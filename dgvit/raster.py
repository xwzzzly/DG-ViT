from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def first_existing_path(paths: list[str | Path | None]) -> Path:
    for path in paths:
        if path is None:
            continue
        candidate = Path(str(path))
        if str(path).strip() and candidate.exists():
            return candidate
    raise FileNotFoundError("No source raster exists for this record")


def reference_grid(path: str | Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        return {
            "crs": dataset.crs,
            "transform": dataset.transform,
            "width": dataset.width,
            "height": dataset.height,
            "profile": dataset.profile.copy(),
        }


def is_same_grid(dataset: rasterio.io.DatasetReader, reference: dict[str, Any]) -> bool:
    return (
        dataset.crs == reference["crs"]
        and dataset.transform.almost_equals(reference["transform"])
        and dataset.width == reference["width"]
        and dataset.height == reference["height"]
    )


def valid_data_mask(
    array: np.ndarray,
    nodata: float | int | None,
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> np.ndarray:
    mask = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        mask &= array != nodata
    if valid_min is not None:
        mask &= array >= valid_min
    if valid_max is not None:
        mask &= array <= valid_max
    return mask


def read_aligned(
    path: str | Path,
    reference: dict[str, Any],
    categorical: bool = False,
    valid_min: float | None = None,
    valid_max: float | None = None,
    destination_nodata: float = -9999.0,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as source:
        if is_same_grid(source, reference):
            data = source.read(1).astype(np.float32)
            mask = valid_data_mask(data, source.nodata, valid_min, valid_max)
            return data, mask

        data = np.full(
            (reference["height"], reference["width"]),
            destination_nodata,
            dtype=np.float32,
        )
        reproject(
            source=rasterio.band(source, 1),
            destination=data,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference["transform"],
            dst_crs=reference["crs"],
            dst_nodata=destination_nodata,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
        mask = valid_data_mask(data, destination_nodata, valid_min, valid_max)
        return data, mask


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    reference: dict[str, Any],
    nodata: float = -9999.0,
    dtype: str = "float32",
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference["profile"].copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    can_tile = reference["width"] >= 16 and reference["height"] >= 16
    profile.update(
        driver="GTiff",
        width=reference["width"],
        height=reference["height"],
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype.startswith("float") else 2,
        tiled=can_tile,
        BIGTIFF="IF_SAFER",
    )
    if can_tile:
        profile.update(
            blockxsize=min(256, (reference["width"] // 16) * 16),
            blockysize=min(256, (reference["height"] // 16) * 16),
        )
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(array.astype(dtype, copy=False), 1)
