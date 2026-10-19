from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from torch.utils.data import Dataset

from .raster import first_existing_path, reference_grid, valid_data_mask


def load_manifest(path: str | Path, source_names: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"date": str})
    required = {"date", *source_names}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def split_manifest(
    frame: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(frame["date"].astype(str).unique())
    split_index = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * train_fraction)))
    train_dates = set(unique_dates[:split_index])
    train = frame[frame["date"].astype(str).isin(train_dates)].reset_index(drop=True)
    validation = frame[~frame["date"].astype(str).isin(train_dates)].reset_index(drop=True)
    if validation.empty:
        validation = train.copy()
    return train, validation


def _clean_manifest_path(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _read_window_on_grid(
    path: str | Path,
    grid: dict,
    window: Window,
    categorical: bool,
    fill_value: float,
) -> tuple[np.ndarray, float | int | None]:
    with rasterio.open(path) as source:
        same_grid = (
            source.crs == grid["crs"]
            and source.transform.almost_equals(grid["transform"])
            and source.width == grid["width"]
            and source.height == grid["height"]
        )
        if same_grid:
            return (
                source.read(1, window=window, boundless=True, fill_value=fill_value),
                source.nodata,
            )
        with WarpedVRT(
            source,
            crs=grid["crs"],
            transform=grid["transform"],
            width=grid["width"],
            height=grid["height"],
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
            nodata=fill_value,
        ) as warped:
            return (
                warped.read(1, window=window, boundless=True, fill_value=fill_value),
                fill_value,
            )


class RandomRasterPatchDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        source_names: Sequence[str],
        land_cover_path: str | Path,
        samples_per_epoch: int,
        patch_size: int = 224,
        valid_min_db: float = -80.0,
        valid_max_db: float = 0.0,
        nodata: float = -9999.0,
        seed: int = 20260727,
        augment: bool = True,
        min_overlap_fraction: float = 0.01,
        max_sampling_attempts: int = 8,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.source_names = list(source_names)
        self.land_cover_path = Path(land_cover_path)
        self.samples_per_epoch = int(samples_per_epoch)
        self.patch_size = int(patch_size)
        self.valid_min_db = valid_min_db
        self.valid_max_db = valid_max_db
        self.nodata = float(nodata)
        self.seed = int(seed)
        self.augment = augment
        self.min_overlap_fraction = float(min_overlap_fraction)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.epoch = 0

        candidate_paths = []
        for source in self.source_names:
            candidate_paths.extend(
                _clean_manifest_path(value)
                for value in self.manifest[source].tolist()
            )
        reference_path = first_existing_path(candidate_paths)
        self.grid = reference_grid(reference_path)
        if self.grid["height"] < self.patch_size or self.grid["width"] < self.patch_size:
            raise ValueError("Raster dimensions are smaller than the requested patch")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_window(self, rng: np.random.Generator) -> Window:
        row = int(rng.integers(0, self.grid["height"] - self.patch_size + 1))
        column = int(rng.integers(0, self.grid["width"] - self.patch_size + 1))
        return Window(column, row, self.patch_size, self.patch_size)

    def _read_sample(
        self,
        manifest_row: pd.Series,
        window: Window,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = []
        masks = []
        for source in self.source_names:
            path = _clean_manifest_path(manifest_row[source])
            if path is None or not Path(path).exists():
                arrays.append(
                    np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                )
                masks.append(
                    np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                )
                continue
            values, source_nodata = _read_window_on_grid(
                path,
                self.grid,
                window,
                categorical=False,
                fill_value=self.nodata,
            )
            valid = valid_data_mask(
                values,
                source_nodata,
                self.valid_min_db,
                self.valid_max_db,
            )
            values = np.where(valid, values, 0.0).astype(np.float32)
            arrays.append(values)
            masks.append(valid.astype(np.float32))

        land_cover, _ = _read_window_on_grid(
            self.land_cover_path,
            self.grid,
            window,
            categorical=True,
            fill_value=0,
        )
        land_cover = np.rint(land_cover).astype(np.int64)
        land_cover[(land_cover < 0) | (land_cover > 9)] = 0
        return np.stack(arrays), np.stack(masks), land_cover

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(
            self.seed + self.epoch * self.samples_per_epoch + int(index)
        )
        best_sample = None
        best_overlap = -1.0
        selected_date = ""
        for _ in range(self.max_sampling_attempts):
            row_index = int(rng.integers(0, len(self.manifest)))
            row = self.manifest.iloc[row_index]
            sample = self._read_sample(row, self._sample_window(rng))
            overlap = float((sample[1].sum(axis=0) >= 2).mean())
            if overlap > best_overlap:
                best_sample = sample
                best_overlap = overlap
                selected_date = str(row["date"])
            if overlap >= self.min_overlap_fraction:
                break

        if best_sample is None:
            raise RuntimeError("Unable to sample a raster patch")
        values, masks, land_cover = best_sample

        if self.augment:
            rotations = int(rng.integers(0, 4))
            values = np.rot90(values, rotations, axes=(-2, -1)).copy()
            masks = np.rot90(masks, rotations, axes=(-2, -1)).copy()
            land_cover = np.rot90(land_cover, rotations, axes=(-2, -1)).copy()
            if rng.random() < 0.5:
                values = values[..., ::-1].copy()
                masks = masks[..., ::-1].copy()
                land_cover = land_cover[..., ::-1].copy()
            if rng.random() < 0.5:
                values = values[..., ::-1, :].copy()
                masks = masks[..., ::-1, :].copy()
                land_cover = land_cover[..., ::-1, :].copy()

        return {
            "sources": torch.from_numpy(values),
            "masks": torch.from_numpy(masks),
            "land_cover": torch.from_numpy(land_cover),
            "date": selected_date,
        }
