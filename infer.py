from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dgvit.data import load_manifest
from dgvit.factory import model_from_checkpoint
from dgvit.priors import LAND_COVER_NAMES
from dgvit.raster import (
    first_existing_path,
    read_aligned,
    reference_grid,
    write_geotiff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic daily fused SR rasters with DG-ViT."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--land-cover", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["checkpoint", "dg_vit", "vit", "prior_only", "mean"],
        default="checkpoint",
    )
    parser.add_argument("--lambda-prior", type=float, default=None)
    parser.add_argument(
        "--active-prior-classes",
        default="",
        help="Comma-separated CLCD codes; blank activates all classes.",
    )
    parser.add_argument("--date", default="", help="Optional YYYYMMDD subset")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-gates", action="store_true")
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_path(value: object) -> str | None:
    text = str(value).strip()
    return None if not text or text.lower() == "nan" else text


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def blending_window(tile: int) -> np.ndarray:
    one_dimensional = np.hanning(tile).astype(np.float32)
    window = np.outer(one_dimensional, one_dimensional)
    return np.clip(window, 0.05, None)


def padded_tile(
    array: np.ndarray,
    row: int,
    column: int,
    tile: int,
    fill: float,
) -> tuple[np.ndarray, int, int]:
    height = min(tile, array.shape[-2] - row)
    width = min(tile, array.shape[-1] - column)
    result = np.full((*array.shape[:-2], tile, tile), fill, dtype=array.dtype)
    result[..., :height, :width] = array[..., row : row + height, column : column + width]
    return result, height, width


def fuse_scene(
    model,
    sources: np.ndarray,
    masks: np.ndarray,
    land_cover: np.ndarray,
    mode: str,
    lambda_prior: float | None,
    overlap: int,
    batch_size: int,
    device: torch.device,
    save_gates: bool,
    active_prior_classes: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    tile = model.image_size
    height, width = sources.shape[-2:]
    rows = tile_starts(height, tile, overlap)
    columns = tile_starts(width, tile, overlap)
    positions = [(row, column) for row in rows for column in columns]
    window = blending_window(tile)
    fused_sum = np.zeros((height, width), dtype=np.float64)
    fused_weight = np.zeros((height, width), dtype=np.float64)
    gate_sum = (
        np.zeros((model.num_sources, height, width), dtype=np.float64)
        if save_gates
        else None
    )
    gate_weight = (
        np.zeros((height, width), dtype=np.float64) if save_gates else None
    )

    model.eval()
    for batch_start in range(0, len(positions), batch_size):
        batch_positions = positions[batch_start : batch_start + batch_size]
        source_tiles = []
        mask_tiles = []
        land_cover_tiles = []
        shapes = []
        for row, column in batch_positions:
            source_tile, actual_height, actual_width = padded_tile(
                sources,
                row,
                column,
                tile,
                0.0,
            )
            mask_tile, _, _ = padded_tile(masks, row, column, tile, 0.0)
            lc_tile, _, _ = padded_tile(land_cover, row, column, tile, 0)
            source_tiles.append(source_tile)
            mask_tiles.append(mask_tile)
            land_cover_tiles.append(lc_tile)
            shapes.append((actual_height, actual_width))

        source_tensor = torch.from_numpy(np.stack(source_tiles)).to(device)
        mask_tensor = torch.from_numpy(np.stack(mask_tiles)).to(device)
        land_cover_tensor = torch.from_numpy(np.stack(land_cover_tiles)).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            fused, diagnostics = model(
                source_tensor,
                mask_tensor,
                land_cover_tensor,
                mode=mode,
                lambda_override=lambda_prior,
                sample_dirichlet=False,
                active_prior_classes=active_prior_classes,
                return_diagnostics=True,
            )
            if save_gates:
                patch_weights = diagnostics["patch_weights"].transpose(1, 2).reshape(
                    len(batch_positions),
                    model.num_sources,
                    model.grid_size,
                    model.grid_size,
                )
                pixel_gates = F.interpolate(
                    patch_weights,
                    size=(tile, tile),
                    mode="nearest",
                )

        fused_batch = fused[:, 0].float().cpu().numpy()
        gates_batch = (
            pixel_gates.float().cpu().numpy() if save_gates else None
        )
        masks_batch = np.stack(mask_tiles)
        for index, ((row, column), shape) in enumerate(
            zip(batch_positions, shapes)
        ):
            actual_height, actual_width = shape
            valid = masks_batch[index, :, :actual_height, :actual_width].sum(axis=0) > 0
            local_window = window[:actual_height, :actual_width] * valid
            fused_sum[
                row : row + actual_height,
                column : column + actual_width,
            ] += fused_batch[index, :actual_height, :actual_width] * local_window
            fused_weight[
                row : row + actual_height,
                column : column + actual_width,
            ] += local_window
            if save_gates and gate_sum is not None and gate_weight is not None:
                gate_sum[
                    :,
                    row : row + actual_height,
                    column : column + actual_width,
                ] += gates_batch[
                    index, :, :actual_height, :actual_width
                ] * local_window
                gate_weight[
                    row : row + actual_height,
                    column : column + actual_width,
                ] += local_window

    fused_result = np.full((height, width), np.nan, dtype=np.float32)
    valid_result = fused_weight > 0
    fused_result[valid_result] = (
        fused_sum[valid_result] / fused_weight[valid_result]
    ).astype(np.float32)
    gate_result = None
    if save_gates and gate_sum is not None and gate_weight is not None:
        gate_result = np.full(
            (model.num_sources, height, width),
            np.nan,
            dtype=np.float32,
        )
        valid_gates = gate_weight > 0
        gate_result[:, valid_gates] = (
            gate_sum[:, valid_gates] / gate_weight[valid_gates]
        ).astype(np.float32)
    return fused_result, gate_result


def gate_summary(
    gates: np.ndarray,
    land_cover: np.ndarray,
    source_names: list[str],
    date: str,
) -> pd.DataFrame:
    records = []
    for code, name in LAND_COVER_NAMES.items():
        selection = (
            np.isfinite(gates).all(axis=0)
            if code == 0
            else (land_cover == code) & np.isfinite(gates).all(axis=0)
        )
        count = int(selection.sum())
        for source_index, source in enumerate(source_names):
            values = gates[source_index][selection]
            records.append(
                {
                    "date": date,
                    "lc_code": code,
                    "lc_name": name,
                    "source": source,
                    "N": count,
                    "gate_mean": float(np.mean(values)) if count else np.nan,
                    "gate_std": float(np.std(values)) if count else np.nan,
                    "gate_p05": float(np.quantile(values, 0.05)) if count else np.nan,
                    "gate_p95": float(np.quantile(values, 0.95)) if count else np.nan,
                }
            )
    return pd.DataFrame(records)


def write_point_csv(
    path: Path,
    fused: np.ndarray,
    reference: dict,
) -> None:
    valid = np.isfinite(fused)
    rows, columns = np.where(valid)
    transform = reference["transform"]
    x = transform.c + (columns + 0.5) * transform.a + (rows + 0.5) * transform.b
    y = transform.f + (columns + 0.5) * transform.d + (rows + 0.5) * transform.e
    pd.DataFrame(
        {
            "EASE_X": x,
            "EASE_Y": y,
            "FUSED_SR_DB": fused[valid],
        }
    ).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, config = model_from_checkpoint(checkpoint, device)
    source_names = list(config["source_names"])
    frame = load_manifest(args.manifest, source_names)
    if args.date:
        frame = frame[frame["date"].astype(str) == args.date].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Date {args.date} is absent from the manifest")

    mode = checkpoint.get("mode", "dg_vit") if args.mode == "checkpoint" else args.mode
    overlap = int(config["inference"]["overlap"])
    batch_size = args.batch_size or int(config["inference"]["batch_size"])
    nodata = float(config["data"]["nodata"])
    valid_min = float(config["data"]["valid_min_db"])
    valid_max = float(config["data"]["valid_max_db"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_prior_classes = (
        tuple(
            int(code)
            for code in args.active_prior_classes.split(",")
            if code.strip()
        )
        or None
    )

    all_summaries = []
    for _, row in frame.iterrows():
        date = str(row["date"])
        paths = [clean_path(row[source]) for source in source_names]
        reference = reference_grid(first_existing_path(paths))
        source_arrays = []
        source_masks = []
        for path in paths:
            if path is None or not Path(path).exists():
                source_arrays.append(
                    np.zeros(
                        (reference["height"], reference["width"]),
                        dtype=np.float32,
                    )
                )
                source_masks.append(
                    np.zeros(
                        (reference["height"], reference["width"]),
                        dtype=np.float32,
                    )
                )
                continue
            values, valid = read_aligned(
                path,
                reference,
                categorical=False,
                valid_min=valid_min,
                valid_max=valid_max,
                destination_nodata=nodata,
            )
            source_arrays.append(np.where(valid, values, 0.0).astype(np.float32))
            source_masks.append(valid.astype(np.float32))

        land_cover_raw, land_cover_valid = read_aligned(
            args.land_cover,
            reference,
            categorical=True,
            destination_nodata=0,
        )
        land_cover = np.rint(land_cover_raw).astype(np.int64)
        land_cover[~land_cover_valid] = 0
        land_cover[(land_cover < 0) | (land_cover > 9)] = 0
        fused, gates = fuse_scene(
            model,
            np.stack(source_arrays),
            np.stack(source_masks),
            land_cover,
            mode,
            args.lambda_prior,
            overlap,
            batch_size,
            device,
            args.save_gates,
            active_prior_classes,
        )

        fused_for_output = np.where(np.isfinite(fused), fused, nodata).astype(np.float32)
        fused_path = output_dir / f"{date}_{mode}_fused_sr_db.tif"
        write_geotiff(fused_path, fused_for_output, reference, nodata=nodata)
        if args.write_csv:
            write_point_csv(
                output_dir / f"{date}_{mode}_fused_sr_db.csv",
                fused,
                reference,
            )
        if gates is not None:
            summary = gate_summary(gates, land_cover, source_names, date)
            summary.to_csv(
                output_dir / f"{date}_{mode}_gate_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            all_summaries.append(summary)
            for source_index, source in enumerate(source_names):
                gate_output = np.where(
                    np.isfinite(gates[source_index]),
                    gates[source_index],
                    nodata,
                )
                safe_name = source.replace("-", "")
                write_geotiff(
                    output_dir / f"{date}_{mode}_gate_{safe_name}.tif",
                    gate_output,
                    reference,
                    nodata=nodata,
                )
        print(f"Completed {date}: {fused_path}")

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            output_dir / f"{mode}_gate_summary_all_dates.csv",
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()
