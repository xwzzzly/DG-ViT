from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import xy

from dgvit.config import load_config
from dgvit.data import load_manifest
from dgvit.metrics import RunningPairMetrics, weighted_median
from dgvit.priors import LAND_COVER_NAMES
from dgvit.raster import first_existing_path, read_aligned, reference_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate LC-specific prior weights from Eqs. (2)-(6)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--land-cover", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta-bias", type=float, default=5.0)
    parser.add_argument("--beta-rmse", type=float, default=5.0)
    parser.add_argument("--minimum-pair-samples", type=int, default=30)
    parser.add_argument(
        "--cygnss-latitude-limit",
        type=float,
        default=38.0,
        help=(
            "Absolute latitude limit of the CYGNSS orbital coverage domain. "
            "Other missions use the full study region."
        ),
    )
    return parser.parse_args()


def clean_path(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def class_mask(land_cover: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return (land_cover >= 1) & (land_cover <= 9)
    return land_cover == code


def grid_center_latitudes(reference: dict) -> np.ndarray:
    rows, columns = np.indices(
        (reference["height"], reference["width"]),
        dtype=np.int32,
    )
    x_coordinates, y_coordinates = xy(
        reference["transform"],
        rows,
        columns,
        offset="center",
    )
    transformer = Transformer.from_crs(
        reference["crs"],
        "EPSG:4326",
        always_xy=True,
    )
    _, latitudes = transformer.transform(
        np.asarray(x_coordinates),
        np.asarray(y_coordinates),
    )
    return np.asarray(latitudes, dtype=np.float64).reshape(
        reference["height"],
        reference["width"],
    )


def source_domain_mask(
    source: str,
    latitudes: np.ndarray,
    cygnss_latitude_limit: float,
) -> np.ndarray:
    if source.strip().upper() == "CYGNSS":
        return (
            np.isfinite(latitudes)
            & (np.abs(latitudes) <= cygnss_latitude_limit)
        )
    return np.ones(latitudes.shape, dtype=bool)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sources = config["source_names"]
    frame = load_manifest(args.manifest, sources)
    all_paths = [
        clean_path(value)
        for source in sources
        for value in frame[source].tolist()
    ]
    reference = reference_grid(first_existing_path(all_paths))
    land_cover_raw, land_cover_valid = read_aligned(
        args.land_cover,
        reference,
        categorical=True,
        destination_nodata=0,
    )
    land_cover = np.rint(land_cover_raw).astype(np.int16)
    land_cover[~land_cover_valid] = 0
    land_cover[(land_cover < 0) | (land_cover > 9)] = 0
    latitudes = grid_center_latitudes(reference)
    source_domains = [
        source_domain_mask(source, latitudes, args.cygnss_latitude_limit)
        for source in sources
    ]

    pair_accumulators = defaultdict(RunningPairMetrics)
    daily_coverage: dict[tuple[int, int], list[float]] = defaultdict(list)
    valid_min = float(config["data"]["valid_min_db"])
    valid_max = float(config["data"]["valid_max_db"])
    nodata = float(config["data"]["nodata"])

    for row_number, row in frame.iterrows():
        arrays = []
        masks = []
        for source in sources:
            path = clean_path(row[source])
            if path is None or not Path(path).exists():
                arrays.append(
                    np.full(
                        (reference["height"], reference["width"]),
                        nodata,
                        dtype=np.float32,
                    )
                )
                masks.append(
                    np.zeros(
                        (reference["height"], reference["width"]),
                        dtype=bool,
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
            arrays.append(values)
            masks.append(valid)

        for code in LAND_COVER_NAMES:
            lc_mask = class_mask(land_cover, code)
            if not np.any(lc_mask):
                for source_index in range(len(sources)):
                    daily_coverage[(code, source_index)].append(0.0)
                continue
            for source_index in range(len(sources)):
                candidate_mask = lc_mask & source_domains[source_index]
                candidate_count = int(candidate_mask.sum())
                coverage = (
                    float((masks[source_index] & candidate_mask).sum())
                    / candidate_count
                    if candidate_count > 0
                    else 0.0
                )
                daily_coverage[(code, source_index)].append(coverage)
            for first, second in itertools.combinations(range(len(sources)), 2):
                collocated = (
                    lc_mask
                    & source_domains[first]
                    & source_domains[second]
                    & masks[first]
                    & masks[second]
                )
                if np.any(collocated):
                    pair_accumulators[(code, first, second)].update(
                        arrays[first][collocated],
                        arrays[second][collocated],
                    )
        print(f"[{row_number + 1}/{len(frame)}] processed {row['date']}")

    pair_rows = []
    pair_results = {}
    for key, accumulator in pair_accumulators.items():
        code, first, second = key
        metrics = accumulator.calculate()
        pair_results[key] = metrics
        pair_rows.append(
            {
                "lc_code": code,
                "lc_name": LAND_COVER_NAMES[code],
                "source_x": sources[first],
                "source_y": sources[second],
                **metrics,
            }
        )

    metric_rows = []
    prior_rows = []
    for code, name in LAND_COVER_NAMES.items():
        scores = []
        source_metrics = []
        lc_mask = class_mask(land_cover, code)
        candidate_count = int(lc_mask.sum())
        for source_index, source in enumerate(sources):
            source_candidate_count = int(
                (lc_mask & source_domains[source_index]).sum()
            )
            correlations = []
            biases = []
            ubrmses = []
            pair_counts = []
            for other in range(len(sources)):
                if other == source_index:
                    continue
                key = (
                    (code, source_index, other)
                    if source_index < other
                    else (code, other, source_index)
                )
                metrics = pair_results.get(key)
                if metrics is None or metrics["n"] < args.minimum_pair_samples:
                    continue
                correlations.append(metrics["r"])
                biases.append(abs(metrics["bias"]))
                ubrmses.append(metrics["ubrmse"])
                pair_counts.append(metrics["n"])

            counts = np.asarray(pair_counts, dtype=np.float64)
            r_value = weighted_median(np.asarray(correlations), counts)
            bias_value = weighted_median(np.asarray(biases), counts)
            ubrmse_value = weighted_median(np.asarray(ubrmses), counts)
            coverage_values = np.asarray(
                daily_coverage[(code, source_index)],
                dtype=np.float64,
            )
            coverage = float(np.mean(coverage_values)) if coverage_values.size else 0.0
            coverage_std = (
                float(np.std(coverage_values)) if coverage_values.size else 0.0
            )
            stability = (
                float(np.exp(-coverage_std / coverage))
                if coverage > 1e-8
                else 0.0
            )
            r_score = max(float(r_value), 0.0) if np.isfinite(r_value) else 0.0
            bias_score = (
                float(np.exp(-bias_value / args.beta_bias))
                if np.isfinite(bias_value)
                else 0.0
            )
            rmse_score = (
                float(np.exp(-ubrmse_value / args.beta_rmse))
                if np.isfinite(ubrmse_value)
                else 0.0
            )
            score = r_score * bias_score * rmse_score * coverage * stability
            scores.append(score)
            source_metrics.append(
                {
                    "lc_code": code,
                    "lc_name": name,
                    "source": source,
                    "N_candidate": source_candidate_count,
                    "coverage_domain": (
                        (
                            f"TP_within_{args.cygnss_latitude_limit:g}S_"
                            f"to_{args.cygnss_latitude_limit:g}N"
                        )
                        if source.strip().upper() == "CYGNSS"
                        else "TP_full"
                    ),
                    "N_pair_support": int(np.sum(counts)) if counts.size else 0,
                    "R": r_value,
                    "abs_bias_db": bias_value,
                    "ubRMSE_db": ubrmse_value,
                    "Cov": coverage,
                    "Stab": stability,
                    "E_bias": bias_score,
                    "E_rmse": rmse_score,
                    "S_cp": score,
                }
            )

        scores = np.asarray(scores, dtype=np.float64)
        if scores.sum() <= 0:
            scores = np.asarray(
                [
                    metrics["Cov"] * metrics["Stab"]
                    for metrics in source_metrics
                ],
                dtype=np.float64,
            )
        if scores.sum() <= 0:
            scores = np.ones(len(sources), dtype=np.float64)
        weights = scores / scores.sum()
        prior_row = {
            "lc_code": code,
            "lc_name": name,
            "N_candidate": candidate_count,
        }
        for source_index, source in enumerate(sources):
            prior_row[source] = float(weights[source_index])
            prior_row[f"{source}_N_candidate"] = int(
                (lc_mask & source_domains[source_index]).sum()
            )
            source_metrics[source_index]["W_cp"] = float(weights[source_index])
        prior_rows.append(prior_row)
        metric_rows.extend(source_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prior_rows).to_csv(
        output_dir / "prior_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "prior_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "pairwise_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved prior products to {output_dir}")


if __name__ == "__main__":
    main()
