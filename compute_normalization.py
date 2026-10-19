from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from dgvit.config import load_config, save_config
from dgvit.data import load_manifest
from dgvit.raster import valid_data_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate mission-specific dB normalization statistics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--max-samples-per-raster", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sources = config["source_names"]
    frame = load_manifest(args.manifest, sources)
    rng = np.random.default_rng(int(config["data"]["seed"]))
    valid_min = float(config["data"]["valid_min_db"])
    valid_max = float(config["data"]["valid_max_db"])
    sampled = {source: [] for source in sources}

    for _, row in frame.iterrows():
        for source in sources:
            raw_path = str(row[source]).strip()
            if not raw_path or raw_path.lower() == "nan" or not Path(raw_path).exists():
                continue
            with rasterio.open(raw_path) as dataset:
                values = dataset.read(1)
                mask = valid_data_mask(
                    values,
                    dataset.nodata,
                    valid_min,
                    valid_max,
                )
            values = values[mask].astype(np.float32)
            if values.size > args.max_samples_per_raster:
                selection = rng.choice(
                    values.size,
                    args.max_samples_per_raster,
                    replace=False,
                )
                values = values[selection]
            if values.size:
                sampled[source].append(values)

    means = []
    stds = []
    for source in sources:
        if not sampled[source]:
            raise RuntimeError(f"No valid observations were found for {source}")
        values = np.concatenate(sampled[source])
        mean = float(np.mean(values))
        std = float(np.std(values))
        means.append(mean)
        stds.append(max(std, 1e-6))
        print(f"{source:10s} mean={mean:.6f}, std={std:.6f}, n={values.size}")

    config["normalization"]["means"] = means
    config["normalization"]["stds"] = stds
    save_config(config, args.output_config)
    print(f"Saved normalized configuration to {args.output_config}")


if __name__ == "__main__":
    main()
