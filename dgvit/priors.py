from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


LAND_COVER_NAMES = {
    0: "ALL",
    1: "Cropland",
    2: "Forest",
    3: "Shrub",
    4: "Grassland",
    5: "Water",
    6: "Snow_Ice",
    7: "Barren",
    8: "Impervious",
    9: "Wetland",
}


def normalize_prior_rows(weights: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError("Prior weights must have shape [land-cover classes, sources].")
    weights = np.clip(weights, 0.0, None)
    row_sum = weights.sum(axis=1, keepdims=True)
    fallback = np.full_like(weights, 1.0 / weights.shape[1])
    return np.where(row_sum > eps, weights / np.maximum(row_sum, eps), fallback)


def load_prior_table(
    csv_path: str | Path,
    source_names: Sequence[str],
    num_classes: int = 10,
) -> np.ndarray:
    frame = pd.read_csv(csv_path)
    required = {"lc_code", *source_names}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in prior table: {sorted(missing)}")

    table = np.zeros((num_classes, len(source_names)), dtype=np.float32)
    seen = set()
    for _, row in frame.iterrows():
        code = int(row["lc_code"])
        if 0 <= code < num_classes:
            table[code] = row[list(source_names)].to_numpy(dtype=np.float32)
            seen.add(code)

    if 0 not in seen:
        raise ValueError("The prior table must contain lc_code=0 as the ALL fallback.")
    for code in range(num_classes):
        if code not in seen:
            table[code] = table[0]
    return normalize_prior_rows(table)


class LandCoverPrior(nn.Module):
    """Maps integer land-cover codes to source-specific prior weights."""

    def __init__(self, weights: np.ndarray | torch.Tensor):
        super().__init__()
        tensor = torch.as_tensor(weights, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError("weights must have shape [classes, sources]")
        tensor = tensor.clamp_min(0.0)
        tensor = tensor / tensor.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("weights", tensor)

    @property
    def num_sources(self) -> int:
        return int(self.weights.shape[1])

    def forward(self, land_cover: torch.Tensor) -> torch.Tensor:
        """
        Args:
            land_cover: [B,H,W] or [B,1,H,W], integer CLCD codes.
        Returns:
            Prior map [B,S,H,W].
        """
        if land_cover.ndim == 4:
            if land_cover.shape[1] != 1:
                raise ValueError("land_cover must contain one categorical channel")
            land_cover = land_cover[:, 0]
        if land_cover.ndim != 3:
            raise ValueError("land_cover must have shape [B,H,W] or [B,1,H,W]")

        codes = land_cover.long()
        valid = (codes >= 0) & (codes < self.weights.shape[0])
        codes = torch.where(valid, codes, torch.zeros_like(codes))
        mapped = self.weights[codes]
        return mapped.permute(0, 3, 1, 2).contiguous()
