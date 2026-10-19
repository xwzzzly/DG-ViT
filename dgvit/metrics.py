from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RunningPairMetrics:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0
    sum_diff: float = 0.0
    sum_diff2: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            return
        x = np.asarray(x[valid], dtype=np.float64)
        y = np.asarray(y[valid], dtype=np.float64)
        difference = y - x
        self.n += int(x.size)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float(np.dot(x, x))
        self.sum_y2 += float(np.dot(y, y))
        self.sum_xy += float(np.dot(x, y))
        self.sum_diff += float(difference.sum())
        self.sum_diff2 += float(np.dot(difference, difference))

    def calculate(self) -> dict[str, float]:
        if self.n < 2:
            return {
                "n": float(self.n),
                "r": np.nan,
                "bias": np.nan,
                "rmse": np.nan,
                "ubrmse": np.nan,
            }
        n = float(self.n)
        covariance = self.sum_xy - self.sum_x * self.sum_y / n
        variance_x = self.sum_x2 - self.sum_x**2 / n
        variance_y = self.sum_y2 - self.sum_y**2 / n
        denominator = math.sqrt(max(variance_x * variance_y, 0.0))
        correlation = covariance / denominator if denominator > 0 else np.nan
        bias = self.sum_diff / n
        rmse = math.sqrt(max(self.sum_diff2 / n, 0.0))
        ubrmse = math.sqrt(max(rmse**2 - bias**2, 0.0))
        return {
            "n": n,
            "r": correlation,
            "bias": bias,
            "rmse": rmse,
            "ubrmse": ubrmse,
        }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])
