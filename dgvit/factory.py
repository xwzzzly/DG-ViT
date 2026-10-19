from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model import DGVisionTransformer


def model_kwargs_from_config(
    config: dict[str, Any],
    prior_weights: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    model_config = config["model"]
    normalization = config["normalization"]
    return {
        "prior_weights": torch.as_tensor(prior_weights, dtype=torch.float32),
        "source_means": normalization["means"],
        "source_stds": normalization["stds"],
        "image_size": int(model_config["image_size"]),
        "patch_size": int(model_config["patch_size"]),
        "embed_dim": int(model_config["embed_dim"]),
        "depth": int(model_config["depth"]),
        "num_heads": int(model_config["num_heads"]),
        "mlp_ratio": float(model_config["mlp_ratio"]),
        "dropout": float(model_config["dropout"]),
        "use_checkpoint": bool(model_config.get("use_checkpoint", False)),
        "lambda_prior": float(model_config.get("lambda_prior", 1.0)),
        "alpha_floor": float(model_config.get("alpha_floor", 0.05)),
        "sample_dirichlet_during_training": bool(
            model_config.get("sample_dirichlet_during_training", True)
        ),
    }


def build_model(
    config: dict[str, Any],
    prior_weights: np.ndarray | torch.Tensor,
) -> DGVisionTransformer:
    return DGVisionTransformer(**model_kwargs_from_config(config, prior_weights))


def model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[DGVisionTransformer, dict[str, Any]]:
    config = checkpoint["config"]
    prior_weights = checkpoint["prior_weights"]
    model = build_model(config, prior_weights).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, config
