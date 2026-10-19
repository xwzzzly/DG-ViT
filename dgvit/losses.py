from __future__ import annotations

import torch


def total_variation(field: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    horizontal_valid = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
    vertical_valid = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
    horizontal = (
        (field[:, :, :, 1:] - field[:, :, :, :-1]).abs() * horizontal_valid
    )
    vertical = (
        (field[:, :, 1:, :] - field[:, :, :-1, :]).abs() * vertical_valid
    )
    horizontal_loss = horizontal.sum() / horizontal_valid.sum().clamp_min(1.0)
    vertical_loss = vertical.sum() / vertical_valid.sum().clamp_min(1.0)
    return horizontal_loss + vertical_loss


def reconstruction_loss(
    fused: torch.Tensor,
    sources: torch.Tensor,
    masks: torch.Tensor,
    source_weights: torch.Tensor | None = None,
    tv_weight: float = 5e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Masked unsupervised reconstruction loss corresponding to Eq. (13).

    source_weights can be the LC-specific pixel prior [B,S,H,W]. When omitted,
    all valid sources receive equal reliability before mask normalization.
    """
    if source_weights is None:
        source_weights = torch.ones_like(sources)
    weights = source_weights.clamp_min(0.0) * masks
    squared_error = (fused - sources).square()
    reconstruction = (squared_error * weights).sum() / weights.sum().clamp_min(1.0)
    union_mask = (masks.sum(dim=1, keepdim=True) > 0).to(fused.dtype)
    tv = total_variation(fused, union_mask)
    total = reconstruction + float(tv_weight) * tv
    return total, {
        "loss": total.detach(),
        "reconstruction": reconstruction.detach(),
        "tv": tv.detach(),
    }
