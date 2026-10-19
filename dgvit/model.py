from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.checkpoint import checkpoint

from .priors import LandCoverPrior


FusionMode = Literal["dg_vit", "vit", "prior_only", "mean"]


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.projection = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        x = x + attended
        return x + self.mlp(self.norm2(x))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        use_checkpoint: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


def _masked_normalize(
    values: torch.Tensor,
    availability: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    weighted = values.clamp_min(0.0) * availability.clamp_min(0.0)
    denominator = weighted.sum(dim=-1, keepdim=True)
    fallback = availability / availability.sum(dim=-1, keepdim=True).clamp_min(eps)
    normalized = weighted / denominator.clamp_min(eps)
    return torch.where(denominator > eps, normalized, fallback)


class DirichletLandCoverGate(nn.Module):
    """
    Implements the manuscript relation
        alpha_(c,p) = f_phi(E_p) + lambda * W_(c,p)
        gamma_c ~ Dirichlet(alpha_c)
    at every sample and spatial patch.
    """

    def __init__(
        self,
        embed_dim: int,
        num_sources: int,
        lambda_prior: float = 1.0,
        alpha_floor: float = 0.05,
        sample_during_training: bool = True,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.lambda_prior = float(lambda_prior)
        self.alpha_floor = float(alpha_floor)
        self.sample_during_training = sample_during_training
        hidden_dim = max(32, embed_dim // 2)
        self.source_embedding = nn.Parameter(
            torch.zeros(1, 1, num_sources, embed_dim)
        )
        nn.init.trunc_normal_(self.source_embedding, std=0.02)
        self.gating_network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        availability: torch.Tensor,
        prior_weights: torch.Tensor,
        lambda_override: float | None = None,
        sample_override: bool | None = None,
        prior_activation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            source_tokens: [B,N,S,D]
            availability: [B,N,S], patch valid fractions in [0,1]
            prior_weights: [B,N,S], normalized LC-specific W_(c,p)
        Returns:
            dynamic weights, concentration parameters, feature concentrations.
        """
        if source_tokens.ndim != 4:
            raise ValueError("source_tokens must have shape [B,N,S,D]")
        if source_tokens.shape[2] != self.num_sources:
            raise ValueError("Unexpected number of source branches")

        features = source_tokens + self.source_embedding
        token_dtype = source_tokens.dtype
        feature_alpha = F.softplus(
            self.gating_network(features).squeeze(-1).float()
        )
        feature_alpha = feature_alpha + self.alpha_floor

        prior_lambda = self.lambda_prior if lambda_override is None else lambda_override
        if prior_activation is None:
            prior_activation = torch.ones_like(prior_weights[..., :1])
        prior_weights = prior_weights.float()
        prior_activation = prior_activation.float()
        availability_float = availability.float()
        concentration = (
            feature_alpha
            + float(prior_lambda) * prior_weights * prior_activation
        )

        active = availability_float > 0
        safe_concentration = torch.where(
            active,
            concentration,
            torch.full_like(concentration, self.alpha_floor),
        )
        should_sample = (
            self.training and self.sample_during_training
            if sample_override is None
            else sample_override
        )
        if should_sample:
            raw_weights = Dirichlet(safe_concentration).rsample()
        else:
            raw_weights = safe_concentration / safe_concentration.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

        dynamic_weights = _masked_normalize(raw_weights, availability_float)
        return (
            dynamic_weights.to(token_dtype),
            concentration,
            feature_alpha,
        )


class DGVisionTransformer(nn.Module):
    """Dirichlet-prior-guided multi-mission GNSS-R reflectivity fusion."""

    def __init__(
        self,
        prior_weights: torch.Tensor,
        source_means: list[float] | torch.Tensor,
        source_stds: list[float] | torch.Tensor,
        image_size: int = 224,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        lambda_prior: float = 1.0,
        alpha_floor: float = 0.05,
        sample_dirichlet_during_training: bool = True,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        prior_weights = torch.as_tensor(prior_weights, dtype=torch.float32)
        self.num_sources = int(prior_weights.shape[1])
        if len(source_means) != self.num_sources or len(source_stds) != self.num_sources:
            raise ValueError("Normalization statistics must match the source count")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size**2
        self.embed_dim = int(embed_dim)

        means = torch.as_tensor(source_means, dtype=torch.float32).view(
            1, self.num_sources, 1, 1
        )
        stds = torch.as_tensor(source_stds, dtype=torch.float32).clamp_min(1e-6).view(
            1, self.num_sources, 1, 1
        )
        self.register_buffer("source_means", means)
        self.register_buffer("source_stds", stds)

        self.land_cover_prior = LandCoverPrior(prior_weights)
        self.patch_embeddings = nn.ModuleList(
            [
                PatchEmbed(self.image_size, self.patch_size, self.embed_dim)
                for _ in range(self.num_sources)
            ]
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.dirichlet_gate = DirichletLandCoverGate(
            self.embed_dim,
            self.num_sources,
            lambda_prior=lambda_prior,
            alpha_floor=alpha_floor,
            sample_during_training=sample_dirichlet_during_training,
        )
        self.encoder = TransformerEncoder(
            self.embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            dropout,
            use_checkpoint,
        )
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.delta_head = nn.Linear(self.embed_dim, 1)
        self.refinement_gate = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid(),
        )

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        land_cover: torch.Tensor,
    ) -> None:
        expected = (
            inputs.ndim == 4
            and inputs.shape == masks.shape
            and inputs.shape[1] == self.num_sources
            and inputs.shape[-2:] == (self.image_size, self.image_size)
        )
        if not expected:
            raise ValueError(
                "inputs and masks must have shape "
                f"[B,{self.num_sources},{self.image_size},{self.image_size}]"
            )
        if land_cover.shape[-2:] != inputs.shape[-2:]:
            raise ValueError("land_cover and source rasters must share a grid")

    def _source_tokens(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (inputs - self.source_means) / self.source_stds
        normalized = normalized * masks
        token_list = []
        availability_list = []
        for source_index, embedding in enumerate(self.patch_embeddings):
            token_list.append(embedding(normalized[:, source_index : source_index + 1]))
            patch_mask = F.avg_pool2d(
                masks[:, source_index : source_index + 1],
                self.patch_size,
                self.patch_size,
            )
            availability_list.append(patch_mask.flatten(2).transpose(1, 2))
        source_tokens = torch.stack(token_list, dim=2)
        availability = torch.cat(availability_list, dim=-1)
        return source_tokens, availability

    def _patch_priors(self, land_cover: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_prior = self.land_cover_prior(land_cover)
        patch_prior = F.avg_pool2d(
            pixel_prior,
            self.patch_size,
            self.patch_size,
        )
        patch_prior = patch_prior.flatten(2).transpose(1, 2)
        patch_prior = patch_prior / patch_prior.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return patch_prior, pixel_prior

    def _pixel_baseline(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        patch_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        weight_grid = patch_weights.transpose(1, 2).reshape(
            batch_size,
            self.num_sources,
            self.grid_size,
            self.grid_size,
        )
        pixel_weights = F.interpolate(
            weight_grid,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        pixel_weights = pixel_weights * masks
        pixel_weights = pixel_weights / pixel_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        return (inputs * pixel_weights).sum(dim=1, keepdim=True)

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        land_cover: torch.Tensor,
        mode: FusionMode = "dg_vit",
        lambda_override: float | None = None,
        sample_dirichlet: bool | None = None,
        active_prior_classes: tuple[int, ...] | list[int] | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(inputs, masks, land_cover)
        masks = masks.to(dtype=inputs.dtype).clamp(0.0, 1.0)
        source_tokens, availability = self._source_tokens(inputs, masks)
        patch_prior, pixel_prior = self._patch_priors(land_cover)
        prior_activation = torch.ones_like(patch_prior[..., :1])
        pixel_prior_activation = torch.ones_like(pixel_prior[:, :1])
        if active_prior_classes is not None:
            codes = land_cover[:, 0] if land_cover.ndim == 4 else land_cover
            active_pixels = torch.zeros_like(codes, dtype=inputs.dtype)
            for code in active_prior_classes:
                active_pixels = torch.maximum(
                    active_pixels,
                    (codes == int(code)).to(inputs.dtype),
                )
            pixel_prior_activation = active_pixels.unsqueeze(1)
            prior_activation = F.avg_pool2d(
                pixel_prior_activation,
                self.patch_size,
                self.patch_size,
            ).flatten(2).transpose(1, 2)

        concentration = torch.zeros_like(patch_prior)
        feature_concentration = torch.zeros_like(patch_prior)
        if mode == "dg_vit":
            patch_weights, concentration, feature_concentration = self.dirichlet_gate(
                source_tokens,
                availability,
                patch_prior,
                lambda_override=lambda_override,
                sample_override=sample_dirichlet,
                prior_activation=prior_activation,
            )
        elif mode == "vit":
            patch_weights, concentration, feature_concentration = self.dirichlet_gate(
                source_tokens,
                availability,
                patch_prior,
                lambda_override=0.0,
                sample_override=sample_dirichlet,
                prior_activation=prior_activation,
            )
        elif mode == "prior_only":
            uniform = torch.full_like(patch_prior, 1.0 / self.num_sources)
            activated_prior = (
                prior_activation * patch_prior
                + (1.0 - prior_activation) * uniform
            )
            patch_weights = _masked_normalize(activated_prior, availability)
        elif mode == "mean":
            patch_weights = _masked_normalize(
                torch.ones_like(availability),
                availability,
            )
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

        baseline = self._pixel_baseline(inputs, masks, patch_weights)
        if mode in {"dg_vit", "vit"}:
            fused_tokens = (source_tokens * patch_weights.unsqueeze(-1)).sum(dim=2)
            encoded = self.encoder(fused_tokens + self.position_embedding)
            encoded = self.final_norm(encoded)
            delta = self.delta_head(encoded)
            refine = self.refinement_gate(encoded)
            delta_grid = delta.transpose(1, 2).reshape(
                inputs.shape[0], 1, self.grid_size, self.grid_size
            )
            refine_grid = refine.transpose(1, 2).reshape(
                inputs.shape[0], 1, self.grid_size, self.grid_size
            )
            delta_pixels = F.interpolate(
                delta_grid,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            refine_pixels = F.interpolate(
                refine_grid,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            fused = baseline + refine_pixels * delta_pixels
        else:
            fused = baseline

        valid_count = masks.sum(dim=1, keepdim=True)
        union_mask = valid_count > 0
        single_source = valid_count == 1
        single_value = (inputs * masks).sum(dim=1, keepdim=True)
        fused = torch.where(single_source, single_value, fused)
        fused = torch.where(union_mask, fused, torch.zeros_like(fused))

        if not return_diagnostics:
            return fused
        diagnostics = {
            "patch_weights": patch_weights,
            "concentration": concentration,
            "feature_concentration": feature_concentration,
            "patch_prior": patch_prior,
            "patch_availability": availability,
            "prior_activation": prior_activation,
            "pixel_prior_activation": pixel_prior_activation,
            "pixel_prior": pixel_prior,
            "baseline": baseline,
            "union_mask": union_mask.to(inputs.dtype),
        }
        return fused, diagnostics
