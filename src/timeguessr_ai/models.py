"""Unified ViT backbones (StreetCLIP & SigLIP2), dual MLP heads, and soft-label losses."""

from __future__ import annotations

import gc
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel, SiglipVisionModel

from .metrics import EARTH_RADIUS_KM, NUM_YEARS, YEAR_MIN, latlon_to_xyz, xyz_to_latlon


class VisionBackbone(nn.Module):
    """Unified wrapper around Hugging Face vision encoders (StreetCLIP & SigLIP2)."""

    def __init__(self, model_id: str, gradient_checkpointing: bool = False):
        super().__init__()
        self.model_id = model_id
        lower_id = model_id.lower()

        if "streetclip" in lower_id or "clip" in lower_id and "siglip" not in lower_id:
            vision_wrapper = CLIPVisionModel.from_pretrained(model_id)
            self.vision_model = getattr(vision_wrapper, "vision_model", vision_wrapper)
            self.embed_dim = int(vision_wrapper.config.hidden_size)
        elif "siglip" in lower_id:
            try:
                vision_wrapper = SiglipVisionModel.from_pretrained(model_id)
                self.vision_model = getattr(vision_wrapper, "vision_model", vision_wrapper)
                self.embed_dim = int(vision_wrapper.config.hidden_size)
            except Exception:
                full = AutoModel.from_pretrained(model_id)
                self.vision_model = getattr(full, "vision_model", full)
                cfg = getattr(full.config, "vision_config", full.config)
                self.embed_dim = int(cfg.hidden_size)
                del full
                gc.collect()
        else:
            full = AutoModel.from_pretrained(model_id)
            self.vision_model = getattr(full, "vision_model", full)
            cfg = getattr(full.config, "vision_config", full.config)
            self.embed_dim = int(cfg.hidden_size)
            del full
            gc.collect()

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.vision_model, "gradient_checkpointing_enable"):
            self.vision_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        elif hasattr(self.vision_model, "encoder"):
            self.vision_model.encoder.gradient_checkpointing = True

    def freeze_all(self) -> None:
        for p in self.vision_model.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n_blocks: int = 6) -> int:
        """Freeze entire vision backbone, then unfreeze the final n_blocks + final norm/pooler."""
        self.freeze_all()
        layers = self.vision_model.encoder.layers
        total_layers = len(layers)
        n_unfreeze = min(max(int(n_blocks), 0), total_layers)

        if n_unfreeze > 0:
            for layer in layers[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # Always unfreeze final layernorm and pooler head (e.g., SigLIP2 MAP head) when fine-tuning
        for attr in ("post_layernorm", "head"):
            mod = getattr(self.vision_model, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
        return trainable

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state.mean(dim=1)


class DualHeadClassifier(nn.Module):
    """Dual MLP heads predicting 127 discrete years (1900-2026) and discrete geographic tiles."""

    def __init__(
        self,
        embed_dim: int,
        num_tiles: int,
        num_years: int = NUM_YEARS,
        time_hidden: int = 512,
        space_hidden: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tiles = num_tiles
        self.num_years = num_years

        self.input_norm = nn.LayerNorm(embed_dim)
        self.time_head = nn.Sequential(
            nn.Linear(embed_dim, time_hidden),
            nn.LayerNorm(time_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, num_years),
        )
        self.space_head = nn.Sequential(
            nn.Linear(embed_dim, space_hidden),
            nn.LayerNorm(space_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(space_hidden, num_tiles),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(features)
        return self.time_head(x), self.space_head(x)


class DualHeadViT(nn.Module):
    """End-to-end VisionBackbone + DualHeadClassifier for Stage 2 fine-tuning."""

    def __init__(self, backbone: VisionBackbone, heads: DualHeadClassifier):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(pixel_values)
        return self.heads(feats)


class GaussianYearLoss(nn.Module):
    """Soft cross-entropy against a Gaussian-smoothed target over 127 year classes."""

    def __init__(self, num_years: int = NUM_YEARS, sigma: float = 2.0):
        super().__init__()
        self.num_years = num_years
        self.sigma = float(sigma)
        # Precompute (num_years, num_years) soft target lookup table
        grid = torch.arange(num_years, dtype=torch.float32)
        diff = grid.unsqueeze(0) - grid.unsqueeze(1)  # (C, C)
        targets = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        targets = targets / targets.sum(dim=-1, keepdim=True)
        self.register_buffer("target_table", targets)

    def forward(self, logits: torch.Tensor, year_idx: torch.Tensor) -> torch.Tensor:
        soft_targets = self.target_table[year_idx]  # (B, num_years)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


class GeodesicTileLoss(nn.Module):
    """Geodesic distance-based soft-label cross-entropy over geographic tiles.

    Computes exact great-circle distance (km) on GPU from each training image's
    ground-truth (lat, lon) to every tile centroid, then forms soft targets:
        q(j | i) = softmax_j( -(d_ij - d_i_min) / tau_km )

    Handles unequal tile sizes naturally: nearby neighboring tiles receive strong
    partial credit proportional to their actual physical proximity.
    """

    def __init__(self, cells_df: pd.DataFrame, tau_km: float = 135.0):
        super().__init__()
        self.tau_km = float(tau_km)
        centroid_xyz = latlon_to_xyz(
            cells_df["centroid_lat"].to_numpy(),
            cells_df["centroid_lon"].to_numpy(),
        ).astype(np.float32)
        self.register_buffer("centroid_xyz", torch.from_numpy(centroid_xyz))  # (C, 3)

    def forward(
        self,
        logits: torch.Tensor,
        lats_deg: torch.Tensor,
        lons_deg: torch.Tensor,
    ) -> torch.Tensor:
        lat_r = torch.deg2rad(lats_deg.float())
        lon_r = torch.deg2rad(lons_deg.float())
        cos_lat = torch.cos(lat_r)
        sample_xyz = torch.stack(
            [cos_lat * torch.cos(lon_r), cos_lat * torch.sin(lon_r), torch.sin(lat_r)],
            dim=-1,
        )  # (B, 3)

        # Cosine similarity on unit sphere -> exact spherical angle -> km
        cos_sim = torch.matmul(sample_xyz, self.centroid_xyz.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        dist_km = torch.acos(cos_sim) * EARTH_RADIUS_KM  # (B, C)
        dist_shifted = dist_km - dist_km.min(dim=-1, keepdim=True).values
        soft_targets = F.softmax(-dist_shifted / self.tau_km, dim=-1)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


@torch.no_grad()
def decode_predictions(
    time_logits: np.ndarray,
    space_logits: np.ndarray,
    cells_df: pd.DataFrame,
    topk_space: int = 5,
) -> dict[str, np.ndarray]:
    """Decode model logits into predicted year and (lat, lon) coordinates.

    Returns both primary predictions (expected year, top-1 tile centroid) and
    alternative decodings (argmax year, top-k probability-weighted spherical centroid).
    """
    t_logits = torch.from_numpy(np.asarray(time_logits, dtype=np.float32))
    t_probs = F.softmax(t_logits, dim=-1).numpy()
    years_grid = np.arange(YEAR_MIN, YEAR_MIN + t_probs.shape[1], dtype=np.float64)
    pred_year_exp = (t_probs * years_grid[None, :]).sum(axis=1)
    pred_year_argmax = years_grid[t_probs.argmax(axis=1)]

    s_logits = torch.from_numpy(np.asarray(space_logits, dtype=np.float32))
    s_probs = F.softmax(s_logits, dim=-1)

    centroid_lat = cells_df["centroid_lat"].to_numpy(dtype=np.float64)
    centroid_lon = cells_df["centroid_lon"].to_numpy(dtype=np.float64)
    centroid_xyz = latlon_to_xyz(centroid_lat, centroid_lon)

    pred_tile_id = s_logits.argmax(dim=-1).numpy()
    pred_lat_top1 = centroid_lat[pred_tile_id]
    pred_lon_top1 = centroid_lon[pred_tile_id]

    # Top-k probability-weighted spherical centroid
    k = min(topk_space, s_probs.shape[1])
    top_vals, top_idx = torch.topk(s_probs, k=k, dim=-1)
    top_w = (top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)).numpy()
    top_xyz = centroid_xyz[top_idx.numpy()]  # (N, k, 3)
    weighted_xyz = (top_xyz * top_w[:, :, None]).sum(axis=1)
    pred_lat_topk, pred_lon_topk = xyz_to_latlon(weighted_xyz)

    return {
        "pred_year": pred_year_exp,
        "pred_year_argmax": pred_year_argmax,
        "pred_tile_id": pred_tile_id,
        "pred_lat": pred_lat_top1,
        "pred_lon": pred_lon_top1,
        "pred_lat_topk": pred_lat_topk,
        "pred_lon_topk": pred_lon_topk,
    }
