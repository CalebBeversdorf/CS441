"""Unified single-image inference interface for Linear Probe and Fine-Tuned ViT models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import build_transforms_from_processor, load_dataset_and_tiles
from .metrics import YEAR_MAX, YEAR_MIN
from .models import DualHeadClassifier, DualHeadViT, VisionBackbone, decode_predictions
from .train import resolve_device


@dataclass
class PredictionResult:
    """Standardized output returned by every TimeGuessr AI predictor."""

    pred_year: int
    pred_lat: float
    pred_lon: float
    pred_year_raw: float
    model_name: str
    location_label: str = ""
    reasoning: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTimeguessrPredictor(ABC):
    """Common interface for all TimeGuessr AI models.

    Subclass this and implement `predict(image)` to plug any additional model
    (e.g. FAISS KNN, OCR-enhanced model, or Multimodal LLM) into Notebook 07.
    """

    name: str = "BasePredictor"

    @abstractmethod
    def predict(self, image: Image.Image) -> PredictionResult:
        """Predict `(year, lat, lon)` for a single RGB PIL Image."""


def _reverse_geocode_label(lat: float, lon: float) -> str:
    """Best-effort offline reverse geocoding via `reverse_geocoder`."""
    try:
        import reverse_geocoder as rg

        res = rg.search([(lat, lon)], mode=1)[0]
        parts = [p for p in (res.get("name"), res.get("admin1"), res.get("cc")) if p]
        return ", ".join(parts)
    except Exception:
        return ""


class LinearProbePredictor(BaseTimeguessrPredictor):
    """Frozen VisionBackbone (StreetCLIP or SigLIP2) + trained DualHeadClassifier checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        project_root: str | Path,
        device: str | torch.device | None = None,
        use_topk_centroid: bool = True,
        topk_space: int = 5,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.project_root = Path(project_root)
        self.device = resolve_device(device)
        self.use_topk_centroid = use_topk_centroid
        self.topk_space = topk_space

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        self.model_id = cfg.get("model_id", "geolocal/StreetCLIP")
        self.tile_scheme = cfg.get("tile_scheme", "semantic_medium")
        self.name = f"Linear Probe ({cfg.get('model_label', self.model_id)})"

        _, self.cells_df, _ = load_dataset_and_tiles(self.project_root, tile_scheme=self.tile_scheme)
        _, self.eval_transform = build_transforms_from_processor(self.model_id)

        self.backbone = VisionBackbone(self.model_id, gradient_checkpointing=False).to(self.device)
        self.backbone.eval()

        self.heads = DualHeadClassifier(
            embed_dim=int(ckpt.get("embed_dim", self.backbone.embed_dim)),
            num_tiles=int(ckpt.get("num_tiles", len(self.cells_df))),
            dropout=float(ckpt.get("dropout", 0.2)),
        ).to(self.device)
        self.heads.load_state_dict(ckpt["heads_state_dict"])
        self.heads.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> PredictionResult:
        px = self.eval_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        feats = self.backbone(px)
        t_logits, s_logits = self.heads(feats)
        decoded = decode_predictions(
            t_logits.float().cpu().numpy(),
            s_logits.float().cpu().numpy(),
            self.cells_df,
            topk_space=self.topk_space,
        )
        lat = float(decoded["pred_lat_topk"][0] if self.use_topk_centroid else decoded["pred_lat"][0])
        lon = float(decoded["pred_lon_topk"][0] if self.use_topk_centroid else decoded["pred_lon"][0])
        year_raw = float(decoded["pred_year"][0])
        year_int = int(np.clip(round(year_raw), YEAR_MIN, YEAR_MAX))
        return PredictionResult(
            pred_year=year_int,
            pred_lat=lat,
            pred_lon=lon,
            pred_year_raw=round(year_raw, 2),
            model_name=self.name,
            location_label=_reverse_geocode_label(lat, lon),
            extra={
                "pred_tile_id": int(decoded["pred_tile_id"][0]),
                "pred_year_argmax": int(decoded["pred_year_argmax"][0]),
            },
        )


class FineTunedViTPredictor(BaseTimeguessrPredictor):
    """End-to-end fine-tuned DualHeadViT checkpoint (unfrozen ViT blocks + dual MLP heads)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        project_root: str | Path,
        device: str | torch.device | None = None,
        use_topk_centroid: bool = True,
        topk_space: int = 5,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.project_root = Path(project_root)
        self.device = resolve_device(device)
        self.use_topk_centroid = use_topk_centroid
        self.topk_space = topk_space

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        self.model_id = ckpt.get("model_id") or cfg.get("model_id", "geolocal/StreetCLIP")
        self.tile_scheme = cfg.get("tile_scheme", "semantic_medium")
        self.name = f"Fine-Tuned ViT ({cfg.get('model_label', self.model_id)})"

        _, self.cells_df, _ = load_dataset_and_tiles(self.project_root, tile_scheme=self.tile_scheme)
        _, self.eval_transform = build_transforms_from_processor(self.model_id)

        backbone = VisionBackbone(self.model_id, gradient_checkpointing=False)
        heads = DualHeadClassifier(
            embed_dim=int(ckpt.get("embed_dim", backbone.embed_dim)),
            num_tiles=int(ckpt.get("num_tiles", len(self.cells_df))),
            dropout=float(ckpt.get("dropout", 0.2)),
        )
        self.model = DualHeadViT(backbone=backbone, heads=heads)
        self.model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> PredictionResult:
        px = self.eval_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        t_logits, s_logits = self.model(px)
        decoded = decode_predictions(
            t_logits.float().cpu().numpy(),
            s_logits.float().cpu().numpy(),
            self.cells_df,
            topk_space=self.topk_space,
        )
        lat = float(decoded["pred_lat_topk"][0] if self.use_topk_centroid else decoded["pred_lat"][0])
        lon = float(decoded["pred_lon_topk"][0] if self.use_topk_centroid else decoded["pred_lon"][0])
        year_raw = float(decoded["pred_year"][0])
        year_int = int(np.clip(round(year_raw), YEAR_MIN, YEAR_MAX))
        return PredictionResult(
            pred_year=year_int,
            pred_lat=lat,
            pred_lon=lon,
            pred_year_raw=round(year_raw, 2),
            model_name=self.name,
            location_label=_reverse_geocode_label(lat, lon),
            extra={
                "pred_tile_id": int(decoded["pred_tile_id"][0]),
                "pred_year_argmax": int(decoded["pred_year_argmax"][0]),
            },
        )


def load_predictor(
    model_type: str,
    project_root: str | Path,
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> BaseTimeguessrPredictor:
    """Factory to instantiate a linear-probe or fine-tuned TimeGuessr AI predictor.

    Supported `model_type` presets:
      - "streetclip_finetune"           : Fine-tuned StreetCLIP ViT (`outputs/models/streetclip_finetune/best_checkpoint.pt`)
      - "streetclip_linear_probe"       : Linear-probed StreetCLIP (`outputs/models/streetclip_linear_probe/best_checkpoint.pt`)
      - "streetclip_linear_probe_tuned" : Tuned linear-probed StreetCLIP (`outputs/models/streetclip_linear_probe_tuned/best_checkpoint.pt`)
      - "siglip2_linear_probe"          : Linear-probed SigLIP 2 (`outputs/models/siglip2_linear_probe/best_checkpoint.pt`)
      - "siglip2_finetune"              : Fine-tuned SigLIP 2 (`outputs/models/siglip2_finetune/best_checkpoint.pt`)
      - "linear_probe"                  : Custom linear-probe checkpoint via `checkpoint_path=...`
      - "finetuned"                     : Custom fine-tuned checkpoint via `checkpoint_path=...`
    """
    project_root = Path(project_root)
    models_dir = project_root / "outputs" / "models"
    key = model_type.lower().strip()

    if key == "streetclip_finetune":
        return FineTunedViTPredictor(
            checkpoint_path=kwargs.pop("checkpoint_path", models_dir / "streetclip_finetune" / "best_checkpoint.pt"),
            project_root=project_root,
            device=device,
            **kwargs,
        )
    if key == "siglip2_finetune":
        return FineTunedViTPredictor(
            checkpoint_path=kwargs.pop("checkpoint_path", models_dir / "siglip2_finetune" / "best_checkpoint.pt"),
            project_root=project_root,
            device=device,
            **kwargs,
        )
    if key == "streetclip_linear_probe":
        return LinearProbePredictor(
            checkpoint_path=kwargs.pop("checkpoint_path", models_dir / "streetclip_linear_probe" / "best_checkpoint.pt"),
            project_root=project_root,
            device=device,
            **kwargs,
        )
    if key == "streetclip_linear_probe_tuned":
        pred = LinearProbePredictor(
            checkpoint_path=kwargs.pop("checkpoint_path", models_dir / "streetclip_linear_probe_tuned" / "best_checkpoint.pt"),
            project_root=project_root,
            device=device,
            **kwargs,
        )
        pred.name = f"Linear Probe Tuned ({pred.model_id})" if pred.name.startswith("Linear Probe") else pred.name
        return pred
    if key == "siglip2_linear_probe":
        return LinearProbePredictor(
            checkpoint_path=kwargs.pop("checkpoint_path", models_dir / "siglip2_linear_probe" / "best_checkpoint.pt"),
            project_root=project_root,
            device=device,
            **kwargs,
        )
    if key == "finetuned":
        return FineTunedViTPredictor(project_root=project_root, device=device, **kwargs)
    if key == "linear_probe":
        return LinearProbePredictor(project_root=project_root, device=device, **kwargs)

    raise ValueError(
        f"Unknown model_type '{model_type}'. Choose from: "
        "streetclip_finetune, streetclip_linear_probe, streetclip_linear_probe_tuned, "
        "siglip2_linear_probe, siglip2_finetune, linear_probe, finetuned."
    )
