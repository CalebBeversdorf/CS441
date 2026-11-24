"""Dataset loading, geolocation-safe transforms, and country x decade balanced sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

from .metrics import YEAR_MAX, YEAR_MIN

# Guard against slightly truncated JPEG files in web-scraped datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

TILE_SCHEMES = {
    "semantic_medium": ("semantic_medium_cells.csv", "semantic_medium_id"),
    "s2_coarse": ("s2_coarse_cells.csv", "s2_coarse_id"),
    "s2_medium": ("s2_medium_cells.csv", "s2_medium_id"),
    "s2_fine": ("s2_fine_cells.csv", "s2_fine_id"),
}


def load_dataset_and_tiles(
    project_root: str | Path,
    tile_scheme: str = "semantic_medium",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str]:
    """Load train/val/test_timeguessr splits and tile metadata CSV."""
    project_root = Path(project_root)
    tiling_dir = project_root / "data" / "tiling"

    if tile_scheme not in TILE_SCHEMES:
        raise ValueError(f"Unknown tile_scheme '{tile_scheme}'. Choose from {list(TILE_SCHEMES)}")

    cells_file, tile_col = TILE_SCHEMES[tile_scheme]
    cells_df = pd.read_csv(tiling_dir / cells_file, keep_default_na=False).sort_values("tile_id").reset_index(drop=True)
    df = pd.read_csv(tiling_dir / "dataset_with_tiles.csv", keep_default_na=False)

    # Preserve ISO-2 "NA" (Namibia) cleanly as string
    if "country" in df.columns:
        df["country"] = df["country"].fillna("XX").astype(str).str.upper()
    if "primary_country" in cells_df.columns:
        cells_df["primary_country"] = cells_df["primary_country"].fillna("XX").astype(str).str.upper()

    # Clamp years strictly into [YEAR_MIN, YEAR_MAX] just in case
    df["year"] = pd.to_numeric(df["year"], errors="coerce").clip(YEAR_MIN, YEAR_MAX).astype(int)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype(np.float64)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype(np.float64)
    df["year_idx"] = df["year"] - YEAR_MIN
    df["tile_id"] = pd.to_numeric(df[tile_col], errors="coerce").astype(int)

    splits = {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test_timeguessr": df[df["split"] == "test_timeguessr"].reset_index(drop=True),
    }
    return splits, cells_df, tile_col


def compute_balanced_sample_weights(
    df: pd.DataFrame,
    country_alpha: float = 0.7,
    decade_alpha: float = 0.75,
) -> np.ndarray:
    """Sublinear country x decade weights to counter Russia/modern skew in training."""
    countries = df["country"].fillna("XX").astype(str)
    country_counts = countries.value_counts()
    w_country = countries.map(country_counts).to_numpy(np.float64) ** (country_alpha - 1.0)

    decades = (df["year"].to_numpy() // 10) * 10
    decade_series = pd.Series(decades)
    decade_counts = decade_series.value_counts()
    w_decade = decade_series.map(decade_counts).to_numpy(np.float64) ** (decade_alpha - 1.0)

    weights = w_country * w_decade
    weights = weights / weights.mean()
    return weights


def _get_field(obj, key: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_processor_spec(model_id: str) -> tuple[int, int, bool, list[float], list[float]]:
    """Read exact resolution, crop mode, and normalization stats from HF AutoImageProcessor."""
    proc = AutoImageProcessor.from_pretrained(model_id)

    size_cfg = getattr(proc, "size", None)
    crop_cfg = getattr(proc, "crop_size", None)
    do_center_crop = bool(getattr(proc, "do_center_crop", False))

    crop_h, crop_w = _get_field(crop_cfg, "height"), _get_field(crop_cfg, "width")
    size_h, size_w = _get_field(size_cfg, "height"), _get_field(size_cfg, "width")
    shortest = _get_field(size_cfg, "shortest_edge")

    if do_center_crop and crop_h is not None and crop_w is not None:
        h, w = int(crop_h), int(crop_w)
    elif size_h is not None and size_w is not None:
        h, w = int(size_h), int(size_w)
    elif shortest is not None:
        h = w = int(shortest)
    elif isinstance(size_cfg, int):
        h = w = int(size_cfg)
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        vcfg = getattr(cfg, "vision_config", cfg)
        h = w = int(getattr(vcfg, "image_size", 336))

    mean = list(getattr(proc, "image_mean", [0.5, 0.5, 0.5]))
    std = list(getattr(proc, "image_std", [0.5, 0.5, 0.5]))
    return h, w, do_center_crop, mean, std


def build_transforms_from_processor(
    model_id: str,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build fast torchvision train (geolocation-safe) and eval transforms matching the HF processor.

    Note: Intentionally omits RandomHorizontalFlip because flipping reverses street signs
    and left/right-hand driving cues.
    """
    h, w, do_center_crop, mean, std = _extract_processor_spec(model_id)

    if do_center_crop:
        eval_resize = [
            transforms.Resize(h, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((h, w)),
        ]
    else:
        eval_resize = [
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        ]

    eval_transform = transforms.Compose(
        [
            *eval_resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (h, w),
                scale=(0.8, 1.0),
                ratio=(0.85, 1.18),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


class TimeguessrImageDataset(Dataset):
    """Loads images + (year_idx, tile_id, lat, lon) for ViT extraction or fine-tuning."""

    def __init__(
        self,
        df: pd.DataFrame,
        project_root: str | Path,
        transform: transforms.Compose,
    ):
        self.project_root = Path(project_root)
        # Normalize Windows/POSIX separators cleanly
        self.image_paths = [
            self.project_root / Path(str(p).replace("\\", "/"))
            for p in df["image_path"].tolist()
        ]
        self.year_idx = df["year_idx"].to_numpy(dtype=np.int64)
        self.tile_id = df["tile_id"].to_numpy(dtype=np.int64)
        self.lats = df["latitude"].to_numpy(dtype=np.float32)
        self.lons = df["longitude"].to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                pixel_values = self.transform(img)
        except Exception:
            # Fallback black image so a single corrupt file never crashes a multi-hour run
            pixel_values = self.transform(Image.new("RGB", (384, 384), (0, 0, 0)))

        return {
            "pixel_values": pixel_values,
            "year_idx": torch.tensor(self.year_idx[idx], dtype=torch.long),
            "tile_id": torch.tensor(self.tile_id[idx], dtype=torch.long),
            "lat": torch.tensor(self.lats[idx], dtype=torch.float32),
            "lon": torch.tensor(self.lons[idx], dtype=torch.float32),
        }


class EmbeddingDataset(Dataset):
    """Fast in-memory dataset over precomputed ViT embeddings for Stage 1 linear probing."""

    def __init__(self, embeddings: np.ndarray | torch.Tensor, df: pd.DataFrame):
        if isinstance(embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(embeddings)
        else:
            self.embeddings = embeddings
        self.year_idx = torch.from_numpy(df["year_idx"].to_numpy(dtype=np.int64))
        self.tile_id = torch.from_numpy(df["tile_id"].to_numpy(dtype=np.int64))
        self.lats = torch.from_numpy(df["latitude"].to_numpy(dtype=np.float32))
        self.lons = torch.from_numpy(df["longitude"].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return len(self.year_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "embedding": self.embeddings[idx].float(),
            "year_idx": self.year_idx[idx],
            "tile_id": self.tile_id[idx],
            "lat": self.lats[idx],
            "lon": self.lons[idx],
        }
