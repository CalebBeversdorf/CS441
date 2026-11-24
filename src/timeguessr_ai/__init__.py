"""Shared utilities for Timeguessr AI dual-head ViT training and evaluation."""

from .metrics import (
    EARTH_RADIUS_KM,
    YEAR_MAX,
    YEAR_MIN,
    NUM_YEARS,
    haversine_km,
    latlon_to_xyz,
    xyz_to_latlon,
    timeguessr_location_score,
    timeguessr_year_score,
    evaluate_predictions,
)
from .data import (
    load_dataset_and_tiles,
    compute_balanced_sample_weights,
    build_transforms_from_processor,
    TimeguessrImageDataset,
    EmbeddingDataset,
)
from .models import (
    VisionBackbone,
    DualHeadClassifier,
    DualHeadViT,
    GaussianYearLoss,
    GeodesicTileLoss,
    decode_predictions,
)
from .train import (
    extract_and_cache_embeddings,
    train_linear_probe,
    train_finetune,
    evaluate_model_on_dataset,
    save_run_outputs,
)

__all__ = [
    "EARTH_RADIUS_KM",
    "YEAR_MIN",
    "YEAR_MAX",
    "NUM_YEARS",
    "haversine_km",
    "latlon_to_xyz",
    "xyz_to_latlon",
    "timeguessr_location_score",
    "timeguessr_year_score",
    "evaluate_predictions",
    "load_dataset_and_tiles",
    "compute_balanced_sample_weights",
    "build_transforms_from_processor",
    "TimeguessrImageDataset",
    "EmbeddingDataset",
    "VisionBackbone",
    "DualHeadClassifier",
    "DualHeadViT",
    "GaussianYearLoss",
    "GeodesicTileLoss",
    "decode_predictions",
    "extract_and_cache_embeddings",
    "train_linear_probe",
    "train_finetune",
    "evaluate_model_on_dataset",
    "save_run_outputs",
]
