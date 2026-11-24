"""Embedding extraction, Stage 1 linear probing, Stage 2 fine-tuning, and standardized I/O."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .data import (
    EmbeddingDataset,
    TimeguessrImageDataset,
    build_transforms_from_processor,
    compute_balanced_sample_weights,
)
from .metrics import evaluate_predictions
from .models import (
    DualHeadClassifier,
    DualHeadViT,
    GaussianYearLoss,
    GeodesicTileLoss,
    VisionBackbone,
    decode_predictions,
)


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")


def _get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _split_paths_sha256(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 fingerprint of ordered normalized image_paths in a split."""
    h = hashlib.sha256()
    if "image_path" in df.columns:
        for p in df["image_path"].astype(str):
            h.update(p.replace("\\", "/").encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def load_cached_embeddings(
    model_id: str,
    splits: dict[str, pd.DataFrame],
    project_root: str | Path,
) -> dict[str, np.ndarray]:
    """Load precomputed ViT embeddings from disk and sanity-check them against `splits`.

    Checks:
      1. Every split `.npy` exists on disk.
      2. Each array is 2D and `arr.shape[0] == len(splits[split_name])`.
      3. Embedding dimension is identical across all splits.
      4. No NaN/Inf values and non-zero norms.
      5. If `<split>_manifest.json` is present, verifies exact `image_path` order hash (`paths_sha256`).
    """
    project_root = Path(project_root)
    cache_dir = project_root / "data" / "embeddings" / _model_slug(model_id)
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"No cached embeddings directory at {cache_dir}. Set CFG['use_cached_embeddings'] = False to extract."
        )

    cached: dict[str, np.ndarray] = {}
    embed_dim: int | None = None
    manifest_verified: list[str] = []
    manifest_missing: list[str] = []

    for split_name, df in splits.items():
        npy_path = cache_dir / f"{split_name}_embeddings.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Missing cached embeddings for split '{split_name}': {npy_path}. "
                "Set CFG['use_cached_embeddings'] = False to extract."
            )

        arr = np.load(npy_path)
        if arr.ndim != 2:
            raise ValueError(f"Cached '{split_name}' embeddings must be 2D, got shape {arr.shape}.")
        if arr.shape[0] != len(df):
            raise ValueError(
                f"Row count mismatch for split '{split_name}': cached embeddings have {arr.shape[0]:,} rows, "
                f"but loaded dataset has {len(df):,} rows. Set CFG['use_cached_embeddings'] = False to re-extract."
            )
        if embed_dim is None:
            embed_dim = int(arr.shape[1])
        elif int(arr.shape[1]) != embed_dim:
            raise ValueError(
                f"Embedding dim mismatch for split '{split_name}': got {arr.shape[1]}, expected {embed_dim}."
            )

        if not np.isfinite(arr).all():
            raise ValueError(f"Cached '{split_name}' embeddings contain NaN or Inf values.")

        # Quick norm check on a sample (first, middle, last rows) to catch degenerate all-zero caches
        if len(arr) > 0:
            sample_idx = sorted({0, len(arr) // 2, len(arr) - 1})
            sample_norms = np.linalg.norm(arr[sample_idx].astype(np.float32), axis=1)
            if np.any(sample_norms < 1e-6):
                raise ValueError(f"Cached '{split_name}' embeddings contain near-zero norm vectors.")

        manifest_path = cache_dir / f"{split_name}_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hash = _split_paths_sha256(df)
            if manifest.get("paths_sha256") != expected_hash:
                raise ValueError(
                    f"Dataset fingerprint mismatch for split '{split_name}': cached image order/paths do not match "
                    "the loaded dataset_with_tiles.csv. Set CFG['use_cached_embeddings'] = False to re-extract."
                )
            manifest_verified.append(split_name)
        else:
            manifest_missing.append(split_name)

        cached[split_name] = arr

    summary = ", ".join(f"{k}={v.shape}" for k, v in cached.items())
    print(f"[CACHE OK] Loaded & verified cached embeddings for {model_id} from {cache_dir} ({summary})")
    if manifest_verified:
        print(f"  Exact image_path SHA-256 manifest verified for: {manifest_verified}")
    if manifest_missing:
        print(
            f"  Note: No manifest.json found for {manifest_missing} (created before manifest tracking); "
            "row count, dim, finite, and norm checks passed."
        )
    return cached


@torch.no_grad()
def extract_and_cache_embeddings(
    model_id: str,
    splits: dict[str, pd.DataFrame],
    project_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str | torch.device = "cuda",
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Extract frozen ViT vision embeddings once and cache to data/embeddings/<slug>/."""
    project_root = Path(project_root)
    cache_dir = project_root / "data" / "embeddings" / _model_slug(model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: dict[str, np.ndarray] = {}
    missing_splits = []
    for split_name, df in splits.items():
        npy_path = cache_dir / f"{split_name}_embeddings.npy"
        manifest_path = cache_dir / f"{split_name}_manifest.json"
        if npy_path.exists() and not force:
            arr = np.load(npy_path)
            hash_ok = True
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                hash_ok = manifest.get("paths_sha256") == _split_paths_sha256(df)
            if arr.ndim == 2 and len(arr) == len(df) and hash_ok:
                cached[split_name] = arr
                continue
        missing_splits.append(split_name)

    if not missing_splits:
        print(f"Loaded cached embeddings for {model_id} from {cache_dir}")
        return cached

    device = torch.device(device)
    amp_dtype = _get_amp_dtype(device)
    _, eval_tf = build_transforms_from_processor(model_id)

    print(f"Extracting embeddings for {model_id} on {device} ({amp_dtype}) for splits: {missing_splits}")
    backbone = VisionBackbone(model_id, gradient_checkpointing=False).to(device)
    backbone.eval()

    for split_name in missing_splits:
        df = splits[split_name]
        ds = TimeguessrImageDataset(df, project_root=project_root, transform=eval_tf)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        chunks = []
        for batch in tqdm(loader, desc=f"Extract {split_name}", leave=True):
            px = batch["pixel_values"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                feats = backbone(px)
            chunks.append(feats.float().cpu().numpy().astype(np.float16))
        emb = np.concatenate(chunks, axis=0)
        npy_path = cache_dir / f"{split_name}_embeddings.npy"
        np.save(npy_path, emb)
        manifest_path = cache_dir / f"{split_name}_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "split": split_name,
                    "num_rows": int(emb.shape[0]),
                    "embed_dim": int(emb.shape[1]),
                    "dtype": str(emb.dtype),
                    "paths_sha256": _split_paths_sha256(df),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cached[split_name] = emb
        print(f"Saved {split_name}: shape={emb.shape}, size={emb.nbytes / 1e6:.1f} MB -> {npy_path}")

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached


@torch.no_grad()
def evaluate_model_on_dataset(
    model: DualHeadClassifier | DualHeadViT,
    loader: DataLoader,
    df: pd.DataFrame,
    cells_df: pd.DataFrame,
    device: str | torch.device = "cuda",
    is_embedding_loader: bool = False,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and compute exact Timeguessr metrics + raw logits."""
    device = torch.device(device)
    amp_dtype = _get_amp_dtype(device)
    model.eval()

    time_logits_list = []
    space_logits_list = []

    for batch in loader:
        inp_key = "embedding" if is_embedding_loader else "pixel_values"
        x = batch[inp_key].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            t_out, s_out = model(x)
        time_logits_list.append(t_out.float().cpu().numpy())
        space_logits_list.append(s_out.float().cpu().numpy())

    time_logits = np.concatenate(time_logits_list, axis=0)
    space_logits = np.concatenate(space_logits_list, axis=0)

    decoded = decode_predictions(time_logits, space_logits, cells_df)
    # Primary decoding: top-k probability-weighted spherical centroid + expected year
    metrics, pred_df = evaluate_predictions(
        df=df,
        pred_lat=decoded["pred_lat_topk"],
        pred_lon=decoded["pred_lon_topk"],
        pred_year=decoded["pred_year"],
        pred_tile_id=decoded["pred_tile_id"],
        cells_df=cells_df,
    )
    # Also log top-1 tile centroid location score as a diagnostic comparison
    metrics_top1, _ = evaluate_predictions(
        df=df,
        pred_lat=decoded["pred_lat"],
        pred_lon=decoded["pred_lon"],
        pred_year=decoded["pred_year"],
        pred_tile_id=decoded["pred_tile_id"],
        cells_df=cells_df,
    )
    metrics["loc_score_mean_top1"] = metrics_top1["loc_score_mean"]
    metrics["total_score_mean_top1"] = metrics_top1["total_score_mean"]
    metrics["loc_score_mean_topk"] = metrics["loc_score_mean"]
    metrics["total_score_mean_topk"] = metrics["total_score_mean"]
    return metrics, pred_df, time_logits, space_logits


def save_run_outputs(
    output_dir: str | Path,
    run_config: dict[str, Any],
    history: list[dict[str, Any]],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    test_pred_df: pd.DataFrame,
    test_space_logits: np.ndarray,
    checkpoint_state: dict[str, Any] | None = None,
) -> Path:
    """Save standardized outputs for warm-starting, 4-way comparison, and Notebook 05 OCR."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_state is not None:
        torch.save(checkpoint_state, output_dir / "best_checkpoint.pt")

    payload = {
        "config": run_config,
        "val_metrics": val_metrics,
        "test_timeguessr_metrics": test_metrics,
        "history": history,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    np.save(output_dir / "test_space_logits.npy", test_space_logits.astype(np.float16))
    print(f"Saved standardized outputs to {output_dir}")
    return output_dir


def train_linear_probe(
    embeddings: dict[str, np.ndarray],
    splits: dict[str, pd.DataFrame],
    cells_df: pd.DataFrame,
    output_dir: str | Path,
    run_config: dict[str, Any],
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    sigma_year: float = 2.0,
    tau_geo_km: float = 135.0,
    w_time: float = 1.0,
    dropout: float = 0.2,
    country_alpha: float = 0.7,
    device: str | torch.device = "cuda",
    seed: int = 42,
) -> dict[str, Any]:
    """Stage 1: Train DualHeadClassifier on frozen precomputed ViT embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    amp_dtype = _get_amp_dtype(device)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test_timeguessr"]

    train_ds = EmbeddingDataset(embeddings["train"], train_df)
    val_ds = EmbeddingDataset(embeddings["val"], val_df)
    test_ds = EmbeddingDataset(embeddings["test_timeguessr"], test_df)

    sample_weights = compute_balanced_sample_weights(train_df, country_alpha=country_alpha)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_df),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    embed_dim = int(embeddings["train"].shape[1])
    model = DualHeadClassifier(
        embed_dim=embed_dim,
        num_tiles=len(cells_df),
        dropout=dropout,
    ).to(device)

    time_criterion = GaussianYearLoss(sigma=sigma_year).to(device)
    space_criterion = GeodesicTileLoss(cells_df=cells_df, tau_km=tau_geo_km).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_score = -1.0
    best_state = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_time = 0.0
        running_space = 0.0
        n_batches = 0

        for batch in train_loader:
            emb = batch["embedding"].to(device, non_blocking=True)
            year_idx = batch["year_idx"].to(device, non_blocking=True)
            lats = batch["lat"].to(device, non_blocking=True)
            lons = batch["lon"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                t_logits, s_logits = model(emb)
                loss_t = time_criterion(t_logits, year_idx)
                loss_s = space_criterion(s_logits, lats, lons)
                loss = loss_s + w_time * loss_t

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item())
            running_time += float(loss_t.item())
            running_space += float(loss_s.item())
            n_batches += 1

        scheduler.step()
        val_metrics, _, _, _ = evaluate_model_on_dataset(
            model, val_loader, val_df, cells_df, device=device, is_embedding_loader=True
        )
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(n_batches, 1), 4),
            "train_time_loss": round(running_time / max(n_batches, 1), 4),
            "train_space_loss": round(running_space / max(n_batches, 1), 4),
            "val_total_score": val_metrics["total_score_mean"],
            "val_loc_score": val_metrics["loc_score_mean"],
            "val_year_score": val_metrics["year_score_mean"],
            "val_median_km": val_metrics["dist_err_median_km"],
            "val_year_mae": val_metrics["year_mae"],
            "lr": float(scheduler.get_last_lr()[0]),
            "sec": round(elapsed, 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} ({elapsed:4.1f}s) | "
            f"loss={row['train_loss']:.4f} (s={row['train_space_loss']:.3f}, t={row['train_time_loss']:.3f}) | "
            f"val_total={row['val_total_score']:.1f} (loc={row['val_loc_score']:.1f}, yr={row['val_year_score']:.1f}) | "
            f"med_km={row['val_median_km']:.1f} | yr_mae={row['val_year_mae']:.2f}"
        )

        if val_metrics["total_score_mean"] > best_val_score:
            best_val_score = val_metrics["total_score_mean"]
            best_state = {
                "heads_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "embed_dim": embed_dim,
                "num_tiles": len(cells_df),
                "dropout": dropout,
                "config": run_config,
                "best_epoch": epoch,
            }

    assert best_state is not None
    model.load_state_dict(best_state["heads_state_dict"])
    val_metrics, _, _, _ = evaluate_model_on_dataset(
        model, val_loader, val_df, cells_df, device=device, is_embedding_loader=True
    )
    test_metrics, test_pred_df, _, test_space_logits = evaluate_model_on_dataset(
        model, test_loader, test_df, cells_df, device=device, is_embedding_loader=True
    )

    save_run_outputs(
        output_dir=output_dir,
        run_config=run_config,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_pred_df=test_pred_df,
        test_space_logits=test_space_logits,
        checkpoint_state=best_state,
    )
    return {
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_pred_df": test_pred_df,
        "checkpoint_path": Path(output_dir) / "best_checkpoint.pt",
    }


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Write checkpoint to a temporary file and atomically replace target (crash-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def train_finetune(
    model_id: str,
    splits: dict[str, pd.DataFrame],
    cells_df: pd.DataFrame,
    project_root: str | Path,
    output_dir: str | Path,
    run_config: dict[str, Any],
    warmstart_checkpoint: str | Path | None = None,
    unfreeze_last_n: int = 6,
    epochs: int = 5,
    batch_size: int = 16,
    grad_accum_steps: int = 4,
    backbone_lr: float = 2e-5,
    head_lr: float = 3e-4,
    weight_decay: float = 1e-2,
    warmup_ratio: float = 0.05,
    sigma_year: float = 2.0,
    tau_geo_km: float = 135.0,
    w_time: float = 1.0,
    dropout: float = 0.2,
    country_alpha: float = 0.7,
    num_workers: int = 0,
    device: str | torch.device = "cuda",
    seed: int = 42,
    resume: bool = True,
) -> dict[str, Any]:
    """Stage 2: Fine-tune last N ViT blocks + warm-started dual MLP heads in bfloat16.

    Saves `last_checkpoint.pt` and `best_checkpoint.pt` atomically after every epoch.
    If `resume=True` and `output_dir/last_checkpoint.pt` exists, resumes automatically
    from the next epoch with full model, optimizer, scheduler, and history state.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    amp_dtype = _get_amp_dtype(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = output_dir / "last_checkpoint.pt"
    best_ckpt_path = output_dir / "best_checkpoint.pt"

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test_timeguessr"]

    train_tf, eval_tf = build_transforms_from_processor(model_id)
    train_ds = TimeguessrImageDataset(train_df, project_root=project_root, transform=train_tf)
    val_ds = TimeguessrImageDataset(val_df, project_root=project_root, transform=eval_tf)
    test_ds = TimeguessrImageDataset(test_df, project_root=project_root, transform=eval_tf)

    sample_weights = compute_balanced_sample_weights(train_df, country_alpha=country_alpha)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_df),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    eval_batch_size = max(batch_size * 2, 32)
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    backbone = VisionBackbone(model_id, gradient_checkpointing=True)
    trainable_bb_params = backbone.unfreeze_last_n_blocks(unfreeze_last_n)
    heads = DualHeadClassifier(
        embed_dim=backbone.embed_dim,
        num_tiles=len(cells_df),
        dropout=dropout,
    )

    can_resume = resume and last_ckpt_path.exists()
    if not can_resume:
        if warmstart_checkpoint is not None and Path(warmstart_checkpoint).exists():
            ckpt = torch.load(warmstart_checkpoint, map_location="cpu", weights_only=False)
            heads.load_state_dict(ckpt["heads_state_dict"])
            print(f"Warm-started dual heads from {warmstart_checkpoint}")
        else:
            print("Warning: No warm-start checkpoint found; initializing dual heads randomly.")

    model = DualHeadViT(backbone=backbone, heads=heads).to(device)
    trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    trainable_head_params = sum(p.numel() for p in model.heads.parameters() if p.requires_grad)
    print(
        f"Fine-tuning {model_id} | unfrozen last {unfreeze_last_n} blocks "
        f"({trainable_bb_params / 1e6:.2f}M backbone + {trainable_head_params / 1e6:.2f}M heads) | "
        f"eff_batch={batch_size * grad_accum_steps}"
    )

    time_criterion = GaussianYearLoss(sigma=sigma_year).to(device)
    space_criterion = GeodesicTileLoss(cells_df=cells_df, tau_km=tau_geo_km).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.heads.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    total_opt_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    warmup_steps = max(1, int(total_opt_steps * warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val_score = -1.0
    best_state = None
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if can_resume:
        last_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        if last_ckpt.get("model_id") == model_id and last_ckpt.get("unfreeze_last_n") == unfreeze_last_n:
            model.load_state_dict(last_ckpt["trainable_state_dict"], strict=False)
            optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(last_ckpt["scheduler_state_dict"])
            if use_scaler and last_ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(last_ckpt["scaler_state_dict"])
            best_val_score = float(last_ckpt.get("best_val_score", -1.0))
            history = list(last_ckpt.get("history", []))
            start_epoch = int(last_ckpt["epoch"]) + 1
            if best_ckpt_path.exists():
                best_state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            print(
                f"Resuming fine-tuning from epoch {start_epoch}/{epochs} "
                f"(completed={start_epoch - 1}, best_val_score={best_val_score:.1f}) via {last_ckpt_path}"
            )
        else:
            print(
                f"Found {last_ckpt_path} but config mismatch "
                f"(ckpt={last_ckpt.get('model_id')}/{last_ckpt.get('unfreeze_last_n')}); starting fresh."
            )
            if warmstart_checkpoint is not None and Path(warmstart_checkpoint).exists():
                ckpt = torch.load(warmstart_checkpoint, map_location="cpu", weights_only=False)
                model.heads.load_state_dict(ckpt["heads_state_dict"])
                print(f"Warm-started dual heads from {warmstart_checkpoint}")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_time = 0.0
        running_space = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Finetune Ep {epoch}/{epochs}", leave=True)
        for step_idx, batch in enumerate(pbar, start=1):
            px = batch["pixel_values"].to(device, non_blocking=True)
            year_idx = batch["year_idx"].to(device, non_blocking=True)
            lats = batch["lat"].to(device, non_blocking=True)
            lons = batch["lon"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                t_logits, s_logits = model(px)
                loss_t = time_criterion(t_logits, year_idx)
                loss_s = space_criterion(s_logits, lats, lons)
                loss = (loss_s + w_time * loss_t) / grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item() * grad_accum_steps)
            running_time += float(loss_t.item())
            running_space += float(loss_s.item())
            n_batches += 1

            if step_idx % grad_accum_steps == 0 or step_idx == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step_idx % 100 == 0:
                pbar.set_postfix(
                    loss=f"{running_loss / n_batches:.3f}",
                    bb_lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

        val_metrics, _, _, _ = evaluate_model_on_dataset(
            model, val_loader, val_df, cells_df, device=device, is_embedding_loader=False
        )
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(n_batches, 1), 4),
            "train_time_loss": round(running_time / max(n_batches, 1), 4),
            "train_space_loss": round(running_space / max(n_batches, 1), 4),
            "val_total_score": val_metrics["total_score_mean"],
            "val_loc_score": val_metrics["loc_score_mean"],
            "val_year_score": val_metrics["year_score_mean"],
            "val_median_km": val_metrics["dist_err_median_km"],
            "val_year_mae": val_metrics["year_mae"],
            "backbone_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "sec": round(elapsed, 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} ({elapsed / 60:.1f} min) | "
            f"loss={row['train_loss']:.4f} (s={row['train_space_loss']:.3f}, t={row['train_time_loss']:.3f}) | "
            f"val_total={row['val_total_score']:.1f} (loc={row['val_loc_score']:.1f}, yr={row['val_year_score']:.1f}) | "
            f"med_km={row['val_median_km']:.1f} | yr_mae={row['val_year_mae']:.2f}"
        )

        trainable_state_dict = {
            k: v.cpu().clone() for k, v in model.state_dict().items() if k in trainable_keys
        }

        if val_metrics["total_score_mean"] > best_val_score:
            best_val_score = val_metrics["total_score_mean"]
            # Save only trainable weights (unfrozen blocks + heads) to keep checkpoint compact
            best_state = {
                "trainable_state_dict": trainable_state_dict,
                "heads_state_dict": {k: v.cpu().clone() for k, v in model.heads.state_dict().items()},
                "model_id": model_id,
                "unfreeze_last_n": unfreeze_last_n,
                "embed_dim": backbone.embed_dim,
                "num_tiles": len(cells_df),
                "dropout": dropout,
                "config": run_config,
                "best_epoch": epoch,
            }
            _atomic_torch_save(best_state, best_ckpt_path)

        # Save crash-safe resume checkpoint after every finished epoch
        last_state = {
            "epoch": epoch,
            "trainable_state_dict": trainable_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_scaler else None,
            "best_val_score": best_val_score,
            "history": history,
            "model_id": model_id,
            "unfreeze_last_n": unfreeze_last_n,
            "embed_dim": backbone.embed_dim,
            "num_tiles": len(cells_df),
            "dropout": dropout,
            "config": run_config,
        }
        _atomic_torch_save(last_state, last_ckpt_path)
        print(f"  [checkpoint] saved epoch {epoch} -> {last_ckpt_path.name} (best_val={best_val_score:.1f})")

    assert best_state is not None
    model.load_state_dict(best_state["trainable_state_dict"], strict=False)
    val_metrics, _, _, _ = evaluate_model_on_dataset(
        model, val_loader, val_df, cells_df, device=device, is_embedding_loader=False
    )
    test_metrics, test_pred_df, _, test_space_logits = evaluate_model_on_dataset(
        model, test_loader, test_df, cells_df, device=device, is_embedding_loader=False
    )

    save_run_outputs(
        output_dir=output_dir,
        run_config=run_config,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_pred_df=test_pred_df,
        test_space_logits=test_space_logits,
        checkpoint_state=best_state,
    )
    return {
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_pred_df": test_pred_df,
        "checkpoint_path": Path(output_dir) / "best_checkpoint.pt",
    }
