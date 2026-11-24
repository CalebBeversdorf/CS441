"""Geospatial math and exact Timeguessr scoring metrics (from formulas.txt)."""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
YEAR_MIN = 1900
YEAR_MAX = 2026
NUM_YEARS = YEAR_MAX - YEAR_MIN + 1  # 127 discrete classes: 1900..2026


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude (degrees) to 3D unit-sphere vectors (N, 3)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])


def xyz_to_latlon(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3D unit-sphere vectors (N, 3) to (lat_deg, lon_deg)."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = xyz / np.maximum(norms, 1e-12)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return lat, lon


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Vectorized great-circle distance in kilometers."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def timeguessr_year_score(pred_year: np.ndarray | float, true_year: np.ndarray | float) -> np.ndarray:
    """Exact year score formula from formulas.txt:
        year = 5000 / (1 + (yearsOff / 12)^2)
    """
    years_off = np.abs(np.asarray(pred_year, dtype=np.float64) - np.asarray(true_year, dtype=np.float64))
    score = 5000.0 / (1.0 + (years_off / 12.0) ** 2)
    return np.clip(score, 0.0, 5000.0)


def timeguessr_location_score(dist_km: np.ndarray | float) -> np.ndarray:
    """Exact location score formula from formulas.txt:
        d = max(km - 0.05, 0)
        location = 5000 * exp(-(d/6000)^0.62 - (d/3000)^1.5)
    """
    d = np.maximum(np.asarray(dist_km, dtype=np.float64) - 0.05, 0.0)
    score = 5000.0 * np.exp(-((d / 6000.0) ** 0.62) - ((d / 3000.0) ** 1.5))
    return np.clip(score, 0.0, 5000.0)


def evaluate_predictions(
    df: pd.DataFrame,
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    pred_year: np.ndarray,
    pred_tile_id: np.ndarray | None = None,
    cells_df: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute standardized Timeguessr metrics and per-sample prediction DataFrame."""
    true_lat = df["latitude"].to_numpy(dtype=np.float64)
    true_lon = df["longitude"].to_numpy(dtype=np.float64)
    true_year = df["year"].to_numpy(dtype=np.float64)

    pred_lat = np.asarray(pred_lat, dtype=np.float64)
    pred_lon = np.asarray(pred_lon, dtype=np.float64)
    pred_year = np.asarray(pred_year, dtype=np.float64)

    dist_km = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
    year_err = np.abs(pred_year - true_year)

    loc_score = timeguessr_location_score(dist_km)
    yr_score = timeguessr_year_score(pred_year, true_year)
    total_score = loc_score + yr_score

    metrics: dict[str, float] = {
        "n_samples": int(len(df)),
        "total_score_mean": round(float(np.mean(total_score)), 2),
        "total_score_median": round(float(np.median(total_score)), 2),
        "loc_score_mean": round(float(np.mean(loc_score)), 2),
        "loc_score_median": round(float(np.median(loc_score)), 2),
        "year_score_mean": round(float(np.mean(yr_score)), 2),
        "year_score_median": round(float(np.median(yr_score)), 2),
        "dist_err_median_km": round(float(np.median(dist_km)), 2),
        "dist_err_mean_km": round(float(np.mean(dist_km)), 2),
        "within_25km_pct": round(float((dist_km <= 25.0).mean() * 100.0), 2),
        "within_200km_pct": round(float((dist_km <= 200.0).mean() * 100.0), 2),
        "within_750km_pct": round(float((dist_km <= 750.0).mean() * 100.0), 2),
        "within_2500km_pct": round(float((dist_km <= 2500.0).mean() * 100.0), 2),
        "year_mae": round(float(np.mean(year_err)), 2),
        "year_median_err": round(float(np.median(year_err)), 2),
        "year_within_5_pct": round(float((year_err <= 5.0).mean() * 100.0), 2),
        "year_within_10_pct": round(float((year_err <= 10.0).mean() * 100.0), 2),
        "year_within_15_pct": round(float((year_err <= 15.0).mean() * 100.0), 2),
    }

    pred_df = pd.DataFrame(
        {
            "image_path": df["image_path"].to_numpy(),
            "source": df["source"].to_numpy() if "source" in df.columns else "",
            "true_country": df["country"].to_numpy() if "country" in df.columns else "",
            "true_year": true_year.astype(int),
            "pred_year": np.rint(pred_year).astype(int),
            "pred_year_raw": np.round(pred_year, 2),
            "year_err": np.round(year_err, 2),
            "true_lat": true_lat,
            "true_lon": true_lon,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "dist_km": np.round(dist_km, 2),
            "loc_score": np.round(loc_score, 2),
            "year_score": np.round(yr_score, 2),
            "total_score": np.round(total_score, 2),
        }
    )

    if pred_tile_id is not None:
        pred_tile_id = np.asarray(pred_tile_id, dtype=np.int64)
        pred_df["pred_tile_id"] = pred_tile_id
        if cells_df is not None and "primary_country" in cells_df.columns and "country" in df.columns:
            tile_to_cc = dict(zip(cells_df["tile_id"].astype(int), cells_df["primary_country"].astype(str)))
            pred_cc = np.array([tile_to_cc.get(int(t), "XX") for t in pred_tile_id])
            true_cc = df["country"].astype(str).str.upper().to_numpy()
            pred_df["pred_country"] = pred_cc
            metrics["country_acc_pct"] = round(float((pred_cc == true_cc).mean() * 100.0), 2)

    return metrics, pred_df
