"""TimeGuessr live game fetcher, HTML/SvelteKit parser, and player distribution client."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import requests
from PIL import Image

from .metrics import haversine_km, timeguessr_location_score, timeguessr_year_score

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class RoundPlayerStats:
    """Per-round aggregate statistics across all TimeGuessr players."""

    round_num: int
    avg_score: float | None = None
    avg_years_off: float | None = None
    avg_distance_km: float | None = None


@dataclass
class PlayerDistribution:
    """All-player score distribution returned by TimeGuessr's game-stats API."""

    avg_score: float
    bin_size: int
    histogram: list[float]
    avg_year_accuracy: float | None = None
    avg_location_accuracy: float | None = None
    avg_duration_sec: float | None = None
    rounds: list[RoundPlayerStats] = field(default_factory=list)
    source_endpoint: str = ""

    def bin_edges(self) -> np.ndarray:
        return np.arange(len(self.histogram) + 1, dtype=np.float64) * self.bin_size

    def bin_centers(self) -> np.ndarray:
        return (np.arange(len(self.histogram), dtype=np.float64) + 0.5) * self.bin_size

    def percentile_of(self, score: float) -> float:
        """Estimated percentage of players scored below `score` (0..100)."""
        probs = np.asarray(self.histogram, dtype=np.float64)
        total = probs.sum()
        if total <= 0:
            return 0.0
        probs = probs / total
        score = float(np.clip(score, 0.0, len(probs) * self.bin_size))
        idx = int(score // self.bin_size)
        if idx >= len(probs):
            return 100.0
        frac = (score - idx * self.bin_size) / self.bin_size
        cdf = float(probs[:idx].sum() + frac * probs[idx])
        return round(cdf * 100.0, 1)

    def round_stats(self, round_num: int) -> RoundPlayerStats | None:
        for r in self.rounds:
            if r.round_num == round_num:
                return r
        return None


@dataclass
class TimeguessrRound:
    """Single round in a TimeGuessr game."""

    round_num: int
    image_url: str
    true_year: int
    true_lat: float
    true_lon: float
    country: str = ""
    description: str = ""
    license: str = ""
    image_id: str = ""
    daily_id: str = ""
    daily_no: str = ""
    image: Image.Image | None = None


@dataclass
class TimeguessrGame:
    """Complete 5-round TimeGuessr game with optional all-player distribution."""

    url: str
    mode: str
    rounds: list[TimeguessrRound]
    daily_no: str | None = None
    daily_id: str | None = None
    game_id: str | None = None
    distribution: PlayerDistribution | None = None

    def __len__(self) -> int:
        return len(self.rounds)

    def get_round(self, round_num: int) -> TimeguessrRound:
        if not (1 <= round_num <= len(self.rounds)):
            raise IndexError(f"Round must be between 1 and {len(self.rounds)}, got {round_num}.")
        return self.rounds[round_num - 1]


def _extract_balanced_bracket(text: str, start_idx: int, open_ch: str = "[", close_ch: str = "]") -> str:
    """Extract a balanced `[...]` or `{...}` substring while respecting JS string literals."""
    depth = 0
    in_str: str | None = None
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'", "`"):
            in_str = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    raise ValueError(f"Unbalanced {open_ch}...{close_ch} starting at index {start_idx}.")


def _js_object_to_json(js_text: str) -> str:
    """Convert a SvelteKit inline JS data literal (unquoted keys, void 0) into valid JSON."""
    out: list[str] = []
    i = 0
    n = len(js_text)
    in_str = False
    escaped = False

    while i < n:
        ch = js_text[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        # Replace JS `void 0` or `undefined` with `null`
        if js_text.startswith("void 0", i) and (i == 0 or not js_text[i - 1].isalnum()):
            out.append("null")
            i += 6
            continue
        if js_text.startswith("undefined", i) and (i == 0 or not js_text[i - 1].isalnum()):
            out.append("null")
            i += 9
            continue

        # Quote unquoted identifier keys immediately following `{` or `,`
        if ch in "{,":
            out.append(ch)
            i += 1
            while i < n and js_text[i].isspace():
                out.append(js_text[i])
                i += 1
            if i < n and (js_text[i].isalpha() or js_text[i] in "_$"):
                j = i
                while j < n and (js_text[j].isalnum() or js_text[j] in "_$"):
                    j += 1
                k = j
                while k < n and js_text[k].isspace():
                    k += 1
                if k < n and js_text[k] == ":":
                    key = js_text[i:j]
                    out.append(f'"{key}"')
                    i = j
                    continue
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def parse_play_array_from_html(html: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse `playArray` and surrounding metadata (`mode`, `gameId`, `archive`) from TimeGuessr HTML."""
    match = re.search(r"playArray\s*:\s*\[", html)
    if not match:
        raise ValueError(
            "Could not find `playArray` in the HTML. Make sure the link is a valid TimeGuessr game URL "
            "(e.g. https://timeguessr.com/play?mode=daily)."
        )

    bracket_start = html.index("[", match.start())
    raw_array = _extract_balanced_bracket(html, bracket_start, "[", "]")
    json_str = _js_object_to_json(raw_array)
    parsed = json.loads(json_str)

    # SvelteKit's playArray sometimes has a trailing sentinel integer (e.g. `, 0]`)
    round_dicts = [item for item in parsed if isinstance(item, dict) and ("URL" in item or "Location" in item)]
    if not round_dicts:
        raise ValueError("Found `playArray`, but it contained no valid round objects.")

    meta: dict[str, Any] = {}
    for key in ("mode", "gameId", "archive"):
        m = re.search(rf'\b{key}\s*:\s*(?:"([^"]*)"|(\d+)|null)', html[: bracket_start + 500])
        if m:
            meta[key] = m.group(1) if m.group(1) is not None else (m.group(2) if m.group(2) is not None else None)
    return round_dicts, meta


def _parse_distribution_payload(payload: dict[str, Any], endpoint: str) -> PlayerDistribution | None:
    if not isinstance(payload, dict) or payload.get("insufficient"):
        return None
    if "avgScore" not in payload or "histogram" not in payload:
        return None

    round_stats: list[RoundPlayerStats] = []
    for r in payload.get("rounds", []) or []:
        dist_m = r.get("avgDistanceMeters")
        round_stats.append(
            RoundPlayerStats(
                round_num=int(r.get("round", len(round_stats) + 1)),
                avg_score=float(r["avgScore"]) if r.get("avgScore") is not None else None,
                avg_years_off=float(r["avgYearsOff"]) if r.get("avgYearsOff") is not None else None,
                avg_distance_km=(float(dist_m) / 1000.0) if dist_m is not None else None,
            )
        )

    return PlayerDistribution(
        avg_score=float(payload["avgScore"]),
        bin_size=int(payload.get("binSize", 1000)),
        histogram=[float(x) for x in payload["histogram"]],
        avg_year_accuracy=float(payload["avgYearAccuracy"]) if payload.get("avgYearAccuracy") is not None else None,
        avg_location_accuracy=(
            float(payload["avgLocationAccuracy"]) if payload.get("avgLocationAccuracy") is not None else None
        ),
        avg_duration_sec=float(payload["avgDuration"]) if payload.get("avgDuration") is not None else None,
        rounds=round_stats,
        source_endpoint=endpoint,
    )


def fetch_player_distribution(
    mode: str,
    daily_id: str | None = None,
    game_id: str | None = None,
    image_ids: list[str] | None = None,
    session: requests.Session | None = None,
    timeout: int = 15,
) -> PlayerDistribution | None:
    """Fetch the all-player score distribution from TimeGuessr's public game-stats API."""
    sess = session or requests.Session()
    candidates: list[str] = []

    if mode == "daily" and daily_id:
        candidates.append(f"https://timeguessr.com/api/game-stats/daily/{quote(str(daily_id), safe='')}")
    elif mode == "custom" and game_id:
        candidates.append(f"https://timeguessr.com/api/game-stats/custom/{quote(str(game_id), safe='')}")

    if image_ids and len(image_ids) == 5 and all(image_ids):
        joined = ",".join(quote(str(i), safe="") for i in image_ids)
        candidates.append(f"https://timeguessr.com/api/game-stats/play?images={joined}")

    for endpoint in candidates:
        try:
            resp = sess.get(endpoint, headers=DEFAULT_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                dist = _parse_distribution_payload(resp.json(), endpoint)
                if dist is not None:
                    return dist
        except Exception:
            continue
    return None


def download_round_image(
    url: str,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> Image.Image:
    """Download a single TimeGuessr round photo as an RGB PIL Image."""
    sess = session or requests.Session()
    headers = {**DEFAULT_HEADERS, "Referer": "https://timeguessr.com/"}
    resp = sess.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


def fetch_timeguessr_game(
    url: str,
    download_images: bool = True,
    fetch_distribution: bool = True,
    timeout: int = 20,
) -> TimeguessrGame:
    """Fetch a TimeGuessr game URL, extract all 5 rounds, download photos, and pull player stats.

    Supports:
      - Daily challenge:  https://timeguessr.com/play?mode=daily
      - Daily archive:    https://timeguessr.com/play?mode=daily&archive=...
      - Custom / play:    https://timeguessr.com/play?mode=custom&game_id=...
    """
    sess = requests.Session()
    resp = sess.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()

    round_dicts, meta = parse_play_array_from_html(resp.text)
    parsed_url = urlparse(url)
    qs = parse_qs(parsed_url.query)
    mode = str(meta.get("mode") or (qs.get("mode", ["daily"])[0]))
    game_id = meta.get("gameId") or (qs.get("game_id", [None])[0])

    rounds: list[TimeguessrRound] = []
    for idx, item in enumerate(round_dicts, start=1):
        loc = item.get("Location") or {}
        lat = float(loc.get("lat", item.get("lat", 0.0)))
        lon = float(loc.get("lng", loc.get("lon", item.get("lng", 0.0))))
        img_url = str(item.get("URL", ""))
        img = download_round_image(img_url, session=sess, timeout=timeout) if (download_images and img_url) else None

        rounds.append(
            TimeguessrRound(
                round_num=idx,
                image_url=img_url,
                true_year=int(round(float(item.get("Year", 1970)))),
                true_lat=lat,
                true_lon=lon,
                country=str(item.get("Country", "")),
                description=str(item.get("Description", "")),
                license=str(item.get("License", "")),
                image_id=str(item.get("ImageId", "")),
                daily_id=str(item.get("DailyId", "")),
                daily_no=str(item.get("No", "")),
                image=img,
            )
        )

    daily_no = rounds[0].daily_no or None
    daily_id = rounds[0].daily_id or None
    image_ids = [r.image_id for r in rounds]

    distribution = None
    if fetch_distribution:
        distribution = fetch_player_distribution(
            mode=mode,
            daily_id=daily_id,
            game_id=game_id,
            image_ids=image_ids,
            session=sess,
            timeout=timeout,
        )

    return TimeguessrGame(
        url=url,
        mode=mode,
        rounds=rounds,
        daily_no=daily_no,
        daily_id=daily_id,
        game_id=game_id,
        distribution=distribution,
    )


def score_round(
    true_lat: float,
    true_lon: float,
    true_year: int,
    pred_lat: float,
    pred_lon: float,
    pred_year: float,
) -> dict[str, float]:
    """Compute exact TimeGuessr round score using formulas.txt (rounded to integer year guess)."""
    pred_year_int = int(round(float(pred_year)))
    dist_km = float(haversine_km(true_lat, true_lon, pred_lat, pred_lon))
    years_off = abs(pred_year_int - int(true_year))
    loc_score = float(timeguessr_location_score(dist_km))
    year_score = float(timeguessr_year_score(pred_year_int, int(true_year)))
    return {
        "pred_year_int": pred_year_int,
        "years_off": int(years_off),
        "dist_km": round(dist_km, 2),
        "loc_score": round(loc_score),
        "year_score": round(year_score),
        "total_score": round(loc_score) + round(year_score),
    }

