"""On-disk JSON cache for FMP/yfinance payloads (ticker + calendar day)."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, Tuple

from analysis_model.data.Data_gathering import DataGathering


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def cache_dir() -> str:
    path = os.path.join(_project_root(), "cache")
    os.makedirs(path, exist_ok=True)
    return path


def cache_path(ticker: str, day: str | None = None) -> str:
    day_key = day or str(date.today())
    safe = ticker.strip().upper().replace("/", "_")
    return os.path.join(cache_dir(), f"{safe}_{day_key}.json")


def load_cached_payload(ticker: str) -> Dict[str, Any] | None:
    path = cache_path(ticker)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def save_cached_payload(ticker: str, payload: Dict[str, Any]) -> None:
    path = cache_path(ticker)
    with open(path, "w") as handle:
        json.dump(payload, handle)


def load_or_fetch_gatherer(ticker: str) -> Tuple[DataGathering, bool]:
    """Return (gatherer, from_cache). Network fetch only on cache miss."""
    cached = load_cached_payload(ticker)
    if cached:
        return DataGathering.from_payload(cached), True
    gatherer = DataGathering(symbol=ticker)
    gatherer.fetch_all_data()
    save_cached_payload(ticker, gatherer.to_payload())
    return gatherer, False
