"""Direct client for Haver Analytics' HaverView REST API
(https://api.haverview.com), used by 01_build_panel_real.py as the primary
data source for the indicator panel and GDP target.

Adapted from the same pattern already proven in a sibling project
(deindustrialization-dashboard/src/haver_loader.py) - a standalone pipeline
script shouldn't depend on an MCP server being connected in a live Claude
session, it should just make HTTP requests.

Auth: reads HAVER_API_KEY from the environment (via a .env file in the repo
root, git-ignored - never hardcoded in source).
"""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://api.haverview.com"


def _load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
API_KEY = os.environ.get("HAVER_API_KEY")


def _headers() -> dict:
    if not API_KEY:
        raise RuntimeError("HAVER_API_KEY is not set - create a .env file in the repo root (see py/haver_client.py).")
    return {"X-API-Key": API_KEY}


def get_series(database: str, series: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.Series:
    """Fetch one series from Haver and return it as a pd.Series indexed by date.

    Raises on any HTTP error or if the response has no usable data points,
    consistent with fetch_fred_series()'s behavior in 01_build_panel_real.py
    (both are expected to be called inside a try/except by their callers).
    """
    resp = requests.get(
        f"{BASE_URL}/v4/database/{database}/series/{series}",
        headers=_headers(),
        params={"format": "data", "start_date": start_date, "end_date": end_date},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    points = data.get("dataPoints", [])
    if not points:
        raise RuntimeError(f"Haver returned no data points for {series}@{database}")

    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"])
    df["nSeriesData"] = pd.to_numeric(df["nSeriesData"], errors="coerce")
    s = df.set_index("date")["nSeriesData"].dropna()
    if s.empty:
        raise RuntimeError(f"Haver returned only null data points for {series}@{database}")
    return s.rename(series)
