"""FRED (Federal Reserve Economic Data) client.

Generalizes the fetch_fred_series() pattern already proven in this repo's
py/01_build_panel_real.py: the official, authenticated REST API when
FRED_API_KEY is set, else the unauthenticated fredgraph.csv scrape (works
for any public FRED series, just more prone to rate limiting/timeouts).

Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html, then
`export FRED_API_KEY=your_key_here` (already set in the user's shell per
this skill's README).
"""
import io
from typing import Optional

import pandas as pd

from _util import get_key, http_get

API_URL = "https://api.stlouisfed.org/fred/series/observations"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def get_series(series_id: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
    """Fetches one FRED series by its series ID (e.g. "GDP", "UNRATE",
    "T10Y2Y") and returns it as a pd.Series indexed by date.

    start/end: "YYYY-MM-DD" strings, both optional (FRED defaults to full
    history if omitted).
    """
    api_key = get_key("FRED_API_KEY")
    if api_key:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        resp = http_get(API_URL, params=params)
        obs = resp.json().get("observations", [])
        if not obs:
            raise RuntimeError(f"FRED API returned no observations for {series_id}")
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")["value"].dropna().rename(series_id)

    # Unauthenticated fallback — scrapes the same CSV the interactive FRED
    # chart downloads. More prone to rate limiting/timeouts than the
    # authenticated API above.
    params = {"id": series_id}
    if start:
        params["cosd"] = start
    if end:
        params["coed"] = end
    resp = http_get(CSV_URL, params=params)
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna().rename(series_id)


if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "GDP"
    print(get_series(sid).tail(8))
