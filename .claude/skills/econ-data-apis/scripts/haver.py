"""Haver Analytics (HaverView REST API) client.

A near-verbatim port of this repo's py/haver_client.py, kept as a separate
copy so this skill stays self-contained and portable to other projects
(no dependency on this repo's py/ package). Reads HAVER_API_KEY from the
environment (already set in the user's shell) or a .env colocated with
this scripts/ directory.
"""
from typing import Optional

import pandas as pd

from _util import get_key, http_get

BASE_URL = "https://api.haverview.com"


def get_series(database: str, series: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.Series:
    """Fetches one series from Haver (e.g. database="USECON", series="GDPH")
    and returns it as a pd.Series indexed by date.

    Raises on any HTTP error, a missing API key, or an empty response.
    """
    api_key = get_key("HAVER_API_KEY")
    if not api_key:
        raise RuntimeError("HAVER_API_KEY is not set — export it in your shell, or add it to a .env file next to this script.")

    resp = http_get(
        f"{BASE_URL}/v4/database/{database}/series/{series}",
        headers={"X-API-Key": api_key},
        params={"format": "data", "start_date": start_date, "end_date": end_date},
    )
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


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "USECON"
    sid = sys.argv[2] if len(sys.argv) > 2 else "GDPH"
    print(get_series(db, sid).tail(8))
