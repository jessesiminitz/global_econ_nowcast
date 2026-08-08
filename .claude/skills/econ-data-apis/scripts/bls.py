"""BLS (Bureau of Labor Statistics) public API v2 client
(https://api.bls.gov/publicAPI/v2). Works without a key at a lower rate
limit/history depth; a free key raises both — register at
https://data.bls.gov/registrationEngine/, then
`export BLS_API_KEY=your_key_here`.

New to this repo. NOTE: this sandbox's network policy blocks api.bls.gov,
so this client could not be live-tested during development (see
reference/extras.md) — smoke-test it against a real series before relying
on it.
"""
from typing import Optional

import pandas as pd
import requests

from _util import get_key, UA

BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

_MONTH_TO_NUM = {f"M{i:02d}": i for i in range(1, 13)}


def get_series(series_id: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.Series:
    """Fetches one BLS series (e.g. "CUUR0000SA0" — CPI-U, all items,
    U.S. city average) and returns it as a monthly pd.Series indexed by
    date. BLS's v2 API only returns up to 20 years per request — for a
    longer span, call this multiple times with different start/end years
    and concat the results.
    """
    api_key = get_key("BLS_API_KEY")
    body = {"seriesid": [series_id]}
    if start_year:
        body["startyear"] = str(start_year)
    if end_year:
        body["endyear"] = str(end_year)
    if api_key:
        body["registrationkey"] = api_key

    resp = requests.post(BASE_URL, json=body, headers=UA, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API request failed: {payload.get('message', payload)}")

    series_list = payload.get("Results", {}).get("series", [])
    if not series_list or not series_list[0].get("data"):
        raise RuntimeError(f"BLS returned no data for {series_id}")

    rows = []
    for point in series_list[0]["data"]:
        month = _MONTH_TO_NUM.get(point["period"])
        if month is None:  # skip annual-average ("M13") and other non-monthly periods
            continue
        rows.append((pd.Timestamp(f"{point['year']}-{month:02d}-01"), float(point["value"])))

    if not rows:
        raise RuntimeError(f"BLS returned only non-monthly periods for {series_id}")
    s = pd.Series(dict(rows)).sort_index()
    return s.rename(series_id)


if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "CUUR0000SA0"
    print(get_series(sid).tail(8))
