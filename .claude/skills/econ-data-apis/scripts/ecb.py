"""ECB Statistical Data Warehouse (SDW) REST API client
(https://data-api.ecb.europa.eu). No API key needed.

New to this repo. NOTE: this sandbox's network policy blocks
*.europa.eu, so this client could not be live-tested during development
(see reference/ecb.md) — smoke-test it against a real series before
relying on it.
"""
from io import StringIO
from typing import Optional

import pandas as pd

from _util import http_get, parse_period

BASE_URL = "https://data-api.ecb.europa.eu/service/data"


def get_series(flow_ref: str, key: str, start_period: Optional[str] = None, end_period: Optional[str] = None, name: Optional[str] = None) -> pd.Series:
    """Fetches one ECB SDW series and returns it as a pd.Series indexed by
    date.

    flow_ref: dataflow ID, e.g. "EXR" (exchange rates), "FM" (financial
    market data), "ICP" (inflation/HICP), "MIR" (bank interest rates).
    key: dot-separated SDMX series key within that flow, e.g.
    "D.USD.EUR.SP00.A" (daily USD/EUR spot reference rate under EXR).
    start_period/end_period: "YYYY-MM-DD" or "YYYY" bounds, optional.
    """
    url = f"{BASE_URL}/{flow_ref}/{key}"
    params = {"format": "csvdata"}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    resp = http_get(url, params=params)
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise RuntimeError(f"ECB SDW returned no rows for {key}@{flow_ref}")

    value_col = next((c for c in df.columns if "OBS_VALUE" in c.upper()), None)
    time_col = next((c for c in df.columns if "TIME_PERIOD" in c.upper()), None)
    if value_col is None or time_col is None:
        raise RuntimeError(f"Unexpected ECB SDW CSV layout for {key}@{flow_ref}: {list(df.columns)}")

    idx = df[time_col].map(parse_period)
    values = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(values.values, index=idx.values)
    s = s[pd.notna(s.index)].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")].rename(name or key)


if __name__ == "__main__":
    # Daily USD/EUR spot reference rate
    print(get_series("EXR", "D.USD.EUR.SP00.A", name="usd_eur").tail(8))
