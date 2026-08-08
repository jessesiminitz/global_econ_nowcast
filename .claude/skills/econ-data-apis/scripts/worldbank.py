"""World Bank API client (https://api.worldbank.org). No API key needed.

Generalizes py/01_build_panel_real.py's fetch_world_bank_annual_gdp_growth()
(hardcoded to country="WLD", indicator="NY.GDP.MKTP.KD.ZG") to any
(country, indicator) pair. World Bank indicator data is annual.
"""
from typing import Optional

import pandas as pd

from _util import http_get

BASE_URL = "https://api.worldbank.org/v2"


def get_indicator(country: str, indicator: str, start: Optional[int] = None, end: Optional[int] = None) -> pd.Series:
    """Fetches one World Bank indicator for one country/aggregate and
    returns it as an annual pd.Series indexed by Jan-1 of each year.

    country: ISO3 code or aggregate code, e.g. "USA", "WLD" (world), "EUU"
    (EU), "CHN".
    indicator: World Bank indicator code, e.g. "NY.GDP.MKTP.KD.ZG" (real
    GDP growth %), "FP.CPI.TOTL.ZG" (inflation), "SL.UEM.TOTL.ZS"
    (unemployment).
    start/end: 4-digit years, optional.
    """
    url = f"{BASE_URL}/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": 1000}
    if start and end:
        params["date"] = f"{start}:{end}"
    resp = http_get(url, params=params)
    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise RuntimeError(f"World Bank returned no data for {indicator} / {country} (response: {payload})")

    rows = [(int(d["date"]), d["value"]) for d in payload[1] if d["value"] is not None]
    if not rows:
        raise RuntimeError(f"World Bank returned only null values for {indicator} / {country}")
    s = pd.Series({pd.Timestamp(f"{y}-01-01"): v for y, v in rows}).sort_index()
    return s.rename(indicator)


if __name__ == "__main__":
    import sys
    country = sys.argv[1] if len(sys.argv) > 1 else "WLD"
    indicator = sys.argv[2] if len(sys.argv) > 2 else "NY.GDP.MKTP.KD.ZG"
    print(get_indicator(country, indicator).tail(8))
