"""OECD SDMX REST API client (https://sdmx.oecd.org). No API key needed.

Generalizes the pattern already proven in this repo's
py/01_build_panel_real.py (fetch_pmi_proxy(), pulling the Composite Leading
Indicator dataflow) to any OECD dataflow/key. OECD dataflow IDs look like
"OECD.SDD.STES,DSD_STES@DF_CLI,4.1" (agency.division,DSD@dataflow,version)
and SDMX keys are dot-separated dimension filters, e.g. ".M.LI...AA.IX..H"
— see reference/oecd.md for how to find both for a given dataset via the
OECD Data Explorer.
"""
from io import StringIO
from typing import Optional

import pandas as pd

from _util import http_get, parse_period

BASE_URL = "https://sdmx.oecd.org/public/rest/data"


def get_sdmx_data(dataflow: str, key: str = "all", start_period: Optional[str] = None) -> pd.DataFrame:
    """Pulls an OECD SDMX dataflow as a raw pd.DataFrame — one row per
    observation. Use this to explore a dataflow's columns before narrowing
    down with to_series().

    dataflow: e.g. "OECD.SDD.STES,DSD_STES@DF_CLI,4.1".
    key: SDMX dimension key filter, or "all"/"" for every series in the
    dataflow (can be large).
    start_period: "YYYY-MM" lower bound, optional.
    """
    url = f"{BASE_URL}/{dataflow}/{key}"
    params = {"format": "csv"}
    if start_period:
        params["startPeriod"] = start_period
    headers = {"Accept": "text/csv, application/vnd.sdmx.data+csv; charset=utf-8"}
    resp = http_get(url, params=params, headers=headers)
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise RuntimeError(f"OECD returned no rows for dataflow={dataflow} key={key}")
    return df


def to_series(df: pd.DataFrame, value_col: Optional[str] = None, time_col: Optional[str] = None, name: str = "value") -> pd.Series:
    """Narrows an already-filtered single-series slice of get_sdmx_data()'s
    output down to a pd.Series indexed by date."""
    value_col = value_col or next((c for c in df.columns if "OBS_VALUE" in c.upper() or c.upper() == "VALUE"), None)
    time_col = time_col or next((c for c in df.columns if "TIME_PERIOD" in c.upper() or c.upper() == "PERIOD"), None)
    if value_col is None or time_col is None:
        raise RuntimeError("Could not find OBS_VALUE/TIME_PERIOD columns — pass value_col/time_col explicitly.")

    idx = df[time_col].map(parse_period)
    values = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(values.values, index=idx.values)
    s = s[pd.notna(s.index)].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")].rename(name)


def composite_leading_indicator(area: str = "G20", start_period: Optional[str] = None) -> pd.Series:
    """Worked recipe: OECD Composite Leading Indicator for a given area
    (default G20) — exactly what py/01_build_panel_real.py's
    fetch_pmi_proxy() pulls before rescaling it to a PMI-like range."""
    df = get_sdmx_data(
        "OECD.SDD.STES,DSD_STES@DF_CLI,4.1",
        ".M.LI...AA.IX..H",
        start_period=start_period,
    )
    if "REF_AREA" in df.columns and area in df["REF_AREA"].values:
        df = df[df["REF_AREA"] == area]
    return to_series(df, name="cli")


if __name__ == "__main__":
    print(composite_leading_indicator().tail(8))
