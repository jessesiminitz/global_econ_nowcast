"""Eurostat REST API client (https://ec.europa.eu/eurostat/api/dissemination).
No API key needed. New to this repo — no prior Eurostat code existed here
to generalize from, so this follows Eurostat's documented SDMX-CSV output
(the same flat OBS_VALUE/TIME_PERIOD shape used by imf.py/oecd.py in this
skill) rather than its JSON-stat format, to keep parsing consistent.

NOTE: this sandbox's network policy blocks ec.europa.eu, so this client
could not be live-tested during development (see reference/eurostat.md) —
smoke-test it against a real dataset before relying on it.
"""
from io import StringIO
from typing import Optional

import pandas as pd

from _util import http_get, parse_period

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def get_dataset(dataset_code: str, filters: Optional[dict] = None, start_period: Optional[str] = None) -> pd.DataFrame:
    """Pulls a Eurostat dataset as a raw pd.DataFrame — one row per
    observation. Use this to explore a dataset's columns before narrowing
    down with to_series().

    dataset_code: Eurostat dataset code, e.g. "namq_10_gdp" (quarterly
    national accounts GDP), "prc_hicp_manr" (HICP inflation).
    filters: dimension filters as a dict, e.g. {"geo": "DE", "na_item":
    "B1GQ", "unit": "CLV_PCH_PRE", "s_adj": "SCA"} — narrow the query as
    much as possible, Eurostat datasets can be huge unfiltered.
    start_period: "YYYY" or "YYYY-Qn"/"YYYY-MM" lower bound, optional.
    """
    url = f"{BASE_URL}/{dataset_code}"
    params = {"format": "SDMX-CSV", "lang": "en", **(filters or {})}
    if start_period:
        params["sinceTimePeriod"] = start_period
    resp = http_get(url, params=params)
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise RuntimeError(f"Eurostat returned no rows for dataset={dataset_code} filters={filters}")
    return df


def to_series(df: pd.DataFrame, value_col: Optional[str] = None, time_col: Optional[str] = None, name: str = "value") -> pd.Series:
    """Narrows an already-filtered single-series slice of get_dataset()'s
    output down to a pd.Series indexed by date."""
    value_col = value_col or next((c for c in df.columns if "OBS_VALUE" in c.upper()), None)
    time_col = time_col or next((c for c in df.columns if "TIME_PERIOD" in c.upper()), None)
    if value_col is None or time_col is None:
        raise RuntimeError("Could not find OBS_VALUE/TIME_PERIOD columns — pass value_col/time_col explicitly.")

    idx = df[time_col].map(parse_period)
    values = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(values.values, index=idx.values)
    s = s[pd.notna(s.index)].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")].rename(name)


def get_series(dataset_code: str, filters: dict, start_period: Optional[str] = None, name: str = "value") -> pd.Series:
    """Convenience one-shot: get_dataset() + to_series() for a filter set
    that's already narrow enough to identify a single series."""
    df = get_dataset(dataset_code, filters=filters, start_period=start_period)
    return to_series(df, name=name)


if __name__ == "__main__":
    # Euro area real GDP, seasonally adjusted, QoQ % change
    print(get_series(
        "namq_10_gdp",
        {"geo": "EA20", "na_item": "B1GQ", "unit": "CLV_PCH_PRE", "s_adj": "SCA"},
        name="ea_gdp_growth",
    ).tail(8))
