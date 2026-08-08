"""BEA (Bureau of Economic Analysis) API client (https://apps.bea.gov/api).
Needs a free key — register at https://apps.bea.gov/API/signup/, then
`export BEA_API_KEY=your_key_here`.

New to this repo. NOTE: this sandbox's network policy blocks apps.bea.gov,
so this client could not be live-tested during development (see
reference/extras.md) — smoke-test it against a real table before relying
on it.
"""
from typing import Optional

import pandas as pd

from _util import get_key, http_get

BASE_URL = "https://apps.bea.gov/api/data"


def get_data(dataset_name: str, **params) -> pd.DataFrame:
    """Pulls a raw BEA dataset query as a pd.DataFrame — one row per
    observation, exactly as BEA's API returns it (columns vary by
    dataset). Use this to explore before narrowing to one series.

    dataset_name: e.g. "NIPA" (national accounts), "Regional", "GDPbyIndustry".
    **params: dataset-specific query params, e.g. for NIPA:
    TableName="T10101", Frequency="Q", Year="ALL".
    """
    api_key = get_key("BEA_API_KEY")
    if not api_key:
        raise RuntimeError("BEA_API_KEY is not set — sign up free at https://apps.bea.gov/API/signup/, export it, or add it to a .env file next to this script.")

    query = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": dataset_name,
        "ResultFormat": "JSON",
        **params,
    }
    resp = http_get(BASE_URL, params=query)
    payload = resp.json()
    results = payload.get("BEAAPI", {}).get("Results", {})
    if "Error" in results:
        raise RuntimeError(f"BEA API error: {results['Error']}")
    data = results.get("Data", [])
    if not data:
        raise RuntimeError(f"BEA returned no data for {dataset_name} / {params} (response: {payload})")
    return pd.DataFrame(data)


def to_series(df: pd.DataFrame, line_description: Optional[str] = None, name: str = "value") -> pd.Series:
    """Narrows a NIPA-style get_data() DataFrame (TimePeriod/DataValue/
    LineDescription columns) down to a pd.Series indexed by date, filtered
    to one LineDescription if given (required if the raw pull has more
    than one line)."""
    if line_description is not None and "LineDescription" in df.columns:
        df = df[df["LineDescription"].astype(str).str.strip() == line_description]
        if df.empty:
            raise RuntimeError(f"No rows matched LineDescription={line_description!r}")

    if "TimePeriod" not in df.columns or "DataValue" not in df.columns:
        raise RuntimeError(f"Expected TimePeriod/DataValue columns, got: {list(df.columns)}")

    period = df["TimePeriod"].astype(str)
    dates = pd.to_datetime(period.str.replace(r"Q(\d)", lambda m: f"-{(int(m.group(1)) - 1) * 3 + 1:02d}", regex=True), errors="coerce")
    dates = dates.where(dates.notna(), pd.to_datetime(period, format="%Y", errors="coerce"))
    values = pd.to_numeric(df["DataValue"].astype(str).str.replace(",", ""), errors="coerce")

    s = pd.Series(values.values, index=dates.values)
    s = s[pd.notna(s.index)].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")].rename(name)


if __name__ == "__main__":
    # NIPA Table 1.1.1: real GDP, quarterly
    df = get_data("NIPA", TableName="T10101", Frequency="Q", Year="ALL")
    print(to_series(df, line_description="Gross domestic product", name="gdp").tail(8))
