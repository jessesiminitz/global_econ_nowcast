"""IMF SDMX 3.0 REST API client (https://api.imf.org).

Generalizes the pattern already proven in this repo's
py/01_build_panel_real.py (fetch_imf_quarterly_gdp(), pulling the
QGDP_WCA dataflow) to any dataflow. No API key needed.

IMF SDMX responses are wide and inconsistent about exact column names
across dataflows (REF_AREA vs COUNTRY vs REGION, etc.) and often carry
multiple TYPE_OF_TRANSFORMATION variants (level vs. % change) for the same
series — see reference/imf.md for how to navigate a new dataflow, and the
worked recipe below (based on the real code this generalizes).
"""
import io
from typing import Optional

import pandas as pd

from _util import http_get, parse_period

BASE_URL = "https://api.imf.org/external/sdmx/3.0/data/dataflow"


def get_sdmx_data(dataflow: str, key: str = "*", database_id: str = "IMF.STA", start_period: Optional[str] = None) -> pd.DataFrame:
    """Pulls a full (or key-filtered) SDMX dataflow as a raw pd.DataFrame —
    one row per observation, with whatever dimension/label columns the IMF
    returns (varies by dataflow). Use this to explore an unfamiliar
    dataflow before narrowing down to one series.

    dataflow: e.g. "QGDP_WCA" (Quarterly GDP, World and Country Aggregates).
    key: SDMX dimension key filter (e.g. "Q.W00.B1GQ_S1_Q..."), or "*" for
    everything.
    database_id: IMF database/agency ID, "IMF.STA" covers most macro data.
    start_period: "YYYY" or "YYYY-Qn" lower bound, optional.
    """
    url = f"{BASE_URL}/{database_id}/{dataflow}/~/{key}"
    params = {"labels": "both"}
    if start_period:
        params["c[TIME_PERIOD]"] = f"ge:{start_period}"
    headers = {"Accept": "application/vnd.sdmx.data+csv;labels=both, text/csv;q=0.9, */*;q=0.1"}
    resp = http_get(url, params=params, headers=headers)
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        raise RuntimeError(f"IMF returned no rows for dataflow={dataflow} key={key}")
    return df


def to_series(df: pd.DataFrame, value_col: Optional[str] = None, time_col: Optional[str] = None, name: str = "value") -> pd.Series:
    """Narrows an already-filtered single-series slice of get_sdmx_data()'s
    output down to a pd.Series indexed by date. Auto-detects the
    OBS_VALUE/TIME_PERIOD columns if not given. Handles "2024-Q1"-style
    quarterly periods as well as monthly/annual/full-date ones (see
    _util.parse_period).
    """
    value_col = value_col or next((c for c in df.columns if "OBS_VALUE" in c.upper()), None)
    time_col = time_col or next((c for c in df.columns if "TIME_PERIOD" in c.upper()), None)
    if value_col is None or time_col is None:
        raise RuntimeError("Could not find OBS_VALUE/TIME_PERIOD columns — pass value_col/time_col explicitly.")

    idx = df[time_col].map(parse_period)
    values = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(values.values, index=idx.values)
    s = s[pd.notna(s.index)].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")].rename(name)


def world_real_gdp_growth_saar() -> pd.Series:
    """Worked recipe: the IMF's own quarterly, seasonally-adjusted world
    real GDP growth series (QGDP_WCA), compounded to an annualized rate —
    exactly what py/01_build_panel_real.py's fetch_imf_quarterly_gdp() does,
    reimplemented here on top of the generic functions above as a runnable
    example. See reference/imf.md for how the World-area / real-GDP-growth
    row gets identified without hardcoding IMF's dimension codes.
    """
    df = get_sdmx_data("QGDP_WCA", database_id="IMF.STA")

    freq_cols = [c for c in df.columns if c.upper() == "FREQ"]
    if freq_cols:
        df = df[df[freq_cols[0]].astype(str).str.upper() == "Q"]

    area_cols = [c for c in df.columns if any(k in c.upper() for k in ("AREA", "COUNTRY", "REGION"))]
    indicator_cols = [c for c in df.columns if "INDICATOR" in c.upper()]
    if not area_cols or not indicator_cols:
        raise RuntimeError("Unexpected QGDP_WCA layout: missing REF_AREA/INDICATOR columns.")

    is_world = pd.Series(False, index=df.index)
    for c in area_cols:
        vals = df[c].astype(str).str.strip()
        is_world |= vals.str.upper().eq("W00") | vals.str.contains("world", case=False, na=False)
    world_df = df[is_world]
    if world_df.empty:
        raise RuntimeError("Could not find a 'World' REF_AREA row in QGDP_WCA.")

    indicator_text = world_df[indicator_cols[0]].astype(str)
    indicator_code = indicator_text.str.split(":", n=1).str[0].str.strip()
    transformation_cols = [c for c in df.columns if "TRANSFORMATION" in c.upper()]

    def restrict_to_pop_pch(frame: pd.DataFrame) -> pd.DataFrame:
        if not transformation_cols:
            return frame
        mask = pd.Series(False, index=frame.index)
        for c in transformation_cols:
            mask |= frame[c].astype(str).str.contains("POP_PCH", case=False, na=False)
        return frame[mask] if mask.any() else frame

    if "B1GQ_S1_Q" in indicator_code.values:
        world_df = restrict_to_pop_pch(world_df[indicator_code == "B1GQ_S1_Q"])
    else:
        raise RuntimeError("Expected indicator code B1GQ_S1_Q not found — inspect get_sdmx_data('QGDP_WCA') directly.")

    s = to_series(world_df, name="gdp_growth_qoq")
    if (s.abs() > 50).all():
        s = s.pct_change().dropna() * 100
    saar = ((1 + s / 100) ** 4 - 1) * 100
    return saar.rename("gdp_growth_saar")


if __name__ == "__main__":
    print(world_real_gdp_growth_saar().tail(8))
