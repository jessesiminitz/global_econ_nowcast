"""
Build panel_monthly.csv and gdp_quarterly.csv from REAL, LIVE data sources.
"""
import io
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

START_DATE = "2005-01-01"
FETCH_START_DATE = "2004-01-01"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


# =========================================================================
# 1. FRED (with Yahoo Finance fallback if FRED blocks/times out)
# =========================================================================
def fetch_fred_series(series_id: str) -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": series_id, "cosd": FETCH_START_DATE}
    r = requests.get(url, params=params, headers=UA, timeout=10)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()


def fetch_oil_shock() -> pd.Series:
    """Brent crude, monthly average, converted to a year-over-year % change."""
    try:
        brent = fetch_fred_series("DCOILBRENTEU")
    except Exception as e:
        print(f"  [fallback] FRED oil download failed ({e}), pulling Brent oil (BZ=F) via yfinance...")
        import yfinance as yf
        df_yf = yf.download("BZ=F", start=FETCH_START_DATE, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            brent = df_yf["Close"]["BZ=F"]
        else:
            brent = df_yf["Close"]
        if isinstance(brent, pd.DataFrame):
            brent = brent.iloc[:, 0]

    monthly = brent.resample("MS").mean()
    yoy = monthly.pct_change(12) * 100
    return yoy.rename("oil")


def fetch_financial_conditions() -> pd.Series:
    """
    NFCI is weekly, positive = TIGHT, negative = LOOSE.
    If FRED times out, fallback to ^VIX from Yahoo Finance.
    """
    try:
        nfci = fetch_fred_series("NFCI")
    except Exception as e:
        print(f"  [fallback] FRED NFCI download failed ({e}), pulling ^VIX via yfinance...")
        import yfinance as yf
        df_yf = yf.download("^VIX", start=FETCH_START_DATE, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            nfci = df_yf["Close"]["^VIX"]
        else:
            nfci = df_yf["Close"]
        if isinstance(nfci, pd.DataFrame):
            nfci = nfci.iloc[:, 0]

    monthly = nfci.resample("MS").mean()
    score = -monthly / monthly.std() * 1.0  # sign-flipped z-score
    return score.rename("fin")


def fetch_copper() -> pd.Series:
    """
    "Dr. Copper" — IMF Global Price of Copper (FRED PCOPPUSDM, USD/metric ton,
    monthly), converted to year-over-year % change. Widely used as a leading
    proxy for global industrial demand (China construction/manufacturing in
    particular). Falls back to CME copper futures (HG=F) via yfinance if
    FRED is unreachable.
    """
    try:
        level = fetch_fred_series("PCOPPUSDM")
        monthly = level.resample("MS").mean()
    except Exception as e:
        print(f"  [fallback] FRED copper download failed ({e}), pulling HG=F via yfinance...")
        import yfinance as yf
        df_yf = yf.download("HG=F", start=FETCH_START_DATE, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            px = df_yf["Close"]["HG=F"]
        else:
            px = df_yf["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        monthly = px.resample("MS").mean()

    yoy = monthly.pct_change(12) * 100
    return yoy.rename("copper")


def fetch_yield_curve() -> pd.Series:
    """
    10y-2y Treasury term spread (FRED T10Y2Y), daily, resampled to monthly
    mean. A classic leading recession indicator (inversions have preceded
    every US recession since the 1970s with a ~12-18 month lead). No clean
    non-FRED fallback exists for the 2y leg, so this indicator is skipped
    (with a warning) rather than substituted if FRED is unreachable.
    """
    try:
        spread = fetch_fred_series("T10Y2Y")
    except Exception as e:
        print(f"  [skip] FRED yield curve download failed ({e}), dropping 'yield_curve' from the panel.")
        return None

    monthly = spread.resample("MS").mean()
    return monthly.rename("yield_curve")


def fetch_dollar_index() -> pd.Series:
    """
    Fed broad trade-weighted USD index (FRED DTWEXBGS), year-over-year %
    change, sign-flipped so + = dollar EASING. A strengthening dollar
    tightens financial conditions for EM/dollar-debt borrowers and is
    historically a headwind for global growth, hence the flip (consistent
    with the sign convention used for `fin`). Falls back to ICE USD index
    futures (DX-Y.NYB) via yfinance if FRED is unreachable.
    """
    try:
        level = fetch_fred_series("DTWEXBGS")
        monthly = level.resample("MS").mean()
    except Exception as e:
        print(f"  [fallback] FRED dollar index download failed ({e}), pulling DX-Y.NYB via yfinance...")
        import yfinance as yf
        df_yf = yf.download("DX-Y.NYB", start=FETCH_START_DATE, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            px = df_yf["Close"]["DX-Y.NYB"]
        else:
            px = df_yf["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        monthly = px.resample("MS").mean()

    yoy = monthly.pct_change(12) * 100
    return (-yoy).rename("usd")


def fetch_credit_spread() -> pd.Series:
    """
    ICE BofA US High Yield Index option-adjusted spread (FRED
    BAMLH0A0HYM2), daily, resampled to monthly mean and sign-flipped so
    + = spreads TIGHTENING (easy credit) — same convention as `fin`. Captures
    corporate credit-market stress more directly than NFCI. No clean
    non-FRED fallback exists, so this indicator is skipped (with a warning)
    if FRED is unreachable.
    """
    try:
        spread = fetch_fred_series("BAMLH0A0HYM2")
    except Exception as e:
        print(f"  [skip] FRED credit spread download failed ({e}), dropping 'credit' from the panel.")
        return None

    monthly = spread.resample("MS").mean()
    return (-monthly).rename("credit")


# =========================================================================
# 2. CPB World Trade Monitor — world trade volume + industrial production
# =========================================================================
def fetch_cpb_world_trade_monitor():
    landing = "https://www.cpb.nl/en/worldtrademonitor/latest"
    r = requests.get(landing, headers=UA, timeout=30)
    r.raise_for_status()
    html = r.text

    matches = re.findall(r'href="([^"]+\.xlsx)"', html, flags=re.IGNORECASE)
    if not matches:
        raise RuntimeError("Could not find an .xlsx link on the CPB landing page.")
    xlsx_url = matches[0]
    if xlsx_url.startswith("/"):
        xlsx_url = "https://www.cpb.nl" + xlsx_url
    print(f"  CPB: downloading {xlsx_url}")

    r2 = requests.get(xlsx_url, headers=UA, timeout=60)
    r2.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r2.content))
    return xls, xlsx_url


def parse_cpb_sheet(xls: pd.ExcelFile, sheet_keywords, series_name):
    target_sheet = None
    for sheet in xls.sheet_names:
        if any(kw in sheet.lower() for kw in sheet_keywords):
            target_sheet = sheet
            break
    if target_sheet is None:
        return None

    df = xls.parse(target_sheet, header=None)
    date_row_idx = None
    for i in range(min(12, len(df))):
        row_str = [str(x).lower() for x in df.iloc[i].values]
        if any("m01" in val or "m1" in val for val in row_str):
            date_row_idx = i
            break

    if date_row_idx is None:
        return None

    world_row_idx = None
    for i in range(date_row_idx + 1, len(df)):
        cell = str(df.iloc[i, 1]).strip().lower()
        if "world" in cell:
            world_row_idx = i
            break

    if world_row_idx is None:
        return None

    date_raw = df.iloc[date_row_idx].values
    val_raw = df.iloc[world_row_idx].values

    dates, vals = [], []
    for d_val, v_val in zip(date_raw, val_raw):
        d_str = str(d_val).strip()
        if "m" in d_str.lower() and len(d_str) in (6, 7):
            try:
                parts = d_str.lower().split("m")
                dt = pd.Timestamp(f"{parts[0]}-{int(parts[1]):02d}-01")
                v = pd.to_numeric(v_val, errors="coerce")
                if not pd.isna(v):
                    dates.append(dt)
                    vals.append(v)
            except Exception:
                pass
    s = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    return s


def parse_cpb_trade_and_ip(xls: pd.ExcelFile):
    trade_level = parse_cpb_sheet(xls, ["trade"], "trade")
    ip_level = parse_cpb_sheet(xls, ["inpro", "production", "ip"], "ip")

    if trade_level is None or ip_level is None:
        raise RuntimeError("Could not parse CPB trade or production sheets.")

    def to_3mo_annualized(level: pd.Series) -> pd.Series:
        level = level.sort_index()
        g = (level / level.shift(3)) ** 4 - 1
        return g * 100

    return to_3mo_annualized(trade_level).rename("trade"), to_3mo_annualized(ip_level).rename("ip")


# =========================================================================
# 3. OECD Composite Leading Indicator — free PMI substitute
# =========================================================================
def fetch_pmi_proxy() -> pd.Series:
    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/.M.LI...AA.IX..H"
        f"?format=csv&startPeriod={FETCH_START_DATE[:7]}"
    )
    headers = {**UA, "Accept": "text/csv, application/vnd.sdmx.data+csv; charset=utf-8"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    # Filter for G20 or main aggregate
    if "REF_AREA" in df.columns:
        if "G20" in df["REF_AREA"].values:
            df = df[df["REF_AREA"] == "G20"]
        elif "G7" in df["REF_AREA"].values:
            df = df[df["REF_AREA"] == "G7"]

    time_col = next(c for c in df.columns if "TIME_PERIOD" in c.upper() or c.upper() == "PERIOD")
    val_col = next(c for c in df.columns if "OBS_VALUE" in c.upper() or c.upper() == "VALUE")

    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[time_col], errors="coerce"),
    )
    s = s[s.index.notna()].dropna().sort_index()

    rescaled = 50 + (s - 100) * 2.5
    return rescaled.resample("MS").mean().rename("pmi_proxy_oecd_cli")


# =========================================================================
# 4. IMF Quarterly GDP database — world real GDP growth (quarterly)
# =========================================================================
def _parse_sdmx_quarter(raw) -> "pd.Timestamp | None":
    """Parse a TIME_PERIOD value like '2024-Q1' or '2024Q1' into a quarter-start Timestamp."""
    s = str(raw).strip().upper().replace(" ", "")
    m = re.match(r"^(\d{4})-?Q([1-4])$", s)
    if not m:
        return None
    year, q = m.groups()
    month = (int(q) - 1) * 3 + 1
    return pd.Timestamp(f"{year}-{month:02d}-01")


def fetch_imf_quarterly_gdp() -> pd.Series:
    """
    IMF Quarterly GDP database — "World and Country Aggregates" dataflow
    (QGDP_WCA), served via the IMF's SDMX 3.0 REST API at api.imf.org. This
    is the IMF's own quarterly, seasonally-adjusted world real GDP growth
    series (the "World GDP grew X% quarter-on-quarter" figure quoted in the
    IMF's Quarterly GDP data briefs) — genuine quarterly data, rather than
    an interpolation of an annual figure.

    The exact REF_AREA/INDICATOR codes for the World aggregate and the
    real/SA/QoQ growth indicator aren't hardcoded: the full dataflow is
    pulled as SDMX-CSV with both codes and labels, then the area and
    indicator columns are searched for "World" and for real/SA/QoQ growth
    keywords. This is more robust to the IMF reordering or renaming its
    codelists than a fixed dimension key would be.
    """
    url = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QGDP_WCA/~/*"
    params = {"labels": "both", "c[TIME_PERIOD]": f"ge:{FETCH_START_DATE[:4]}"}
    headers = {
        **UA,
        "Accept": "application/vnd.sdmx.data+csv;labels=both, text/csv;q=0.9, */*;q=0.1",
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    freq_cols = [c for c in df.columns if c.upper() == "FREQ"]
    if freq_cols:
        df = df[df[freq_cols[0]].astype(str).str.upper() == "Q"]

    area_cols = [c for c in df.columns if any(k in c.upper() for k in ("AREA", "COUNTRY", "REGION"))]
    indicator_cols = [c for c in df.columns if "INDICATOR" in c.upper()]
    time_col = next((c for c in df.columns if "TIME_PERIOD" in c.upper()), None)
    val_col = next((c for c in df.columns if "OBS_VALUE" in c.upper()), None)

    if not area_cols or not indicator_cols or time_col is None or val_col is None:
        raise RuntimeError("Unexpected QGDP_WCA CSV layout: missing REF_AREA/INDICATOR/TIME_PERIOD/OBS_VALUE columns.")

    is_world = pd.Series(False, index=df.index)
    for c in area_cols:
        vals = df[c].astype(str).str.strip()
        is_world |= vals.str.upper().eq("W00") | vals.str.lower().eq("world") | vals.str.contains("world", case=False, na=False)
    world_df = df[is_world]
    if world_df.empty:
        raise RuntimeError("Could not find a 'World' REF_AREA row in QGDP_WCA.")

    def indicator_score(text: str) -> int:
        t = text.lower()
        score = 0
        if "real" in t:
            score += 1
        if "gdp" in t:
            score += 1
        if "growth" in t or "change" in t:
            score += 1
        if "seasonally adjusted" in t or re.search(r"\bsa\b", t):
            score += 2
        if "quarter" in t or "qoq" in t or "q-o-q" in t:
            score += 2
        if "annual" in t or "yoy" in t or "y-o-y" in t:
            score -= 1
        return score

    indicator_text = world_df[indicator_cols[0]].astype(str)
    indicator_code = indicator_text.str.split(":", 1).str[0].str.strip()
    if "B1GQ_S1_Q" in indicator_code.values:
        world_df = world_df[indicator_code == "B1GQ_S1_Q"]
    else:
        combined_text = indicator_text.agg(" ".join)
        scores = combined_text.map(indicator_score)
        best = scores.max()
        if best <= 0:
            raise RuntimeError("Could not identify a real/SA/QoQ GDP growth indicator row in QGDP_WCA.")
        world_df = world_df[scores == best]

    quarters = world_df[time_col].map(_parse_sdmx_quarter)
    values = pd.to_numeric(world_df[val_col], errors="coerce")
    s = pd.Series(values.values, index=quarters.values)
    s = s[s.index.notna()].dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]

    if s.empty:
        raise RuntimeError("QGDP_WCA query returned no usable World GDP growth observations.")

    # If IMF returns a level/index series instead of YOY or QoQ growth,
    # convert it to quarter-on-quarter growth before annualizing.
    if (s.abs() > 50).all():
        s = s.pct_change() * 100
        s = s.dropna()

    # QGDP_WCA publishes non-annualized quarter-on-quarter growth; compound
    # to an annualized rate to match the gdp_growth_saar convention used
    # elsewhere in this pipeline.
    saar = ((1 + s / 100) ** 4 - 1) * 100
    return saar.rename("gdp_growth_saar")


# =========================================================================
# 4b. World Bank — world real GDP growth (annual) — fallback for the above
# =========================================================================
def fetch_world_bank_annual_gdp_growth() -> pd.Series:
    url = "https://api.worldbank.org/v2/country/WLD/indicator/NY.GDP.MKTP.KD.ZG"
    params = {"format": "json", "per_page": 100}
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    data = r.json()[1]
    rows = [(int(d["date"]), d["value"]) for d in data if d["value"] is not None]
    s = pd.Series({pd.Timestamp(f"{y}-01-01"): v for y, v in rows}).sort_index()
    return s.rename("world_gdp_growth_annual")


def get_quarterly_gdp_target(annual_gdp: pd.Series) -> pd.Series:
    monthly = annual_gdp.resample("MS").ffill()
    idx = pd.date_range(monthly.index.min(), monthly.index.max() + pd.DateOffset(months=11), freq="MS")
    monthly = monthly.reindex(idx).ffill()
    smoothed = monthly.rolling(6, center=True, min_periods=1).mean()
    return smoothed.resample("QS").mean().rename("gdp_growth_saar")


# =========================================================================
# 5. AI / tech capex proxy — Yahoo Finance via yfinance
# =========================================================================
def fetch_ai_tech_proxy() -> pd.Series:
    import yfinance as yf

    df_yf = yf.download("^SOX", start=FETCH_START_DATE, progress=False)
    if isinstance(df_yf.columns, pd.MultiIndex):
        px = df_yf["Close"]["^SOX"]
    else:
        px = df_yf["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]

    monthly = px.resample("MS").last()
    yoy = monthly.pct_change(12) * 100
    z = (yoy - yoy.mean()) / yoy.std()
    score = np.tanh(z / 2) * 2
    return score.rename("ai")


# =========================================================================
# main
# =========================================================================
def main():
    print("Fetching Brent crude...")
    oil = fetch_oil_shock()

    print("Fetching financial conditions...")
    fin = fetch_financial_conditions()

    print("Fetching CPB World Trade Monitor (world trade + industrial production)...")
    xls, xlsx_url = fetch_cpb_world_trade_monitor()
    trade, ip = parse_cpb_trade_and_ip(xls)

    print("Fetching OECD Composite Leading Indicator (PMI substitute)...")
    pmi_proxy = fetch_pmi_proxy()

    print("Fetching AI/tech capex proxy (Yahoo Finance: ^SOX)...")
    ai = fetch_ai_tech_proxy()

    print("Fetching copper price (Dr. Copper)...")
    copper = fetch_copper()

    print("Fetching 10y-2y yield curve slope...")
    yield_curve = fetch_yield_curve()

    print("Fetching broad USD index...")
    usd = fetch_dollar_index()

    print("Fetching high-yield credit spread...")
    credit = fetch_credit_spread()

    print("Fetching IMF quarterly world real GDP growth (QGDP_WCA)...")
    try:
        gdp_quarterly = fetch_imf_quarterly_gdp()
    except Exception as e:
        print(f"  [fallback] IMF quarterly GDP fetch failed ({e}), falling back to World Bank annual GDP (interpolated to quarterly)...")
        gdp_annual = fetch_world_bank_annual_gdp_growth()
        gdp_quarterly = get_quarterly_gdp_target(gdp_annual)

    series = [pmi_proxy, trade, ip, oil, fin, ai, copper, yield_curve, usd, credit]
    panel = pd.concat([s for s in series if s is not None], axis=1)
    panel = panel.loc[START_DATE:].resample("MS").mean()
    panel = panel.rename(columns={"pmi_proxy_oecd_cli": "pmi"})
    panel = panel.interpolate(limit=2)

    panel.to_csv("panel_monthly.csv")
    gdp_quarterly.to_csv("gdp_quarterly.csv")

    print("\nSaved panel_monthly.csv:")
    print(panel.tail(8))
    print("\nSaved gdp_quarterly.csv:")
    print(gdp_quarterly.tail(8))


if __name__ == "__main__":
    main()
