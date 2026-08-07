"""
Build panel_monthly.csv and gdp_quarterly.csv from REAL, LIVE data sources.

Haver Analytics (via py/haver_client.py) is the primary source for every
indicator below - each fetcher tries Haver first and falls back to the
original FRED/yfinance/CPB/OECD/IMF/World Bank fetcher (unchanged) if the
Haver call fails, following the same tiered-fallback pattern this file
already used for FRED-vs-yfinance before Haver was added.
"""
import io
import os
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import haver_client
from paths import CSV

warnings.filterwarnings("ignore")

START_DATE = "2005-01-01"
FETCH_START_DATE = "2004-01-01"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# Set this in your shell (`export FRED_API_KEY=...`) to use the official,
# authenticated FRED REST API instead of the unauthenticated fredgraph.csv
# scrape below — the latter has been unreliable (aggressive rate limiting /
# timeouts) for the wider set of series this pipeline now pulls. Get a free
# key at https://fred.stlouisfed.org/docs/api/api_key.html
# Only used as a fallback now that Haver is the primary source (see
# py/haver_client.py / HAVER_API_KEY) for every indicator below.
FRED_API_KEY = os.environ.get("FRED_API_KEY")


def fetch_haver(database: str, series: str) -> pd.Series:
    """Thin wrapper so every fetcher below has a one-line Haver call to try
    first, matching the shape of fetch_fred_series()."""
    return haver_client.get_series(database, series, start_date=FETCH_START_DATE)


# =========================================================================
# 1. FRED (with Yahoo Finance fallback if FRED blocks/times out) — fallback
#    tier only now; Haver is tried first in every fetcher below.
# =========================================================================
def fetch_fred_series(series_id: str) -> pd.Series:
    if FRED_API_KEY:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": FETCH_START_DATE,
        }
        r = requests.get(url, params=params, headers=UA, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            raise RuntimeError(f"FRED API returned no observations for {series_id}")
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")["value"].dropna()

    # Unauthenticated fallback (no FRED_API_KEY set) — scrapes the same CSV
    # the interactive FRED chart downloads, but with no API key this is more
    # aggressively rate-limited and prone to timeouts.
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": series_id, "cosd": FETCH_START_DATE}
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()


def fetch_oil_shock() -> pd.Series:
    """Brent crude, monthly average, converted to a year-over-year % change.
    Haver (PEOBR@WBPRICES) first, FRED/yfinance fallback."""
    try:
        brent = fetch_haver("WBPRICES", "PEOBR")
    except Exception as e:
        print(f"  [fallback] Haver oil download failed ({e}), falling back to FRED/yfinance...")
        try:
            brent = fetch_fred_series("DCOILBRENTEU")
        except Exception as e2:
            print(f"  [fallback] FRED oil download failed ({e2}), pulling Brent oil (BZ=F) via yfinance...")
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
    Haver (NFCIM1@BCI) first, FRED/^VIX fallback.
    """
    try:
        nfci = fetch_haver("BCI", "NFCIM1")
    except Exception as e:
        print(f"  [fallback] Haver NFCI download failed ({e}), falling back to FRED/yfinance...")
        try:
            nfci = fetch_fred_series("NFCI")
        except Exception as e2:
            print(f"  [fallback] FRED NFCI download failed ({e2}), pulling ^VIX via yfinance...")
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
    "Dr. Copper" — LME copper price (Haver PNMCOP@WBPRICES, USD/metric ton,
    monthly), converted to year-over-year % change. Widely used as a leading
    proxy for global industrial demand (China construction/manufacturing in
    particular). Falls back to FRED's IMF-mirrored copper series, then CME
    copper futures (HG=F) via yfinance.
    """
    try:
        level = fetch_haver("WBPRICES", "PNMCOP")
        monthly = level.resample("MS").mean()
    except Exception as e:
        print(f"  [fallback] Haver copper download failed ({e}), falling back to FRED/yfinance...")
        try:
            level = fetch_fred_series("PCOPPUSDM")
            monthly = level.resample("MS").mean()
        except Exception as e2:
            print(f"  [fallback] FRED copper download failed ({e2}), pulling HG=F via yfinance...")
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
    10y-2y Treasury term spread, daily, resampled to monthly mean. A classic
    leading recession indicator (inversions have preceded every US recession
    since the 1970s with a ~12-18 month lead). Haver (FCM10/FCM2@USECON,
    computed as a spread) first, FRED T10Y2Y fallback; skipped (with a
    warning) if both fail, since no other clean fallback exists.
    """
    try:
        y10 = fetch_haver("USECON", "FCM10")
        y2 = fetch_haver("USECON", "FCM2")
        spread = (y10 - y2).dropna()
    except Exception as e:
        print(f"  [fallback] Haver yield curve download failed ({e}), falling back to FRED...")
        try:
            spread = fetch_fred_series("T10Y2Y")
        except Exception as e2:
            print(f"  [skip] FRED yield curve download failed ({e2}), dropping 'yield_curve' from the panel.")
            return None

    monthly = spread.resample("MS").mean()
    return monthly.rename("yield_curve")


def fetch_dollar_index() -> pd.Series:
    """
    Broad trade-weighted USD index, year-over-year % change, sign-flipped so
    + = dollar EASING. A strengthening dollar tightens financial conditions
    for EM/dollar-debt borrowers and is historically a headwind for global
    growth, hence the flip (consistent with the sign convention used for
    `fin`). Haver (FXTWBDI@USECON — the same Fed series FRED's DTWEXBGS
    mirrors) first, FRED/yfinance fallback.
    """
    try:
        level = fetch_haver("USECON", "FXTWBDI")
        monthly = level.resample("MS").mean()
    except Exception as e:
        print(f"  [fallback] Haver dollar index download failed ({e}), falling back to FRED/yfinance...")
        try:
            level = fetch_fred_series("DTWEXBGS")
            monthly = level.resample("MS").mean()
        except Exception as e2:
            print(f"  [fallback] FRED dollar index download failed ({e2}), pulling DX-Y.NYB via yfinance...")
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
    US High Yield option-adjusted spread, sign-flipped so + = spreads
    TIGHTENING (easy credit) — same convention as `fin`. Captures
    corporate credit-market stress more directly than NFCI.

    Haver's Bloomberg US Corporate HY OAS (LDCHOA@BONDINDX) only has history
    back to 2012 — spliced with FRED's longer ICE BofA HY OAS
    (BAMLH0A0HYM2) for everything before that, so the panel doesn't lose
    2005-2012 history. The two vendor indices aren't identical, so there's a
    methodology discontinuity at the splice point — accepted tradeoff to
    keep the column's full history rather than dropping it or truncating
    the whole panel to 2012+.
    """
    try:
        haver_spread = fetch_haver("BONDINDX", "LDCHOA")
    except Exception as e:
        print(f"  [fallback] Haver credit spread download failed ({e}), falling back to FRED...")
        try:
            spread = fetch_fred_series("BAMLH0A0HYM2")
        except Exception as e2:
            print(f"  [skip] FRED credit spread download failed ({e2}), dropping 'credit' from the panel.")
            return None
        monthly = spread.resample("MS").mean()
        return (-monthly).rename("credit")

    haver_monthly = haver_spread.resample("MS").mean()
    haver_start = haver_monthly.index.min()
    try:
        fred_monthly = fetch_fred_series("BAMLH0A0HYM2").resample("MS").mean()
    except Exception as e:
        print(f"  [warn] FRED HY OAS fetch failed ({e}) — using Haver credit spread alone "
              f"(history starts {haver_start.date()}, no pre-2012 splice).")
        return (-haver_monthly).rename("credit")

    if fred_monthly.index.min() < haver_start:
        print(f"  Splicing credit spread: FRED/ICE BofA HY OAS before {haver_start.date()}, "
              f"Bloomberg HY OAS (Haver) from {haver_start.date()} onward.")
        combined = pd.concat([fred_monthly[fred_monthly.index < haver_start], haver_monthly]).sort_index()
        return (-combined).rename("credit")
    return (-haver_monthly).rename("credit")


# =========================================================================
# 1b. Additional financial-conditions block (VIX, equities, bank equities,
#     lending standards, EM spread, housing, leverage) — Haver primary,
#     FRED/yfinance fallback.
# =========================================================================
def fetch_vix() -> pd.Series:
    """Cboe VIX, monthly average, sign-flipped z-score so + = calm (low vol),
    consistent with the fin/usd/credit sign convention. Haver
    (SPVIX@USECON) first, yfinance ^VIX fallback."""
    try:
        px = fetch_haver("USECON", "SPVIX")
    except Exception as e:
        print(f"  [fallback] Haver VIX download failed ({e}), falling back to yfinance...")
        import yfinance as yf
        df_yf = yf.download("^VIX", start=FETCH_START_DATE, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            px = df_yf["Close"]["^VIX"]
        else:
            px = df_yf["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]

    monthly = px.resample("MS").mean()
    z = (monthly - monthly.mean()) / monthly.std()
    return (-z).rename("vix")


def fetch_equity() -> pd.Series:
    """
    S&P 500 Composite, YoY % change — used in preference to a global ETF
    (e.g. ACWI) because global-equity ETFs mostly launched in 2008+ and
    would truncate the panel's usable history right before the GFC; the
    S&P 500 has decades of history and is highly correlated with global
    equity cycles anyway. US proxy for global equity performance. Haver
    (SP5COM@SPD, the real index) first, yfinance ^GSPC fallback.
    """
    try:
        px = fetch_haver("SPD", "SP5COM")
    except Exception as e:
        print(f"  [fallback] Haver S&P 500 download failed ({e}), falling back to yfinance...")
        import yfinance as yf
        df_yf = yf.download("^GSPC", start=FETCH_START_DATE, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            px = df_yf["Close"]["^GSPC"]
        else:
            px = df_yf["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]

    monthly = px.resample("MS").last()
    yoy = monthly.pct_change(12) * 100
    return yoy.rename("equity")


def fetch_bank_equity() -> pd.Series:
    """KBW Bank Index, YoY % change — global banking-sector health proxy (a
    common leading indicator of financial-system stress). Haver
    (SPKBW@USECON, the same index Yahoo's ^BKX mirrors) first, yfinance
    fallback."""
    try:
        px = fetch_haver("USECON", "SPKBW")
    except Exception as e:
        print(f"  [fallback] Haver KBW bank index download failed ({e}), falling back to yfinance...")
        import yfinance as yf
        df_yf = yf.download("^BKX", start=FETCH_START_DATE, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            px = df_yf["Close"]["^BKX"]
        else:
            px = df_yf["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]

    monthly = px.resample("MS").last()
    yoy = monthly.pct_change(12) * 100
    return yoy.rename("bank_equity")


def fetch_lending_standards() -> pd.Series:
    """
    Fed Senior Loan Officer Opinion Survey (Haver LCIQ157@BCI — the same
    survey FRED's DRTSCILM mirrors): net % of banks tightening C&I loan
    standards, quarterly. Sign-flipped so + = easing (consistent with
    fin/usd/credit), resampled to monthly via forward-fill. US-only (no
    clean global SLOOS equivalent), used as a bank-lending-cycle proxy — the
    same honest US-proxy tradeoff as `jobless_claims`/`housing` below.
    """
    try:
        q = fetch_haver("BCI", "LCIQ157")
    except Exception as e:
        print(f"  [fallback] Haver SLOOS download failed ({e}), falling back to FRED...")
        q = fetch_fred_series("DRTSCILM")

    monthly = (-q).resample("MS").ffill()
    return monthly.rename("lending_standards")


def fetch_em_spread() -> pd.Series:
    """
    EMBI Global sovereign spread (Haver GS@EMBI) — the real JPMorgan EMBI
    Global index, in basis points. Year-over-year change in the spread,
    sign-flipped so + = spreads TIGHTENING (easy EM conditions), same
    convention as `credit`. This closes what was previously an honest gap:
    real EMBI is a JPMorgan-licensed product with no free feed, so this
    indicator used to be proxied by the iShares MSCI EM ETF (EEM)'s YoY
    price return instead — that proxy is now the fallback only.
    """
    try:
        spread = fetch_haver("EMBI", "GS")
        monthly = spread.resample("MS").mean()
        yoy_change = monthly.diff(12)
        return (-yoy_change).rename("em_spread")
    except Exception as e:
        print(f"  [fallback] Haver EMBI download failed ({e}), falling back to EEM ETF proxy via yfinance...")
        import yfinance as yf
        df_yf = yf.download("EEM", start=FETCH_START_DATE, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            px = df_yf["Close"]["EEM"]
        else:
            px = df_yf["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        monthly = px.resample("MS").last()
        yoy = monthly.pct_change(12) * 100
        return yoy.rename("em_spread")


def fetch_housing() -> pd.Series:
    """
    S&P/Case-Shiller US National Home Price Index (Haver CASUSXAM@USECON),
    YoY % change. US-only proxy standing in for a global housing-price cycle
    — a genuine global property-price aggregate doesn't exist even via
    Haver, same honest-caveat tradeoff as before. FRED fallback.
    """
    try:
        level = fetch_haver("USECON", "CASUSXAM")
    except Exception as e:
        print(f"  [fallback] Haver Case-Shiller download failed ({e}), falling back to FRED...")
        level = fetch_fred_series("CSUSHPINSA")

    monthly = level.resample("MS").mean()
    yoy = monthly.pct_change(12) * 100
    return yoy.rename("housing")


def fetch_leverage() -> pd.Series:
    """
    BIS "Total Credit to Private Non-Financial Sector" for the US, as a % of
    GDP (Haver Q111RCRD@BIS — real BIS data, rather than the FRED mirror
    used previously), YoY change in the ratio (pp). US-only proxy for a
    global credit/leverage cycle — BIS's own global credit-to-GDP dataset
    has no single aggregate feed, so this stays a US proxy either way.
    """
    try:
        q = fetch_haver("BIS", "Q111RCRD")
    except Exception as e:
        print(f"  [fallback] Haver BIS credit-to-GDP download failed ({e}), falling back to FRED...")
        q = fetch_fred_series("CRDQUSAPABIS")

    monthly = q.resample("MS").ffill()
    yoy = monthly.diff(12)
    return yoy.rename("leverage")


def fetch_jobless_claims() -> pd.Series:
    """
    US initial jobless claims (Haver LICM@USECON — the same series FRED's
    ICSA mirrors), weekly, resampled to monthly mean and sign-flipped (YoY %
    change, negated) so + = claims falling = labor market improving. US-only
    proxy for global labor-market stress.
    """
    try:
        weekly = fetch_haver("USECON", "LICM")
    except Exception as e:
        print(f"  [fallback] Haver jobless claims download failed ({e}), falling back to FRED...")
        weekly = fetch_fred_series("ICSA")

    monthly = weekly.resample("MS").mean()
    yoy = monthly.pct_change(12) * 100
    return (-yoy).rename("jobless_claims")


# =========================================================================
# 1c. OECD-sourced block (retail sales, consumer/business confidence) —
#     Haver pulls these directly from OECD (C0*@OECDMEI), rather than via a
#     FRED mirror of the same underlying OECD MEI series.
# =========================================================================
def fetch_retail_sales() -> pd.Series:
    """OECD Total retail trade volume index (Haver C003ROI@OECDMEI),
    3-month-annualized growth (matching the trade/ip transform convention)."""
    try:
        level = fetch_haver("OECDMEI", "C003ROI")
    except Exception as e:
        print(f"  [fallback] Haver OECD retail sales download failed ({e}), falling back to FRED...")
        level = fetch_fred_series("SLRTTO01OEQ659S")

    monthly = level.resample("MS").ffill()
    g = (monthly / monthly.shift(3)) ** 4 - 1
    return (g * 100).rename("retail_sales")


def fetch_consumer_confidence() -> pd.Series:
    """OECD Total Consumer Confidence Indicator, amplitude-adjusted (Haver
    C003CCE@OECDMEI), de-meaned so 0 = long-run-average confidence."""
    try:
        level = fetch_haver("OECDMEI", "C003CCE")
    except Exception as e:
        print(f"  [fallback] Haver OECD consumer confidence download failed ({e}), falling back to FRED...")
        level = fetch_fred_series("CSCICP03O9M665S")

    monthly = level.resample("MS").mean()
    return (monthly - monthly.mean()).rename("consumer_conf")


def fetch_business_confidence() -> pd.Series:
    """OECD Total Manufacturing Industrial Confidence Indicator,
    amplitude-adjusted (Haver C003BMA@OECDMEI) — the closest OECD Total
    composite available; not identical to the all-sector Business Tendency
    Survey composite FRED's BSCICP03O9M665S mirrored, but the same concept
    (economy-wide business sentiment). De-meaned so 0 = long-run average."""
    try:
        level = fetch_haver("OECDMEI", "C003BMA")
    except Exception as e:
        print(f"  [fallback] Haver OECD business confidence download failed ({e}), falling back to FRED...")
        level = fetch_fred_series("BSCICP03O9M665S")

    monthly = level.resample("MS").mean()
    return (monthly - monthly.mean()).rename("business_conf")


# =========================================================================
# 1d. COVID stringency index — Oxford COVID-19 Government Response Tracker
#     (no Haver equivalent exists; unchanged)
# =========================================================================
def fetch_covid_stringency() -> pd.Series:
    """
    Oxford COVID-19 Government Response Tracker (OxCGRT), StringencyIndex
    averaged across all reporting countries per day, resampled to monthly,
    sign-flipped so + = less stringent (supportive of growth). Explicitly
    0 outside the pandemic's active-tracking window (2020-2022) — that's the
    real value (no lockdown in force), not a missing observation.
    """
    url = (
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/"
        "data/OxCGRT_compact_national_v1.csv"
    )
    df = pd.read_csv(url, usecols=["Date", "StringencyIndex_Average"], low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"])
    daily_avg = df.groupby("Date")["StringencyIndex_Average"].mean()
    monthly = daily_avg.resample("MS").mean()
    # Reindex through "now" (not just the source data's own last date) —
    # OxCGRT stopped tracking in ~Dec 2022, but that means 0 stringency for
    # every month since, all the way to the present, not a missing value.
    today = pd.Timestamp.today().normalize().replace(day=1)
    idx = pd.date_range(FETCH_START_DATE, max(monthly.index.max(), today), freq="MS")
    monthly = monthly.reindex(idx).fillna(0.0)
    return (-monthly).rename("covid_stringency")


# =========================================================================
# 2. World trade + industrial production — Haver primary (S001IQXM/S001XDG
#    @G10), CPB World Trade Monitor scrape as last-ditch fallback if BOTH
#    Haver calls fail.
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


def _to_3mo_annualized(level: pd.Series) -> pd.Series:
    level = level.sort_index()
    g = (level / level.shift(3)) ** 4 - 1
    return g * 100


def fetch_trade_and_ip():
    """World trade volume (S001IQXM@G10) and world industrial production
    (S001XDG@G10, World[incl US], World Trade Weight), both monthly SA
    indexes, converted to 3-month-annualized growth. Replaces the CPB World
    Trade Monitor Excel scrape as the primary source — CPB stays as a
    last-ditch fallback only if BOTH Haver calls fail, since it's the only
    other source that provides both series together."""
    try:
        trade_level = fetch_haver("G10", "S001IQXM")
        ip_level = fetch_haver("G10", "S001XDG")
        return _to_3mo_annualized(trade_level).rename("trade"), _to_3mo_annualized(ip_level).rename("ip")
    except Exception as e:
        print(f"  [fallback] Haver world trade/IP download failed ({e}), falling back to CPB World Trade Monitor scrape...")
        xls, xlsx_url = fetch_cpb_world_trade_monitor()
        return parse_cpb_trade_and_ip(xls)


# =========================================================================
# 3. Global Composite PMI — real S&P Global data via Haver
#    (SGBLVPTG@INTSRVYS), OECD Composite Leading Indicator as fallback
#    substitute if Haver is unreachable (same free-PMI-substitute caveat
#    this file used to carry as its primary approach).
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
    return rescaled.resample("MS").mean().rename("pmi")


def fetch_pmi() -> pd.Series:
    """Global Composite PMI — the real, licensed S&P Global Composite PMI
    (Haver SGBLVPTG@INTSRVYS), used directly at its natural 50+=expansion
    scale (no rescaling needed, unlike the OECD CLI substitute below).

    Haver's real PMI only has history back to 2021 — spliced with the OECD
    Composite Leading Indicator (rescaled to a PMI-like range) for
    everything before that, so the panel doesn't lose 2005-2021 history.
    This is the single largest-weighted indicator in the DFM, so dropping
    it (rather than splicing) would be a real regression, not just a
    quality tradeoff. The rescaled OECD CLI and the real PMI are different
    concepts at different scales, so there's a methodology discontinuity at
    the splice point — accepted tradeoff to keep the column's full history.
    """
    try:
        haver_pmi = fetch_haver("INTSRVYS", "SGBLVPTG").resample("MS").mean()
    except Exception as e:
        print(f"  [fallback] Haver Global Composite PMI download failed ({e}), falling back to OECD CLI substitute...")
        return fetch_pmi_proxy()

    haver_start = haver_pmi.index.min()
    try:
        proxy = fetch_pmi_proxy()
    except Exception as e:
        print(f"  [warn] OECD CLI PMI proxy fetch failed ({e}) — using real PMI alone "
              f"(history starts {haver_start.date()}, no pre-2021 splice).")
        return haver_pmi.rename("pmi")

    if proxy.index.min() < haver_start:
        print(f"  Splicing PMI: OECD CLI substitute before {haver_start.date()}, "
              f"real S&P Global Composite PMI (Haver) from {haver_start.date()} onward.")
        combined = pd.concat([proxy[proxy.index < haver_start], haver_pmi]).sort_index()
        return combined.rename("pmi")
    return haver_pmi.rename("pmi")


# =========================================================================
# 4. World real GDP growth (quarterly) — Haver primary
#    (S001XGPP@G10, World[incl US] Real GDP, PPP-weighted), IMF
#    SDMX/World Bank chain as fallback.
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

    # QGDP_WCA carries multiple TYPE_OF_TRANSFORMATION variants per indicator
    # per quarter (e.g. an index/level series AND a period-over-period %
    # change series) — without filtering on this, the two get silently mixed
    # together after the later dedup step, producing a nonsense series that
    # alternates between GDP levels and growth rates quarter to quarter.
    transformation_cols = [c for c in df.columns if "TRANSFORMATION" in c.upper()]

    def restrict_to_pop_pch(frame: pd.DataFrame) -> pd.DataFrame:
        if not transformation_cols:
            return frame
        is_pop_pch = pd.Series(False, index=frame.index)
        for c in transformation_cols:
            is_pop_pch |= frame[c].astype(str).str.contains("POP_PCH", case=False, na=False)
        return frame[is_pop_pch] if is_pop_pch.any() else frame

    indicator_text = world_df[indicator_cols[0]].astype(str)
    indicator_code = indicator_text.str.split(":", n=1).str[0].str.strip()
    if "B1GQ_S1_Q" in indicator_code.values:
        world_df = restrict_to_pop_pch(world_df[indicator_code == "B1GQ_S1_Q"])
    else:
        # (per-row scoring: the previous version aggregated indicator_text
        # into a single joined string before scoring, which both discarded
        # any per-row distinction and doesn't support .map() on a plain str)
        scores = indicator_text.map(indicator_score)
        best = scores.max()
        if best <= 0:
            raise RuntimeError("Could not identify a real/SA/QoQ GDP growth indicator row in QGDP_WCA.")
        world_df = restrict_to_pop_pch(world_df[scores == best])

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


def fetch_gdp_target() -> pd.Series:
    """World[incl US] Real GDP, PPP-weighted (Haver S001XGPP@G10) —
    genuinely quarterly, seasonally adjusted, real, global (not just OECD)
    GDP index. Converted to quarter-on-quarter annualized growth the same
    way fetch_imf_quarterly_gdp() does. Falls back to the IMF SDMX
    QGDP_WCA fetch, then World Bank annual GDP (interpolated to quarterly),
    if Haver is unreachable — that ~150-line fallback chain is kept as-is
    rather than deleted, since it's tested and working."""
    try:
        level = fetch_haver("G10", "S001XGPP")
        quarterly = level.resample("QS").mean()
        qoq = quarterly.pct_change()
        saar = ((1 + qoq) ** 4 - 1) * 100
        return saar.dropna().rename("gdp_growth_saar")
    except Exception as e:
        print(f"  [fallback] Haver world GDP download failed ({e}), falling back to IMF quarterly GDP (QGDP_WCA)...")
        try:
            return fetch_imf_quarterly_gdp()
        except Exception as e2:
            print(f"  [fallback] IMF quarterly GDP fetch failed ({e2}), falling back to World Bank annual GDP (interpolated to quarterly)...")
            gdp_annual = fetch_world_bank_annual_gdp_growth()
            return get_quarterly_gdp_target(gdp_annual)


# =========================================================================
# 5. AI / tech capex proxy — Philadelphia Semiconductor Index
# =========================================================================
def fetch_ai_tech_proxy() -> pd.Series:
    """Philadelphia Semiconductor Index — Haver (SPSOX@DAILY) is the same
    index Yahoo's ^SOX mirrors; yfinance is the fallback."""
    try:
        px = fetch_haver("DAILY", "SPSOX")
    except Exception as e:
        print(f"  [fallback] Haver semiconductor index download failed ({e}), falling back to yfinance...")
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
    if haver_client.API_KEY:
        print("HAVER_API_KEY detected — Haver Analytics is the primary data source.")
    else:
        print("No HAVER_API_KEY set — every fetch below will immediately fall back to "
              "FRED/yfinance/CPB/OECD/IMF. Set HAVER_API_KEY (see py/haver_client.py) "
              "to use Haver as the primary source.")
    if FRED_API_KEY:
        print("FRED_API_KEY detected — fallback FRED calls use the authenticated REST API.")
    else:
        print("No FRED_API_KEY set — fallback FRED calls (if triggered) use the "
              "unauthenticated fredgraph.csv scrape, more prone to rate limiting/timeouts.")

    print("Fetching Brent crude...")
    oil = fetch_oil_shock()

    print("Fetching financial conditions...")
    fin = fetch_financial_conditions()

    print("Fetching world trade + industrial production...")
    trade, ip = fetch_trade_and_ip()

    print("Fetching Global Composite PMI...")
    pmi = fetch_pmi()

    print("Fetching AI/tech capex proxy (semiconductor index)...")
    ai = fetch_ai_tech_proxy()

    print("Fetching copper price (Dr. Copper)...")
    copper = fetch_copper()

    print("Fetching 10y-2y yield curve slope...")
    yield_curve = fetch_yield_curve()

    print("Fetching broad USD index...")
    usd = fetch_dollar_index()

    print("Fetching high-yield credit spread...")
    credit = fetch_credit_spread()

    def try_fetch(label, fn):
        print(f"Fetching {label}...")
        try:
            return fn()
        except Exception as e:
            print(f"  [skip] {label} fetch failed ({e}), dropping from the panel.")
            return None

    vix = try_fetch("VIX", fetch_vix)
    equity = try_fetch("equity index (S&P 500)", fetch_equity)
    bank_equity = try_fetch("bank equity index (KBW)", fetch_bank_equity)
    lending_standards = try_fetch("bank lending standards (SLOOS)", fetch_lending_standards)
    em_spread = try_fetch("EM sovereign spread (EMBI Global)", fetch_em_spread)
    housing = try_fetch("housing prices (Case-Shiller)", fetch_housing)
    leverage = try_fetch("leverage / credit-to-GDP", fetch_leverage)
    jobless_claims = try_fetch("weekly jobless claims", fetch_jobless_claims)
    retail_sales = try_fetch("retail sales (OECD)", fetch_retail_sales)
    consumer_conf = try_fetch("consumer confidence (OECD)", fetch_consumer_confidence)
    business_conf = try_fetch("business confidence (OECD)", fetch_business_confidence)
    covid_stringency = try_fetch("COVID stringency index (OxCGRT)", fetch_covid_stringency)

    print("Fetching world real GDP growth (quarterly)...")
    gdp_quarterly = fetch_gdp_target()

    series = [
        pmi, trade, ip, oil, fin, ai, copper, yield_curve, usd, credit,
        vix, equity, bank_equity, lending_standards, em_spread, housing,
        leverage, jobless_claims, retail_sales, consumer_conf, business_conf,
        covid_stringency,
    ]

    # A fetch can technically succeed but return a series that's been
    # discontinued (e.g. a ticker that stopped updating years ago) or newly
    # created (recent index rebase means little history before some cutoff)
    # — either way, that column would silently truncate every OTHER
    # column's usable history once the modeling scripts do
    # dropna(subset=cols), since all columns must be non-null on the same
    # row. Better to drop a column like that here, with a clear reason,
    # than to let it wreck the whole panel's date range downstream.
    panel_start = pd.Timestamp(START_DATE)
    panel_end = pd.Timestamp.today().normalize().replace(day=1)
    checked_series = []
    for s in series:
        if s is None:
            continue
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            continue
        start_lag_months = (first.year - panel_start.year) * 12 + (first.month - panel_start.month)
        end_lag_months = (panel_end.year - last.year) * 12 + (panel_end.month - last.month)
        if start_lag_months > 30:
            print(f"  [skip] '{s.name}' data doesn't start until {first.date()} "
                  f"(panel starts {panel_start.date()}) — dropping to avoid truncating "
                  "every other column's usable history.")
            continue
        if end_lag_months > 6:
            print(f"  [skip] '{s.name}' data ends at {last.date()} (panel needs data "
                  f"through ~{panel_end.date()}) — likely discontinued, dropping.")
            continue
        checked_series.append(s)
    series = checked_series

    panel = pd.concat([s for s in series if s is not None], axis=1)
    panel = panel.loc[START_DATE:].resample("MS").mean()
    panel = panel.interpolate(limit=2)

    panel.to_csv(CSV / "panel_monthly.csv")
    gdp_quarterly.to_csv(CSV / "gdp_quarterly.csv")

    print("\nSaved panel_monthly.csv:")
    print(panel.tail(8))
    print("\nSaved gdp_quarterly.csv:")
    print(gdp_quarterly.tail(8))


if __name__ == "__main__":
    main()
