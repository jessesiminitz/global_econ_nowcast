# Global GDP Dynamic Factor Model — pipeline

Files are organized by type — `py/` (scripts), `csv/`, `json/`, `png/`
(charts), `txt/` (model params) — see [Files](#files) below. Every script
resolves these folders relative to its own location (`py/paths.py`), so run
from the repo root or from inside `py/`, either works.

Run this on a machine with internet access, from the repo root:

```
pip install requests pandas numpy scipy scikit-learn openpyxl yfinance matplotlib
export FRED_API_KEY=your_key_here   # optional but recommended, see below
python3 py/01_build_panel_real.py   # csv/panel_monthly.csv + csv/gdp_quarterly.csv
python3 py/02_estimate_dfm.py       # DFM baseline -> csv/quarterly_decomposition.csv
python3 py/03_estimate_ml.py        # elastic-net/factor-ML challenger -> csv/quarterly_decomposition_ml.csv
python3 py/04_backtest.py           # rolling out-of-sample comparison -> json/backtest_summary.json
python3 py/05_export.py             # png/*.png charts + json/decomposition_export.json
```

`01_build_panel_real.py` pulls real data with no manual CSV assembly. The
original 10-indicator core:

| Column | Source | Notes |
|---|---|---|
| `oil` | FRED `DCOILBRENTEU` (Brent crude) | no API key needed |
| `fin` | FRED `NFCI` (Chicago Fed) | sign-flipped so + = easy |
| `trade`, `ip` | CPB World Trade Monitor, live "latest" release | scrapes the current month's Excel automatically |
| `pmi` | OECD Composite Leading Indicator | **substitute for the real PMI — see caveat below** |
| `ai` | Yahoo Finance `^SOX` (semiconductor index), via `yfinance` | optional; skipped gracefully if not installed |
| `copper` | FRED `PCOPPUSDM` (IMF global copper price) | "Dr. Copper" — global industrial/China demand proxy; YoY % change |
| `yield_curve` | FRED `T10Y2Y` (10y-2y Treasury spread) | classic leading recession indicator; dropped from the panel (with a warning) if FRED is unreachable — no clean fallback |
| `usd` | FRED `DTWEXBGS` (broad trade-weighted USD index) | sign-flipped YoY so + = dollar easing (typically supportive of global growth) |
| `credit` | FRED `BAMLH0A0HYM2` (ICE BofA US HY OAS) | sign-flipped so + = spreads tightening (easy credit); dropped from the panel (with a warning) if FRED is unreachable |
| GDP target | IMF Quarterly GDP database, World and Country Aggregates (`QGDP_WCA`), via the IMF SDMX 3.0 API | real quarterly, seasonally-adjusted world GDP growth, compounded to an annualized rate; falls back to World Bank `NY.GDP.MKTP.KD.ZG` (annual, interpolated to quarterly) with a warning if the IMF API is unreachable |

Plus 12 more indicators added to widen the panel (core activity, financial
conditions, and a COVID-crisis proxy):

| Column | Source | Notes |
|---|---|---|
| `vix` | Yahoo Finance `^VIX` | sign-flipped z-score so + = calm |
| `equity` | Yahoo Finance `^GSPC` (S&P 500) | YoY % change; US proxy for global equities (global-equity ETFs mostly launched 2008+, which would truncate the panel right before the GFC) |
| `bank_equity` | Yahoo Finance `^BKX` (KBW Bank Index) | YoY % change — bank-sector health/stress proxy |
| `lending_standards` | FRED `DRTSCILM` (Fed SLOOS, net % tightening C&I loans) | US-only proxy for the global bank-lending cycle; sign-flipped so + = easing |
| `em_spread` | Yahoo Finance `EEM` (MSCI Emerging Markets ETF) | YoY % change, EM-risk-appetite proxy standing in for a true EMBI sovereign spread (JPMorgan-licensed, no free feed) |
| `housing` | FRED `CSUSHPINSA` (Case-Shiller US National) | YoY % change; US proxy for a global housing cycle (no free global BIS aggregate feed) |
| `leverage` | FRED `CRDQUSAPABIS` (BIS credit-to-GDP for the US, mirrored on FRED) | YoY change in the ratio (pp); US proxy for a global credit/leverage cycle |
| `jobless_claims` | FRED `ICSA` (US initial claims) | sign-flipped YoY % change so + = claims falling |
| `retail_sales` | FRED `SLRTTO01OEQ659S` (OECD Total retail trade volume, mirrored on FRED) | 3-month-annualized growth |
| `consumer_conf` | FRED `CSCICP03O9M665S` (OECD Consumer Confidence, mirrored on FRED) | de-meaned level |
| `business_conf` | FRED `BSCICP03O9M665S` (OECD Business Confidence, mirrored on FRED) | de-meaned level |
| `covid_stringency` | Oxford COVID-19 Government Response Tracker (OxCGRT), static historical CSV | sign-flipped so + = less stringent; genuinely 0 outside 2020-2022, not a missing value |

Two honest caveats, worth reading before you trust the output:
1. **The real J.P.Morgan/S&P Global Composite PMI is a paid, licensed
   dataset with no free API.** This script substitutes the OECD's
   Composite Leading Indicator, a genuinely free and conceptually similar
   (but numerically different) leading indicator. If you have a Refinitiv/
   Bloomberg/S&P subscription, swap in the real series.
2. **Several of the 12 newer indicators are honest US-only proxies for a
   global concept** (`equity`, `lending_standards`, `housing`, `leverage`,
   `jobless_claims`) because no clean free global feed exists — the same
   tradeoff as the PMI substitute above. A handful of items on the original
   candidate list (freight rates/Baltic Exchange, container throughput,
   IATA airline data, Google/Apple Mobility, Google Trends, hospitalization
   data, real EMBI sovereign spreads, Bloomberg's Financial Conditions
   Index, restaurant/card-spend proxies, IMF BOP capital-flow data) were
   **not implemented** — they're either paid-vendor-only or their free
   feeds have been discontinued.

If the CPB scraper breaks (their workbook layout can change release to
release), run `python3 py/01_build_panel_real.py --inspect-cpb` to dump the
sheet names and a preview of each, then adjust `parse_cpb_trade_and_ip()`.
The OECD/BIS-derived FRED series above are similarly liable to need a ticker
adjustment if FRED renames/retires one of them — each fetcher fails
gracefully and drops its column (with a printed warning) rather than
crashing the rest of the pipeline, so a single broken source degrades the
panel rather than blocking it.

### FRED API key (recommended)

`fetch_fred_series()` (used by 13 of the 22 indicators, plus the GDP-target
fallback) defaults to scraping the same unauthenticated CSV endpoint the
interactive FRED chart uses (`fredgraph.csv`), which has proven prone to
aggressive rate limiting and timeouts now that this pipeline pulls many more
FRED series than before. Get a free key at
https://fred.stlouisfed.org/docs/api/api_key.html, then:

```
export FRED_API_KEY=your_key_here
```

before running `py/01_build_panel_real.py`. With the key set, every FRED fetch
uses the official authenticated `api.stlouisfed.org/fred/series/observations`
endpoint instead — same series IDs, just a more reliable transport. Without
it, the pipeline still runs, just falling back to the old scrape (with a
startup message saying so).

## Model comparison: DFM baseline vs. elastic-net/factor-ML challenger

`02_estimate_dfm.py` is the baseline (unchanged single-factor DFM).
`03_estimate_ml.py` is a challenger: the quarterly-averaged, standardized
indicator panel plus its top PCA factors, fed into `ElasticNetCV` (shrinkage/
selection via time-series cross-validation). Both share their core fitting
logic via `model_lib.py` (`fit_dfm` / `fit_elastic_net`), which is what lets
`04_backtest.py` refit either model on any training window.

`04_backtest.py` runs a walk-forward comparison: starting from a short
burn-in period, each subsequent quarter is nowcast by both models using only
data through the prior quarter, then compared to the realized GDP print.
Errors are split into **stressed** quarters (GFC: 2008Q3-2009Q2, COVID:
2020Q1-2020Q3) and **calm** quarters (everything else), with RMSE/MAE
computed for both. The winner is whichever model minimizes the *worse* of
its two regime RMSEs — not the pooled average — so a model can't win by
being fine in the (much larger) set of calm quarters while quietly failing
in a crisis. Both models' regime numbers are always written to
`csv/backtest_report.csv`/`json/backtest_summary.json` regardless of which
one wins, so the tradeoff stays visible. `05_export.py` reads the declared
winner and exports its decomposition to the dashboard (`decomposition_export.json`).

### Charts

Every chart comes in a full-history version and a version zoomed to 2022
onward (`_2022` suffix) — the full-history stacked-bar charts compress
15+ years into one plot, so the zoomed versions are usually the more
readable day-to-day view.

`validation_fit[_2022].png` is the one chart that compares both models —
actual GDP growth vs. both the DFM's and the elastic net's fitted/forecast
lines. Every *decomposition* chart, by contrast, is single-model: a DFM
decomposition chart shows only DFM drivers and the DFM's own fit; an elastic
net decomposition chart shows only elastic net drivers — never a line from
the other model. There are four per model:

- `decomposition_dfm[_2022].png` / `decomposition_elastic_net[_2022].png` —
  every indicator that model uses, individually.
- `decomposition_dfm_bucketed[_2022].png` / `decomposition_elastic_net_bucketed[_2022].png` —
  the same decomposition with indicators grouped into 5 categories (Core
  Activity, Financial Conditions, Commodities & Tech, GFC Credit Cycle, COVID
  Stringency) instead of ~17-22 individual bars.

Color and layout follow the `dataviz` skill (a Kieran-Healy/Tufte-style
approach): 8 categorical hues in a fixed order, validated for colorblind-safe
adjacent-pair separation — never hand-picked or reassigned based on what's
present. Since 17-22 individual indicators is well past what any categorical
palette can carry distinctly, the **bucketed charts are the validated,
accessible view** (5 categories, straight from the palette); the detailed
per-indicator charts shade each indicator within its category's hue family
(organized by family, but not independently validated the same way). The
residual bar uses a neutral gray rather than a categorical hue, since it
isn't a real driver. Chart chrome (off-white surface, hairline recessive
gridlines, muted axis ink, no top/right border) follows the same skill.

## Files

Organized by type — every script resolves these paths relative to its own
location (`py/paths.py`), regardless of your current directory:

- `py/01_build_panel_real.py` — pulls real live data (run locally, needs internet)
- `py/02_estimate_dfm.py` — DFM baseline (Kalman filter/smoother, bridge regression, decomposition)
- `py/03_estimate_ml.py` — elastic-net/factor-ML challenger
- `py/04_backtest.py` — rolling out-of-sample comparison + winner selection
- `py/05_export.py` — charts + JSON export for the React dashboard, all for the winning model (plus dual-model chart variants)
- `py/model_lib.py` — shared `fit_dfm()`/`fit_elastic_net()`/`forecast_quarters()` used by all three modeling scripts above
- `py/paths.py` — shared `CSV`/`JSON`/`PNG`/`TXT` folder constants
- `jsx/global_gdp_nowcast_dfm.jsx` — React/Recharts dashboard; renders whichever indicator keys are in the loaded export
- `csv/panel_monthly.csv` — monthly indicator panel
- `csv/gdp_quarterly.csv` — quarterly GDP growth target
- `csv/quarterly_decomposition.csv` / `csv/quarterly_decomposition_ml.csv` — full decomposition history (plus 4 forecast quarters), DFM / elastic-net respectively
- `csv/backtest_report.csv` — per-quarter out-of-sample errors, both models
- `json/backtest_summary.json` — calm/stressed/overall RMSE & MAE per model, plus the declared winner
- `json/decomposition_export.json` — last 48 quarters (including forecast quarters) of the winning model's decomposition, feeds the React chart
- `png/validation_fit[_2022].png` — actual GDP growth vs. both models' fitted/forecast values (the one cross-model chart)
- `png/decomposition_dfm[_2022].png` / `png/decomposition_elastic_net[_2022].png` — per-model stacked-bar decomposition, every indicator
- `png/decomposition_dfm_bucketed[_2022].png` / `png/decomposition_elastic_net_bucketed[_2022].png` — same, indicators grouped into 5 categories
- `txt/model_params.txt` / `txt/model_params_ml.txt` — estimated parameters for the DFM / elastic-net model respectively
