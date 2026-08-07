# Global GDP Dynamic Factor Model — pipeline

Files are organized by type — `py/` (scripts), `csv/`, `json/`, `png/`
(charts), `txt/` (model params) — see [Files](#files) below. Every script
resolves these folders relative to its own location (`py/paths.py`), so run
from the repo root or from inside `py/`, either works.

Run this on a machine with internet access, from the repo root:

```
pip install requests pandas numpy scipy scikit-learn openpyxl yfinance matplotlib pytest
echo "HAVER_API_KEY=your_key_here" > .env   # required, see below
export FRED_API_KEY=your_key_here           # optional, used only as a fallback if Haver is unreachable
python3 py/01_build_panel_real.py   # csv/panel_monthly.csv + csv/gdp_quarterly.csv
python3 py/02_estimate_dfm.py       # DFM baseline -> csv/quarterly_decomposition.csv
python3 py/03_estimate_ml.py        # elastic-net/factor-ML challenger -> csv/quarterly_decomposition_ml.csv
python3 py/04_backtest.py           # rolling out-of-sample comparison -> json/backtest_summary.json
python3 py/05_export.py             # png/*.png charts + json/decomposition_export.json
```

### Running tests

```
pytest tests/
```

`tests/` covers `py/model_lib.py` — the shared `fit_dfm`/`fit_elastic_net`/
`forecast_quarters` logic every other script depends on — against a small
fixed synthetic panel (`tests/conftest.py`), not the real Haver-pulled data
(that needs network access and drifts every re-pull, which would make tests
flaky). It locks in the mechanisms behind this project's own accuracy fixes
so a future change can't silently reintroduce them: the decomposition's
additivity (bars must sum to the fitted line, for both models and for
forecast quarters, even when the output cap below rescales them), the
trailing-window trend actually using recent data, the DFM's ridge
regularization actually shrinking its bridge-regression coefficients
relative to the unregularized fit it replaced, the elastic-net's 1-SE rule
always picking at least as much regularization as the raw CV-optimal
choice, and both `fit_elastic_net.predict()` and `forecast_quarters()`
staying inside their plausibility bound on deliberately extreme inputs.
That last set of tests was checked to actually fail when the bound they
cover is disabled (not just pass regardless) — see the comments in
`tests/test_model_lib.py` for why that check matters: an earlier draft of
those tests computed its own tolerance from the same constant the code
under test reads, so it couldn't have caught the bound being removed.

`01_build_panel_real.py` pulls real data with no manual CSV assembly. Haver
Analytics is the primary source for every indicator (via `py/haver_client.py`,
a standalone HaverView REST client — see "Haver API key" below); each fetcher
falls back to the original free source (FRED/yfinance/CPB/OECD/IMF/World Bank)
if the Haver call fails, so the pipeline still runs without a Haver
subscription, just on lower-quality proxies. The original 10-indicator core:

| Column | Haver source | Fallback | Notes |
|---|---|---|---|
| `oil` | `PEOBR@WBPRICES` (Brent spot) | FRED `DCOILBRENTEU` / yfinance `BZ=F` | YoY % change |
| `fin` | `NFCIM1@BCI` (Chicago Fed NFCI) | FRED `NFCI` / yfinance `^VIX` | sign-flipped so + = easy |
| `trade`, `ip` | `S001IQXM@G10` (world trade volume), `S001XDG@G10` (world IP) | CPB World Trade Monitor scrape | 3-month-annualized growth |
| `pmi` | `SGBLVPTG@INTSRVYS` — **real S&P Global Composite PMI** | OECD Composite Leading Indicator (rescaled) | used at its natural 50+=expansion scale |
| `ai` | `SPSOX@DAILY` (Philadelphia Semiconductor Index) | yfinance `^SOX` | YoY % change, z-scored, tanh-compressed |
| `copper` | `PNMCOP@WBPRICES` (LME copper) | FRED `PCOPPUSDM` / yfinance `HG=F` | "Dr. Copper" — global industrial/China demand proxy; YoY % change |
| `yield_curve` | `FCM10@USECON` − `FCM2@USECON` (10y-2y spread) | FRED `T10Y2Y` | classic leading recession indicator; dropped from the panel (with a warning) if both are unreachable |
| `usd` | `FXTWBDI@USECON` (broad trade-weighted USD index) | FRED `DTWEXBGS` / yfinance `DX-Y.NYB` | sign-flipped YoY so + = dollar easing (typically supportive of global growth) |
| `credit` | `LDCHOA@BONDINDX` (Bloomberg US Corporate HY OAS) | FRED `BAMLH0A0HYM2` (ICE BofA) | sign-flipped so + = spreads tightening (easy credit); dropped from the panel (with a warning) if both are unreachable |
| GDP target | `S001XGPP@G10` — **World[incl US] Real GDP, PPP-weighted, quarterly** | IMF SDMX `QGDP_WCA`, then World Bank `NY.GDP.MKTP.KD.ZG` (annual, interpolated) | real quarterly, seasonally-adjusted world GDP growth, compounded to an annualized rate |

Plus 12 more indicators added to widen the panel (core activity, financial
conditions, and a COVID-crisis proxy):

| Column | Haver source | Fallback | Notes |
|---|---|---|---|
| `vix` | `SPVIX@USECON` (CBOE VIX) | yfinance `^VIX` | sign-flipped z-score so + = calm |
| `equity` | `SP5COM@SPD` (S&P 500 Composite) | yfinance `^GSPC` | YoY % change; US proxy for global equities (global-equity ETFs mostly launched 2008+, which would truncate the panel right before the GFC) |
| `bank_equity` | `SPKBW@USECON` (KBW Bank Index) | yfinance `^BKX` | YoY % change — bank-sector health/stress proxy |
| `lending_standards` | `LCIQ157@BCI` (Fed SLOOS, net % tightening C&I loans) | FRED `DRTSCILM` | US-only proxy for the global bank-lending cycle; sign-flipped so + = easing |
| `em_spread` | `GS@EMBI` — **real JPMorgan EMBI Global sovereign spread (bp)** | yfinance `EEM` (MSCI EM ETF, YoY price return) | YoY change in spread (bp), sign-flipped so + = spreads tightening |
| `housing` | `CASUSXAM@USECON` (Case-Shiller US National) | FRED `CSUSHPINSA` | YoY % change; US proxy for a global housing cycle (no global aggregate feed exists, even via Haver) |
| `leverage` | `Q111RCRD@BIS` (US private nonfinancial credit-to-GDP ratio) | FRED `CRDQUSAPABIS` | YoY change in the ratio (pp); US proxy for a global credit/leverage cycle |
| `jobless_claims` | `LICM@USECON` (US initial claims) | FRED `ICSA` | sign-flipped YoY % change so + = claims falling |
| `retail_sales` | `C003ROI@OECDMEI` (OECD Total retail trade volume) | FRED `SLRTTO01OEQ659S` (same series, FRED-mirrored) | 3-month-annualized growth |
| `consumer_conf` | `C003CCE@OECDMEI` (OECD Total Consumer Confidence) | FRED `CSCICP03O9M665S` (same series, FRED-mirrored) | de-meaned level |
| `business_conf` | `C003BMA@OECDMEI` (OECD Total Mfg Industrial Confidence) | FRED `BSCICP03O9M665S` | de-meaned level; closest OECD Total composite, not identical to FRED's all-sector Business Tendency Survey mirror |
| `covid_stringency` | *(no Haver equivalent)* | Oxford COVID-19 Government Response Tracker (OxCGRT), static historical CSV | sign-flipped so + = less stringent; genuinely 0 outside 2020-2022, not a missing value |

One honest caveat remains, worth reading before you trust the output:
**Two indicators are still honest US-only proxies for a global concept**
(`housing`, `leverage`) because no clean global feed exists — not even via
Haver (BIS's own global credit-to-GDP and property-price aggregates don't
come as a single feed). `lending_standards` and `jobless_claims` are also
US-only, but as US-cycle indicators (labor market, bank lending) that's a
smaller stretch than for a price index. A handful of items on the original
candidate list (freight rates/Baltic Exchange, container throughput, IATA
airline data, Google/Apple Mobility, Google Trends, hospitalization data,
Bloomberg's Financial Conditions Index, restaurant/card-spend proxies, IMF
BOP capital-flow data) were **not implemented** — mostly discontinued free
feeds or concepts with no clean Haver series identified.

If a Haver mnemonic is ever retired/renamed, or the CPB scraper breaks
(their workbook layout can change release to release), each fetcher fails
gracefully and drops its column (with a printed warning) rather than
crashing the rest of the pipeline, so a single broken source degrades the
panel rather than blocking it. Run `python3 py/01_build_panel_real.py
--inspect-cpb` to dump the CPB workbook's sheet names and a preview of each
if you need to adjust `parse_cpb_trade_and_ip()`.

### Haver API key (required)

`py/haver_client.py` calls Haver Analytics' HaverView REST API
(`https://api.haverview.com`) directly — a standalone client, not the MCP
Haver tools (which only exist inside a live Claude Code session). It reads
`HAVER_API_KEY` from the environment, loaded from a `.env` file in the repo
root (git-ignored — never commit a key):

```
echo "HAVER_API_KEY=your_key_here" > .env
```

Without it, every fetcher in `01_build_panel_real.py` immediately falls back
to its free-source equivalent (FRED/yfinance/CPB/OECD/IMF) — the pipeline
still runs, just on the lower-quality proxies documented in the table above.

### FRED API key (optional, fallback only)

`fetch_fred_series()` is now only reached when a Haver call fails. It
defaults to scraping the same unauthenticated CSV endpoint the interactive
FRED chart uses (`fredgraph.csv`), which has proven prone to aggressive rate
limiting and timeouts. Get a free key at
https://fred.stlouisfed.org/docs/api/api_key.html, then:

```
export FRED_API_KEY=your_key_here
```

to make any fallback FRED call use the official authenticated
`api.stlouisfed.org/fred/series/observations` endpoint instead.

## Model comparison: DFM baseline vs. elastic-net/factor-ML challenger

`02_estimate_dfm.py` is the baseline (a 3-factor DFM — an earlier
unregularized attempt at this overfit and was reverted to single-factor,
but ridge-regularizing the bridge regression across K=3 factors instead
(`model_lib.DFM_RIDGE_LAMBDA`) beats single-factor on calm RMSE, stressed
RMSE, *and* overall RMSE simultaneously, measured via `04_backtest.py` —
not just a calm/stressed tradeoff, which is what ridge-regularizing a
single factor alone produces instead; see `model_lib.py`'s module-level
comment for the full sweep). The DFM still fits a scalar AR(1) + Kalman
filter/smoother on the first factor, but only as a diagnostic — wiring the
smoothed factor(s) into the bridge regression itself (the Doz-Giannone-
Reichlin two-step estimator's actual design) was tried and reverted after
an ablation against the real backtest moved overall RMSE by nothing and
calm RMSE by noise-level amounts either way; see `fit_dfm`'s docstring for
the numbers. Both models anchor their trend/level to a trailing
20-quarter window rather than a full-sample intercept, so an unrepresentative
early period like 2005-2007's pre-GFC boom can't permanently bias the
forecast under an expanding backtest window — see `model_lib.py`'s
module-level comments for the numbers behind both changes.
`03_estimate_ml.py` is a challenger: the quarterly-averaged, standardized
indicator panel plus its top PCA factors, fed into `ElasticNetCV` (shrinkage/
selection via time-series cross-validation). The regularization strength
itself is picked by the "1-SE rule" (largest alpha within one standard error
of the CV-optimal one), not the raw CV-minimizing alpha — measured directly
via `04_backtest.py`, this cut calm-regime RMSE by ~27% and nearly
eliminated a -0.27 lag-1 autocorrelation in the walk-forward errors, at the
cost of a much sparser, more conservative model (most indicators now get
exactly zero weight — see `model_lib.py`'s module-level comment for the
full numbers). Its `predict()` also clips its output to the training
window's own trend +- 8 standard deviations of realized GDP growth — a
walk-forward-safe plausibility bound (derived only from data the fold has
already seen) against wild extrapolation on genuinely out-of-distribution
quarters like 2020Q2; this cut stressed-regime RMSE by ~39% and roughly
halved the single worst backtest error, with zero effect on ordinary
predictions (an earlier attempt at the same goal — clipping the *inputs*
instead — was tried and reverted after making things worse; see
`model_lib.py`). Both share their core fitting logic via `model_lib.py`
(`fit_dfm` / `fit_elastic_net`), which is what lets `04_backtest.py` refit
either model on any training window.

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

**Is the winner robust, or an artifact of one particular backtest window?**
Checked directly rather than left as a vague caveat: elastic-net wins under
every regime configuration tested against the same 73-quarter backtest —
full data (worst-regime RMSE 8.41 vs DFM's 13.95), with the 3 COVID
quarters excluded entirely (2.06 vs 4.87), with the 4 GFC quarters excluded
entirely (12.63 vs 20.56), and calm-only with both stressed windows
dropped (1.50 vs 1.57 — the closest margin of the four, but ML still
ahead). The win isn't being carried by one crisis window.

**Splice-point check.** `01_build_panel_real.py` splices two indicators to
extend their history: PMI (real S&P Global Composite PMI from Haver only
goes back to 2021, backfilled with a rescaled OECD CLI before that) and
credit spreads (Bloomberg HY OAS from Haver only goes back to 2012,
backfilled with FRED's ICE BofA HY OAS before that). Checked both splice
points for backtest error clustering: the credit splice (2012) shows no
effect — mean absolute error in the surrounding year is *better* than the
73-quarter average for both models. The PMI splice (2021) window does show
elevated error for both models (DFM 2.95 vs 1.88 overall mean-abs-error;
ML 3.57 vs 1.37), concentrated in the first two quarters after the splice
date — but that window is also the volatile vaccine-rollout reopening
period, so this can't be cleanly attributed to the splice discontinuity
itself versus genuine regime volatility with only 2 quarters of
splice-adjacent data to go on. No code change from this — noted here as an
open, unresolved question rather than a fix.

**Ensemble/blend of both models.** Tested directly using the real backtest's
already-computed out-of-sample predictions (`csv/backtest_report.csv`'s
`dfm_pred`/`ml_pred` columns), sweeping every weighted blend from pure DFM
to pure elastic-net: every blend with nonzero DFM weight makes stressed
RMSE worse than pure elastic-net's 8.41 (e.g. a 50/50 blend: stressed RMSE
10.74), and the small calm-RMSE gain a blend buys (best case 1.40 vs pure
ML's 1.50, at roughly 40-50% DFM weight) doesn't offset it under the
project's own win criterion (minimize the worse of the two regime RMSEs).
Pure elastic-net dominates every blend tested. No ensemble adopted.

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
- `py/haver_client.py` — standalone Haver Analytics (HaverView REST API) client, the primary data source for `01_build_panel_real.py`
- `py/02_estimate_dfm.py` — DFM baseline (Kalman filter/smoother, bridge regression, decomposition)
- `py/03_estimate_ml.py` — elastic-net/factor-ML challenger
- `py/04_backtest.py` — rolling out-of-sample comparison + winner selection
- `py/05_export.py` — charts + JSON export for the React dashboard, all for the winning model (plus dual-model chart variants)
- `py/model_lib.py` — shared `fit_dfm()`/`fit_elastic_net()`/`forecast_quarters()` used by all three modeling scripts above
- `py/paths.py` — shared `CSV`/`JSON`/`PNG`/`TXT` folder constants
- `tests/test_model_lib.py` / `tests/conftest.py` — regression tests for `model_lib.py` against a small synthetic fixture (`pytest tests/`, see "Running tests" above)
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
