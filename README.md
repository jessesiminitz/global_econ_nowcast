# Global GDP Dynamic Factor Model — pipeline

Two ways to build the panel:

## Option A — real live data (run this on your own machine)

```
pip install requests pandas numpy scipy openpyxl yfinance matplotlib
python3 01_build_panel_real.py
python3 02_estimate_dfm.py
python3 03_export.py
python3 04_plot_decomposition.py   # optional: decomposition_chart.png (matplotlib alternative to the React dashboard)
```

`01_build_panel_real.py` pulls real data with no manual CSV assembly:

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
| GDP target | World Bank `NY.GDP.MKTP.KD.ZG` (annual), interpolated to quarterly | **weakest link — see caveat below** |

Two honest caveats baked into the script's docstring, worth reading before
you trust the output:
1. **The real J.P.Morgan/S&P Global Composite PMI is a paid, licensed
   dataset with no free API.** This script substitutes the OECD's
   Composite Leading Indicator, a genuinely free and conceptually similar
   (but numerically different) leading indicator. If you have a Refinitiv/
   Bloomberg/S&P subscription, swap in the real series.
2. **There's no clean free API for quarterly world GDP.** The script
   interpolates the World Bank's annual figure — fine for a trend line,
   not a substitute for real quarterly data. This is the single highest-
   value manual upgrade if you have access to a better GDP series (e.g.
   OECD quarterly national accounts, or a data vendor's tracker).

If the CPB scraper breaks (their workbook layout can change release to
release), run `python3 01_build_panel_real.py --inspect-cpb` to dump the
sheet names and a preview of each, then adjust `parse_cpb_trade_and_ip()`.

## Option B — synthetic panel calibrated to documented history

Use this if you don't have (or don't yet want to debug) live internet
access to the sources above — e.g. to test the estimation code itself.

```
python3 01_build_panel.py     # builds panel_monthly.csv + gdp_quarterly.csv
python3 02_estimate_dfm.py    # PCA -> Kalman smoother -> bridge regression -> decomposition
python3 03_export.py          # validation_fit.png + decomposition_export.json
python3 04_plot_decomposition.py  # optional: decomposition_chart.png
```

Requires: `numpy`, `pandas`, `scipy`, `matplotlib`. The dynamic factor
model (Kalman filter, RTS smoother, EM-style two-step estimation) is
implemented from scratch in `02_estimate_dfm.py` — no `statsmodels`
dependency.

`01_build_panel.py` *constructs* `panel_monthly.csv` from documented macro
history (see the caveat at the top of that file) rather than pulling real
vintages — it has no live network access. `02_estimate_dfm.py`,
`03_export.py`, and `04_plot_decomposition.py` are identical either way;
they just expect `panel_monthly.csv` with a `gdp_quarterly.csv` target and
any subset of the columns
`pmi, trade, ip, oil, fin, ai, copper, yield_curve, usd, credit`
(all three scripts, plus the `global_gdp_nowcast_dfm.jsx` dashboard, auto-detect
which of these are actually present, so the pipeline still runs unchanged on
an older 6-column panel).

## Files

- `01_build_panel_real.py` — pulls real live data (run locally, needs internet)
- `01_build_panel.py` — builds a synthetic, history-calibrated panel (works offline)
- `02_estimate_dfm.py` — DFM estimation (Kalman filter/smoother, bridge regression, decomposition)
- `03_export.py` — validation chart + JSON export for the React dashboard
- `04_plot_decomposition.py` — optional matplotlib stacked-bar decomposition chart (alternative to the React dashboard), reads `decomposition_export.json`
- `global_gdp_nowcast_dfm.jsx` — React/Recharts dashboard; renders whichever indicator keys are in the loaded export
- `panel_monthly.csv` — monthly indicator panel (from whichever builder you ran)
- `gdp_quarterly.csv` — quarterly GDP growth target
- `quarterly_decomposition.csv` — full decomposition history (all quarters)
- `decomposition_export.json` — last 48 quarters, feeds the React chart
- `validation_fit.png` — actual vs. model-fitted GDP growth
- `decomposition_chart.png` — output of `04_plot_decomposition.py`
- `model_params.txt` — estimated φ, loadings, bridge α/β, R²

