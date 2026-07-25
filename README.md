# Global GDP Dynamic Factor Model — pipeline

Two ways to build the panel:

## Option A — real live data (run this on your own machine)

```
pip install requests pandas numpy scipy openpyxl yfinance
python3 01_build_panel_real.py
python3 02_estimate_dfm.py
python3 03_export.py
```

`01_build_panel_real.py` pulls real data with no manual CSV assembly:

| Column | Source | Notes |
|---|---|---|
| `oil` | FRED `DCOILBRENTEU` (Brent crude) | no API key needed |
| `fin` | FRED `NFCI` (Chicago Fed) | sign-flipped so + = easy |
| `trade`, `ip` | CPB World Trade Monitor, live "latest" release | scrapes the current month's Excel automatically |
| `pmi` | OECD Composite Leading Indicator | **substitute for the real PMI — see caveat below** |
| `ai` | Yahoo Finance `^SOX` (semiconductor index), via `yfinance` | optional; skipped gracefully if not installed |
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
```

Requires: `numpy`, `pandas`, `scipy`, `matplotlib`. The dynamic factor
model (Kalman filter, RTS smoother, EM-style two-step estimation) is
implemented from scratch in `02_estimate_dfm.py` — no `statsmodels`
dependency.

`01_build_panel.py` *constructs* `panel_monthly.csv` from documented macro
history (see the caveat at the top of that file) rather than pulling real
vintages — it has no live network access. `02_estimate_dfm.py` and
`03_export.py` are identical either way; they just expect
`panel_monthly.csv` with columns `pmi, trade, ip, oil, fin, ai` and a
`gdp_quarterly.csv` target.

## Files

- `01_build_panel_real.py` — pulls real live data (run locally, needs internet)
- `01_build_panel.py` — builds a synthetic, history-calibrated panel (works offline)
- `02_estimate_dfm.py` — DFM estimation (Kalman filter/smoother, bridge regression, decomposition)
- `03_export.py` — validation chart + JSON export for the React dashboard
- `panel_monthly.csv` — monthly indicator panel (from whichever builder you ran)
- `gdp_quarterly.csv` — quarterly GDP growth target
- `quarterly_decomposition.csv` — full decomposition history (all quarters)
- `decomposition_export.json` — last 48 quarters, feeds the React chart
- `validation_fit.png` — actual vs. model-fitted GDP growth
- `model_params.txt` — estimated φ, loadings, bridge α/β, R²

