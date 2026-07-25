# Global GDP Dynamic Factor Model — pipeline

Run in order:

```
python3 01_build_panel.py     # builds panel_monthly.csv + gdp_quarterly.csv
python3 02_estimate_dfm.py    # PCA -> Kalman smoother -> bridge regression -> decomposition
python3 03_export.py          # validation_fit.png + decomposition_export.json
```

Requires: `numpy`, `pandas`, `scipy`, `matplotlib`. No other dependencies —
the dynamic factor model (Kalman filter, RTS smoother, EM-style two-step
estimation) is implemented from scratch in `02_estimate_dfm.py`, since this
environment doesn't have `statsmodels` or internet access to install it.

## Swapping in real data

`01_build_panel.py` currently *constructs* `panel_monthly.csv` from
documented macro history because this sandbox has no live network access —
see the caveat at the top of that file. To run on real vintage data:

1. Skip `01_build_panel.py`.
2. Build your own `panel_monthly.csv` with a `DatetimeIndex` and columns
   `pmi, trade, ip, oil, fin, ai` (rename/add columns as you like — just
   update the `cols` list at the top of `02_estimate_dfm.py`). Good real
   sources: S&P Global/J.P.Morgan Composite PMI, CPB World Trade Monitor
   (trade + industrial production), Brent crude spot (FRED: DCOILBRENTEU),
   a financial conditions index (e.g. Goldman Sachs GSFCI, Chicago Fed
   NFCI), and a capex/semiconductor-billings series as an AI/tech proxy.
3. Build `gdp_quarterly.csv` from actual quarterly world real GDP growth
   (IMF, World Bank, or a GDP-weighted aggregate of national accounts).
4. Run `02_estimate_dfm.py` and `03_export.py` unchanged.

## Files

- `panel_monthly.csv` — monthly indicator panel
- `gdp_quarterly.csv` — quarterly GDP growth target
- `quarterly_decomposition.csv` — full decomposition history (all quarters)
- `decomposition_export.json` — last 48 quarters, feeds the React chart
- `validation_fit.png` — actual vs. model-fitted GDP growth
- `model_params.txt` — estimated φ, loadings, bridge α/β, R²
