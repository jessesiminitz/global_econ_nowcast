"""
Estimate a single-factor dynamic factor model on the monthly indicator
panel, following the standard two-step estimator (Doz, Giannone & Reichlin,
2011, "A two-step estimator for large approximate dynamic factor models
based on Kalman filtering", J. Econometrics):

  Step 1 (static):  standardize indicators -> first principal component
                     gives factor loadings w and the static factor F_t.
                     model_lib.fit_dfm supports more factors via its
                     n_factors parameter (K=3 clears 60.9% cumulative
                     variance vs. 38.8% for K=1) — tried as a fix here,
                     but reverted: verified via 04_backtest.py's
                     walk-forward harness that K=2/K=3 both have WORSE
                     calm-regime RMSE than K=1 (more parameters fit on
                     training windows as short as 12-19 quarters early in
                     the backtest overfits, despite the better full-sample
                     variance coverage). See model_lib.fit_dfm's docstring.
  Step 2 (dynamic):  fit an AR(1) to F_t, then run a Kalman filter/smoother
                     over the state-space {f_t = phi f_{t-1} + w_t,
                     x_t = Lambda f_t + e_t} to get the smoothed monthly
                     factor f_t|T — diagnostic only today, doesn't feed the
                     bridge regression, decomposition, or predict().

Then: bridge regression of quarterly GDP growth (demeaned by a trailing
20-quarter trend, not a full-sample intercept — see
model_lib.TREND_WINDOW_QUARTERS; an unrepresentative early period like
2005-2007's pre-GFC boom otherwise permanently biases every fold of an
expanding backtest window) on the quarterly-averaged factor gives the
mapping from "common cycle" to growth units. Because the Step-1 factor is
an exact linear combination of the standardized indicators
(F_t = sum_i w_i * z_i,t), the bridge-implied GDP growth is *exactly*
decomposable into additive per-indicator contributions — this is what
feeds the stacked-bar chart.

This is the pipeline's DFM *baseline* — see 03_estimate_ml.py for the
elastic-net/factor-ML challenger, and 04_backtest.py for the rolling
out-of-sample comparison between the two. The actual Kalman-filter/bridge
math lives in model_lib.fit_dfm() so it can be reused unchanged by the
backtest.

The saved decomposition also carries FORECAST_QUARTERS quarters beyond the
last published GDP print (an `is_forecast` column marks them) — see
model_lib.forecast_quarters() for how those are produced.
"""
import pandas as pd

from model_lib import fit_dfm, forecast_quarters
from paths import CSV, TXT

FORECAST_QUARTERS = 4

panel = pd.read_csv(CSV / "panel_monthly.csv", index_col=0, parse_dates=True)
gdp = pd.read_csv(CSV / "gdp_quarterly.csv", index_col=0, parse_dates=True).iloc[:, 0]

cols = [
    "pmi", "trade", "ip", "oil", "fin", "ai", "copper", "yield_curve", "usd", "credit",
    "vix", "equity", "bank_equity", "lending_standards", "em_spread", "housing",
    "leverage", "jobless_claims", "retail_sales", "consumer_conf", "business_conf",
    "covid_stringency",
]
cols = [c for c in cols if c in panel.columns]

result = fit_dfm(panel, gdp, cols)
contrib = result["contrib"]
p = result["params"]

factor_word = "factor" if p["n_factors"] == 1 else "factors"
print(f"Share of common variance explained by {p['n_factors']} {factor_word}: {p['var_explained']:.1%}")
print("Standardized weights (bridge-scaled, per unit z-score):")
for c in cols:
    print(f"  {c:18s}  w={p['weights'][c]:+.3f}")

if p["n_factors"] == 1:
    beta_str = f"{p['beta'][0]:+.2f} * Factor"
else:
    beta_str = " ".join(f"{b:+.2f}*F{k}" for k, b in enumerate(p["beta"]))
print(f"\nBridge regression: GDP_growth = {p['alpha']:.2f} {beta_str}   (R^2={p['r2']:.2f})")

check = (contrib["fitted"] - (contrib[cols].sum(axis=1) + p["alpha"])).abs().max() < 1e-8
print(f"Decomposition additivity check (should be True): {check}")

forecast = forecast_quarters(panel, gdp, cols, p["mu"], p["sd"], p["const"],
                              p["effective_weights"], n_ahead=FORECAST_QUARTERS)
contrib["is_forecast"] = False
contrib = pd.concat([contrib, forecast])

contrib.to_csv(CSV / "quarterly_decomposition.csv")
result["panel_with_factor"].to_csv(CSV / "panel_with_factor.csv")

with open(TXT / "model_params.txt", "w") as f:
    f.write(f"n_factors={p['n_factors']}\nvar_explained={p['var_explained']}\nphi={p['phi']}\nsigma_w2={p['sigma_w2']}\n")
    f.write(f"alpha={p['alpha']}\n")
    f.write("beta=" + ",".join(f"F{k}:{b:.4f}" for k, b in enumerate(p["beta"])) + "\n")
    f.write(f"r2={p['r2']}\n")
    f.write("weights=" + ",".join(f"{c}:{p['weights'][c]:.4f}" for c in cols) + "\n")
    f.write("loadings_on_F=" + ",".join(f"{c}:{p['loadings_on_F'][c]:.4f}" for c in cols) + "\n")

print("\nLatest 6 quarters, decomposition (pp contribution to annualized GDP growth):")
print(contrib.tail(6).round(2))
