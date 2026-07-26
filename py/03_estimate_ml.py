"""
Elastic-net / factor-ML challenger to the DFM baseline (02_estimate_dfm.py).

Feature set: the quarterly-averaged, standardized indicator panel plus the
top few PCA factors of that same panel (a "factor-augmented" design) fed
into ElasticNetCV, which shrinks/selects among them via time-series cross-
validation. Because the whole pipeline is an affine function of the
quarterly z-scored indicators, it decomposes additively per indicator the
same way the DFM's bridge regression does — see model_lib.fit_elastic_net()
for the actual math (shared with 04_backtest.py's rolling comparison).

The saved decomposition also carries FORECAST_QUARTERS quarters beyond the
last published GDP print (an `is_forecast` column marks them) — see
model_lib.forecast_quarters() for how those are produced.
"""
import pandas as pd

from model_lib import fit_elastic_net, forecast_quarters
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

result = fit_elastic_net(panel, gdp, cols)
contrib = result["contrib"]
p = result["params"]

print(f"Elastic net: alpha={p['alpha_enet']:.4f}  l1_ratio={p['l1_ratio_enet']:.2f}  "
      f"n_factors={p['n_factors']}  R^2={p['r2']:.2f}")
print("Effective per-indicator weights (post factor fold-back):")
for c in cols:
    print(f"  {c:18s}  w={p['weights'][c]:+.3f}")

check = (contrib["fitted"] - (contrib[cols].sum(axis=1) + p["intercept"])).abs().max() < 1e-8
print(f"Decomposition additivity check (should be True): {check}")

forecast = forecast_quarters(panel, gdp, cols, p["mu"], p["sd"], p["const"],
                              p["effective_weights"], n_ahead=FORECAST_QUARTERS)
contrib["is_forecast"] = False
contrib = pd.concat([contrib, forecast])

contrib.to_csv(CSV / "quarterly_decomposition_ml.csv")

with open(TXT / "model_params_ml.txt", "w") as f:
    f.write(f"alpha_enet={p['alpha_enet']}\nl1_ratio_enet={p['l1_ratio_enet']}\n")
    f.write(f"n_factors={p['n_factors']}\nr2={p['r2']}\nintercept={p['intercept']}\n")
    f.write("weights=" + ",".join(f"{c}:{p['weights'][c]:.4f}" for c in cols) + "\n")

print("\nLatest 6 quarters, decomposition (pp contribution to annualized GDP growth):")
print(contrib.tail(6).round(2))
