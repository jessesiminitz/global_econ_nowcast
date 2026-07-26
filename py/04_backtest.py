"""
Rolling out-of-sample comparison of the DFM baseline (02_estimate_dfm.py)
against the elastic-net/factor-ML challenger (03_estimate_ml.py), and winner
selection.

Walk-forward design: starting after a minimum training window, each
subsequent quarter t is nowcast by re-fitting BOTH models using only data
through t-1, then applying the fitted transform to quarter t's own
(already-realized) indicator readings — this is a nowcast, not a pure
ex-ante forecast, consistent with how the DFM's bridge regression already
works in-sample. Forecast errors are split into "stressed" (GFC 2008Q3-
2009Q2, COVID 2020Q1-2020Q3) vs. "calm" (everything else) quarters.

Winner rule: minimize max(calm_RMSE, stressed_RMSE) — the worst-case regime
error — rather than a naive pooled RMSE, so a model can't win purely by being
fine in the (numerically much larger) set of calm quarters while quietly
failing in crises. Both regimes' numbers are reported regardless of winner.

MIN_TRAIN_QUARTERS is deliberately short (3 years) so the walk-forward test
period starts before the GFC — a longer, more "proper" burn-in would push
the first test quarter past 2008 and make it impossible to evaluate either
model's crisis performance out-of-sample at all. Early folds are thin and
noisier as a result; this is a known, disclosed tradeoff, not a bug.
"""
import json

import numpy as np
import pandas as pd

from model_lib import fit_dfm, fit_elastic_net
from paths import CSV, JSON

MIN_TRAIN_QUARTERS = 12
STRESSED_WINDOWS = [
    ("2008-07-01", "2009-06-30"),  # GFC
    ("2020-01-01", "2020-09-30"),  # COVID
]

panel = pd.read_csv(CSV / "panel_monthly.csv", index_col=0, parse_dates=True)
gdp = pd.read_csv(CSV / "gdp_quarterly.csv", index_col=0, parse_dates=True).iloc[:, 0]

cols = [
    "pmi", "trade", "ip", "oil", "fin", "ai", "copper", "yield_curve", "usd", "credit",
    "vix", "equity", "bank_equity", "lending_standards", "em_spread", "housing",
    "leverage", "jobless_claims", "retail_sales", "consumer_conf", "business_conf",
    "covid_stringency",
]
cols = [c for c in cols if c in panel.columns]

usable_panel = panel.dropna(subset=cols)
test_quarters = gdp.index.intersection(usable_panel.resample("QS").mean().index)
test_quarters = test_quarters.sort_values()

if len(test_quarters) <= MIN_TRAIN_QUARTERS + 4:
    raise RuntimeError(
        f"Only {len(test_quarters)} usable quarters available (need > "
        f"{MIN_TRAIN_QUARTERS + 4} for a meaningful backtest) — check panel_monthly.csv "
        "coverage for the new indicator columns."
    )

records = []
print(f"Walk-forward backtest: {len(test_quarters) - MIN_TRAIN_QUARTERS} out-of-sample quarters "
      f"(training on the preceding {MIN_TRAIN_QUARTERS}+ quarters each time)...")

for i in range(MIN_TRAIN_QUARTERS, len(test_quarters)):
    target_q = test_quarters[i]
    train_cutoff_q = test_quarters[i - 1]
    train_panel_end = train_cutoff_q + pd.DateOffset(months=2)

    train_panel = panel.loc[:train_panel_end]
    train_gdp = gdp.loc[:train_cutoff_q]
    target_panel_slice = panel.loc[target_q: target_q + pd.DateOffset(months=2)]

    if target_panel_slice[cols].isna().any().any() or len(target_panel_slice) == 0:
        continue

    dfm_fit = fit_dfm(train_panel, train_gdp, cols)
    ml_fit = fit_elastic_net(train_panel, train_gdp, cols)

    dfm_pred_series = dfm_fit["predict"](target_panel_slice)
    ml_pred_series = ml_fit["predict"](target_panel_slice)

    if target_q not in dfm_pred_series.index or target_q not in ml_pred_series.index:
        continue

    records.append({
        "quarter": target_q,
        "actual": gdp.loc[target_q],
        "dfm_pred": dfm_pred_series.loc[target_q],
        "ml_pred": ml_pred_series.loc[target_q],
    })
    if (i - MIN_TRAIN_QUARTERS) % 10 == 0:
        print(f"  ...{i - MIN_TRAIN_QUARTERS + 1}/{len(test_quarters) - MIN_TRAIN_QUARTERS} done")

report = pd.DataFrame(records).set_index("quarter")
report["dfm_err"] = report["actual"] - report["dfm_pred"]
report["ml_err"] = report["actual"] - report["ml_pred"]

is_stressed = pd.Series(False, index=report.index)
for start, end in STRESSED_WINDOWS:
    is_stressed |= (report.index >= start) & (report.index <= end)
report["regime"] = np.where(is_stressed, "stressed", "calm")

report.to_csv(CSV / "backtest_report.csv")


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def mae(e):
    return float(np.mean(np.abs(e))) if len(e) else float("nan")


summary = {}
for model, err_col in [("dfm", "dfm_err"), ("elastic_net", "ml_err")]:
    calm_err = report.loc[report["regime"] == "calm", err_col]
    stressed_err = report.loc[report["regime"] == "stressed", err_col]
    overall_err = report[err_col]
    summary[model] = {
        "calm_rmse": rmse(calm_err), "calm_mae": mae(calm_err), "n_calm": int(len(calm_err)),
        "stressed_rmse": rmse(stressed_err), "stressed_mae": mae(stressed_err), "n_stressed": int(len(stressed_err)),
        "overall_rmse": rmse(overall_err), "overall_mae": mae(overall_err),
    }

winner = min(summary, key=lambda m: max(summary[m]["calm_rmse"], summary[m]["stressed_rmse"]))

print("\n=== Backtest summary (annualized pp of GDP growth) ===")
for model in summary:
    s = summary[model]
    flag = "  <-- winner (min worst-case regime RMSE)" if model == winner else ""
    print(f"{model:12s}  calm RMSE={s['calm_rmse']:.2f} (n={s['n_calm']})  "
          f"stressed RMSE={s['stressed_rmse']:.2f} (n={s['n_stressed']})  "
          f"overall RMSE={s['overall_rmse']:.2f}{flag}")

with open(JSON / "backtest_summary.json", "w") as f:
    json.dump({"winner": winner, **summary}, f, indent=2)

print(f"\nWinner: {winner}")
print("Wrote backtest_report.csv and backtest_summary.json")
