import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c = pd.read_csv("quarterly_decomposition.csv", index_col=0, parse_dates=True)

# ---- validation chart ----
fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
ax.plot(c.index, c["actual"], color="#1f2933", lw=1.8, label="Actual (model panel)")
ax.plot(c.index, c["fitted"], color="#e0724a", lw=1.8, ls="--", label="DFM bridge fit")
ax.axhline(0, color="#999", lw=0.6)
ax.set_title("Global GDP growth (annualized): actual vs. dynamic-factor bridge fit")
ax.set_ylabel("%, annualized q/q")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("validation_fit.png")
print("saved validation_fit.png")

# ---- JSON export (last 48 quarters) for the front-end chart ----
tail = c.tail(48).copy()
tail.index = tail.index.strftime("%Y-%m")
cols_order = ["pmi", "trade", "ip", "oil", "fin", "ai", "copper", "yield_curve", "usd", "credit"]
cols_order = [col for col in cols_order if col in tail.columns]
records = []
for idx, row in tail.iterrows():
    records.append({
        "q": idx,
        **{k: round(float(row[k]), 3) for k in cols_order},
        "trend": round(float(row["trend"]), 3),
        "fitted": round(float(row["fitted"]), 3),
        "actual": round(float(row["actual"]), 3),
        "residual": round(float(row["residual"]), 3),
    })

with open("decomposition_export.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"exported {len(records)} quarters to decomposition_export.json")
print(json.dumps(records[-1], indent=2))