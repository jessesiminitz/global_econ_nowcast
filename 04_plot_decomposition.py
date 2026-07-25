"""
Plot decomposition_export.json as a stacked-bar chart: quarterly GDP-growth
driver contributions, plus a line for actual GDP growth.

Usage:
    python3 04_plot_decomposition.py                     # reads ./decomposition_export.json
    python3 04_plot_decomposition.py path/to/other.json   # or point at a specific file

Output: decomposition_chart.png in the same folder.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "decomposition_export.json"

DRIVERS = [
    ("pmi", "Composite PMI", "#45d6c0"),
    ("trade", "Trade volumes", "#5b8def"),
    ("ip", "Industrial production", "#9b7cf0"),
    ("oil", "Oil price shock", "#ef7a56"),
    ("fin", "Financial conditions", "#e3b24e"),
    ("ai", "AI / tech capex", "#6bd66b"),
    ("residual", "Idiosyncratic residual", "#9aa5ad"),
]

with open(PATH) as f:
    data = json.load(f)

quarters = [d["q"] for d in data]
n = len(quarters)
x = np.arange(n)

fig, ax = plt.subplots(figsize=(max(11, n * 0.32), 5.2), dpi=150)

pos_bottom = np.zeros(n)
neg_bottom = np.zeros(n)
for key, label, color in DRIVERS:
    vals = np.array([d[key] for d in data])
    pos = np.clip(vals, 0, None)
    neg = np.clip(vals, None, 0)
    ax.bar(x, pos, bottom=pos_bottom, color=color, width=0.75, label=label)
    ax.bar(x, neg, bottom=neg_bottom, color=color, width=0.75)
    pos_bottom += pos
    neg_bottom += neg

actual = [d.get("actual") for d in data]
actual_x = [xi for xi, a in zip(x, actual) if a is not None]
actual_y = [a for a in actual if a is not None]
ax.plot(actual_x, actual_y, color="#1f2933", lw=1.8, marker="o", markersize=3,
        label="Actual GDP growth")

ax.axhline(0, color="#999", lw=0.7)
ax.set_xticks(x[::2])
ax.set_xticklabels(quarters[::2], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("pp contribution, annualized")
ax.set_title("Global GDP growth: driver decomposition by quarter")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("decomposition_chart.png", bbox_inches="tight")
print(f"Read {n} quarters from {PATH}")
print("Saved decomposition_chart.png")
