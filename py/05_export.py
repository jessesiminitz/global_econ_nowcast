"""
Export both models' decompositions (per 04_backtest.py for the winner flag)
and render, for EACH model separately (never mixed — a DFM chart shows only
DFM drivers and the DFM's own fit; same for elastic net):

  1. decomposition_dfm[_2022].png              - DFM stacked-bar decomposition
                                                  (all indicators), full + 2022-on.
  2. decomposition_elastic_net[_2022].png       - same, elastic net.
  3. decomposition_dfm_bucketed[_2022].png      - DFM decomposition with
                                                  indicators grouped into 5
                                                  categories instead of 22
                                                  individual bars.
  4. decomposition_elastic_net_bucketed[_2022].png - same, elastic net.
  5. validation_fit[_2022].png                  - actual vs. BOTH models'
                                                  fitted/forecast values (the
                                                  one chart that's explicitly
                                                  a model comparison).
  6. decomposition_export.json                  - last 48 quarters of the
                                                  winning model's decomposition,
                                                  feeds the React dashboard.

Color/layout follows a Kieran-Healy/Tufte-style approach (see the `dataviz`
skill): categorical hues assigned in a fixed order from a CVD-validated
8-slot palette (never eyeballed/cycled), a neutral (non-categorical) color
for the residual since it isn't a real driver, minimal chart chrome
(off-white surface, hairline recessive gridlines, muted axis ink, no
top/right border), and — since 22 individual indicators is well past the
~8 hues a categorical palette can carry — the "bucketed" charts are the
accessible, validated view (5 categories, straight from the palette); the
detailed 22-indicator charts shade each indicator within its category's hue
family (organized, but not independently CVD-validated the way the 5-bucket
view is).

Usage (from the repo root):
    python3 py/05_export.py
"""
import colorsys
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import CSV, JSON, PNG

ZOOM_START = "2022-01-01"

# ---- chart chrome (dataviz skill: references/palette.md, light mode) ----
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ---- validated categorical palette, fixed order, never cycled ----
# (references/palette.md — this order clears the adjacent-pair CVD/contrast
# gates in both light and dark mode; a prefix of an already-validated
# ordered sequence only drops pairs, so using the first 5 slots is covered.)
CATEGORIES = [
    ("Core Activity", ["pmi", "trade", "ip", "retail_sales", "consumer_conf", "business_conf", "jobless_claims"], "#2a78d6"),
    ("Financial Conditions", ["fin", "vix", "credit", "yield_curve", "usd", "equity", "bank_equity", "lending_standards", "em_spread"], "#eb6834"),
    ("Commodities & Tech", ["oil", "copper", "ai"], "#1baf7a"),
    ("GFC Credit Cycle", ["housing", "leverage"], "#eda100"),
    ("COVID Stringency", ["covid_stringency"], "#e87ba4"),
]
RESIDUAL_COLOR = INK_MUTED  # not a real driver, so it doesn't spend a categorical slot

INDICATOR_LABELS = {
    "pmi": "Composite PMI", "trade": "Trade volumes", "ip": "Industrial production",
    "oil": "Oil price shock", "fin": "Financial conditions", "ai": "AI / tech capex",
    "copper": "Copper (Dr. Copper)", "yield_curve": "Yield curve (10y-2y)",
    "usd": "USD index (easing = +)", "credit": "Credit spreads (HY OAS)",
    "vix": "VIX (calm = +)", "equity": "Equity market performance",
    "bank_equity": "Bank equity index", "lending_standards": "Bank lending standards (easing = +)",
    "em_spread": "EM / sovereign spread proxy", "housing": "Housing prices",
    "leverage": "Leverage / credit-to-GDP", "jobless_claims": "Jobless claims (falling = +)",
    "retail_sales": "Retail sales", "consumer_conf": "Consumer confidence",
    "business_conf": "Business confidence", "covid_stringency": "COVID stringency (less strict = +)",
}
ALL_COLS_ORDER = list(INDICATOR_LABELS.keys())


def _shade_family(base_hex, n, l_range=(0.38, 0.72)):
    """n distinguishable shades of one hue (lightness ramp, same hue/sat) —
    used to color several indicators within one category family on the
    detailed (non-bucketed) chart. Not independently CVD-validated; the
    bucketed chart (straight from the palette) is the validated view."""
    if n <= 0:
        return []
    if n == 1:
        return [base_hex]
    h = base_hex.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    lo, hi = l_range
    shades = []
    for i in range(n):
        li = lo + (hi - lo) * i / (n - 1)
        rr, gg, bb = colorsys.hls_to_rgb(hue, li, sat)
        shades.append("#" + "".join(f"{max(0, min(255, round(v * 255))):02x}" for v in (rr, gg, bb)))
    return shades


def build_detailed_drivers(cols_present):
    """[(key, label, color)] for every indicator present, colored by category
    family (fixed hue per category, shaded per member) — used by the
    non-bucketed decomposition charts."""
    drivers = []
    for _cat_name, members, base_color in CATEGORIES:
        present = [m for m in members if m in cols_present]
        for m, shade in zip(present, _shade_family(base_color, len(present))):
            drivers.append((m, INDICATOR_LABELS[m], shade))
    drivers.append(("residual", "Idiosyncratic residual", RESIDUAL_COLOR))
    return drivers


def build_bucket_drivers(cols_present):
    """[(category_name, category_name, color)] for categories with at least
    one present member — the validated, accessible view (<=5 hues + residual,
    straight from the fixed palette, well within the 8-slot CVD budget)."""
    drivers = []
    for cat_name, members, base_color in CATEGORIES:
        if any(m in cols_present for m in members):
            drivers.append((cat_name, cat_name, base_color))
    drivers.append(("residual", "Idiosyncratic residual", RESIDUAL_COLOR))
    return drivers


def add_bucket_sums(records, cols_present):
    """Return a copy of `records` with one extra key per category — the sum
    of that category's present member contributions — for the bucketed
    charts. The underlying per-indicator values are additive by construction
    (see model_lib.py), so category sums are exact, not approximations."""
    out = []
    for rec in records:
        new = dict(rec)
        for cat_name, members, _ in CATEGORIES:
            present = [m for m in members if m in cols_present]
            new[cat_name] = round(sum(rec.get(m, 0.0) for m in present), 3) if present else 0.0
        out.append(new)
    return out


if os.path.exists(JSON / "backtest_summary.json"):
    with open(JSON / "backtest_summary.json") as f:
        winner = json.load(f)["winner"]
else:
    winner = "dfm"
    print("[note] backtest_summary.json not found (run 04_backtest.py first) — defaulting to the DFM baseline.")

dfm_c = pd.read_csv(CSV / "quarterly_decomposition.csv", index_col=0, parse_dates=True)
ml_c = None
if os.path.exists(CSV / "quarterly_decomposition_ml.csv"):
    ml_c = pd.read_csv(CSV / "quarterly_decomposition_ml.csv", index_col=0, parse_dates=True)
winner_frame = dfm_c if winner == "dfm" else ml_c
print(f"Winning model: {winner}")


def _nz(v):
    """None (-> JSON null) for NaN, else a plain float — json.dumps emits a
    bare `NaN` token for float('nan'), which is not valid JSON and breaks
    JS's JSON.parse, so forecast rows' actual/residual must go through this."""
    return None if pd.isna(v) else round(float(v), 3)


def _split_forecast(frame):
    hist = frame[~frame.get("is_forecast", False).astype(bool)]
    fore = frame[frame.get("is_forecast", False).astype(bool)]
    return hist, fore


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_validation(dfm_frame, ml_frame, out_path, title_suffix=""):
    """Actual vs. BOTH models' fitted values — the one chart that's
    explicitly a model comparison, forecast segment visually distinct."""
    dfm_hist, dfm_fore = _split_forecast(dfm_frame)
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.plot(dfm_frame.index, dfm_frame["actual"], color=INK_PRIMARY, lw=2, label="Actual (model panel)")
    ax.plot(dfm_hist.index, dfm_hist["fitted"], color="#eb6834", lw=2, ls="--", label="DFM baseline fit")
    # Forecast segment: connect from the last historical point so the line
    # reads continuously, but render it visually distinct (lighter, dotted).
    dfm_bridge = pd.concat([dfm_hist.iloc[[-1]], dfm_fore]) if len(dfm_fore) and len(dfm_hist) else dfm_fore
    ax.plot(dfm_bridge.index, dfm_bridge["fitted"], color="#eb6834", lw=2, ls=":", alpha=0.7,
            label="DFM forecast (next 4Q)")
    if ml_frame is not None:
        ml_hist, ml_fore = _split_forecast(ml_frame)
        ax.plot(ml_hist.index, ml_hist["fitted"], color="#2a78d6", lw=2, ls="-.", label="Elastic-net / factor-ML fit")
        ml_bridge = pd.concat([ml_hist.iloc[[-1]], ml_fore]) if len(ml_fore) and len(ml_hist) else ml_fore
        ax.plot(ml_bridge.index, ml_bridge["fitted"], color="#2a78d6", lw=2, ls=":", alpha=0.7,
                label="Elastic-net forecast (next 4Q)")
    if len(dfm_fore):
        ax.axvspan(dfm_bridge.index[0], dfm_frame.index[-1], color=INK_MUTED, alpha=0.08, zorder=0)
        ax.axvline(dfm_bridge.index[0], color=INK_MUTED, lw=0.8, ls="--")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title(f"Global GDP growth (annualized): actual vs. model fit  (winner: {winner}){title_suffix}")
    ax.set_ylabel("%, annualized q/q")
    legend = ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path.name}")


def make_records(frame, cols_order):
    df2 = frame.copy()
    df2.index = df2.index.strftime("%Y-%m")
    records = []
    for idx, row in df2.iterrows():
        records.append({
            "q": idx,
            **{k: round(float(row[k]), 3) for k in cols_order},
            "trend": round(float(row["trend"]), 3),
            "fitted": round(float(row["fitted"]), 3),
            "actual": _nz(row["actual"]),
            "residual": _nz(row["residual"]),
            "is_forecast": bool(row.get("is_forecast", False)),
        })
    return records


def plot_decomposition(records, drivers, out_path, model_label, is_winner, model_color, title_suffix=""):
    """
    Stacked-bar driver decomposition for ONE model only — `drivers` is
    whatever set of (key, label, color) triples that model's own chart
    should show (individual indicators or category buckets). Alongside the
    bars: "Actual GDP growth" and THIS model's own fitted/forecast line —
    never a line from the other model.

    Bars are based at the model's long-term trend (its bridge-regression
    intercept/`alpha` for the DFM, its intercept `c0` for elastic net) rather
    than at zero, so they visually read as "how far above/below the
    long-run average did each driver push growth this quarter" — the same
    absolute %-growth scale the "Actual GDP growth" line is already plotted
    on. (Previously the bars were zero-based, i.e. implicitly showing
    deviation-from-trend without the trend drawn anywhere, so the bar stack
    and the actual-growth line never visually lined up — they were offset
    by exactly the trend amount.)
    """
    quarters = [d["q"] for d in records]
    n = len(quarters)
    x = np.arange(n)
    trend = np.array([d["trend"] for d in records])

    active_drivers = [(key, label, color) for key, label, color in drivers if key in records[0]]

    fig, ax = plt.subplots(figsize=(max(11, n * 0.32), 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    is_forecast = np.array([d.get("is_forecast", False) for d in records])
    # Forecast bars use a hatch + lower alpha so they read as distinct from
    # realized history at a glance, without needing a separate color scale.
    hatch = np.where(is_forecast, "///", None)
    alpha = np.where(is_forecast, 0.55, 1.0)

    pos_bottom = trend.copy()
    neg_bottom = trend.copy()
    for key, label, color in active_drivers:
        # `residual` is None for forecast quarters (no actual to compare
        # against) — nothing to show there, so treat it as 0 rather than
        # crashing np.clip on a None.
        vals = np.array([d[key] if d.get(key) is not None else 0.0 for d in records])
        pos = np.clip(vals, 0, None)
        neg = np.clip(vals, None, 0)
        for i in range(n):
            # A thin surface-color edge on every segment is the "gap" that
            # separates touching bar segments, instead of a heavier border.
            ax.bar(x[i], pos[i], bottom=pos_bottom[i], color=color, width=0.75,
                   alpha=alpha[i], hatch=hatch[i], edgecolor=SURFACE, linewidth=0.6,
                   label=label if i == 0 else None)
            ax.bar(x[i], neg[i], bottom=neg_bottom[i], color=color, width=0.75,
                   alpha=alpha[i], hatch=hatch[i], edgecolor=SURFACE, linewidth=0.6)
        pos_bottom += pos
        neg_bottom += neg

    actual = [d.get("actual") for d in records]
    actual_x = [xi for xi, a in zip(x, actual) if a is not None]
    actual_y = [a for a in actual if a is not None]
    ax.plot(actual_x, actual_y, color=INK_PRIMARY, lw=2, marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=0.6, label="Actual GDP growth")

    # This model's own fitted/forecast line — solid through history, dotted
    # (bridging from the last historical point) over the forecast horizon.
    fitted = np.array([d["fitted"] for d in records])
    hist_idx = np.where(~is_forecast)[0]
    fore_idx = np.where(is_forecast)[0]
    if len(hist_idx):
        ax.plot(x[hist_idx], fitted[hist_idx], color=model_color, lw=1.8, ls="--",
                label=f"{model_label} fitted")
    if len(fore_idx):
        bridge_idx = (np.append(hist_idx[-1:], fore_idx) if len(hist_idx) else fore_idx)
        ax.plot(x[bridge_idx], fitted[bridge_idx], color=model_color, lw=1.8, ls=":", alpha=0.8,
                label=f"{model_label} forecast")

    if is_forecast.any():
        first_forecast_x = x[is_forecast][0]
        ax.axvline(first_forecast_x - 0.5, color=INK_MUTED, lw=0.8, ls="--")
        ax.axvspan(first_forecast_x - 0.5, x[-1] + 0.5, color=INK_MUTED, alpha=0.06, zorder=0)
        ax.text(first_forecast_x - 0.5, ax.get_ylim()[1], " forecast →", fontsize=8,
                color=INK_SECONDARY, va="top", ha="left", style="italic")

    ax.axhline(trend[0], color=INK_SECONDARY, lw=1.2, ls="--", zorder=2,
               label=f"Long-term trend ({trend[0]:+.1f}pp)")
    ax.axhline(0, color=AXIS, lw=0.6, alpha=0.6)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(
        [f"{quarters[i]}*" if is_forecast[i] else quarters[i] for i in range(0, n, 2)],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("%, annualized q/q (bars deviate from long-term trend)")
    ax.set_title(f"Global GDP growth: {model_label} driver decomposition by quarter  "
                 f"({'winner' if is_winner else 'challenger'}; * = forecast){title_suffix}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path.name} ({n} quarters)")


# ---- full-history + zoomed validation chart (the one model-comparison chart) ----
plot_validation(dfm_c, ml_c, PNG / "validation_fit.png")
dfm_zoom_frame = dfm_c[dfm_c.index >= ZOOM_START]
ml_zoom_frame = ml_c[ml_c.index >= ZOOM_START] if ml_c is not None else None
plot_validation(dfm_zoom_frame, ml_zoom_frame, PNG / "validation_fit_2022.png", title_suffix=" · 2022 onward")

# ---- JSON export (last 48 quarters, plus the forecast quarters already
#      appended at the tail by 02_estimate_dfm.py/03_estimate_ml.py) for the
#      front-end chart — winning model only ----
cols_order = [c for c in ALL_COLS_ORDER if c in winner_frame.columns]
records_48 = make_records(winner_frame.tail(48), cols_order)

with open(JSON / "decomposition_export.json", "w") as f:
    json.dump(records_48, f, indent=2)

print(f"exported {len(records_48)} quarters to decomposition_export.json")
print(json.dumps(records_48[-1], indent=2))

# ---- per-model decomposition charts: detailed (all indicators) + bucketed
#      (5 categories), each full-history and zoomed to 2022 onward. Each
#      model's chart shows ONLY that model's own drivers and fit — never a
#      line from the other model. ----
# Same two hues plot_validation() uses for these models, so a reader moving
# between validation_fit.png and a decomposition chart sees consistent colors.
MODEL_FRAMES = [
    ("dfm", "DFM", dfm_c, "#eb6834"),
    ("elastic_net", "elastic-net/factor-ML", ml_c, "#2a78d6"),
]

for model_key, model_label, frame, model_color in MODEL_FRAMES:
    if frame is None:
        print(f"[note] no decomposition available for {model_key} — skipping its charts.")
        continue
    is_winner = (model_key == winner)
    cols_present = [c for c in ALL_COLS_ORDER if c in frame.columns]
    detailed_drivers = build_detailed_drivers(cols_present)
    bucket_drivers = build_bucket_drivers(cols_present)

    for suffix, zoom_frame in [("", frame), ("_2022", frame[frame.index >= ZOOM_START])]:
        title_suffix = " · 2022 onward" if suffix else ""
        recs = make_records(zoom_frame, cols_present)
        recs_bucketed = add_bucket_sums(recs, cols_present)

        plot_decomposition(recs, detailed_drivers, PNG / f"decomposition_{model_key}{suffix}.png",
                            model_label, is_winner, model_color, title_suffix)
        plot_decomposition(recs_bucketed, bucket_drivers, PNG / f"decomposition_{model_key}_bucketed{suffix}.png",
                            model_label, is_winner, model_color, title_suffix)
