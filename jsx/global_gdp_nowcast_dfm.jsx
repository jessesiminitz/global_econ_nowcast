import React, { useMemo, useState } from "react";
import {
  ComposedChart, Bar, Cell, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ReferenceArea, ResponsiveContainer, Legend,
} from "recharts";

/* =================================================================
   GLOBAL GDP NOWCAST — 3-factor Dynamic Factor Model (ridge-regularized)
   Estimated with the two-step Doz–Giannone–Reichlin procedure, extended
   to multiple static factors:
     Step 1: PCA on the standardized monthly indicator panel -> top-3
             static factors (exact linear combinations, so the
             decomposition below is additive by construction across all
             3, not an approximation). An earlier UNREGULARIZED attempt
             at K=2/K=3 overfit the short early training windows and made
             calm-regime RMSE worse despite explaining more full-sample
             variance (60.9% at K=3 vs. 38.8% at K=1) — reverted at the
             time. What actually worked: ridge-regularizing the bridge
             regression below (model_lib.DFM_RIDGE_LAMBDA) instead of
             dropping back to K=1 — this beats single-factor on calm,
             stressed, AND overall RMSE simultaneously (1.65/15.62/5.09 ->
             1.57/13.95/4.57), not just a calm/stressed tradeoff, which is
             what ridge-regularizing a single factor alone produces
             instead. See model_lib.py's module-level comment for the
             full sweep.
     Step 2: AR(1) factor dynamics fit + Kalman filter/smoother, still on
             the first factor alone — diagnostic only, doesn't feed the
             bridge regression or decomposition below.
   Bridge regression: quarterly GDP growth ~ trend + beta·factors, where
   trend is a trailing 20-quarter average (not a full-sample intercept —
   an early, unrepresentative period like 2005-2007's pre-GFC boom
   otherwise permanently biases the forecast under an expanding backtest
   window).
   The panel now supports up to 22 indicators — the original 10 (PMI,
   trade, industrial production, oil, financial conditions, AI/tech
   capex, copper, the 10y-2y yield curve slope, the broad USD index, and
   high-yield credit spreads) plus 12 more covering VIX, equities, bank
   equities, lending standards, an EM/sovereign spread proxy, housing,
   leverage, jobless claims, retail sales, consumer/business confidence,
   and a COVID stringency index — see
   01_build_panel_real.py. A second model (elastic-net/factor-ML, see
   03_estimate_ml.py) competes against the DFM via a rolling backtest
   (04_backtest.py); 05_export.py exports whichever one wins. This
   component renders whichever driver keys are present in the loaded
   export, so it works unchanged for any subset of the known indicators.
   Model fit stats below (54.4% var. explained, bridge R² = 0.93) are
   from the bundled 6-indicator example run. See methodology panel for
   the important caveat about this environment having no live
   data-vendor access.
================================================================= */

// ---- output of 02_estimate_dfm.py / 03_estimate_ml.py / 05_export.py, last 48 quarters ----
// This is the bundled example run (6-indicator panel, predates the
// copper/yield-curve/USD/credit additions below). Use "Load
// decomposition_export.json" in the header to swap in your own
// freshly-generated 10-indicator export.
const DEFAULT_DECOMP_DATA = [{"q":"2014-10","pmi":0.299,"trade":0.321,"ip":0.015,"oil":-0.612,"fin":0.014,"ai":-0.242,"trend":2.922,"fitted":2.718,"actual":3.831,"residual":1.113},{"q":"2015-01","pmi":0.043,"trade":0.06,"ip":0.052,"oil":-0.8,"fin":0.235,"ai":-0.264,"trend":2.922,"fitted":2.246,"actual":3.769,"residual":1.522},{"q":"2015-04","pmi":-0.231,"trade":-0.309,"ip":-0.219,"oil":-0.8,"fin":0.146,"ai":-0.267,"trend":2.922,"fitted":1.242,"actual":2.114,"residual":0.872},{"q":"2015-07","pmi":-0.224,"trade":-0.56,"ip":-0.069,"oil":-0.861,"fin":0.025,"ai":-0.23,"trend":2.922,"fitted":1.002,"actual":1.564,"residual":0.561},{"q":"2015-10","pmi":-0.261,"trade":-0.194,"ip":0.152,"oil":-0.863,"fin":0.096,"ai":-0.145,"trend":2.922,"fitted":1.706,"actual":2.009,"residual":0.304},{"q":"2016-01","pmi":-0.316,"trade":-0.239,"ip":-0.045,"oil":-0.958,"fin":0.001,"ai":-0.236,"trend":2.922,"fitted":1.13,"actual":1.893,"residual":0.763},{"q":"2016-04","pmi":-0.188,"trade":-0.176,"ip":-0.035,"oil":-0.631,"fin":0.031,"ai":-0.323,"trend":2.922,"fitted":1.599,"actual":2.914,"residual":1.315},{"q":"2016-07","pmi":-0.008,"trade":-0.171,"ip":0.133,"oil":-0.083,"fin":0.064,"ai":-0.293,"trend":2.922,"fitted":2.565,"actual":2.676,"residual":0.111},{"q":"2016-10","pmi":0.089,"trade":0.036,"ip":0.274,"oil":0.291,"fin":-0.028,"ai":-0.292,"trend":2.922,"fitted":3.291,"actual":3.875,"residual":0.584},{"q":"2017-01","pmi":0.252,"trade":0.127,"ip":0.056,"oil":0.337,"fin":-0.09,"ai":-0.309,"trend":2.922,"fitted":3.296,"actual":3.902,"residual":0.606},{"q":"2017-04","pmi":0.523,"trade":0.766,"ip":0.617,"oil":0.362,"fin":-0.311,"ai":-0.27,"trend":2.922,"fitted":4.608,"actual":5.125,"residual":0.517},{"q":"2017-07","pmi":0.628,"trade":1.035,"ip":0.726,"oil":0.359,"fin":-0.423,"ai":-0.198,"trend":2.922,"fitted":5.049,"actual":5.309,"residual":0.26},{"q":"2017-10","pmi":0.706,"trade":0.649,"ip":0.543,"oil":0.324,"fin":-0.391,"ai":-0.246,"trend":2.922,"fitted":4.507,"actual":5.101,"residual":0.594},{"q":"2018-01","pmi":0.438,"trade":0.311,"ip":0.402,"oil":0.308,"fin":-0.477,"ai":-0.186,"trend":2.922,"fitted":3.718,"actual":4.596,"residual":0.878},{"q":"2018-04","pmi":0.645,"trade":-0.056,"ip":0.441,"oil":0.359,"fin":-0.296,"ai":-0.112,"trend":2.922,"fitted":3.902,"actual":4.688,"residual":0.785},{"q":"2018-07","pmi":0.34,"trade":0.14,"ip":0.228,"oil":0.374,"fin":-0.353,"ai":-0.034,"trend":2.922,"fitted":3.616,"actual":4.538,"residual":0.922},{"q":"2018-10","pmi":-0.013,"trade":0.041,"ip":0.197,"oil":0.295,"fin":-0.485,"ai":0.008,"trend":2.922,"fitted":2.963,"actual":2.741,"residual":-0.222},{"q":"2019-01","pmi":-0.023,"trade":-0.389,"ip":0.04,"oil":0.03,"fin":-0.281,"ai":-0.057,"trend":2.922,"fitted":2.242,"actual":3.294,"residual":1.052},{"q":"2019-04","pmi":-0.216,"trade":-0.218,"ip":-0.245,"oil":-0.29,"fin":-0.133,"ai":-0.121,"trend":2.922,"fitted":1.698,"actual":2.219,"residual":0.521},{"q":"2019-07","pmi":-0.302,"trade":-0.359,"ip":-0.294,"oil":-0.513,"fin":0.064,"ai":-0.083,"trend":2.922,"fitted":1.435,"actual":2.358,"residual":0.922},{"q":"2019-10","pmi":-0.164,"trade":-0.542,"ip":0.004,"oil":-0.688,"fin":0.075,"ai":-0.157,"trend":2.922,"fitted":1.448,"actual":2.613,"residual":1.164},{"q":"2020-01","pmi":-1.074,"trade":-0.597,"ip":-0.633,"oil":-0.942,"fin":-0.487,"ai":-0.161,"trend":2.922,"fitted":-0.972,"actual":-0.96,"residual":0.012},{"q":"2020-04","pmi":-4.093,"trade":-3.412,"ip":-3.429,"oil":-0.982,"fin":-0.207,"ai":-0.116,"trend":2.922,"fitted":-9.317,"actual":-10.821,"residual":-1.504},{"q":"2020-07","pmi":-0.468,"trade":-0.339,"ip":-0.086,"oil":-0.881,"fin":0.541,"ai":-0.069,"trend":2.922,"fitted":1.619,"actual":0.97,"residual":-0.649},{"q":"2020-10","pmi":0.665,"trade":0.826,"ip":0.468,"oil":-0.578,"fin":0.513,"ai":0.118,"trend":2.922,"fitted":4.934,"actual":4.807,"residual":-0.127},{"q":"2021-01","pmi":1.223,"trade":1.545,"ip":1.306,"oil":-0.108,"fin":0.593,"ai":0.344,"trend":2.922,"fitted":7.825,"actual":6.817,"residual":-1.008},{"q":"2021-04","pmi":1.417,"trade":1.083,"ip":1.518,"oil":0.438,"fin":0.49,"ai":0.401,"trend":2.922,"fitted":8.269,"actual":7.889,"residual":-0.38},{"q":"2021-07","pmi":1.135,"trade":1.062,"ip":0.994,"oil":0.533,"fin":0.642,"ai":0.421,"trend":2.922,"fitted":7.707,"actual":7.577,"residual":-0.13},{"q":"2021-10","pmi":0.598,"trade":0.222,"ip":0.561,"oil":0.519,"fin":0.463,"ai":0.277,"trend":2.922,"fitted":5.562,"actual":5.241,"residual":-0.32},{"q":"2022-01","pmi":0.42,"trade":0.18,"ip":0.357,"oil":0.432,"fin":0.109,"ai":0.019,"trend":2.922,"fitted":4.439,"actual":5.095,"residual":0.656},{"q":"2022-04","pmi":-0.115,"trade":-0.36,"ip":-0.422,"oil":0.447,"fin":-0.23,"ai":-0.354,"trend":2.922,"fitted":1.888,"actual":2.279,"residual":0.391},{"q":"2022-07","pmi":-0.502,"trade":0.0,"ip":-0.27,"oil":0.347,"fin":-0.596,"ai":-0.384,"trend":2.922,"fitted":1.516,"actual":1.497,"residual":-0.019},{"q":"2022-10","pmi":-0.509,"trade":-0.991,"ip":-0.276,"oil":0.084,"fin":-0.508,"ai":-0.36,"trend":2.922,"fitted":0.361,"actual":1.433,"residual":1.072},{"q":"2023-01","pmi":-0.199,"trade":-0.628,"ip":0.034,"oil":-0.316,"fin":-0.399,"ai":-0.324,"trend":2.922,"fitted":1.09,"actual":2.899,"residual":1.809},{"q":"2023-04","pmi":-0.176,"trade":-0.447,"ip":-0.114,"oil":-0.64,"fin":-0.243,"ai":-0.258,"trend":2.922,"fitted":1.043,"actual":2.27,"residual":1.227},{"q":"2023-07","pmi":-0.026,"trade":-0.231,"ip":-0.074,"oil":-0.561,"fin":-0.106,"ai":-0.123,"trend":2.922,"fitted":1.801,"actual":2.896,"residual":1.096},{"q":"2023-10","pmi":0.198,"trade":-0.342,"ip":0.101,"oil":-0.431,"fin":-0.05,"ai":0.073,"trend":2.922,"fitted":2.47,"actual":3.376,"residual":0.905},{"q":"2024-01","pmi":-0.048,"trade":0.287,"ip":0.23,"oil":-0.314,"fin":-0.108,"ai":0.321,"trend":2.922,"fitted":3.289,"actual":2.698,"residual":-0.591},{"q":"2024-04","pmi":0.157,"trade":0.473,"ip":0.625,"oil":-0.205,"fin":0.032,"ai":0.357,"trend":2.922,"fitted":4.361,"actual":3.478,"residual":-0.883},{"q":"2024-07","pmi":0.264,"trade":0.445,"ip":0.359,"oil":-0.204,"fin":0.184,"ai":0.469,"trend":2.922,"fitted":4.439,"actual":4.022,"residual":-0.417},{"q":"2024-10","pmi":0.306,"trade":0.162,"ip":0.342,"oil":-0.224,"fin":0.283,"ai":0.44,"trend":2.922,"fitted":4.229,"actual":3.283,"residual":-0.946},{"q":"2025-01","pmi":0.179,"trade":-0.128,"ip":-0.26,"oil":-0.182,"fin":0.265,"ai":0.412,"trend":2.922,"fitted":3.207,"actual":3.824,"residual":0.617},{"q":"2025-04","pmi":0.123,"trade":0.395,"ip":-0.034,"oil":-0.265,"fin":0.361,"ai":0.418,"trend":2.922,"fitted":3.919,"actual":3.877,"residual":-0.042},{"q":"2025-07","pmi":0.215,"trade":0.113,"ip":-0.049,"oil":-0.245,"fin":0.262,"ai":0.432,"trend":2.922,"fitted":3.649,"actual":3.211,"residual":-0.438},{"q":"2025-10","pmi":-0.43,"trade":-0.299,"ip":-0.899,"oil":-0.342,"fin":-0.042,"ai":0.451,"trend":2.922,"fitted":1.361,"actual":1.298,"residual":-0.062},{"q":"2026-01","pmi":-0.37,"trade":-0.118,"ip":-0.537,"oil":-0.117,"fin":-0.3,"ai":0.511,"trend":2.922,"fitted":1.991,"actual":2.118,"residual":0.128},{"q":"2026-04","pmi":-0.297,"trade":0.232,"ip":-0.385,"oil":0.332,"fin":-0.232,"ai":0.557,"trend":2.922,"fitted":3.129,"actual":1.847,"residual":-1.282},{"q":"2026-07","pmi":0.292,"trade":-0.026,"ip":0.236,"oil":0.378,"fin":0.283,"ai":0.595,"trend":2.922,"fitted":4.679,"actual":4.496,"residual":-0.183}];

// Core fields every export must have. Indicator keys are NOT required here
// on purpose — the panel can legitimately carry 6 or 10 (or any subset of
// the known) indicators, so driver keys are resolved dynamically below from
// whichever of them are actually present in the loaded data.
const REQUIRED_KEYS = ["q", "trend", "fitted", "actual", "residual"];

function validateDecompData(arr) {
  if (!Array.isArray(arr) || arr.length === 0) {
    throw new Error("File must contain a non-empty JSON array.");
  }
  const missing = REQUIRED_KEYS.filter((k) => !(k in arr[0]));
  if (missing.length) {
    throw new Error(`Missing expected field(s): ${missing.join(", ")}. Is this a decomposition_export.json from 05_export.py?`);
  }
  const hasIndicator = KNOWN_DRIVERS.some((d) => d.key in arr[0]);
  if (!hasIndicator) {
    throw new Error("No recognized indicator columns found (pmi, trade, ip, oil, fin, ai, copper, yield_curve, usd, credit).");
  }
  return arr;
}

// All indicators the pipeline can produce. A given export may only carry a
// subset (e.g. the bundled 6-indicator example predates copper/yield_curve/
// usd/credit) — components below filter this list down to whatever keys are
// actually present in the loaded data.
const KNOWN_DRIVERS = [
  { key: "pmi", label: "Composite PMI", color: "#45d6c0" },
  { key: "trade", label: "Trade volumes", color: "#5b8def" },
  { key: "ip", label: "Industrial production", color: "#9b7cf0" },
  { key: "oil", label: "Oil price shock", color: "#ef7a56" },
  { key: "fin", label: "Financial conditions", color: "#e3b24e" },
  { key: "ai", label: "AI / tech capex", color: "#6bd66b" },
  { key: "copper", label: "Copper (Dr. Copper)", color: "#c77b3c" },
  { key: "yield_curve", label: "Yield curve (10y-2y)", color: "#7f9cf5" },
  { key: "usd", label: "USD index (easing = +)", color: "#d65f9e" },
  { key: "credit", label: "Credit spreads (HY OAS)", color: "#8fd6e8" },
  { key: "vix", label: "VIX (calm = +)", color: "#e05252" },
  { key: "equity", label: "Equity market performance", color: "#3fa7d6" },
  { key: "bank_equity", label: "Bank equity index", color: "#8e6c88" },
  { key: "lending_standards", label: "Bank lending standards (easing = +)", color: "#f2a65a" },
  { key: "em_spread", label: "EM / sovereign spread proxy", color: "#4059ad" },
  { key: "housing", label: "Housing prices", color: "#b0413e" },
  { key: "leverage", label: "Leverage / credit-to-GDP", color: "#6b9080" },
  { key: "jobless_claims", label: "Jobless claims (falling = +)", color: "#f7b32b" },
  { key: "retail_sales", label: "Retail sales", color: "#5adbb5" },
  { key: "consumer_conf", label: "Consumer confidence", color: "#ee6f57" },
  { key: "business_conf", label: "Business confidence", color: "#849b96" },
  { key: "covid_stringency", label: "COVID stringency (less strict = +)", color: "#d4a5a5" },
  { key: "residual", label: "Idiosyncratic residual", color: "#4a5560" },
];

const MODEL_STATS = {
  varExplained: 0.544,
  r2: 0.93,
  alpha: 2.92,
  beta: 2.74,
  phi: 0.83,
};

function fmtPP(v) {
  const s = v >= 0 ? "+" : "";
  return `${s}${v.toFixed(2)}pp`;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  const rowDrivers = KNOWN_DRIVERS.filter((d) => d.key in row);
  return (
    <div className="ttip">
      <div className="ttip-q">{label}</div>
      {rowDrivers.map((d) => (
        <div className="ttip-row" key={d.key}>
          <span><i style={{ background: d.color }} />{d.label}</span>
          <span className="mono">{fmtPP(row[d.key])}</span>
        </div>
      ))}
      <div className="ttip-row ttip-total">
        <span>Trend</span>
        <span className="mono">{fmtPP(row.trend)}</span>
      </div>
      <div className="ttip-row ttip-total">
        <span>{row.actualKnown ? "Fitted / Actual" : "Forecast"}</span>
        <span className="mono">
          {row.fitted.toFixed(2)}% {row.actualKnown ? `/ ${row.actual.toFixed(2)}%` : ""}
        </span>
      </div>
    </div>
  );
}

export default function GlobalGDPNowcastDFM() {
  const [showMethod, setShowMethod] = useState(false);
  const [rawData, setRawData] = useState(DEFAULT_DECOMP_DATA);
  const [fileName, setFileName] = useState(null);
  const [loadError, setLoadError] = useState(null);

  // Newer exports (05_export.py) carry an explicit `is_forecast` flag on
  // every row, marking however many quarters were forecast ahead of the
  // last published GDP print (see model_lib.forecast_quarters()). Older
  // exports/the bundled example predate that field — for those, fall back
  // to the original behavior of treating just the last row as an
  // in-progress nowcast.
  const chartData = useMemo(() => {
    const hasForecastFlag = rawData.some((d) => "is_forecast" in d);
    if (hasForecastFlag) {
      return rawData.map((d) => ({ ...d, actualKnown: !d.is_forecast }));
    }
    return rawData.map((d, i) =>
      i === rawData.length - 1
        ? { ...d, actual: null, actualKnown: false, is_forecast: true }
        : { ...d, actualKnown: true, is_forecast: false }
    );
  }, [rawData]);

  const latest = chartData[chartData.length - 1];
  const firstForecastIdx = chartData.findIndex((d) => d.is_forecast);
  // The headline/driver breakdown should reflect the *nearest-term* nowcast
  // (the first forecast quarter), not whichever forecast quarter happens to
  // be last in the chart — those further out are extrapolated, not fresh.
  const nowcastRow = firstForecastIdx >= 0 ? chartData[firstForecastIdx] : latest;
  const forecastQuarterCount = chartData.filter((d) => d.is_forecast).length;
  // Which indicator keys this particular export actually carries (6- or
  // 10-indicator panel, or anything in between).
  const activeDrivers = useMemo(
    () => KNOWN_DRIVERS.filter((d) => d.key !== "residual" && d.key in nowcastRow),
    [nowcastRow]
  );
  const drivers = useMemo(
    () => activeDrivers.map((d) => ({ ...d, pp: nowcastRow[d.key] })).sort((a, b) => b.pp - a.pp),
    [activeDrivers, nowcastRow]
  );
  // Stacked-bar series: active indicators plus the residual bar (residual is
  // always present per REQUIRED_KEYS).
  const chartBars = useMemo(
    () => [...activeDrivers, KNOWN_DRIVERS.find((d) => d.key === "residual")],
    [activeDrivers]
  );

  function handleFile(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const parsed = JSON.parse(ev.target.result);
        validateDecompData(parsed);
        setRawData(parsed);
        setFileName(file.name);
        setLoadError(null);
      } catch (err) {
        setLoadError(err.message);
      }
    };
    reader.onerror = () => setLoadError("Could not read that file.");
    reader.readAsText(file);
    e.target.value = ""; // allow re-selecting the same file name later
  }

  function resetToExample() {
    setRawData(DEFAULT_DECOMP_DATA);
    setFileName(null);
    setLoadError(null);
  }

  return (
    <div className="ngdp-root">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        .ngdp-root {
          --bg: #0a0e10; --panel: #12181b; --panel2: #151d21;
          --border: rgba(255,255,255,0.09); --text: #ece8e0; --dim: #8fa0a8;
          --teal: #45d6c0; --coral: #ef7a56; --gold: #e3b24e;
          font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text);
          padding: 28px 22px 34px; border-radius: 14px; max-width: 1040px;
          margin: 0 auto; box-sizing: border-box;
        }
        .ngdp-root * { box-sizing: border-box; }
        .mono { font-family: 'IBM Plex Mono', monospace; }
        .serif { font-family: 'Source Serif 4', serif; }
        .eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: .16em;
          color: var(--dim); text-transform: uppercase; margin-bottom: 10px; }

        .headline-row { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 22px;
          border-bottom: 1px solid var(--border); padding-bottom: 22px; margin-bottom: 26px; }
        .headline-num { font-family: 'Source Serif 4', serif; font-size: 60px; font-weight: 600;
          line-height: 1; color: var(--gold); }
        .headline-sub { font-size: 13px; color: var(--dim); max-width: 400px; line-height: 1.55; }
        .headline-sub b { color: var(--text); font-weight: 600; }

        .stat-strip { display: flex; gap: 20px; margin-left: auto; flex-wrap: wrap; }
        .stat { text-align: right; }
        .stat .v { font-family: 'IBM Plex Mono', monospace; font-size: 17px; color: var(--text); }
        .stat .l { font-size: 10px; color: var(--dim); letter-spacing: .05em; text-transform: uppercase; margin-top: 2px; }

        .panel-title { font-family: 'Source Serif 4', serif; font-size: 17px; font-weight: 600; margin: 0 0 4px; }
        .panel-sub { font-size: 12px; color: var(--dim); margin-bottom: 16px; }

        .chart-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
          padding: 18px 14px 8px; }

        .ttip { background: #0d1215; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
          font-size: 11.5px; color: var(--text); min-width: 190px; }
        .ttip-q { font-family: 'IBM Plex Mono', monospace; color: var(--dim); margin-bottom: 6px; font-size: 11px; }
        .ttip-row { display: flex; justify-content: space-between; gap: 14px; padding: 1.5px 0; }
        .ttip-row span:first-child { display: flex; align-items: center; gap: 6px; }
        .ttip-row i { width: 7px; height: 7px; border-radius: 2px; display: inline-block; }
        .ttip-total { border-top: 1px solid var(--border); margin-top: 4px; padding-top: 4px; font-weight: 600; }

        .now-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin-top: 22px; }
        .now-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
        .now-card .nm { font-size: 12px; font-weight: 600; }
        .now-card .np { font-family: 'IBM Plex Mono', monospace; font-size: 15px; margin-top: 4px; }
        .pos { color: var(--teal); } .neg { color: var(--coral); }

        .method-toggle { margin-top: 24px; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
          letter-spacing: .05em; color: var(--dim); background: none; border: none; cursor: pointer;
          padding: 0; text-decoration: underline; text-underline-offset: 3px; }
        .method-body { margin-top: 12px; font-size: 12.5px; color: var(--dim); line-height: 1.65;
          border-left: 2px solid var(--border); padding-left: 14px; }
        .method-body b { color: var(--text); }
        .method-body table { border-collapse: collapse; margin-top: 10px; width: 100%; }
        .method-body td, .method-body th { font-family: 'IBM Plex Mono', monospace; font-size: 11px;
          padding: 4px 8px 4px 0; text-align: left; border-bottom: 1px solid var(--border); color: var(--dim); }
        .method-body th { color: var(--text); }
        .warn { background: rgba(239,122,86,0.09); border: 1px solid rgba(239,122,86,0.35);
          border-radius: 8px; padding: 10px 12px; margin-top: 10px; color: #f0a488; font-size: 12px; }

        .disclaimer { margin-top: 22px; font-size: 11px; color: #5f6d76; line-height: 1.6; }

        .loader-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
        .load-btn {
          position: relative; display: inline-flex; align-items: center; gap: 6px;
          background: var(--panel2); border: 1px solid var(--border); color: var(--text);
          font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: .03em;
          padding: 8px 14px; border-radius: 7px; cursor: pointer;
        }
        .load-btn:hover { border-color: rgba(255,255,255,0.3); }
        .load-btn input[type="file"] {
          position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
        }
        .load-status { font-family: 'IBM Plex Mono', monospace; font-size: 11px; }
        .load-status.ok { color: var(--teal); }
        .load-status.err { color: var(--coral); }
        .reset-btn {
          background: none; border: 1px solid var(--border); color: var(--dim);
          font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 6px 10px;
          border-radius: 7px; cursor: pointer;
        }
        .reset-btn:hover { color: var(--text); border-color: rgba(255,255,255,0.3); }

        @media (max-width: 640px) {
          .headline-num { font-size: 42px; }
          .stat-strip { margin-left: 0; }
        }
      `}</style>

      <div className="eyebrow">Global GDP Nowcast · Estimated Dynamic Factor Model</div>

      <div className="loader-row">
        <label className="load-btn">
          ↑ Load decomposition_export.json
          <input type="file" accept="application/json,.json" onChange={handleFile} />
        </label>
        {fileName && (
          <>
            <span className="load-status ok">loaded: {fileName} ({rawData.length} quarters)</span>
            <button className="reset-btn" onClick={resetToExample}>↺ reset to example</button>
          </>
        )}
        {loadError && <span className="load-status err">{loadError}</span>}
      </div>

      <div className="headline-row">
        <div>
          <div className="headline-num">{nowcastRow.fitted >= 0 ? "+" : ""}{nowcastRow.fitted.toFixed(1)}%</div>
          <div className="mono" style={{ fontSize: 11, color: "var(--dim)", marginTop: 4 }}>
            annualized · {nowcastRow.q} {nowcastRow.is_forecast ? "nowcast" : "actual"}
            {forecastQuarterCount > 1 && ` · ${forecastQuarterCount} quarters forecast ahead in the chart below`}
          </div>
        </div>
        <div className="headline-sub">
          Single-factor DFM (two-step Doz–Giannone–Reichlin): PCA-extracted common
          factor from 6 monthly indicators, Kalman-smoothed, bridged to GDP growth
          by <b>OLS: growth = {MODEL_STATS.alpha.toFixed(2)} + {MODEL_STATS.beta.toFixed(2)} × factor</b> (R² = {MODEL_STATS.r2.toFixed(2)}).
          {fileName && (
            <> <i>Stats at right are from the bundled example run — decomposition_export.json
            doesn't carry them; check your own model_params.txt for the real values.</i></>
          )}
        </div>
        <div className="stat-strip">
          <div className="stat"><div className="v">{(MODEL_STATS.varExplained * 100).toFixed(0)}%</div><div className="l">Var. explained</div></div>
          <div className="stat"><div className="v">{MODEL_STATS.r2.toFixed(2)}</div><div className="l">Bridge R²</div></div>
          <div className="stat"><div className="v">{MODEL_STATS.phi.toFixed(2)}</div><div className="l">Factor AR(1) φ</div></div>
        </div>
      </div>

      <div className="panel-title">Drivers of growth, by quarter</div>
      <div className="panel-sub">Stacked pp-contributions from the estimated factor loadings · {chartData[0].q} – {latest.q} · line = actual GDP growth</div>

      <div className="chart-card">
        <ResponsiveContainer width="100%" height={420}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 12, left: -6, bottom: 4 }} barCategoryGap={2}>
            <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.06)" vertical={false} />
            <XAxis dataKey="q" tick={{ fill: "#8fa0a8", fontSize: 10, fontFamily: "IBM Plex Mono" }}
              interval={2} axisLine={{ stroke: "rgba(255,255,255,0.12)" }} tickLine={false} />
            <YAxis tick={{ fill: "#8fa0a8", fontSize: 10, fontFamily: "IBM Plex Mono" }}
              axisLine={false} tickLine={false} width={38}
              label={{ value: "pp, annualized", angle: -90, position: "insideLeft", fill: "#5f6d76", fontSize: 10 }} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.25)" />
            {firstForecastIdx > 0 && (
              <ReferenceArea x1={chartData[firstForecastIdx].q} x2={latest.q}
                fill="#ffffff" fillOpacity={0.035} strokeOpacity={0} ifOverflow="extendDomain" />
            )}
            {firstForecastIdx > 0 && (
              <ReferenceLine x={chartData[firstForecastIdx].q} stroke="rgba(255,255,255,0.3)" strokeDasharray="3 3" />
            )}
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
            {chartBars.map((d) => (
              <Bar key={d.key} dataKey={d.key} stackId="s" fill={d.color} isAnimationActive={false}>
                {chartData.map((row, i) => (
                  <Cell key={i} fillOpacity={row.is_forecast ? 0.4 : 1} />
                ))}
              </Bar>
            ))}
            <Line type="monotone" dataKey="actual" stroke="#ece8e0" strokeWidth={1.6} dot={false}
              connectNulls={false} isAnimationActive={false} />
            <Legend
              wrapperStyle={{ fontSize: 11, fontFamily: "Inter", color: "#8fa0a8", paddingTop: 10 }}
              formatter={(v) => {
                const d = KNOWN_DRIVERS.find((x) => x.key === v);
                return d ? d.label : v === "actual" ? "Actual GDP growth" : v;
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="panel-title" style={{ marginTop: 28 }}>This quarter's breakdown</div>
      <div className="panel-sub">{nowcastRow.q}{nowcastRow.is_forecast ? " (nowcast)" : ""}, ranked by contribution size</div>
      <div className="now-grid">
        {drivers.map((d) => (
          <div className="now-card" key={d.key}>
            <div className="nm" style={{ color: d.color }}>{d.label}</div>
            <div className={`np ${d.pp >= 0 ? "pos" : "neg"}`}>{fmtPP(d.pp)}</div>
          </div>
        ))}
        <div className="now-card">
          <div className="nm" style={{ color: "var(--dim)" }}>Trend</div>
          <div className="np mono">{fmtPP(nowcastRow.trend)}</div>
        </div>
      </div>

      <button className="method-toggle" onClick={() => setShowMethod((s) => !s)}>
        {showMethod ? "− HIDE METHODOLOGY & CAVEATS" : "+ SHOW METHODOLOGY & CAVEATS"}
      </button>

      {showMethod && (
        <div className="method-body">
          <p><b>Estimation.</b> Monthly series — Composite PMI, world trade volume growth,
          world industrial production growth, an oil-price shock term, a financial-conditions
          score, an AI/tech-capex score, and (as of the latest panel) copper price growth, the
          10y-2y yield curve slope, a broad USD index, and high-yield credit spreads — are
          standardized and reduced to a single common factor via PCA — the "static" step of the
          Doz–Giannone–Reichlin (2011) two-step estimator. An AR(1) is fit to that factor
          (φ = {MODEL_STATS.phi.toFixed(2)}), and a Kalman filter/smoother is run over the
          resulting state-space to produce a dynamically-consistent monthly factor. A bridge
          regression of quarterly GDP growth on the quarterly-averaged factor gives the growth
          units. Because the PCA factor is an <i>exact</i> linear combination of the standardized
          indicators, every quarter's fitted growth decomposes additively into the bars shown for
          whichever indicators are in the loaded export — no approximation error.</p>

          <p style={{ marginTop: 10 }}>The table below reflects the <b>bundled 6-indicator
          example run</b> (loadings will shift once you load a fresh export built from the
          wider panel — rerun <code className="mono">py/01_build_panel_real.py</code>
          through <code className="mono">py/05_export.py</code> to get real weights for
          copper/yield_curve/usd/credit and the newer indicators).</p>

          <table>
            <thead><tr><th>Indicator</th><th>Loading (w)</th><th>Reads as</th></tr></thead>
            <tbody>
              <tr><td>Composite PMI</td><td>+0.297</td><td>largest single weight — services + manufacturing momentum</td></tr>
              <tr><td>Trade volumes</td><td>+0.287</td><td>external demand</td></tr>
              <tr><td>Industrial production</td><td>+0.284</td><td>goods-side output</td></tr>
              <tr><td>Oil price</td><td>+0.189</td><td>historically procyclical (demand-driven) on average</td></tr>
              <tr><td>Financial conditions</td><td>+0.109</td><td>policy/credit backdrop</td></tr>
              <tr><td>AI / tech capex</td><td>+0.086</td><td>smallest weight — shorter, more idiosyncratic history</td></tr>
            </tbody>
          </table>

          <div className="warn">
            <b>Two honest limitations worth flagging:</b><br/>
            1) <b>No live data access in this environment.</b> This sandbox has no network
            connection, so the ~250-month panel above is not pulled vintage-by-vintage from
            Haver Analytics (with FRED/CPB/OECD as fallback sources). It's constructed to match documented history — real
            recession dates, the actual JPMorgan PMI troughs (≈33 in Nov-2008, 26.2 in
            Apr-2020), actual IMF annual growth figures, and the real 2026 Brent move — but
            between those anchors the paths are interpolated, not observed. The estimation
            code (PCA, AR(1), Kalman filter/smoother, bridge regression) is genuine and would
            run unchanged on real pulled data.<br/><br/>
            2) <b>Oil's positive loading is a known limitation, and more factors don't fix it.</b>
            Historically oil prices are procyclical on average (demand pulls both up together), so
            the model learned a positive weight. A genuine <i>supply</i> shock (e.g. a Strait of
            Hormuz disruption) isn't something this can distinguish from a demand-driven rise — it
            will show up as a modeled tailwind when the real-world channel (cost, uncertainty) is
            arguably a drag. This model now uses 3 static factors (ridge-regularized — see the
            header comment above for why that combination, and only that combination, actually
            improved walk-forward accuracy), but that doesn't resolve this specific issue: plain
            PCA factors aren't constrained to align with economically meaningful shock types
            regardless of how many are used, ridge-regularized or not. Genuinely fixing this needs a structural
            (sign-restricted) model, not just more factors.
          </div>

          <p style={{ marginTop: 10 }}>To put this on real data: run <code className="mono">py/01_build_panel_real.py</code>
          on a machine with internet access — it now pulls 22 indicators total, including
          copper, the 10y-2y yield curve, a broad USD index, and high-yield credit spreads
          (via Haver Analytics), plus VIX/equities/bank-lending-standards/housing/leverage/confidence
          surveys/jobless-claims/COVID-stringency — then rerun
          <code className="mono"> py/02_estimate_dfm.py</code>, <code className="mono">py/03_estimate_ml.py</code>,
          <code className="mono"> py/04_backtest.py</code>, and <code className="mono">py/05_export.py</code>
          unchanged and load the fresh <code className="mono">json/decomposition_export.json</code> above.</p>
        </div>
      )}

      <div className="disclaimer">
        Illustrative model, not an official or investment forecast. Estimation code and full
        panel-construction script are provided alongside this file. Update the panel with real
        pulled data to turn this into a production nowcast.
      </div>
    </div>
  );
}
