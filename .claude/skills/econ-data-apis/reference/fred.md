# FRED (Federal Reserve Economic Data)

Client: `scripts/fred.py` — `get_series(series_id, start=None, end=None) -> pd.Series`

## Auth

- `FRED_API_KEY` env var, optional but recommended. Free key:
  https://fred.stlouisfed.org/docs/api/api_key.html
- With a key: official `api.stlouisfed.org/fred/series/observations`
  endpoint, JSON.
- Without a key: falls back to scraping `fredgraph.csv` (the same CSV the
  interactive FRED chart downloads) — works for any public series but is
  more prone to aggressive rate limiting and timeouts. Proven unreliable
  enough at scale that this repo's own pipeline treats it as a last-resort
  fallback, not the default.

## Finding a series ID

Every FRED series has a short, stable ID shown on its page URL, e.g.
`fred.stlouisfed.org/series/GDP` → `GDP`. Common ones already used
elsewhere in this repo (see `py/01_build_panel_real.py` for the full
context each was chosen for):

| Series ID | What it is |
|---|---|
| `GDP` | US nominal GDP |
| `GDPC1` | US real GDP |
| `UNRATE` | US unemployment rate |
| `CPIAUCSL` | US CPI, all items |
| `T10Y2Y` | 10y-2y Treasury yield spread |
| `DTWEXBGS` | Broad trade-weighted USD index |
| `NFCI` | Chicago Fed National Financial Conditions Index |
| `ICSA` | Initial jobless claims |
| `BAMLH0A0HYM2` | ICE BofA US High Yield OAS |
| `DCOILBRENTEU` | Brent crude spot price |

Search the full catalog at https://fred.stlouisfed.org/tags/series or via
FRED's `series/search` endpoint (not wrapped here — add if you need
programmatic search rather than a known ID).

## Gotchas

- FRED represents missing observations as the literal string `"."`, not a
  blank — `get_series()` already coerces this to `NaN` and drops it via
  `pd.to_numeric(errors="coerce")` + `dropna()`.
- Some series (e.g. anything seasonally-adjusted-vs-not) have near-duplicate
  IDs — double check you have the right variant (`_SA`, `NSA`, etc. suffixes
  or prefixes vary by series family).
