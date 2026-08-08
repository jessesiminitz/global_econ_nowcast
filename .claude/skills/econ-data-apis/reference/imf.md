# IMF (SDMX 3.0 REST API)

Client: `scripts/imf.py`:
- `get_sdmx_data(dataflow, key="*", database_id="IMF.STA", start_period=None) -> pd.DataFrame` — raw pull, explore first
- `to_series(df, ...) -> pd.Series` — narrow a filtered slice down to one series
- `world_real_gdp_growth_saar()` — worked recipe (see below)

No API key needed. Base: `https://api.imf.org/external/sdmx/3.0/data/dataflow`.

## Working with an unfamiliar dataflow

IMF SDMX responses are wide and inconsistent across dataflows:
- The area dimension column is sometimes `REF_AREA`, sometimes bundled with
  `COUNTRY`/`REGION` — search for any column containing those substrings
  rather than assuming one name.
- Dataflows often carry multiple `TYPE_OF_TRANSFORMATION` variants for the
  same indicator (e.g. a level/index series AND a period-over-period %
  change series in the same pull) — filter on this column
  (`POP_PCH` = period-over-period % change is common) or you'll silently
  mix levels and growth rates together.
- Time periods come as `"2024-Q1"`-style strings for quarterly data —
  `to_series()` / `_util.parse_period()` handle this.

Workflow for a new dataflow: call `get_sdmx_data(dataflow_id)` with a broad
key (or `"*"`), inspect `df.columns` and `df[<area_col>].unique()` /
`df[<indicator_col>].unique()` to find the row(s) you want, then filter and
call `to_series()`.

## Known dataflows

| Dataflow | What it is |
|---|---|
| `QGDP_WCA` | Quarterly GDP, World and Country Aggregates — real, SA, QoQ |
| `IFS` | International Financial Statistics — huge, broad macro/financial catalog |
| `WEO` | World Economic Outlook — annual, forward-looking projections included |

Discover more dataflows via the IMF Data API's dataflow catalog endpoint
(`GET /external/sdmx/3.0/structure/dataflow/IMF.STA` — not wrapped here)
or https://data.imf.org.

## `world_real_gdp_growth_saar()` recipe

Reimplements exactly what this repo's `py/01_build_panel_real.py`
(`fetch_imf_quarterly_gdp()`) does: finds the World-area row(s) in
`QGDP_WCA`, restricts to the `POP_PCH` transformation, and compounds the
quarter-on-quarter growth rate to an annualized rate. Read the function
body in `scripts/imf.py` as a template for writing similar recipes against
other dataflows.

## Gotchas / caveats

Not live-testable in this sandbox (network policy blocks `api.imf.org`) —
verified only against a hand-built mock CSV matching the documented
response shape. Smoke-test against the real API before relying on this,
especially `world_real_gdp_growth_saar()`'s indicator-selection logic,
which assumes indicator code `B1GQ_S1_Q` stays stable.
