# OECD (SDMX REST API)

Client: `scripts/oecd.py`:
- `get_sdmx_data(dataflow, key="all", start_period=None) -> pd.DataFrame` — raw pull, explore first
- `to_series(df, ...) -> pd.Series` — narrow a filtered slice down to one series
- `composite_leading_indicator(area="G20", start_period=None)` — worked recipe

No API key needed. Base: `https://sdmx.oecd.org/public/rest/data`.

## Dataflow IDs and keys

OECD dataflow IDs look like `OECD.SDD.STES,DSD_STES@DF_CLI,4.1`
(`agency.division,DSD@dataflow,version`). SDMX keys are dot-separated
dimension filters specific to each dataflow's dimension order, e.g.
`.M.LI...AA.IX..H` for the Composite Leading Indicator dataflow (monthly,
leading indicator, amplitude-adjusted index, ...). Use `"all"` (or `""`)
to skip filtering and get every series in the dataflow — fine for
small/medium dataflows, but the CLI dataflow filtered by area is much
smaller and faster.

Find the right dataflow ID and key for a dataset via the OECD Data
Explorer (https://data-explorer.oecd.org) — build the query there in the
UI, then copy the "API request" URL it generates, which shows both.

## Known dataflows

| Dataflow | What it is |
|---|---|
| `OECD.SDD.STES,DSD_STES@DF_CLI,4.1` | Composite Leading Indicators |
| `OECD.SDD.STES,DSD_STES@DF_MEI,...` | Main Economic Indicators (retail sales, confidence surveys, etc. — this repo's Haver-sourced `retail_sales`/`consumer_conf`/`business_conf` indicators pull the Haver mirror of this) |

## Gotchas

- `REF_AREA` values are area codes like `G20`, `G7`, `OECD`, or individual
  country ISO3 codes — filter the raw DataFrame on this column if the key
  alone doesn't narrow it down (see `composite_leading_indicator()`'s
  fallback logic in `scripts/oecd.py`).
- Not live-testable in this sandbox (network policy blocks
  `sdmx.oecd.org`) — verified only against a hand-built mock CSV. Smoke-test
  before relying on this.
