# Eurostat

Client: `scripts/eurostat.py`:
- `get_dataset(dataset_code, filters=None, start_period=None) -> pd.DataFrame` — raw pull
- `to_series(df, ...) -> pd.Series`
- `get_series(dataset_code, filters, ...) -> pd.Series` — one-shot convenience

No API key needed. Base:
`https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data`.
New to this repo — no prior Eurostat code existed to generalize from, so
this was written from Eurostat's documented API rather than adapted from
working code (see caveat below).

## Dataset codes and filters

Every Eurostat dataset has a short code (e.g. `namq_10_gdp` for quarterly
national accounts GDP, `prc_hicp_manr` for HICP inflation,
`une_rt_m` for monthly unemployment). Find a dataset's code and its exact
dimension names/values via the Eurostat data browser
(https://ec.europa.eu/eurostat/databrowser) — the code is in the URL, and
the browser's "download"/API panel shows the exact filter dimension names.

Common dimensions: `geo` (country/area code, e.g. `DE`, `EA20` for euro
area, `EU27_2020`), `freq` (`A`/`Q`/`M`), `unit`, plus dataset-specific
dimensions like `na_item` (national accounts item, e.g. `B1GQ` for GDP)
or `s_adj` (seasonal adjustment).

Filter as narrowly as possible — an unfiltered dataset pull can be huge.

## Example

```python
eurostat.get_series(
    "namq_10_gdp",
    {"geo": "EA20", "na_item": "B1GQ", "unit": "CLV_PCH_PRE", "s_adj": "SCA"},
    name="ea_gdp_growth",
)
```

Euro area real GDP, quarter-on-quarter % change, seasonally adjusted.

## Gotchas / caveats

- This client requests `format=SDMX-CSV` for parsing consistency with the
  other SDMX-based clients in this skill (`imf.py`, `oecd.py`) — Eurostat's
  default/native format is JSON-stat, which is more compact for sparse
  multi-dimensional pulls but needs different (more involved) parsing;
  not implemented here.
- **This sandbox's network policy blocks `ec.europa.eu` entirely**, so
  this client has never made a real request — it's built from documented
  API behavior and unit-tested only against a hand-built mock CSV.
  Verify the dataset code, filter dimension names, and response shape
  against the real API before relying on this for anything.
