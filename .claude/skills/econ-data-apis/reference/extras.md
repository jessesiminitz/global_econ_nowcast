# Extras: BEA, BLS, ECB SDW

Less deeply documented than the 6 primary sources — working clients, but
newer/thinner reference material.

## BEA (Bureau of Economic Analysis)

Client: `scripts/bea.py` — `get_data(dataset_name, **params) -> pd.DataFrame`,
`to_series(df, line_description=None, name="value") -> pd.Series`.

- **Requires** `BEA_API_KEY` — free signup at
  https://apps.bea.gov/API/signup/. No keyless tier.
- Base: `https://apps.bea.gov/api/data`.
- Most useful dataset: `NIPA` (National Income and Product Accounts).
  Common `TableName` values: `T10101` (real GDP and components, quarterly),
  `T20100` (personal income). Pass `Frequency="Q"` or `"A"` and
  `Year="ALL"` (or a specific year/range) alongside `TableName`.
- `to_series()` filters a NIPA pull down to one line via its
  `LineDescription` text (e.g. `"Gross domestic product"`) — inspect
  `df["LineDescription"].unique()` on a raw `get_data()` pull first if
  you're not sure of the exact wording for the line you want.
- Not live-testable in this sandbox (network policy blocks
  `apps.bea.gov`) — smoke-test with a real key before relying on this.

## BLS (Bureau of Labor Statistics)

Client: `scripts/bls.py` — `get_series(series_id, start_year=None, end_year=None) -> pd.Series`.

- `BLS_API_KEY` optional — works keyless (v2 API, lower rate limit and a
  shorter max history window per request); a free key
  (https://data.bls.gov/registrationEngine/) raises both.
- Base: `https://api.bls.gov/publicAPI/v2/timeseries/data/`, POST request
  with a JSON body (not a simple GET).
- Series IDs are BLS's own codes, e.g. `CUUR0000SA0` (CPI-U, all items,
  US city average, not seasonally adjusted), `LNS14000000` (unemployment
  rate, seasonally adjusted). Look up IDs via https://data.bls.gov/cgi-bin/surveymost
  or the series-ID construction guide for the relevant BLS program.
- **Max ~20 years of history per request** — for a longer span, call
  `get_series()` multiple times with different `start_year`/`end_year`
  windows and `pd.concat()` the results.
- Annual-average data points (`period == "M13"`) are filtered out
  automatically — only real monthly observations are returned.
- Not live-testable in this sandbox (network policy blocks `api.bls.gov`).

## ECB SDW (Statistical Data Warehouse)

Client: `scripts/ecb.py` — `get_series(flow_ref, key, start_period=None, end_period=None) -> pd.Series`.

- No API key needed. Base: `https://data-api.ecb.europa.eu/service/data`
  (the newer ECB Data Portal API).
- `flow_ref` is a dataflow ID (`EXR` for exchange rates, `FM` for
  financial market data, `ICP` for HICP inflation, `MIR` for bank interest
  rates); `key` is a dot-separated SDMX series key within that flow, e.g.
  `D.USD.EUR.SP00.A` (daily USD/EUR spot reference rate under `EXR`).
  Find both via the ECB Data Portal browser (https://data.ecb.europa.eu).
- Not live-testable in this sandbox (network policy blocks
  `*.europa.eu`, which includes `data-api.ecb.europa.eu`) — verified only
  against a hand-built mock CSV. Smoke-test before relying on this.
