# World Bank API

Client: `scripts/worldbank.py` — `get_indicator(country, indicator, start=None, end=None) -> pd.Series`

No API key needed. Base: `https://api.worldbank.org/v2`. Data is annual.

## Country/aggregate codes

ISO3 country codes (`USA`, `CHN`, `DEU`, ...) or World Bank aggregate codes:

| Code | What it is |
|---|---|
| `WLD` | World |
| `EUU` | European Union |
| `OED` | OECD members |
| `HIC` / `MIC` / `LIC` | High/middle/low income countries |

Full list: https://api.worldbank.org/v2/country?format=json&per_page=400

## Common indicator codes

| Indicator | What it is |
|---|---|
| `NY.GDP.MKTP.KD.ZG` | Real GDP growth, annual % |
| `NY.GDP.MKTP.CD` | GDP, current US$ |
| `FP.CPI.TOTL.ZG` | Inflation, consumer prices, annual % |
| `SL.UEM.TOTL.ZS` | Unemployment, % of labor force |
| `NE.TRD.GNFS.ZS` | Trade (% of GDP) |
| `SP.POP.TOTL` | Total population |

Search the full indicator catalog at
https://data.worldbank.org/indicator or
`https://api.worldbank.org/v2/indicator?format=json&per_page=20000`
(large — not wrapped here).

## Gotchas

- Response is a two-element JSON array: `[metadata, data]` — `get_indicator()`
  already unwraps this and raises a clear error if `data` is empty or the
  country/indicator combination doesn't exist.
- Missing years come through as `null` values in the data list — dropped
  automatically.
- Data is annual only; there's no monthly/quarterly World Bank indicator
  API. For sub-annual global data, use IMF or OECD instead.
