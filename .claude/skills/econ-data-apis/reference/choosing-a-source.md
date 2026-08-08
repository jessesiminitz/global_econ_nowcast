# Choosing a source

Rough decision guide, informed by what this repo's own pipeline
(`py/01_build_panel_real.py`) already had to work out empirically.

| Need | Best source | Why |
|---|---|---|
| Any specific US series (GDP, CPI, unemployment, rates, ...) | **FRED** | Enormous catalog, stable series IDs, well-documented |
| A licensed/premium series (real S&P Global PMI, real EMBI, Bloomberg HY OAS, ...) | **Haver** | Aggregates vendor-licensed data FRED/free sources don't carry |
| Any country's annual macro indicator (GDP growth, inflation, poverty, etc.) | **World Bank** | Broadest country/aggregate coverage, keyless, simple API |
| Genuinely quarterly, cross-country GDP/national-accounts data | **IMF** (SDMX) | IMF's own quarterly national accounts, not an annual figure interpolated down |
| OECD-member-specific indicators (leading indicators, confidence surveys, retail sales) | **OECD** (SDMX) | OECD's own MEI/CLI datasets, keyless |
| EU/euro-area specific data (national accounts, HICP, unemployment by member state) | **Eurostat** | The authoritative EU statistical source; World Bank/OECD often lag or omit EU-specific detail |
| Euro area monetary/financial data (exchange rates, interest rates, bank lending) | **ECB SDW** | ECB's own statistical warehouse, keyless |
| Detailed US national accounts (NIPA tables, GDP by industry, regional data) | **BEA** | More granular than FRED's own mirrors of the same BEA data |
| Detailed US labor-market series (CPI by item/area, employment by industry, wage data) | **BLS** | The source FRED itself mirrors many labor series from — more series, before FRED picks them up |

General patterns:
- **Keyless (no signup needed at all)**: World Bank, IMF, OECD, Eurostat,
  ECB SDW.
- **Free key required**: FRED (optional, but recommended — see
  `reference/fred.md`), BEA (required), BLS (optional, raises limits).
- **Paid subscription required**: Haver Analytics only.
- **Frequency**: FRED/Haver/BLS/BEA are typically the highest-frequency
  (monthly/weekly/daily) sources for US data; World Bank is almost always
  annual; IMF/OECD/Eurostat/ECB vary by dataset (quarterly national
  accounts, monthly surveys, daily financial data all exist).
- When a series exists in more than one place (e.g. US CPI is in both FRED
  and BLS), prefer FRED for convenience (stable series IDs, one simple
  call) unless you need detail FRED doesn't mirror.
