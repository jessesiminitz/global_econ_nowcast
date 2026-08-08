---
name: econ-data-apis
description: Pull economic, macro, and financial time-series data — GDP, inflation, unemployment, interest rates, trade, PMI, and similar indicators — from FRED, Haver Analytics, IMF, World Bank, OECD, Eurostat, ECB SDW, BEA, or BLS. Use whenever the user asks to fetch/pull/download an economic data series, build an indicator panel, or look up a specific FRED/IMF/World Bank/OECD/Eurostat/Haver/BEA/BLS/ECB series.
---

# Economic data APIs

Working, tested clients for 9 economic-data sources, each returning a plain
`pandas.Series` indexed by date. Every fetch is auto-cached to a local
DuckDB file so a series pulled once is queryable again without hitting the
network. Self-contained — no dependency on anything else in this repo — so
this whole folder is copy-paste portable into any other project (see
"Using this in other repos" below).

## Quickstart

```python
import sys
sys.path.insert(0, ".claude/skills/econ-data-apis/scripts")  # adjust to wherever this folder lives

import fred, worldbank, imf, oecd, eurostat, haver

fred.get_series("GDP", start="2010-01-01")                 # US real GDP
worldbank.get_indicator("WLD", "NY.GDP.MKTP.KD.ZG")          # world GDP growth %, annual
imf.world_real_gdp_growth_saar()                             # IMF world GDP growth, quarterly SAAR
oecd.composite_leading_indicator(area="G20")                  # OECD CLI
eurostat.get_series("namq_10_gdp", {"geo": "EA20", "na_item": "B1GQ", "unit": "CLV_PCH_PRE", "s_adj": "SCA"})
haver.get_series("USECON", "GDPH")                             # needs HAVER_API_KEY
```

Or from the terminal, which also auto-caches the result:

```
python3 .claude/skills/econ-data-apis/scripts/fetch_cli.py fred GDP --start 2010-01-01
python3 .claude/skills/econ-data-apis/scripts/fetch_cli.py worldbank WLD NY.GDP.MKTP.KD.ZG
python3 .claude/skills/econ-data-apis/scripts/fetch_cli.py --query fred GDP    # read the cache, no network
python3 .claude/skills/econ-data-apis/scripts/fetch_cli.py --list-cached
```

## Sources at a glance

| Source | Module | Needs key? | Env var | Notes |
|---|---|---|---|---|
| FRED | `fred.py` | Optional | `FRED_API_KEY` | Keyless fallback scrapes `fredgraph.csv`, more rate-limited |
| Haver Analytics | `haver.py` | **Required** | `HAVER_API_KEY` | Paid subscription; premium, high-quality real data |
| IMF | `imf.py` | No | — | SDMX 3.0 REST API |
| World Bank | `worldbank.py` | No | — | Annual indicators, any country/aggregate |
| OECD | `oecd.py` | No | — | SDMX REST API |
| Eurostat | `eurostat.py` | No | — | SDMX-CSV; new, see caveat below |
| ECB (SDW) | `ecb.py` | No | — | New, see caveat below |
| BEA | `bea.py` | **Required** | `BEA_API_KEY` | Free signup; US national accounts detail |
| BLS | `bls.py` | Optional | `BLS_API_KEY` | Keyless works, key raises rate limit/history depth |

`FRED_API_KEY` and `HAVER_API_KEY` are read straight from your shell
environment if already exported (as they are in this project's setup) —
no extra config needed. See `.env.example` (copy to `.env` next to the
scripts) for the rest.

Full per-source details, example series/dataflow IDs, and gotchas: see
`reference/*.md`. Start with `reference/choosing-a-source.md` if you're not
sure which source fits what you're looking for.

## Caching (DuckDB)

`scripts/db.py` provides `save_series()` / `load_series()` / `list_cached()`
against a local `econ_data.duckdb` file (defaults to the current working
directory; override with `ECON_DATA_DB_PATH`). Every source's data lands in
one long-format table (`series_observations`), so it's all queryable
together regardless of which API it came from. `fetch_cli.py` caches by
default; call `db.save_series()` yourself from Python if you're not using
the CLI. See `reference/caching.md` for the schema and example SQL.

The cache is a convenience layer, not a source of truth — re-fetch if a
series gets revised upstream.

## Using this in other repos

This build can only land inside `global_econ_nowcast` (this session can't
write to your local machine), but the whole point of the design is that
`scripts/` and `reference/` don't reference anything else in this repo — so
to make it available in **every** project, copy or symlink this folder to
Claude Code's personal skill location on your machine:

```
cp -r .claude/skills/econ-data-apis ~/.claude/skills/econ-data-apis
# or, to keep it updated automatically when this repo changes:
ln -s "$(pwd)/.claude/skills/econ-data-apis" ~/.claude/skills/econ-data-apis
```

Once there, Claude Code loads it in any project, not just this one.

## Relationship to this repo's own pipeline

`py/01_build_panel_real.py` (this repo's real nowcasting pipeline) has its
own, older, indicator-specific fetchers with a Haver-primary/free-fallback
chain — untouched by this skill. It's a good worked example of chaining
several of these sources together for one production pipeline; this skill
is the general-purpose, parameterized version for one-off or ad hoc pulls
of *any* series, from a fresh Claude Code session in any project.

## A note on live testing

Every client here was built by generalizing (FRED, Haver, IMF, World Bank,
OECD) or newly writing (Eurostat, ECB, BEA, BLS) from documented API
shapes, and unit-tested against realistic mocked responses — but this
sandbox's network policy blocks every one of these API hosts, so none of
it has been exercised against the *live* APIs. Smoke-test each source you
actually plan to use once on a machine with real internet access before
relying on it for anything important; `reference/*.md` flags this
per-source where relevant.
