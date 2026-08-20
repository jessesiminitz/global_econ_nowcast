# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that nowcasts world real GDP growth (PPP-weighted, annualized q/q) from a ~20-indicator
monthly panel (trade, industrial production, PMI, financial conditions, labor market, housing, etc.),
pulled live from Haver Analytics with FRED/yfinance/OECD/IMF/World Bank as fallbacks. Two competing
models — a ridge-regularized dynamic factor model and an elastic-net regression — are backtested
against history on every run; whichever wins is exported to the dashboard.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install requests pandas numpy scipy scikit-learn openpyxl yfinance matplotlib pytest

python3 py/01_build_panel_real.py   # pull live data -> csv/panel_monthly.csv, csv/gdp_quarterly.csv
python3 py/02_estimate_dfm.py       # DFM baseline -> csv/quarterly_decomposition.csv
python3 py/03_estimate_ml.py        # elastic-net challenger -> csv/quarterly_decomposition_ml.csv
python3 py/04_backtest.py           # walk-forward comparison, picks the winner -> json/backtest_summary.json
python3 py/05_export.py             # charts + json/decomposition_export.json, winning model only

pytest tests/                              # full suite
pytest tests/test_model_lib.py -k fit_dfm  # single test / test group
```

Every script resolves `csv/`, `json/`, `png/`, `txt/` relative to its own location (`py/paths.py`), so
run from the repo root or from inside `py/` — either works.

## Data refresh: API keys (read before running `01_build_panel_real.py` non-interactively)

`HAVER_API_KEY` and `FRED_API_KEY` live in this user's `~/.zshrc`, which zsh only sources for
**interactive** shells — invisible to a subprocess, agent session, or any non-interactive invocation.
Without them, every fetcher silently falls back to the free/unauthenticated FRED endpoint, which is
aggressively rate-limited — expect most indicators to drop out of the panel, not just degrade.

Before an automated or agent-driven refresh, copy both keys into this repo's own git-ignored `.env`
(`py/haver_client.py` loads it automatically at import time; `01_build_panel_real.py`'s `FRED_API_KEY`
lookup then picks it up too, since it runs after that import):

```bash
grep '^FRED_API_KEY=\|^HAVER_API_KEY=' ~/.config/econ-data/.env > .env
```

That path is the user's Economic Data API skill's own shell-independent key location (see
`~/.claude/skills/economic-data-api-skill-lkvl80/SKILL.md`) — it documents this exact trap and is the
preferred source of truth for these keys over asking the user to retype them.

## Architecture

- `py/model_lib.py` is the shared core (`fit_dfm`, `fit_elastic_net`, `forecast_quarters`) that both
  models and the backtest call — changes here affect every downstream script. `tests/` covers this file
  specifically against a small synthetic fixture (`tests/conftest.py`), not real pulled data, which
  drifts on every re-pull and would make the suite flaky.
- Both models fit independently (`02_estimate_dfm.py` / `03_estimate_ml.py`) but never pick a winner
  themselves — `04_backtest.py` does that via walk-forward out-of-sample error, split into **calm** vs
  **stressed** (GFC/COVID) quarters. The winner is whichever minimizes the *worse* of the two regimes,
  not the pooled average, so a model can't win by being fine in the much-larger calm-quarter set while
  quietly failing in a crisis. Both models' numbers always land in `json/backtest_summary.json`
  regardless of which wins.
- `05_export.py` reads the declared winner and exports only that model's decomposition — every
  downstream chart/JSON is single-model; nothing mixes DFM and elastic-net output in one view except
  `png/validation_fit.png`.
- A broken data source degrades gracefully rather than blocking the run: each fetcher in
  `01_build_panel_real.py` drops its own column with a printed warning on failure, so the indicator
  count (and therefore which model wins) can genuinely vary run to run — check the console output
  before trusting a fresh pull's results, and treat "the winner flipped" as a real possible outcome,
  not a bug.
- `01_build_panel_real.py` has an `EXCLUDED_INDICATORS` set near the top of the file for deliberately
  setting indicators aside without deleting their fetchers — currently used to run a **global-only**
  panel by excluding six US-only proxy series (`lending_standards`, `housing`, `leverage`,
  `jobless_claims`, `bank_equity`, `fin`). Each fetcher stays fully defined; removing a name from the
  set brings that indicator straight back into the next run. `credit` (US HY OAS) is separately absent
  from the panel for a data reason, not this toggle — its Haver history only starts in 2012, short
  enough that the panel-truncation check in `main()` drops it automatically.

## Dashboards

`jsx/global_gdp_nowcast_dfm.jsx` is a React/Recharts component checked into this repo — it renders
whichever indicator keys are present in a loaded `decomposition_export.json`. The dashboard actually
in current use is a separate, self-contained HTML build (assessment paragraph, two tabs — a data
Dashboard and a one-page non-technical Memo in Treasury-memo voice — dual-scale charts, per-indicator
breakdown, model comparison) published as a Claude Artifact at
`https://claude.ai/code/artifact/9e283bfd-34f0-49dd-a58c-5ed52cd14601`. Its generator script is not
checked into this repo (it lives in the session that built it) — ask the user before assuming it needs
to be rebuilt from scratch.

Two static snapshots of that build are checked in and regenerated by hand when the dashboard is
refreshed (not by any script in `py/`):
- `html/dashboard.html` — the same self-contained build as the published Artifact (no external
  dependencies; embedded data), openable directly in a browser.
- `pdf/global_gdp_nowcast.pdf` — both tabs (Dashboard, then Memo) rendered into one PDF via headless
  Chrome (`chrome --headless --print-to-pdf`) against a local copy of `html/dashboard.html`, using a
  print stylesheet that forces both tabs visible and light-themed regardless of the tab UI/dark-mode
  state a live viewer would see. Landscape 17"x11" to fit the wide charts/tables without cropping.
