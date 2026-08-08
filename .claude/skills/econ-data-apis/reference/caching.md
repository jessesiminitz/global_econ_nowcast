# DuckDB caching

`scripts/db.py` persists every fetch into a local DuckDB file so a series
pulled once is queryable again without hitting the network. Requires
`pip install duckdb`.

## Location

Defaults to `./econ_data.duckdb`, resolved **relative to the current
working directory**, not this skill's install location — so whichever
project you're working in when you call these functions gets its own
cache file, consistent with how the skill needs to behave identically
whether it's installed here or copied elsewhere. Override with the
`ECON_DATA_DB_PATH` env var, or pass `db_path=` explicitly to any `db.py`
function.

## Schema

```sql
CREATE TABLE series_observations (
    source TEXT NOT NULL,      -- e.g. "fred", "worldbank", "haver"
    series_id TEXT NOT NULL,   -- e.g. "GDP", "NY.GDP.MKTP.KD.ZG"
    date DATE NOT NULL,
    value DOUBLE,
    PRIMARY KEY (source, series_id, date)
);

CREATE TABLE series_catalog (
    source TEXT NOT NULL,
    series_id TEXT NOT NULL,
    description TEXT,
    last_fetched TIMESTAMP,
    PRIMARY KEY (source, series_id)
);
```

One long-format observations table across every source — the point is
being able to query across sources without juggling per-source files,
e.g.:

```sql
-- everything cached for a given series_id, regardless of source
SELECT * FROM series_observations WHERE series_id = 'GDP' ORDER BY date;

-- what's been pulled, and when
SELECT * FROM series_catalog ORDER BY last_fetched DESC;

-- join for a quick sanity check of coverage
SELECT source, series_id, min(date), max(date), count(*)
FROM series_observations GROUP BY 1, 2;
```

Query directly with the `duckdb` CLI (`duckdb econ_data.duckdb`) or from
Python (`duckdb.connect("econ_data.duckdb").sql("...")`).

## API

- `save_series(data, source, series_id, description=None, db_path=None)`
  — upserts a `pd.Series` (or single-column `pd.DataFrame`). Re-running the
  same fetch updates existing rows in place rather than duplicating them
  (`ON CONFLICT (source, series_id, date) DO UPDATE`).
- `load_series(source, series_id, db_path=None) -> pd.Series` — reads back
  without hitting the network. Raises if nothing's cached yet.
- `list_cached(db_path=None) -> pd.DataFrame` — the catalog: what's been
  pulled, when, and how many observations/what date range each has.

## Cache invalidation

There's no automatic staleness/expiry — this is a convenience layer, not a
source of truth. If a series gets revised upstream (common for recent
GDP/employment prints) and you need the current value, just re-fetch and
re-`save_series()` — the upsert overwrites the stale rows. To drop
something entirely, delete the corresponding rows directly
(`DELETE FROM series_observations WHERE source = ... AND series_id = ...`)
or just delete the whole `.duckdb` file to start fresh.
