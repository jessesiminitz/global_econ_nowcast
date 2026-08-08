"""DuckDB caching layer shared by every client in this skill.

Every fetch returns a plain pd.Series/pd.DataFrame — this module is an
optional layer on top that persists what's been pulled so it's queryable
later without hitting the network again. One long-format table
(series_observations) holds every source's data so it's all queryable
together; a small catalog table tracks what's been fetched and when.

The database file defaults to ./econ_data.duckdb, resolved relative to the
current working directory (not this skill's install location) — so if this
skill is copied into another project, each project naturally gets its own
cache alongside its own data, exactly like this repo's own csv/json/ folders
are project-local. Override with the ECON_DATA_DB_PATH env var.

Requires the `duckdb` package (`pip install duckdb`).
"""
from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd

from _util import get_key

_SCHEMA = """
CREATE TABLE IF NOT EXISTS series_observations (
    source TEXT NOT NULL,
    series_id TEXT NOT NULL,
    date DATE NOT NULL,
    value DOUBLE,
    PRIMARY KEY (source, series_id, date)
);
CREATE TABLE IF NOT EXISTS series_catalog (
    source TEXT NOT NULL,
    series_id TEXT NOT NULL,
    description TEXT,
    last_fetched TIMESTAMP,
    PRIMARY KEY (source, series_id)
);
"""


def get_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    override = get_key("ECON_DATA_DB_PATH")
    if override:
        return Path(override)
    return Path.cwd() / "econ_data.duckdb"


def _connect(db_path: Optional[Union[str, Path]] = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(get_db_path(db_path)))
    con.execute(_SCHEMA)
    return con


def save_series(
    data: Union[pd.Series, pd.DataFrame],
    source: str,
    series_id: str,
    description: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Upserts a fetched series into the cache. Accepts a pd.Series (or a
    single-column pd.DataFrame) indexed by date. Re-running the same fetch
    is idempotent: existing (source, series_id, date) rows get their value
    updated in place rather than duplicated. Returns the number of rows
    written.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(f"save_series() expects a single-column DataFrame or a Series, got {data.shape[1]} columns — call it once per column.")
        data = data.iloc[:, 0]

    if data.empty:
        return 0

    df = pd.DataFrame({
        "source": source,
        "series_id": series_id,
        "date": pd.to_datetime(data.index).date,
        "value": data.values,
    })

    con = _connect(db_path)
    try:
        con.register("_incoming", df)
        con.execute("""
            INSERT INTO series_observations (source, series_id, date, value)
            SELECT source, series_id, date, value FROM _incoming
            ON CONFLICT (source, series_id, date) DO UPDATE SET value = excluded.value
        """)
        con.execute("""
            INSERT INTO series_catalog (source, series_id, description, last_fetched)
            VALUES (?, ?, ?, now())
            ON CONFLICT (source, series_id) DO UPDATE
                SET description = COALESCE(excluded.description, series_catalog.description),
                    last_fetched = excluded.last_fetched
        """, [source, series_id, description])
    finally:
        con.close()
    return len(df)


def load_series(source: str, series_id: str, db_path: Optional[Union[str, Path]] = None) -> pd.Series:
    """Reads a previously-cached series back out, without hitting the
    network. Raises if nothing's cached for (source, series_id)."""
    con = _connect(db_path)
    try:
        df = con.execute(
            "SELECT date, value FROM series_observations WHERE source = ? AND series_id = ? ORDER BY date",
            [source, series_id],
        ).fetchdf()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError(f"No cached data for {series_id}@{source} in {get_db_path(db_path)} — fetch it first.")
    s = df.set_index(pd.to_datetime(df["date"]))["value"]
    return s.rename(series_id)


def list_cached(db_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Returns the catalog of every series cached so far: source, series_id,
    description, last_fetched, and row count."""
    con = _connect(db_path)
    try:
        return con.execute("""
            SELECT c.source, c.series_id, c.description, c.last_fetched,
                   count(o.date) AS n_observations,
                   min(o.date) AS start_date, max(o.date) AS end_date
            FROM series_catalog c
            LEFT JOIN series_observations o
                ON o.source = c.source AND o.series_id = c.series_id
            GROUP BY c.source, c.series_id, c.description, c.last_fetched
            ORDER BY c.last_fetched DESC
        """).fetchdf()
    finally:
        con.close()
