#!/usr/bin/env python3
"""CLI for one-off pulls: fetch a series from the terminal, print a
preview, and (by default) cache it to DuckDB via db.py. Also supports
reading straight from the cache with --query, skipping the network
entirely — useful once something's already been pulled once.

Examples:
    python3 fetch_cli.py fred GDP --start 2010-01-01
    python3 fetch_cli.py worldbank WLD NY.GDP.MKTP.KD.ZG
    python3 fetch_cli.py haver USECON GDPH
    python3 fetch_cli.py ecb EXR D.USD.EUR.SP00.A
    python3 fetch_cli.py bls CUUR0000SA0 --start 2015 --end 2023
    python3 fetch_cli.py eurostat namq_10_gdp geo=EA20 na_item=B1GQ unit=CLV_PCH_PRE s_adj=SCA
    python3 fetch_cli.py bea T10101
    python3 fetch_cli.py imf                          # recipe: world real GDP growth (no extra args)
    python3 fetch_cli.py oecd                          # recipe: G20 composite leading indicator
    python3 fetch_cli.py --query fred GDP               # read back from the cache, no network
    python3 fetch_cli.py --list-cached                  # show everything cached so far
"""
import argparse
import sys

import bea as bea_mod
import bls as bls_mod
import db
import ecb as ecb_mod
import eurostat as eurostat_mod
import fred as fred_mod
import haver as haver_mod
import imf as imf_mod
import oecd as oecd_mod
import worldbank as worldbank_mod


def _fetch(source: str, ids: list, start, end):
    """Returns (pd.Series, series_id_for_caching)."""
    if source == "fred":
        (series_id,) = ids
        return fred_mod.get_series(series_id, start=start, end=end), series_id
    if source == "haver":
        database, series = ids
        return haver_mod.get_series(database, series, start_date=start, end_date=end), series
    if source == "worldbank":
        country, indicator = ids
        start_yr = int(start[:4]) if start else None
        end_yr = int(end[:4]) if end else None
        return worldbank_mod.get_indicator(country, indicator, start=start_yr, end=end_yr), indicator
    if source == "ecb":
        flow_ref, key = ids
        return ecb_mod.get_series(flow_ref, key, start_period=start, end_period=end), key
    if source == "bls":
        (series_id,) = ids
        start_yr = int(start) if start else None
        end_yr = int(end) if end else None
        return bls_mod.get_series(series_id, start_year=start_yr, end_year=end_yr), series_id
    if source == "eurostat":
        dataset_code, *filter_parts = ids
        filters = dict(p.split("=", 1) for p in filter_parts)
        return eurostat_mod.get_series(dataset_code, filters, start_period=start, name=dataset_code), dataset_code
    if source == "imf":
        return imf_mod.world_real_gdp_growth_saar(), "world_gdp_growth_saar"
    if source == "oecd":
        return oecd_mod.composite_leading_indicator(start_period=start), "cli_g20"
    if source == "bea":
        (table_name,) = ids
        raw = bea_mod.get_data("NIPA", TableName=table_name, Frequency="Q", Year="ALL")
        return bea_mod.to_series(raw, name=table_name), table_name
    raise ValueError(f"Unknown source: {source!r}. Choices: fred, haver, worldbank, imf, oecd, eurostat, ecb, bea, bls")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="fred | haver | worldbank | imf | oecd | eurostat | ecb | bea | bls")
    parser.add_argument("ids", nargs="*", help="source-specific identifiers, e.g. a FRED series ID, or 'database series' for Haver")
    parser.add_argument("--start", help="start date/year, format depends on source (e.g. 2010-01-01 or 2010)")
    parser.add_argument("--end", help="end date/year")
    parser.add_argument("--no-cache", action="store_true", help="don't write the result to DuckDB")
    parser.add_argument("--query", action="store_true", help="read from the DuckDB cache instead of fetching (usage: --query <source> <series_id>)")
    parser.add_argument("--db-path", help="override the DuckDB cache file path (defaults to ./econ_data.duckdb or $ECON_DATA_DB_PATH)")
    parser.add_argument("--list-cached", action="store_true", help="print the cache catalog and exit")
    args = parser.parse_args()

    if args.list_cached:
        print(db.list_cached(db_path=args.db_path).to_string(index=False))
        return

    if not args.source:
        parser.error("a source is required unless --list-cached is given")

    if args.query:
        if len(args.ids) != 1:
            parser.error("--query usage: fetch_cli.py --query <source> <series_id>")
        (series_id,) = args.ids
        s = db.load_series(args.source, series_id, db_path=args.db_path)
        print(s)
        return

    try:
        series, series_id = _fetch(args.source, args.ids, args.start, args.end)
    except Exception as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(series.tail(12))
    print(f"\n{len(series)} observations, {series.index.min().date()} to {series.index.max().date()}")

    if not args.no_cache:
        n = db.save_series(series, args.source, series_id, db_path=args.db_path)
        print(f"Cached {n} rows to {db.get_db_path(args.db_path)} ({args.source}/{series_id}).")


if __name__ == "__main__":
    main()
