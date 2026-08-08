# Haver Analytics (HaverView REST API)

Client: `scripts/haver.py` — `get_series(database, series, start_date=None, end_date=None) -> pd.Series`

A near-verbatim port of this repo's own `py/haver_client.py` — same API,
same auth, same response parsing. Kept as a separate copy so this skill
stays self-contained (no import from `py/`).

## Auth

`HAVER_API_KEY` env var, **required** — Haver is a paid subscription
product with no keyless tier. Already exported in this project's shell
setup per the user; the client also reads a `.env` file next to the
scripts as a fallback (see `.env.example`).

## Identifying a series

Haver series are identified by `mnemonic@database` (e.g. `GDPH@USECON`).
This client takes them as two separate args: `get_series("USECON", "GDPH")`.
There's no public catalog search wrapped here — mnemonics come from your
Haver subscription's data browser/documentation, or from an existing
consumer like this repo's `py/01_build_panel_real.py`, which documents
~20 real Haver mnemonics and what they represent (see that file's
module-level table in `README.md`) — e.g.:

| Mnemonic@Database | What it is |
|---|---|
| `S001XGPP@G10` | World (incl. US) real GDP, PPP-weighted, quarterly |
| `SGBLVPTG@INTSRVYS` | Real S&P Global Composite PMI |
| `NFCIM1@BCI` | Chicago Fed NFCI |
| `GS@EMBI` | Real JPMorgan EMBI Global sovereign spread |
| `LDCHOA@BONDINDX` | Bloomberg US Corporate HY OAS |

## Gotchas

- Every request needs both `database` and `series` — there's no
  cross-database series lookup by mnemonic alone.
- The API returns `dataPoints` as a list of `{date, nSeriesData}` objects;
  nulls come through as non-numeric strings, coerced to `NaN` and dropped.
- Not live-testable in this sandbox (network policy blocks
  `api.haverview.com`) or with a placeholder key — smoke-test with your
  real `HAVER_API_KEY` before relying on this.
