"""Shared helpers for every client in this skill: .env loading, a common
requests wrapper, and small parsing utilities. Self-contained on purpose (no
imports from this repo's py/ package) so the whole econ-data-apis/ folder
stays copy-paste portable into any other project or into ~/.claude/skills/.

.env lookup order: real environment variables win (so keys already exported
in the user's shell, e.g. `export FRED_API_KEY=...` in ~/.zshrc, just work),
then a .env file colocated with this scripts/ directory (git-ignored, never
commit a key) as a fallback for projects that prefer that route.
"""
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

SCRIPTS_DIR = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

_dotenv_loaded = False


def load_dotenv(path: Optional[Path] = None) -> None:
    """Loads KEY=value lines from a .env file into os.environ, without
    overwriting variables already set in the real environment. Idempotent —
    safe to call from every module at import time."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    path = path or (SCRIPTS_DIR / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv()


def get_key(env_var: str, required_for: str = "") -> Optional[str]:
    """Reads an API key from the environment (after load_dotenv()). Returns
    None rather than raising — callers decide whether their source can run
    keyless (e.g. World Bank, IMF, OECD, Eurostat, ECB) or must raise
    (e.g. Haver)."""
    return os.environ.get(env_var)


def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 30) -> requests.Response:
    """Common GET wrapper: merges in a browser-like User-Agent (several of
    these APIs rate-limit or reject bare `python-requests` UAs), applies a
    30s timeout by default, and raises on HTTP errors."""
    merged_headers = {**UA, **(headers or {})}
    resp = requests.get(url, params=params, headers=merged_headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def parse_period(raw) -> "pd.Timestamp | None":
    """Parses an SDMX-style TIME_PERIOD value into a period-start
    pd.Timestamp: "2024-Q1"/"2024Q1" (quarterly), "2024-01"/"2024M01"
    (monthly), or "2024" (annual) — the three shapes IMF/OECD/Eurostat/ECB
    all use somewhere. Falls back to pd.to_datetime for anything else
    (e.g. a full "2024-01-15" date). Returns None if nothing parses,
    rather than raising — callers filter those rows out.

    Plain pd.to_datetime() doesn't understand "2024-Q1" natively and falls
    back to a slow, warning-emitting dateutil guess for it — this handles
    the common SDMX period shapes explicitly first instead.
    """
    s = str(raw).strip().upper().replace(" ", "")
    m = re.match(r"^(\d{4})-?Q([1-4])$", s)
    if m:
        year, q = m.groups()
        month = (int(q) - 1) * 3 + 1
        return pd.Timestamp(f"{year}-{month:02d}-01")
    m = re.match(r"^(\d{4})-?M(\d{2})$", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return pd.Timestamp(f"{m.group(1)}-{int(m.group(2)):02d}-01")
    if re.match(r"^\d{4}-\d{2}$", s) and 1 <= int(s[5:7]) <= 12:
        return pd.Timestamp(f"{s}-01")
    if re.match(r"^\d{4}$", s):
        return pd.Timestamp(f"{s}-01-01")
    parsed = pd.to_datetime(raw, errors="coerce")
    return None if pd.isna(parsed) else parsed


def series_from_records(dates, values, name: str) -> pd.Series:
    """Builds a sorted, deduplicated (keep last) pd.Series indexed by date
    from parallel date/value iterables — the shape every client in this
    skill returns."""
    s = pd.Series(pd.to_numeric(pd.Series(values), errors="coerce").values, index=pd.to_datetime(pd.Series(dates), errors="coerce"))
    s = s[s.index.notna()].dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(name)
