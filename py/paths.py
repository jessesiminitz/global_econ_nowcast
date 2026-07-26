"""
Shared output-folder locations, resolved relative to this file (not the
current working directory) so every script works the same whether it's run
as `python3 py/01_build_panel_real.py` from the repo root or `python3
01_build_panel_real.py` from inside py/. Files are organized by type:
csv/, json/, png/, txt/ — see README.md.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV = ROOT / "csv"
JSON = ROOT / "json"
PNG = ROOT / "png"
TXT = ROOT / "txt"

for _d in (CSV, JSON, PNG, TXT):
    _d.mkdir(exist_ok=True)
