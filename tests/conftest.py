"""
Adds py/ to sys.path (model_lib.py isn't a package, and this repo's own
scripts resolve paths relative to __file__, not CWD — see py/paths.py's
docstring) so `import model_lib` works regardless of where pytest is
invoked from, and defines the synthetic panel/GDP fixtures shared across
the test files.

Deliberately synthetic, not the real Haver-pulled panel_monthly.csv/
gdp_quarterly.csv: those need network access and drift every time the
pipeline is re-run, which would make tests flaky and non-reproducible.
model_lib.fit_dfm/fit_elastic_net/forecast_quarters are pure functions of
(panel, gdp, cols, ...) with no file I/O, so a small fixed synthetic
dataset exercises the same code paths deterministically.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))

SYNTH_COLS = ["pmi", "trade", "ip", "oil", "fin", "covid_stringency"]


@pytest.fixture
def synthetic_panel_gdp():
    """
    15 years of monthly data (2005-01 through 2019-12, 180 months / 60
    quarters) — long enough for K=3-factor PCA, ElasticNetCV's
    TimeSeriesSplit, and the walk-forward-style slicing tests below to all
    have enough observations to be well-posed, without needing real data.

    Five indicators share a common latent factor (so the panel has genuine
    exploitable structure, not pure noise) plus independent noise; GDP
    growth is a known linear function of that same factor plus its own
    noise, so fitted models should recover a meaningfully-nonzero
    relationship. `covid_stringency` is included as an all-zero column
    (matching the real pipeline's own pre-2020 case) to exercise the
    constant-column guards in fit_dfm/fit_elastic_net.
    """
    rng = np.random.default_rng(42)
    months = pd.date_range("2005-01-01", "2019-12-01", freq="MS")
    n = len(months)

    latent = rng.normal(0, 1, n).cumsum()
    latent = (latent - latent.mean()) / latent.std()

    panel = pd.DataFrame(index=months)
    loadings = {"pmi": 0.8, "trade": 0.6, "ip": 0.7, "oil": 0.3, "fin": -0.4}
    for col, loading in loadings.items():
        panel[col] = loading * latent + rng.normal(0, 0.5, n)
    panel["covid_stringency"] = 0.0

    quarters = pd.date_range("2005-01-01", "2019-10-01", freq="QS")
    latent_q = pd.Series(latent, index=months).resample("QS").mean().reindex(quarters)
    gdp = pd.Series(3.5 + 2.0 * latent_q.values + rng.normal(0, 0.8, len(quarters)), index=quarters)

    return panel, gdp, SYNTH_COLS
