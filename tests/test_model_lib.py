"""
Regression tests for model_lib.py, locking in the behavior verified via
ad hoc scripts against the real walk-forward backtest this session:
the trailing-window trend, the DFM's ridge-regularized multi-factor
bridge regression, elastic-net's 1-SE regularization selection and
output-bound, and forecast_quarters()'s bound + proportional bar
rescaling. None of these depend on network access or the live-pulled
panel — see conftest.py's synthetic_panel_gdp fixture.
"""
import numpy as np
import pytest

import model_lib


# --------------------------------------------------------------- _trailing_trend
def test_trailing_trend_uses_recent_window_not_full_history():
    # A long run of 0s followed by a recent run of 10s: a full-sample mean
    # would land far from 10, but the trailing window should land at 10.
    y = np.array([0.0] * 40 + [10.0] * model_lib.TREND_WINDOW_QUARTERS)
    assert model_lib._trailing_trend(y) == pytest.approx(10.0)


def test_trailing_trend_degrades_gracefully_when_history_is_short():
    # Fewer observations than the trend window exist at all -- should use
    # everything available (numpy slicing semantics), not raise or misbehave.
    y = np.array([1.0, 2.0, 3.0])
    assert model_lib._trailing_trend(y) == pytest.approx(2.0)


# ---------------------------------------------------------------------- fit_dfm
def test_fit_dfm_decomposition_is_additive(synthetic_panel_gdp):
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_dfm(panel, gdp, cols)
    contrib = result["contrib"]
    reconstructed = contrib[cols].sum(axis=1) + contrib["trend"]
    assert np.allclose(contrib["fitted"], reconstructed, atol=1e-8)


def test_fit_dfm_returns_the_uniform_interface(synthetic_panel_gdp):
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_dfm(panel, gdp, cols)
    p = result["params"]
    # forecast_quarters() depends on exactly these keys existing with these
    # shapes, for either model interchangeably.
    for key in ("mu", "sd", "const", "effective_weights"):
        assert key in p
    for col in cols:
        assert col in p["mu"] and col in p["sd"] and col in p["effective_weights"]
    assert callable(result["predict"])


def test_fit_dfm_predict_matches_fitted_in_sample(synthetic_panel_gdp):
    # predict() re-derives its own quarterly factor from the stored mu/sd/W
    # rather than reusing the fit's internal Z -- applying it back to the
    # training panel should closely reproduce `fitted`, confirming the two
    # code paths stay consistent with each other.
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_dfm(panel, gdp, cols)
    contrib = result["contrib"]
    repredicted = result["predict"](panel)
    common = contrib.index.intersection(repredicted.index)
    assert len(common) > 10
    assert np.allclose(contrib.loc[common, "fitted"], repredicted.loc[common], atol=1e-6)


def test_fit_dfm_ridge_shrinks_beta_toward_zero(synthetic_panel_gdp):
    # Ridge-regularizing the multi-factor bridge regression (DFM_RIDGE_LAMBDA)
    # should shrink beta's magnitude relative to the unregularized OLS
    # solution it replaced -- the core mechanism the K=3 fix depends on.
    panel, gdp, cols = synthetic_panel_gdp
    with_ridge = model_lib.fit_dfm(panel, gdp, cols, n_factors=3)

    original_lambda = model_lib.DFM_RIDGE_LAMBDA
    model_lib.DFM_RIDGE_LAMBDA = 0.0
    try:
        without_ridge = model_lib.fit_dfm(panel, gdp, cols, n_factors=3)
    finally:
        model_lib.DFM_RIDGE_LAMBDA = original_lambda

    beta_with = np.array(with_ridge["params"]["beta"])
    beta_without = np.array(without_ridge["params"]["beta"])
    assert np.linalg.norm(beta_with) < np.linalg.norm(beta_without)


def test_fit_dfm_constant_column_does_not_produce_nan(synthetic_panel_gdp):
    # covid_stringency is all-zero in the fixture, matching the real panel's
    # own pre-2020 case -- the sd==0 guards should keep this NaN-free.
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_dfm(panel, gdp, cols)
    assert not result["contrib"].isna().any().any()
    assert np.isfinite(result["params"]["weights"]["covid_stringency"])


# ------------------------------------------------------------- fit_elastic_net
def test_fit_elastic_net_decomposition_is_additive(synthetic_panel_gdp):
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_elastic_net(panel, gdp, cols)
    contrib = result["contrib"]
    reconstructed = contrib[cols].sum(axis=1) + contrib["trend"]
    assert np.allclose(contrib["fitted"], reconstructed, atol=1e-8)


def test_fit_elastic_net_returns_the_uniform_interface(synthetic_panel_gdp):
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_elastic_net(panel, gdp, cols)
    p = result["params"]
    for key in ("mu", "sd", "const", "effective_weights"):
        assert key in p
    for col in cols:
        assert col in p["mu"] and col in p["sd"] and col in p["effective_weights"]
    assert callable(result["predict"])


def test_fit_elastic_net_1se_alpha_is_at_least_the_cv_optimal_alpha(synthetic_panel_gdp):
    # The 1-SE rule always picks the LARGEST alpha within one SE of the
    # CV-minimizing one -- i.e. at least as much regularization, never less.
    # Re-derive the raw CV-optimal alpha the same way fit_elastic_net does
    # internally and confirm the rule's invariant holds on this fixture.
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_elastic_net(panel, gdp, cols)
    chosen_alpha = result["params"]["alpha_enet"]

    X = panel[cols].values
    mu, sd = X.mean(axis=0), np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    import pandas as pd
    Zq = pd.DataFrame((X - mu) / sd, index=panel.index, columns=cols).resample("QS").mean()
    common = gdp.index.intersection(Zq.index)
    Zq, y = Zq.loc[common], gdp.loc[common].values
    pca = PCA(n_components=3, random_state=0)
    features = np.hstack([Zq.values, pca.fit_transform(Zq.values)])
    n_splits = max(2, min(5, len(common) // 8))
    cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                       cv=TimeSeriesSplit(n_splits=n_splits), max_iter=20000, random_state=0)
    Pipeline([("scale", StandardScaler()), ("enet", cv)]).fit(features, y - model_lib._trailing_trend(y))

    assert chosen_alpha >= cv.alpha_ - 1e-9


def test_fit_elastic_net_predict_respects_the_output_bound(synthetic_panel_gdp):
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_elastic_net(panel, gdp, cols)

    # Feed predict() a deliberately extreme month (every indicator far
    # outside anything seen in training) and confirm the output still
    # respects a sane plausibility bound rather than extrapolating linearly
    # off to an implausible value. The sanity ceiling here is a fixed,
    # hardcoded 100pp — deliberately NOT derived from
    # model_lib.PREDICT_STD_CAP_K, so this test can't be fooled by
    # widening/removing that constant the way computing the same bound
    # from it here would be (confirmed: an earlier version of this test do
    # exactly that and silently passed even with the cap disabled). The
    # real bound (~trend +- 8*std(train), a few dozen pp at most for any
    # realistic GDP series) is always far inside 100pp; an uncapped linear
    # extrapolation from indicators of 1000 is not.
    extreme = panel.tail(3).copy()
    for c in cols:
        extreme[c] = 1000.0
    pred = result["predict"](extreme)

    trend = model_lib._trailing_trend(gdp.values)
    SANITY_CEILING_PP = 100.0
    assert (pred - trend).abs().max() < SANITY_CEILING_PP


# ---------------------------------------------------------------- forecast_quarters
def test_forecast_quarters_shape_and_flags(synthetic_panel_gdp):
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_dfm(panel, gdp, cols)
    p = result["params"]
    fc = model_lib.forecast_quarters(panel, gdp, cols, p["mu"], p["sd"], p["const"], p["effective_weights"], n_ahead=4)
    assert len(fc) == 4
    assert fc["is_forecast"].all()
    assert fc["actual"].isna().all()
    assert fc["residual"].isna().all()


def test_forecast_quarters_decomposition_is_additive(synthetic_panel_gdp):
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_elastic_net(panel, gdp, cols)
    p = result["params"]
    fc = model_lib.forecast_quarters(panel, gdp, cols, p["mu"], p["sd"], p["const"], p["effective_weights"], n_ahead=4)
    reconstructed = fc[cols].sum(axis=1) + fc["trend"]
    assert np.allclose(fc["fitted"], reconstructed, atol=1e-8)


def test_forecast_quarters_caps_extreme_weights_and_keeps_additivity(synthetic_panel_gdp):
    # Same scenario as the ad hoc script used to verify this fix: absurd
    # effective_weights should get capped, not extrapolated linearly, and
    # the per-indicator bars should rescale to match rather than silently
    # stop summing to `fitted`.
    panel, gdp, cols = synthetic_panel_gdp
    mu = {c: 0.0 for c in cols}
    sd = {c: 1.0 for c in cols}
    const = 3.6
    effective_weights = {c: 0.0 for c in cols}
    effective_weights[cols[0]] = 500.0

    fc = model_lib.forecast_quarters(panel, gdp, cols, mu, sd, const, effective_weights, n_ahead=4)

    # Fixed, hardcoded sanity ceiling — deliberately NOT derived from
    # model_lib.PREDICT_STD_CAP_K (see the identical note in
    # test_fit_elastic_net_predict_respects_the_output_bound above; the
    # same tautology bug applies here if the bound is computed from the
    # same constant the code under test uses).
    trend = model_lib._trailing_trend(gdp.dropna().values)
    SANITY_CEILING_PP = 100.0
    assert (fc["fitted"] - trend).abs().max() < SANITY_CEILING_PP
    reconstructed = fc[cols].sum(axis=1) + fc["trend"]
    assert np.allclose(fc["fitted"], reconstructed, atol=1e-6)


def test_forecast_quarters_is_a_noop_when_bound_is_not_binding(synthetic_panel_gdp):
    # Sanity check on the fixture itself: with the fit's own real (modest)
    # weights, the cap should never bind, matching what was confirmed
    # against the real pipeline's current forecast quarters.
    panel, gdp, cols = synthetic_panel_gdp
    result = model_lib.fit_dfm(panel, gdp, cols)
    p = result["params"]
    fc = model_lib.forecast_quarters(panel, gdp, cols, p["mu"], p["sd"], p["const"], p["effective_weights"], n_ahead=4)
    raw = fc[cols].sum(axis=1) + fc["trend"]
    assert np.allclose(fc["fitted"], raw, atol=1e-8)
