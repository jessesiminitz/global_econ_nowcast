"""
Shared model-fitting logic for the DFM baseline and the elastic-net/factor-ML
challenger. Both `fit_dfm` and `fit_elastic_net` are pure functions of
(panel, gdp, cols) — no file I/O — so they can be called standalone by
02_estimate_dfm.py / 03_estimate_ml.py, or repeatedly on expanding training
windows by 04_backtest.py's rolling out-of-sample comparison.

Each returns a dict with the same shape:
  {"contrib": DataFrame,   # quarterly per-indicator decomposition (matches
                           # quarterly_decomposition.csv's existing columns:
                           # trend, fitted, actual, residual, plus one column
                           # per indicator)
   "params": {...},        # scalars/diagnostics worth printing or logging
   "predict": callable}    # predict(monthly_panel_slice) -> quarterly
                           # pd.Series of nowcast growth, using only the
                           # training-fitted transform (for out-of-sample use)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Both fit_dfm and fit_elastic_net anchor their level/trend to the trailing
# N quarters of the training window's own GDP prints, rather than a
# full-sample least-squares intercept. 04_backtest.py's walk-forward window
# is EXPANDING (train_panel = panel.loc[:cutoff]), so a full-sample
# intercept keeps 2005-2007's real, one-off pre-GFC boom (+5.17% avg growth,
# vs +2.85%-+3.66% every era since) baked into the "trend" for every fold
# from 2008 through the present — both models shared an almost identical
# ~-0.47pp calm-regime overprediction bias as a result. A trailing window
# lets that stale boom drop out of the trend estimate once enough more
# recent history accumulates, while still using the FULL training window
# for the factor/weight structure (which isn't what was biased). This is a
# partial fix, not a complete one — measured directly, an expanding-window
# mean runs ~0.15-0.35pp above a trailing-20q mean at typical backtest
# cutoffs, smaller than the full observed bias — so re-measure after
# changing this rather than assuming it's fully closed.
TREND_WINDOW_QUARTERS = 20

# fit_elastic_net's inner ElasticNetCV picks its regularization strength via
# TimeSeriesSplit(n_splits=max(2, min(5, T_train // 8))) — with as few as 12
# training quarters (04_backtest.py's MIN_TRAIN_QUARTERS), that's only 2 CV
# folds of ~6 observations each, tuning across 23 features (20 indicators +
# 3 PCA factors). Measured this session: elastic-net's backtest errors carry
# a -0.27 lag-1 autocorrelation (vs DFM's -0.04) — a plausible symptom of
# noisy quarter-to-quarter hyperparameter selection on folds this thin.
#
# TRIED FIRST AND REPLACED: a hard gate (below some MIN_CV_QUARTERS, skip CV
# entirely and use a fixed, guessed alpha/l1_ratio). Measured: it left the
# aggregate lag-1 autocorrelation essentially unchanged (-0.261 vs -0.267) —
# it only touched ~8 of 73 folds, and the fallback values were arbitrary
# guesses, not derived from anything.
#
# WHAT ACTUALLY WORKED: the standard "1-SE rule" (Hastie/Tibshirani) —
# instead of taking ElasticNetCV's own argmin-CV-MSE alpha (which is exactly
# what a noisy 2-fold/~6-obs CV curve is worst at estimating precisely),
# pick the LARGEST alpha (most regularization, i.e. the simplest model)
# whose mean CV MSE is still within one standard error of the minimum.
# Applied to every fold, not just thin ones, since the underlying CV noise
# problem isn't unique to the earliest folds. Measured on the full 73-quarter
# backtest: calm RMSE 2.05->1.50, calm bias -0.481->-0.195, lag-1
# autocorrelation -0.267->-0.035 (matching DFM's own -0.03 to -0.04 baseline)
# — a real, broad improvement, not just a fix for the diagnosed symptom.
# Stressed RMSE ticked up slightly (13.29->13.83, n=7, noise-level). Adding
# the old hard-gate fallback BACK on top of this made things marginally
# worse across the board (tested), so it's not used alongside this.

# TRIED AND REVERTED: clipping standardized inputs to +-4 before the linear
# combination, to bound the -52.4%-vs-actual-(-21.3%) 2020-04 prediction.
# Measured result: it made things WORSE, not better — clipped calm RMSE
# 2.76 vs 2.05 unclipped, and the single worst error actually grew to 40.6
# (vs 31.1 unclipped). Mechanism: clipping the INPUT doesn't bound the
# OUTPUT — fit on truncated training extremes, the model compensated with
# larger coefficients (e.g. ip's weight grew to ~4.2), so a clipped input
# times an inflated weight can overshoot even more than an unclipped input
# times the original weight did. Bounding the prediction itself (e.g. a
# Huber loss on the target, or capping the final output) would be a
# different, untested approach — not implemented here.


def _trailing_trend(y: np.ndarray, window: int = TREND_WINDOW_QUARTERS) -> float:
    return float(np.mean(y[-window:]))


def fit_dfm(panel: pd.DataFrame, gdp: pd.Series, cols: list, n_factors: int = 1) -> dict:
    """
    DFM: standardize -> top-`n_factors` principal components (static
    factors) -> multivariate bridge regression of quarterly GDP growth
    (demeaned by a trailing-window trend, see _trailing_trend) on the
    quarterly-averaged static factors -> exact per-indicator decomposition
    (each static factor is an exact linear combination of the standardized
    indicators, so the bridge-implied growth decomposes additively across
    indicators regardless of how many factors are used). A scalar AR(1) +
    Kalman filter/smoother is still fit on the FIRST factor alone as a
    diagnostic (`factor_smoothed` on the returned panel, `phi`/`sigma_w2`
    in `params`) — it does not feed `predict()`, `contrib`, or `fitted`
    today (never did).

    n_factors defaults to 1 (single-factor), not because more factors
    wouldn't explain more full-sample variance — they do (K=3 clears 60.9%
    cumulative variance vs. 38.8% for K=1, measured on this panel) — but
    because that doesn't hold up walk-forward: tested directly via
    04_backtest.py's harness, K=2 and K=3 both had WORSE calm-regime RMSE
    than K=1 (1.65 at K=1, vs 2.34 at K=2 and 2.05 at K=3). More factors
    means more parameters fit on training windows as short as 12-19
    quarters early in the backtest — a textbook overfitting risk that a
    full-sample variance-explained check doesn't catch. The K-factor
    machinery is kept (parameterized, not hardcoded to K=1) in case a
    future change to MIN_TRAIN_QUARTERS or the panel makes more factors
    viable, but re-verify against the walk-forward backtest, not just
    var_explained, before raising this default.
    """
    panel = panel.dropna(subset=cols).copy()
    X = panel[cols].values
    T, N = X.shape

    mu = X.mean(axis=0)
    # A column can be constant within a given training window (e.g.
    # covid_stringency is exactly 0 for any window entirely before 2020) —
    # dividing by a zero std there is 0/0 = NaN, which poisons the whole
    # covariance matrix. Treat a constant column as contributing nothing
    # (z-score 0) instead of NaN; `sd` (with zeros replaced by 1) is reused
    # by `predict` below so training and prediction stay consistent.
    sd = np.where(X.std(axis=0, ddof=0) == 0, 1.0, X.std(axis=0, ddof=0))
    Z = (X - mu) / sd

    cov = np.cov(Z, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]

    # Top-K static factors (K=1 by default — see the docstring above for why
    # more factors were tried and reverted). Each is independently
    # unit-scaled the same way the original single factor was.
    K = max(1, min(n_factors, N, T - 2, len(eigval)))
    W = np.zeros((N, K))
    F_static = np.zeros((T, K))
    for k in range(K):
        vk = eigvec[:, k]
        if vk[0] < 0:
            vk = -vk
        raw_fk = Z @ vk
        f_scale = raw_fk.std()
        wk = vk / f_scale if f_scale > 0 else vk
        W[:, k] = wk
        F_static[:, k] = Z @ wk

    # Everything below through the Kalman smoother is a DIAGNOSTIC fit on
    # the FIRST factor alone (F1) — it feeds `factor_smoothed`/`phi`/
    # `sigma_w2` for logging only. The bridge regression a few lines down
    # uses the full K-factor F_static, not F1 or f_smooth.
    F1 = F_static[:, 0]
    # Z[:, i] can be constant within the training window (see the sd guard
    # above), which makes corrcoef divide-by-zero (undefined correlation for
    # a constant series) — this `lam` is diagnostic-only, so 0 is fine.
    lam = np.array([
        0.0 if np.allclose(Z[:, i], Z[0, i]) else np.corrcoef(Z[:, i], F1)[0, 1] * sd[i]
        for i in range(N)
    ])

    var_explained = eigval[:K].sum() / eigval.sum()

    phi = np.sum(F1[1:] * F1[:-1]) / np.sum(F1[:-1] ** 2)
    resid_f = F1[1:] - phi * F1[:-1]
    sigma_w2 = resid_f.var()

    lam_on_F = np.array([np.polyfit(F1, Z[:, i], 1)[0] for i in range(N)])
    idio_var = np.array([np.var(Z[:, i] - lam_on_F[i] * F1) for i in range(N)])

    # A constant column within the training window (e.g. covid_stringency
    # pre-2020, or any indicator with sd==0 above) has exactly zero
    # idiosyncratic variance, which would divide-by-zero in the Kalman gain
    # below. This only affects the diagnostic smoothed factor/loadings (the
    # bridge regression and decomposition use the static factor F_static,
    # not the smoothed one), but a tiny floor keeps it numerically clean.
    R = np.where(idio_var == 0, 1e-8, idio_var)
    Lam = lam_on_F

    f_pred = np.zeros(T)
    P_pred = np.zeros(T)
    f_filt = np.zeros(T)
    P_filt = np.zeros(T)

    f_pred[0], P_pred[0] = 0.0, 1.0 / (1 - phi ** 2) if abs(phi) < 1 else 1.0
    for t in range(T):
        if t > 0:
            f_pred[t] = phi * f_filt[t - 1]
            P_pred[t] = phi ** 2 * P_filt[t - 1] + sigma_w2
        prec = 1.0 / P_pred[t] + np.sum(Lam ** 2 / R)
        P_filt[t] = 1.0 / prec
        f_filt[t] = P_filt[t] * (f_pred[t] / P_pred[t] + np.sum(Lam * Z[t, :] / R))

    f_smooth = np.zeros(T)
    P_smooth = np.zeros(T)
    f_smooth[-1], P_smooth[-1] = f_filt[-1], P_filt[-1]
    for t in range(T - 2, -1, -1):
        J = phi * P_filt[t] / P_pred[t + 1]
        f_smooth[t] = f_filt[t] + J * (f_smooth[t + 1] - phi * f_filt[t])
        P_smooth[t] = P_filt[t] + J ** 2 * (P_smooth[t + 1] - P_pred[t + 1])

    panel["factor_static"] = F1
    panel["factor_smoothed"] = f_smooth

    Fq_static = pd.DataFrame(F_static, index=panel.index, columns=[f"f{k}" for k in range(K)]).resample("QS").mean()
    common_idx = gdp.index.intersection(Fq_static.index)
    y = gdp.loc[common_idx].values
    Xf = Fq_static.loc[common_idx].values  # T_train x K

    # Estimate beta by demeaning y with the FULL training-window mean, not
    # the trailing trend — Xf's columns are already ~mean-zero over this
    # same full window (built from mean-zero Z), so this reproduces the
    # original well-posed with-intercept OLS's beta (regressing y on
    # [1, Xf] jointly gives alpha_ols ~ full_mean when Xf is ~mean-zero,
    # so demeaning by full_mean and dropping the intercept column is
    # numerically equivalent). Demeaning by the trailing trend INSTEAD,
    # here, would mismatch Xf's own full-window centering and distort
    # beta — confirmed by testing: it made bias and RMSE worse, not
    # better. The trailing trend is used only below, as the reported
    # level/intercept for `fitted`, decoupled from beta estimation.
    full_mean = float(np.mean(y))
    beta_vec, *_ = np.linalg.lstsq(Xf, y - full_mean, rcond=None)
    trend = _trailing_trend(y)
    alpha = trend
    fitted = alpha + Xf @ beta_vec
    resid = y - fitted
    r2 = 1 - resid.var() / y.var()

    Zdf = pd.DataFrame(Z, index=panel.index, columns=cols)
    Zq = Zdf.resample("QS").mean().loc[common_idx]

    # Combined per-indicator sensitivity across all K factors — indicator i's
    # effective weight is sum_k(beta_vec[k] * W[i, k]), so the decomposition
    # stays additive per indicator even with K>1 factors.
    combined_w = W @ beta_vec  # length N

    contrib = pd.DataFrame(index=common_idx, columns=cols, dtype=float)
    for i, c in enumerate(cols):
        contrib[c] = combined_w[i] * Zq[c].values
    contrib["trend"] = alpha
    contrib["fitted"] = contrib[cols].sum(axis=1) + alpha
    contrib["actual"] = y
    contrib["residual"] = contrib["actual"] - contrib["fitted"]

    def predict(monthly_panel_slice: pd.DataFrame) -> pd.Series:
        Xp = monthly_panel_slice[cols].values
        Zp = (Xp - mu) / sd
        Fp = Zp @ W  # T x K
        Fpq = pd.DataFrame(Fp, index=monthly_panel_slice.index).resample("QS").mean()
        return pd.Series(alpha + Fpq.values @ beta_vec, index=Fpq.index)

    params = {
        "var_explained": var_explained,
        "n_factors": K,
        "phi": phi,
        "sigma_w2": sigma_w2,
        "alpha": alpha,
        "beta": beta_vec.tolist(),
        "r2": r2,
        # Combined (bridge-scaled) per-indicator weight — same semantics as
        # fit_elastic_net's own `weights` below (contribution per unit
        # z-score), not the raw pre-bridge factor loading.
        "weights": dict(zip(cols, combined_w)),
        "loadings_on_F": dict(zip(cols, Lam)),
        # Uniform fields (same keys fit_elastic_net exposes below) so
        # forecast_quarters() can extend either model without special-casing:
        # fitted = const + sum_i effective_weights[i] * Zq_i.
        "mu": dict(zip(cols, mu)),
        "sd": dict(zip(cols, sd)),
        "const": alpha,
        "effective_weights": dict(zip(cols, combined_w)),
    }
    return {"contrib": contrib, "panel_with_factor": panel, "params": params, "predict": predict}


def fit_elastic_net(panel: pd.DataFrame, gdp: pd.Series, cols: list, n_factors: int = 3,
                     random_state: int = 0) -> dict:
    """
    Factor-augmented elastic net: quarterly-averaged, standardized indicators
    plus the top-`n_factors` PCA components of that same quarterly panel are
    fed into ElasticNetCV (walk-forward-safe TimeSeriesSplit for the internal
    alpha/l1_ratio search). Because the whole pipeline (PCA -> StandardScaler
    -> ElasticNet) is an affine function of the quarterly z-scored
    indicators, its per-indicator sensitivity is recovered by probing that
    affine map with basis vectors — giving an exact additive decomposition,
    the same shape as the DFM's.
    """
    panel = panel.dropna(subset=cols).copy()
    X = panel[cols].values
    mu = X.mean(axis=0)
    # See fit_dfm's identical guard: a constant column within a training
    # window (e.g. covid_stringency pre-2020) would otherwise divide by zero.
    sd = np.where(X.std(axis=0, ddof=0) == 0, 1.0, X.std(axis=0, ddof=0))
    # No input clipping here — tried and reverted, see the module-level
    # "TRIED AND REVERTED" comment above.
    Z = (X - mu) / sd
    Zdf = pd.DataFrame(Z, index=panel.index, columns=cols)
    Zq_full = Zdf.resample("QS").mean()

    common_idx = gdp.index.intersection(Zq_full.index)
    Zq = Zq_full.loc[common_idx]
    y = gdp.loc[common_idx].values
    N = len(cols)
    T_train = len(common_idx)

    # Trend/level from the trailing window, same mechanism and same
    # rationale as fit_dfm's identical step — fit the elastic net on the
    # DEMEANED target so a stale early-history level can't anchor it.
    trend = _trailing_trend(y)
    y_demeaned = y - trend

    n_factors_eff = max(1, min(n_factors, N, T_train - 2))
    pca = PCA(n_components=n_factors_eff, random_state=random_state)
    pca_scores = pca.fit_transform(Zq.values)
    features = np.hstack([Zq.values, pca_scores])

    n_splits = max(2, min(5, T_train // 8))
    l1_ratio_grid = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
    cv_pipe = Pipeline([("scale", StandardScaler()), ("enet", ElasticNetCV(
        l1_ratio=l1_ratio_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        max_iter=20000,
        random_state=random_state,
    ))])
    cv_pipe.fit(features, y_demeaned)
    enet_cv = cv_pipe.named_steps["enet"]

    # 1-SE rule: rather than the argmin-CV-MSE alpha (exactly what a noisy
    # small-sample CV curve estimates worst), pick the LARGEST alpha (most
    # regularization) whose mean CV MSE is still within one standard error
    # of the minimum — see the module-level comment above for the measured
    # effect (this materially reduced both bias and error autocorrelation).
    l1_idx = l1_ratio_grid.index(enet_cv.l1_ratio_)
    mse_path = enet_cv.mse_path_[l1_idx]  # n_alphas x n_folds
    alphas_path = enet_cv.alphas_[l1_idx]
    mean_mse = mse_path.mean(axis=1)
    se_mse = mse_path.std(axis=1) / np.sqrt(mse_path.shape[1])
    best_idx = np.argmin(mean_mse)
    within_1se = np.where(mean_mse <= mean_mse[best_idx] + se_mse[best_idx])[0]
    chosen_idx = within_1se[np.argmax(alphas_path[within_1se])]
    alpha_used, l1_ratio_used = float(alphas_path[chosen_idx]), float(enet_cv.l1_ratio_)

    pipe = Pipeline([("scale", StandardScaler()), ("enet", ElasticNet(
        alpha=alpha_used, l1_ratio=l1_ratio_used, max_iter=20000, random_state=random_state,
    ))])
    pipe.fit(features, y_demeaned)

    def _predict_from_zq_row(zq_row: np.ndarray) -> float:
        pca_row = pca.transform(zq_row.reshape(1, -1))
        feat_row = np.hstack([zq_row.reshape(1, -1), pca_row])
        return float(pipe.predict(feat_row)[0])

    resid_intercept = _predict_from_zq_row(np.zeros(N))
    c0 = trend + resid_intercept
    g = np.array([_predict_from_zq_row(np.eye(N)[j]) - resid_intercept for j in range(N)])

    fitted = c0 + Zq.values @ g
    resid = y - fitted
    r2 = 1 - resid.var() / y.var()

    contrib = pd.DataFrame(index=common_idx, columns=cols, dtype=float)
    for i, c in enumerate(cols):
        contrib[c] = g[i] * Zq[c].values
    contrib["trend"] = c0
    contrib["fitted"] = contrib[cols].sum(axis=1) + c0
    contrib["actual"] = y
    contrib["residual"] = contrib["actual"] - contrib["fitted"]

    def predict(monthly_panel_slice: pd.DataFrame) -> pd.Series:
        Xp = monthly_panel_slice[cols].values
        Zp = (Xp - mu) / sd
        Zpq = pd.DataFrame(Zp, index=monthly_panel_slice.index, columns=cols).resample("QS").mean()
        preds = {qi: c0 + row.values @ g for qi, row in Zpq.iterrows()}
        return pd.Series(preds)

    params = {
        "alpha_enet": alpha_used,
        "l1_ratio_enet": l1_ratio_used,
        "n_factors": n_factors_eff,
        "r2": r2,
        "weights": dict(zip(cols, g)),
        "intercept": c0,
        # Uniform fields — see fit_dfm's identical block.
        "mu": dict(zip(cols, mu)),
        "sd": dict(zip(cols, sd)),
        "const": c0,
        "effective_weights": dict(zip(cols, g)),
    }
    return {"contrib": contrib, "params": params, "predict": predict}


def forecast_quarters(panel: pd.DataFrame, gdp: pd.Series, cols: list, mu: dict, sd: dict,
                       const: float, effective_weights: dict, n_ahead: int = 4) -> pd.DataFrame:
    """
    Extend either model's decomposition `n_ahead` quarters past the last
    published GDP print, reusing its already-fitted linear map (fitted =
    const + sum_i effective_weights[i] * Zq_i) — this works identically for
    the DFM and the elastic net since both are affine in the quarterly
    z-scored indicators by construction (see fit_dfm/fit_elastic_net above).

    Two regimes, quarter by quarter:
      - "Nowcast gap": indicator data for that quarter is already published
        in `panel` even though GDP hasn't been reported yet (indicators lead
        GDP releases) — use the real z-scored readings directly, same as
        `predict()`.
      - Genuine forecast: no indicator data exists yet for that quarter —
        extrapolate each indicator's own quarterly z-score forward via its
        own AR(1) persistence (fit on its historical quarterly series),
        i.e. mean-reverting toward 0 (its long-run standardized average).
        This is deliberately simple (no attempt to forecast realistic future
        indicator *levels*) — it's a persistence-based placeholder, not a
        claim about what will actually happen to e.g. oil prices.

    Returns a DataFrame shaped like `contrib` above, plus an `is_forecast`
    column (all True) and `actual`/`residual` left as NaN (nothing to
    compare against yet).
    """
    cols = list(cols)
    mu_arr = np.array([mu[c] for c in cols])
    sd_arr = np.array([sd[c] for c in cols])

    X = panel[cols].values
    Z = (X - mu_arr) / sd_arr
    Zdf = pd.DataFrame(Z, index=panel.index, columns=cols)
    Zq_all = Zdf.resample("QS").mean()

    last_actual_q = gdp.index.max()
    target_quarters = pd.date_range(last_actual_q + pd.DateOffset(months=3), periods=n_ahead, freq="QS")

    hist_idx = gdp.index.intersection(Zq_all.index)
    Zq_hist = Zq_all.loc[hist_idx].sort_index()
    phi = {}
    for c in cols:
        s = Zq_hist[c].dropna().values
        denom = np.sum(s[:-1] ** 2) if len(s) > 1 else 0.0
        p = np.sum(s[1:] * s[:-1]) / denom if denom > 0 else 0.0
        phi[c] = float(np.clip(p, -0.98, 0.98))

    real_quarterly_available = Zq_all.dropna(subset=cols)
    last_known = {
        c: (Zq_all[c].dropna().iloc[-1] if Zq_all[c].dropna().size else 0.0)
        for c in cols
    }

    rows = {}
    for q in target_quarters:
        if q in real_quarterly_available.index:
            row = {c: real_quarterly_available.loc[q, c] for c in cols}
        else:
            row = {c: phi[c] * last_known[c] for c in cols}
        rows[q] = row
        last_known = row

    Zq_fore = pd.DataFrame(rows).T[cols]

    contrib = pd.DataFrame(index=Zq_fore.index, columns=cols, dtype=float)
    for c in cols:
        contrib[c] = effective_weights[c] * Zq_fore[c].values
    contrib["trend"] = const
    contrib["fitted"] = contrib[cols].sum(axis=1) + const
    contrib["actual"] = np.nan
    contrib["residual"] = np.nan
    contrib["is_forecast"] = True
    return contrib
