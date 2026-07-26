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
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


def fit_dfm(panel: pd.DataFrame, gdp: pd.Series, cols: list) -> dict:
    """
    Single-factor DFM: standardize -> first principal component (static
    factor) -> AR(1) + Kalman filter/smoother -> bridge regression of
    quarterly GDP growth on the quarterly-averaged static factor -> exact
    per-indicator decomposition (the static factor is an exact linear
    combination of the standardized indicators, so the bridge-implied growth
    decomposes additively). Same math as the original 02_estimate_dfm.py.
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

    v1 = eigvec[:, 0]
    if v1[0] < 0:
        v1 = -v1
    raw_f = Z @ v1
    f_scale = raw_f.std()
    w = v1 / f_scale
    F_static = Z @ w
    # Z[:, i] can be constant within the training window (see the sd guard
    # above), which makes corrcoef divide-by-zero (undefined correlation for
    # a constant series) — this `lam` is diagnostic-only, so 0 is fine.
    lam = np.array([
        0.0 if np.allclose(Z[:, i], Z[0, i]) else np.corrcoef(Z[:, i], F_static)[0, 1] * sd[i]
        for i in range(N)
    ])

    var_explained = eigval[0] / eigval.sum()

    phi = np.sum(F_static[1:] * F_static[:-1]) / np.sum(F_static[:-1] ** 2)
    resid_f = F_static[1:] - phi * F_static[:-1]
    sigma_w2 = resid_f.var()

    lam_on_F = np.array([np.polyfit(F_static, Z[:, i], 1)[0] for i in range(N)])
    idio_var = np.array([np.var(Z[:, i] - lam_on_F[i] * F_static) for i in range(N)])

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

    panel["factor_static"] = F_static
    panel["factor_smoothed"] = f_smooth

    Fq_static = pd.Series(F_static, index=panel.index).resample("QS").mean()
    common_idx = gdp.index.intersection(Fq_static.index)
    y = gdp.loc[common_idx].values
    x = Fq_static.loc[common_idx].values
    Xd = np.column_stack([np.ones_like(x), x])
    beta_hat, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    alpha, beta = beta_hat
    fitted = alpha + beta * x
    resid = y - fitted
    r2 = 1 - resid.var() / y.var()

    Zdf = pd.DataFrame(Z, index=panel.index, columns=cols)
    Zq = Zdf.resample("QS").mean().loc[common_idx]

    contrib = pd.DataFrame(index=common_idx, columns=cols, dtype=float)
    for i, c in enumerate(cols):
        contrib[c] = beta * w[i] * Zq[c].values
    contrib["trend"] = alpha
    contrib["fitted"] = contrib[cols].sum(axis=1) + alpha
    contrib["actual"] = y
    contrib["residual"] = contrib["actual"] - contrib["fitted"]

    def predict(monthly_panel_slice: pd.DataFrame) -> pd.Series:
        Xp = monthly_panel_slice[cols].values
        Zp = (Xp - mu) / sd
        Fp = Zp @ w
        Fpq = pd.Series(Fp, index=monthly_panel_slice.index).resample("QS").mean()
        return alpha + beta * Fpq

    params = {
        "var_explained": var_explained,
        "phi": phi,
        "sigma_w2": sigma_w2,
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
        "weights": dict(zip(cols, w)),
        "loadings_on_F": dict(zip(cols, Lam)),
        # Uniform fields (same keys fit_elastic_net exposes below) so
        # forecast_quarters() can extend either model without special-casing:
        # fitted = const + sum_i effective_weights[i] * Zq_i.
        "mu": dict(zip(cols, mu)),
        "sd": dict(zip(cols, sd)),
        "const": alpha,
        "effective_weights": {c: beta * w[i] for i, c in enumerate(cols)},
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
    Z = (X - mu) / sd
    Zdf = pd.DataFrame(Z, index=panel.index, columns=cols)
    Zq_full = Zdf.resample("QS").mean()

    common_idx = gdp.index.intersection(Zq_full.index)
    Zq = Zq_full.loc[common_idx]
    y = gdp.loc[common_idx].values
    N = len(cols)
    T_train = len(common_idx)

    n_factors_eff = max(1, min(n_factors, N, T_train - 2))
    pca = PCA(n_components=n_factors_eff, random_state=random_state)
    pca_scores = pca.fit_transform(Zq.values)
    features = np.hstack([Zq.values, pca_scores])

    n_splits = max(2, min(5, T_train // 8))
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=TimeSeriesSplit(n_splits=n_splits),
            max_iter=20000,
            random_state=random_state,
        )),
    ])
    pipe.fit(features, y)

    def _predict_from_zq_row(zq_row: np.ndarray) -> float:
        pca_row = pca.transform(zq_row.reshape(1, -1))
        feat_row = np.hstack([zq_row.reshape(1, -1), pca_row])
        return float(pipe.predict(feat_row)[0])

    c0 = _predict_from_zq_row(np.zeros(N))
    g = np.array([_predict_from_zq_row(np.eye(N)[j]) - c0 for j in range(N)])

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
        "alpha_enet": pipe.named_steps["enet"].alpha_,
        "l1_ratio_enet": pipe.named_steps["enet"].l1_ratio_,
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
