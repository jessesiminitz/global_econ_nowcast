"""
Estimate a single-factor dynamic factor model on the 6-indicator monthly
panel, following the standard two-step estimator (Doz, Giannone & Reichlin,
2011, "A two-step estimator for large approximate dynamic factor models
based on Kalman filtering", J. Econometrics):

  Step 1 (static):  standardize indicators -> first principal component
                     gives factor loadings Lambda and an initial factor F_t.
  Step 2 (dynamic):  fit an AR(1) to F_t, then run a Kalman filter/smoother
                     over the state-space {f_t = phi f_{t-1} + w_t,
                     x_t = Lambda f_t + e_t} to get the smoothed monthly
                     factor f_t|T, properly accounting for idiosyncratic
                     noise and factor dynamics.

Then: bridge regression of quarterly GDP growth on the quarterly-averaged
factor gives the mapping from "common cycle" to growth units. Because the
Step-1 factor is an exact linear combination of the standardized
indicators (F_t = sum_i w_i * z_i,t), the bridge-implied GDP growth is
*exactly* decomposable into additive per-indicator contributions — this is
what feeds the stacked-bar chart.
"""
import numpy as np
import pandas as pd

panel = pd.read_csv("panel_monthly.csv", index_col=0, parse_dates=True)
gdp = pd.read_csv("gdp_quarterly.csv", index_col=0, parse_dates=True).iloc[:, 0]

cols = ["pmi", "trade", "ip", "oil", "fin", "ai"]
panel = panel.dropna(subset=cols)
X = panel[cols].values
T, N = X.shape

# ---------- Step 1: standardize + PCA (static factor) ----------
mu = X.mean(axis=0)
sd = X.std(axis=0, ddof=0)
Z = (X - mu) / sd

cov = np.cov(Z, rowvar=False)
eigval, eigvec = np.linalg.eigh(cov)
order = np.argsort(eigval)[::-1]
eigval, eigvec = eigval[order], eigvec[:, order]

v1 = eigvec[:, 0]                      # first PC loading vector (unit norm)
if v1[0] < 0:                          # sign normalization: PMI should load positively
    v1 = -v1
raw_f = Z @ v1
f_scale = raw_f.std()
w = v1 / f_scale                       # so that F_t = Z_t @ w has unit variance
F_static = Z @ w                       # exact linear static factor
lam = np.array([np.corrcoef(Z[:, i], F_static)[0, 1] * sd[i] for i in range(N)])  # for reporting

var_explained = eigval[0] / eigval.sum()
print(f"Share of common variance explained by factor 1: {var_explained:.1%}")
print("Standardized loadings (w_i, exact linear weights in F_t = sum w_i * z_i,t):")
for c, wi in zip(cols, w):
    print(f"  {c:6s}  w={wi:+.3f}")

# ---------- Step 2: AR(1) dynamics + Kalman smoother ----------
phi = np.sum(F_static[1:] * F_static[:-1]) / np.sum(F_static[:-1] ** 2)
resid_f = F_static[1:] - phi * F_static[:-1]
sigma_w2 = resid_f.var()

# idiosyncratic variances from static-factor regression residuals
idio_var = np.array([
    np.var(Z[:, i] - w[i] * F_static * f_scale / f_scale * (1 / w[i]) * w[i])  # placeholder, replaced below
    for i in range(N)
])
# simpler & correct: e_it = z_it - lam_i * F_t  where lam_i is the loading
# of z on F (regression coefficient of z_i on F_static)
lam_on_F = np.array([np.polyfit(F_static, Z[:, i], 1)[0] for i in range(N)])
idio_var = np.array([
    np.var(Z[:, i] - lam_on_F[i] * F_static) for i in range(N)
])

# Kalman filter (scalar state) over Z_t = lam_on_F * f_t + e_t, f_t = phi f_{t-1} + w_t
R = idio_var  # diagonal idiosyncratic variances
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
    # update using all 6 series (all observed every month in this panel)
    S = Lam * P_pred[t]                       # (N,) = Lam_i * P
    K_denom = (Lam ** 2) * P_pred[t] + R       # (N,)
    K = S / K_denom                            # Kalman gain per series
    # joint update using full vector (Gaussian, independent idio errors):
    prec = 1.0 / P_pred[t] + np.sum(Lam ** 2 / R)
    P_filt[t] = 1.0 / prec
    f_filt[t] = P_filt[t] * (f_pred[t] / P_pred[t] + np.sum(Lam * Z[t, :] / R))

# RTS smoother
f_smooth = np.zeros(T)
P_smooth = np.zeros(T)
f_smooth[-1], P_smooth[-1] = f_filt[-1], P_filt[-1]
for t in range(T - 2, -1, -1):
    J = phi * P_filt[t] / P_pred[t + 1]
    f_smooth[t] = f_filt[t] + J * (f_smooth[t + 1] - phi * f_filt[t])
    P_smooth[t] = P_filt[t] + J ** 2 * (P_smooth[t + 1] - P_pred[t + 1])

panel["factor_static"] = F_static
panel["factor_smoothed"] = f_smooth

# ---------- Bridge regression: quarterly GDP growth ~ factor ----------
Fq_static = pd.Series(F_static, index=panel.index).resample("QS").mean()
Fq_smooth = pd.Series(f_smooth, index=panel.index).resample("QS").mean()

common_idx = gdp.index.intersection(Fq_static.index)
y = gdp.loc[common_idx].values
x = Fq_static.loc[common_idx].values
Xd = np.column_stack([np.ones_like(x), x])
beta_hat, *_ = np.linalg.lstsq(Xd, y, rcond=None)
alpha, beta = beta_hat
fitted = alpha + beta * x
resid = y - fitted
r2 = 1 - resid.var() / y.var()
print(f"\nBridge regression: GDP_growth = {alpha:.2f} + {beta:.2f} * Factor   (R^2={r2:.2f})")

# ---------- Exact per-indicator decomposition ----------
# F_t = sum_i w_i * z_i,t  (Step-1 static factor is an exact linear combo)
# => quarterly Fq = sum_i w_i * mean_{t in q}(z_i,t)
# => fitted growth_q = alpha + beta * Fq = alpha + sum_i [beta * w_i * mean_q(z_i,t)]
Zdf = pd.DataFrame(Z, index=panel.index, columns=cols)
Zq = Zdf.resample("QS").mean().loc[common_idx]

contrib = pd.DataFrame(index=common_idx, columns=cols, dtype=float)
for i, c in enumerate(cols):
    contrib[c] = beta * w[i] * Zq[c].values

contrib["trend"] = alpha
contrib["fitted"] = contrib[cols].sum(axis=1) + alpha
contrib["actual"] = y
contrib["residual"] = contrib["actual"] - contrib["fitted"]

check = np.allclose(contrib["fitted"].values, fitted, atol=1e-8)
print(f"Decomposition additivity check (should be True): {check}")

contrib.to_csv("quarterly_decomposition.csv")
panel.to_csv("panel_with_factor.csv")

with open("model_params.txt", "w") as f:
    f.write(f"var_explained={var_explained}\nphi={phi}\nsigma_w2={sigma_w2}\n")
    f.write(f"alpha={alpha}\nbeta={beta}\nr2={r2}\n")
    f.write("weights=" + ",".join(f"{c}:{wi:.4f}" for c, wi in zip(cols, w)) + "\n")
    f.write("loadings_on_F=" + ",".join(f"{c}:{li:.4f}" for c, li in zip(cols, Lam)) + "\n")

print("\nLatest 6 quarters, decomposition (pp contribution to annualized GDP growth):")
print(contrib.tail(6).round(2))
