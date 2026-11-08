"""Reference values for tests/evaluation.rs: an independent NumPy/SciPy
translation of BeforeIT's dmtest.jl, mztest.jl, bias_ttest.jl, varx.jl and
correlation_utils.jl, applied to pseudo-random inputs that are stored with
the results.

Usage: python3 evaluation_reference.py <output file>
Needs numpy and scipy.
"""
import json
import sys

import numpy as np
from scipy import stats

rng = np.random.RandomState(20101)


def cov(x, y):
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1))


def newey_west(d, maxlag):
    n = len(d)
    gamma = [cov(d[: n - i], d[i:]) / n for i in range(maxlag + 1)]
    return gamma[0] + 2 * sum(gamma[1:])


def dmtest_modified(e1, e2, h):
    n = len(e1)
    d = np.abs(e1) ** 2 - np.abs(e2) ** 2
    var_d = newey_west(d, h - 1) if h > 1 else np.var(d, ddof=1)
    if var_d < 0:
        h = 1
        var_d = np.var(d, ddof=1)
    k = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    dm = (d.mean() / np.sqrt(var_d / n)) * k
    return [float(dm), float(2 * stats.norm.cdf(-abs(dm)))]


def bias_ttest(e, h):
    n = len(e)
    maxlag = min(h - 1, n - 2)
    var_d = newey_west(e, maxlag) if maxlag >= 1 else np.var(e, ddof=1)
    if var_d < 0:
        var_d = np.var(e, ddof=1)
    t = e.mean() / np.sqrt(var_d / n)
    return [float(t), float(2 * stats.norm.cdf(-abs(t)))]


def mztest(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    xxi = np.linalg.inv(X.T @ X)
    beta = xxi @ X.T @ y
    r = y - X @ beta
    s2 = np.sum(r**2) / (len(x) - 2)
    c = beta - np.array([0.0, 1.0])
    stat = c @ np.linalg.inv(s2 * xxi) @ c
    return [float(beta[0]), float(beta[1]), float(1 - stats.chi2.cdf(stat, 2))]


def hpfilter(y, lam=1600.0):
    T = len(y)
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i : i + 3] = [1.0, -2.0, 1.0]
    trend = np.linalg.solve(np.eye(T) + lam * D.T @ D, y)
    return trend, y - trend


def cross_correlation(x, y, maxlag):
    n = len(x)
    xc, yc = x - x.mean(), y - y.mean()
    sx, sy = np.sqrt(np.sum(xc**2) / n), np.sqrt(np.sum(yc**2) / n)
    out = []
    for lag in range(-maxlag, maxlag + 1):
        s = abs(lag)
        if s >= n:
            out.append(0.0)
        elif lag >= 0:
            out.append(np.sum(xc[: n - s] * yc[s:]) / n / (sx * sy))
        else:
            out.append(np.sum(xc[s:] * yc[: n - s]) / n / (sx * sy))
    return [float(v) for v in out]


def auto_correlation(x, lags):
    n = len(x)
    xc = x - x.mean()
    v = np.sum(xc**2) / n
    out = []
    for lag in lags:
        if lag == 0:
            out.append(1.0)
        elif lag >= n:
            out.append(0.0)
        else:
            out.append(float(np.sum(xc[: n - lag] * xc[lag:]) / n / v))
    return out


def var_coefficients(y, lags, x):
    """rfvar3 for full-rank regressors: least squares; rows of B are the
    regressors (lags of each variable, lag after lag, then x), columns the
    equations."""
    T = y.shape[0]
    X = [np.concatenate([y[t - l - 1] for l in range(lags)] + [x[t]]) for t in range(lags, T)]
    X = np.array(X)
    B = np.linalg.lstsq(X, y[lags:], rcond=None)[0]
    return B


def forecast_var(y, n, intercept, lags):
    x = np.ones((len(y), 1)) if intercept else np.zeros((len(y), 0))
    B = var_coefficients(y, lags, x)
    nv = y.shape[1]
    cur = [row for row in y]
    out = []
    for _ in range(n):
        lag = np.concatenate([cur[-1 - l] for l in range(lags)])
        nxt = B[: nv * lags].T @ lag
        if intercept:
            nxt = nxt + B[nv * lags]
        out.append(nxt)
        cur.append(nxt)
    return np.array(out)


def forecast_varx(y, exo, n, lags):
    T = len(y)
    x = np.column_stack([np.ones(T), exo[1 : T + 1]])
    B = var_coefficients(y, lags, x)
    nv = y.shape[1]
    cur = [row for row in y]
    out = []
    for i in range(1, n + 1):
        lag = np.concatenate([cur[-1 - l] for l in range(lags)])
        nxt = B[: nv * lags].T @ lag + B[nv * lags + 1 :].T @ exo[len(exo) - n + i - 1] + B[nv * lags]
        out.append(nxt)
        cur.append(nxt)
    return np.array(out)


def correlation_stats(gdp_cols, var_cols, L, H):
    xc, ac, vol = [], [], []
    for g, v in zip(gdp_cols, var_cols):
        _, gc = hpfilter(g)
        _, vc = hpfilter(v)
        m = min(len(gc), len(vc))
        xc.append(cross_correlation(gc[len(gc) - m :], vc[len(vc) - m :], L))
        ac.append(auto_correlation(vc, range(H + 1)))
        vol.append(float(np.std(v, ddof=1)))
    # rows: lags, columns: samples
    return {"crosscor": np.array(xc).T.tolist(), "autocor": np.array(ac).T.tolist(), "volatility": vol}


out = {}

e1 = rng.normal(0.0, 1.0, 40)
e2 = rng.normal(0.2, 1.3, 40)
out["dm"] = {"e1": e1.tolist(), "e2": e2.tolist(), "results": {str(h): dmtest_modified(e1, e2, h) for h in [1, 2, 4, 8, 12]}}

errors = rng.normal(0.3, 1.0, 30)
out["bias"] = {"errors": errors.tolist(), "results": {str(h): bias_ttest(errors, h) for h in [1, 2, 4, 8, 12]}}

xf = rng.normal(0.0, 1.0, 50)
yv = 0.3 + 0.8 * xf + rng.normal(0.0, 0.5, 50)
out["mz"] = {"y": yv.tolist(), "x": xf.tolist(), "result": mztest(yv, xf)}
yv2 = xf + rng.normal(0.0, 0.5, 50)
out["mz_unbiased"] = {"y": yv2.tolist(), "x": xf.tolist(), "result": mztest(yv2, xf)}

walk = np.cumsum(rng.normal(0.5, 1.0, 48)) + 100.0
trend, cycle = hpfilter(walk)
out["hp"] = {"y": walk.tolist(), "trend": trend.tolist(), "cycle": cycle.tolist()}

a = rng.normal(0.0, 1.0, 50).cumsum()
b = a + rng.normal(0.0, 1.0, 50)
out["corr"] = {
    "x": a.tolist(),
    "y": b.tolist(),
    "cross": cross_correlation(a, b, 5),
    "auto": auto_correlation(a, range(9)),
}

T = 60
gdp = np.cumsum(rng.normal(0.4, 1.0, T)) + 50.0
wages = 0.6 * gdp + rng.normal(0.0, 1.0, T)
sims_gdp = np.column_stack([np.cumsum(rng.normal(0.4, 1.0, T)) + 50.0 for _ in range(3)])
sims_wages = 0.6 * sims_gdp + rng.normal(0.0, 1.0, (T, 3))
out["corr_stats"] = {
    "gdp": gdp.tolist(),
    "wages": wages.tolist(),
    "data": correlation_stats([gdp], [wages], 5, 8),
    "sims_gdp": sims_gdp.tolist(),
    "sims_wages": sims_wages.tolist(),
    "sims": correlation_stats(list(sims_gdp.T), list(sims_wages.T), 5, 8),
}

nv = 2
y = np.zeros((T, nv))
y[0] = [1.0, 2.0]
y[1] = [1.5, 1.0]
A1 = np.array([[0.5, 0.1], [-0.2, 0.4]])
A2 = np.array([[0.1, 0.0], [0.05, -0.1]])
for t in range(2, T):
    y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + np.array([0.2, -0.1]) + rng.normal(0.0, 0.3, nv)
exo = rng.normal(0.0, 1.0, (T + 9, 1))
out["var"] = {
    "y": y.tolist(),
    "forecasts": {
        f"{intercept}_{lags}": forecast_var(y, 8, intercept, lags).tolist()
        for intercept in [True, False]
        for lags in [1, 2]
    },
    "exo": exo.tolist(),
    "varx": forecast_varx(y, exo, 8, 1).tolist(),
}

with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, separators=(",", ":"), sort_keys=True)
    fh.write("\n")
