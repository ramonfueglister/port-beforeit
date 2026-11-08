"""Reference values for tests/analysis.rs: an independent NumPy/SciPy
translation of BeforeIT's error_tables.jl and analysis_utils.jl
(error_table_ar, error_table_abm, error_table_validation_var,
error_table_validation_abm and the tables they write), applied to the
predictions that tests/analysis.rs writes for 2010Q1 to 2011Q4 of
ITALY_CALIBRATION. The output holds the series of the predictions that the
tables use and the text of every table.

Usage, from the root of the crate:
    cargo test --test analysis   # writes target/tmp/analysis
    python3 tests/data/analysis_reference.py target/tmp/analysis \
        data/italy/calibration.json tests/data/analysis_reference.json
Needs numpy and scipy.
"""
import datetime
import json
import math
import os
import sys
from decimal import Decimal

import numpy as np
from scipy import stats

folder, calibration_file, output = sys.argv[1:4]

VARIANTS = ["Model", "ModelGR"]
HORIZONS = [1, 2, 4, 8, 12]
BASE = ["Real GDP", "GDP Deflator Growth", "Real Consumption", "Real Investment", "Euribor"]
VALIDATION = [
    "Real GDP", "GDP Deflator Growth", "Real Gov Consumption", "Real Exports",
    "Real Imports", "Real GDP (EA)", "GDP Deflator Growth (EA)", "Euribor",
]
FIRST, LAST = datetime.date(2010, 3, 31), datetime.date(2011, 12, 31)


def quarterly_rate(x):
    return (1 + x) ** (1 / 4)


def log1(x):
    return math.log(1 + x)


TRAINING_SPECS = [
    ("real_gdp_quarterly", math.log, "data"),
    ("gdp_deflator_growth_quarterly", log1, "data"),
    ("real_household_consumption_quarterly", math.log, "data"),
    ("real_fixed_capitalformation_quarterly", math.log, "data"),
    ("euribor", quarterly_rate, "data"),
]
VALIDATION_SPECS = [
    ("real_gdp_quarterly", math.log, "data"),
    ("gdp_deflator_growth_quarterly", log1, "data"),
    ("real_government_consumption_quarterly", math.log, "data"),
    ("real_exports_quarterly", math.log, "data"),
    ("real_imports_quarterly", math.log, "data"),
    ("real_gdp_quarterly", math.log, "ea"),
    ("gdp_deflator_growth_quarterly", log1, "ea"),
    ("euribor", quarterly_rate, "data"),
]
PREDICTED_KEYS = ["quarters_num"] + sorted(
    {k for k, _, s in TRAINING_SPECS + VALIDATION_SPECS if s == "data"}
    | {k.replace("_quarterly", "_ea_quarterly") for k, _, s in VALIDATION_SPECS if s == "ea"}
)

# --- dates: MATLAB date numbers ------------------------------------------------


def datenum(d):
    return float(d.toordinal() + 366)


def num2date(n):
    return datetime.date.fromordinal(int(n) - 366)


def add_months(d, m):
    k = d.month - 1 + m
    y, mo = d.year + k // 12, k % 12 + 1
    day = d.day
    while True:
        try:
            return datetime.date(y, mo, day)
        except ValueError:
            day -= 1


def last_day_of_month(d):
    return add_months(datetime.date(d.year, d.month, 1), 1) - datetime.timedelta(days=1)


def forecast_quarter(qn, h):
    return datenum(last_day_of_month(add_months(num2date(qn), 3 * h)))


def quarters():
    out, k = [], 0
    while add_months(FIRST, 3 * k) <= LAST:
        out.append(add_months(FIRST, 3 * k))
        k += 1
    return out


# --- Julia's printing and rounding ----------------------------------------------


def julia_string(x):
    """Julia string(x::Float64): shortest digits, Ryu's layout."""
    x = float(x)
    if x == 0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    _, digits, exp = Decimal(repr(abs(x))).as_tuple()
    digits = "".join(map(str, digits))
    stripped = digits.rstrip("0")
    exp += len(digits) - len(stripped)
    digits = stripped
    olength = len(digits)
    pt = exp + olength
    sign = "-" if x < 0 else ""
    if -4 < pt <= 6:
        if pt <= 0:
            s = "0." + "0" * (-pt) + digits
        elif pt >= olength:
            s = digits + "0" * (pt - olength) + ".0"
        else:
            s = digits[:pt] + "." + digits[pt:]
    else:
        s = digits[0] + "." + (digits[1:] if olength > 1 else "0") + "e" + str(pt - 1)
    return sign + s


def julia_round(x, d):
    """Julia round(x; digits = d): round(x * 10^d) / 10^d, ties to even."""
    if not math.isfinite(x):
        return x
    y = float(np.round(x * 10.0**d)) / 10.0**d
    return y if math.isfinite(y) else x


# --- tests (as in evaluation_reference.py, with Julia's n < 2 case) --------------


def cov(x, y):
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1))


def newey_west(d, maxlag):
    n = len(d)
    gamma = [cov(d[: n - i], d[i:]) / n for i in range(maxlag + 1)]
    return gamma[0] + 2 * sum(gamma[1:])


def dmtest_modified(e1, e2, h):
    n = len(e1)
    assert h <= n
    d = np.abs(e1) ** 2 - np.abs(e2) ** 2
    var_d = newey_west(d, h - 1) if h > 1 else np.var(d, ddof=1)
    if var_d < 0:
        h = 1
        var_d = np.var(d, ddof=1)
    k = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    dm = (d.mean() / np.sqrt(var_d / n)) * k
    return float(2 * stats.norm.cdf(-abs(dm)))


def bias_ttest(e, h):
    n = len(e)
    if n < 2:
        return math.nan
    maxlag = min(h - 1, n - 2)
    var_d = newey_west(e, maxlag) if maxlag >= 1 else np.var(e, ddof=1)
    if var_d < 0:
        var_d = np.var(e, ddof=1)
    t = e.mean() / np.sqrt(var_d / n)
    return float(2 * stats.norm.cdf(-abs(t)))


# --- benchmarks: least squares on the lags and a constant -----------------------


def var_forecast(y, h, lags):
    """VAR with intercept on the rows of y (rfvar3 for full-rank regressors)."""
    T, nv = y.shape
    X = np.array([np.concatenate([y[t - l - 1] for l in range(lags)] + [[1.0]]) for t in range(lags, T)])
    B = np.linalg.lstsq(X, y[lags:], rcond=None)[0]
    cur = list(y)
    out = []
    for _ in range(h):
        lag = np.concatenate([cur[-1 - l] for l in range(lags)])
        nxt = B[: nv * lags].T @ lag + B[nv * lags]
        out.append(nxt)
        cur.append(nxt)
    return np.array(out)


# --- tables ---------------------------------------------------------------------

tables = {}
saved = {}


def latex(variant, name, cells):
    lines = []
    for row, c in enumerate(cells):
        line = " & ".join([f"{HORIZONS[row]}q"] + list(c))
        lines.append(line + (" \\\\ " if row < len(cells) - 1 else ""))
    tables[f"{variant}/{name}"] = "".join(line + "\n" for line in lines)


def csv(variant, name, m, names):
    text = ",".join(["Horizon"] + names) + "\n"
    for row, h in enumerate(HORIZONS):
        text += ",".join([f"{h}q"] + [julia_string(m[row][l]) for l in range(len(names))]) + "\n"
    tables[f"{variant}/{name}"] = text


def nanmean(v):
    v = v[~np.isnan(v)]
    return float(np.mean(v)) if len(v) else math.nan


def forecast_errors(F, A):
    E = F - A
    nh, nv = E.shape[1:]
    rmse = [[100 * math.sqrt(nanmean(E[:, j, l] ** 2)) for l in range(nv)] for j in range(nh)]
    bias = [[nanmean(E[:, j, l]) for l in range(nv)] for j in range(nh)]
    return rmse, bias, E


def rounded(m, d):
    return [[julia_round(x, d) for x in r] for r in m]


def improvement(r1, r2):
    return [[-julia_round(100 * (a - b) / b, 1) for a, b in zip(x, y)] for x, y in zip(r1, r2)]


def dm_comparison(E1, E2, r1, r2):
    imp = improvement(r1, r2)
    cells = [[None] * len(imp[0]) for _ in imp]
    pvals = [[math.nan] * len(imp[0]) for _ in imp]
    for j, h in enumerate(HORIZONS):
        for l in range(len(imp[0])):
            e1, e2 = E1[:, j, l], E2[:, j, l]
            mask = ~np.isnan(e1) & ~np.isnan(e2)
            p = math.nan if mask.sum() < 2 else dmtest_modified(e2[mask], e1[mask], h)
            cells[j][l] = f"{julia_string(imp[j][l])}({julia_string(julia_round(p, 2))}, {stars(p)})"
            pvals[j][l] = p
    return cells, pvals, imp


def bias_cells(E, bias):
    b = rounded(bias, 4)
    cells = [[None] * len(b[0]) for _ in b]
    pvals = [[math.nan] * len(b[0]) for _ in b]
    for j, h in enumerate(HORIZONS):
        for l in range(len(b[0])):
            e = E[:, j, l]
            p = bias_ttest(e[~np.isnan(e)], h)
            cells[j][l] = f"{julia_string(b[j][l])} ({julia_string(julia_round(p, 3))}, {stars(p)})"
            pvals[j][l] = p
    return cells, pvals, b


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def tables_var(F, A, ftype, mtype, names, k, variant):
    name = ("validation_" if ftype == "validation" else "") + mtype
    if k == 1:
        saved[(variant, name)] = F
        rmse, bias, E = forecast_errors(F, A)
        r = rounded(rmse, 2)
        latex(variant, f"rmse_{name}.tex", [[julia_string(x) for x in row] for row in r])
        csv(variant, f"rmse_{name}.csv", r, names)
        cells, pvals, b = bias_cells(E, bias)
        latex(variant, f"bias_{name}.tex", cells)
        csv(variant, f"bias_{name}.csv", b, names)
        csv(variant, f"pval_bias_{name}.csv", pvals, names)
    else:
        rmse_k, _, E_k = forecast_errors(F, A)
        rmse_1, _, E_1 = forecast_errors(saved[(variant, name)], A)
        cells, pvals, imp = dm_comparison(E_k, E_1, rmse_k, rmse_1)
        latex(variant, f"rmse_{name}_{k}.tex", cells)
        csv(variant, f"rmse_{name}_{k}.csv", imp, names)
        csv(variant, f"pval_{name}_{k}.csv", pvals, names)


def tables_abm(F, A, ftype, names, variant):
    prefix = "validation_" if ftype == "validation" else ""
    benchmark = "var" if ftype == "validation" else "ar"
    saved[(variant, prefix + "abm")] = F
    rmse, bias, E = forecast_errors(F, A)
    r = rounded(rmse, 2)
    csv(variant, f"rmse_{prefix}abm.csv", r, names)
    latex(variant, f"rmse_{prefix}abm.tex", [[julia_string(x) for x in row] for row in r])
    comparisons = [(benchmark, saved[(variant, prefix + benchmark)])]
    if variant != "Model":
        comparisons.append(("base", saved[("Model", prefix + "abm")]))
    for other, F_other in comparisons:
        rmse_o, _, E_o = forecast_errors(F_other, A)
        cells, pvals, imp = dm_comparison(E, E_o, rmse, rmse_o)
        latex(variant, f"rmse_{prefix}abm_vs_{other}.tex", cells)
        csv(variant, f"rmse_{prefix}abm_vs_{other}.csv", imp, names)
        csv(variant, f"pval_{prefix}abm_vs_{other}.csv", pvals, names)
    cells, pvals, b = bias_cells(E, bias)
    latex(variant, f"bias_{prefix}abm.tex", cells)
    csv(variant, f"bias_{prefix}abm.csv", b, names)
    csv(variant, f"pval_bias_{prefix}abm.csv", pvals, names)


# --- data -----------------------------------------------------------------------


def column(t, key):
    return np.array([np.nan if v is None else v for v in t[key]["data"]], dtype=float)


with open(calibration_file) as fh:
    cal = json.load(fh)
data = {k: column(cal["data"], k) for k in cal["data"] if isinstance(cal["data"][k], dict)}
ea = {k: column(cal["ea"], k) for k in cal["ea"] if isinstance(cal["ea"][k], dict)}
# BeforeIT's calibration overwrites the GDP deflator of the data.
data["gdp_deflator_quarterly"] = data["nominal_gdp_quarterly"] / data["real_gdp_quarterly"]
dq = data["quarters_num"]


def actual_values(fq, specs):
    mask = dq == fq
    out = []
    for key, f, source in specs:
        raw = (ea if source == "ea" else data)[key][mask]
        assert len(raw) == 1
        out.append(f(float(raw[0])))
    return out


def loop(qs, fn):
    max_year = qs[-1].year
    for i, q in enumerate(qs):
        qn = datenum(q)
        for j, h in enumerate(HORIZONS):
            fq = forecast_quarter(qn, h)
            if num2date(fq) > datetime.date(max_year, 12, 31):
                break
            if not np.any(dq == fq):
                continue
            fn(i, j, qn, h, fq)


def error_table_ar(variant, qs):
    for k in range(1, 4):
        F = np.full((len(qs), len(HORIZONS), 5), np.nan)
        A = F.copy()

        def fn(i, j, qn, h, fq):
            A[i, j] = actual_values(fq, TRAINING_SPECS)
            m = dq <= qn
            Y0 = np.column_stack([
                np.log(data["real_gdp_quarterly"][m]),
                np.log(data["gdp_deflator_quarterly"][m]),
                np.log(data["real_household_consumption_quarterly"][m]),
                np.log(data["real_fixed_capitalformation_quarterly"][m]),
                np.cumsum((1 + data["euribor"][m]) ** (1 / 4)),
            ])
            D = np.diff(Y0[4 - k - 1:], axis=0)
            Y = np.column_stack([var_forecast(D[:, [l]], h, k)[:, 0] for l in range(5)])
            last = Y[-1].copy()
            for l in [0, 2, 3]:
                last[l] = Y0[-1, l] + Y[:, l].sum()
            F[i, j] = last

        loop(qs, fn)
        tables_var(F, A, "training", "ar", BASE, k, variant)


def error_table_validation_var(variant, qs):
    for k in range(1, 4):
        F = np.full((len(qs), len(HORIZONS), 8), np.nan)
        A = F.copy()

        def fn(i, j, qn, h, fq):
            A[i, j] = actual_values(fq, VALIDATION_SPECS)
            m = dq <= qn
            Y0 = np.column_stack([
                np.log(data["real_gdp_quarterly"][m]),
                np.log(1 + data["gdp_deflator_growth_quarterly"][m]),
                np.log(data["real_government_consumption_quarterly"][m]),
                np.log(data["real_exports_quarterly"][m]),
                np.log(data["real_imports_quarterly"][m]),
                np.log(ea["real_gdp_quarterly"][m]),
                np.log(1 + ea["gdp_deflator_growth_quarterly"][m]),
                np.cumsum((1 + data["euribor"][m]) ** (1 / 4)),
            ])
            D = np.diff(Y0[4 - k:], axis=0)
            Y = var_forecast(D, h, k)
            last = Y[-1].copy()
            for l in [0, 2, 3, 4, 5]:
                last[l] = Y0[-1, l] + Y[:, l].sum()
            F[i, j] = last

        loop(qs, fn)
        tables_var(F, A, "validation", "var", VALIDATION, k, variant)


def load_predictions(path):
    with open(path) as fh:
        t = json.load(fh)
    out = {}
    for key, v in t.items():
        a = np.array([np.nan if x == "NaN" else x for x in v["data"]], dtype=float)
        out[key] = a.reshape(v["size"][::-1]).T if len(v["size"]) == 2 else a
    return out


predictions = {}


def error_table_abm(variant, qs, specs, names, ftype):
    files = {q: os.path.join(folder, "data", "it", "abm_predictions", variant, f"{q.year}Q{(q.month - 1) // 3 + 1}.json") for q in qs}
    F = np.full((len(qs), len(HORIZONS), len(names)), np.nan)
    A = F.copy()
    models = {q: load_predictions(f) for q, f in files.items()}
    for q, m in models.items():
        predictions[f"{variant}/{q.year}Q{(q.month - 1) // 3 + 1}"] = {
            k: {"size": list(m[k].shape), "data": m[k].T.ravel().tolist()} for k in PREDICTED_KEYS
        }

    def fn(i, j, qn, h, fq):
        model = models[qs[i]]
        A[i, j] = actual_values(fq, specs)
        row = model["quarters_num"] == fq
        vals = []
        for key, f, source in specs:
            k = key.replace("_quarterly", "_ea_quarterly") if source == "ea" else key
            vals.append(f(float(np.mean(model[k][row, :]))))
        F[i, j] = vals

    loop(qs, fn)
    tables_abm(F, A, ftype, names, variant)


qs = quarters()
for variant in VARIANTS:
    error_table_ar(variant, qs)
    error_table_abm(variant, qs, TRAINING_SPECS, BASE, "training")
    error_table_validation_var(variant, qs)
    error_table_abm(variant, qs, VALIDATION_SPECS, VALIDATION, "validation")

with open(output, "w") as fh:
    json.dump({"predictions": predictions, "tables": tables}, fh, separators=(",", ":"), sort_keys=True)
    fh.write("\n")
