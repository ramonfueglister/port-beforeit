"""Independent NumPy translation of `get_params_and_initial_conditions` in
BeforeIT.jl `src/utils/calibration.jl` (commit 09263989b56bd2d58c28657bb8e940142d995cf4),
written from the Julia source for the tests of this port. It shares no code
with the Rust port: the tests compare the two, which checks that both read
the Julia code the same way, not that the Julia code is right.

The regressions on two columns follow BeforeIT's `rfvar3` with the analytic
`svd_mx2` of `src/utils/julia_svd.jl`, as Julia does; the Taylor rule
regression (three columns) uses `numpy.linalg.lstsq`.

Usage:
  python3 calibration_reference.py <calibration json> <YYYY-MM-DD> <scale> <use_growth_rate_ar1 0/1> <output json>

<calibration json> is a file written by `data/convert_calibration.py`. The
output holds "parameters", "initial_conditions" (null for NaN) and
"warnings", the messages of Julia's `@warn` in order. Needs numpy.
"""
import datetime, json, math, sys
import numpy as np

src, date_s, scale, growth, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
co = json.load(open(src))


def arr(v):
    """Julia array: (size, column-major entries with None for missing)."""
    if isinstance(v, dict):
        conv = []
        for x in v["data"]:
            if x is None:
                conv.append(None)
            elif isinstance(x, str):
                conv.append({"NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}[x])
            else:
                conv.append(float(x))
        return v["size"], conv
    return [], [float(v)]


class D:
    def __init__(self, t):
        self.t = {k: arr(v) for k, v in t.items()}

    def has(self, k):
        return k in self.t

    def size(self, k, d):
        s = self.t[k][0]
        return s[d - 1] if d <= len(s) else 1

    def lin(self, k, i):  # 1-based linear index, error on missing
        x = self.t[k][1][i - 1]
        assert i >= 1
        if x is None:
            raise ValueError(f"{k}[{i}] missing")
        return x

    def vec(self, k):
        return self.t[k][1]

    def rng(self, k, a, b):
        return np.array([self.lin(k, i) for i in range(a, b + 1)])

    def col(self, k, t):
        r = self.size(k, 1)
        return self.rng(k, (t - 1) * r + 1, t * r)

    def slab(self, k, t):
        r, c = self.size(k, 1), self.size(k, 2)
        flat = self.rng(k, (t - 1) * r * c + 1, t * r * c)
        return flat.reshape((c, r)).T  # column-major


def date2num(d):
    return d.toordinal() + 366.0


def findfirst(vals, x):
    for i, v in enumerate(vals):
        if v == x:
            return i + 1
    raise ValueError(f"{x} not found")


def has_quarterly(d, key, idx):
    return d.has(key) and len(d.vec(key)) >= idx and d.vec(key)[idx - 1] is not None


def matlab_round(x):
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def jmax(a, b):
    return np.maximum(a, b)


def pos(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), 0.0, np.where(x > 0, x, 0.0))


def neg(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), 0.0, np.where(x < 0, x, 0.0))


def ols_ar1(y):
    # rfvar3 with the analytic svd_mx2 of BeforeIT src/utils/julia_svd.jl,
    # which Julia uses for a regression on two columns.
    y = np.asarray(y, dtype=float)
    Y = y[1:]
    X = np.column_stack([y[:-1], np.ones(len(Y))])
    c1, c2 = X[:, 0], X[:, 1]
    a, b, g = c1 @ c1, c2 @ c2, c1 @ c2
    th = 0.5 * math.atan2(2 * g, a - b)
    c, s = math.cos(th), math.sin(th)
    V = np.array([[c, -s], [s, c]])
    W = X @ V
    S = np.array([np.linalg.norm(W[:, 0]), np.linalg.norm(W[:, 1])])
    U = W / S
    keep = S > 100 * np.finfo(float).eps
    B = (V[:, keep] / S[keep]) @ (U[:, keep].T @ Y)
    u = Y - X @ B
    sigma = math.sqrt(np.var(u, ddof=1))
    return B[0], B[1], sigma, u


def taylor(r, p, g):
    r, p, g = map(np.asarray, (r, p, g))
    Y = r[1:]
    X = np.column_stack([r[:-1], p[1:], g[1:]])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    rho = b[0]
    xi_pi = b[1] / (1 - rho)
    xi_gamma = b[2] / (1 - rho)
    pi_star = (0.02 + 1) ** (1 / 4) - 1
    r_star = pi_star * (xi_pi - 1)
    return rho, r_star, xi_pi, xi_gamma, pi_star


cd, figaro, data, ea = D(co["calibration"]), D(co["figaro"]), D(co["data"]), D(co["ea"])
max_cal = datetime.date.fromisoformat(co["max_calibration_date"])
est = datetime.date.fromisoformat(co["estimation_date"])
cal = datetime.date.fromisoformat(date_s)
warnings = []

T_calibration = findfirst(cd.vec("years_num"), date2num(datetime.date(min(cal, max_cal).year, 12, 31)))
T_calibration_quarterly = findfirst(cd.vec("quarters_num"), date2num(cal))
T_estimation_exo = findfirst(data.vec("quarters_num"), date2num(est))
T_calibration_exo = findfirst(data.vec("quarters_num"), date2num(cal))
T_calibration_exo_max = len(data.vec("quarters_num"))
Tc, Tq, Te, Tx = T_calibration, T_calibration_quarterly, T_estimation_exo, T_calibration_exo


def defl(t, a, b):
    return t.rng("nominal_gdp_quarterly", a, b) / t.rng("real_gdp_quarterly", a, b)


intermediate_consumption = figaro.slab("intermediate_consumption", Tc)
G = intermediate_consumption.shape[0]
S = G
household_consumption = figaro.col("household_consumption", Tc)
fixed_capitalformation = figaro.col("fixed_capitalformation", Tc)
exports = figaro.col("exports", Tc)
compensation_employees = figaro.col("compensation_employees", Tc)
household_cash_quarterly = cd.lin("household_cash_quarterly", Tq)
property_income = cd.lin("property_income", Tc)
mixed_income = cd.lin("mixed_income", Tc)
operating_surplus = figaro.col("operating_surplus", Tc)
firm_cash_quarterly = cd.lin("firm_cash_quarterly", Tq)
firm_debt_quarterly = cd.lin("firm_debt_quarterly", Tq)
if has_quarterly(cd, "firm_debt_consolidation_ratio_quarterly", Tq):
    firm_debt_quarterly *= cd.lin("firm_debt_consolidation_ratio_quarterly", Tq)
hq_fi = has_quarterly(cd, "firm_interest_quarterly", Tq)
hq_gi = has_quarterly(cd, "interest_government_debt_quarterly", Tq)
firm_interest_quarterly = cd.lin("firm_interest_quarterly", Tq) if hq_fi else cd.lin("firm_interest", Tc)
government_debt_quarterly = cd.lin("government_debt_quarterly", Tq)
if hq_gi:
    igdq = cd.lin("interest_government_debt_quarterly", Tq)
else:
    warnings.append("Using annual 'interest_government_debt' - will apply timescale conversion")
    igdq = cd.lin("interest_government_debt", Tc)
government_consumption = figaro.col("government_consumption", Tc)
social_benefits = cd.lin("social_benefits", Tc)
unemployment_benefits = cd.lin("unemployment_benefits", Tc)
pension_benefits = cd.lin("pension_benefits", Tc)
corporate_tax = cd.lin("corporate_tax", Tc)
has_sw = cd.has("wages_by_sector") and cd.size("wages_by_sector", 2) >= Tc
if has_sw:
    wages_by_sector = cd.col("wages_by_sector", Tc)
else:
    warnings.append("wages_by_sector not available, estimating from aggregate wages")
    wages_scalar = cd.lin("wages", Tc)
    wages_by_sector = wages_scalar * (compensation_employees / compensation_employees.sum())
tph = figaro.lin("taxes_products_household", Tc)
social_contributions = cd.lin("social_contributions", Tc)
income_tax = cd.lin("income_tax", Tc)
capital_taxes = cd.lin("capital_taxes", Tc)
tpfcf = figaro.lin("taxes_products_capitalformation", Tc)
taxes_production = figaro.col("taxes_production", Tc)
tpg = figaro.lin("taxes_products_government", Tc)
bank_equity_quarterly = cd.lin("bank_equity_quarterly", Tq)
taxes_products = np.zeros(G)
hq_gd = has_quarterly(cd, "government_deficit_quarterly", Tq)
if hq_gd:
    gdq = cd.lin("government_deficit_quarterly", Tq)
else:
    warnings.append("Using annual 'government_deficit' - will NOT apply timescale conversion")
gda = cd.lin("government_deficit", Tc)
firms = cd.col("firms", Tc)
employees = cd.col("employees", Tc)
population = cd.lin("population", Tc)
r_bar = (data.lin("euribor", Tx) + 1.0) ** (1.0 / 4.0) - 1
omega = 0.85

intermediate_consumption = np.maximum(0, intermediate_consumption)
output = intermediate_consumption.sum(axis=0) + taxes_products + taxes_production + compensation_employees + operating_surplus

fa1 = cd.size("fixed_assets", 1)
dw1 = cd.size("dwellings", 1)
if fa1 == (G & dw1) and (G & dw1) == G:
    fixed_assets = cd.col("fixed_assets", Tc)
    dwellings = cd.col("dwellings", Tc)
    fao = fixed_assets - dwellings
else:
    fixed_assets = cd.lin("fixed_assets", Tc)
    dwellings = cd.lin("dwellings", Tc)
    w = (cd.col("fixed_assets_eu7", Tc) - cd.col("dwellings_eu7", Tc)) / cd.col("nominal_nace64_output_eu7", Tc) * output
    fao = (fixed_assets - dwellings) * w / w.sum()

if cd.size("capital_consumption", 1) == len(compensation_employees):
    capital_consumption = cd.col("capital_consumption", Tc)
else:
    capital_consumption = cd.col("nace64_capital_consumption", Tc) / cd.col("nominal_nace64_output", Tc) * output

urq = data.lin("unemployment_rate_quarterly", Tx)
tpe = 0.0
if has_sw:
    esc = compensation_employees - wages_by_sector
else:
    sesc = min(social_contributions, compensation_employees.sum() - wages_scalar)
    esc = sesc * (compensation_employees / compensation_employees.sum())
fixed_capitalformation = pos(fixed_capitalformation)
gcfd = cd.lin("gross_capitalformation_dwellings", Tc)
tpcfd = gcfd * (1 - 1 / (1 + tpfcf / fixed_capitalformation.sum()))
timescale = data.lin("nominal_gdp_quarterly", Tx) / (
    (compensation_employees + operating_surplus + taxes_production + taxes_products).sum() + tph + tpcfd + tpg + tpe
)
if not hq_fi:
    firm_interest_quarterly = timescale * firm_interest_quarterly
if not hq_gi:
    igdq = timescale * igdq
cfd = (gcfd - tpcfd) * fixed_capitalformation / fixed_capitalformation.sum()
fcfo = fixed_capitalformation - cfd
exports = pos(exports)
net_imports = (intermediate_consumption.sum(axis=1) + household_consumption + government_consumption
               + fcfo * capital_consumption.sum() / fcfo.sum() + cfd + exports - output)
imports = pos(net_imports)
reexports = neg(net_imports)
hsc = social_contributions - esc.sum()
wages = wages_by_sector if has_sw else compensation_employees * (1 - esc.sum() / compensation_employees.sum())
hit = income_tax - corporate_tax
gdt = gdq / timescale if hq_gd else gda
ont = (tph + tpcfd + tpe + taxes_products.sum() + taxes_production.sum() + esc.sum() + hsc + hit + corporate_tax
       + capital_taxes - social_benefits - government_consumption.sum() - igdq / timescale - gdt)
di = wages.sum() + mixed_income + property_income + social_benefits + ont - hsc - hit - capital_taxes
if cd.has("unemployed_census"):
    unemployed = cd.lin("unemployed_census", 1)
else:
    unemployed = matlab_round((urq * employees.sum()) / (1 - urq))
if cd.has("inactive_census"):
    inactive = cd.lin("inactive_census", 1)
else:
    inactive = population - np.maximum(np.maximum(1, firms), employees).sum() - unemployed - np.maximum(1, firms).sum() - 1

firms = np.maximum(1, np.array([matlab_round(scale * f) for f in firms]))
employees = np.maximum(firms, np.array([matlab_round(scale * e) for e in employees]))
inactive = max(1, matlab_round(scale * inactive))
unemployed = max(1, matlab_round(scale * unemployed))

with np.errstate(divide="ignore", invalid="ignore"):
    alpha_s = timescale * output / employees
    beta_s = output / intermediate_consumption.sum(axis=0)
    kappa_s = timescale * output / fao / omega
    delta_s = np.nan_to_num(timescale * capital_consumption / fao / omega, nan=0.0, posinf=np.inf, neginf=-np.inf)
    w_s = timescale * wages / employees
    nz = lambda v: np.where(np.isnan(v), 0.0, v)
    alpha_s, kappa_s, w_s, beta_s = nz(alpha_s), nz(kappa_s), nz(w_s), nz(beta_s)
    alpha_s = np.maximum(alpha_s, 1e-6)
    kappa_s = np.maximum(kappa_s, 1e-6)
    tau_Y_s = nz(taxes_products / output)
    tau_K_s = nz(taxes_production / output)
    b_CF_g = fcfo / fcfo.sum()
    b_CFH_g = cfd / cfd.sum()
    b_HH_g = household_consumption / household_consumption.sum()
    a_sg = nz(intermediate_consumption / intermediate_consumption.sum(axis=0, keepdims=True))
    c_G_g = government_consumption / government_consumption.sum()
    c_E_g = (exports - reexports) / (exports - reexports).sum()
    c_I_g = imports / imports.sum()

T_prime = Tx - Te + 1
T = 12
T_max = T - max(0, Tx + T - T_calibration_exo_max)
H_act = employees.sum() + unemployed + firms.sum() + 1
H_inact = inactive
J = matlab_round(firms.sum() / 4)
L = matlab_round(firms.sum() / 2)
mu = firm_interest_quarterly / firm_debt_quarterly - r_bar
tau_INC = (hit + capital_taxes) / (wages.sum() + property_income + mixed_income - hsc)
tau_SIF = esc.sum() / wages.sum()
Y_s = timescale * output
with np.errstate(divide="ignore", invalid="ignore"):
    pi_bar_s = 1 - (1 + tau_SIF) * w_s / alpha_s - delta_s / kappa_s - 1 / beta_s - tau_K_s - tau_Y_s
pi_bar_s = np.where(np.isfinite(pi_bar_s), pi_bar_s, 0.0)
mps = pi_bar_s * Y_s
ps = pos(mps).sum()
D_s = pos(mps) / ps if ps > 0 else np.zeros(G)
L_s = fao / fao.sum()
pais = mps - (r_bar + mu) * firm_debt_quarterly * L_s + r_bar * firm_cash_quarterly * D_s
tau_FIRM = timescale * corporate_tax / (pos(pais).sum() + firm_interest_quarterly - r_bar * (firm_debt_quarterly - bank_equity_quarterly))
tau_VAT = tph / household_consumption.sum()
tau_SIW = hsc / wages.sum()
tau_EXPORT = tpe / (exports - reexports).sum()
tau_CF = tpcfd / cfd.sum()
tau_G = tpg / government_consumption.sum()
psi = (household_consumption.sum() + tph) / di
psi_H = (cfd.sum() + tpcfd) / di
theta_DIV = timescale * (mixed_income + property_income) / (
    pos(pais).sum() + firm_interest_quarterly - r_bar * (firm_debt_quarterly - bank_equity_quarterly) - timescale * corporate_tax)
r_G = igdq / government_debt_quarterly
theta_UB = 0.55 * (1 - tau_INC) * (1 - tau_SIW)

ea_def = defl(ea, Te - 1, Tx)
alpha_pi_EA, beta_pi_EA, sigma_pi_EA, _ = ols_ar1(np.diff(np.log(ea_def)))
a1 = (data.rng("euribor", Te, Tx) + 1) ** (1 / 4) - 1
a2 = np.exp(np.diff(np.log(ea_def))) - 1
a3 = np.exp(np.diff(np.log(ea.rng("real_gdp_quarterly", Te - 1, Tx)))) - 1
rho, r_star, xi_pi, xi_gamma, pi_star = taylor(a1, a2, a3)
G_lev = timescale * government_consumption.sum() * data.rng("real_government_consumption_quarterly", Te, Tx) / data.lin("real_government_consumption_quarterly", Tx)
E_lev = timescale * (exports - reexports).sum() * data.rng("real_exports_quarterly", Te, Tx) / data.lin("real_exports_quarterly", Tx)
I_lev = timescale * imports.sum() * data.rng("real_imports_quarterly", Te, Tx) / data.lin("real_imports_quarterly", Tx)
Y_EA_raw = ea.rng("real_gdp_quarterly", Te, Tx)
alpha_Y_EA, beta_Y_EA, sigma_Y_EA, eY = ols_ar1(np.log(Y_EA_raw))
extra = {}
if growth:
    gG, gE, gI = np.diff(np.log(G_lev)), np.diff(np.log(E_lev)), np.diff(np.log(I_lev))
    alpha_G, beta_G, sigma_G, _ = ols_ar1(gG)
    alpha_E, beta_E, sigma_E, eE = ols_ar1(gE)
    alpha_I, beta_I, sigma_I, eI = ols_ar1(gI)
    extra = {"g_G": gG[-1], "g_E": gE[-1], "g_I": gI[-1]}
    C = np.cov(np.column_stack([eY[1:], eE, eI]), rowvar=False)
else:
    alpha_G, beta_G, sigma_G, _ = ols_ar1(np.log(G_lev))
    alpha_E, beta_E, sigma_E, eE = ols_ar1(np.log(E_lev))
    alpha_I, beta_I, sigma_I, eI = ols_ar1(np.log(I_lev))
    C = np.cov(np.column_stack([eY, eE, eI]), rowvar=False)

params = dict(T=T, T_max=T_max, S=S, G=G, H_act=H_act, H_inact=H_inact, J=J, L=L, tau_INC=tau_INC, tau_FIRM=tau_FIRM,
              tau_VAT=tau_VAT, tau_SIF=tau_SIF, tau_SIW=tau_SIW, tau_EXPORT=tau_EXPORT, tau_CF=tau_CF, tau_G=tau_G,
              theta_UB=theta_UB, psi=psi, psi_H=psi_H, theta_DIV=theta_DIV, theta=0.05, mu=mu, r_G=r_G, zeta=0.03,
              zeta_LTV=0.6, zeta_b=0.5, I_s=firms, alpha_s=alpha_s, beta_s=beta_s, kappa_s=kappa_s, delta_s=delta_s,
              w_s=w_s, tau_Y_s=tau_Y_s, tau_K_s=tau_K_s, b_CF_g=b_CF_g, b_CFH_g=b_CFH_g, b_HH_g=b_HH_g, c_G_g=c_G_g,
              c_E_g=c_E_g, c_I_g=c_I_g, a_sg=a_sg, T_prime=T_prime, alpha_pi_EA=alpha_pi_EA, beta_pi_EA=beta_pi_EA,
              sigma_pi_EA=sigma_pi_EA, alpha_Y_EA=alpha_Y_EA, beta_Y_EA=beta_Y_EA, sigma_Y_EA=sigma_Y_EA, rho=rho,
              r_star=r_star, xi_pi=xi_pi, xi_gamma=xi_gamma, pi_star=pi_star, alpha_G=alpha_G, beta_G=beta_G,
              sigma_G=sigma_G, alpha_E=alpha_E, beta_E=beta_E, sigma_E=sigma_E, alpha_I=alpha_I, beta_I=beta_I,
              sigma_I=sigma_I, C=C, use_growth_rate_ar1=float(growth))
nan_fill = [math.nan] * max(0, Tx + T - T_calibration_exo_max)
end = min(Tx + T, T_calibration_exo_max)
hist = lambda tot, k: list(timescale * tot * data.rng(k, Te, end) / data.lin(k, Tx)) + nan_fill
ic = dict(D_I=firm_cash_quarterly, L_I=firm_debt_quarterly, omega=omega, w_UB=timescale * unemployment_benefits / unemployed,
          sb_inact=timescale * pension_benefits / inactive,
          sb_other=timescale * (social_benefits + ont - unemployment_benefits - pension_benefits) / (employees.sum() + unemployed + inactive + firms.sum() + 1),
          D_H=household_cash_quarterly, K_H=float(np.sum(dwellings)), L_G=government_debt_quarterly, E_k=bank_equity_quarterly,
          E_CB=government_debt_quarterly + firm_debt_quarterly - firm_cash_quarterly - household_cash_quarterly - bank_equity_quarterly,
          D_RoW=0.0, N_s=employees,
          Y=timescale * output.sum() * data.rng("real_gdp_quarterly", Te, Tx) / data.lin("real_gdp_quarterly", Tx),
          pi=np.diff(np.log(defl(data, Te - 1, Tx))), Y_EA=ea.lin("real_gdp_quarterly", Tx),
          pi_EA=defl(ea, Tx, Tx)[0] / defl(ea, Tx - 1, Tx - 1)[0] - 1, r_bar=r_bar,
          C_G=hist(government_consumption.sum(), "real_government_consumption_quarterly"),
          C_E=hist((exports - reexports).sum(), "real_exports_quarterly"),
          Y_I=hist(imports.sum(), "real_imports_quarterly"),
          Y_EA_series=timescale * output.sum() * Y_EA_raw / ea.lin("real_gdp_quarterly", Tx),
          pi_EA_series=np.diff(np.log(ea_def)), r_bar_series=a1, **extra)


def js(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        return None if math.isnan(v) else float(v)
    return [js(x) for x in v]


json.dump({"parameters": {k: js(v) for k, v in params.items()},
           "initial_conditions": {k: js(v) for k, v in ic.items()},
           "warnings": warnings}, open(out, "w"))
