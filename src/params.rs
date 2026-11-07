//! Model inputs. The names are the keys of BeforeIT's `parameters` and
//! `initial_conditions`; their meaning is given in the online appendix of
//! the paper, tables D.2 and D.3 (parameters) and E.1 and E.2 (initial
//! conditions). BeforeIT keeps the firm parameters per sector (`alpha_s`
//! and so on), and every firm of a sector gets the sector's value.

/// Parameters (BeforeIT `parameters`). Matrices are stored as rows:
/// `a_sg[g][s]` is Julia's `a_sg[g, s]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameters {
    /// Number of products and industries (S = G).
    pub G: usize,
    /// Length of the history used for the expectations, in quarters.
    pub T_prime: usize,
    /// Economically active persons, including the firm and bank owners.
    pub H_act: usize,
    /// Economically inactive persons.
    pub H_inact: usize,
    /// Government entities.
    pub J: usize,
    /// Foreign consumers.
    pub L: usize,
    /// Firms per industry.
    pub I_s: Vec<usize>,
    /// Average productivity of labour per industry.
    pub alpha_s: Vec<f64>,
    /// Productivity of intermediate consumption per industry.
    pub beta_s: Vec<f64>,
    /// Productivity of capital per industry.
    pub kappa_s: Vec<f64>,
    /// Depreciation rate for capital per industry.
    pub delta_s: Vec<f64>,
    /// Average wage rate per industry.
    pub w_s: Vec<f64>,
    /// Net tax rate on products per industry.
    pub tau_Y_s: Vec<f64>,
    /// Net tax rate on production per industry.
    pub tau_K_s: Vec<f64>,
    /// Income tax rate.
    pub tau_INC: f64,
    /// Corporate tax rate.
    pub tau_FIRM: f64,
    /// Value-added tax rate.
    pub tau_VAT: f64,
    /// Social insurance rate, employers' contributions.
    pub tau_SIF: f64,
    /// Social insurance rate, employees' contributions.
    pub tau_SIW: f64,
    /// Export tax rate.
    pub tau_EXPORT: f64,
    /// Tax rate on capital formation.
    pub tau_CF: f64,
    /// Tax rate on government consumption.
    pub tau_G: f64,
    /// Unemployment benefit replacement rate.
    pub theta_UB: f64,
    /// Fraction of income devoted to consumption.
    pub psi: f64,
    /// Fraction of income devoted to investment in housing.
    pub psi_H: f64,
    /// Risk premium on the policy rate.
    pub mu: f64,
    /// Dividend payout ratio.
    pub theta_DIV: f64,
    /// Rate of instalment on debt.
    pub theta: f64,
    /// Bank's capital ratio.
    pub zeta: f64,
    /// Loan-to-value ratio.
    pub zeta_LTV: f64,
    /// Loan-to-capital ratio for new firms after bankruptcy.
    pub zeta_b: f64,
    /// Share of product g in firm investment.
    pub b_CF_g: Vec<f64>,
    /// Share of product g in household investment.
    pub b_CFH_g: Vec<f64>,
    /// Share of product g in household consumption.
    pub b_HH_g: Vec<f64>,
    /// Share of product g in government consumption.
    pub c_G_g: Vec<f64>,
    /// Share of product g in exports.
    pub c_E_g: Vec<f64>,
    /// Share of product g in imports.
    pub c_I_g: Vec<f64>,
    /// Technology coefficients, `a_sg[g][s]`: units of product g per unit of
    /// the intermediate inputs of industry s.
    pub a_sg: Vec<Vec<f64>>,
    /// Covariance matrix of the shocks to euro area GDP, exports and imports,
    /// in this order.
    pub C: [[f64; 3]; 3],
    /// AR(1) coefficient, constant and standard deviation of the shock for
    /// log government consumption.
    pub alpha_G: f64,
    pub beta_G: f64,
    pub sigma_G: f64,
    /// The same for log exports. The shocks to exports, imports and euro
    /// area GDP come from `C`; as in the Julia code, `sigma_E`, `sigma_I` and
    /// `sigma_Y_EA` are stored but not used.
    pub alpha_E: f64,
    pub beta_E: f64,
    pub sigma_E: f64,
    /// The same for log imports.
    pub alpha_I: f64,
    pub beta_I: f64,
    pub sigma_I: f64,
    /// The same for log(1 + euro area inflation).
    pub alpha_pi_EA: f64,
    pub beta_pi_EA: f64,
    pub sigma_pi_EA: f64,
    /// The same for log euro area GDP.
    pub alpha_Y_EA: f64,
    pub beta_Y_EA: f64,
    pub sigma_Y_EA: f64,
    /// Interest rate on government bonds.
    pub r_G: f64,
    /// Adjustment coefficient of the policy rate.
    pub rho: f64,
    /// Real equilibrium interest rate.
    pub r_star: f64,
    /// Inflation target of the central bank.
    pub pi_star: f64,
    /// Weight of the inflation target.
    pub xi_pi: f64,
    /// Weight of economic growth.
    pub xi_gamma: f64,
    /// Number of banks. BeforeIT.jl uses 1 when the key `n_banks` is
    /// missing, as in the paper.
    pub n_banks: usize,
    /// Weight of last quarter's inflation in expected inflation, between 0
    /// (only the estimate) and 1 (only last quarter). BeforeIT.jl uses 0 when
    /// the key `omega_pi` is missing.
    pub omega_pi: f64,
    /// Share of the firms, drawn at random, that adjust their price each
    /// quarter; only the model type CANVAS uses it. BeforeIT.jl uses 1 (all
    /// firms) when the key `theta_calvo` is missing.
    pub theta_calvo: f64,
    /// Whether a bank's capital constraint on new loans also counts the
    /// overdrafts (negative deposits) of its customers. BeforeIT.jl uses
    /// `false` (loans only) when the key `overdrafts_in_capital_ratio` is
    /// missing.
    pub overdrafts_in_capital_ratio: bool,
    /// Number of industries `S`, forecast horizon `T` and the number of
    /// those quarters for which data exist, `T_max`. The calibration writes
    /// them and the data sets have them; the model does not use them.
    pub S: Option<usize>,
    pub T: Option<usize>,
    pub T_max: Option<usize>,
    /// Whether the calibration estimated the AR(1) processes of government
    /// consumption, exports and imports on growth rates (see
    /// [`crate::calibration::get_params_and_initial_conditions`]); the
    /// model does not use it.
    pub use_growth_rate_ar1: Option<bool>,
}

/// Initial conditions (BeforeIT `initial_conditions`). Time series start with
/// the oldest quarter.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialConditions {
    /// Deposits of the household sector.
    pub D_H: f64,
    /// Deposits of the firm sector.
    pub D_I: f64,
    /// Net creditor or debtor position towards the rest of the world.
    pub D_RoW: f64,
    /// Central bank equity.
    pub E_CB: f64,
    /// Bank equity.
    pub E_k: f64,
    /// Capital (dwellings) of the household sector.
    pub K_H: f64,
    /// Government debt.
    pub L_G: f64,
    /// Loans of the firm sector.
    pub L_I: f64,
    /// Employees per industry.
    pub N_s: Vec<f64>,
    /// Real GDP history, at least `T_prime` quarters. The model uses the
    /// first `T_prime` and stores each simulated quarter after them.
    pub Y: Vec<f64>,
    /// Inflation history, at least `T_prime` quarters, used like `Y`.
    pub pi: Vec<f64>,
    /// Euro area GDP.
    pub Y_EA: f64,
    /// Euro area inflation.
    pub pi_EA: f64,
    /// Histories of imports, exports and government consumption; the model
    /// uses the value of quarter `T_prime` (counted from 1).
    pub Y_I: Vec<f64>,
    pub C_E: Vec<f64>,
    pub C_G: Vec<f64>,
    /// Desired capacity utilisation rate.
    pub omega: f64,
    /// Policy rate.
    pub r_bar: f64,
    /// Social benefits per inactive person.
    pub sb_inact: f64,
    /// Other social benefits, received by every person.
    pub sb_other: f64,
    /// Unemployment benefits.
    pub w_UB: f64,
    /// Histories of the policy rate, euro area GDP and euro area inflation,
    /// at least `T_prime` quarters. Only the CANVAS model uses them; they
    /// are empty when the initial conditions do not have them.
    pub r_bar_series: Vec<f64>,
    pub Y_EA_series: Vec<f64>,
    pub pi_EA_series: Vec<f64>,
    /// Last growth rates of government consumption, exports and imports,
    /// which the calibration adds when it estimates on growth rates; the
    /// model does not use them.
    pub g_G: Option<f64>,
    pub g_E: Option<f64>,
    pub g_I: Option<f64>,
}

/// Values that stay fixed during a run, BeforeIT `Properties`.
#[derive(Clone, Debug, PartialEq)]
pub struct Properties {
    pub G: usize,
    pub T_prime: usize,
    pub H_act: usize,
    pub H_inact: usize,
    pub J: usize,
    pub L: usize,
    pub I_s: Vec<usize>,
    /// Number of firms.
    pub I: usize,
    /// Number of persons, `H_act + H_inact`.
    pub H: usize,
    pub tau_INC: f64,
    pub tau_FIRM: f64,
    pub tau_VAT: f64,
    pub tau_SIF: f64,
    pub tau_SIW: f64,
    pub tau_EXPORT: f64,
    pub tau_CF: f64,
    pub tau_G: f64,
    pub theta_UB: f64,
    pub psi: f64,
    pub psi_H: f64,
    pub mu: f64,
    pub theta_DIV: f64,
    pub theta: f64,
    pub zeta: f64,
    pub zeta_LTV: f64,
    pub zeta_b: f64,
    pub omega_pi: f64,
    pub theta_calvo: f64,
    pub overdrafts_in_capital_ratio: bool,
    pub b_CF_g: Vec<f64>,
    pub b_CFH_g: Vec<f64>,
    pub b_HH_g: Vec<f64>,
    pub c_G_g: Vec<f64>,
    pub c_E_g: Vec<f64>,
    pub c_I_g: Vec<f64>,
    pub a_sg: Vec<Vec<f64>>,
    pub C: [[f64; 3]; 3],
    pub D_H: f64,
    pub K_H: f64,
    pub sb_other: f64,
    pub E_k: f64,
    pub r_bar: f64,
    pub n_banks: usize,
}

impl Properties {
    /// `Properties(parameters, initial_conditions)`, after checking the
    /// inputs, which the Julia code uses without checking: the lists per
    /// sector have `G` entries, the histories are long enough for the
    /// estimates, there is at least one bank, every firm and every bank have
    /// an owner among the active persons, and the shocks can be drawn from
    /// `C`. For the last four the Julia code throws an exception, at the
    /// latest in the first quarter.
    pub fn new(p: &Parameters, ic: &InitialConditions) -> Result<Self, String> {
        check(p, ic)?;
        Ok(Properties {
            G: p.G,
            T_prime: p.T_prime,
            H_act: p.H_act,
            H_inact: p.H_inact,
            J: p.J,
            L: p.L,
            I_s: p.I_s.clone(),
            I: p.I_s.iter().sum(),
            H: p.H_act + p.H_inact,
            tau_INC: p.tau_INC,
            tau_FIRM: p.tau_FIRM,
            tau_VAT: p.tau_VAT,
            tau_SIF: p.tau_SIF,
            tau_SIW: p.tau_SIW,
            tau_EXPORT: p.tau_EXPORT,
            tau_CF: p.tau_CF,
            tau_G: p.tau_G,
            theta_UB: p.theta_UB,
            psi: p.psi,
            psi_H: p.psi_H,
            mu: p.mu,
            theta_DIV: p.theta_DIV,
            theta: p.theta,
            zeta: p.zeta,
            zeta_LTV: p.zeta_LTV,
            zeta_b: p.zeta_b,
            omega_pi: p.omega_pi,
            theta_calvo: p.theta_calvo,
            overdrafts_in_capital_ratio: p.overdrafts_in_capital_ratio,
            b_CF_g: p.b_CF_g.clone(),
            b_CFH_g: p.b_CFH_g.clone(),
            b_HH_g: p.b_HH_g.clone(),
            c_G_g: p.c_G_g.clone(),
            c_E_g: p.c_E_g.clone(),
            c_I_g: p.c_I_g.clone(),
            a_sg: p.a_sg.clone(),
            C: p.C,
            D_H: ic.D_H,
            K_H: ic.K_H,
            sb_other: ic.sb_other,
            E_k: ic.E_k,
            r_bar: ic.r_bar,
            n_banks: p.n_banks,
        })
    }
}

/// The checks of [`Properties::new`].
fn check(p: &Parameters, ic: &InitialConditions) -> Result<(), String> {
    let G = p.G;
    if G == 0 {
        return Err("G is 0".into());
    }
    let per_sector: [(&str, usize); 16] = [
        ("I_s", p.I_s.len()),
        ("alpha_s", p.alpha_s.len()),
        ("beta_s", p.beta_s.len()),
        ("kappa_s", p.kappa_s.len()),
        ("delta_s", p.delta_s.len()),
        ("w_s", p.w_s.len()),
        ("tau_Y_s", p.tau_Y_s.len()),
        ("tau_K_s", p.tau_K_s.len()),
        ("b_CF_g", p.b_CF_g.len()),
        ("b_CFH_g", p.b_CFH_g.len()),
        ("b_HH_g", p.b_HH_g.len()),
        ("c_G_g", p.c_G_g.len()),
        ("c_E_g", p.c_E_g.len()),
        ("c_I_g", p.c_I_g.len()),
        ("a_sg", p.a_sg.len()),
        ("N_s", ic.N_s.len()),
    ];
    for (name, len) in per_sector {
        if len != G {
            return Err(format!("{name} has {len} entries instead of G = {G}"));
        }
    }
    if let Some(row) = p.a_sg.iter().find(|row| row.len() != G) {
        return Err(format!(
            "a_sg has a row with {} entries instead of G = {G}",
            row.len()
        ));
    }
    if p.T_prime < 2 {
        return Err(format!(
            "T_prime is {}, the AR(1) estimation needs at least 2 quarters",
            p.T_prime
        ));
    }
    let history: [(&str, usize); 5] = [
        ("Y", ic.Y.len()),
        ("pi", ic.pi.len()),
        ("Y_I", ic.Y_I.len()),
        ("C_E", ic.C_E.len()),
        ("C_G", ic.C_G.len()),
    ];
    for (name, len) in history {
        if len < p.T_prime {
            return Err(format!(
                "{name} has {len} quarters, T_prime = {} are needed",
                p.T_prime
            ));
        }
    }
    // Julia's `rand(1:n_banks, ·)` in the model constructor throws.
    if p.n_banks == 0 {
        return Err("n_banks is 0".into());
    }
    let I: usize = p.I_s.iter().sum();
    if p.H_act < I + p.n_banks {
        return Err(format!(
            "H_act = {} is less than the number of firm and bank owners, I + n_banks = {}",
            p.H_act,
            I + p.n_banks
        ));
    }
    // Julia's `epsilon(C)` throws in the first quarter for such a matrix.
    crate::estimate::shock_factor(&p.C)?;
    Ok(())
}

/// The JSON object of a data set.
type Object = serde_json::Map<String, serde_json::Value>;

fn object(text: &str, what: &str) -> Result<Object, String> {
    match serde_json::from_str(text) {
        Ok(serde_json::Value::Object(o)) => Ok(o),
        Ok(_) => Err(format!("{what}: not a JSON object")),
        Err(e) => Err(format!("{what}: {e}")),
    }
}

/// Julia `dict[key]`, which throws a `KeyError` for a missing key.
fn get<'a>(o: &'a Object, key: &str) -> Result<&'a serde_json::Value, String> {
    o.get(key).ok_or_else(|| format!("{key} is missing"))
}

/// A number; `null` stands for NaN, which JSON cannot express.
fn as_number(v: &serde_json::Value, key: &str) -> Result<f64, String> {
    if v.is_null() {
        return Ok(f64::NAN);
    }
    v.as_f64().ok_or_else(|| format!("{key} is not a number"))
}

fn number(o: &Object, key: &str) -> Result<f64, String> {
    as_number(get(o, key)?, key)
}

/// A count, which the data sets store as an integer or a float, Julia
/// `Int(x)`.
fn count(o: &Object, key: &str) -> Result<usize, String> {
    as_count(number(o, key)?, key)
}

fn as_count(x: f64, key: &str) -> Result<usize, String> {
    if x >= 0.0 && x.fract() == 0.0 && x < 2f64.powi(53) {
        Ok(x as usize)
    } else {
        Err(format!("{key} = {x} is not a count"))
    }
}

/// A list, Julia `vec(x)`: a matrix (a list of rows) is read column by
/// column.
fn list(o: &Object, key: &str) -> Result<Vec<f64>, String> {
    let rows = get(o, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not a list"))?;
    if rows.iter().all(|x| x.is_number() || x.is_null()) {
        return rows.iter().map(|x| as_number(x, key)).collect();
    }
    let m = matrix(o, key)?;
    let n = m.first().map_or(0, |r| r.len());
    Ok((0..n).flat_map(|c| m.iter().map(move |r| r[c])).collect())
}

/// A matrix, as a list of rows of equal length.
fn matrix(o: &Object, key: &str) -> Result<Vec<Vec<f64>>, String> {
    let rows = get(o, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not a matrix"))?;
    let m: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| format!("{key} is not a matrix"))?
                .iter()
                .map(|x| as_number(x, key))
                .collect()
        })
        .collect::<Result<_, _>>()?;
    if m.iter().any(|r| r.len() != m[0].len()) {
        return Err(format!("{key} has rows of different lengths"));
    }
    Ok(m)
}

/// Julia `get(dict, key, default)`.
fn number_or(o: &Object, key: &str, default: f64) -> Result<f64, String> {
    o.get(key).map_or(Ok(default), |v| as_number(v, key))
}

fn optional_number(o: &Object, key: &str) -> Result<Option<f64>, String> {
    o.get(key).map(|v| as_number(v, key)).transpose()
}

fn optional_count(o: &Object, key: &str) -> Result<Option<usize>, String> {
    optional_number(o, key)?
        .map(|x| as_count(x, key))
        .transpose()
}

/// A list of numbers in JSON; NaN becomes `null`.
fn numbers(v: &[f64]) -> serde_json::Value {
    v.iter().map(|&x| serde_json::Value::from(x)).collect()
}

impl Parameters {
    /// Reads `parameters` from a JSON object whose keys are BeforeIT's, such
    /// as the files in the crate's folder `data`. Counts may be floats, as in
    /// BeforeIT's JLD2 files. `n_banks`, `omega_pi`, `theta_calvo` and
    /// `overdrafts_in_capital_ratio` may be missing; they then get
    /// BeforeIT's defaults 1, 0, 1 and false; `S`, `T`, `T_max` and
    /// `use_growth_rate_ar1` may be missing too. `null` stands for NaN.
    /// Other keys are ignored.
    pub fn from_json(text: &str) -> Result<Parameters, String> {
        let o = &object(text, "parameters")?;
        let c = matrix(o, "C")?;
        if c.len() != 3 || c[0].len() != 3 {
            return Err("C is not a 3 × 3 matrix".into());
        }
        let C = [0, 1, 2].map(|r| [c[r][0], c[r][1], c[r][2]]);
        let n = |key| number(o, key);
        let overdrafts = match o.get("overdrafts_in_capital_ratio") {
            None => false,
            Some(serde_json::Value::Bool(b)) => *b,
            // Julia `Bool(x)` accepts the numbers 0 and 1.
            Some(v) => match as_number(v, "overdrafts_in_capital_ratio")? {
                0.0 => false,
                1.0 => true,
                x => return Err(format!("overdrafts_in_capital_ratio = {x} is not a Bool")),
            },
        };
        Ok(Parameters {
            G: count(o, "G")?,
            T_prime: count(o, "T_prime")?,
            H_act: count(o, "H_act")?,
            H_inact: count(o, "H_inact")?,
            J: count(o, "J")?,
            L: count(o, "L")?,
            I_s: list(o, "I_s")?
                .into_iter()
                .map(|x| as_count(x, "I_s"))
                .collect::<Result<_, _>>()?,
            alpha_s: list(o, "alpha_s")?,
            beta_s: list(o, "beta_s")?,
            kappa_s: list(o, "kappa_s")?,
            delta_s: list(o, "delta_s")?,
            w_s: list(o, "w_s")?,
            tau_Y_s: list(o, "tau_Y_s")?,
            tau_K_s: list(o, "tau_K_s")?,
            tau_INC: n("tau_INC")?,
            tau_FIRM: n("tau_FIRM")?,
            tau_VAT: n("tau_VAT")?,
            tau_SIF: n("tau_SIF")?,
            tau_SIW: n("tau_SIW")?,
            tau_EXPORT: n("tau_EXPORT")?,
            tau_CF: n("tau_CF")?,
            tau_G: n("tau_G")?,
            theta_UB: n("theta_UB")?,
            psi: n("psi")?,
            psi_H: n("psi_H")?,
            mu: n("mu")?,
            theta_DIV: n("theta_DIV")?,
            theta: n("theta")?,
            zeta: n("zeta")?,
            zeta_LTV: n("zeta_LTV")?,
            zeta_b: n("zeta_b")?,
            b_CF_g: list(o, "b_CF_g")?,
            b_CFH_g: list(o, "b_CFH_g")?,
            b_HH_g: list(o, "b_HH_g")?,
            c_G_g: list(o, "c_G_g")?,
            c_E_g: list(o, "c_E_g")?,
            c_I_g: list(o, "c_I_g")?,
            a_sg: matrix(o, "a_sg")?,
            C,
            alpha_G: n("alpha_G")?,
            beta_G: n("beta_G")?,
            sigma_G: n("sigma_G")?,
            alpha_E: n("alpha_E")?,
            beta_E: n("beta_E")?,
            sigma_E: n("sigma_E")?,
            alpha_I: n("alpha_I")?,
            beta_I: n("beta_I")?,
            sigma_I: n("sigma_I")?,
            alpha_pi_EA: n("alpha_pi_EA")?,
            beta_pi_EA: n("beta_pi_EA")?,
            sigma_pi_EA: n("sigma_pi_EA")?,
            alpha_Y_EA: n("alpha_Y_EA")?,
            beta_Y_EA: n("beta_Y_EA")?,
            sigma_Y_EA: n("sigma_Y_EA")?,
            r_G: n("r_G")?,
            rho: n("rho")?,
            r_star: n("r_star")?,
            pi_star: n("pi_star")?,
            xi_pi: n("xi_pi")?,
            xi_gamma: n("xi_gamma")?,
            n_banks: as_count(number_or(o, "n_banks", 1.0)?, "n_banks")?,
            omega_pi: number_or(o, "omega_pi", 0.0)?,
            theta_calvo: number_or(o, "theta_calvo", 1.0)?,
            overdrafts_in_capital_ratio: overdrafts,
            S: optional_count(o, "S")?,
            T: optional_count(o, "T")?,
            T_max: optional_count(o, "T_max")?,
            use_growth_rate_ar1: match o.get("use_growth_rate_ar1") {
                None => None,
                Some(v) => Some(
                    v.as_bool()
                        .ok_or("use_growth_rate_ar1 is not a Bool".to_string())?,
                ),
            },
        })
    }

    /// Writes the parameters as a JSON object with BeforeIT's keys, which
    /// [`Parameters::from_json`] reads back. Missing optional values are
    /// left out; NaN is written as `null`.
    pub fn to_json(&self) -> String {
        let mut o = Object::new();
        let mut put = |key: &str, v: serde_json::Value| {
            o.insert(key.to_string(), v);
        };
        let counts = |v: &[usize]| serde_json::Value::from(v.to_vec());
        put("G", self.G.into());
        put("T_prime", self.T_prime.into());
        put("H_act", self.H_act.into());
        put("H_inact", self.H_inact.into());
        put("J", self.J.into());
        put("L", self.L.into());
        put("I_s", counts(&self.I_s));
        for (key, v) in [
            ("alpha_s", &self.alpha_s),
            ("beta_s", &self.beta_s),
            ("kappa_s", &self.kappa_s),
            ("delta_s", &self.delta_s),
            ("w_s", &self.w_s),
            ("tau_Y_s", &self.tau_Y_s),
            ("tau_K_s", &self.tau_K_s),
            ("b_CF_g", &self.b_CF_g),
            ("b_CFH_g", &self.b_CFH_g),
            ("b_HH_g", &self.b_HH_g),
            ("c_G_g", &self.c_G_g),
            ("c_E_g", &self.c_E_g),
            ("c_I_g", &self.c_I_g),
        ] {
            put(key, numbers(v));
        }
        put("a_sg", self.a_sg.iter().map(|r| numbers(r)).collect());
        put("C", self.C.iter().map(|r| numbers(r)).collect());
        for (key, x) in [
            ("tau_INC", self.tau_INC),
            ("tau_FIRM", self.tau_FIRM),
            ("tau_VAT", self.tau_VAT),
            ("tau_SIF", self.tau_SIF),
            ("tau_SIW", self.tau_SIW),
            ("tau_EXPORT", self.tau_EXPORT),
            ("tau_CF", self.tau_CF),
            ("tau_G", self.tau_G),
            ("theta_UB", self.theta_UB),
            ("psi", self.psi),
            ("psi_H", self.psi_H),
            ("mu", self.mu),
            ("theta_DIV", self.theta_DIV),
            ("theta", self.theta),
            ("zeta", self.zeta),
            ("zeta_LTV", self.zeta_LTV),
            ("zeta_b", self.zeta_b),
            ("alpha_G", self.alpha_G),
            ("beta_G", self.beta_G),
            ("sigma_G", self.sigma_G),
            ("alpha_E", self.alpha_E),
            ("beta_E", self.beta_E),
            ("sigma_E", self.sigma_E),
            ("alpha_I", self.alpha_I),
            ("beta_I", self.beta_I),
            ("sigma_I", self.sigma_I),
            ("alpha_pi_EA", self.alpha_pi_EA),
            ("beta_pi_EA", self.beta_pi_EA),
            ("sigma_pi_EA", self.sigma_pi_EA),
            ("alpha_Y_EA", self.alpha_Y_EA),
            ("beta_Y_EA", self.beta_Y_EA),
            ("sigma_Y_EA", self.sigma_Y_EA),
            ("r_G", self.r_G),
            ("rho", self.rho),
            ("r_star", self.r_star),
            ("pi_star", self.pi_star),
            ("xi_pi", self.xi_pi),
            ("xi_gamma", self.xi_gamma),
            ("omega_pi", self.omega_pi),
            ("theta_calvo", self.theta_calvo),
        ] {
            put(key, x.into());
        }
        put("n_banks", self.n_banks.into());
        put(
            "overdrafts_in_capital_ratio",
            self.overdrafts_in_capital_ratio.into(),
        );
        for (key, x) in [("S", self.S), ("T", self.T), ("T_max", self.T_max)] {
            if let Some(x) = x {
                put(key, x.into());
            }
        }
        if let Some(b) = self.use_growth_rate_ar1 {
            put("use_growth_rate_ar1", b.into());
        }
        serde_json::Value::Object(o).to_string()
    }
}

impl InitialConditions {
    /// Reads `initial_conditions` from a JSON object whose keys are
    /// BeforeIT's, such as the files in the crate's folder `data`. The
    /// series of the CANVAS model and the growth rates `g_G`, `g_E` and `g_I`
    /// may be missing. `null` stands for NaN. Other keys are ignored.
    pub fn from_json(text: &str) -> Result<InitialConditions, String> {
        let o = &object(text, "initial_conditions")?;
        let n = |key| number(o, key);
        let optional = |key| {
            if o.contains_key(key) {
                list(o, key)
            } else {
                Ok(Vec::new())
            }
        };
        Ok(InitialConditions {
            D_H: n("D_H")?,
            D_I: n("D_I")?,
            D_RoW: n("D_RoW")?,
            E_CB: n("E_CB")?,
            E_k: n("E_k")?,
            K_H: n("K_H")?,
            L_G: n("L_G")?,
            L_I: n("L_I")?,
            N_s: list(o, "N_s")?,
            Y: list(o, "Y")?,
            pi: list(o, "pi")?,
            Y_EA: n("Y_EA")?,
            pi_EA: n("pi_EA")?,
            Y_I: list(o, "Y_I")?,
            C_E: list(o, "C_E")?,
            C_G: list(o, "C_G")?,
            omega: n("omega")?,
            r_bar: n("r_bar")?,
            sb_inact: n("sb_inact")?,
            sb_other: n("sb_other")?,
            w_UB: n("w_UB")?,
            r_bar_series: optional("r_bar_series")?,
            Y_EA_series: optional("Y_EA_series")?,
            pi_EA_series: optional("pi_EA_series")?,
            g_G: optional_number(o, "g_G")?,
            g_E: optional_number(o, "g_E")?,
            g_I: optional_number(o, "g_I")?,
        })
    }

    /// Writes the initial conditions as a JSON object with BeforeIT's keys,
    /// which [`InitialConditions::from_json`] reads back. Empty series of
    /// the CANVAS model and missing growth rates are left out; NaN, which
    /// the histories `C_G`, `C_E` and `Y_I` of a recent calibration contain,
    /// is written as `null`.
    pub fn to_json(&self) -> String {
        let mut o = Object::new();
        for (key, x) in [
            ("D_H", self.D_H),
            ("D_I", self.D_I),
            ("D_RoW", self.D_RoW),
            ("E_CB", self.E_CB),
            ("E_k", self.E_k),
            ("K_H", self.K_H),
            ("L_G", self.L_G),
            ("L_I", self.L_I),
            ("Y_EA", self.Y_EA),
            ("pi_EA", self.pi_EA),
            ("omega", self.omega),
            ("r_bar", self.r_bar),
            ("sb_inact", self.sb_inact),
            ("sb_other", self.sb_other),
            ("w_UB", self.w_UB),
        ] {
            o.insert(key.to_string(), x.into());
        }
        for (key, v) in [
            ("N_s", &self.N_s),
            ("Y", &self.Y),
            ("pi", &self.pi),
            ("Y_I", &self.Y_I),
            ("C_E", &self.C_E),
            ("C_G", &self.C_G),
        ] {
            o.insert(key.to_string(), numbers(v));
        }
        for (key, v) in [
            ("r_bar_series", &self.r_bar_series),
            ("Y_EA_series", &self.Y_EA_series),
            ("pi_EA_series", &self.pi_EA_series),
        ] {
            if !v.is_empty() {
                o.insert(key.to_string(), numbers(v));
            }
        }
        for (key, x) in [("g_G", self.g_G), ("g_E", self.g_E), ("g_I", self.g_I)] {
            if let Some(x) = x {
                o.insert(key.to_string(), x.into());
            }
        }
        serde_json::Value::Object(o).to_string()
    }
}
