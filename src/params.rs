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
}

/// Values that stay fixed during a run, BeforeIT `Properties`. The options
/// of BeforeIT's model extensions (`n_banks`, `omega_pi`, `theta_calvo`,
/// `overdrafts_in_capital_ratio`) are not ported.
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
}

impl Properties {
    /// `Properties(parameters, initial_conditions)`, after checking the
    /// inputs, which the Julia code uses without checking: the lists per
    /// sector have `G` entries, the histories are long enough for the
    /// estimates, every firm and the bank have an owner among the active
    /// persons, and the shocks can be drawn from `C`. For the last three the
    /// Julia code throws an exception, at the latest in the first quarter.
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
    let I: usize = p.I_s.iter().sum();
    if p.H_act < I + 1 {
        return Err(format!(
            "H_act = {} is less than the {I} firm owners plus the bank owner",
            p.H_act
        ));
    }
    // Julia's `epsilon(C)` throws in the first quarter for such a matrix.
    crate::estimate::shock_factor(&p.C)?;
    Ok(())
}
