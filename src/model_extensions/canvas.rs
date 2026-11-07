//! The CANVAS model, BeforeIT `src/model_extensions/init_CANVAS.jl`: of the
//! four changes of the CANVAS model (Hommes et al., 2025) with respect to
//! Poledna et al. (2023), BeforeIT.jl implements two, as does this port:
//! firms set prices and quantities by demand pull, and the central bank
//! re-estimates the Taylor rule every quarter (adaptive learning). The other
//! two need data on single households and firms.

use crate::agents::Firms;
use crate::estimate::estimate_taylor_rule;
use crate::firms::FirmsDecisions;
use crate::julia::max;
use crate::random::Randomness;
use crate::{AbstractModel, Extension, InitialConditions, Parameters, base, taylor_rule};

/// The fields that the CANVAS model adds, BeforeIT's `CentralBankCANVAS`
/// (`r_bar_series`) and `RestOfTheWorldCANVAS` (`Y_EA_series`,
/// `pi_EA_series`): the histories of the policy rate and of euro area GDP
/// and inflation, which grow by one entry per quarter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Canvas {
    pub r_bar_series: Vec<f64>,
    pub Y_EA_series: Vec<f64>,
    pub pi_EA_series: Vec<f64>,
}

/// BeforeIT's `ModelCANVAS`.
pub type ModelCANVAS = AbstractModel<Canvas>;

impl ModelCANVAS {
    /// `ModelCANVAS(parameters, initial_conditions)`: the model of
    /// [`crate::Model::new`] with the series `Y_EA_series`, `pi_EA_series`
    /// and `r_bar_series` of the initial conditions, and with each firm's
    /// expected sales `Q_s_i` set to its demand `Q_d_i`, so that the first
    /// price and quantity setting does not divide by 0.
    ///
    /// Returns the errors of [`crate::Model::new`], and an error if one of
    /// the three series is missing (Julia throws a `KeyError`) or shorter
    /// than `T_prime` (Julia throws in the first quarter).
    pub fn new<R: Randomness + ?Sized>(
        p: &Parameters,
        ic: &InitialConditions,
        rng: &mut R,
    ) -> Result<ModelCANVAS, String> {
        for (name, series) in [
            ("Y_EA_series", &ic.Y_EA_series),
            ("pi_EA_series", &ic.pi_EA_series),
            ("r_bar_series", &ic.r_bar_series),
        ] {
            if series.is_empty() {
                return Err(format!("initial_conditions has no {name}"));
            }
            if series.len() < p.T_prime {
                return Err(format!(
                    "{name} has {} quarters, T_prime = {} are needed",
                    series.len(),
                    p.T_prime
                ));
            }
        }
        let mut agents = crate::agents_of_the_model(p, ic, rng)?;
        let firms = &mut agents.2;
        firms.Q_s_i.clone_from(&firms.Q_d_i);
        let ext = Canvas {
            r_bar_series: ic.r_bar_series.clone(),
            Y_EA_series: ic.Y_EA_series.clone(),
            pi_EA_series: ic.pi_EA_series.clone(),
        };
        AbstractModel::from_agents(agents, ext, rng)
    }
}

/// Julia `x * b` for a float `x` and a `Bool` `b`: `x` or a zero with the
/// sign of `x`, also for NaN.
fn times_bool(x: f64, b: bool) -> f64 {
    if b { x } else { 0.0f64.copysign(x) }
}

impl Extension for Canvas {
    /// Demand pull: a firm whose sales reached its expected sales raises the
    /// quantity if its price is at least the sector's price index and the
    /// price otherwise; a firm that sold less lowers the price if its price
    /// is at least the index and the quantity otherwise, each by the relative
    /// gap between demand and expected sales. Cost-push and demand-pull price
    /// changes apply only to a random share `theta_calvo` of the firms
    /// (Calvo); the rest is as in the model of the paper.
    // The four cases of the Julia code, two of which act alike.
    #[allow(clippy::if_same_then_else)]
    fn firms_expectations_and_decisions<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> FirmsDecisions {
        let m = &*m;
        let f: &Firms = &m.firms;
        let (P_bar_g, gamma_e, pi_e) = (&m.agg.P_bar_g, m.agg.gamma_e, m.agg.pi_e);
        let I = f.G_i.len();
        let mut gamma_d_i = vec![0.0; I];
        let mut pi_d_i = vec![0.0; I];
        for i in 0..I {
            let (Q_s, Q_d, P, P_g) = (f.Q_s_i[i], f.Q_d_i[i], f.P_i[i], P_bar_g[f.G_i[i]]);
            if Q_s <= Q_d && P >= P_g {
                gamma_d_i[i] = Q_d / Q_s - 1.0;
                pi_d_i[i] = 0.0;
            } else if Q_s <= Q_d && P < P_g {
                gamma_d_i[i] = 0.0;
                pi_d_i[i] = Q_d / Q_s - 1.0;
            } else if Q_s > Q_d && P >= P_g {
                gamma_d_i[i] = 0.0;
                pi_d_i[i] = Q_d / Q_s - 1.0;
            } else if Q_s > Q_d && P < P_g {
                gamma_d_i[i] = Q_d / Q_s - 1.0;
                pi_d_i[i] = 0.0;
            }
        }
        let Q_s_i: Vec<f64> = (0..I)
            .map(|i| f.Q_s_i[i] * (1.0 + gamma_e) * (1.0 + gamma_d_i[i]))
            .collect();
        let mut pi_c_i = Self::cost_push_inflation(m, f);
        let theta_calvo = m.prop.theta_calvo;
        let adjust_mask: Vec<bool> = (0..f.P_i.len()).map(|_| rng.rand() < theta_calvo).collect();
        for i in 0..I {
            pi_d_i[i] = times_bool(pi_d_i[i], adjust_mask[i]);
            pi_c_i[i] = times_bool(pi_c_i[i], adjust_mask[i]);
        }
        let new_P_i = (0..I)
            .map(|i| f.P_i[i] * (1.0 + pi_c_i[i]) * (1.0 + pi_e) * (1.0 + pi_d_i[i]))
            .collect();
        let (I_d_i, DM_d_i, N_d_i) = Self::desired_capital_material_employment(f, &Q_s_i);
        let Pi_e_i: Vec<f64> = f
            .Pi_i
            .iter()
            .map(|pi| pi * (1.0 + pi_e) * (1.0 + gamma_e))
            .collect();
        let (DD_e_i, K_e_i, L_e_i) = Self::expected_deposits_capital_loans(m, f, &Pi_e_i);
        let DL_d_i = (0..I).map(|i| max(0.0, -DD_e_i[i] - f.D_i[i])).collect();
        FirmsDecisions {
            Q_s_i,
            I_d_i,
            DM_d_i,
            N_d_i,
            Pi_e_i,
            DL_d_i,
            K_e_i,
            L_e_i,
            new_P_i,
        }
    }

    /// Adaptive learning: re-estimates the parameters of the Taylor rule
    /// with [`estimate_taylor_rule`] on the first `T_prime + t − 1` entries
    /// of the histories, stores them in the central bank, applies the rule
    /// and appends the new rate to `r_bar_series`. As in BeforeIT.jl, the
    /// estimate receives euro area GDP in levels in the place of inflation
    /// and euro area inflation in the place of growth.
    fn central_bank_rate(m: &mut AbstractModel<Self>) -> f64 {
        let n = m.prop.T_prime + m.agg.t - 1;
        let e = &m.ext;
        let (rho, r_star, xi_pi, xi_gamma, pi_star) = estimate_taylor_rule(
            &e.r_bar_series[..n],
            &e.Y_EA_series[..n],
            &e.pi_EA_series[..n],
        );
        let cb = &mut m.cb;
        (cb.rho, cb.r_star, cb.xi_pi, cb.xi_gamma, cb.pi_star) =
            (rho, r_star, xi_pi, xi_gamma, pi_star);
        let r_bar = taylor_rule(
            cb.rho,
            cb.r_bar,
            cb.r_star,
            cb.pi_star,
            cb.xi_pi,
            cb.xi_gamma,
            m.rotw.gamma_EA,
            m.rotw.pi_EA,
        );
        m.ext.r_bar_series.push(r_bar);
        r_bar
    }

    /// Euro area GDP, growth and inflation as in the model of the paper,
    /// appended to `Y_EA_series` and `pi_EA_series`.
    fn growth_inflation_EA<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, f64, f64) {
        let (Y_EA, gamma_EA, pi_EA) = base::growth_inflation_EA(m, rng);
        m.ext.Y_EA_series.push(Y_EA);
        m.ext.pi_EA_series.push(pi_EA);
        (Y_EA, gamma_EA, pi_EA)
    }
}
