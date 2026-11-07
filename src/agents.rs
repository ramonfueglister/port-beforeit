//! The agents and their initial state, BeforeIT `src/model_init/` (online
//! appendix E). As in the Julia code, each variable is a list over all agents
//! of a group: firm `i` is entry `i` of every firm list.

use crate::julia::{round_int, sum};
use crate::params::{InitialConditions, Parameters};
use crate::random::Randomness;

/// A group of households: the fields of BeforeIT's agent types that
/// `households_deposits(households, r, model)` reads. [`Workers`], [`Firms`]
/// (their owners) and [`Bank`] (its owner) are households.
pub trait Households {
    /// Deposits.
    fn D_h(&self) -> &[f64];
    /// Disposable income.
    fn Y_h(&self) -> &[f64];
    /// Realised consumption.
    fn C_h(&self) -> &[f64];
    /// Realised investment.
    fn I_h(&self) -> &[f64];
}

macro_rules! households {
    ($t:ty) => {
        impl Households for $t {
            fn D_h(&self) -> &[f64] {
                &self.D_h
            }
            fn Y_h(&self) -> &[f64] {
                &self.Y_h
            }
            fn C_h(&self) -> &[f64] {
                &self.C_h
            }
            fn I_h(&self) -> &[f64] {
                &self.I_h
            }
        }
    };
}
households!(Workers);
households!(Firms);

impl Households for Bank {
    fn D_h(&self) -> &[f64] {
        std::slice::from_ref(&self.D_h)
    }
    fn Y_h(&self) -> &[f64] {
        std::slice::from_ref(&self.Y_h)
    }
    fn C_h(&self) -> &[f64] {
        std::slice::from_ref(&self.C_h)
    }
    fn I_h(&self) -> &[f64] {
        std::slice::from_ref(&self.I_h)
    }
}

/// Persons: the workers (`w_act`, employed and unemployed) or the
/// economically inactive persons (`w_inact`), BeforeIT `Workers`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Workers {
    /// Disposable income (A.53).
    pub Y_h: Vec<f64>,
    /// Deposits (A.54).
    pub D_h: Vec<f64>,
    /// Capital stock, dwellings (A.52).
    pub K_h: Vec<f64>,
    /// Wage (A.41).
    pub w_h: Vec<f64>,
    /// Employer: `Some(i)` works for firm `i`; `None` is unemployed (Julia 0)
    /// or economically inactive (Julia −1).
    pub O_h: Vec<Option<usize>>,
    /// Consumption budget (A.46).
    pub C_d_h: Vec<f64>,
    /// Investment budget (A.49).
    pub I_d_h: Vec<f64>,
    /// Realised consumption (A.48).
    pub C_h: Vec<f64>,
    /// Realised investment (A.51).
    pub I_h: Vec<f64>,
}

impl Workers {
    pub fn len(&self) -> usize {
        self.Y_h.len()
    }

    pub fn is_empty(&self) -> bool {
        self.Y_h.is_empty()
    }

    /// `Workers(parameters, initial_conditions)`: all workers start
    /// unemployed; the inactive persons receive their social benefits.
    pub fn new(p: &Parameters, ic: &InitialConditions) -> (Workers, Workers) {
        let I: usize = p.I_s.iter().sum();
        let H_W = p.H_act - I - 1;
        let w_act = Workers {
            Y_h: vec![0.0; H_W],
            D_h: vec![0.0; H_W],
            K_h: vec![0.0; H_W],
            w_h: vec![ic.w_UB / p.theta_UB; H_W],
            O_h: vec![None; H_W],
            C_d_h: vec![0.0; H_W],
            I_d_h: vec![0.0; H_W],
            C_h: vec![0.0; H_W],
            I_h: vec![0.0; H_W],
        };
        let P_bar_HH = 1.0;
        let Y_h = vec![(ic.sb_inact + ic.sb_other) * P_bar_HH; p.H_inact];
        let w_inact = Workers {
            D_h: Y_h.iter().map(|y| ic.D_H * y).collect(),
            K_h: Y_h.iter().map(|y| ic.K_H * y).collect(),
            Y_h,
            w_h: vec![0.0; p.H_inact],
            O_h: vec![None; p.H_inact],
            C_d_h: vec![0.0; p.H_inact],
            I_d_h: vec![0.0; p.H_inact],
            C_h: vec![0.0; p.H_inact],
            I_h: vec![0.0; p.H_inact],
        };
        (w_act, w_inact)
    }
}

/// Firms and their owners (one investor household per firm), BeforeIT
/// `Firms`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Firms {
    /// Sector, from 0 (Julia from 1).
    pub G_i: Vec<usize>,
    pub alpha_bar_i: Vec<f64>,
    pub beta_i: Vec<f64>,
    pub kappa_i: Vec<f64>,
    /// Wage (A.26).
    pub w_i: Vec<f64>,
    pub w_bar_i: Vec<f64>,
    pub delta_i: Vec<f64>,
    pub tau_Y_i: Vec<f64>,
    pub tau_K_i: Vec<f64>,
    /// Employees (A.24).
    pub N_i: Vec<i64>,
    /// Production (A.13).
    pub Y_i: Vec<f64>,
    /// Sales (A.2).
    pub Q_i: Vec<f64>,
    /// Demand (A.1).
    pub Q_d_i: Vec<f64>,
    /// Price (A.8).
    pub P_i: Vec<f64>,
    /// Inventories of finished goods (A.4).
    pub S_i: Vec<f64>,
    /// Capital stock (A.17).
    pub K_i: Vec<f64>,
    /// Stock of intermediate inputs (A.21).
    pub M_i: Vec<f64>,
    /// Loans (A.35).
    pub L_i: Vec<f64>,
    /// Operating margin (appendix E.1).
    pub pi_bar_i: Vec<f64>,
    /// Deposits (A.34).
    pub D_i: Vec<f64>,
    /// Profits (A.30).
    pub Pi_i: Vec<f64>,
    /// Vacancies; 0 after initialisation, the labour market computes them
    /// anew each quarter.
    pub V_i: Vec<i64>,
    /// Realised investment (A.16).
    pub I_i: Vec<f64>,
    /// Equity (A.36).
    pub E_i: Vec<f64>,
    /// Average price paid for intermediate inputs.
    pub P_bar_i: Vec<f64>,
    /// Average price paid for investment goods.
    pub P_CF_i: Vec<f64>,
    /// Change in inventories (A.3).
    pub DS_i: Vec<f64>,
    /// Intermediate inputs bought (A.20).
    pub DM_i: Vec<f64>,
    /// New loans granted (A.64).
    pub DL_i: Vec<f64>,
    /// Desired new loans (A.29).
    pub DL_d_i: Vec<f64>,
    /// Expected capital stock, at expected prices (A.63).
    pub K_e_i: Vec<f64>,
    /// Expected loans (A.63).
    pub L_e_i: Vec<f64>,
    /// Expected sales, which is the planned supply (A.5).
    pub Q_s_i: Vec<f64>,
    /// Desired investment (A.14).
    pub I_d_i: Vec<f64>,
    /// Desired intermediate inputs (A.18).
    pub DM_d_i: Vec<f64>,
    /// Desired employees (A.22).
    pub N_d_i: Vec<i64>,
    /// Expected profits (A.28).
    pub Pi_e_i: Vec<f64>,
    /// The owner household: income, budgets, consumption, investment,
    /// capital stock and deposits, as in [`Workers`].
    pub Y_h: Vec<f64>,
    pub C_d_h: Vec<f64>,
    pub I_d_h: Vec<f64>,
    pub C_h: Vec<f64>,
    pub I_h: Vec<f64>,
    pub K_h: Vec<f64>,
    pub D_h: Vec<f64>,
}

impl Firms {
    pub fn len(&self) -> usize {
        self.G_i.len()
    }

    /// The firms of each of the `G` sectors, in firm order.
    pub fn by_sector(&self, G: usize) -> Vec<Vec<usize>> {
        let mut firms = vec![Vec::new(); G];
        for (i, &g) in self.G_i.iter().enumerate() {
            firms[g].push(i);
        }
        firms
    }

    pub fn is_empty(&self) -> bool {
        self.G_i.is_empty()
    }

    /// `Firms(parameters, initial_conditions)` (online appendix E.1).
    pub fn new<R: Randomness + ?Sized>(
        p: &Parameters,
        ic: &InitialConditions,
        rng: &mut R,
    ) -> Result<Firms, String> {
        let I: usize = p.I_s.iter().sum();
        let mut G_i = Vec::with_capacity(I);
        for (g, &n) in p.I_s.iter().enumerate() {
            G_i.extend(std::iter::repeat_n(g, n));
        }
        let by_sector = |v: &[f64]| -> Vec<f64> { G_i.iter().map(|&g| v[g]).collect() };
        let alpha_bar_i = by_sector(&p.alpha_s);
        let beta_i = by_sector(&p.beta_s);
        let kappa_i = by_sector(&p.kappa_s);
        let delta_i = by_sector(&p.delta_s);
        let w_bar_i = by_sector(&p.w_s);
        let tau_Y_i = by_sector(&p.tau_Y_s);
        let tau_K_i = by_sector(&p.tau_K_s);

        let mut N_i = Vec::with_capacity(I);
        for (g, &n) in p.I_s.iter().enumerate() {
            let N_s = round_int(ic.N_s[g]).map_err(|e| format!("N_s: {e}"))?;
            N_i.extend(rng.randpl(n, 2.0, N_s)?);
        }

        let omega = ic.omega;
        let Y_i: Vec<f64> = (0..I).map(|i| alpha_bar_i[i] * N_i[i] as f64).collect();
        let K_i: Vec<f64> = (0..I).map(|i| Y_i[i] / (omega * kappa_i[i])).collect();
        let M_i: Vec<f64> = (0..I).map(|i| Y_i[i] / (omega * beta_i[i])).collect();
        let sum_K = sum(&K_i);
        let L_i: Vec<f64> = K_i.iter().map(|k| ic.L_I * k / sum_K).collect();
        let pi_bar_i: Vec<f64> = (0..I)
            .map(|i| {
                1.0 - (1.0 + p.tau_SIF) * w_bar_i[i] / alpha_bar_i[i]
                    - delta_i[i] / kappa_i[i]
                    - 1.0 / beta_i[i]
                    - tau_K_i[i]
                    - tau_Y_i[i]
            })
            .collect();
        let surplus: Vec<f64> = (0..I)
            .map(|i| crate::julia::max(0.0, pi_bar_i[i] * Y_i[i]))
            .collect();
        let sum_surplus = sum(&surplus);
        let D_i: Vec<f64> = surplus.iter().map(|s| ic.D_I * s / sum_surplus).collect();
        let r = ic.r_bar + p.mu;
        let Pi_i: Vec<f64> = (0..I)
            .map(|i| pi_bar_i[i] * Y_i[i] - r * L_i[i] + ic.r_bar * crate::julia::max(0.0, D_i[i]))
            .collect();
        let P_bar_HH = 1.0;
        let Y_h: Vec<f64> = Pi_i
            .iter()
            .map(|&pi| {
                p.theta_DIV * (1.0 - p.tau_INC) * (1.0 - p.tau_FIRM) * crate::julia::max(0.0, pi)
                    + ic.sb_other * P_bar_HH
            })
            .collect();
        let zeros = vec![0.0; I];
        Ok(Firms {
            V_i: N_i.clone(),
            Q_d_i: Y_i.clone(),
            K_h: Y_h.iter().map(|y| ic.K_H * y).collect(),
            D_h: Y_h.iter().map(|y| ic.D_H * y).collect(),
            G_i,
            alpha_bar_i,
            beta_i,
            kappa_i,
            w_i: zeros.clone(),
            w_bar_i,
            delta_i,
            tau_Y_i,
            tau_K_i,
            N_i,
            Y_i,
            Q_i: zeros.clone(),
            P_i: vec![1.0; I],
            S_i: zeros.clone(),
            K_i,
            M_i,
            L_i,
            pi_bar_i,
            D_i,
            Pi_i,
            I_i: zeros.clone(),
            E_i: zeros.clone(),
            P_bar_i: zeros.clone(),
            P_CF_i: zeros.clone(),
            DS_i: zeros.clone(),
            DM_i: zeros.clone(),
            DL_i: zeros.clone(),
            DL_d_i: zeros.clone(),
            K_e_i: zeros.clone(),
            L_e_i: zeros.clone(),
            Q_s_i: zeros.clone(),
            I_d_i: zeros.clone(),
            DM_d_i: zeros.clone(),
            N_d_i: vec![0; I],
            Pi_e_i: zeros.clone(),
            Y_h,
            C_d_h: zeros.clone(),
            I_d_h: zeros.clone(),
            C_h: zeros.clone(),
            I_h: zeros,
        })
    }
}

/// The bank and its owner household (appendix A.4: one bank, as in the
/// paper). BeforeIT `Banks` with one bank.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bank {
    /// Equity (A.67).
    pub E_k: f64,
    /// Profits (A.65).
    pub Pi_k: f64,
    /// Expected profits (A.45).
    pub Pi_e_k: f64,
    /// Net position at the central bank: reserves if positive, advances if
    /// negative (A.68).
    pub D_k: f64,
    /// Lending rate (A.66).
    pub r: f64,
    /// The owner household: income, budgets, consumption, investment,
    /// capital stock and deposits, as in [`Workers`].
    pub Y_h: f64,
    pub C_d_h: f64,
    pub I_d_h: f64,
    pub C_h: f64,
    pub I_h: f64,
    pub K_h: f64,
    pub D_h: f64,
}

impl Bank {
    /// `Banks(parameters, initial_conditions)` with one bank. Profit,
    /// reserves and the owner household are set in [`crate::Model::new`].
    pub fn new(p: &Parameters, ic: &InitialConditions) -> Bank {
        Bank {
            E_k: ic.E_k,
            r: ic.r_bar + p.mu,
            ..Bank::default()
        }
    }
}

/// The central bank, BeforeIT `CentralBank` (appendix A.5).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CentralBank {
    /// Policy rate (A.69).
    pub r_bar: f64,
    /// Interest rate on government bonds.
    pub r_G: f64,
    pub rho: f64,
    pub r_star: f64,
    pub pi_star: f64,
    pub xi_pi: f64,
    pub xi_gamma: f64,
    /// Equity (A.74).
    pub E_CB: f64,
}

impl CentralBank {
    pub fn new(p: &Parameters, ic: &InitialConditions) -> CentralBank {
        CentralBank {
            r_bar: ic.r_bar,
            r_G: p.r_G,
            rho: p.rho,
            r_star: p.r_star,
            pi_star: p.pi_star,
            xi_pi: p.xi_pi,
            xi_gamma: p.xi_gamma,
            E_CB: ic.E_CB,
        }
    }
}

/// The government, BeforeIT `Government` (appendix A.3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Government {
    pub alpha_G: f64,
    pub beta_G: f64,
    pub sigma_G: f64,
    /// Revenues (A.59).
    pub Y_G: f64,
    /// Real government consumption, desired (A.55).
    pub C_G: f64,
    /// Debt (A.61).
    pub L_G: f64,
    /// Social benefits per inactive person (A.42).
    pub sb_inact: f64,
    /// Other social benefits, received by every person (A.43).
    pub sb_other: f64,
    /// Budget of each government entity (A.56).
    pub C_d_j: Vec<f64>,
    /// Nominal government consumption (A.58).
    pub C_j: f64,
    /// Average price paid for government consumption.
    pub P_j: f64,
}

impl Government {
    pub fn new(p: &Parameters, ic: &InitialConditions) -> Government {
        Government {
            alpha_G: p.alpha_G,
            beta_G: p.beta_G,
            sigma_G: p.sigma_G,
            Y_G: 0.0,
            C_G: ic.C_G[p.T_prime - 1],
            L_G: ic.L_G,
            sb_inact: ic.sb_inact,
            sb_other: ic.sb_other,
            C_d_j: vec![0.0; p.J],
            C_j: 0.0,
            P_j: 0.0,
        }
    }
}

/// The rest of the world: euro area, imports and exports, BeforeIT
/// `RestOfTheWorld` (appendix A.5.1 and A.6). One foreign seller `m` per
/// product.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RestOfTheWorld {
    pub alpha_E: f64,
    pub beta_E: f64,
    pub sigma_E: f64,
    pub alpha_I: f64,
    pub beta_I: f64,
    pub sigma_I: f64,
    /// Euro area GDP (A.71).
    pub Y_EA: f64,
    /// Euro area growth (A.72).
    pub gamma_EA: f64,
    /// Euro area inflation (A.70).
    pub pi_EA: f64,
    pub alpha_pi_EA: f64,
    pub beta_pi_EA: f64,
    pub sigma_pi_EA: f64,
    pub alpha_Y_EA: f64,
    pub beta_Y_EA: f64,
    pub sigma_Y_EA: f64,
    /// Net creditor or debtor position of the national economy towards the
    /// rest of the world (A.75).
    pub D_RoW: f64,
    /// Real supply of imports (A.76).
    pub Y_I: f64,
    /// Real demand for exports (A.81).
    pub C_E: f64,
    /// Budget of each foreign consumer (A.82).
    pub C_d_l: Vec<f64>,
    /// Nominal exports (A.84).
    pub C_l: f64,
    /// Import supply per product (A.77).
    pub Y_m: Vec<f64>,
    /// Import sales per product (A.79).
    pub Q_m: Vec<f64>,
    /// Import demand per product (A.80).
    pub Q_d_m: Vec<f64>,
    /// Import price per product (A.78).
    pub P_m: Vec<f64>,
    /// Average price paid for exports.
    pub P_l: f64,
}

impl RestOfTheWorld {
    pub fn new(p: &Parameters, ic: &InitialConditions) -> RestOfTheWorld {
        RestOfTheWorld {
            alpha_E: p.alpha_E,
            beta_E: p.beta_E,
            sigma_E: p.sigma_E,
            alpha_I: p.alpha_I,
            beta_I: p.beta_I,
            sigma_I: p.sigma_I,
            Y_EA: ic.Y_EA,
            gamma_EA: 0.0,
            pi_EA: ic.pi_EA,
            alpha_pi_EA: p.alpha_pi_EA,
            beta_pi_EA: p.beta_pi_EA,
            sigma_pi_EA: p.sigma_pi_EA,
            alpha_Y_EA: p.alpha_Y_EA,
            beta_Y_EA: p.beta_Y_EA,
            sigma_Y_EA: p.sigma_Y_EA,
            D_RoW: ic.D_RoW,
            Y_I: ic.Y_I[p.T_prime - 1],
            C_E: ic.C_E[p.T_prime - 1],
            C_d_l: vec![0.0; p.L],
            C_l: 0.0,
            Y_m: vec![0.0; p.G],
            Q_m: vec![0.0; p.G],
            Q_d_m: vec![0.0; p.G],
            P_m: vec![0.0; p.G],
            P_l: 0.0,
        }
    }
}

/// Economy-wide variables, BeforeIT `Aggregates`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Aggregates {
    /// Real GDP: the given history, then one value per simulated quarter
    /// (used in A.6).
    pub Y: Vec<f64>,
    /// Inflation, history and simulated quarters (A.11).
    pub pi_: Vec<f64>,
    /// Producer price index (A.12, computed as in the authors' code).
    pub P_bar: f64,
    /// Price index per product (A.32).
    pub P_bar_g: Vec<f64>,
    /// Consumer price index (A.31).
    pub P_bar_HH: f64,
    /// Price index of capital formation (A.37).
    pub P_bar_CF: f64,
    /// Average price paid for household consumption.
    pub P_bar_h: f64,
    /// Average price paid for household investment.
    pub P_bar_CF_h: f64,
    /// Expected real GDP (A.6).
    pub Y_e: f64,
    /// Expected growth (A.7).
    pub gamma_e: f64,
    /// Expected inflation (A.10).
    pub pi_e: f64,
    pub epsilon_Y_EA: f64,
    pub epsilon_E: f64,
    pub epsilon_I: f64,
    /// Current quarter, from 1.
    pub t: usize,
}

impl Aggregates {
    pub fn new(p: &Parameters, ic: &InitialConditions) -> Aggregates {
        Aggregates {
            Y: ic.Y.clone(),
            pi_: ic.pi.clone(),
            P_bar: 1.0,
            P_bar_g: vec![1.0; p.G],
            P_bar_HH: 1.0,
            P_bar_CF: 1.0,
            P_bar_h: 0.0,
            P_bar_CF_h: 0.0,
            Y_e: 0.0,
            gamma_e: 0.0,
            pi_e: 0.0,
            epsilon_Y_EA: 0.0,
            epsilon_E: 0.0,
            epsilon_I: 0.0,
            t: 1,
        }
    }
}
