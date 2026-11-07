//! Government, BeforeIT `src/agent_actions/government.jl` (appendix A.3).

use crate::julia::{pos, sum, sum_by};
use crate::random::Randomness;
use crate::{AbstractModel, Extension};

/// `gov_social_benefits` (A.42, A.43): social benefits grow with expected
/// growth. Returns `sb_other` and `sb_inact`.
pub fn gov_social_benefits<X: Extension>(m: &mut AbstractModel<X>) -> (f64, f64) {
    let gamma_e = m.agg.gamma_e;
    let sb_other = m.gov.sb_other * (1.0 + gamma_e);
    let sb_inact = m.gov.sb_inact * (1.0 + gamma_e);
    (sb_other, sb_inact)
}

/// `gov_expenditure` (A.55, A.56): real government consumption by an AR(1)
/// rule, split equally among the government entities as nominal budgets at
/// expected prices. Returns `C_G` and `C_d_j`.
pub fn gov_expenditure<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    rng: &mut R,
) -> (f64, Vec<f64>) {
    let gov = &m.gov;
    let (c_G_g, P_bar_g, pi_e) = (&m.prop.c_G_g, &m.agg.P_bar_g, m.agg.pi_e);
    let epsilon_G = rng.randn() * gov.sigma_G;
    let C_G = libm::exp(gov.alpha_G * libm::log(gov.C_G) + gov.beta_G + epsilon_G);
    let J = gov.C_d_j.len();
    let price = sum_by(c_G_g.len(), |g| c_G_g[g] * P_bar_g[g]);
    let C_d_j = vec![C_G / J as f64 * price * (1.0 + pi_e); J];
    (C_G, C_d_j)
}

/// `gov_revenues` (A.59): social security contributions and taxes on wages,
/// consumption (VAT), dividends, corporate profits, household investment,
/// products, production and exports.
pub fn gov_revenues<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let p = &m.prop;
    let P_bar_HH = m.agg.P_bar_HH;
    let (w, f, b) = (&m.w_act, &m.firms, &m.bank);
    let wages_emp: Vec<f64> = (0..w.w_h.len())
        .filter(|&h| w.O_h[h].is_some())
        .map(|h| w.w_h[h])
        .collect();
    let tot_wages_emp = sum(&wages_emp);
    let tot_C_h = sum(&w.C_h) + sum(&m.w_inact.C_h) + sum(&f.C_h) + b.C_h;
    let tot_I_h = sum(&w.I_h) + sum(&m.w_inact.I_h) + sum(&f.I_h) + b.I_h;
    let profits = sum_by(f.Pi_i.len(), |i| pos(f.Pi_i[i])) + pos(b.Pi_k);

    let social_security = (p.tau_SIF + p.tau_SIW) * tot_wages_emp * P_bar_HH;
    let labour_income = p.tau_INC * (1.0 - p.tau_SIW) * P_bar_HH * tot_wages_emp;
    let value_added = p.tau_VAT * tot_C_h;
    let capital_income = p.tau_INC * (1.0 - p.tau_FIRM) * p.theta_DIV * profits;
    let corporate_income = p.tau_FIRM * profits;
    let capital_formation = p.tau_CF * tot_I_h;
    let products = sum_by(f.Y_i.len(), |i| f.tau_Y_i[i] * f.P_i[i] * f.Y_i[i]);
    let production = sum_by(f.Y_i.len(), |i| f.tau_K_i[i] * f.P_i[i] * f.Y_i[i]);
    let export = p.tau_EXPORT * m.rotw.C_l;

    social_security
        + labour_income
        + value_added
        + capital_income
        + corporate_income
        + capital_formation
        + products
        + production
        + export
}

/// `gov_loans` (A.60, A.61): the deficit, social benefits plus consumption
/// plus interest minus revenues, is added to the debt.
pub fn gov_loans<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let p = &m.prop;
    let P_bar_HH = m.agg.P_bar_HH;
    let w = &m.w_act;
    let gov = &m.gov;
    let wages_unemp: Vec<f64> = (0..w.w_h.len())
        .filter(|&h| w.O_h[h].is_none())
        .map(|h| w.w_h[h])
        .collect();
    let tot_wages_unemp = sum(&wages_unemp);
    let social_benefits = p.H_inact as f64 * gov.sb_inact * P_bar_HH
        + p.theta_UB * tot_wages_unemp * P_bar_HH
        + p.H as f64 * gov.sb_other * P_bar_HH;
    let Pi_G = social_benefits + gov.C_j + m.cb.r_G * gov.L_G - gov.Y_G;
    gov.L_G + Pi_G
}

impl<X: Extension> AbstractModel<X> {
    /// `gov_social_benefits(model)`; see
    /// [`crate::base::gov_social_benefits`].
    pub fn gov_social_benefits(&mut self) -> (f64, f64) {
        X::gov_social_benefits(self)
    }

    /// `set_gov_social_benefits!`: `gov.sb_other` and `gov.sb_inact`.
    pub fn set_gov_social_benefits(&mut self) {
        (self.gov.sb_other, self.gov.sb_inact) = self.gov_social_benefits();
    }

    /// `gov_expenditure(model)`; see [`crate::base::gov_expenditure`].
    pub fn gov_expenditure<R: Randomness + ?Sized>(&mut self, rng: &mut R) -> (f64, Vec<f64>) {
        X::gov_expenditure(self, rng)
    }

    /// `set_gov_expenditure!`: `gov.C_G` and `gov.C_d_j`.
    pub fn set_gov_expenditure<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let (C_G, C_d_j) = self.gov_expenditure(rng);
        self.gov.C_G = C_G;
        self.gov.C_d_j.copy_from_slice(&C_d_j);
    }

    /// `gov_revenues(model)`; see [`crate::base::gov_revenues`].
    pub fn gov_revenues(&mut self) -> f64 {
        X::gov_revenues(self)
    }

    /// `set_gov_revenues!`: `gov.Y_G`.
    pub fn set_gov_revenues(&mut self) {
        self.gov.Y_G = self.gov_revenues();
    }

    /// `gov_loans(model)`; see [`crate::base::gov_loans`].
    pub fn gov_loans(&mut self) -> f64 {
        X::gov_loans(self)
    }

    /// `set_gov_loans!`: `gov.L_G`.
    pub fn set_gov_loans(&mut self) {
        self.gov.L_G = self.gov_loans();
    }
}
