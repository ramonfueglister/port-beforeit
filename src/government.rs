//! Government, BeforeIT `src/agent_actions/government.jl` (appendix A.3).

use crate::Model;
use crate::julia::{pos, sum, sum_by};
use crate::random::Randomness;

impl Model {
    /// `gov_social_benefits` (A.42, A.43): social benefits grow with expected
    /// growth.
    pub fn set_gov_social_benefits(&mut self) {
        let gamma_e = self.agg.gamma_e;
        self.gov.sb_other *= 1.0 + gamma_e;
        self.gov.sb_inact *= 1.0 + gamma_e;
    }

    /// `gov_expenditure` (A.55, A.56): real government consumption by an
    /// AR(1) rule, split equally among the government entities as nominal
    /// budgets at expected prices.
    pub fn set_gov_expenditure<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let (c_G_g, P_bar_g) = (&self.prop.c_G_g, &self.agg.P_bar_g);
        let price = sum_by(c_G_g.len(), |g| c_G_g[g] * P_bar_g[g]);
        let pi_e = self.agg.pi_e;
        let gov = &mut self.gov;
        let epsilon_G = rng.randn() * gov.sigma_G;
        let C_G = libm::exp(gov.alpha_G * libm::log(gov.C_G) + gov.beta_G + epsilon_G);
        let C_d_j = C_G / gov.C_d_j.len() as f64 * price * (1.0 + pi_e);
        gov.C_G = C_G;
        gov.C_d_j.fill(C_d_j);
    }

    /// `gov_revenues` (A.59): social security contributions and taxes on
    /// wages, consumption (VAT), dividends, corporate profits, household
    /// investment, products, production and exports.
    pub fn set_gov_revenues(&mut self) {
        let p = &self.prop;
        let P_bar_HH = self.agg.P_bar_HH;
        let (w, f, b) = (&self.w_act, &self.firms, &self.bank);
        let wages_emp: Vec<f64> = (0..w.len())
            .filter(|&h| w.O_h[h].is_some())
            .map(|h| w.w_h[h])
            .collect();
        let tot_wages_emp = sum(&wages_emp);
        let tot_C_h = sum(&w.C_h) + sum(&self.w_inact.C_h) + sum(&f.C_h) + b.C_h;
        let tot_I_h = sum(&w.I_h) + sum(&self.w_inact.I_h) + sum(&f.I_h) + b.I_h;
        let profits = sum_by(f.Pi_i.len(), |i| pos(f.Pi_i[i])) + pos(b.Pi_k);

        let social_security = (p.tau_SIF + p.tau_SIW) * tot_wages_emp * P_bar_HH;
        let labour_income = p.tau_INC * (1.0 - p.tau_SIW) * P_bar_HH * tot_wages_emp;
        let value_added = p.tau_VAT * tot_C_h;
        let capital_income = p.tau_INC * (1.0 - p.tau_FIRM) * p.theta_DIV * profits;
        let corporate_income = p.tau_FIRM * profits;
        let capital_formation = p.tau_CF * tot_I_h;
        let products = sum_by(f.Y_i.len(), |i| f.tau_Y_i[i] * f.P_i[i] * f.Y_i[i]);
        let production = sum_by(f.Y_i.len(), |i| f.tau_K_i[i] * f.P_i[i] * f.Y_i[i]);
        let export = p.tau_EXPORT * self.rotw.C_l;

        self.gov.Y_G = social_security
            + labour_income
            + value_added
            + capital_income
            + corporate_income
            + capital_formation
            + products
            + production
            + export;
    }

    /// `gov_loans` (A.60, A.61): the deficit, social benefits plus
    /// consumption plus interest minus revenues, is added to the debt.
    pub fn set_gov_loans(&mut self) {
        let p = &self.prop;
        let P_bar_HH = self.agg.P_bar_HH;
        let w = &self.w_act;
        let gov = &self.gov;
        let wages_unemp: Vec<f64> = (0..w.len())
            .filter(|&h| w.O_h[h].is_none())
            .map(|h| w.w_h[h])
            .collect();
        let tot_wages_unemp = sum(&wages_unemp);
        let social_benefits = p.H_inact as f64 * gov.sb_inact * P_bar_HH
            + p.theta_UB * tot_wages_unemp * P_bar_HH
            + p.H as f64 * gov.sb_other * P_bar_HH;
        let Pi_G = social_benefits + gov.C_j + self.cb.r_G * gov.L_G - gov.Y_G;
        self.gov.L_G += Pi_G;
    }
}
