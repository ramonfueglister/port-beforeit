//! Households, BeforeIT `src/agent_actions/households.jl` (appendix A.2).
//! The households are the workers, the inactive persons, the firm owners and
//! the bank owner.

use crate::Model;
use crate::julia::max;

impl Model {
    /// `update_workers_wages!` (A.41): employed workers get the wage of their
    /// firm; the unemployed keep their last wage.
    pub fn update_workers_wages(&mut self) {
        let w = &mut self.w_act;
        for h in 0..w.len() {
            if let Some(i) = w.O_h[h] {
                w.w_h[h] = self.firms.w_i[i];
            }
        }
    }

    /// `households_income_act` (A.44 and A.53, employed or unemployed),
    /// expected or realised.
    fn households_income_act(&self, expected: bool) -> Vec<f64> {
        let p = &self.prop;
        let sb_other = self.gov.sb_other;
        let P_bar_HH = self.agg.P_bar_HH;
        let pi_e = if expected { self.agg.pi_e } else { 0.0 };
        let w = &self.w_act;
        (0..w.len())
            .map(|h| {
                if w.O_h[h].is_some() {
                    (w.w_h[h] * (1.0 - p.tau_SIW - p.tau_INC * (1.0 - p.tau_SIW)) + sb_other)
                        * P_bar_HH
                        * (1.0 + pi_e)
                } else {
                    (p.theta_UB * w.w_h[h] + sb_other) * P_bar_HH * (1.0 + pi_e)
                }
            })
            .collect()
    }

    /// `households_income_inact` (A.44 and A.53, inactive), expected or
    /// realised.
    fn households_income_inact(&self, expected: bool) -> Vec<f64> {
        let P_bar_HH = self.agg.P_bar_HH;
        let pi_e = if expected { self.agg.pi_e } else { 0.0 };
        let y = (self.gov.sb_inact + self.gov.sb_other) * P_bar_HH * (1.0 + pi_e);
        vec![y; self.w_inact.len()]
    }

    /// Income of an owner household from the profits `Pi` (A.44 and A.53).
    fn owner_income(&self, Pi: f64, expected: bool) -> f64 {
        let p = &self.prop;
        let P_bar_HH = self.agg.P_bar_HH;
        let pi_e = if expected { self.agg.pi_e } else { 0.0 };
        p.theta_DIV * (1.0 - p.tau_INC) * (1.0 - p.tau_FIRM) * max(0.0, Pi)
            + self.gov.sb_other * P_bar_HH * (1.0 + pi_e)
    }

    /// `households_income_firms`: expected income from the firm's expected
    /// profits, realised income from its realised profits.
    fn households_income_firms(&self, expected: bool) -> Vec<f64> {
        let Pi = if expected {
            &self.firms.Pi_e_i
        } else {
            &self.firms.Pi_i
        };
        Pi.iter()
            .map(|&pi| self.owner_income(pi, expected))
            .collect()
    }

    /// `households_income_banks`.
    fn households_income_banks(&self, expected: bool) -> f64 {
        let Pi = if expected {
            self.bank.Pi_e_k
        } else {
            self.bank.Pi_k
        };
        self.owner_income(Pi, expected)
    }

    /// Consumption budget (A.46) and investment budget (A.49) from expected
    /// income.
    fn budget(&self, Y_e_h: f64) -> (f64, f64) {
        let p = &self.prop;
        (
            p.psi * Y_e_h / (1.0 + p.tau_VAT),
            p.psi_H * Y_e_h / (1.0 + p.tau_CF),
        )
    }

    /// `households_budget_act`.
    pub fn set_households_budget_act(&mut self) {
        let Y_e_h = self.households_income_act(true);
        for (h, &y) in Y_e_h.iter().enumerate() {
            (self.w_act.C_d_h[h], self.w_act.I_d_h[h]) = self.budget(y);
        }
    }

    /// `households_budget_inact`.
    pub fn set_households_budget_inact(&mut self) {
        let Y_e_h = self.households_income_inact(true);
        for (h, &y) in Y_e_h.iter().enumerate() {
            (self.w_inact.C_d_h[h], self.w_inact.I_d_h[h]) = self.budget(y);
        }
    }

    /// `households_budget_firms`.
    pub fn set_households_budget_firms(&mut self) {
        let Y_e_h = self.households_income_firms(true);
        for (i, &y) in Y_e_h.iter().enumerate() {
            (self.firms.C_d_h[i], self.firms.I_d_h[i]) = self.budget(y);
        }
    }

    /// `households_budget_banks`.
    pub fn set_households_budget_banks(&mut self) {
        let Y_e_h = self.households_income_banks(true);
        (self.bank.C_d_h, self.bank.I_d_h) = self.budget(Y_e_h);
    }

    /// `set_households_income_act!` (A.53).
    pub fn set_households_income_act(&mut self) {
        self.w_act.Y_h = self.households_income_act(false);
    }

    /// `set_households_income_inact!` (A.53).
    pub fn set_households_income_inact(&mut self) {
        self.w_inact.Y_h = self.households_income_inact(false);
    }

    /// `set_households_income_firms!` (A.53).
    pub fn set_households_income_firms(&mut self) {
        self.firms.Y_h = self.households_income_firms(false);
    }

    /// `set_households_income_banks!` (A.53).
    pub fn set_households_income_banks(&mut self) {
        self.bank.Y_h = self.households_income_banks(false);
    }

    /// `households_deposits` (A.54) for one household.
    fn deposits(&self, D_h: f64, Y_h: f64, C_h: f64, I_h: f64) -> f64 {
        let p = &self.prop;
        let (r_bar, r) = (self.cb.r_bar, self.bank.r);
        let DD_h = Y_h - (1.0 + p.tau_VAT) * C_h - (1.0 + p.tau_CF) * I_h + r_bar * max(0.0, D_h)
            - r * max(0.0, -D_h);
        D_h + DD_h
    }

    /// `set_households_deposits_act!` (A.54).
    pub fn set_households_deposits_act(&mut self) {
        for h in 0..self.w_act.len() {
            let w = &self.w_act;
            self.w_act.D_h[h] = self.deposits(w.D_h[h], w.Y_h[h], w.C_h[h], w.I_h[h]);
        }
    }

    /// `set_households_deposits_inact!` (A.54).
    pub fn set_households_deposits_inact(&mut self) {
        for h in 0..self.w_inact.len() {
            let w = &self.w_inact;
            self.w_inact.D_h[h] = self.deposits(w.D_h[h], w.Y_h[h], w.C_h[h], w.I_h[h]);
        }
    }

    /// `set_households_deposits_firms!` (A.54).
    pub fn set_households_deposits_firms(&mut self) {
        for i in 0..self.firms.len() {
            let f = &self.firms;
            self.firms.D_h[i] = self.deposits(f.D_h[i], f.Y_h[i], f.C_h[i], f.I_h[i]);
        }
    }

    /// `set_households_deposits_banks!` (A.54).
    pub fn set_households_deposits_banks(&mut self) {
        let b = &self.bank;
        self.bank.D_h = self.deposits(b.D_h, b.Y_h, b.C_h, b.I_h);
    }
}
