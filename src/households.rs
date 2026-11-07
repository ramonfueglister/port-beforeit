//! Households, BeforeIT `src/agent_actions/households.jl` (appendix A.2).
//! The households are the workers, the inactive persons, the firm owners and
//! the bank owner.

use crate::agents::Households;
use crate::julia::max;
use crate::{AbstractModel, Extension};

/// `update_workers_wages!` (A.41): employed workers get the wage of their
/// firm; the unemployed keep their last wage.
pub fn update_workers_wages<X: Extension>(m: &mut AbstractModel<X>) {
    let w = &mut m.w_act;
    for h in 0..w.len() {
        if let Some(i) = w.O_h[h] {
            w.w_h[h] = m.firms.w_i[i];
        }
    }
}

/// `households_income_act` (A.44 and A.53, employed or unemployed),
/// expected or realised.
pub fn households_income_act<X: Extension>(m: &mut AbstractModel<X>, expected: bool) -> Vec<f64> {
    let p = &m.prop;
    let sb_other = m.gov.sb_other;
    let P_bar_HH = m.agg.P_bar_HH;
    let pi_e = if expected { m.agg.pi_e } else { 0.0 };
    let w = &m.w_act;
    (0..w.w_h.len())
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
pub fn households_income_inact<X: Extension>(m: &mut AbstractModel<X>, expected: bool) -> Vec<f64> {
    let P_bar_HH = m.agg.P_bar_HH;
    let pi_e = if expected { m.agg.pi_e } else { 0.0 };
    let y = (m.gov.sb_inact + m.gov.sb_other) * P_bar_HH * (1.0 + pi_e);
    vec![y; m.w_inact.len()]
}

/// Income of an owner household from the profits `Pi` (A.44 and A.53).
fn owner_income<X>(m: &AbstractModel<X>, Pi: f64, expected: bool) -> f64 {
    let p = &m.prop;
    let P_bar_HH = m.agg.P_bar_HH;
    let pi_e = if expected { m.agg.pi_e } else { 0.0 };
    p.theta_DIV * (1.0 - p.tau_INC) * (1.0 - p.tau_FIRM) * max(0.0, Pi)
        + m.gov.sb_other * P_bar_HH * (1.0 + pi_e)
}

/// `households_income_firms`: expected income from the firm's expected
/// profits, realised income from its realised profits.
pub fn households_income_firms<X: Extension>(m: &mut AbstractModel<X>, expected: bool) -> Vec<f64> {
    let Pi = if expected {
        &m.firms.Pi_e_i
    } else {
        &m.firms.Pi_i
    };
    Pi.iter().map(|&pi| owner_income(m, pi, expected)).collect()
}

/// `households_income_banks`: income of the bank owner from the bank's
/// expected or realised profits.
pub fn households_income_banks<X: Extension>(m: &mut AbstractModel<X>, expected: bool) -> f64 {
    let Pi = if expected { m.bank.Pi_e_k } else { m.bank.Pi_k };
    owner_income(m, Pi, expected)
}

/// Consumption budget (A.46) and investment budget (A.49) from expected
/// income.
fn budget<X>(m: &AbstractModel<X>, Y_e_h: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let p = &m.prop;
    let C_d_h = Y_e_h
        .iter()
        .map(|y| p.psi * y / (1.0 + p.tau_VAT))
        .collect();
    let I_d_h = Y_e_h
        .iter()
        .map(|y| p.psi_H * y / (1.0 + p.tau_CF))
        .collect();
    (C_d_h, I_d_h)
}

/// `households_budget_act` (A.46, A.49).
pub fn households_budget_act<X: Extension>(m: &mut AbstractModel<X>) -> (Vec<f64>, Vec<f64>) {
    let Y_e_h = X::households_income_act(m, true);
    budget(m, &Y_e_h)
}

/// `households_budget_inact` (A.46, A.49).
pub fn households_budget_inact<X: Extension>(m: &mut AbstractModel<X>) -> (Vec<f64>, Vec<f64>) {
    let Y_e_h = X::households_income_inact(m, true);
    budget(m, &Y_e_h)
}

/// `households_budget_firms` (A.46, A.49).
pub fn households_budget_firms<X: Extension>(m: &mut AbstractModel<X>) -> (Vec<f64>, Vec<f64>) {
    let Y_e_h = X::households_income_firms(m, true);
    budget(m, &Y_e_h)
}

/// `households_budget_banks` (A.46, A.49).
pub fn households_budget_banks<X: Extension>(m: &mut AbstractModel<X>) -> (f64, f64) {
    let Y_e_h = X::households_income_banks(m, true);
    let (C_d_h, I_d_h) = budget(m, &[Y_e_h]);
    (C_d_h[0], I_d_h[0])
}

/// `households_deposits(households, r, model)` (A.54): income minus
/// consumption and investment including taxes, plus interest on deposits,
/// minus interest at the lending rates `r` on overdrafts.
pub fn households_deposits<X: Extension, H: Households + ?Sized>(
    m: &AbstractModel<X>,
    households: &H,
    r: &[f64],
) -> Vec<f64> {
    let p = &m.prop;
    let r_bar = m.cb.r_bar;
    let (D_h, Y_h, C_h, I_h) = (
        households.D_h(),
        households.Y_h(),
        households.C_h(),
        households.I_h(),
    );
    (0..D_h.len())
        .map(|h| {
            let DD_h = Y_h[h] - (1.0 + p.tau_VAT) * C_h[h] - (1.0 + p.tau_CF) * I_h[h]
                + r_bar * max(0.0, D_h[h])
                - r[h] * max(0.0, -D_h[h]);
            D_h[h] + DD_h
        })
        .collect()
}

impl<X: Extension> AbstractModel<X> {
    /// `update_workers_wages!(model)`; see
    /// [`crate::base::update_workers_wages`].
    pub fn update_workers_wages(&mut self) {
        X::update_workers_wages(self)
    }

    /// `households_income_act(model; expected)`; see
    /// [`crate::base::households_income_act`].
    pub fn households_income_act(&mut self, expected: bool) -> Vec<f64> {
        X::households_income_act(self, expected)
    }

    /// `set_households_income_act!(model; expected)`: `w_act.Y_h`. `step`
    /// stores the realised income (`expected = false`).
    pub fn set_households_income_act(&mut self, expected: bool) {
        let Y_h = self.households_income_act(expected);
        self.w_act.Y_h.copy_from_slice(&Y_h);
    }

    /// `households_income_inact(model; expected)`; see
    /// [`crate::base::households_income_inact`].
    pub fn households_income_inact(&mut self, expected: bool) -> Vec<f64> {
        X::households_income_inact(self, expected)
    }

    /// `set_households_income_inact!(model; expected)`: `w_inact.Y_h`.
    pub fn set_households_income_inact(&mut self, expected: bool) {
        let Y_h = self.households_income_inact(expected);
        self.w_inact.Y_h.copy_from_slice(&Y_h);
    }

    /// `households_income_firms(model; expected)`; see
    /// [`crate::base::households_income_firms`].
    pub fn households_income_firms(&mut self, expected: bool) -> Vec<f64> {
        X::households_income_firms(self, expected)
    }

    /// `set_households_income_firms!(model; expected)`: `firms.Y_h`.
    pub fn set_households_income_firms(&mut self, expected: bool) {
        let Y_h = self.households_income_firms(expected);
        self.firms.Y_h.copy_from_slice(&Y_h);
    }

    /// `households_income_banks(model; expected)`; see
    /// [`crate::base::households_income_banks`].
    pub fn households_income_banks(&mut self, expected: bool) -> f64 {
        X::households_income_banks(self, expected)
    }

    /// `set_households_income_banks!(model; expected)`: `bank.Y_h`.
    pub fn set_households_income_banks(&mut self, expected: bool) {
        self.bank.Y_h = self.households_income_banks(expected);
    }

    /// `households_budget_act(model)`; see
    /// [`crate::base::households_budget_act`].
    pub fn households_budget_act(&mut self) -> (Vec<f64>, Vec<f64>) {
        X::households_budget_act(self)
    }

    /// `set_households_budget_act!`: `w_act.C_d_h` and `w_act.I_d_h`.
    pub fn set_households_budget_act(&mut self) {
        let (C_d_h, I_d_h) = self.households_budget_act();
        self.w_act.C_d_h.copy_from_slice(&C_d_h);
        self.w_act.I_d_h.copy_from_slice(&I_d_h);
    }

    /// `households_budget_inact(model)`; see
    /// [`crate::base::households_budget_inact`].
    pub fn households_budget_inact(&mut self) -> (Vec<f64>, Vec<f64>) {
        X::households_budget_inact(self)
    }

    /// `set_households_budget_inact!`: `w_inact.C_d_h` and `w_inact.I_d_h`.
    pub fn set_households_budget_inact(&mut self) {
        let (C_d_h, I_d_h) = self.households_budget_inact();
        self.w_inact.C_d_h.copy_from_slice(&C_d_h);
        self.w_inact.I_d_h.copy_from_slice(&I_d_h);
    }

    /// `households_budget_firms(model)`; see
    /// [`crate::base::households_budget_firms`].
    pub fn households_budget_firms(&mut self) -> (Vec<f64>, Vec<f64>) {
        X::households_budget_firms(self)
    }

    /// `set_households_budget_firms!`: `firms.C_d_h` and `firms.I_d_h`.
    pub fn set_households_budget_firms(&mut self) {
        let (C_d_h, I_d_h) = self.households_budget_firms();
        self.firms.C_d_h.copy_from_slice(&C_d_h);
        self.firms.I_d_h.copy_from_slice(&I_d_h);
    }

    /// `households_budget_banks(model)`; see
    /// [`crate::base::households_budget_banks`].
    pub fn households_budget_banks(&mut self) -> (f64, f64) {
        X::households_budget_banks(self)
    }

    /// `set_households_budget_banks!`: `bank.C_d_h` and `bank.I_d_h`.
    pub fn set_households_budget_banks(&mut self) {
        (self.bank.C_d_h, self.bank.I_d_h) = self.households_budget_banks();
    }

    /// `households_deposits(households, r, model)`; see
    /// [`crate::base::households_deposits`].
    pub fn households_deposits<H: Households + ?Sized>(
        &self,
        households: &H,
        r: &[f64],
    ) -> Vec<f64> {
        X::households_deposits(self, households, r)
    }

    /// `set_households_deposits_act!` (A.54): `w_act.D_h`.
    pub fn set_households_deposits_act(&mut self) {
        let r = vec![self.bank.r; self.w_act.D_h.len()];
        let D_h = self.households_deposits(&self.w_act, &r);
        self.w_act.D_h.copy_from_slice(&D_h);
    }

    /// `set_households_deposits_inact!` (A.54): `w_inact.D_h`.
    pub fn set_households_deposits_inact(&mut self) {
        let r = vec![self.bank.r; self.w_inact.D_h.len()];
        let D_h = self.households_deposits(&self.w_inact, &r);
        self.w_inact.D_h.copy_from_slice(&D_h);
    }

    /// `set_households_deposits_firms!` (A.54): `firms.D_h`.
    pub fn set_households_deposits_firms(&mut self) {
        let r = vec![self.bank.r; self.firms.D_h.len()];
        let D_h = self.households_deposits(&self.firms, &r);
        self.firms.D_h.copy_from_slice(&D_h);
    }

    /// `set_households_deposits_banks!` (A.54): `bank.D_h`.
    pub fn set_households_deposits_banks(&mut self) {
        let D_h = self.households_deposits(&self.bank, &[self.bank.r]);
        self.bank.D_h = D_h[0];
    }
}
