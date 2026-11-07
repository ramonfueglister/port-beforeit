//! Bank and central bank, BeforeIT `src/agent_actions/bank.jl` and
//! `central_bank.jl` (appendix A.4 and A.5), with one bank as in the paper.

use crate::julia::{max, pos, sum, sum_by};
use crate::{AbstractModel, Extension};

/// `taylor_rule` (A.69): policy rate by the Taylor rule from euro area
/// growth and inflation. As in the Julia code, the rate never falls below 0;
/// A.69 in the paper and the authors' MATLAB code have no such floor.
#[allow(clippy::too_many_arguments)]
pub fn taylor_rule(
    rho: f64,
    r_bar: f64,
    r_star: f64,
    pi_star: f64,
    xi_pi: f64,
    xi_gamma: f64,
    gamma_EA: f64,
    pi_EA: f64,
) -> f64 {
    let rate = rho * r_bar
        + (1.0 - rho) * (r_star + pi_star + xi_pi * (pi_EA - pi_star) + xi_gamma * gamma_EA);
    pos(rate)
}

/// `central_bank_rate` (A.69): the Taylor rule with the parameters of the
/// central bank and the euro area growth and inflation of the quarter.
pub fn central_bank_rate<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let cb = &m.cb;
    taylor_rule(
        cb.rho,
        cb.r_bar,
        cb.r_star,
        cb.pi_star,
        cb.xi_pi,
        cb.xi_gamma,
        m.rotw.gamma_EA,
        m.rotw.pi_EA,
    )
}

/// `central_bank_profits` (A.73): interest on government debt minus
/// interest on the bank's reserves.
pub fn central_bank_profits<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let (D_k, L_G) = (m.bank.D_k, m.gov.L_G);
    m.cb.r_G * L_G - m.cb.r_bar * D_k
}

/// `central_bank_equity` (A.74).
pub fn central_bank_equity<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let Pi_CB = X::central_bank_profits(m);
    m.cb.E_CB + Pi_CB
}

/// `banks_rate` (A.66): the policy rate plus the risk premium.
pub fn banks_rate<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    m.cb.r_bar + m.prop.mu
}

/// `banks_expected_profits` (A.45): last quarter's profits grown with
/// expected inflation and expected growth.
pub fn banks_expected_profits<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let (pi_e, gamma_e) = (m.agg.pi_e, m.agg.gamma_e);
    m.bank.Pi_k * (1.0 + pi_e) * (1.0 + gamma_e)
}

/// Deposits of all households in the order of the Julia code: workers,
/// inactive persons, firm owners, bank owner.
fn all_households_D_h<X>(m: &AbstractModel<X>) -> Vec<f64> {
    m.w_act
        .D_h
        .iter()
        .chain(&m.w_inact.D_h)
        .chain(&m.firms.D_h)
        .copied()
        .chain(std::iter::once(m.bank.D_h))
        .collect()
}

/// `banks_profits` (A.65): interest on loans and overdrafts plus interest on
/// the net position at the central bank, minus interest on deposits.
pub fn banks_profits<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let f = &m.firms;
    let D_h = all_households_D_h(m);
    let r_terms = sum(&f.L_i)
        + sum_by(f.D_i.len(), |i| max(0.0, -f.D_i[i]))
        + sum_by(D_h.len(), |h| max(0.0, -D_h[h]));
    let r_bar_terms = m.bank.D_k
        - sum_by(f.D_i.len(), |i| max(0.0, f.D_i[i]))
        - sum_by(D_h.len(), |h| max(0.0, D_h[h]));
    m.bank.r * r_terms + m.cb.r_bar * r_bar_terms
}

/// `banks_equity` (A.67): equity plus profits after dividends and corporate
/// tax. The write-off of bad debt in A.67 happens in
/// [`finance_insolvent_firms`].
pub fn banks_equity<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let p = &m.prop;
    let Pi_k = m.bank.Pi_k;
    let DE_k =
        Pi_k - p.theta_DIV * (1.0 - p.tau_FIRM) * max(0.0, Pi_k) - p.tau_FIRM * max(0.0, Pi_k);
    m.bank.E_k + DE_k
}

/// `banks_deposits` (A.68): the net position at the central bank as the
/// balancing item of the bank's balance sheet.
pub fn banks_deposits<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let f = &m.firms;
    let tot_D_h = sum(&m.w_act.D_h) + sum(&m.w_inact.D_h) + sum(&f.D_h) + m.bank.D_h;
    sum(&f.D_i) + tot_D_h + m.bank.E_k - sum(&f.L_i)
}

/// `finance_insolvent_firms!` (A.38 to A.40, and the write-off in A.67): a
/// firm with negative deposits and negative equity is replaced by a new one.
/// The bank writes off the firm's loans and overdraft above
/// `zeta_b · P̄_CF · K_i`, which the new firm keeps as its loans.
pub fn finance_insolvent_firms<X: Extension>(m: &mut AbstractModel<X>) {
    let (P_bar_CF, zeta_b) = (m.agg.P_bar_CF, m.prop.zeta_b);
    let f = &mut m.firms;
    for i in 0..f.len() {
        if f.D_i[i] < 0.0 && f.E_i[i] < 0.0 {
            let refinancing_amount = f.L_i[i] - f.D_i[i] - zeta_b * P_bar_CF * f.K_i[i];
            m.bank.E_k -= refinancing_amount;
            f.E_i[i] += refinancing_amount;
            f.L_i[i] = zeta_b * P_bar_CF * f.K_i[i];
            f.D_i[i] = 0.0;
        }
    }
}

impl<X: Extension> AbstractModel<X> {
    /// `central_bank_rate(model)`; see [`crate::base::central_bank_rate`].
    pub fn central_bank_rate(&mut self) -> f64 {
        X::central_bank_rate(self)
    }

    /// `set_central_bank_rate!`: `cb.r_bar`.
    pub fn set_central_bank_rate(&mut self) {
        self.cb.r_bar = self.central_bank_rate();
    }

    /// `central_bank_profits(cb, model)`; see
    /// [`crate::base::central_bank_profits`].
    pub fn central_bank_profits(&mut self) -> f64 {
        X::central_bank_profits(self)
    }

    /// `central_bank_equity(model)`; see [`crate::base::central_bank_equity`].
    pub fn central_bank_equity(&mut self) -> f64 {
        X::central_bank_equity(self)
    }

    /// `set_central_bank_equity!`: `cb.E_CB`.
    pub fn set_central_bank_equity(&mut self) {
        self.cb.E_CB = self.central_bank_equity();
    }

    /// `banks_rate(model)`; see [`crate::base::banks_rate`].
    pub fn banks_rate(&mut self) -> f64 {
        X::banks_rate(self)
    }

    /// `set_banks_rate!`: `bank.r`.
    pub fn set_banks_rate(&mut self) {
        self.bank.r = self.banks_rate();
    }

    /// `banks_expected_profits(model)`; see
    /// [`crate::base::banks_expected_profits`].
    pub fn banks_expected_profits(&mut self) -> f64 {
        X::banks_expected_profits(self)
    }

    /// `set_banks_expected_profits!`: `bank.Pi_e_k`.
    pub fn set_banks_expected_profits(&mut self) {
        self.bank.Pi_e_k = self.banks_expected_profits();
    }

    /// `banks_profits(model)`; see [`crate::base::banks_profits`].
    pub fn banks_profits(&mut self) -> f64 {
        X::banks_profits(self)
    }

    /// `set_banks_profits!`: `bank.Pi_k`.
    pub fn set_banks_profits(&mut self) {
        self.bank.Pi_k = self.banks_profits();
    }

    /// `banks_equity(model)`; see [`crate::base::banks_equity`].
    pub fn banks_equity(&mut self) -> f64 {
        X::banks_equity(self)
    }

    /// `set_banks_equity!`: `bank.E_k`.
    pub fn set_banks_equity(&mut self) {
        self.bank.E_k = self.banks_equity();
    }

    /// `banks_deposits(model)`; see [`crate::base::banks_deposits`].
    pub fn banks_deposits(&mut self) -> f64 {
        X::banks_deposits(self)
    }

    /// `set_banks_deposits!`: `bank.D_k`.
    pub fn set_banks_deposits(&mut self) {
        self.bank.D_k = self.banks_deposits();
    }

    /// `finance_insolvent_firms!(model)`; see
    /// [`crate::base::finance_insolvent_firms`].
    pub fn finance_insolvent_firms(&mut self) {
        X::finance_insolvent_firms(self)
    }
}
