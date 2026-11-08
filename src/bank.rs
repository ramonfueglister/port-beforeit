//! Banks and central bank, BeforeIT `src/agent_actions/bank.jl` and
//! `central_bank.jl` (appendix A.4 and A.5). The paper has one bank;
//! BeforeIT.jl allows several (`n_banks`), each with its own customers.

use crate::agents::gather;
use crate::julia::{max, pos, sum};
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
/// interest on the banks' reserves.
pub fn central_bank_profits<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let (D_k, L_G) = (sum(&m.banks.D_k), m.gov.L_G);
    m.cb.r_G * L_G - m.cb.r_bar * D_k
}

/// `central_bank_equity` (A.74).
pub fn central_bank_equity<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let Pi_CB = X::central_bank_profits(m);
    m.cb.E_CB + Pi_CB
}

/// `bank_exposure(model, bank_id; expected, overdrafts)`: the loans of the
/// bank's firms, expected loans `L_e_i` if `expected`, plus, if
/// `overdrafts`, the overdrafts (negative deposits) of its firms, their
/// owners, its workers, its inactive persons and its owner.
pub fn bank_exposure<X: Extension>(
    m: &AbstractModel<X>,
    bank_id: usize,
    expected: bool,
    overdrafts: bool,
) -> f64 {
    let (f, b) = (&m.firms, &m.banks);
    let L = if expected { &f.L_e_i } else { &f.L_i };
    // From left to right, as Julia's `sum` with `init`.
    let mut exposure = gather(&b.firms[bank_id], L, |id| f.index(id))
        .iter()
        .fold(0.0, |s, x| s + x);
    if !overdrafts {
        return exposure;
    }
    for &id in &b.firms[bank_id] {
        let i = f.index(id);
        exposure += max(0.0, -f.D_i[i]) + max(0.0, -f.D_h[i]);
    }
    for (w, ids) in [
        (&m.w_act, &b.w_act[bank_id]),
        (&m.w_inact, &b.w_inact[bank_id]),
    ] {
        for &id in ids {
            exposure += max(0.0, -w.D_h[w.index(id)]);
        }
    }
    exposure += max(0.0, -b.D_h[bank_id]);
    exposure
}

/// `banks_rate` (A.66): the policy rate plus the risk premium, the same for
/// every bank.
pub fn banks_rate<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    m.cb.r_bar + m.prop.mu
}

/// `banks_expected_profits` (A.45): last quarter's profits grown with
/// expected inflation and expected growth.
pub fn banks_expected_profits<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let (pi_e, gamma_e) = (m.agg.pi_e, m.agg.gamma_e);
    m.banks
        .Pi_k
        .iter()
        .map(|Pi| Pi * (1.0 + pi_e) * (1.0 + gamma_e))
        .collect()
}

/// `banks_profits` (A.65): for each bank, interest on the loans and
/// overdrafts of its customers plus interest on its net position at the
/// central bank, minus interest on the deposits of its customers and its
/// owner.
pub fn banks_profits<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let (f, b, r_bar) = (&m.firms, &m.banks, m.cb.r_bar);
    let (w_act, w_inact) = (&m.w_act, &m.w_inact);
    (0..m.prop.n_banks)
        .map(|k| {
            let index = |id| f.index(id);
            let L_i = gather(&b.firms[k], &f.L_i, index);
            let D_i = gather(&b.firms[k], &f.D_i, index);
            let D_h_firms = gather(&b.firms[k], &f.D_h, index);
            let D_h_act = gather(&b.w_act[k], &w_act.D_h, |id| w_act.index(id));
            let D_h_inact = gather(&b.w_inact[k], &w_inact.D_h, |id| w_inact.index(id));
            // The households in the order of the Julia code: workers,
            // inactive persons, firm owners, bank owner.
            let mut D_h = D_h_act;
            D_h.extend(D_h_inact);
            D_h.extend(D_h_firms);
            D_h.push(b.D_h[k]);

            let max0_minus = |v: &[f64]| -> Vec<f64> { v.iter().map(|&x| max(0.0, -x)).collect() };
            let max0 = |v: &[f64]| -> Vec<f64> { v.iter().map(|&x| max(0.0, x)).collect() };
            let r_terms = sum(&L_i) + sum(&max0_minus(&D_i)) + sum(&max0_minus(&D_h));
            let r_bar_terms = b.D_k[k] - sum(&max0(&D_i)) - sum(&max0(&D_h));
            b.r[k] * r_terms + r_bar * r_bar_terms
        })
        .collect()
}

/// `banks_equity` (A.67): equity plus profits after dividends and corporate
/// tax. The write-off of bad debt in A.67 happens in
/// [`finance_insolvent_firms`].
pub fn banks_equity<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let p = &m.prop;
    let b = &m.banks;
    (0..b.Pi_k.len())
        .map(|k| {
            let Pi_k = b.Pi_k[k];
            let DE_k = Pi_k
                - p.theta_DIV * (1.0 - p.tau_FIRM) * max(0.0, Pi_k)
                - p.tau_FIRM * max(0.0, Pi_k);
            b.E_k[k] + DE_k
        })
        .collect()
}

/// `banks_deposits` (A.68): each bank's net position at the central bank as
/// the balancing item of its balance sheet.
pub fn banks_deposits<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let (f, b) = (&m.firms, &m.banks);
    let (w_act, w_inact) = (&m.w_act, &m.w_inact);
    (0..m.prop.n_banks)
        .map(|k| {
            let index = |id| f.index(id);
            let fL_i = gather(&b.firms[k], &f.L_i, index);
            let fD_i = gather(&b.firms[k], &f.D_i, index);
            let fD_h = gather(&b.firms[k], &f.D_h, index);
            let waD_h = gather(&b.w_act[k], &w_act.D_h, |id| w_act.index(id));
            let wiD_h = gather(&b.w_inact[k], &w_inact.D_h, |id| w_inact.index(id));
            let tot_D_h = sum(&waD_h) + sum(&wiD_h) + sum(&fD_h) + b.D_h[k];
            sum(&fD_i) + tot_D_h + b.E_k[k] - sum(&fL_i)
        })
        .collect()
}

/// `finance_insolvent_firms!` (A.38 to A.40, and the write-off in A.67): a
/// firm with negative deposits and negative equity is replaced by a new one.
/// The firm's bank writes off the firm's loans and overdraft above
/// `zeta_b · P̄_CF · K_i`, which the new firm keeps as its loans.
pub fn finance_insolvent_firms<X: Extension>(m: &mut AbstractModel<X>) {
    let (P_bar_CF, zeta_b) = (m.agg.P_bar_CF, m.prop.zeta_b);
    let (f, b) = (&mut m.firms, &mut m.banks);
    for i in 0..m.prop.I {
        if f.D_i[i] < 0.0 && f.E_i[i] < 0.0 {
            let bank_id = f.B_i[i];
            let refinancing_amount = f.L_i[i] - f.D_i[i] - zeta_b * P_bar_CF * f.K_i[i];
            b.E_k[bank_id] -= refinancing_amount;
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

    /// `set_banks_rate!`: `banks.r` of every bank.
    pub fn set_banks_rate(&mut self) {
        let r = self.banks_rate();
        self.banks.r.fill(r);
    }

    /// `bank_exposure(model, bank_id; expected, overdrafts)`; see
    /// [`crate::base::bank_exposure`].
    pub fn bank_exposure(&self, bank_id: usize, expected: bool, overdrafts: bool) -> f64 {
        X::bank_exposure(self, bank_id, expected, overdrafts)
    }

    /// `banks_expected_profits(model)`; see
    /// [`crate::base::banks_expected_profits`].
    pub fn banks_expected_profits(&mut self) -> Vec<f64> {
        X::banks_expected_profits(self)
    }

    /// `set_banks_expected_profits!`: `banks.Pi_e_k`.
    pub fn set_banks_expected_profits(&mut self) {
        let Pi_e_k = self.banks_expected_profits();
        self.banks.Pi_e_k.copy_from_slice(&Pi_e_k);
    }

    /// `banks_profits(model)`; see [`crate::base::banks_profits`].
    pub fn banks_profits(&mut self) -> Vec<f64> {
        X::banks_profits(self)
    }

    /// `set_banks_profits!`: `banks.Pi_k`.
    pub fn set_banks_profits(&mut self) {
        let Pi_k = self.banks_profits();
        self.banks.Pi_k.copy_from_slice(&Pi_k);
    }

    /// `banks_equity(model)`; see [`crate::base::banks_equity`].
    pub fn banks_equity(&mut self) -> Vec<f64> {
        X::banks_equity(self)
    }

    /// `set_banks_equity!`: `banks.E_k`.
    pub fn set_banks_equity(&mut self) {
        let E_k = self.banks_equity();
        self.banks.E_k.copy_from_slice(&E_k);
    }

    /// `banks_deposits(model)`; see [`crate::base::banks_deposits`].
    pub fn banks_deposits(&mut self) -> Vec<f64> {
        X::banks_deposits(self)
    }

    /// `set_banks_deposits!`: `banks.D_k`.
    pub fn set_banks_deposits(&mut self) {
        let D_k = self.banks_deposits();
        self.banks.D_k.copy_from_slice(&D_k);
    }

    /// `finance_insolvent_firms!(model)`; see
    /// [`crate::base::finance_insolvent_firms`].
    pub fn finance_insolvent_firms(&mut self) {
        X::finance_insolvent_firms(self)
    }
}
