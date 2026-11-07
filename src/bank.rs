//! Bank and central bank, BeforeIT `src/agent_actions/bank.jl` and
//! `central_bank.jl` (appendix A.4 and A.5), with one bank as in the paper.

use crate::julia::{max, pos, sum, sum_by};
use crate::{CentralBank, Model};

impl CentralBank {
    /// `taylor_rule` (A.69): policy rate by the Taylor rule from euro area
    /// growth and inflation. As in the Julia code, the rate never falls
    /// below 0; A.69 in the paper and the authors' MATLAB code have no such
    /// floor.
    pub fn taylor_rule(&self, gamma_EA: f64, pi_EA: f64) -> f64 {
        let (rho, pi_star) = (self.rho, self.pi_star);
        pos(rho * self.r_bar
            + (1.0 - rho)
                * (self.r_star
                    + pi_star
                    + self.xi_pi * (pi_EA - pi_star)
                    + self.xi_gamma * gamma_EA))
    }
}

impl Model {
    /// Deposits of all households in the order of the Julia code: workers,
    /// inactive persons, firm owners, bank owner.
    fn all_households_D_h(&self) -> impl Iterator<Item = f64> + '_ {
        self.w_act
            .D_h
            .iter()
            .chain(&self.w_inact.D_h)
            .chain(&self.firms.D_h)
            .copied()
            .chain(std::iter::once(self.bank.D_h))
    }

    /// `banks_rate` (A.66).
    pub fn set_banks_rate(&mut self) {
        self.bank.r = self.cb.r_bar + self.prop.mu;
    }

    /// `banks_expected_profits` (A.45): last quarter's profits grown with
    /// expected inflation and expected growth.
    pub fn set_banks_expected_profits(&mut self) {
        let (pi_e, gamma_e) = (self.agg.pi_e, self.agg.gamma_e);
        self.bank.Pi_e_k = self.bank.Pi_k * (1.0 + pi_e) * (1.0 + gamma_e);
    }

    /// `banks_profits` (A.65): interest on loans and overdrafts plus interest
    /// on the net position at the central bank, minus interest on deposits.
    pub fn set_banks_profits(&mut self) {
        let f = &self.firms;
        let D_h: Vec<f64> = self.all_households_D_h().collect();
        let r_terms = sum(&f.L_i)
            + sum_by(f.D_i.len(), |i| max(0.0, -f.D_i[i]))
            + sum_by(D_h.len(), |h| max(0.0, -D_h[h]));
        let r_bar_terms = self.bank.D_k
            - sum_by(f.D_i.len(), |i| max(0.0, f.D_i[i]))
            - sum_by(D_h.len(), |h| max(0.0, D_h[h]));
        self.bank.Pi_k = self.bank.r * r_terms + self.cb.r_bar * r_bar_terms;
    }

    /// `banks_equity` (A.67): profits after dividends and corporate tax. The
    /// write-off of bad debt in A.67 happens in
    /// [`Model::finance_insolvent_firms`].
    pub fn set_banks_equity(&mut self) {
        let p = &self.prop;
        let Pi_k = self.bank.Pi_k;
        let DE_k =
            Pi_k - p.theta_DIV * (1.0 - p.tau_FIRM) * max(0.0, Pi_k) - p.tau_FIRM * max(0.0, Pi_k);
        self.bank.E_k += DE_k;
    }

    /// `banks_deposits` (A.68): the net position at the central bank as the
    /// balancing item of the bank's balance sheet.
    pub fn set_banks_deposits(&mut self) {
        let f = &self.firms;
        let tot_D_h = sum(&self.w_act.D_h) + sum(&self.w_inact.D_h) + sum(&f.D_h) + self.bank.D_h;
        self.bank.D_k = sum(&f.D_i) + tot_D_h + self.bank.E_k - sum(&f.L_i);
    }

    /// `central_bank_rate` (A.69).
    pub fn set_central_bank_rate(&mut self) {
        self.cb.r_bar = self.cb.taylor_rule(self.rotw.gamma_EA, self.rotw.pi_EA);
    }

    /// `central_bank_equity` (A.73, A.74).
    pub fn set_central_bank_equity(&mut self) {
        let Pi_CB = self.cb.r_G * self.gov.L_G - self.cb.r_bar * self.bank.D_k;
        self.cb.E_CB += Pi_CB;
    }
}
