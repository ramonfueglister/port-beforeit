//! Firms, BeforeIT `src/agent_actions/firms.jl` (appendix A.1), and
//! `finance_insolvent_firms!` from `src/agent_actions/bank.jl`.

use crate::Model;
use crate::julia::{max, min, pos, round, round_int, sum_by};

impl Model {
    /// Σ_g a_sg[g][s] · P̄_g for each sector s, as in A.9 and A.36: the
    /// price of one unit of the sector's bundle of intermediate inputs.
    fn material_price_by_sector(&self) -> Vec<f64> {
        let G = self.prop.G;
        let (a_sg, P_bar_g) = (&self.prop.a_sg, &self.agg.P_bar_g);
        (0..G)
            .map(|s| sum_by(G, |g| a_sg[g][s] * P_bar_g[g]))
            .collect()
    }

    /// `firms_expectations_and_decisions` (A.5, A.8, A.9, A.14, A.18, A.22,
    /// A.27 to A.29, A.63): planned supply, price, desired investment,
    /// intermediate inputs and employees, expected profits, desired new
    /// loans, expected capital and expected loans.
    ///
    /// # Panics
    ///
    /// If the desired employment of a firm is not a finite number that fits
    /// an `i64` (Julia throws an `InexactError` there).
    pub fn set_firms_expectations_and_decisions(&mut self) {
        let term = self.material_price_by_sector();
        let (gamma_e, pi_e) = (self.agg.gamma_e, self.agg.pi_e);
        let (P_bar_HH, P_bar_CF) = (self.agg.P_bar_HH, self.agg.P_bar_CF);
        let p = &self.prop;
        let f = &mut self.firms;
        for i in 0..f.len() {
            let Q_s_i = f.Q_d_i[i] * (1.0 + gamma_e);
            // Cost-push inflation (A.9), with the price of the previous quarter.
            let labour_costs =
                (1.0 + p.tau_SIF) * f.w_bar_i[i] / f.alpha_bar_i[i] * (P_bar_HH / f.P_i[i] - 1.0);
            let material_costs = 1.0 / f.beta_i[i] * (term[f.G_i[i]] / f.P_i[i] - 1.0);
            let capital_costs = f.delta_i[i] / f.kappa_i[i] * (P_bar_CF / f.P_i[i] - 1.0);
            let pi_c_i = labour_costs + material_costs + capital_costs;
            let new_P_i = f.P_i[i] * (1.0 + pi_c_i) * (1.0 + pi_e);

            // Per firm (A.14, A.18, A.22). The Julia code takes this minimum
            // over whole vectors for A.14 and A.22; see README, "Differences
            // from BeforeIT.jl".
            let capacity = min(Q_s_i, f.K_i[i] * f.kappa_i[i]);
            let I_d_i = f.delta_i[i] / f.kappa_i[i] * capacity;
            let DM_d_i = capacity / f.beta_i[i];
            let N_d_i = max(1.0, round(capacity / f.alpha_bar_i[i]));

            let Pi_e_i = f.Pi_i[i] * (1.0 + pi_e) * (1.0 + gamma_e);
            let DD_e_i = Pi_e_i
                - p.theta * f.L_i[i]
                - p.tau_FIRM * max(0.0, Pi_e_i)
                - (p.theta_DIV * (1.0 - p.tau_FIRM)) * max(0.0, Pi_e_i);
            let K_e_i = P_bar_CF * (1.0 + pi_e) * f.K_i[i];
            let L_e_i = (1.0 - p.theta) * f.L_i[i];
            let DL_d_i = max(0.0, -DD_e_i - f.D_i[i]);

            f.Q_s_i[i] = Q_s_i;
            f.I_d_i[i] = I_d_i;
            f.DM_d_i[i] = DM_d_i;
            // Julia converts to `Int` when it stores the value.
            f.N_d_i[i] = round_int(N_d_i).unwrap_or_else(|e| panic!("N_d_i: {e}"));
            f.Pi_e_i[i] = Pi_e_i;
            f.P_i[i] = new_P_i;
            f.DL_d_i[i] = DL_d_i;
            f.K_e_i[i] = K_e_i;
            f.L_e_i[i] = L_e_i;
        }
    }

    /// Work effort, the factor in A.25 and A.26: the output that planned
    /// supply, capital and intermediate inputs allow, relative to what the
    /// employees produce at average productivity, at most 1.5.
    fn effort(&self, i: usize) -> f64 {
        let f = &self.firms;
        min(
            1.5,
            min(
                f.Q_s_i[i],
                min(f.K_i[i] * f.kappa_i[i], f.M_i[i] * f.beta_i[i]),
            ) / (f.N_i[i] as f64 * f.alpha_bar_i[i]),
        )
    }

    /// `firms_wages` (A.26).
    pub fn set_firms_wages(&mut self) {
        for i in 0..self.firms.len() {
            self.firms.w_i[i] = self.firms.w_bar_i[i] * self.effort(i);
        }
    }

    /// `firms_production` (A.13, A.25): Leontief production function.
    pub fn set_firms_production(&mut self) {
        for i in 0..self.firms.len() {
            let alpha_i = self.firms.alpha_bar_i[i] * self.effort(i);
            let f = &self.firms;
            self.firms.Y_i[i] = min(
                f.Q_s_i[i],
                min(
                    f.N_i[i] as f64 * alpha_i,
                    min(f.K_i[i] * f.kappa_i[i], f.M_i[i] * f.beta_i[i]),
                ),
            );
        }
    }

    /// `firms_stocks` (A.3, A.4, A.17, A.21).
    pub fn set_firms_stocks(&mut self) {
        let f = &mut self.firms;
        for i in 0..f.len() {
            f.K_i[i] = f.K_i[i] - f.delta_i[i] / f.kappa_i[i] * f.Y_i[i] + f.I_i[i];
            f.M_i[i] = f.M_i[i] - f.Y_i[i] / f.beta_i[i] + f.DM_i[i];
            f.DS_i[i] = f.Y_i[i] - f.Q_i[i];
            f.S_i[i] += f.DS_i[i];
        }
    }

    /// `firms_profits` (A.30).
    pub fn set_firms_profits(&mut self) {
        let (P_bar_HH, r_bar, r) = (self.agg.P_bar_HH, self.cb.r_bar, self.bank.r);
        let tau_SIF = self.prop.tau_SIF;
        let f = &mut self.firms;
        for i in 0..f.len() {
            let in_sales = f.P_i[i] * f.Q_i[i] + f.P_i[i] * f.DS_i[i];
            let in_deposits = r_bar * pos(f.D_i[i]);
            let out_wages = (1.0 + tau_SIF) * f.w_i[i] * f.N_i[i] as f64 * P_bar_HH;
            let out_expenses = 1.0 / f.beta_i[i] * f.P_bar_i[i] * f.Y_i[i];
            let out_depreciation = f.delta_i[i] / f.kappa_i[i] * f.P_CF_i[i] * f.Y_i[i];
            let out_taxes_prods = f.tau_Y_i[i] * f.P_i[i] * f.Y_i[i];
            let out_taxes_capital = f.tau_K_i[i] * f.P_i[i] * f.Y_i[i];
            let out_loans = r * (f.L_i[i] + pos(-f.D_i[i]));
            f.Pi_i[i] = in_sales + in_deposits
                - out_wages
                - out_expenses
                - out_depreciation
                - out_taxes_prods
                - out_taxes_capital
                - out_loans;
        }
    }

    /// `firms_deposits` (A.33, A.34).
    pub fn set_firms_deposits(&mut self) {
        let (P_bar_HH, r_bar, r) = (self.agg.P_bar_HH, self.cb.r_bar, self.bank.r);
        let p = &self.prop;
        let f = &mut self.firms;
        for i in 0..f.len() {
            let sales = f.P_i[i] * f.Q_i[i];
            let labour_cost = -(1.0 + p.tau_SIF) * f.w_i[i] * f.N_i[i] as f64 * P_bar_HH;
            let material_cost = -f.DM_i[i] * f.P_bar_i[i];
            let taxes_products = -f.tau_Y_i[i] * f.P_i[i] * f.Y_i[i];
            let taxes_production = -f.tau_K_i[i] * f.P_i[i] * f.Y_i[i];
            let corporate_tax = -p.tau_FIRM * pos(f.Pi_i[i]);
            let dividend_payments = -p.theta_DIV * (1.0 - p.tau_FIRM) * pos(f.Pi_i[i]);
            let interest_payments = -r * (f.L_i[i] + pos(-f.D_i[i]));
            let interest_received = r_bar * pos(f.D_i[i]);
            let investment_cost = -f.P_CF_i[i] * f.I_i[i];
            let new_credit = f.DL_i[i];
            let debt_installment = -p.theta * f.L_i[i];
            let DD_i = sales
                + labour_cost
                + material_cost
                + taxes_products
                + taxes_production
                + corporate_tax
                + dividend_payments
                + interest_payments
                + interest_received
                + investment_cost
                + new_credit
                + debt_installment;
            f.D_i[i] += DD_i;
        }
    }

    /// `firms_loans` (A.35).
    pub fn set_firms_loans(&mut self) {
        let theta = self.prop.theta;
        let f = &mut self.firms;
        for i in 0..f.len() {
            f.L_i[i] = (1.0 - theta) * f.L_i[i] + f.DL_i[i];
        }
    }

    /// `firms_equity` (A.36).
    pub fn set_firms_equity(&mut self) {
        let term = self.material_price_by_sector();
        let P_bar_CF = self.agg.P_bar_CF;
        let f = &mut self.firms;
        for i in 0..f.len() {
            f.E_i[i] =
                f.D_i[i] + f.M_i[i] * term[f.G_i[i]] + f.P_i[i] * f.S_i[i] + P_bar_CF * f.K_i[i]
                    - f.L_i[i];
        }
    }

    /// `finance_insolvent_firms!` (A.38 to A.40, and the write-off in A.67):
    /// a firm with negative deposits and negative equity is replaced by a new
    /// one. The bank writes off the firm's loans and overdraft above
    /// `zeta_b · P̄_CF · K_i`, which the new firm keeps as its loans.
    pub fn finance_insolvent_firms(&mut self) {
        let (P_bar_CF, zeta_b) = (self.agg.P_bar_CF, self.prop.zeta_b);
        let f = &mut self.firms;
        for i in 0..f.len() {
            if f.D_i[i] < 0.0 && f.E_i[i] < 0.0 {
                let refinancing_amount = f.L_i[i] - f.D_i[i] - zeta_b * P_bar_CF * f.K_i[i];
                self.bank.E_k -= refinancing_amount;
                f.E_i[i] += refinancing_amount;
                f.L_i[i] = zeta_b * P_bar_CF * f.K_i[i];
                f.D_i[i] = 0.0;
            }
        }
    }
}
