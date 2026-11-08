//! Firms, BeforeIT `src/agent_actions/firms.jl` (appendix A.1).

use crate::agents::Firms;
use crate::julia::{max, min, pos, round, round_int, sum_by};
use crate::random::Randomness;
use crate::{AbstractModel, Extension};

/// Σ_g a_sg[g][s] · P̄_g for each sector s, as in A.9 and A.36: the price of
/// one unit of the sector's bundle of intermediate inputs.
fn material_price_by_sector<X>(m: &AbstractModel<X>) -> Vec<f64> {
    let G = m.prop.G;
    let (a_sg, P_bar_g) = (&m.prop.a_sg, &m.agg.P_bar_g);
    (0..G)
        .map(|s| sum_by(G, |g| a_sg[g][s] * P_bar_g[g]))
        .collect()
}

/// The result of `firms_expectations_and_decisions(model)`, in the order of
/// the Julia tuple.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FirmsDecisions {
    /// Expected sales, the planned supply (A.5).
    pub Q_s_i: Vec<f64>,
    /// Desired investment (A.14).
    pub I_d_i: Vec<f64>,
    /// Desired intermediate inputs (A.18).
    pub DM_d_i: Vec<f64>,
    /// Desired employees (A.22), whole numbers as floats as in Julia.
    pub N_d_i: Vec<f64>,
    /// Expected profits (A.28).
    pub Pi_e_i: Vec<f64>,
    /// Desired new loans (A.29).
    pub DL_d_i: Vec<f64>,
    /// Expected capital (A.63).
    pub K_e_i: Vec<f64>,
    /// Expected loans (A.63).
    pub L_e_i: Vec<f64>,
    /// New price (A.8).
    pub new_P_i: Vec<f64>,
}

/// `cost_push_inflation` (A.9): the change in unit costs of labour,
/// intermediate inputs and capital at last quarter's prices.
pub fn cost_push_inflation<X: Extension>(m: &AbstractModel<X>, firms: &Firms) -> Vec<f64> {
    let term = material_price_by_sector(m);
    let (P_bar_HH, P_bar_CF) = (m.agg.P_bar_HH, m.agg.P_bar_CF);
    let tau_SIF = m.prop.tau_SIF;
    let f = firms;
    (0..f.len())
        .map(|i| {
            let labour_costs =
                (1.0 + tau_SIF) * f.w_bar_i[i] / f.alpha_bar_i[i] * (P_bar_HH / f.P_i[i] - 1.0);
            let material_costs = 1.0 / f.beta_i[i] * (term[f.G_i[i]] / f.P_i[i] - 1.0);
            let capital_costs = f.delta_i[i] / f.kappa_i[i] * (P_bar_CF / f.P_i[i] - 1.0);
            labour_costs + material_costs + capital_costs
        })
        .collect()
}

/// `desired_capital_material_employment` (A.14, A.18, A.22): desired
/// investment, intermediate inputs and employees for the planned supply
/// `Q_s_i`, at most at full capacity.
///
/// The minimum of planned supply and capacity is taken firm by firm. The
/// Julia code takes it over the whole vectors for A.14 and A.22; see
/// README, "Differences from BeforeIT.jl".
pub fn desired_capital_material_employment(
    firms: &Firms,
    Q_s_i: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let f = firms;
    let capacity: Vec<f64> = (0..f.len())
        .map(|i| min(Q_s_i[i], f.K_i[i] * f.kappa_i[i]))
        .collect();
    let I_d_i = (0..f.len())
        .map(|i| f.delta_i[i] / f.kappa_i[i] * capacity[i])
        .collect();
    let DM_d_i = (0..f.len()).map(|i| capacity[i] / f.beta_i[i]).collect();
    let N_d_i = (0..f.len())
        .map(|i| max(1.0, round(capacity[i] / f.alpha_bar_i[i])))
        .collect();
    (I_d_i, DM_d_i, N_d_i)
}

/// `expected_deposits_capital_loans` (A.29, A.63): expected change in
/// deposits, expected capital at expected prices and expected loans.
pub fn expected_deposits_capital_loans<X: Extension>(
    m: &AbstractModel<X>,
    firms: &Firms,
    Pi_e_i: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let p = &m.prop;
    let (P_bar_CF, pi_e) = (m.agg.P_bar_CF, m.agg.pi_e);
    let f = firms;
    let DD_e_i = (0..f.len())
        .map(|i| {
            Pi_e_i[i]
                - p.theta * f.L_i[i]
                - p.tau_FIRM * max(0.0, Pi_e_i[i])
                - (p.theta_DIV * (1.0 - p.tau_FIRM)) * max(0.0, Pi_e_i[i])
        })
        .collect();
    let K_e_i = f.K_i.iter().map(|k| P_bar_CF * (1.0 + pi_e) * k).collect();
    let L_e_i = f.L_i.iter().map(|l| (1.0 - p.theta) * l).collect();
    (DD_e_i, K_e_i, L_e_i)
}

/// `firms_expectations_and_decisions` (A.5, A.8, A.9, A.14, A.18, A.22, A.27
/// to A.29, A.63): planned supply, price, desired investment, intermediate
/// inputs and employees, expected profits, desired new loans, expected
/// capital and expected loans. The standard rule draws no random numbers.
pub fn firms_expectations_and_decisions<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    _rng: &mut R,
) -> FirmsDecisions {
    let m = &*m;
    let f = &m.firms;
    let (gamma_e, pi_e) = (m.agg.gamma_e, m.agg.pi_e);
    let Q_s_i: Vec<f64> = f.Q_d_i.iter().map(|q| q * (1.0 + gamma_e)).collect();
    let pi_c_i = X::cost_push_inflation(m, f);
    let new_P_i = (0..f.len())
        .map(|i| f.P_i[i] * (1.0 + pi_c_i[i]) * (1.0 + pi_e))
        .collect();
    let (I_d_i, DM_d_i, N_d_i) = X::desired_capital_material_employment(f, &Q_s_i);
    let Pi_e_i: Vec<f64> = f
        .Pi_i
        .iter()
        .map(|pi| pi * (1.0 + pi_e) * (1.0 + gamma_e))
        .collect();
    let (DD_e_i, K_e_i, L_e_i) = X::expected_deposits_capital_loans(m, f, &Pi_e_i);
    let DL_d_i = (0..f.len())
        .map(|i| max(0.0, -DD_e_i[i] - f.D_i[i]))
        .collect();
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

/// Work effort, the factor in A.25 and A.26: the output that planned supply,
/// capital and intermediate inputs allow, relative to what the employees
/// produce at average productivity, at most 1.5.
fn effort(f: &Firms, i: usize) -> f64 {
    min(
        1.5,
        min(
            f.Q_s_i[i],
            min(f.K_i[i] * f.kappa_i[i], f.M_i[i] * f.beta_i[i]),
        ) / (f.N_i[i] as f64 * f.alpha_bar_i[i]),
    )
}

/// `firms_wages` (A.26).
pub fn firms_wages<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let f = &m.firms;
    (0..f.len()).map(|i| f.w_bar_i[i] * effort(f, i)).collect()
}

/// `firms_production` (A.13, A.25): Leontief production function with the
/// labour productivity of the quarter.
pub fn firms_production<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let f = &m.firms;
    let alpha_i: Vec<f64> = (0..f.len())
        .map(|i| f.alpha_bar_i[i] * effort(f, i))
        .collect();
    X::leontief_production(
        &f.Q_s_i, &f.N_i, &alpha_i, &f.K_i, &f.kappa_i, &f.M_i, &f.beta_i,
    )
}

/// `leontief_production` (A.13).
pub fn leontief_production(
    Q_s_i: &[f64],
    N_i: &[i64],
    alpha_i: &[f64],
    K_i: &[f64],
    kappa_i: &[f64],
    M_i: &[f64],
    beta_i: &[f64],
) -> Vec<f64> {
    (0..Q_s_i.len())
        .map(|i| {
            min(
                Q_s_i[i],
                min(
                    N_i[i] as f64 * alpha_i[i],
                    min(K_i[i] * kappa_i[i], M_i[i] * beta_i[i]),
                ),
            )
        })
        .collect()
}

/// `firms_stocks` (A.3, A.4, A.17, A.21): capital, intermediate inputs,
/// change in inventories and inventories.
pub fn firms_stocks<X: Extension>(
    m: &mut AbstractModel<X>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let f = &m.firms;
    let I = f.len();
    let K_i = (0..I)
        .map(|i| f.K_i[i] - f.delta_i[i] / f.kappa_i[i] * f.Y_i[i] + f.I_i[i])
        .collect();
    let M_i = (0..I)
        .map(|i| f.M_i[i] - f.Y_i[i] / f.beta_i[i] + f.DM_i[i])
        .collect();
    let DS_i: Vec<f64> = (0..I).map(|i| f.Y_i[i] - f.Q_i[i]).collect();
    let S_i = (0..I).map(|i| f.S_i[i] + DS_i[i]).collect();
    (K_i, M_i, DS_i, S_i)
}

/// The lending rate of each firm's bank, `[model.banks.r[firms.B_i[i]] for
/// i in eachfirm(model)]`.
fn firms_rate<X>(m: &AbstractModel<X>) -> Vec<f64> {
    (0..m.prop.I).map(|i| m.banks.r[m.firms.B_i[i]]).collect()
}

/// `firms_profits` (A.30).
pub fn firms_profits<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let (P_bar_HH, r_bar, r) = (m.agg.P_bar_HH, m.cb.r_bar, firms_rate(m));
    let tau_SIF = m.prop.tau_SIF;
    let f = &m.firms;
    (0..f.len())
        .map(|i| {
            let in_sales = f.P_i[i] * f.Q_i[i] + f.P_i[i] * f.DS_i[i];
            let in_deposits = r_bar * pos(f.D_i[i]);
            let out_wages = (1.0 + tau_SIF) * f.w_i[i] * f.N_i[i] as f64 * P_bar_HH;
            let out_expenses = 1.0 / f.beta_i[i] * f.P_bar_i[i] * f.Y_i[i];
            let out_depreciation = f.delta_i[i] / f.kappa_i[i] * f.P_CF_i[i] * f.Y_i[i];
            let out_taxes_prods = f.tau_Y_i[i] * f.P_i[i] * f.Y_i[i];
            let out_taxes_capital = f.tau_K_i[i] * f.P_i[i] * f.Y_i[i];
            let out_loans = r[i] * (f.L_i[i] + pos(-f.D_i[i]));
            in_sales + in_deposits
                - out_wages
                - out_expenses
                - out_depreciation
                - out_taxes_prods
                - out_taxes_capital
                - out_loans
        })
        .collect()
}

/// `firms_deposits` (A.33, A.34).
pub fn firms_deposits<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let (P_bar_HH, r_bar, r) = (m.agg.P_bar_HH, m.cb.r_bar, firms_rate(m));
    let p = &m.prop;
    let f = &m.firms;
    (0..f.len())
        .map(|i| {
            let sales = f.P_i[i] * f.Q_i[i];
            let labour_cost = -(1.0 + p.tau_SIF) * f.w_i[i] * f.N_i[i] as f64 * P_bar_HH;
            let material_cost = -f.DM_i[i] * f.P_bar_i[i];
            let taxes_products = -f.tau_Y_i[i] * f.P_i[i] * f.Y_i[i];
            let taxes_production = -f.tau_K_i[i] * f.P_i[i] * f.Y_i[i];
            let corporate_tax = -p.tau_FIRM * pos(f.Pi_i[i]);
            let dividend_payments = -p.theta_DIV * (1.0 - p.tau_FIRM) * pos(f.Pi_i[i]);
            let interest_payments = -r[i] * (f.L_i[i] + pos(-f.D_i[i]));
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
            f.D_i[i] + DD_i
        })
        .collect()
}

/// `firms_loans` (A.35).
pub fn firms_loans<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let theta = m.prop.theta;
    let f = &m.firms;
    (0..f.len())
        .map(|i| (1.0 - theta) * f.L_i[i] + f.DL_i[i])
        .collect()
}

/// `firms_equity` (A.36).
pub fn firms_equity<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let term = material_price_by_sector(m);
    let P_bar_CF = m.agg.P_bar_CF;
    let f = &m.firms;
    (0..f.len())
        .map(|i| {
            f.D_i[i] + f.M_i[i] * term[f.G_i[i]] + f.P_i[i] * f.S_i[i] + P_bar_CF * f.K_i[i]
                - f.L_i[i]
        })
        .collect()
}

impl<X: Extension> AbstractModel<X> {
    /// `firms_expectations_and_decisions(model)`; see
    /// [`crate::base::firms_expectations_and_decisions`].
    pub fn firms_expectations_and_decisions<R: Randomness + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> FirmsDecisions {
        X::firms_expectations_and_decisions(self, rng)
    }

    /// `set_firms_expectations_and_decisions!`: planned supply, desired
    /// investment, intermediate inputs and employees, expected profits,
    /// price, desired new loans, expected capital and expected loans.
    ///
    /// # Panics
    ///
    /// If the desired employment of a firm is not a finite number that fits
    /// an `i64` (Julia throws an `InexactError` there).
    pub fn set_firms_expectations_and_decisions<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let d = self.firms_expectations_and_decisions(rng);
        let f = &mut self.firms;
        f.Q_s_i.copy_from_slice(&d.Q_s_i);
        f.I_d_i.copy_from_slice(&d.I_d_i);
        f.DM_d_i.copy_from_slice(&d.DM_d_i);
        for (n, &x) in f.N_d_i.iter_mut().zip(&d.N_d_i) {
            // Julia converts to `Int` when it stores the value.
            *n = round_int(x).unwrap_or_else(|e| panic!("N_d_i: {e}"));
        }
        f.Pi_e_i.copy_from_slice(&d.Pi_e_i);
        f.P_i.copy_from_slice(&d.new_P_i);
        f.DL_d_i.copy_from_slice(&d.DL_d_i);
        f.K_e_i.copy_from_slice(&d.K_e_i);
        f.L_e_i.copy_from_slice(&d.L_e_i);
    }

    /// `firms_wages(model)`; see [`crate::base::firms_wages`].
    pub fn firms_wages(&mut self) -> Vec<f64> {
        X::firms_wages(self)
    }

    /// `set_firms_wages!`: `firms.w_i`.
    pub fn set_firms_wages(&mut self) {
        let w_i = self.firms_wages();
        self.firms.w_i.copy_from_slice(&w_i);
    }

    /// `firms_production(model)`; see [`crate::base::firms_production`].
    pub fn firms_production(&mut self) -> Vec<f64> {
        X::firms_production(self)
    }

    /// `set_firms_production!`: `firms.Y_i`.
    pub fn set_firms_production(&mut self) {
        let Y_i = self.firms_production();
        self.firms.Y_i.copy_from_slice(&Y_i);
    }

    /// `firms_stocks(model)`; see [`crate::base::firms_stocks`].
    pub fn firms_stocks(&mut self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        X::firms_stocks(self)
    }

    /// `set_firms_stocks!`: `firms.K_i`, `M_i`, `DS_i` and `S_i`.
    pub fn set_firms_stocks(&mut self) {
        let (K_i, M_i, DS_i, S_i) = self.firms_stocks();
        let f = &mut self.firms;
        f.K_i.copy_from_slice(&K_i);
        f.M_i.copy_from_slice(&M_i);
        f.DS_i.copy_from_slice(&DS_i);
        f.S_i.copy_from_slice(&S_i);
    }

    /// `firms_profits(model)`; see [`crate::base::firms_profits`].
    pub fn firms_profits(&mut self) -> Vec<f64> {
        X::firms_profits(self)
    }

    /// `set_firms_profits!`: `firms.Pi_i`.
    pub fn set_firms_profits(&mut self) {
        let Pi_i = self.firms_profits();
        self.firms.Pi_i.copy_from_slice(&Pi_i);
    }

    /// `firms_deposits(model)`; see [`crate::base::firms_deposits`].
    pub fn firms_deposits(&mut self) -> Vec<f64> {
        X::firms_deposits(self)
    }

    /// `set_firms_deposits!`: `firms.D_i`.
    pub fn set_firms_deposits(&mut self) {
        let D_i = self.firms_deposits();
        self.firms.D_i.copy_from_slice(&D_i);
    }

    /// `firms_loans(model)`; see [`crate::base::firms_loans`].
    pub fn firms_loans(&mut self) -> Vec<f64> {
        X::firms_loans(self)
    }

    /// `set_firms_loans!`: `firms.L_i`.
    pub fn set_firms_loans(&mut self) {
        let L_i = self.firms_loans();
        self.firms.L_i.copy_from_slice(&L_i);
    }

    /// `firms_equity(model)`; see [`crate::base::firms_equity`].
    pub fn firms_equity(&mut self) -> Vec<f64> {
        X::firms_equity(self)
    }

    /// `set_firms_equity!`: `firms.E_i`.
    pub fn set_firms_equity(&mut self) {
        let E_i = self.firms_equity();
        self.firms.E_i.copy_from_slice(&E_i);
    }
}
