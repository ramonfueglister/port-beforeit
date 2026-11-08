//! Single rules with small numbers, as in BeforeIT `test/agent_actions`.

#![allow(non_snake_case)]

mod common;

use beforeit::{Deterministic, Model};
use common::*;

fn model() -> Model {
    let (p, ic) = austria();
    Model::new(&p, &ic, &mut Deterministic).unwrap()
}

/// `test bank_profits`: interest on loans and overdrafts plus interest on
/// the reserves at the central bank, minus interest on deposits. As in the
/// Julia test, the model is cut down to three firms of the first bank.
#[test]
fn bank_profits() {
    let mut m = model();
    m.firms.L_i = vec![1.0, -1.0, 0.0];
    m.firms.D_i = vec![1.0, -1.0, 0.0];
    m.firms.D_h = vec![1.0, -1.0, 0.0];
    m.firms.B_i = vec![0, 0, 0];
    m.prop.I = 3;
    m.w_act.D_h.clear();
    m.w_inact.D_h.clear();
    m.w_act.B_h.clear();
    m.w_inact.B_h.clear();
    m.banks.firms[0].truncate(3);
    m.banks.w_act[0].clear();
    m.banks.w_inact[0].clear();
    m.prop.H = 0;
    m.banks.D_h[0] = 0.0;
    m.banks.D_k[0] = 4.0;
    m.cb.r_bar = 0.1;
    m.banks.r[0] = 0.05;
    let Pi_k = m.banks_profits();
    check("Pi_k", mean(&Pi_k), 0.3, 0.0, 1e-10);
}

/// `test bank_expected_profits`.
#[test]
fn bank_expected_profits() {
    let mut m = model();
    m.banks.Pi_k.fill(1.0);
    m.agg.pi_e = 0.1;
    m.agg.gamma_e = 0.2;
    let Pi_e_k = m.banks_expected_profits();
    check("Pi_e_k", mean(&Pi_e_k), 1.32, 0.0, 1e-10);
}

/// `test bank_deposits`: the reserves are what remains of the bank's
/// balance sheet.
#[test]
fn bank_deposits() {
    let mut m = model();
    m.w_act.D_h = vec![1.0, 2.0, 3.0];
    m.w_inact.D_h = vec![1.0, 2.0, 3.0];
    m.w_act.B_h = vec![0, 0, 0];
    m.w_inact.B_h = vec![0, 0, 0];
    m.firms.D_h = vec![1.0, 2.0, 3.0];
    m.firms.D_i = vec![1.0, 2.0, 3.0];
    m.firms.L_i = vec![6.0, 0.0, 0.0];
    m.firms.B_i = vec![0, 0, 0];
    m.banks.firms[0].truncate(3);
    m.banks.w_act[0].truncate(3);
    m.banks.w_inact[0].truncate(3);
    m.prop.I = 3;
    m.prop.H = 3;
    m.banks.D_h[0] = 6.0;
    m.banks.E_k[0] = 6.0;
    let D_k = m.banks_deposits();
    check("D_k", mean(&D_k), 30.0, 0.0, 1e-10);
}

/// `bank_exposure` (no test in BeforeIT.jl): the loans of the bank's firms,
/// expected loans if asked, plus the overdrafts of all its customers and its
/// owner if asked.
#[test]
fn bank_exposure() {
    let mut m = model();
    m.firms.L_i = vec![1.0, 2.0, 3.0];
    m.firms.L_e_i = vec![0.5, 1.0, 1.5];
    m.firms.D_i = vec![-1.0, 2.0, 0.0];
    m.firms.D_h = vec![0.0, -2.0, 1.0];
    m.firms.B_i = vec![0, 0, 0];
    m.prop.I = 3;
    m.banks.firms[0].truncate(3);
    m.w_act.D_h = vec![-0.5, 1.0];
    m.w_act.B_h = vec![0, 0];
    m.banks.w_act[0].truncate(2);
    m.w_inact.D_h = vec![-0.25];
    m.w_inact.B_h = vec![0];
    m.banks.w_inact[0].truncate(1);
    m.banks.D_h[0] = -0.125;
    assert_eq!(m.bank_exposure(0, false, false), 6.0);
    assert_eq!(m.bank_exposure(0, true, false), 3.0);
    // Loans 6, overdrafts of a firm 1, of a firm owner 2, of a worker 0.5,
    // of an inactive person 0.25 and of the bank owner 0.125.
    assert_eq!(m.bank_exposure(0, false, true), 9.875);
}

/// `omega_pi` (no test in BeforeIT.jl): with weight 1, expected inflation
/// is last quarter's inflation.
#[test]
fn omega_pi() {
    let (mut p, ic) = austria();
    p.omega_pi = 1.0;
    let mut m = Model::new(&p, &ic, &mut Deterministic).unwrap();
    let (_, _, pi_e) = m.growth_inflation_expectations(&mut Deterministic);
    let n = m.prop.T_prime + m.agg.t - 1;
    assert_eq!(pi_e, m.agg.pi_[n - 1]);
}

/// `test taylor_rule`: the policy rate does not fall below 0.
#[test]
fn taylor_rule() {
    let (rho, r_bar, r_star, pi_star, xi_pi, xi_gamma) = (1.0, -0.1, 0.1, 0.1, 0.5, 0.5);
    let (gamma_EA, pi_EA) = (0.1, 0.1);
    let rate = beforeit::taylor_rule(
        rho, r_bar, r_star, pi_star, xi_pi, xi_gamma, gamma_EA, pi_EA,
    );
    check("r_bar", rate, 0.0, 0.0, 1e-10);
}

/// `test inflation_priceindex`: prices weighted by production. The price
/// index itself is checked here as well.
#[test]
fn inflation_priceindex() {
    let mut m = model();
    m.firms.P_i = vec![1.0, 2.0, 3.0];
    m.firms.Y_i = vec![1.0, 2.0, 3.0];
    m.agg.P_bar = 2.0;
    m.set_inflation_priceindex();
    check(
        "pi",
        *m.agg.pi_.last().unwrap(),
        (14.0f64 / 12.0).ln(),
        0.0,
        1e-10,
    );
    check("P_bar", m.agg.P_bar, 14.0 / 6.0, 0.0, 1e-10);
}

/// `test sector_specific_priceindex`: domestic sales and imports.
#[test]
fn sector_specific_priceindex() {
    let mut m = model();
    m.firms.P_i = vec![1.0, 2.0, 3.0];
    m.firms.Q_i = vec![1.0, 2.0, 3.0];
    m.firms.G_i = vec![0, 0, 0];
    m.rotw.P_m[0] = 2.0;
    m.rotw.Q_m[0] = 1.0;
    m.set_sector_specific_priceindex();
    check("P_bar_g", m.agg.P_bar_g[0], 16.0 / 7.0, 0.0, 1e-10);
}

/// `test get_leontief_production`. In the Julia test the productivity of
/// labour is given; here the rule computes it from the work effort, which
/// gives the same values for these numbers.
#[test]
fn leontief_production() {
    let mut m = model();
    let f = &mut m.firms;
    f.G_i = vec![0, 0, 0];
    f.Q_s_i = vec![1.0, 2.0, 0.0];
    f.N_i = vec![1, 2, 3];
    f.alpha_bar_i = vec![1.0, 2.0, 3.0];
    f.K_i = vec![1.0, 2.0, 3.0];
    f.kappa_i = vec![1.0, 0.0, 3.0];
    f.M_i = vec![1.0, 2.0, 3.0];
    f.beta_i = vec![1.0, 2.0, 3.0];
    f.Y_i = vec![0.0; 3];
    m.set_firms_production();
    assert_eq!(m.firms.Y_i, [1.0, 0.0, 0.0]);
}
