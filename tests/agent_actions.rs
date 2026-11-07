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
/// the reserves at the central bank, minus interest on deposits.
#[test]
fn bank_profits() {
    let mut m = model();
    m.firms.L_i = vec![1.0, -1.0, 0.0];
    m.firms.D_i = vec![1.0, -1.0, 0.0];
    m.firms.D_h = vec![1.0, -1.0, 0.0];
    m.w_act.D_h.clear();
    m.w_inact.D_h.clear();
    m.bank.D_h = 0.0;
    m.bank.D_k = 4.0;
    m.cb.r_bar = 0.1;
    m.bank.r = 0.05;
    m.set_banks_profits();
    check("Pi_k", m.bank.Pi_k, 0.3, 0.0, 1e-10);
}

/// `test bank_expected_profits`.
#[test]
fn bank_expected_profits() {
    let mut m = model();
    m.bank.Pi_k = 1.0;
    m.agg.pi_e = 0.1;
    m.agg.gamma_e = 0.2;
    m.set_banks_expected_profits();
    check("Pi_e_k", m.bank.Pi_e_k, 1.32, 0.0, 1e-10);
}

/// `test bank_deposits`: the reserves are what remains of the bank's
/// balance sheet.
#[test]
fn bank_deposits() {
    let mut m = model();
    m.w_act.D_h = vec![1.0, 2.0, 3.0];
    m.w_inact.D_h = vec![1.0, 2.0, 3.0];
    m.firms.D_h = vec![1.0, 2.0, 3.0];
    m.firms.D_i = vec![1.0, 2.0, 3.0];
    m.firms.L_i = vec![6.0, 0.0, 0.0];
    m.bank.D_h = 6.0;
    m.bank.E_k = 6.0;
    m.set_banks_deposits();
    check("D_k", m.bank.D_k, 30.0, 0.0, 1e-10);
}

/// `test taylor_rule`: the policy rate does not fall below 0.
#[test]
fn taylor_rule() {
    let mut m = model();
    let cb = &mut m.cb;
    (
        cb.rho,
        cb.r_bar,
        cb.r_star,
        cb.pi_star,
        cb.xi_pi,
        cb.xi_gamma,
    ) = (1.0, -0.1, 0.1, 0.1, 0.5, 0.5);
    check("r_bar", cb.taylor_rule(0.1, 0.1), 0.0, 0.0, 1e-10);
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
