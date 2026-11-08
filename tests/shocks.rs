//! Shocks, as in BeforeIT `test/shocks/shocks.jl`, and the other shocks of
//! BeforeIT `src/shocks/shocks.jl`.

#![allow(non_snake_case)]

mod common;

use beforeit::{
    ConsumptionShock, Deterministic, InterestRateShock, Model, NoShock, ProductivityShock, Shock,
    Xoshiro,
};
use common::*;

fn model() -> Model {
    let (p, ic) = austria();
    Model::new(&p, &ic, &mut Deterministic).unwrap()
}

/// `test no shock`.
#[test]
fn no_shock_changes_nothing() {
    let mut m = model();
    let before = m.clone();
    NoShock.apply(&mut m);
    assert_eq!(m, before);
}

/// `test time shock`.
#[test]
fn interest_rate_shock_until_final_time() {
    let mut m = model();
    m.cb.r_bar = 0.01;
    m.agg.t = 1;
    let shock = InterestRateShock {
        rate: 0.02,
        final_time: 1,
    };
    shock.apply(&mut m);
    assert_eq!(m.cb.r_bar, 0.02);
    m.cb.r_bar = 0.01;
    m.agg.t = 2;
    shock.apply(&mut m);
    assert_eq!(m.cb.r_bar, 0.01);
}

/// The productivity shock multiplies again each time it is applied.
#[test]
fn productivity_shock_multiplies_each_time() {
    let mut m = model();
    let alpha = m.firms.alpha_bar_i.clone();
    let shock = ProductivityShock {
        productivity_multiplier: 1.02,
    };
    shock.apply(&mut m);
    shock.apply(&mut m);
    for (new, old) in m.firms.alpha_bar_i.iter().zip(&alpha) {
        assert_eq!(*new, old * 1.02 * 1.02);
    }
}

/// The consumption shock raises `psi` at time 1 and lowers it again at
/// `final_time`.
#[test]
fn consumption_shock_from_time_1_to_final_time() {
    let mut m = model();
    let psi = m.prop.psi;
    let shock = ConsumptionShock {
        consumption_multiplier: 1.02,
        final_time: 4,
    };
    let mut values = Vec::new();
    for t in 1..=5 {
        m.agg.t = t;
        shock.apply(&mut m);
        values.push(m.prop.psi);
    }
    let high = psi * 1.02;
    assert_eq!(values, [high, high, high, high / 1.02, high / 1.02]);
}

/// `run_with` applies the shock in every quarter right after the central
/// bank has set the rate; a closure is a shock as well.
#[test]
fn shocks_in_runs() {
    let (p, ic) = austria();
    let run = |shock: &dyn Shock<beforeit::Standard>| {
        let rng = &mut Xoshiro::seed_from_u64(3);
        let mut m = Model::new(&p, &ic, rng).unwrap();
        m.run_with(3, rng, false, shock);
        m
    };
    let rate = InterestRateShock {
        rate: 0.01,
        final_time: 2,
    };
    let shocked = run(&rate);
    let closure = |m: &mut Model| {
        if m.agg.t <= 2 {
            m.cb.r_bar = 0.01;
        }
    };
    assert_eq!(shocked, run(&closure));
    // Data are collected after the shock: the policy rate of quarters 1
    // and 2 is the shock's, that of quarter 3 the Taylor rule's.
    assert_eq!(shocked.data.euribor[1..3], [0.01, 0.01]);
    assert_ne!(shocked.data.euribor[3], 0.01);
    assert_ne!(shocked, run(&NoShock));

    // `steps` runs quarters without collecting data.
    let rng = &mut Xoshiro::seed_from_u64(3);
    let mut m = Model::new(&p, &ic, rng).unwrap();
    m.steps(3, rng, false, &rate);
    assert_eq!(m.cb, shocked.cb);
    assert_eq!(m.data.euribor.len(), 1);
}
