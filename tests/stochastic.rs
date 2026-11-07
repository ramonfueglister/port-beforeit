//! Runs with randomness: the goods market against the mean and standard
//! deviation of the authors' original implementation (BeforeIT
//! `test/markets/search_and_matching.jl`); the same seed gives the same run;
//! a saved state continues exactly like the original.

#![allow(non_snake_case)]

mod common;

use beforeit::{Model, Randomness, Workers, Xoshiro};
use common::*;

/// One quarter up to and including the goods market, as in the authors'
/// test.
fn until_goods_market(m: &mut Model, rng: &mut dyn Randomness) {
    m.finance_insolvent_firms();
    m.set_growth_inflation_expectations(rng);
    m.set_epsilon(rng);
    m.set_growth_inflation_EA(rng);
    m.set_central_bank_rate();
    m.set_banks_rate();
    m.set_firms_expectations_and_decisions(rng);
    m.search_and_matching_credit(rng);
    m.search_and_matching_labour(rng);
    m.set_firms_wages();
    m.set_firms_production();
    m.update_workers_wages();
    m.set_gov_social_benefits();
    m.set_banks_expected_profits();
    m.set_households_budget_act();
    m.set_households_budget_inact();
    m.set_households_budget_firms();
    m.set_households_budget_banks();
    m.set_gov_expenditure(rng);
    m.set_rotw_import_export(rng);
    m.search_and_matching(rng);
}

/// Mean and standard deviation of ten quantities after the goods market in
/// the original implementation, from the authors' test.
const EXPECTED: [(&str, f64, f64); 10] = [
    ("I_h", 0.32975, 0.0025351),
    ("C_h", 3.973, 0.029366),
    ("I_i", 20.5075, 0.12763),
    ("DM_i", 109.3163, 0.68033),
    ("P_bar_i", 1.0031, 0.0044726),
    ("P_CF_i", 1.0031, 0.0044726),
    ("C_j", 14752.2413, 126.7441),
    ("C_l", 34188.1258, 666.275),
    ("Q_d_i", 216.2474, 1.2275),
    ("Q_d_m", 535.7522, 9.6082),
];

fn market_values(seed: u64) -> [f64; 10] {
    let (p, ic) = austria();
    let rng = &mut Xoshiro::seed_from_u64(seed);
    let mut m = Model::new(&p, &ic, rng).unwrap();
    until_goods_market(&mut m, rng);
    // All households in the order of the authors' test: bank owners,
    // workers, inactive persons, firm owners.
    let all = |field: fn(&Workers) -> &Vec<f64>, firms: &[f64], banks: &[f64]| {
        let mut v = banks.to_vec();
        v.extend(field(&m.w_act));
        v.extend(field(&m.w_inact));
        v.extend(firms);
        v
    };
    let f = &m.firms;
    [
        mean(&all(|w| &w.I_h, &f.I_h, &m.banks.I_h)),
        mean(&all(|w| &w.C_h, &f.C_h, &m.banks.C_h)),
        mean(&f.I_i),
        mean(&f.DM_i),
        mean(&f.P_bar_i),
        mean(&f.P_CF_i),
        m.gov.C_j,
        m.rotw.C_l,
        mean(&f.Q_d_i),
        mean(&m.rotw.Q_d_m),
    ]
}

/// As in the authors' test: one run, each quantity within three standard
/// deviations of the original's mean.
#[test]
fn goods_market_one_run_like_the_original() {
    for ((name, expected, sigma), x) in EXPECTED.iter().zip(market_values(1)) {
        check(name, x, *expected, 0.0, 3.0 * sigma);
    }
}

/// Stricter: 50 runs. Their mean lies within three standard errors of the
/// original's mean, and their standard deviation is close to the
/// original's.
#[test]
fn goods_market_mean_and_spread_like_the_original() {
    let n = 50;
    let runs: Vec<[f64; 10]> = (1..=n).map(market_values).collect();
    for (k, (name, expected, sigma)) in EXPECTED.iter().enumerate() {
        let z: Vec<f64> = runs.iter().map(|r| (r[k] - expected) / sigma).collect();
        let mean_z = mean(&z);
        let sd_z = (z.iter().map(|x| (x - mean_z).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        assert!(
            mean_z.abs() <= 3.0 / (n as f64).sqrt(),
            "{name}: mean {mean_z:+.2} standard deviations from the original's"
        );
        assert!(
            (0.6..=1.4).contains(&sd_z),
            "{name}: standard deviation {sd_z:.2} times the original's"
        );
    }
}

fn run(seed: u64, T: usize) -> Model {
    let (p, ic) = austria();
    let rng = &mut Xoshiro::seed_from_u64(seed);
    let mut m = Model::new(&p, &ic, rng).unwrap();
    m.run(T, rng);
    m
}

#[test]
fn same_seed_same_run() {
    assert_eq!(run(7, 3), run(7, 3));
    assert_ne!(run(7, 3).data.nominal_gdp, run(8, 3).data.nominal_gdp);
}

/// Model and generator state saved after two quarters continue exactly like
/// the original run.
#[test]
fn saved_state_continues_identically() {
    let (p, ic) = austria();
    let mut rng = Xoshiro::seed_from_u64(11);
    let mut m = Model::new(&p, &ic, &mut rng).unwrap();
    m.run(2, &mut rng);
    let mut saved = m.clone();
    let mut saved_rng = Xoshiro::from_state(rng.state()).unwrap();
    m.run(2, &mut rng);
    saved.run(2, &mut saved_rng);
    assert_eq!(m, saved);
}
