//! The model extensions CANVAS and growth-rate model, as in BeforeIT
//! `test/model_extensions.jl`, and checks of their rules.

#![allow(non_snake_case)]

use beforeit::estimate::{estimate_for_calibration_script, estimate_taylor_rule};
use beforeit::{
    AUSTRIA2010Q1, Deterministic, ITALY2010Q1, ModelCANVAS, ModelGR, Xoshiro,
    get_accounting_identities, get_accounting_identity_banks,
};

/// `model extensions run`: each model type can be built and stepped; the
/// CANVAS model needs the euro area series of the Italian data set.
#[test]
fn model_extensions_run() {
    let rng = &mut Xoshiro::seed_from_u64(1);
    let s = &*AUSTRIA2010Q1;
    let mut gr = ModelGR::new(&s.parameters, &s.initial_conditions, rng).unwrap();
    gr.run(1, rng);
    assert_eq!(gr.agg.t, 2);
    let s = &*ITALY2010Q1;
    let mut canvas = ModelCANVAS::new(&s.parameters, &s.initial_conditions, rng).unwrap();
    canvas.run(1, rng);
    assert_eq!(canvas.agg.t, 2);
}

/// The accounting identities and balance sheets of the authors' test hold
/// in both model types: for the growth-rate model on the Austrian data
/// within the authors' 1e-8; for the CANVAS model on the Italian data,
/// whose nominal GDP is about 5.5 times larger, within 1e-13 of the summed
/// nominal GDP (the model of the paper on the Italian data reaches 1.7e-8).
#[test]
fn identities_hold_in_both_model_types() {
    let check = |name: &str, atol: f64, residuals: (f64, f64, f64), banks: (f64, f64)| {
        let all = [residuals.0, residuals.1, residuals.2, banks.0, banks.1];
        for (k, r) in all.iter().enumerate() {
            assert!(r.abs() <= atol, "{name}, identity {k}: {r:e}");
        }
    };
    for seed in [1, 2] {
        let rng = &mut Xoshiro::seed_from_u64(seed);
        let s = &*AUSTRIA2010Q1;
        let mut gr = ModelGR::new(&s.parameters, &s.initial_conditions, rng).unwrap();
        gr.run(3, rng);
        let (data, banks) = (
            get_accounting_identities(&gr.data),
            get_accounting_identity_banks(&gr),
        );
        check("GR", 1e-8, data, banks);
        let s = &*ITALY2010Q1;
        let mut canvas = ModelCANVAS::new(&s.parameters, &s.initial_conditions, rng).unwrap();
        canvas.run(3, rng);
        let atol = 1e-13 * canvas.data.nominal_gdp.iter().sum::<f64>();
        let data = get_accounting_identities(&canvas.data);
        check("CANVAS", atol, data, get_accounting_identity_banks(&canvas));
    }
}

/// The central bank of the CANVAS model re-estimates the Taylor rule from
/// the histories each quarter, which grow by one entry per quarter.
#[test]
fn canvas_learns_the_taylor_rule() {
    let s = &*ITALY2010Q1;
    let rng = &mut Xoshiro::seed_from_u64(3);
    let mut m = ModelCANVAS::new(&s.parameters, &s.initial_conditions, rng).unwrap();
    let T_prime = m.prop.T_prime;
    assert_eq!(m.ext.r_bar_series.len(), T_prime);
    assert_eq!(m.firms.Q_s_i, m.firms.Q_d_i);
    for t in 1..=3 {
        let e = m.ext.clone();
        m.step(rng);
        m.collect_data();
        let n = T_prime + t - 1;
        for series in [&m.ext.r_bar_series, &m.ext.Y_EA_series, &m.ext.pi_EA_series] {
            assert_eq!(series.len(), n + 1);
        }
        assert_eq!(m.ext.r_bar_series[n], m.cb.r_bar);
        assert_eq!(m.ext.Y_EA_series[n], m.rotw.Y_EA);
        assert_eq!(m.ext.pi_EA_series[n], m.rotw.pi_EA);
        let (rho, r_star, xi_pi, xi_gamma, pi_star) = estimate_taylor_rule(
            &e.r_bar_series[..n],
            &m.ext.Y_EA_series[..n],
            &m.ext.pi_EA_series[..n],
        );
        let cb = &m.cb;
        assert_eq!(
            (cb.rho, cb.r_star, cb.xi_pi, cb.xi_gamma, cb.pi_star),
            (rho, r_star, xi_pi, xi_gamma, pi_star)
        );
    }
}

/// With `theta_calvo = 0` no firm adjusts its price for cost push or demand
/// pull: every price grows with expected inflation only.
#[test]
fn canvas_without_calvo_adjustment() {
    let s = &*ITALY2010Q1;
    let mut p = s.parameters.clone();
    p.theta_calvo = 0.0;
    let rng = &mut Xoshiro::seed_from_u64(4);
    let mut m = ModelCANVAS::new(&p, &s.initial_conditions, rng).unwrap();
    m.step(rng);
    m.collect_data();
    m.finance_insolvent_firms();
    m.set_growth_inflation_expectations(rng);
    let P_i = m.firms.P_i.clone();
    let pi_e = m.agg.pi_e;
    let d = m.firms_expectations_and_decisions(rng);
    for (new, old) in d.new_P_i.iter().zip(&P_i) {
        assert_eq!(*new, old * (1.0 + 0.0) * (1.0 + pi_e) * (1.0 + 0.0));
    }
}

/// The growth-rate model estimates its AR(1) processes on the growth rates
/// of the first `T_prime` quarters, and without shocks government
/// consumption, exports and imports follow them exactly.
#[test]
fn growth_rate_processes() {
    let s = &*AUSTRIA2010Q1;
    let ic = &s.initial_conditions;
    let rng = &mut Deterministic;
    let mut m = ModelGR::new(&s.parameters, ic, rng).unwrap();
    let T_prime = m.prop.T_prime;
    let growth = |x: &[f64]| -> Vec<f64> {
        x[..T_prime]
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    };
    let (alpha, beta, sigma, _) = estimate_for_calibration_script(&growth(&ic.C_G));
    assert_eq!(
        (m.ext.alpha_G_gr, m.ext.beta_G_gr, m.ext.sigma_G_gr),
        (alpha, beta, sigma)
    );
    let (alpha, beta, sigma, _) = estimate_for_calibration_script(&growth(&ic.Y_I));
    assert_eq!(
        (m.ext.alpha_I_gr, m.ext.beta_I_gr, m.ext.sigma_I_gr),
        (alpha, beta, sigma)
    );
    assert_eq!(m.ext.g_prev_C_E, *growth(&ic.C_E).last().unwrap());

    // Deterministic draws 0 for every normal number.
    let e = m.ext.clone();
    let (C_G, C_E, Y_I) = (m.gov.C_G, m.rotw.C_E, m.rotw.Y_I);
    m.step(rng);
    let g_G = e.alpha_G_gr * e.g_prev_C_G + e.beta_G_gr;
    let g_E = e.alpha_E_gr * e.g_prev_C_E + e.beta_E_gr;
    let g_I = e.alpha_I_gr * e.g_prev_Y_I + e.beta_I_gr;
    assert_eq!(m.gov.C_G, C_G * (1.0 + g_G));
    assert_eq!(m.rotw.C_E, C_E * (1.0 + g_E));
    assert_eq!(m.rotw.Y_I, Y_I * (1.0 + g_I));
    assert_eq!(
        (m.ext.g_prev_C_G, m.ext.g_prev_C_E, m.ext.g_prev_Y_I),
        (g_G, g_E, g_I)
    );
}

#[test]
fn canvas_needs_its_series() {
    let s = &*AUSTRIA2010Q1;
    let error =
        ModelCANVAS::new(&s.parameters, &s.initial_conditions, &mut Deterministic).unwrap_err();
    assert_eq!(error, "initial_conditions has no Y_EA_series");
    let mut ic = ITALY2010Q1.initial_conditions.clone();
    ic.r_bar_series.truncate(10);
    let error = ModelCANVAS::new(&ITALY2010Q1.parameters, &ic, &mut Deterministic).unwrap_err();
    assert_eq!(
        error,
        "r_bar_series has 10 quarters, T_prime = 54 are needed"
    );
}
