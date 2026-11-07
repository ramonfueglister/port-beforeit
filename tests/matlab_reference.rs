//! Comparison with the authors' reference values without randomness, as in
//! BeforeIT `test/deterministic`: the initial state, every rule of one
//! quarter, and the aggregates after quarters 1 and 5, against the values
//! of the authors' MATLAB code in `test/matlab_code`.

#![allow(non_snake_case)]

mod common;

use beforeit::{Deterministic, Model, Workers};
use common::*;

fn model() -> Model {
    let (p, ic) = austria();
    Model::new(&p, &ic, &mut Deterministic).unwrap()
}

/// One quantity as computed here and as stored by the MATLAB code, with the
/// relative tolerance of the authors' test.
struct Pair {
    name: &'static str,
    ours: Vec<f64>,
    matlab: Vec<f64>,
    rtol: f64,
}

/// The quantities of `initialize_deterministic.jl`: firms, bank and
/// households after the model is built.
fn initial_state() -> Vec<Pair> {
    let m = model();
    let (f, w_act, w_inact, b) = (&m.firms, &m.w_act, &m.w_inact, &m.banks);
    let pair = |name, ours, matlab| Pair {
        name,
        ours,
        matlab,
        rtol: SQRT_EPS,
    };
    let mut pairs = Vec::new();

    let firms = load("matlab_init_vars_firms");
    // MATLAB numbers sectors and firms from 1.
    let G_i = f.G_i.iter().map(|&g| g as f64 + 1.0).collect();
    let fields: [(&str, Vec<f64>); 20] = [
        ("G_i", G_i),
        ("alpha_bar_i", f.alpha_bar_i.clone()),
        ("beta_i", f.beta_i.clone()),
        ("kappa_i", f.kappa_i.clone()),
        ("w_bar_i", f.w_bar_i.clone()),
        ("delta_i", f.delta_i.clone()),
        ("tau_Y_i", f.tau_Y_i.clone()),
        ("tau_K_i", f.tau_K_i.clone()),
        ("N_i", as_f64(&f.N_i)),
        ("Y_i", f.Y_i.clone()),
        ("Q_d_i", f.Q_d_i.clone()),
        ("P_i", f.P_i.clone()),
        ("S_i", f.S_i.clone()),
        ("K_i", f.K_i.clone()),
        ("M_i", f.M_i.clone()),
        ("L_i", f.L_i.clone()),
        ("pi_bar_i", f.pi_bar_i.clone()),
        ("D_i", f.D_i.clone()),
        ("Pi_i", f.Pi_i.clone()),
        ("V_i", as_f64(&f.V_i)),
    ];
    for (name, ours) in fields {
        pairs.push(pair(name, ours, list(&firms, name)));
    }
    // There, income, capital and deposits of the firm owners are part of the
    // lists of all households: workers, inactive persons, firm owners, bank
    // owner.
    let owners = w_act.len() + w_inact.len()..w_act.len() + w_inact.len() + f.len();
    for (name, ours) in [("Y_h", &f.Y_h), ("K_h", &f.K_h), ("D_h", &f.D_h)] {
        let matlab = list(&firms, name)[owners.clone()].to_vec();
        pairs.push(pair(name, ours.clone(), matlab));
    }

    let bank = load("matlab_init_vars_bank");
    for (name, ours) in [
        ("E_k", &b.E_k),
        ("r", &b.r),
        ("Pi_k", &b.Pi_k),
        ("D_k", &b.D_k),
    ] {
        pairs.push(pair(name, ours.clone(), list(&bank, name)));
    }

    let households = load("matlab_init_vars_households");
    let O_h = w_act
        .O_h
        .iter()
        .map(|o| o.map_or(0.0, |i| i as f64 + 1.0))
        .collect();
    pairs.push(pair("O_h", O_h, list(&households, "O_h")));
    pairs.push(pair("w_h", w_act.w_h.clone(), list(&households, "w_h")));
    let all = |field: fn(&Workers) -> &Vec<f64>, firms: &[f64], bank: &[f64]| {
        let mut v = field(w_act).clone();
        v.extend(field(w_inact));
        v.extend(firms);
        v.extend(bank);
        v
    };
    for (name, ours) in [
        ("Y_h", all(|w| &w.Y_h, &f.Y_h, &b.Y_h)),
        ("D_h", all(|w| &w.D_h, &f.D_h, &b.D_h)),
        ("K_h", all(|w| &w.K_h, &f.K_h, &b.K_h)),
    ] {
        pairs.push(pair(name, ours, list(&households, name)));
    }
    pairs
}

/// The aggregates of `deterministic_ouput_t1_t5.jl` after `T` quarters, one
/// value per quarter without the initial state, as the MATLAB code stores
/// them. Tolerances of the authors: `√eps` after one quarter, `1e-4` for the
/// series and `1e-5` for value added by sector after five.
fn after_quarters(T: usize) -> Vec<Pair> {
    let mut m = model();
    m.run(T, &mut Deterministic);
    assert_eq!(m.data.collection_time, (1..=T + 1).collect::<Vec<_>>());
    let out = load(&format!("matlab_output_t{T}"));
    let d = &m.data;
    let series: [(&str, &Vec<f64>); 25] = [
        ("nominal_gdp", &d.nominal_gdp),
        ("real_gdp", &d.real_gdp),
        ("nominal_gva", &d.nominal_gva),
        ("real_gva", &d.real_gva),
        (
            "nominal_household_consumption",
            &d.nominal_household_consumption,
        ),
        ("real_household_consumption", &d.real_household_consumption),
        (
            "nominal_government_consumption",
            &d.nominal_government_consumption,
        ),
        (
            "real_government_consumption",
            &d.real_government_consumption,
        ),
        ("nominal_capitalformation", &d.nominal_capitalformation),
        ("real_capitalformation", &d.real_capitalformation),
        (
            "nominal_fixed_capitalformation",
            &d.nominal_fixed_capitalformation,
        ),
        (
            "real_fixed_capitalformation",
            &d.real_fixed_capitalformation,
        ),
        (
            "nominal_fixed_capitalformation_dwellings",
            &d.nominal_fixed_capitalformation_dwellings,
        ),
        (
            "real_fixed_capitalformation_dwellings",
            &d.real_fixed_capitalformation_dwellings,
        ),
        ("nominal_exports", &d.nominal_exports),
        ("real_exports", &d.real_exports),
        ("nominal_imports", &d.nominal_imports),
        ("real_imports", &d.real_imports),
        ("operating_surplus", &d.operating_surplus),
        ("compensation_employees", &d.compensation_employees),
        ("wages", &d.wages),
        ("taxes_production", &d.taxes_production),
        ("euribor", &d.euribor),
        ("gdp_deflator_growth_ea", &d.gdp_deflator_growth_ea),
        ("real_gdp_ea", &d.real_gdp_ea),
    ];
    let mut pairs: Vec<Pair> = series
        .into_iter()
        .map(|(name, ours)| Pair {
            name,
            ours: ours[1..].to_vec(),
            matlab: list(&out, name),
            rtol: if T == 1 { SQRT_EPS } else { 1e-4 },
        })
        .collect();
    for (name, ours) in [
        ("nominal_sector_gva", &d.nominal_sector_gva),
        ("real_sector_gva", &d.real_sector_gva),
    ] {
        // Quarter by quarter, one value per sector.
        let matlab = if T == 1 {
            list(&out, name)
        } else {
            matrix(&out, name).into_iter().flatten().collect()
        };
        pairs.push(Pair {
            name,
            ours: ours[1..].iter().flatten().copied().collect(),
            matlab,
            rtol: if T == 1 { SQRT_EPS } else { 1e-5 },
        });
    }
    pairs
}

#[test]
fn initial_state_matches_matlab() {
    for p in initial_state() {
        check_list(p.name, &p.ours, &p.matlab, p.rtol, 0.0);
    }
}

#[test]
fn quarters_1_and_5_match_matlab() {
    for p in after_quarters(1).into_iter().chain(after_quarters(5)) {
        check_list(p.name, &p.ours, &p.matlab, p.rtol, 0.0);
    }
}

/// Relative deviation in the sense of Julia's `isapprox`: the norm of the
/// difference divided by the larger of the two norms.
fn relative_deviation(x: &[f64], y: &[f64]) -> f64 {
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|a| a * a).sum::<f64>().sqrt();
    let size = norm(&mut x.iter().copied()).max(norm(&mut y.iter().copied()));
    if size == 0.0 {
        return 0.0;
    }
    norm(&mut x.iter().zip(y).map(|(a, b)| a - b)) / size
}

/// Stricter than the authors' tolerances: all quantities of the two tests
/// above deviate from the MATLAB values by at most 1e-10 (relative).
#[test]
fn largest_deviation_from_matlab_is_below_1e_10() {
    let runs = [
        ("initial state", initial_state()),
        ("one quarter", after_quarters(1)),
        ("five quarters", after_quarters(5)),
    ];
    let (mut worst, mut name, mut run) = (0.0, "", "");
    for (r, pairs) in &runs {
        for p in pairs {
            assert_eq!(p.ours.len(), p.matlab.len(), "{}", p.name);
            let d = relative_deviation(&p.ours, &p.matlab);
            assert!(d.is_finite(), "{}: deviation {d}", p.name);
            if d > worst {
                (worst, name, run) = (d, p.name, r);
            }
        }
    }
    println!("largest relative deviation from the MATLAB values: {worst:.1e} ({name}, {run})");
    assert!(
        worst <= 1e-10,
        "{name}, {run}: relative deviation {worst:e}"
    );
}

/// `one_epoch_deterministic.jl`: one quarter rule by rule, with the authors'
/// values and tolerances. As there, the price indices are not updated
/// before the stocks.
#[test]
fn one_quarter_rule_by_rule() {
    let mut m = model();
    let rng = &mut Deterministic;

    m.set_growth_inflation_expectations(rng);
    check("Y_e", m.agg.Y_e, 134929.5631, SQRT_EPS, 0.0);
    check("gamma_e", m.agg.gamma_e, 0.0021822, 1e-4, 0.0);

    m.set_epsilon(rng);
    m.set_growth_inflation_EA(rng);
    check("gamma_EA", m.rotw.gamma_EA, 0.0016278, 1e-5, 0.0);
    check("Y_EA", m.rotw.Y_EA, 2358680.8201, 1e-5, 0.0);
    check("pi_EA", m.rotw.pi_EA, 0.0033723, 1e-5, 0.0);

    m.set_central_bank_rate();
    check("r_bar", m.cb.r_bar, 0.0017616, 1e-4, 0.0);
    m.set_banks_rate();
    check("r", mean(&m.banks.r), 0.028476, 1e-4, 0.0);

    m.set_firms_expectations_and_decisions(rng);
    let f = &m.firms;
    check("Q_s_i", mean(&f.Q_s_i), 220.0311, 1e-6, 0.0);
    check("I_d_i", mean(&f.I_d_i), 21.6029, 1e-5, 0.0);
    check("DM_d_i", mean(&f.DM_d_i), 110.8158, 1e-5, 0.0);
    check("N_d_i", mean(&as_f64(&f.N_d_i)), 6.2436, 1e-5, 0.0);
    check("Pi_e_i", mean(&f.Pi_e_i), 17.5269, 1e-5, 0.0);
    check("DL_d_i", mean(&f.DL_d_i), 6.3063, 1e-5, 0.0);
    check("K_e_i", mean(&f.K_e_i), 1265.3191, 1e-5, 0.0);
    check("L_e_i", mean(&f.L_e_i), 360.694, 1e-5, 0.0);
    check("P_i", mean(&f.P_i), 1.0031, 1e-4, 0.0);

    m.search_and_matching_credit(rng);
    let granted: Vec<f64> = m.firms.DL_i.iter().copied().filter(|&x| x > 0.0).collect();
    check("DL_i", mean(&granted), 95.9791, 1e-6, 0.0);

    m.search_and_matching_labour(rng);
    m.set_firms_wages();
    m.set_firms_production();
    check("w_i", mean(&m.firms.w_i), 6.6122, 1e-5, 0.0);
    check("Y_i", mean(&m.firms.Y_i), 220.0311, 1e-6, 0.0);

    m.update_workers_wages();
    check("w_h", mean(&m.w_act.w_h), 7.5221, 1e-5, 0.0);

    m.set_gov_social_benefits();
    check("sb_other", m.gov.sb_other, 0.59157, 1e-5, 0.0);
    check("sb_inact", m.gov.sb_inact, 2.2434, 1e-4, 0.0);

    m.set_banks_expected_profits();
    check("Pi_e_k", mean(&m.banks.Pi_e_k), 6510.4793, 1e-5, 0.0);

    m.set_households_budget_act();
    m.set_households_budget_inact();
    m.set_households_budget_firms();
    m.set_households_budget_banks();
    // Sum over all households: workers, inactive persons, firm owners, bank
    // owners.
    let total = |m: &Model, f: fn(&Workers) -> f64, firms: f64, bank: f64| {
        f(&m.w_act) + f(&m.w_inact) + firms + bank
    };
    let C_d_h = total(
        &m,
        |w| w.C_d_h.iter().sum(),
        m.firms.C_d_h.iter().sum(),
        m.banks.C_d_h.iter().sum(),
    );
    let I_d_h = total(
        &m,
        |w| w.I_d_h.iter().sum(),
        m.firms.I_d_h.iter().sum(),
        m.banks.I_d_h.iter().sum(),
    );
    check("C_d_h", C_d_h, 35538.3159, 1e-9, 1e-6);
    check("I_d_h", I_d_h, 2950.5957, 1e-6, 1e-6);

    m.set_gov_expenditure(rng);
    check("C_G", m.gov.C_G, 14783.2494, 1e-6, 1e-6);
    check("C_d_j", mean(&m.gov.C_d_j), 95.0572, 1e-6, 1e-6);

    m.set_rotw_import_export(rng);
    check("C_E", m.rotw.C_E, 34246.8702, 1e-6, 1e-6);
    check("C_d_l", mean(&m.rotw.C_d_l), 110.1048, 1e-6, 1e-6);
    check("Y_I", m.rotw.Y_I, 33214.9736, 1e-6, 1e-6);
    check("Y_m", mean(&m.rotw.Y_m), 535.7254, 1e-6, 1e-6);
    check("P_m", mean(&m.rotw.P_m), 1.0031, 1e-4, 1e-6);

    m.search_and_matching(rng);
    let C_h = total(
        &m,
        |w| w.C_h.iter().sum(),
        m.firms.C_h.iter().sum(),
        m.banks.C_h.iter().sum(),
    );
    let I_h = total(
        &m,
        |w| w.I_h.iter().sum(),
        m.firms.I_h.iter().sum(),
        m.banks.I_h.iter().sum(),
    );
    let K_h = total(
        &m,
        |w| w.K_h.iter().sum(),
        m.firms.K_h.iter().sum(),
        m.banks.K_h.iter().sum(),
    );
    check("C_h", C_h, 35136.4805, 1e-4, 1e-4);
    check("I_h", I_h, 2699.6511, 1e-6, 1e-6);
    check("K_h", K_h, 408076.5511, 1e-6, 1e-6);
    let f = &m.firms;
    check("Q_d_i", mean(&f.Q_d_i), 220.092, 1e-6, 0.0);
    check("Q_i", mean(&f.Q_i), 216.6644, 1e-6, 0.0);
    check("Q_d_m", mean(&m.rotw.Q_d_m), 527.2969, 1e-6, 0.0);
    check("Q_m", mean(&m.rotw.Q_m), 527.2969, 1e-5, 0.0);
    check("I_i", mean(&f.I_i), 21.6029, 1e-5, 0.0);
    check("DM_i", mean(&f.DM_i), 110.7829, 1e-6, 0.0);
    check("P_CF_i", mean(&f.P_CF_i), 1.0031, 1e-4, 0.0);
    check("P_bar_i", mean(&f.P_bar_i), 1.0031, 1e-4, 0.0);
    check("C_j", m.gov.C_j, 14370.3493, 1e-5, 0.0);
    check("P_j", m.gov.P_j, 1.0031, 1e-4, 0.0);

    m.set_firms_stocks();
    let f = &m.firms;
    check("K_i", mean(&f.K_i), 1261.4216, 1e-6, 0.0);
    check("M_i", mean(&f.M_i), 130.0548, 1e-6, 0.0);
    check("DS_i", mean(&f.DS_i), 3.3667, 0.0, 1e-5);
    check("S_i", mean(&f.S_i), 3.3667, 0.0, 1e-5);

    m.set_firms_profits();
    check("Pi_i", mean(&m.firms.Pi_i), 17.5491, 1e-2, 0.0);
    m.set_banks_profits();
    check("Pi_k", mean(&m.banks.Pi_k), 6486.6381, 1e-5, 0.0);
    m.set_banks_equity();
    check("E_k", mean(&m.banks.E_k), 90742.39, 1e-5, 0.0);

    m.set_households_income_act(false);
    m.set_households_income_inact(false);
    m.set_households_income_firms(false);
    m.set_households_income_banks(false);
    m.set_households_deposits_act();
    m.set_households_deposits_inact();
    m.set_households_deposits_firms();
    m.set_households_deposits_banks();
    let Y_h = total(
        &m,
        |w| w.Y_h.iter().sum(),
        m.firms.Y_h.iter().sum(),
        m.banks.Y_h.iter().sum(),
    );
    let D_h = total(
        &m,
        |w| w.D_h.iter().sum(),
        m.firms.D_h.iter().sum(),
        m.banks.D_h.iter().sum(),
    );
    check("Y_h", Y_h, 45032.3263, 1e-2, 0.0);
    check("D_h", D_h, 221816.6764, 1e-3, 0.0);

    let E_CB = m.cb.E_CB;
    m.set_central_bank_equity();
    check("Pi_CB", m.cb.E_CB - E_CB, 1866.3821, 1e-5, 0.0);

    m.set_gov_revenues();
    check("Y_G", m.gov.Y_G, 28783.0089, 1e-2, 0.0);
    let L_G = m.gov.L_G;
    m.set_gov_loans();
    check("Pi_G", m.gov.L_G - L_G, 3140.6916, 1e-2, 0.0);
    check("L_G", m.gov.L_G, 235751.5916, 1e-4, 0.0);

    let D_i = m.firms.D_i.clone();
    m.set_firms_deposits();
    let DD_i: Vec<f64> = m.firms.D_i.iter().zip(&D_i).map(|(a, b)| a - b).collect();
    m.set_firms_loans();
    m.set_firms_equity();
    check("DD_i", mean(&DD_i), -14.8245, 1e-2, 0.0);
    check("D_i", mean(&m.firms.D_i), 71.7925, 1e-3, 0.0);
    check("L_i", mean(&m.firms.L_i), 367.0003, 1e-5, 0.0);
    check("E_i", mean(&m.firms.E_i), 1103.945, 1e-2, 0.0);
    check("E_CB", m.cb.E_CB, 108046.2821, 1e-2, 0.0);

    m.set_rotw_deposits();
    check("D_RoW", m.rotw.D_RoW, -644.0817, 1e-5, 0.0);
    m.set_banks_deposits();
    check("D_k", mean(&m.banks.D_k), 128349.3912, 1e-3, 0.0);
}
