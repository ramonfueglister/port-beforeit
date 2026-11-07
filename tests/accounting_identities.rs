//! Accounting identities, as in BeforeIT `test/accounting_identities.jl`:
//! income and production give the same value added, GDP is the sum of its
//! uses, and the balance sheets of the central bank and the bank add up.

#![allow(non_snake_case)]

mod common;

use beforeit::{Deterministic, Model, Randomness, Xoshiro};
use common::*;

/// The identities of the GDP accounts, one value per collection of data.
fn gdp_residuals(m: &Model) -> [Vec<f64>; 3] {
    let d = &m.data;
    let n = d.collection_time.len();
    let income = (0..n)
        .map(|t| {
            d.nominal_gva[t]
                - d.compensation_employees[t]
                - d.operating_surplus[t]
                - d.taxes_production[t]
        })
        .collect();
    let nominal = (0..n)
        .map(|t| {
            d.nominal_gdp[t]
                - d.nominal_household_consumption[t]
                - d.nominal_government_consumption[t]
                - d.nominal_capitalformation[t]
                - d.nominal_exports[t]
                + d.nominal_imports[t]
        })
        .collect();
    let real = (0..n)
        .map(|t| {
            d.real_gdp[t]
                - d.real_household_consumption[t]
                - d.real_government_consumption[t]
                - d.real_capitalformation[t]
                - d.real_exports[t]
                + d.real_imports[t]
        })
        .collect();
    [income, nominal, real]
}

fn balance_sheets(m: &Model) {
    let zero = m.cb.E_CB + m.rotw.D_RoW - m.gov.L_G + m.bank.D_k;
    check("central bank balance sheet", zero, 0.0, 0.0, 1e-8);

    let D_h = m.w_act.D_h.iter().sum::<f64>()
        + m.w_inact.D_h.iter().sum::<f64>()
        + m.firms.D_h.iter().sum::<f64>()
        + m.bank.D_h;
    let zero = m.firms.D_i.iter().sum::<f64>() + D_h + m.bank.E_k
        - m.firms.L_i.iter().sum::<f64>()
        - m.bank.D_k;
    check("bank balance sheet", zero, 0.0, 0.0, 1e-8);
}

fn run(rng: &mut dyn Randomness) -> Model {
    let (p, ic) = austria();
    let mut m = Model::new(&p, &ic, rng).unwrap();
    m.run(3, rng);
    m
}

const NAMES: [&str; 3] = [
    "income against production",
    "nominal GDP against its uses",
    "real GDP against its uses",
];

/// The authors' test: three quarters with randomness, each identity summed
/// over all four collections of data within 1e-8, and the balance sheets
/// at the end within 1e-8.
#[test]
fn identities_hold_like_in_the_authors_test() {
    for seed in [1, 2, 3] {
        let m = run(&mut Xoshiro::seed_from_u64(seed));
        for (name, residuals) in NAMES.iter().zip(gdp_residuals(&m)) {
            check(name, residuals.iter().sum(), 0.0, 0.0, 1e-8);
        }
        balance_sheets(&m);
    }
}

/// Each collection on its own, within 1e-13 of GDP. The markets add up
/// thousands of small purchases, so rounding leaves residuals of the order
/// of 1e-14 of GDP.
#[test]
fn identities_hold_in_every_quarter() {
    for seed in [1, 2, 3] {
        let m = run(&mut Xoshiro::seed_from_u64(seed));
        for (name, residuals) in NAMES.iter().zip(gdp_residuals(&m)) {
            for (t, zero) in residuals.iter().enumerate() {
                let tol = 1e-13 * m.data.nominal_gdp[t].abs();
                check(&format!("{name}, collection {t}"), *zero, 0.0, 0.0, tol);
            }
        }
    }
}

/// Without randomness every firm of a sector gets the same number of
/// employees, which together differ slightly from the employees in the
/// initial conditions. The initial state then does not match the national
/// accounts; the quarters after it do.
#[test]
fn identities_hold_without_randomness() {
    let m = run(&mut Deterministic);
    for (name, residuals) in NAMES.iter().zip(gdp_residuals(&m)) {
        for (t, zero) in residuals.iter().enumerate().skip(1) {
            let tol = 1e-13 * m.data.nominal_gdp[t].abs();
            check(&format!("{name}, collection {t}"), *zero, 0.0, 0.0, tol);
        }
    }
    balance_sheets(&m);
}
