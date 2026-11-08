//! Accounting identities, as in BeforeIT `test/accounting_identities.jl`:
//! income and production give the same value added, GDP is the sum of its
//! uses, and the balance sheets of the central bank and of each bank add
//! up.

#![allow(non_snake_case)]

mod common;

use beforeit::{
    Deterministic, Model, Randomness, Xoshiro, get_accounting_identities,
    get_accounting_identity_banks,
};
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
    let zero = m.cb.E_CB + m.rotw.D_RoW - m.gov.L_G + m.banks.D_k.iter().sum::<f64>();
    check("central bank balance sheet", zero, 0.0, 0.0, 1e-8);

    let (f, b) = (&m.firms, &m.banks);
    for k in 0..b.len() {
        let L_i: f64 = b.firms[k].iter().map(|&i| f.L_i[i]).sum();
        let D_i: f64 = b.firms[k].iter().map(|&i| f.D_i[i]).sum();
        let D_h: f64 = b.w_act[k].iter().map(|&h| m.w_act.D_h[h]).sum::<f64>()
            + b.w_inact[k].iter().map(|&h| m.w_inact.D_h[h]).sum::<f64>()
            + b.firms[k].iter().map(|&i| f.D_h[i]).sum::<f64>()
            + b.D_h[k];
        let zero = D_i + D_h + b.E_k[k] - L_i - b.D_k[k];
        check(&format!("balance sheet of bank {k}"), zero, 0.0, 0.0, 1e-8);
    }
}

fn model(n_banks: usize, rng: &mut dyn Randomness) -> Model {
    let (mut p, ic) = austria();
    p.n_banks = n_banks;
    Model::new(&p, &ic, rng).unwrap()
}

fn run(rng: &mut dyn Randomness) -> Model {
    let mut m = model(1, rng);
    m.run(3, rng);
    m
}

/// Loans of each bank's firms, `[sum(f.L_i for f in model.banks.firms[k];
/// init = 0.0) for k in 1:n_banks]`.
fn loans_per_bank(m: &Model) -> Vec<f64> {
    let b = &m.banks;
    (0..b.len())
        .map(|k| b.firms[k].iter().fold(0.0, |s, &i| s + m.firms.L_i[i]))
        .collect()
}

const NAMES: [&str; 3] = [
    "income against production",
    "nominal GDP against its uses",
    "real GDP against its uses",
];

/// The authors' test, with one and with three banks: three quarters with
/// randomness, each identity summed over the collections of data within
/// 1e-8, the balance sheets at the end within 1e-8, and the loans per bank
/// in the data equal to the loans of the bank's firms. With more than one
/// bank the GDP identities skip the initial state: the extra bank owners
/// replace workers and receive social benefits, so the initial state no
/// longer matches the calibrated household income.
#[test]
fn identities_hold_like_in_the_authors_test() {
    for n_banks in [1, 3] {
        for seed in [1, 2, 3] {
            let rng = &mut Xoshiro::seed_from_u64(seed);
            let mut m = model(n_banks, rng);
            assert_eq!(m.banks.len(), n_banks);
            let mut L_i_per_bank = vec![loans_per_bank(&m)];
            for _ in 0..3 {
                m.step(rng);
                m.collect_data();
                L_i_per_bank.push(loans_per_bank(&m));
            }
            let first = if n_banks == 1 { 0 } else { 1 };
            for (j, (name, residuals)) in NAMES.iter().zip(gdp_residuals(&m)).enumerate() {
                let from = if j == 0 { 0 } else { first };
                check(name, residuals[from..].iter().sum(), 0.0, 0.0, 1e-8);
            }
            balance_sheets(&m);
            for (t, loans) in L_i_per_bank.iter().enumerate() {
                for (k, &loans) in loans.iter().enumerate() {
                    check(
                        &format!("credit_stock_per_bank, collection {t}, bank {k}"),
                        m.data.credit_stock_per_bank[t][k],
                        loans,
                        0.0,
                        1e-8,
                    );
                }
            }
        }
    }
}

/// The authors' test `bank initialisation invariant to n_banks`: splitting
/// the sector into 1, 2 or 5 banks creates or destroys no equity, initial
/// profits or reserves.
#[test]
fn bank_initialisation_does_not_depend_on_the_number_of_banks() {
    let mut reference: Option<[f64; 3]> = None;
    for n_banks in [1, 2, 5] {
        let m = model(n_banks, &mut Xoshiro::seed_from_u64(1234));
        assert_eq!(m.banks.len(), n_banks);
        let b = &m.banks;
        let aggregates = [
            b.E_k.iter().sum::<f64>(),
            b.Pi_k.iter().sum::<f64>(),
            b.D_k.iter().sum::<f64>(),
        ];
        match reference {
            None => reference = Some(aggregates),
            Some(r) => {
                for (x, y) in aggregates.iter().zip(r) {
                    check("bank aggregates", *x, y, 1e-10, 0.0);
                }
            }
        }
    }
}

/// `get_accounting_identities` and `get_accounting_identity_banks` of the
/// library give the same zeros, with one and with three banks.
#[test]
fn library_functions_give_zero() {
    for n_banks in [1, 3] {
        let rng = &mut Xoshiro::seed_from_u64(5);
        let mut m = model(n_banks, rng);
        m.run(3, rng);
        let (income, nominal, real) = get_accounting_identities(&m.data);
        check("income against production", income, 0.0, 0.0, 1e-8);
        if n_banks == 1 {
            check("nominal GDP against its uses", nominal, 0.0, 0.0, 1e-8);
            check("real GDP against its uses", real, 0.0, 0.0, 1e-8);
        }
        let (cb_balance, bank_balance) = get_accounting_identity_banks(&m);
        check("central bank balance sheet", cb_balance, 0.0, 0.0, 1e-8);
        check("bank balance sheet", bank_balance, 0.0, 0.0, 1e-8);
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
