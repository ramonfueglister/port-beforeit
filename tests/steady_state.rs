//! The steady-state data set, as in BeforeIT `test/steady_state.jl`.

#![allow(non_snake_case)]

use beforeit::{Data, Model, STEADY_STATE2010Q1, Xoshiro};

/// Sum of the absolute deviations from the mean, Julia
/// `sum(abs.(x .- mean(x)))`.
fn spread(x: &[f64]) -> f64 {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| (v - mean).abs()).sum()
}

macro_rules! s {
    ($d:ident.$f:ident) => {
        (stringify!($f), &$d.$f)
    };
}

/// After five quarters, every national accounts series of the data is
/// constant within the tolerances of the Julia test: 1e-7 for each series,
/// 1e-6 for each sector of the series per sector. The Julia test loops over
/// all fields of `Data`, which today also include `collection_time`, the
/// quarter counter, and the series per bank that BeforeIT.jl added later;
/// the test is not part of BeforeIT's `runtests.jl` and fails on those
/// fields, since they are not constant (the loans and reserves of the bank
/// change from quarter to quarter in this data set). They are left out here.
#[test]
fn every_series_stays_constant() {
    let s = &*STEADY_STATE2010Q1;
    for seed in [1, 2, 3] {
        let rng = &mut Xoshiro::seed_from_u64(seed);
        let mut m = Model::new(&s.parameters, &s.initial_conditions, rng).unwrap();
        m.run(5, rng);
        let d: &Data = &m.data;
        let flat: [(&str, &Vec<f64>); 25] = [
            s!(d.nominal_gdp),
            s!(d.real_gdp),
            s!(d.nominal_gva),
            s!(d.real_gva),
            s!(d.nominal_household_consumption),
            s!(d.real_household_consumption),
            s!(d.nominal_government_consumption),
            s!(d.real_government_consumption),
            s!(d.nominal_capitalformation),
            s!(d.real_capitalformation),
            s!(d.nominal_fixed_capitalformation),
            s!(d.real_fixed_capitalformation),
            s!(d.nominal_fixed_capitalformation_dwellings),
            s!(d.real_fixed_capitalformation_dwellings),
            s!(d.nominal_exports),
            s!(d.real_exports),
            s!(d.nominal_imports),
            s!(d.real_imports),
            s!(d.operating_surplus),
            s!(d.compensation_employees),
            s!(d.wages),
            s!(d.taxes_production),
            s!(d.gdp_deflator_growth_ea),
            s!(d.real_gdp_ea),
            s!(d.euribor),
        ];
        for (name, x) in flat {
            assert_eq!(x.len(), 6);
            assert!(spread(x) <= 1e-7, "seed {seed}, {name}: {:e}", spread(x));
        }
        for (name, x) in [s!(d.nominal_sector_gva), s!(d.real_sector_gva)] {
            for g in 0..x[0].len() {
                let column: Vec<f64> = x.iter().map(|row| row[g]).collect();
                let z = spread(&column);
                assert!(z <= 1e-6, "seed {seed}, {name}[{g}]: {z:e}");
            }
        }
    }
}
