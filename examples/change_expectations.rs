//! Changing the expectations of the agents, BeforeIT
//! `examples/change_expectations.jl`: the model of Austria run for 40
//! quarters with the AR(1) forecasts of the paper (forward looking) and,
//! from the same seed, with agents who expect the last value of a series
//! (backward looking); both plotted in the same panels.
//!
//! ```sh
//! cargo run --release --features plots --example change_expectations
//! ```
//!
//! The plots are written as SVG to
//! `target/example-output/change_expectations`.
//!
//! Julia redefines `Bit.estimate_next_value` for all models, after which it
//! builds the second model with `Bit.Model`; here the second model is a
//! model of a type that replaces `estimate_next_value`, built from the same
//! seed.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_SIZE, Legend, Plot, layout_svg};
use beforeit::{AUSTRIA2010Q1, Data, Extension, Model, Randomness, Xoshiro};

/// Backward-looking expectations: the next value of a series is its last.
#[derive(Clone, Debug)]
struct BackwardLooking;

impl Extension for BackwardLooking {
    fn estimate_next_value<R: Randomness + ?Sized>(data: &[f64], _rng: &mut R) -> f64 {
        data[data.len() - 1]
    }
}

/// Plots.jl's `plot(y1, title = title, titlefont = 10)` followed by
/// `plot!(p, y2)`.
fn compare(title: &str, y1: Vec<f64>, y2: Vec<f64>) -> Plot {
    let mut p = Plot::new(title);
    p.title_font = 10;
    p.add_line(common::one_to(y1.len()), y1);
    p.add_line(common::one_to(y2.len()), y2);
    p
}

fn deflator(d: &Data) -> Vec<f64> {
    d.nominal_gdp
        .iter()
        .zip(&d.real_gdp)
        .map(|(n, r)| n / r)
        .collect()
}

fn main() -> Result<(), String> {
    let dir = common::output_dir("change_expectations");
    let par = &AUSTRIA2010Q1.parameters;
    let init = &AUSTRIA2010Q1.initial_conditions;

    // Set the seed, initialise the model and run one simulation.
    let mut rng = Xoshiro::seed_from_u64(1234);
    let mut model = Model::new(par, init, &mut rng)?;
    model.run(40, &mut rng);

    // The same with backward-looking expectations and the same seed.
    let mut rng = Xoshiro::seed_from_u64(1234);
    let mut model2 = common::with_rules(Model::new(par, init, &mut rng)?, BackwardLooking);
    model2.run(40, &mut rng);
    let (d1, d2) = (&model.data, &model2.data);

    // The two cases as different lines.
    let mut p1 = compare("gdp", d1.real_gdp.clone(), d2.real_gdp.clone());
    let mut p2 = compare(
        "consumption",
        d1.real_household_consumption.clone(),
        d2.real_household_consumption.clone(),
    );
    for p in [&mut p1, &mut p2] {
        p.series[0].label = Some("forward looking".to_string());
        p.series[1].label = Some("backward looking".to_string());
    }
    common::save(
        &dir,
        "gdp_consumption.svg",
        &layout_svg(&[p1, p2], (2, 1), DEFAULT_SIZE)?,
    )?;

    // All time series, without legend.
    let mut ps = vec![
        compare("gdp", d1.real_gdp.clone(), d2.real_gdp.clone()),
        compare(
            "household cons.",
            d1.real_household_consumption.clone(),
            d2.real_household_consumption.clone(),
        ),
        compare(
            "gov. cons.",
            d1.real_government_consumption.clone(),
            d2.real_government_consumption.clone(),
        ),
        compare(
            "capital form.",
            d1.real_capitalformation.clone(),
            d2.real_capitalformation.clone(),
        ),
        compare("exports", d1.real_exports.clone(), d2.real_exports.clone()),
        compare("imports", d1.real_imports.clone(), d2.real_imports.clone()),
        compare("wages", d1.wages.clone(), d2.wages.clone()),
        compare("euribor", d1.euribor.clone(), d2.euribor.clone()),
        compare("gdp deflator", deflator(d1), deflator(d2)),
    ];
    for p in &mut ps {
        p.legend = Legend::None;
    }
    common::save(&dir, "all.svg", &layout_svg(&ps, (3, 3), DEFAULT_SIZE)?)?;
    Ok(())
}
