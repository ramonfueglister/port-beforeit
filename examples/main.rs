//! The script at the top of BeforeIT.jl, `main.jl`: a model of Austria from
//! 2010Q1 run for 20 quarters with the goods markets in parallel, and nine
//! of its series in a layout of 3 × 3 plots without legends.
//!
//! ```sh
//! cargo run --release --features plots --example main
//! ```
//!
//! The plot is written as SVG to `target/example-output/main`; Julia shows
//! it. The run draws its own random numbers (seed 1), so the curves are not
//! those of a Julia session.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_SIZE, Legend, Plot, layout_svg};
use beforeit::{AUSTRIA2010Q1, Model, NoShock, Xoshiro};

fn main() -> Result<(), String> {
    let dir = common::output_dir("main");
    let mut rng = Xoshiro::seed_from_u64(1);

    let parameters = &AUSTRIA2010Q1.parameters;
    let initial_conditions = &AUSTRIA2010Q1.initial_conditions;

    // Or with a calibration object of the Zenodo record (see the examples
    // `simulate_and_create_tables` and `crosscorrelations`):
    // let co = ...; // `Bit.download_zenodo_calibration_object("FR")`
    // let c = get_params_and_initial_conditions(&co, Date::new(2022, 12, 31), 1.0 / 1000.0, false)?;

    let mut model = Model::new(parameters, initial_conditions, &mut rng)?;

    let T = 20;
    for t in 1..=T {
        println!("Step: {t}");
        model.step_with(&mut rng, true, &NoShock);
        model.collect_data();
    }

    let data = &model.data;

    let nominal_over_real: Vec<f64> = data
        .nominal_gdp
        .iter()
        .zip(&data.real_gdp)
        .map(|(n, r)| n / r)
        .collect();
    let series = [
        ("gdp", &data.real_gdp),
        ("household cons.", &data.real_household_consumption),
        ("gov. cons.", &data.real_government_consumption),
        ("capital form.", &data.real_capitalformation),
        ("exports", &data.real_exports),
        ("imports", &data.real_imports),
        ("wages", &data.wages),
        ("euribor", &data.euribor),
        ("gdp deflator", &nominal_over_real),
    ];
    // `plot(y, title = ..., titlefont = 10)` for each, then
    // `plot(p1, ..., p9, layout = (3, 3), legend = false)`.
    let ps: Vec<Plot> = series
        .iter()
        .map(|(title, y)| {
            let mut p = Plot::new(*title);
            p.title_font = 10;
            p.add_line(common::one_to(y.len()), y.to_vec());
            p.legend = Legend::None;
            p
        })
        .collect();
    common::save(&dir, "main.svg", &layout_svg(&ps, (3, 3), DEFAULT_SIZE)?)?;
    Ok(())
}
