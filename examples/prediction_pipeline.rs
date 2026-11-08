//! The whole prediction pipeline, BeforeIT `examples/prediction_pipeline.jl`:
//! the calibration object of Italy, the model calibrated on 2014Q1 at scale
//! 1:10 000, 3 runs of 20 quarters, the predictions from the runs, and six
//! quarterly series of the predictions against the data.
//!
//! ```sh
//! cargo run --release --features plots --example prediction_pipeline
//! ```
//!
//! The plot is written as SVG to `target/example-output/prediction_pipeline`.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_SIZE, Legend, layout_svg, plot_model_vs_real};
use beforeit::{
    DataVector, Date, ITALY_CALIBRATION, Model, NoShock, Xoshiro, ensemblerun,
    get_params_and_initial_conditions, get_predictions_from_sims,
};

fn main() -> Result<(), String> {
    let dir = common::output_dir("prediction_pipeline");
    let mut rng = Xoshiro::seed_from_u64(1);

    // The calibration object of Italy with its four tables `calibration`,
    // `figaro`, `data` and `ea`. BeforeIT.jl contains them for Italy; other
    // countries need them made in the same form.
    let mut cal = ITALY_CALIBRATION.clone();
    // Julia lists the fields with `fieldnames(typeof(cal))`; here they are
    // the public fields of `CalibrationData`.
    println!("calibration, figaro, data, ea, max_calibration_date, estimation_date");

    // The keys of the four tables (sorted; Julia prints those of a `Dict`
    // in the order of their hashes),
    println!("{:?}", cal.calibration.keys().collect::<Vec<_>>());
    println!("{:?}", cal.figaro.keys().collect::<Vec<_>>());
    println!("{:?}", cal.data.keys().collect::<Vec<_>>());
    println!("{:?}", cal.ea.keys().collect::<Vec<_>>());

    // and the two dates of the data.
    println!("{}", common::datetime(cal.max_calibration_date));
    println!("{}", common::datetime(cal.estimation_date));

    // The calibration on one quarter.
    let calibration_date = Date::new(2014, 3, 31);
    let c = get_params_and_initial_conditions(&cal, calibration_date, 0.0001, false)?;
    for w in &c.warnings {
        eprintln!("Warning: {w}");
    }
    // Julia's calibration also stores the GDP deflators in the data it
    // is given, which the predictions below use.
    cal.set_gdp_deflators()?;

    // Runs of T quarters.
    let T = 20;
    let n_sims = 3;
    let model = Model::new(&c.parameters, &c.initial_conditions, &mut rng)?;
    let mut model_vector = vec![model; n_sims];
    ensemblerun(&mut model_vector, T, &mut rng, true, &NoShock);

    // The predictions from the runs.
    let real_data = &cal.data;
    let predictions = get_predictions_from_sims(
        &DataVector::from_models(&model_vector),
        real_data,
        calibration_date,
    )?;
    let predictions_dict = predictions.to_table();

    // The predictions against the data.
    let ps = [
        "real_gdp_quarterly",
        "real_household_consumption_quarterly",
        "real_fixed_capitalformation_quarterly",
        "real_government_consumption_quarterly",
        "real_exports_quarterly",
        "real_imports_quarterly",
    ]
    .iter()
    .map(|v| {
        let mut p = plot_model_vs_real(&predictions_dict, real_data, v, true)?;
        p.legend = Legend::None;
        Ok(p)
    })
    .collect::<Result<Vec<_>, String>>()?;
    common::save(
        &dir,
        "predictions.svg",
        &layout_svg(&ps, (3, 2), DEFAULT_SIZE)?,
    )?;
    Ok(())
}
