//! Predictions from several quarters, BeforeIT
//! `examples/prediction_pipeline_multiple.jl`: the parameters and initial
//! conditions of Italy for each quarter from 2014Q1 to 2014Q4 at scale
//! 1:2000, 2 runs of 16 quarters from each, the predictions from the runs,
//! and six quarterly series of the predictions from 2014Q2 against the data.
//!
//! ```sh
//! cargo run --release --features plots --example prediction_pipeline_multiple
//! ```
//!
//! As in Julia, the files go to the folder `data/italy` of the working
//! directory, which the example sets to `target/example-output`; **the
//! folder is deleted first if it exists**. The files are JSON instead of
//! JLD2. The plot is written as SVG to
//! `target/example-output/prediction_pipeline_multiple`.

#![allow(non_snake_case)]

mod common;

use beforeit::analysis::{
    save_all_params_and_initial_conditions, save_all_predictions_from_sims, save_all_simulations,
};
use beforeit::calibration::table_from_json;
use beforeit::plots::{DEFAULT_SIZE, Legend, layout_svg, plot_model_vs_real};
use beforeit::{Date, ITALY_CALIBRATION, Model, Xoshiro};

fn main() -> Result<(), String> {
    let dir = common::output_dir("prediction_pipeline_multiple");
    common::enter_work_dir()?;
    let mut rng = Xoshiro::seed_from_u64(1);

    // The folder for all the data.
    let folder_name = "data/italy";

    // Parameters and initial conditions from a first to a last date,
    let mut cal = ITALY_CALIBRATION.clone();
    let first_calibration_date = Date::new(2014, 3, 31);
    let last_calibration_date = Date::new(2014, 12, 31);

    // saved in the folder.
    save_all_params_and_initial_conditions(
        &cal,
        folder_name,
        0.0005,
        first_calibration_date,
        last_calibration_date,
    )?;
    // Julia's calibration also stores the GDP deflators in the data it is
    // given, which the predictions below use.
    cal.set_gdp_deflators()?;

    // n_sims runs of T quarters from the parameters and initial conditions
    // of each quarter in the folder, each ensemble saved as a `DataVector`
    // in `simulations/YYYYQn.json`.
    save_all_simulations(folder_name, 16, 2, Model::new, None, &mut rng)?;

    // The runs aligned with the data, as predictions.
    let real_data = &cal.data;
    save_all_predictions_from_sims(folder_name, real_data, "simulations", "abm_predictions")?;

    // The predictions from one quarter against the data.
    let (y, q) = (2014, 2);
    let file = format!("data/italy/abm_predictions/{y}Q{q}.json");
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
    let predictions_dict = table_from_json(&text)?;

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
