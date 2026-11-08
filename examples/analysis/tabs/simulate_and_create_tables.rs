//! Forecast error tables, BeforeIT
//! `examples/analysis/tabs/simulate_and_create_tables.jl`: the RMSE and the
//! bias of the forecasts of the CANVAS model for Austria from 2010Q1 to
//! 2010Q4 (2 runs of 12 quarters from each), against AR and VAR
//! benchmarks.
//!
//! ```sh
//! BEFOREIT_CALIBRATION_DIR=<folder> cargo run --release --example simulate_and_create_tables
//! ```
//!
//! Julia downloads the calibration object of Austria from Zenodo; the port
//! reads neither HTTP nor JLD2, so the example reads `AT.json` in the folder
//! `BEFOREIT_CALIBRATION_DIR`, converted from `AT_calibration_object.jld2`
//! of the record by `data/convert_calibration.py`. As in Julia, the files go
//! to the folder `data/AT` of the working directory, which the example sets
//! to `target/example-output`; **the folder is deleted first if it exists**
//! (`run_simulation`). The tables are in `data/AT/analysis/ModelCANVAS/`.

#![allow(non_snake_case)]

#[path = "../../common/mod.rs"]
mod common;

use beforeit::analysis::{
    FORECAST_HORIZONS, error_table_abm, error_table_ar, error_table_validation_abm,
    error_table_validation_var, save_all_params_and_initial_conditions,
    save_all_predictions_from_sims, save_all_simulations,
};
use beforeit::{Date, ModelCANVAS, Xoshiro};

fn main() -> Result<(), String> {
    common::enter_work_dir()?;
    let mut rng = Xoshiro::seed_from_u64(1);

    // Configuration.
    let country = "AT";
    let T = 12; // forecast horizon (quarters)
    let n_sims = 2; // ensemble size

    let horizons = FORECAST_HORIZONS; // [1, 2, 4, 8, 12]

    let starting_quarter = Date::new(2010, 3, 31);
    let ending_quarter = Date::new(2010, 12, 31);
    // let ending_quarter = Date::new(2019, 12, 31);

    // Set to false after simulations and predictions have been saved once.
    let run_simulation = true;

    // The model: `Model::new`, `ModelGR::new` or `ModelCANVAS::new`, and the
    // name of its variant, which names the output folders
    // `data/<country>/analysis/<variant>/` (Julia derives it from the type).
    let model_constructor = ModelCANVAS::new;
    let model_variant = "ModelCANVAS";

    eprintln!("Forecast error tables for {country} (variant: {model_variant})");

    let mut calibration = common::zenodo_calibration_object(country)?;

    let quarters = Date::every_months(starting_quarter, ending_quarter, 3);

    let folder = format!("data/{country}");

    // Parameters, simulations and predictions, in folders per variant.
    // `save_all_params_and_initial_conditions` makes `folder` anew and
    // DELETES it first if it exists. Use a scratch folder (such as
    // "data/IT"), not a kept one (such as "data/italy").
    if run_simulation {
        eprintln!("Generating parameters and initial conditions...");
        save_all_params_and_initial_conditions(
            &calibration,
            &folder,
            0.0005,
            starting_quarter,
            ending_quarter,
        )?;
        // Julia's calibration also stores the GDP deflators in the data it
        // is given, which the tables below use.
        calibration.set_gdp_deflators()?;

        eprintln!("Running simulations and extracting predictions (T={T}, n_sims={n_sims})...");
        let sim_subdir = format!("simulations/{model_variant}");
        let pred_subdir = format!("abm_predictions/{model_variant}");
        save_all_simulations(
            &folder,
            T,
            n_sims,
            model_constructor,
            Some(&sim_subdir),
            &mut rng,
        )?;
        save_all_predictions_from_sims(&folder, &calibration.data, &sim_subdir, &pred_subdir)?;
    }
    let (data, ea) = (&calibration.data, &calibration.ea);

    // Error tables.
    eprintln!("Computing error tables...");

    let prediction_folder = format!("abm_predictions/{model_variant}");

    error_table_ar(country, ea, data, &quarters, &horizons, model_variant)?;
    eprintln!("✓ AR error table");

    error_table_abm(
        country,
        ea,
        data,
        &quarters,
        &horizons,
        model_variant,
        &prediction_folder,
    )?;
    eprintln!("✓ ABM error table");

    error_table_validation_var(country, ea, data, &quarters, &horizons, model_variant)?;
    eprintln!("✓ VAR validation table");

    error_table_validation_abm(
        country,
        ea,
        data,
        &quarters,
        &horizons,
        model_variant,
        &prediction_folder,
    )?;
    eprintln!("✓ ABM validation table");

    eprintln!("Done. Results saved to data/{country}/analysis/{model_variant}/");
    Ok(())
}
