//! Forecast error tables of several countries, BeforeIT
//! `examples/analysis/tabs/simulate_and_create_tables_multicountry.jl`: the
//! prediction pipeline of the CANVAS model for Italy, Austria and Spain
//! from 2010Q1 to 2011Q1 (4 runs of 12 quarters from each quarter), then
//! the error tables of each country.
//!
//! ```sh
//! BEFOREIT_CALIBRATION_DIR=<folder> cargo run --release --example simulate_and_create_tables_multicountry
//! ```
//!
//! Julia downloads the calibration object of each country from Zenodo; the
//! port reads neither HTTP nor JLD2, so the example reads `IT.json`,
//! `AT.json` and `ES.json` in the folder `BEFOREIT_CALIBRATION_DIR`,
//! converted from the JLD2 files of the record by
//! `data/convert_calibration.py`. As in Julia, the files go to the folders
//! `data/<country>` of the working directory, which the example sets to
//! `target/example-output`; **a country's folder is deleted first if it
//! exists but has no folder `parameters`**.

#![allow(non_snake_case)]

#[path = "../../common/mod.rs"]
mod common;

use std::path::Path;

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

    // The countries (two-letter codes of `AVAILABLE_COUNTRIES`), each with
    // its calibration object from Zenodo.
    let countries = ["IT", "AT", "ES"];

    let T = 12; // forecast horizon (quarters)
    let n_sims = 4; // number of simulations per quarter

    let quarters = Date::every_months(Date::new(2010, 3, 31), Date::new(2011, 3, 31), 3);

    // The model: `Model::new`, `ModelGR::new` or `ModelCANVAS::new`, and the
    // name of its variant, which names the output folders
    // `data/<country>/analysis/<variant>/` (Julia derives it from the type).
    let model_constructor = ModelCANVAS::new;
    let model_variant = "ModelCANVAS";

    // Simulations.
    // `save_all_params_and_initial_conditions` makes `data/<country>` anew
    // and DELETES it first if it exists.

    eprintln!(
        "Starting simulations (variant: {model_variant}) for: {}",
        countries.join(", ")
    );

    for country in countries {
        eprintln!("Processing {country}");

        let mut calibration = common::zenodo_calibration_object(country)?;
        let folder = format!("data/{country}");
        let sim_subdir = format!("simulations/{model_variant}");
        let pred_subdir = format!("abm_predictions/{model_variant}");

        // Parameters and initial conditions do not depend on the model: they
        // are made once per country, so that a later run with another
        // model does not wipe the outputs of the previous variant
        // (`save_all_params_and_initial_conditions` deletes the whole folder
        // of the country if it exists).
        if !Path::new(&folder).join("parameters").is_dir() {
            save_all_params_and_initial_conditions(
                &calibration,
                &folder,
                0.0005,
                quarters[0],
                quarters[quarters.len() - 1],
            )?;
            // Julia's calibration also stores the GDP deflators in the data
            // it is given, which the predictions below use.
            calibration.set_gdp_deflators()?;
        }
        save_all_simulations(
            &folder,
            T,
            n_sims,
            model_constructor,
            Some(&sim_subdir),
            &mut rng,
        )?;
        save_all_predictions_from_sims(&folder, &calibration.data, &sim_subdir, &pred_subdir)?;
        eprintln!("Completed {country}");
    }

    // Analysis: each country's calibration object anew, as Julia downloads
    // it again (without the GDP deflators that its calibration stored).

    eprintln!("Generating error tables (variant: {model_variant})...");

    for country in countries {
        eprintln!("Generating tables for {country}");

        let calibration = common::zenodo_calibration_object(country)?;
        let dir = Path::new("data")
            .join(country)
            .join("analysis")
            .join(model_variant);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let (data, ea) = (&calibration.data, &calibration.ea);
        let prediction_folder = format!("abm_predictions/{model_variant}");

        let h = &FORECAST_HORIZONS;
        error_table_ar(country, ea, data, &quarters, h, model_variant)?;
        error_table_validation_var(country, ea, data, &quarters, h, model_variant)?;
        error_table_abm(
            country,
            ea,
            data,
            &quarters,
            h,
            model_variant,
            &prediction_folder,
        )?;
        error_table_validation_abm(
            country,
            ea,
            data,
            &quarters,
            h,
            model_variant,
            &prediction_folder,
        )?;

        eprintln!("Completed {country}");
    }

    eprintln!("Done.");
    Ok(())
}
