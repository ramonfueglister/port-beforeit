//! Cross-correlations and autocorrelations of one country, BeforeIT
//! `examples/analysis/figs/crosscorrelations.jl`: predictions of the model
//! for Austria from each quarter of 2010Q1 to 2012Q4 (4 runs of 20 quarters
//! each; one file of predictions per quarter, as in
//! `prediction_pipeline_multiple`), and the business-cycle
//! statistics of the predictions and of the data: the cross-correlations
//! with real GDP and the autocorrelations of four series.
//!
//! ```sh
//! BEFOREIT_CALIBRATION_DIR=<folder> cargo run --release --features plots --example crosscorrelations
//! ```
//!
//! Julia downloads the calibration object of Austria from Zenodo; the port
//! reads neither HTTP nor JLD2, so the example reads `AT.json` in the folder
//! `BEFOREIT_CALIBRATION_DIR`, converted from `AT_calibration_object.jld2`
//! of the record by `data/convert_calibration.py`. As in Julia, the files go
//! to the folder `data/AT` of the working directory, which the example sets
//! to `target/example-output`; **the folder is deleted first if it exists**
//! (`prediction_from_scratch`). The plots are SVG files in `analysis/figs/AT`
//! instead of PNG.

#![allow(non_snake_case)]

#[path = "../../common/mod.rs"]
mod common;

use std::path::Path;

use beforeit::analysis::{
    save_all_params_and_initial_conditions, save_all_predictions_from_sims, save_all_simulations,
};
use beforeit::calibration::table_from_json;
use beforeit::correlation::{CorrelationStats, correlation_stats, correlation_stats_folder};
use beforeit::plots::{Color, LineStyle, Plot, layout_svg_with_title};
use beforeit::{Date, Model, Xoshiro};

/// The mean of the ABM's samples with a band of one standard deviation, and
/// the data (if `real` has the variable), at the lags, Julia's
/// `plot!(p, subplot = k, ...)`.
fn subplot(var: &str, lags: &[f64], abm: (Vec<f64>, Vec<f64>), real: Option<Vec<f64>>) -> Plot {
    let mut p = Plot::new(var.replace("_quarterly", ""));
    let s = p.add_line(lags.to_vec(), abm.0);
    s.ribbon = Some(abm.1);
    s.label = Some("ABM".to_string());
    s.color = Color::STEELBLUE;
    s.line_width = 2.0;
    if let Some(real) = real {
        let s = p.add_line(lags.to_vec(), real);
        s.label = Some("Real".to_string());
        s.color = Color::CRIMSON;
        s.line_width = 2.0;
    }
    p.add_hline(0.0, Color::GRAY, LineStyle::Dot);
    p
}

fn main() -> Result<(), String> {
    common::enter_work_dir()?;
    let mut rng = Xoshiro::seed_from_u64(1);

    // Configuration.

    // The country: a two-letter code of `AVAILABLE_COUNTRIES`, its
    // calibration object from Zenodo.
    let country = "AT";

    let correlation_lags = 15;
    let autocorr_lags = 20;

    let plot_variables = [
        "real_capitalformation_quarterly",
        "wages_quarterly",
        "real_household_consumption_quarterly",
        "gdp_deflator_quarterly",
    ];

    // The ensembles: predictions in `data/<country>/abm_predictions/YYYYQn.json`.
    let folder = format!("data/{country}");
    let first_calibration_date = Date::new(2010, 3, 31);
    let last_calibration_date = Date::new(2012, 12, 31);
    let T = 20; // quarters simulated per prediction
    let n_sims = 4; // simulations per quarter

    // The model: `Model::new`, `ModelGR::new` or `ModelCANVAS::new`.
    let model_constructor = Model::new;

    // The prediction files, as in `prediction_pipeline_multiple`: the
    // calibration on each quarter, `n_sims` runs of `T` quarters, aligned
    // with the data, one file per quarter.
    let mut calibration = common::zenodo_calibration_object(country)?;
    let prediction_folder = format!("{folder}/abm_predictions");

    // `true` makes the prediction files anew; with `false` the example uses
    // those in `prediction_folder`.
    let prediction_from_scratch = true;

    if prediction_from_scratch {
        eprintln!("Generating prediction files in {prediction_folder}");
        save_all_params_and_initial_conditions(
            &calibration,
            &folder,
            0.0005,
            first_calibration_date,
            last_calibration_date,
        )?;
        // Julia's calibration also stores the GDP deflators in the data it
        // is given, which the predictions and statistics below use.
        calibration.set_gdp_deflators()?;
        save_all_simulations(&folder, T, n_sims, model_constructor, None, &mut rng)?;
        save_all_predictions_from_sims(
            &folder,
            &calibration.data,
            "simulations",
            "abm_predictions",
        )?;
    }
    let real_data = &calibration.data;

    // The statistics.

    // The quarterly variables of the predictions, from the first file.
    let mut files: Vec<String> = std::fs::read_dir(&prediction_folder)
        .map_err(|e| format!("{prediction_folder}: {e}"))?
        .filter_map(|e| e.ok().and_then(|e| e.file_name().into_string().ok()))
        .collect();
    files.sort();
    let first = files
        .first()
        .ok_or_else(|| format!("{prediction_folder} is empty"))?;
    let path = Path::new(&prediction_folder).join(first);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let first_pred = table_from_json(&text)?;
    let vars: Vec<&str> = first_pred
        .keys()
        .map(String::as_str)
        .filter(|v| v.ends_with("_quarterly"))
        .collect();

    // The statistics of the predictions and of the data.
    let gdp = "real_gdp_quarterly";
    let abm_stats = correlation_stats_folder(
        Path::new(&prediction_folder),
        &vars,
        gdp,
        correlation_lags,
        autocorr_lags,
    )?;
    let real_stats = correlation_stats(real_data, &vars, gdp, correlation_lags, autocorr_lags)?;

    // The plots.

    let output_folder = format!("analysis/figs/{country}");
    std::fs::create_dir_all(&output_folder).map_err(|e| format!("{output_folder}: {e}"))?;
    let output_folder = Path::new(&output_folder);

    // Cross-correlations; a panel stays empty if a statistic is missing.
    let cross_lags: Vec<f64> = (-(correlation_lags as i64)..=correlation_lags as i64)
        .map(|l| l as f64)
        .collect();
    let has = |s: &CorrelationStats, var: &str| s.crosscor.contains_key(var);
    let mut p1 = Vec::new();
    for var in plot_variables.iter().take(4) {
        if has(&abm_stats, var) && has(&real_stats, var) {
            p1.push(subplot(
                var,
                &cross_lags,
                (abm_stats.mean_crosscor(var)?, abm_stats.std_crosscor(var)?),
                Some(real_stats.mean_crosscor(var)?),
            ));
        } else {
            p1.push(Plot::new(""));
        }
    }
    common::save(
        output_folder,
        "crosscorrelations_abm.svg",
        &layout_svg_with_title(&p1, (2, 2), (1200, 800), "Cross-Correlations with Real GDP")?,
    )?;
    eprintln!("✓ Saved cross-correlation plot");

    // Autocorrelations.
    let auto_lags: Vec<f64> = (0..=autocorr_lags).map(|l| l as f64).collect();
    let mut p2 = Vec::new();
    for var in plot_variables.iter().take(4) {
        if abm_stats.autocor.contains_key(*var) {
            let real = if real_stats.autocor.contains_key(*var) {
                Some(real_stats.mean_autocor(var)?)
            } else {
                None
            };
            p2.push(subplot(
                var,
                &auto_lags,
                (abm_stats.mean_autocor(var)?, abm_stats.std_autocor(var)?),
                real,
            ));
        } else {
            p2.push(Plot::new(""));
        }
    }
    common::save(
        output_folder,
        "autocorrelations_abm.svg",
        &layout_svg_with_title(&p2, (2, 2), (1200, 800), "Autocorrelations")?,
    )?;
    eprintln!("✓ Saved autocorrelation plot");
    Ok(())
}
