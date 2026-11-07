//! Heatmaps of the forecast errors across countries, BeforeIT
//! `examples/analysis/figs/aggregate_forecast_results.jl`: for each kind of
//! table, the values of every country (rows) and variable (columns) at the
//! horizon of 12 quarters, coloured and written in their cells with the
//! stars of the tests.
//!
//! ```sh
//! cargo run --release --features plots --example figs_aggregate_forecast_results
//! ```
//!
//! As in Julia, the example reads `data/<country>/analysis/<variant>/` and
//! writes `analysis/figs/forecast_performance/<variant>/` in the working
//! directory, which it sets to `target/example-output`, where
//! `simulate_and_create_tables` and `simulate_and_create_tables_multicountry`
//! write their tables. The heatmaps are SVG files instead of PNG; their
//! margins are those of `plotters` instead of 15 mm.

#![allow(non_snake_case)]

#[path = "../../common/mod.rs"]
mod common;

use std::path::Path;

use beforeit::analysis::{
    AGGREGATE_TABLE_TYPES, build_12q_matrices, discover_countries_of_variant,
    load_country_forecast_data, stars,
};
use beforeit::julia::{float_string, round_digits};
use beforeit::plots::{Color, Gradient, Plot};

fn main() -> Result<(), String> {
    common::enter_work_dir()?;

    // The model variant whose results to draw: "Model", "ModelGR" or
    // "ModelCANVAS".
    let model_variant = "Model";

    eprintln!("Creating forecast heatmaps for variant: {model_variant}...");

    let countries = discover_countries_of_variant(model_variant, Path::new("data"))?;
    eprintln!("Found {} countries", countries.len());

    let output_dir = Path::new("analysis")
        .join("figs")
        .join("forecast_performance")
        .join(model_variant);
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("{}: {e}", output_dir.display()))?;

    for table_type in AGGREGATE_TABLE_TYPES {
        let Some(result) = load_country_forecast_data(model_variant, table_type, &countries) else {
            continue;
        };
        let (matrix, pval_matrix) = build_12q_matrices(&result, "12q")?;
        if matrix.iter().flatten().all(|x| x.is_nan()) {
            continue;
        }

        let valid_countries = &result.valid_countries;
        let variables = &result.variables;
        let n_countries = valid_countries.len();
        let n_variables = variables.len();
        let valid_vals: Vec<f64> = matrix
            .iter()
            .flatten()
            .copied()
            .filter(|x| !x.is_nan())
            .collect();

        let is_bias = table_type.contains("bias");
        let is_raw_rmse = [
            "rmse_abm",
            "rmse_ar",
            "rmse_validation_abm",
            "rmse_validation_var",
        ]
        .contains(&table_type);
        let is_diverging = is_bias || !is_raw_rmse;

        // The bounds of the colour scale.
        let max_abs = valid_vals
            .iter()
            .map(|x| x.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        let val_min = valid_vals.iter().copied().fold(f64::INFINITY, f64::min);
        let val_max = valid_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let val_range = val_max - val_min;

        let clean_vars: Vec<String> = variables.iter().map(|v| v.replace(' ', "\n")).collect();

        let mut p = if is_diverging {
            let mut p = Plot::heatmap(matrix.clone(), Gradient::RdBu);
            if let Some(h) = &mut p.heatmap {
                h.clims = Some((-max_abs, max_abs));
            }
            p
        } else {
            Plot::heatmap(matrix.clone(), Gradient::YlOrRd)
        };
        p.x_ticks = Some(((1..=n_variables).map(|j| j as f64).collect(), clean_vars));
        p.y_ticks = Some((
            (1..=n_countries).map(|i| i as f64).collect(),
            valid_countries.clone(),
        ));
        p.x_rotation = 45.0;
        let size = (
            600.max(n_variables as u32 * 120),
            400.max(n_countries as u32 * 40),
        );

        for i in 0..n_countries {
            for j in 0..n_variables {
                let val = matrix[i][j];
                if val.is_nan() {
                    continue;
                }
                let mut txt = float_string(round_digits(val, if is_bias { 4 } else { 1 }));
                if result.has_pvals && !pval_matrix[i][j].is_nan() {
                    txt.push_str(stars(pval_matrix[i][j]));
                }
                let white = if is_diverging {
                    val.abs() / max_abs > 0.65
                } else {
                    let normalized = if val_range > 0.0 {
                        (val - val_min) / val_range
                    } else {
                        0.0
                    };
                    normalized > 0.7
                };
                let txt_color = if white { Color::WHITE } else { Color::BLACK };
                p.annotate((j + 1) as f64, (i + 1) as f64, txt, 6, txt_color);
            }
        }

        common::save(
            &output_dir,
            &format!("{table_type}_heatmap.svg"),
            &p.to_svg(size)?,
        )?;
        eprintln!("✓ {table_type}");
    }

    eprintln!("Done.");
    Ok(())
}
