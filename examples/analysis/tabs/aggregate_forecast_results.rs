//! Forecast error tables across countries, BeforeIT
//! `examples/analysis/tabs/aggregate_forecast_results.jl`: the tables of
//! every country that has predictions of a model variant, gathered in CSV
//! tables of countries × variables at the horizon of 12 quarters (with the
//! stars of the tests) and in LaTeX tables of horizons × variables with the
//! mean over the countries and its standard error.
//!
//! ```sh
//! cargo run --release --example tabs_aggregate_forecast_results
//! ```
//!
//! As in Julia, the example reads `data/<country>/analysis/<variant>/` and
//! writes `analysis/tabs/forecast_performance/<variant>/` in the working
//! directory, which it sets to `target/example-output`, where
//! `simulate_and_create_tables` and `simulate_and_create_tables_multicountry`
//! write their tables. A country counts if it has predictions in
//! `data/<country>/abm_predictions/<variant>/`.

#![allow(non_snake_case)]

#[path = "../../common/mod.rs"]
mod common;

use std::path::Path;

use beforeit::analysis::{
    AGGREGATE_TABLE_TYPES, CsvColumn, FORECAST_HORIZONS, build_12q_matrices, csv_text,
    discover_countries_of_variant, load_country_forecast_data, stars,
};
use beforeit::julia::{float_string, round_digits};

/// Statistics.jl's `mean` and `std` (with n − 1, NaN for one value).
fn mean_std(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0);
    (m, v.sqrt())
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> Result<(), String> {
    common::enter_work_dir()?;

    // The model variant whose results to gather: "Model", "ModelGR" or
    // "ModelCANVAS".
    let model_variant = "Model";

    eprintln!("Aggregating forecast results for variant: {model_variant}...");

    let countries = discover_countries_of_variant(model_variant, Path::new("data"))?;
    eprintln!("Found {} countries", countries.len());

    let output_base = Path::new("analysis")
        .join("tabs")
        .join("forecast_performance")
        .join(model_variant);
    for sub in ["countries", "latex"] {
        let dir = output_base.join(sub);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }

    for table_type in AGGREGATE_TABLE_TYPES {
        let Some(result) = load_country_forecast_data(model_variant, table_type, &countries) else {
            eprintln!("Warning: No data for {table_type}");
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
        let is_bias = table_type.starts_with("bias_");

        // CSV of countries × variables (12 quarters, with stars), the
        // column `Country` first, as CSV.jl writes the `DataFrame`.
        let mut text = std::iter::once("Country")
            .chain(variables.iter().map(String::as_str))
            .map(csv_text)
            .collect::<Vec<_>>()
            .join(",");
        text.push('\n');
        for i in 0..n_countries {
            let mut cells = vec![csv_text(&valid_countries[i])];
            for j in 0..n_variables {
                let x = matrix[i][j];
                cells.push(if result.has_pvals {
                    let digits = if is_bias { 4 } else { 1 };
                    if x.is_nan() {
                        csv_text("")
                    } else {
                        let val_str = float_string(round_digits(x, digits));
                        let p = pval_matrix[i][j];
                        let star_str = if p.is_nan() { "" } else { stars(p) };
                        csv_text(&(val_str + star_str))
                    }
                } else {
                    float_string(x)
                });
            }
            text.push_str(&cells.join(","));
            text.push('\n');
        }
        write(
            &output_base
                .join("countries")
                .join(format!("{table_type}_countries.csv")),
            &text,
        )?;

        // LaTeX table of horizons × variables: the mean over the countries
        // and its standard error.
        let n_horizons = FORECAST_HORIZONS.len();
        let mut mean_vals = vec![vec![f64::NAN; n_variables]; n_horizons];
        let mut std_errs = mean_vals.clone();
        for (i, h) in FORECAST_HORIZONS.iter().enumerate() {
            for (j, var) in variables.iter().enumerate() {
                let mut vals = Vec::new();
                for country in valid_countries {
                    let df = &result.country_data[country];
                    let Some(column) = df.column(var) else {
                        continue;
                    };
                    // Julia's `df.Horizon .== "12q"` throws without the
                    // column and matches no row of a column of numbers.
                    let horizon = match df.column("Horizon") {
                        Some(CsvColumn::Text(horizon)) => horizon,
                        Some(CsvColumn::Numbers(_)) => continue,
                        None => {
                            return Err(format!("{country}: {table_type} has no column Horizon"));
                        }
                    };
                    let label = format!("{h}q");
                    let rows: Vec<usize> = (0..horizon.len())
                        .filter(|&r| horizon[r].as_deref() == Some(label.as_str()))
                        .collect();
                    if rows.len() != 1 {
                        continue;
                    }
                    match column {
                        CsvColumn::Numbers(v) => {
                            if let Some(x) = v[rows[0]].filter(|x| !x.is_nan()) {
                                vals.push(x);
                            }
                        }
                        CsvColumn::Text(v) if v[rows[0]].is_some() => {
                            return Err(format!(
                                "{country}: the column {var} of {table_type} holds text \
                                 (Julia throws a MethodError for isnan)"
                            ));
                        }
                        CsvColumn::Text(_) => {}
                    }
                }
                if !vals.is_empty() {
                    let (m, s) = mean_std(&vals);
                    mean_vals[i][j] = m;
                    std_errs[i][j] = s / (vals.len() as f64).sqrt();
                }
            }
        }

        let mut tex = format!("% Cross-country aggregated table for {table_type}\n");
        for (i, h) in FORECAST_HORIZONS.iter().enumerate() {
            let mut row = vec![format!("{h}q")];
            for j in 0..n_variables {
                if !mean_vals[i][j].is_nan() && !std_errs[i][j].is_nan() {
                    // `@sprintf("%.1f(%.1f)", ...)`.
                    row.push(format!("{:.1}({:.1})", mean_vals[i][j], std_errs[i][j]));
                } else {
                    row.push("N/A".to_string());
                }
            }
            tex.push_str(&row.join(" & "));
            tex.push_str(if i + 1 < n_horizons { " \\\\ " } else { "" });
            tex.push('\n');
        }
        write(
            &output_base
                .join("latex")
                .join(format!("cross_country_{table_type}.tex")),
            &tex,
        )?;

        eprintln!("✓ {table_type}: {n_countries} countries");
    }

    eprintln!("Done.");
    Ok(())
}
