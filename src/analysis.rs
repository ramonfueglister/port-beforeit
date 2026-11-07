//! Forecast error tables and the batch runs that produce their inputs,
//! BeforeIT `src/utils/analysis_utils.jl`, `src/utils/error_tables.jl` and
//! `src/utils/save_all_predictions.jl`.
//!
//! The batch runs compute parameters and initial conditions for a range of
//! quarters ([`save_all_params_and_initial_conditions`]), simulate each
//! quarter ([`save_all_simulations`]) and turn the simulations into
//! forecasts ([`save_all_predictions_from_sims`]). The error tables compare
//! these forecasts, and those of AR and VAR benchmarks, with the data: root
//! mean squared errors, mean errors (bias) with a t-test, and
//! Diebold-Mariano tests of one forecast against another.
//!
//! As in the Julia code, the functions read and write files at paths
//! relative to the working directory, such as
//! `data/<country>/analysis/<variant>/rmse_abm.csv`. JSON files take the
//! place of Julia's JLD2 files, with the same names and `.json` in place of
//! `.jld2`: parameters and initial conditions as written by
//! [`Parameters::to_json`] and [`InitialConditions::to_json`], simulations
//! as [`DataVector::to_json`], predictions as the table of
//! [`Predictions::to_table`](crate::Predictions::to_table) in
//! [`table_to_json`], and the arrays of forecasts as a table with the key
//! `forecast`. The CSV and LaTeX tables are written as by the Julia code,
//! numbers as Julia prints them ([`float_string`]). What the Julia code logs
//! with `@warn` is printed to standard error, what it prints with `println`
//! to standard output.

use std::collections::BTreeMap;
use std::ops::{Index, IndexMut};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use crate::calibration::{
    Array, CalibrationData, Date, Table, get_params_and_initial_conditions, table_from_json,
    table_to_json,
};
use crate::correlation::{extract_yq, read_dir_names};
use crate::data::DataVector;
use crate::evaluation::{bias_ttest, dmtest_modified, mean};
use crate::julia::{float_string, round_digits};
use crate::params::{InitialConditions, Parameters};
use crate::predictions::get_predictions_from_sims;
use crate::random::Randomness;
use crate::shocks::{NoShock, ensemblerun};
use crate::varx::forecast_k_steps_VAR;
use crate::{AbstractModel, Extension};

/// A matrix given by its rows. In the error tables, one row per forecast
/// horizon and one column per variable.
pub type Matrix = Vec<Vec<f64>>;

/// A three-dimensional array in Julia's column-major order, such as the
/// forecasts of the error tables, `quarters × horizons × variables`.
/// `a[(i, j, l)]` is Julia's `a[i + 1, j + 1, l + 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Array3 {
    pub size: [usize; 3],
    pub data: Vec<f64>,
}

impl Array3 {
    /// Julia `fill(NaN, size...)`.
    pub fn nan(size: [usize; 3]) -> Array3 {
        Array3 {
            size,
            data: vec![f64::NAN; size.iter().product()],
        }
    }

    fn offset(&self, (i, j, l): (usize, usize, usize)) -> usize {
        let [n1, n2, n3] = self.size;
        assert!(
            i < n1 && j < n2 && l < n3,
            "index ({i}, {j}, {l}) outside an array of size {:?}",
            self.size
        );
        i + n1 * (j + n2 * l)
    }

    /// Julia `a[:, j + 1, l + 1]`.
    fn column(&self, j: usize, l: usize) -> Result<&[f64], String> {
        let [n1, n2, n3] = self.size;
        if j >= n2 || l >= n3 {
            return Err(format!(
                "column ({j}, {l}) outside an array of size {:?} (Julia throws a BoundsError)",
                self.size
            ));
        }
        let start = n1 * (j + n2 * l);
        Ok(&self.data[start..start + n1])
    }

    fn to_array(&self) -> Array {
        Array {
            size: self.size.to_vec(),
            data: self.data.iter().map(|&x| Some(x)).collect(),
        }
    }

    fn from_array(a: &Array, what: &str) -> Result<Array3, String> {
        let size: [usize; 3] = a
            .size
            .as_slice()
            .try_into()
            .map_err(|_| format!("{what} is not a three-dimensional array"))?;
        let data = a
            .data
            .iter()
            .map(|x| x.ok_or_else(|| format!("{what} has a missing entry")))
            .collect::<Result<_, _>>()?;
        Ok(Array3 { size, data })
    }
}

impl Index<(usize, usize, usize)> for Array3 {
    type Output = f64;

    fn index(&self, i: (usize, usize, usize)) -> &f64 {
        &self.data[self.offset(i)]
    }
}

impl IndexMut<(usize, usize, usize)> for Array3 {
    fn index_mut(&mut self, i: (usize, usize, usize)) -> &mut f64 {
        let k = self.offset(i);
        &mut self.data[k]
    }
}

// =============================================================================
// Constants
// =============================================================================

/// `agg_base_variables`: the variables of the error tables of the ABM and
/// the AR benchmarks.
pub const AGG_BASE_VARIABLES: [&str; 5] = [
    "Real GDP",
    "GDP Deflator Growth",
    "Real Consumption",
    "Real Investment",
    "Euribor",
];

/// `agg_validation_variables`: the variables of the validation tables of
/// the ABM and the VAR benchmarks.
pub const AGG_VALIDATION_VARIABLES: [&str; 8] = [
    "Real GDP",
    "GDP Deflator Growth",
    "Real Gov Consumption",
    "Real Exports",
    "Real Imports",
    "Real GDP (EA)",
    "GDP Deflator Growth (EA)",
    "Euribor",
];

/// `forecast_horizons`: the horizons of the error tables, in quarters.
pub const FORECAST_HORIZONS: [usize; 5] = [1, 2, 4, 8, 12];

/// `aggregate_table_types`: the tables of each country that BeforeIT's
/// examples gather across countries.
pub const AGGREGATE_TABLE_TYPES: [&str; 12] = [
    "rmse_abm",
    "bias_abm",
    "rmse_validation_abm",
    "bias_validation_abm",
    "rmse_abm_vs_ar",
    "rmse_ar",
    "bias_ar",
    "rmse_validation_abm_vs_var",
    "rmse_validation_var",
    "bias_validation_var",
    "rmse_abm_vs_base",
    "rmse_validation_abm_vs_base",
];

/// `get_variables_for_table(t)`: [`AGG_VALIDATION_VARIABLES`] if `t`
/// contains "validation", otherwise [`AGG_BASE_VARIABLES`].
pub fn get_variables_for_table(t: &str) -> &'static [&'static str] {
    if t.contains("validation") {
        &AGG_VALIDATION_VARIABLES
    } else {
        &AGG_BASE_VARIABLES
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// `stars(p_value)`: "***" below 0.01, "**" below 0.05, "*" below 0.1,
/// otherwise (also for NaN) "".
pub fn stars(p_value: f64) -> &'static str {
    if p_value < 0.01 {
        "***"
    } else if p_value < 0.05 {
        "**"
    } else if p_value < 0.1 {
        "*"
    } else {
        ""
    }
}

/// `nanmean(x)`: the mean of the numbers that are not NaN (NaN if there
/// are none).
pub fn nanmean(x: &[f64]) -> f64 {
    let kept: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    mean(&kept)
}

fn same_shape(a: &Matrix, b: &Matrix) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(r, s)| r.len() == s.len())
}

/// `calculate_forecast_errors(forecast, actual)`: the errors `forecast −
/// actual`, and for each horizon and variable their root mean squared
/// error in percent, `100 sqrt(nanmean(error²))`, and their mean (bias),
/// both over the quarters without NaN. Returns `(rmse, bias, error)`.
///
/// Returns an error if the arrays have different sizes (Julia throws a
/// `DimensionMismatch`).
pub fn calculate_forecast_errors(
    forecast: &Array3,
    actual: &Array3,
) -> Result<(Matrix, Matrix, Array3), String> {
    if forecast.size != actual.size {
        return Err(format!(
            "calculate_forecast_errors: arrays of sizes {:?} and {:?}",
            forecast.size, actual.size
        ));
    }
    let error = Array3 {
        size: forecast.size,
        data: forecast
            .data
            .iter()
            .zip(&actual.data)
            .map(|(f, a)| f - a)
            .collect(),
    };
    let [_, nh, nv] = error.size;
    let mut rmse = vec![vec![0.0; nv]; nh];
    let mut bias = vec![vec![0.0; nv]; nh];
    for j in 0..nh {
        for l in 0..nv {
            let e = error.column(j, l)?;
            let squares: Vec<f64> = e.iter().map(|x| x * x).collect();
            rmse[j][l] = 100.0 * nanmean(&squares).sqrt();
            bias[j][l] = nanmean(e);
        }
    }
    Ok((rmse, bias, error))
}

/// `rmse_improvement(rmse1, rmse2)`: `−round(100 (rmse1 − rmse2) / rmse2;
/// digits = 1)`, the reduction of the error in percent, positive where
/// `rmse1` is smaller.
///
/// Returns an error if the matrices have different shapes (Julia
/// broadcasts dimensions of length 1 and throws otherwise).
pub fn rmse_improvement(rmse1: &Matrix, rmse2: &Matrix) -> Result<Matrix, String> {
    if !same_shape(rmse1, rmse2) {
        return Err("rmse_improvement: the matrices have different shapes".into());
    }
    Ok(rmse1
        .iter()
        .zip(rmse2)
        .map(|(r1, r2)| {
            r1.iter()
                .zip(r2)
                .map(|(a, b)| -round_digits(100.0 * (a - b) / b, 1))
                .collect()
        })
        .collect())
}

fn round_matrix(m: &Matrix, digits: u32) -> Matrix {
    m.iter()
        .map(|r| r.iter().map(|&x| round_digits(x, digits)).collect())
        .collect()
}

fn strings(m: &Matrix) -> Vec<Vec<String>> {
    m.iter()
        .map(|r| r.iter().map(|&x| float_string(x)).collect())
        .collect()
}

fn io<T>(path: &Path, r: std::io::Result<T>) -> Result<T, String> {
    r.map_err(|e| format!("{}: {e}", path.display()))
}

fn write_file(path: &Path, contents: &str) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        io(dir, std::fs::create_dir_all(dir))?;
    }
    io(path, std::fs::write(path, contents))
}

fn read_file(path: &Path) -> Result<String, String> {
    io(path, std::fs::read_to_string(path))
}

/// `data/<country>/analysis/<model_variant>`.
fn analysis_dir(country: &str, model_variant: &str) -> PathBuf {
    Path::new("data")
        .join(country)
        .join("analysis")
        .join(model_variant)
}

/// Julia `save(path, "forecast", forecast)`.
fn save_forecast(path: &Path, forecast: &Array3) -> Result<(), String> {
    let t: Table = [("forecast".to_string(), forecast.to_array())].into();
    write_file(path, &table_to_json(&t))
}

/// Julia `load(path)["forecast"]`.
fn load_forecast(path: &Path) -> Result<Array3, String> {
    let t = table_from_json(&read_file(path)?).map_err(|e| format!("{}: {e}", path.display()))?;
    let a = t
        .get("forecast")
        .ok_or_else(|| format!("{}: no \"forecast\"", path.display()))?;
    Array3::from_array(a, &format!("{}: \"forecast\"", path.display()))
}

// =============================================================================
// Table writing
// =============================================================================

/// `write_latex_table(filename, country, input_data_S, horizons;
/// model_variant)`: writes the rows of LaTeX table cells `input_data_S`,
/// each after the label of its horizon (`1q`, `2q`, …) and joined by
/// ` & `, every row but the last ending in ` \\ `, to
/// `data/<country>/analysis/<model_variant>/<filename>`, creating the
/// folders.
///
/// Returns an error if there are fewer horizons than rows (Julia throws a
/// `BoundsError`) or the file cannot be written.
pub fn write_latex_table(
    filename: &str,
    country: &str,
    input_data_S: &[Vec<String>],
    horizons: &[usize],
    model_variant: &str,
) -> Result<(), String> {
    let nrows = input_data_S.len();
    if horizons.len() < nrows {
        return Err(format!(
            "write_latex_table: {nrows} rows and {} horizons",
            horizons.len()
        ));
    }
    let mut text = String::new();
    for (row, cells) in input_data_S.iter().enumerate() {
        let mut line = vec![format!("{}q", horizons[row])];
        line.extend(cells.iter().cloned());
        text.push_str(&line.join(" & "));
        if row + 1 < nrows {
            text.push_str(" \\\\ ");
        }
        text.push('\n');
    }
    write_file(&analysis_dir(country, model_variant).join(filename), &text)
}

/// A text cell as CSV.jl writes it: in quotes, with quotes doubled, if it
/// is empty, begins or ends with a space, or contains a comma, a quote or
/// a line break. Numbers are written as Julia prints them
/// ([`float_string`]).
pub fn csv_text(s: &str) -> String {
    let quote = s.is_empty()
        || s.starts_with(' ')
        || s.ends_with(' ')
        || s.contains([',', '"', '\n', '\r']);
    if quote {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// `write_csv_table(filename, country, input_data, horizons,
/// variable_names; model_variant)`: writes a CSV table with the column
/// `Horizon` (`1q`, `2q`, …) and a column per variable from the columns of
/// `input_data` to `data/<country>/analysis/<model_variant>/<filename>`,
/// creating the folders, as CSV.jl writes a `DataFrame`: comma-separated,
/// a header line, numbers as Julia prints them, NaN as `NaN`.
///
/// Returns an error if `input_data` does not have a row per horizon and a
/// column per variable (Julia throws), or the file cannot be written.
pub fn write_csv_table(
    filename: &str,
    country: &str,
    input_data: &Matrix,
    horizons: &[usize],
    variable_names: &[&str],
    model_variant: &str,
) -> Result<(), String> {
    if input_data.len() != horizons.len()
        || input_data.iter().any(|r| r.len() < variable_names.len())
    {
        return Err(format!(
            "write_csv_table: {} rows for {} horizons, or fewer columns than {} variables",
            input_data.len(),
            horizons.len(),
            variable_names.len()
        ));
    }
    let mut header = vec![csv_text("Horizon")];
    header.extend(variable_names.iter().map(|v| csv_text(v)));
    let mut text = header.join(",") + "\n";
    for (h, row) in horizons.iter().zip(input_data) {
        let mut cells = vec![csv_text(&format!("{h}q"))];
        cells.extend(row[..variable_names.len()].iter().map(|&x| float_string(x)));
        text.push_str(&cells.join(","));
        text.push('\n');
    }
    write_file(&analysis_dir(country, model_variant).join(filename), &text)
}

// =============================================================================
// Statistical test wrappers
// =============================================================================

/// `generate_dm_test_comparison(error1, error2, rmse1, rmse2, horizons)`:
/// for each horizon and variable, [`rmse_improvement`] of `rmse1` over
/// `rmse2` and the p-value of the Diebold-Mariano test
/// ([`dmtest_modified`]`(e2, e1, h)`) on the quarters where both errors are
/// observed (NaN with fewer than two). Returns the LaTeX cells
/// `improvement(p, stars)`, with `p` rounded to 2 digits, and the
/// p-values.
///
/// Returns an error where Julia throws: if the shapes do not fit or the
/// test fails (more quarters of horizon than observations).
pub fn generate_dm_test_comparison(
    error1: &Array3,
    error2: &Array3,
    rmse1: &Matrix,
    rmse2: &Matrix,
    horizons: &[usize],
) -> Result<(Vec<Vec<String>>, Matrix), String> {
    let input_data = rmse_improvement(rmse1, rmse2)?;
    let ncols = input_data.first().map_or(0, |r| r.len());
    if horizons.len() > input_data.len() {
        return Err("generate_dm_test_comparison: more horizons than rows".into());
    }
    let mut input_data_S = vec![vec![String::new(); ncols]; input_data.len()];
    let mut pval_matrix = vec![vec![f64::NAN; ncols]; input_data.len()];
    for (j, &h) in horizons.iter().enumerate() {
        for l in 0..ncols {
            let e1 = error1.column(j, l)?;
            let e2 = error2.column(j, l)?;
            if e1.len() != e2.len() {
                return Err(
                    "generate_dm_test_comparison: errors of different lengths (Julia throws a DimensionMismatch)"
                        .into(),
                );
            }
            let (a, b): (Vec<f64>, Vec<f64>) = e1
                .iter()
                .zip(e2)
                .filter(|(x, y)| !x.is_nan() && !y.is_nan())
                .unzip();
            let p_value = if a.len() < 2 {
                f64::NAN
            } else {
                dmtest_modified(&b, &a, h as i64)?.1
            };
            input_data_S[j][l] = format!(
                "{}({}, {})",
                float_string(input_data[j][l]),
                float_string(round_digits(p_value, 2)),
                stars(p_value)
            );
            pval_matrix[j][l] = p_value;
        }
    }
    Ok((input_data_S, pval_matrix))
}

/// `generate_bias_ttest(error, bias, horizons)`: for each horizon and
/// variable, the bias rounded to 4 digits and the p-value of
/// [`bias_ttest`] on the errors that are not NaN. Returns the LaTeX cells
/// `bias (p, stars)`, with `p` rounded to 3 digits, and the p-values.
///
/// Returns an error if the shapes do not fit (Julia throws a
/// `BoundsError`).
pub fn generate_bias_ttest(
    error: &Array3,
    bias: &Matrix,
    horizons: &[usize],
) -> Result<(Vec<Vec<String>>, Matrix), String> {
    let input_data = round_matrix(bias, 4);
    let ncols = bias.first().map_or(0, |r| r.len());
    if horizons.len() > input_data.len() {
        return Err("generate_bias_ttest: more horizons than rows".into());
    }
    let mut input_data_S = vec![vec![String::new(); ncols]; input_data.len()];
    let mut pval_matrix = vec![vec![f64::NAN; ncols]; input_data.len()];
    for (j, &h) in horizons.iter().enumerate() {
        for l in 0..ncols {
            let e: Vec<f64> = error
                .column(j, l)?
                .iter()
                .copied()
                .filter(|x| !x.is_nan())
                .collect();
            let (_, p_value) = bias_ttest(&e, h as i64);
            input_data_S[j][l] = format!(
                "{} ({}, {})",
                float_string(input_data[j][l]),
                float_string(round_digits(p_value, 3)),
                stars(p_value)
            );
            pval_matrix[j][l] = p_value;
        }
    }
    Ok((input_data_S, pval_matrix))
}

// =============================================================================
// Per-country table generation
// =============================================================================

/// `create_bias_rmse_tables_abm(forecast, actual, horizons, type,
/// variable_names, country; model_variant)`: the tables of the ABM in
/// `data/<country>/analysis/<model_variant>/`, with the prefix
/// `validation_` in the names if `forecast_type` (Julia's `type`) is
/// "validation":
///
/// - `rmse_abm` (CSV and LaTeX): the RMSE, rounded to 2 digits;
/// - `rmse_abm_vs_ar` (VAR for validation; LaTeX with the DM tests, CSV)
///   and `pval_abm_vs_ar.csv`: against the benchmark, whose forecasts are
///   read from `forecast_ar.json` (`forecast_validation_var.json`);
/// - for a variant other than "Model", `rmse_abm_vs_base` and
///   `pval_abm_vs_base.csv` against the forecasts of the base model in
///   `data/<country>/analysis/Model/forecast_abm.json`, if that file exists
///   (otherwise a warning);
/// - `bias_abm` (LaTeX with the t-tests, CSV rounded to 4 digits) and
///   `pval_bias_abm.csv`.
///
/// Returns an error where Julia throws, for example if the benchmark has
/// not been computed.
pub fn create_bias_rmse_tables_abm(
    forecast: &Array3,
    actual: &Array3,
    horizons: &[usize],
    forecast_type: &str,
    variable_names: &[&str],
    country: &str,
    model_variant: &str,
) -> Result<(), String> {
    let validation = forecast_type == "validation";
    let type_prefix = if validation { "validation_" } else { "" };
    let comparison_model = if validation { "var" } else { "ar" };

    let (rmse_abm, bias_abm, error_abm) = calculate_forecast_errors(forecast, actual)?;

    // 1. Absolute RMSE
    let rmse_abs = round_matrix(&rmse_abm, 2);
    let name = format!("rmse_{type_prefix}abm");
    write_csv_table(
        &format!("{name}.csv"),
        country,
        &rmse_abs,
        horizons,
        variable_names,
        model_variant,
    )?;
    write_latex_table(
        &format!("{name}.tex"),
        country,
        &strings(&rmse_abs),
        horizons,
        model_variant,
    )?;

    // 2. Relative to the AR/VAR benchmark
    let benchmark_file = analysis_dir(country, model_variant)
        .join(format!("forecast_{type_prefix}{comparison_model}.json"));
    let forecast_benchmark = load_forecast(&benchmark_file)?;
    let (rmse_benchmark, _, error_benchmark) =
        calculate_forecast_errors(&forecast_benchmark, actual)?;
    let (latex, pval) = generate_dm_test_comparison(
        &error_abm,
        &error_benchmark,
        &rmse_abm,
        &rmse_benchmark,
        horizons,
    )?;
    let name = format!("{type_prefix}abm_vs_{comparison_model}");
    write_latex_table(
        &format!("rmse_{name}.tex"),
        country,
        &latex,
        horizons,
        model_variant,
    )?;
    write_csv_table(
        &format!("rmse_{name}.csv"),
        country,
        &rmse_improvement(&rmse_abm, &rmse_benchmark)?,
        horizons,
        variable_names,
        model_variant,
    )?;
    write_csv_table(
        &format!("pval_{name}.csv"),
        country,
        &pval,
        horizons,
        variable_names,
        model_variant,
    )?;

    // 3. Relative to the base ABM (variants other than the base model)
    if model_variant != "Model" {
        let base_abm_file =
            analysis_dir(country, "Model").join(format!("forecast_{type_prefix}abm.json"));
        if base_abm_file.is_file() {
            let forecast_base_abm = load_forecast(&base_abm_file)?;
            let (rmse_base_abm, _, error_base_abm) =
                calculate_forecast_errors(&forecast_base_abm, actual)?;
            let (latex, pval) = generate_dm_test_comparison(
                &error_abm,
                &error_base_abm,
                &rmse_abm,
                &rmse_base_abm,
                horizons,
            )?;
            let name = format!("{type_prefix}abm_vs_base");
            write_latex_table(
                &format!("rmse_{name}.tex"),
                country,
                &latex,
                horizons,
                model_variant,
            )?;
            write_csv_table(
                &format!("rmse_{name}.csv"),
                country,
                &rmse_improvement(&rmse_abm, &rmse_base_abm)?,
                horizons,
                variable_names,
                model_variant,
            )?;
            write_csv_table(
                &format!("pval_{name}.csv"),
                country,
                &pval,
                horizons,
                variable_names,
                model_variant,
            )?;
        } else {
            eprintln!(
                "Warning: Skipping base ABM comparison for {country}: base variant not yet available"
            );
        }
    }

    // 4. Bias with HAC t-test
    let (latex, pval) = generate_bias_ttest(&error_abm, &bias_abm, horizons)?;
    let name = format!("{type_prefix}abm");
    write_latex_table(
        &format!("bias_{name}.tex"),
        country,
        &latex,
        horizons,
        model_variant,
    )?;
    write_csv_table(
        &format!("bias_{name}.csv"),
        country,
        &round_matrix(&bias_abm, 4),
        horizons,
        variable_names,
        model_variant,
    )?;
    write_csv_table(
        &format!("pval_bias_{name}.csv"),
        country,
        &pval,
        horizons,
        variable_names,
        model_variant,
    )
}

/// `create_bias_rmse_tables_var(forecast, actual, horizons, forecast_type,
/// model_type, variable_names, k, country; model_variant)`: the tables of
/// a benchmark (`model_type` "ar" or "var") with `k` lags in
/// `data/<country>/analysis/<model_variant>/`, with the prefix
/// `validation_` if `forecast_type` is "validation". For `k = 1` it saves
/// the forecasts as `forecast_ar.json` and writes `rmse_ar` (CSV and
/// LaTeX, rounded to 2 digits), `bias_ar` (LaTeX with the t-tests, CSV
/// rounded to 4 digits) and `pval_bias_ar.csv`; for other `k` it saves
/// `forecast_ar_<k>.json` and compares with the forecasts of `k = 1`:
/// `rmse_ar_<k>` (LaTeX with the DM tests, CSV) and `pval_ar_<k>.csv`.
///
/// Returns an error where Julia throws, for example if `k > 1` comes before
/// `k = 1`.
#[allow(clippy::too_many_arguments)]
pub fn create_bias_rmse_tables_var(
    forecast: &Array3,
    actual: &Array3,
    horizons: &[usize],
    forecast_type: &str,
    model_type: &str,
    variable_names: &[&str],
    k: usize,
    country: &str,
    model_variant: &str,
) -> Result<(), String> {
    let type_prefix = if forecast_type == "validation" {
        "validation_"
    } else {
        ""
    };
    let dir = analysis_dir(country, model_variant);
    io(&dir, std::fs::create_dir_all(&dir))?;
    let name = format!("{type_prefix}{model_type}");
    if k == 1 {
        save_forecast(&dir.join(format!("forecast_{name}.json")), forecast)?;
        let (rmse_var, bias_var, error_var) = calculate_forecast_errors(forecast, actual)?;
        let rmse_rounded = round_matrix(&rmse_var, 2);
        write_latex_table(
            &format!("rmse_{name}.tex"),
            country,
            &strings(&rmse_rounded),
            horizons,
            model_variant,
        )?;
        write_csv_table(
            &format!("rmse_{name}.csv"),
            country,
            &rmse_rounded,
            horizons,
            variable_names,
            model_variant,
        )?;
        let (latex, pval) = generate_bias_ttest(&error_var, &bias_var, horizons)?;
        write_latex_table(
            &format!("bias_{name}.tex"),
            country,
            &latex,
            horizons,
            model_variant,
        )?;
        write_csv_table(
            &format!("bias_{name}.csv"),
            country,
            &round_matrix(&bias_var, 4),
            horizons,
            variable_names,
            model_variant,
        )?;
        write_csv_table(
            &format!("pval_bias_{name}.csv"),
            country,
            &pval,
            horizons,
            variable_names,
            model_variant,
        )
    } else {
        save_forecast(&dir.join(format!("forecast_{name}_{k}.json")), forecast)?;
        let (rmse_var_k, _, error_var_k) = calculate_forecast_errors(forecast, actual)?;
        let forecast_base_var = load_forecast(&dir.join(format!("forecast_{name}.json")))?;
        let (rmse_base_var, _, error_base_var) =
            calculate_forecast_errors(&forecast_base_var, actual)?;
        let (latex, pval) = generate_dm_test_comparison(
            &error_var_k,
            &error_base_var,
            &rmse_var_k,
            &rmse_base_var,
            horizons,
        )?;
        write_latex_table(
            &format!("rmse_{name}_{k}.tex"),
            country,
            &latex,
            horizons,
            model_variant,
        )?;
        write_csv_table(
            &format!("rmse_{name}_{k}.csv"),
            country,
            &rmse_improvement(&rmse_var_k, &rmse_base_var)?,
            horizons,
            variable_names,
            model_variant,
        )?;
        write_csv_table(
            &format!("pval_{name}_{k}.csv"),
            country,
            &pval,
            horizons,
            variable_names,
            model_variant,
        )
    }
}

// =============================================================================
// Shared data loading for aggregate analysis
// =============================================================================

/// A column of a [`CsvTable`]: numbers, or text if a cell is not a number;
/// `None` for an empty cell (Julia's `missing`).
#[derive(Clone, Debug, PartialEq)]
pub enum CsvColumn {
    Numbers(Vec<Option<f64>>),
    Text(Vec<Option<String>>),
}

/// A table read by [`read_csv`], Julia's `CSV.read(file, DataFrame)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CsvTable {
    pub names: Vec<String>,
    pub columns: Vec<CsvColumn>,
}

impl CsvTable {
    /// The column called `name`.
    pub fn column(&self, name: &str) -> Option<&CsvColumn> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| &self.columns[i])
    }
}

/// The cells of the lines of a CSV text: separated by commas, cells in
/// quotes may contain commas, quotes (doubled) and line breaks. A cell is
/// `(text, quoted)`. Empty lines are skipped.
fn csv_records(text: &str) -> Result<Vec<Vec<(String, bool)>>, String> {
    let mut records = Vec::new();
    let mut record: Vec<(String, bool)> = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();
    let mut line_has_content = false;
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cell.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                cell.push(c);
            }
            continue;
        }
        match c {
            '"' if cell.is_empty() && !quoted => {
                in_quotes = true;
                quoted = true;
                line_has_content = true;
            }
            ',' => {
                record.push((std::mem::take(&mut cell), quoted));
                quoted = false;
                line_has_content = true;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' | '\r' => {
                if line_has_content || !cell.is_empty() {
                    record.push((std::mem::take(&mut cell), quoted));
                    records.push(std::mem::take(&mut record));
                }
                quoted = false;
                line_has_content = false;
            }
            _ => {
                cell.push(c);
                line_has_content = true;
            }
        }
    }
    if in_quotes {
        return Err("a quoted cell is not closed".into());
    }
    if line_has_content || !cell.is_empty() {
        record.push((cell, quoted));
        records.push(record);
    }
    Ok(records)
}

/// Reads a CSV file with a header line, Julia `CSV.read(file, DataFrame)`
/// for the tables of [`write_csv_table`]: a column whose cells are numbers
/// (also `NaN` and `Inf`) or empty is [`CsvColumn::Numbers`], any other
/// [`CsvColumn::Text`]; empty cells are missing, a quoted empty cell is
/// an empty text. CSV.jl also recognises dates, times and booleans, which
/// the tables do not contain.
///
/// Returns an error if the file cannot be read or a line does not have a
/// cell for each column (CSV.jl warns and fills or widens the table).
pub fn read_csv(path: &Path) -> Result<CsvTable, String> {
    let text = read_file(path)?;
    let mut records = csv_records(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if records.is_empty() {
        return Ok(CsvTable {
            names: Vec::new(),
            columns: Vec::new(),
        });
    }
    let names: Vec<String> = records.remove(0).into_iter().map(|(s, _)| s).collect();
    if let Some(r) = records.iter().position(|r| r.len() != names.len()) {
        return Err(format!(
            "{}: line {} has {} cells for {} columns",
            path.display(),
            r + 2,
            records[r].len(),
            names.len()
        ));
    }
    let columns = (0..names.len())
        .map(|c| {
            let cells: Vec<&(String, bool)> = records.iter().map(|r| &r[c]).collect();
            let missing = |(s, q): &(String, bool)| s.is_empty() && !q;
            let numbers: Option<Vec<Option<f64>>> = cells
                .iter()
                .map(|cell| {
                    if missing(cell) {
                        Some(None)
                    } else {
                        cell.0.parse::<f64>().ok().map(Some)
                    }
                })
                .collect();
            match numbers {
                Some(v) => CsvColumn::Numbers(v),
                None => CsvColumn::Text(
                    cells
                        .iter()
                        .map(|cell| (!missing(cell)).then(|| cell.0.clone()))
                        .collect(),
                ),
            }
        })
        .collect();
    Ok(CsvTable { names, columns })
}

/// The result of [`load_country_forecast_data`], Julia's named tuple
/// `(country_data, pval_data, valid_countries, variables, has_pvals)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CountryForecastData {
    /// The table of each country.
    pub country_data: BTreeMap<String, CsvTable>,
    /// The p-values of each country, for comparisons and bias tables.
    pub pval_data: BTreeMap<String, CsvTable>,
    /// The countries with a table, sorted.
    pub valid_countries: Vec<String>,
    /// [`get_variables_for_table`] of the table type.
    pub variables: Vec<String>,
    /// Whether the table type has p-values (comparisons `_vs_` and bias
    /// tables).
    pub has_pvals: bool,
}

/// `load_country_forecast_data(model_variant, table_type, countries)`: the
/// tables `data/<country>/analysis/<model_variant>/<table_type>.csv` of the
/// countries that have one, and for comparisons (`_vs_`) and bias tables
/// the p-values (`pval_…`). A table of absolute RMSE with a negative value
/// is skipped with a warning, as is a file that cannot be read. `None` if
/// no country has the table.
pub fn load_country_forecast_data(
    model_variant: &str,
    table_type: &str,
    countries: &[impl AsRef<str>],
) -> Option<CountryForecastData> {
    let variables = get_variables_for_table(table_type);
    let file = |country: &str, name: &str| {
        Path::new("data")
            .join(country)
            .join("analysis")
            .join(model_variant)
            .join(format!("{name}.csv"))
    };

    let mut country_data = BTreeMap::new();
    for country in countries {
        let country = country.as_ref();
        let csv_file = file(country, table_type);
        if !csv_file.is_file() {
            continue;
        }
        match read_csv(&csv_file) {
            Ok(df) => {
                if table_type.starts_with("rmse_") && !table_type.contains("vs_") {
                    // Julia checks the columns of numbers without missing
                    // values (`eltype <: Number`).
                    let negative = df.names.iter().zip(&df.columns).any(|(name, col)| {
                        name != "Horizon"
                            && matches!(col, CsvColumn::Numbers(v)
                                if v.iter().all(Option::is_some)
                                    && v.iter().flatten().any(|&x| x < 0.0))
                    });
                    if negative {
                        eprintln!(
                            "Warning: Skipping {}: negative values in absolute RMSE",
                            csv_file.display()
                        );
                        continue;
                    }
                }
                country_data.insert(country.to_string(), df);
            }
            Err(e) => eprintln!("Warning: Failed to load {}: {e}", csv_file.display()),
        }
    }

    if country_data.is_empty() {
        return None;
    }
    let valid_countries: Vec<String> = country_data.keys().cloned().collect();

    let is_comparison = table_type.contains("_vs_");
    let is_bias = table_type.starts_with("bias_");
    let has_pvals = is_comparison || is_bias;

    let mut pval_data = BTreeMap::new();
    if has_pvals {
        let pval_name = if is_comparison {
            table_type.replace("rmse_", "pval_")
        } else {
            format!("pval_{table_type}")
        };
        for country in &valid_countries {
            let pval_file = file(country, &pval_name);
            if !pval_file.is_file() {
                continue;
            }
            match read_csv(&pval_file) {
                Ok(df) => {
                    pval_data.insert(country.clone(), df);
                }
                Err(e) => eprintln!("Warning: Failed to load {}: {e}", pval_file.display()),
            }
        }
    }

    Some(CountryForecastData {
        country_data,
        pval_data,
        valid_countries,
        variables: variables.iter().map(|v| v.to_string()).collect(),
        has_pvals,
    })
}

/// The values of the variables in the row of `df` whose `Horizon` is
/// `horizon`, into `out` where present and not NaN; `false` if there is
/// not exactly one such row.
fn horizon_row(
    df: &CsvTable,
    horizon: &str,
    variables: &[String],
    out: &mut [f64],
) -> Result<bool, String> {
    let Some(h) = df.column("Horizon") else {
        return Err("the table has no column Horizon (Julia throws)".into());
    };
    let rows: Vec<usize> = match h {
        CsvColumn::Text(v) => (0..v.len())
            .filter(|&r| v[r].as_deref() == Some(horizon))
            .collect(),
        CsvColumn::Numbers(_) => Vec::new(),
    };
    let [row] = rows[..] else {
        return Ok(false);
    };
    for (j, var) in variables.iter().enumerate() {
        match df.column(var) {
            Some(CsvColumn::Numbers(v)) => {
                if let Some(x) = v[row].filter(|x| !x.is_nan()) {
                    out[j] = x;
                }
            }
            Some(CsvColumn::Text(v)) if v[row].is_some() => {
                return Err(format!(
                    "the column {var} holds text (Julia throws a MethodError for isnan)"
                ));
            }
            _ => {}
        }
    }
    Ok(true)
}

/// `build_12q_matrices(result; horizon = "12q")`: the values of the row
/// `horizon` of each country's table, `countries × variables`, and the
/// p-values in the same layout; NaN where a value is missing.
///
/// Returns an error if a table has no column `Horizon` or a variable's
/// column holds text (Julia throws).
pub fn build_12q_matrices(
    result: &CountryForecastData,
    horizon: &str,
) -> Result<(Matrix, Matrix), String> {
    let n_variables = result.variables.len();
    let mut matrix = vec![vec![f64::NAN; n_variables]; result.valid_countries.len()];
    let mut pval_matrix = matrix.clone();
    for (i, country) in result.valid_countries.iter().enumerate() {
        let df = &result.country_data[country];
        if !horizon_row(df, horizon, &result.variables, &mut matrix[i])? {
            continue;
        }
        if result.has_pvals
            && let Some(pdf) = result.pval_data.get(country)
        {
            horizon_row(pdf, horizon, &result.variables, &mut pval_matrix[i])?;
        }
    }
    Ok((matrix, pval_matrix))
}

// =============================================================================
// Country discovery
// =============================================================================

/// `discover_countries(; folder, subfolder)`: the entries of `folder` (in
/// Julia `"data"` by default) that have a folder `subfolder` (by default
/// `"abm_predictions/Model"`) with a `.json` file (`.jld2` in Julia),
/// sorted. No countries if `folder` does not exist.
pub fn discover_countries(folder: &Path, subfolder: &str) -> Result<Vec<String>, String> {
    let mut countries = Vec::new();
    if !folder.is_dir() {
        return Ok(countries);
    }
    for entry in read_dir_names(folder)? {
        let dir = folder.join(&entry).join(subfolder);
        if dir.is_dir() && read_dir_names(&dir)?.iter().any(|f| f.ends_with(".json")) {
            countries.push(entry);
        }
    }
    countries.sort();
    Ok(countries)
}

/// `discover_countries(variant; folder)`: [`discover_countries`] with the
/// subfolder `abm_predictions/<variant>`.
pub fn discover_countries_of_variant(variant: &str, folder: &Path) -> Result<Vec<String>, String> {
    discover_countries(folder, &format!("abm_predictions/{variant}"))
}

// =============================================================================
// Variable specifications of the error tables
// =============================================================================

/// The data of a variable of the error tables: the country's or the euro
/// area's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Data,
    Ea,
}

/// A variable of the error tables, Julia's named tuple `(key, transform,
/// source)`: the key of the series, the transformation of its values and
/// whose data it is.
#[derive(Clone, Copy, Debug)]
pub struct VarSpec {
    pub key: &'static str,
    pub transform: fn(f64) -> f64,
    pub source: Source,
}

fn log(x: f64) -> f64 {
    libm::log(x)
}

/// `x -> log(1 + x)`.
fn log_one_plus(x: f64) -> f64 {
    libm::log(1.0 + x)
}

/// `x -> (1 + x)^(1 / 4)`.
fn quarterly_rate(x: f64) -> f64 {
    libm::pow(1.0 + x, 1.0 / 4.0)
}

const fn spec(key: &'static str, transform: fn(f64) -> f64, source: Source) -> VarSpec {
    VarSpec {
        key,
        transform,
        source,
    }
}

/// `training_var_specs`: the variables of [`AGG_BASE_VARIABLES`]: the
/// logarithms of real GDP, consumption and investment, `log(1 + x)` of the
/// growth of the GDP deflator and `(1 + x)^(1/4)` of the Euribor.
pub const TRAINING_VAR_SPECS: [VarSpec; 5] = [
    spec("real_gdp_quarterly", log, Source::Data),
    spec("gdp_deflator_growth_quarterly", log_one_plus, Source::Data),
    spec("real_household_consumption_quarterly", log, Source::Data),
    spec("real_fixed_capitalformation_quarterly", log, Source::Data),
    spec("euribor", quarterly_rate, Source::Data),
];

/// `validation_var_specs`: the variables of [`AGG_VALIDATION_VARIABLES`].
pub const VALIDATION_VAR_SPECS: [VarSpec; 8] = [
    spec("real_gdp_quarterly", log, Source::Data),
    spec("gdp_deflator_growth_quarterly", log_one_plus, Source::Data),
    spec("real_government_consumption_quarterly", log, Source::Data),
    spec("real_exports_quarterly", log, Source::Data),
    spec("real_imports_quarterly", log, Source::Data),
    spec("real_gdp_quarterly", log, Source::Ea),
    spec("gdp_deflator_growth_quarterly", log_one_plus, Source::Ea),
    spec("euribor", quarterly_rate, Source::Data),
];

// =============================================================================
// Shared helpers of the error tables
// =============================================================================

fn series<'a>(t: &'a Table, name: &str, key: &str) -> Result<&'a Array, String> {
    t.get(key)
        .ok_or_else(|| format!("{name}[\"{key}\"] is missing (Julia throws a KeyError)"))
}

/// `data["quarters_num"] .== x` and the like: `f` of each date number.
fn quarter_mask(data: &Table, f: impl Fn(f64) -> bool) -> Result<Vec<bool>, String> {
    series(data, "data", "quarters_num")?
        .data
        .iter()
        .map(|q| {
            q.map(&f)
                .ok_or_else(|| "data[\"quarters_num\"] has a missing entry".to_string())
        })
        .collect()
}

/// Julia `src[key][mask]`: the values where `mask` holds, none missing.
fn select(src: &Table, name: &str, key: &str, mask: &[bool]) -> Result<Vec<f64>, String> {
    let a = series(src, name, key)?;
    if a.data.len() != mask.len() {
        return Err(format!(
            "{name}[\"{key}\"] has {} entries, data[\"quarters_num\"] {} (Julia throws a BoundsError)",
            a.data.len(),
            mask.len()
        ));
    }
    a.data
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(x, _)| x.ok_or_else(|| format!("{name}[\"{key}\"] has a missing value")))
        .collect()
}

/// Julia `src[key][mask]` with `missing` as `None`.
fn select_some(
    src: &Table,
    name: &str,
    key: &str,
    mask: &[bool],
) -> Result<Vec<Option<f64>>, String> {
    let a = series(src, name, key)?;
    if a.data.len() != mask.len() {
        return Err(format!(
            "{name}[\"{key}\"] has {} entries, data[\"quarters_num\"] {} (Julia throws a BoundsError)",
            a.data.len(),
            mask.len()
        ));
    }
    Ok(a.data
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(x, _)| *x)
        .collect())
}

fn table_of<'a>(spec: &VarSpec, data: &'a Table, ea: &'a Table) -> (&'a Table, &'static str) {
    match spec.source {
        Source::Data => (data, "data"),
        Source::Ea => (ea, "ea"),
    }
}

/// `extract_actual_values(data, ea, forecast_quarter_num, var_specs)`: the
/// transformed value of each variable in the quarter with the date number
/// `forecast_quarter_num`. As in Julia, the quarter is looked up in
/// `data["quarters_num"]` also for the series of the euro area.
///
/// Returns an error if a series is missing, the quarter is not there
/// exactly once, or its value is missing (Julia throws).
pub fn extract_actual_values(
    data: &Table,
    ea: &Table,
    forecast_quarter_num: f64,
    var_specs: &[VarSpec],
) -> Result<Vec<f64>, String> {
    let mask = quarter_mask(data, |q| q == forecast_quarter_num)?;
    var_specs
        .iter()
        .map(|spec| {
            let (src, name) = table_of(spec, data, ea);
            match select(src, name, spec.key, &mask)?[..] {
                [x] => Ok((spec.transform)(x)),
                ref v => Err(format!(
                    "{name}[\"{}\"] has {} values at date number {forecast_quarter_num}, one is needed",
                    spec.key,
                    v.len()
                )),
            }
        })
        .collect()
}

/// `extract_abm_forecast(model, forecast_quarter_num, number_of_seeds,
/// var_specs)`: the transformed mean over the runs of each variable of
/// the predictions `model` (the table of
/// [`Predictions::to_table`](crate::Predictions::to_table)) in the quarter
/// `forecast_quarter_num`. The series of the euro area are those with
/// `_ea_quarterly` in place of `_quarterly`.
///
/// Returns an error if a series is missing or is not a `quarters ×
/// number_of_seeds` matrix (Julia throws).
pub fn extract_abm_forecast(
    model: &Table,
    forecast_quarter_num: f64,
    number_of_seeds: usize,
    var_specs: &[VarSpec],
) -> Result<Vec<f64>, String> {
    let quarters = series(model, "model", "quarters_num")?;
    let mask: Vec<bool> = quarters
        .data
        .iter()
        .map(|q| {
            q.map(|q| q == forecast_quarter_num)
                .ok_or_else(|| "model[\"quarters_num\"] has a missing entry".to_string())
        })
        .collect::<Result<_, _>>()?;
    var_specs
        .iter()
        .map(|spec| {
            let key = match spec.source {
                Source::Ea => spec.key.replace("_quarterly", "_ea_quarterly"),
                Source::Data => spec.key.to_string(),
            };
            let a = series(model, "model", &key)?;
            if a.size != [mask.len(), number_of_seeds] {
                return Err(format!(
                    "model[\"{key}\"] has the size {:?}, not {} × {number_of_seeds} (Julia throws a BoundsError)",
                    a.size,
                    mask.len()
                ));
            }
            // Column-major order: the runs one after the other.
            let values = (0..number_of_seeds)
                .flat_map(|c| (0..mask.len()).map(move |r| (c, r)))
                .filter(|&(_, r)| mask[r])
                .map(|(c, r)| {
                    a.data[c * mask.len() + r]
                        .ok_or_else(|| format!("model[\"{key}\"] has a missing value"))
                })
                .collect::<Result<Vec<f64>, String>>()?;
            Ok((spec.transform)(mean(&values)))
        })
        .collect()
}

/// `date2num(lastdayofmonth(num2date(quarter_num) + Month(3 * horizon)))`.
fn forecast_quarter(quarter_num: f64, horizon: usize) -> f64 {
    Date::from_datenum(quarter_num)
        .add_months(3 * horizon as i64)
        .last_day_of_month()
        .datenum()
}

/// `YYYYQn.json` of the quarter with the date number `q`.
fn quarter_file(q: f64) -> String {
    let d = Date::from_datenum(q);
    format!("{}Q{}.json", d.year(), (d.month() - 1) / 3 + 1)
}

/// `Date(year(quarters[end]), 12, 31)`.
fn year_end(quarters: &[Date]) -> Result<Date, String> {
    let last = quarters
        .last()
        .ok_or("no quarters (Julia throws a BoundsError)")?;
    Ok(Date::new(last.year(), 12, 31))
}

/// The loop over quarters and horizons of the error tables: `f(i, j,
/// quarter_num, horizon, forecast_quarter_num)` for each quarter and each
/// horizon up to the end of the year of the last quarter whose forecast
/// quarter is in the data.
fn for_each_forecast(
    data: &Table,
    quarters: &[Date],
    horizons: &[usize],
    mut f: impl FnMut(usize, usize, f64, usize, f64) -> Result<(), String>,
) -> Result<(), String> {
    let year_end = year_end(quarters)?;
    let data_quarters = series(data, "data", "quarters_num")?;
    for (i, q) in quarters.iter().enumerate() {
        let quarter_num = q.datenum();
        for (j, &horizon) in horizons.iter().enumerate() {
            let forecast_quarter_num = forecast_quarter(quarter_num, horizon);
            if Date::from_datenum(forecast_quarter_num) > year_end {
                break;
            }
            if !data_quarters.data.contains(&Some(forecast_quarter_num)) {
                continue;
            }
            f(i, j, quarter_num, horizon, forecast_quarter_num)?;
        }
    }
    Ok(())
}

/// `Y0[from:end, :]` differenced, `diff(…; dims = 1)`, for the columns of
/// `Y0` (`None` for `missing`).
///
/// Returns an error if a value from row `from` on is missing: the
/// differences are then a matrix with `missing` values, which Julia's
/// `estimate_VAR` does not take (it throws a `MethodError`). Missing values
/// in earlier rows are not used.
fn differences(Y0: &[Vec<Option<f64>>], from: usize) -> Result<Vec<Vec<f64>>, String> {
    let rows = Y0.first().map_or(0, |c| c.len());
    if from > rows {
        return Err("too few quarters of data before the forecast (Julia throws)".into());
    }
    let used: Option<Vec<Vec<f64>>> = Y0
        .iter()
        .map(|c| c[from..].iter().copied().collect())
        .collect();
    let used = used.ok_or_else(|| {
        "a value of the data that the benchmark is estimated on is missing \
         (Julia's estimate_VAR throws a MethodError)"
            .to_string()
    })?;
    Ok((1..rows - from)
        .map(|t| used.iter().map(|c| c[t] - c[t - 1]).collect())
        .collect())
}

/// `cumsum(x)`, `missing` (`None`) from a missing value on.
fn cumsum(x: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut s: Vec<Option<f64>> = Vec::with_capacity(x.len());
    for (t, v) in x.iter().enumerate() {
        s.push(if t == 0 {
            *v
        } else {
            s[t - 1].zip(*v).map(|(a, b)| a + b)
        });
    }
    s
}

/// `Y[end, cols] = Y0[end, cols]' + sum(Y[:, cols], dims = 1)`, then
/// `Y[end, :]`: the levels of the differenced variables `cols` at the
/// horizon. The sums run from 0 in the order of the quarters (Julia adds
/// within a column with `@simd`, which can change the last digit).
///
/// Returns an error if a last level is missing (Julia cannot store
/// `missing` in `Y` and throws).
fn last_forecast(
    Y0: &[Vec<Option<f64>>],
    Y: &[Vec<f64>],
    cols: &[usize],
) -> Result<Vec<f64>, String> {
    let mut out = Y.last().cloned().unwrap_or_default();
    for &l in cols {
        let s = Y.iter().fold(0.0, |acc, r| acc + r[l]);
        let last = match Y0[l].last() {
            Some(Some(x)) => *x,
            Some(None) => {
                return Err("the last value of a series before the forecast is missing \
                     (Julia throws)"
                    .to_string());
            }
            None => f64::NAN,
        };
        out[l] = last + s;
    }
    Ok(out)
}

fn set_row(a: &mut Array3, i: usize, j: usize, values: &[f64]) {
    for (l, &v) in values.iter().enumerate() {
        a[(i, j, l)] = v;
    }
}

// =============================================================================
// Error table functions
// =============================================================================

/// The forecasts of an AR or VAR benchmark with `k` lags: `forecast(Y0, k,
/// horizon)` gives the row of forecasts at the horizon from the series up
/// to the quarter.
#[allow(clippy::too_many_arguments)]
fn error_table_benchmark(
    country: &str,
    ea: &Table,
    data: &Table,
    quarters: &[Date],
    horizons: &[usize],
    model_variant: &str,
    specs: &[VarSpec],
    variable_names: &[&str],
    forecast_type: &str,
    model_type: &str,
    columns: impl Fn(&[bool]) -> Result<Vec<Vec<Option<f64>>>, String>,
    forecast: impl Fn(&[Vec<Option<f64>>], usize, usize) -> Result<Vec<f64>, String>,
) -> Result<(), String> {
    let size = [quarters.len(), horizons.len(), variable_names.len()];
    for k in 1..=3 {
        let mut forecasts = Array3::nan(size);
        let mut actual = Array3::nan(size);
        for_each_forecast(
            data,
            quarters,
            horizons,
            |i, j, quarter_num, horizon, fq| {
                let values = extract_actual_values(data, ea, fq, specs)?;
                set_row(&mut actual, i, j, &values);
                let mask = quarter_mask(data, |q| q <= quarter_num)?;
                let Y0 = columns(&mask)?;
                set_row(&mut forecasts, i, j, &forecast(&Y0, k, horizon)?);
                Ok(())
            },
        )?;
        create_bias_rmse_tables_var(
            &forecasts,
            &actual,
            horizons,
            forecast_type,
            model_type,
            variable_names,
            k,
            country,
            model_variant,
        )?;
    }
    Ok(())
}

/// `error_table_ar(country, ea, data, quarters, horizons; model_variant)`:
/// for `k = 1, 2, 3` lags, the forecasts of univariate AR(k) benchmarks
/// with intercept ([`forecast_k_steps_VAR`]) from each quarter of
/// `quarters` (ends of quarters) to each horizon, and their tables
/// ([`create_bias_rmse_tables_var`], `model_type` "ar"). The AR models are
/// estimated on the differences, from the quarter `4 − k` on, of the
/// logarithms of real GDP, the GDP deflator (`gdp_deflator_quarterly`),
/// real consumption and real investment, and of the cumulated
/// `(1 + euribor)^(1/4)`; the levels at the horizon are the last value
/// plus the forecast differences, except for the deflator and the
/// Euribor, which keep the forecast difference. Horizons that end after
/// the year of the last quarter or outside the data are left as NaN.
///
/// BeforeIT's calibration overwrites `gdp_deflator_quarterly` in the data
/// it is given; for Julia's values after a calibration, call
/// [`CalibrationData::set_gdp_deflators`] on the data first.
///
/// Returns an error where Julia throws, for example if a series is missing.
pub fn error_table_ar(
    country: &str,
    ea: &Table,
    data: &Table,
    quarters: &[Date],
    horizons: &[usize],
    model_variant: &str,
) -> Result<(), String> {
    let presample = 4;
    let columns = |mask: &[bool]| -> Result<Vec<Vec<Option<f64>>>, String> {
        let logs = |key: &str| -> Result<Vec<Option<f64>>, String> {
            Ok(select_some(data, "data", key, mask)?
                .into_iter()
                .map(|x| x.map(log))
                .collect())
        };
        let rate: Vec<Option<f64>> = select_some(data, "data", "euribor", mask)?
            .into_iter()
            .map(|x| x.map(quarterly_rate))
            .collect();
        Ok(vec![
            logs("real_gdp_quarterly")?,
            logs("gdp_deflator_quarterly")?,
            logs("real_household_consumption_quarterly")?,
            logs("real_fixed_capitalformation_quarterly")?,
            cumsum(&rate),
        ])
    };
    let forecast =
        |Y0: &[Vec<Option<f64>>], k: usize, horizon: usize| -> Result<Vec<f64>, String> {
            let Y0_diff = differences(Y0, presample - k - 1)?;
            let nv = Y0.len();
            let mut Y = vec![vec![0.0; nv]; horizon];
            for l in 0..nv {
                let column: Vec<Vec<f64>> = Y0_diff.iter().map(|r| vec![r[l]]).collect();
                let f = forecast_k_steps_VAR(&column, horizon, true, k, None);
                for (t, r) in f.iter().enumerate() {
                    Y[t][l] = r[0];
                }
            }
            last_forecast(Y0, &Y, &[0, 2, 3])
        };
    error_table_benchmark(
        country,
        ea,
        data,
        quarters,
        horizons,
        model_variant,
        &TRAINING_VAR_SPECS,
        &AGG_BASE_VARIABLES,
        "training",
        "ar",
        columns,
        forecast,
    )
}

/// `error_table_validation_var(country, ea, data, quarters, horizons;
/// model_variant)`: for `k = 1, 2, 3` lags, the forecasts of a VAR(k) with
/// intercept ([`forecast_k_steps_VAR`]) of the eight variables of
/// [`AGG_VALIDATION_VARIABLES`] and their tables
/// ([`create_bias_rmse_tables_var`], `model_type` "var", prefix
/// `validation_`). The VAR is estimated on the differences, from the
/// quarter `5 − k` on, of the logarithms of real GDP, government
/// consumption, exports, imports and euro area GDP, of `log(1 + x)` of the
/// growth of the GDP deflator of the country and of the euro area, and of
/// the cumulated `(1 + euribor)^(1/4)`; the levels at the horizon are the
/// last value plus the forecast differences for the logarithms of levels,
/// the forecast difference for the others. As in Julia, the euro area
/// series are taken at the quarters of the country's data.
///
/// Returns an error where Julia throws, for example if a series is missing.
pub fn error_table_validation_var(
    country: &str,
    ea: &Table,
    data: &Table,
    quarters: &[Date],
    horizons: &[usize],
    model_variant: &str,
) -> Result<(), String> {
    let presample = 4;
    let columns = |mask: &[bool]| -> Result<Vec<Vec<Option<f64>>>, String> {
        let apply = |t: &Table, name: &str, key: &str, f: fn(f64) -> f64| {
            select_some(t, name, key, mask)
                .map(|v| v.into_iter().map(|x| x.map(f)).collect::<Vec<_>>())
        };
        Ok(vec![
            apply(data, "data", "real_gdp_quarterly", log)?,
            apply(data, "data", "gdp_deflator_growth_quarterly", log_one_plus)?,
            apply(data, "data", "real_government_consumption_quarterly", log)?,
            apply(data, "data", "real_exports_quarterly", log)?,
            apply(data, "data", "real_imports_quarterly", log)?,
            apply(ea, "ea", "real_gdp_quarterly", log)?,
            apply(ea, "ea", "gdp_deflator_growth_quarterly", log_one_plus)?,
            cumsum(&apply(data, "data", "euribor", quarterly_rate)?),
        ])
    };
    let forecast =
        |Y0: &[Vec<Option<f64>>], k: usize, horizon: usize| -> Result<Vec<f64>, String> {
            let Y0_diff = differences(Y0, presample - k)?;
            let Y = forecast_k_steps_VAR(&Y0_diff, horizon, true, k, None);
            last_forecast(Y0, &Y, &[0, 2, 3, 4, 5])
        };
    error_table_benchmark(
        country,
        ea,
        data,
        quarters,
        horizons,
        model_variant,
        &VALIDATION_VAR_SPECS,
        &AGG_VALIDATION_VARIABLES,
        "validation",
        "var",
        columns,
        forecast,
    )
}

/// The ABM tables of [`error_table_abm`] and
/// [`error_table_validation_abm`].
#[allow(clippy::too_many_arguments)]
fn error_table_of_abm(
    country: &str,
    ea: &Table,
    data: &Table,
    quarters: &[Date],
    horizons: &[usize],
    model_variant: &str,
    prediction_folder: &str,
    specs: &[VarSpec],
    variable_names: &[&str],
    forecast_type: &str,
) -> Result<(), String> {
    let year_end = year_end(quarters)?;
    let predictions_dir = Path::new(".")
        .join("data")
        .join(country)
        .join(prediction_folder);
    let load = |path: &Path| -> Result<Table, String> {
        table_from_json(&read_file(path)?).map_err(|e| format!("{}: {e}", path.display()))
    };

    // The number of runs, from the first prediction file there is.
    let mut number_of_seeds = None;
    for q in quarters {
        let fname = predictions_dir.join(quarter_file(q.datenum()));
        if fname.is_file() {
            let model = load(&fname)?;
            number_of_seeds = Some(series(&model, "model", "real_gdp_quarterly")?.size(2));
            break;
        }
    }
    let Some(number_of_seeds) = number_of_seeds else {
        eprintln!(
            "Warning: No prediction files found for {country} in {}, skipping",
            predictions_dir.display()
        );
        return Ok(());
    };

    let size = [quarters.len(), horizons.len(), variable_names.len()];
    let mut forecast = Array3::nan(size);
    let mut actual = Array3::nan(size);
    let data_quarters = series(data, "data", "quarters_num")?;
    for (i, q) in quarters.iter().enumerate() {
        let quarter_num = q.datenum();
        let fname = predictions_dir.join(quarter_file(quarter_num));
        // Quarters without a prediction file stay NaN.
        if !fname.is_file() {
            continue;
        }
        let model = load(&fname)?;
        for (j, &horizon) in horizons.iter().enumerate() {
            let fq = forecast_quarter(quarter_num, horizon);
            if Date::from_datenum(fq) > year_end {
                break;
            }
            if !data_quarters.data.contains(&Some(fq)) {
                continue;
            }
            set_row(
                &mut actual,
                i,
                j,
                &extract_actual_values(data, ea, fq, specs)?,
            );
            set_row(
                &mut forecast,
                i,
                j,
                &extract_abm_forecast(&model, fq, number_of_seeds, specs)?,
            );
        }
    }

    let prefix = if forecast_type == "validation" {
        "validation_"
    } else {
        ""
    };
    save_forecast(
        &analysis_dir(country, model_variant).join(format!("forecast_{prefix}abm.json")),
        &forecast,
    )?;
    create_bias_rmse_tables_abm(
        &forecast,
        &actual,
        horizons,
        forecast_type,
        variable_names,
        country,
        model_variant,
    )
}

/// `error_table_abm(country, ea, data, quarters, horizons; model_variant,
/// prediction_folder)`: the forecasts of the ABM from the prediction files
/// `./data/<country>/<prediction_folder>/YYYYQn.json` (in Julia by default
/// `abm_predictions/Model`) for each quarter of `quarters` and horizon,
/// the mean over the runs of the variables of [`TRAINING_VAR_SPECS`],
/// saved as `forecast_abm.json`, and their tables
/// ([`create_bias_rmse_tables_abm`]). Quarters without a prediction file
/// are left as NaN; without any prediction file it only prints a warning.
///
/// Returns an error where Julia throws, for example if the AR benchmark
/// ([`error_table_ar`]) has not been computed.
pub fn error_table_abm(
    country: &str,
    ea: &Table,
    data: &Table,
    quarters: &[Date],
    horizons: &[usize],
    model_variant: &str,
    prediction_folder: &str,
) -> Result<(), String> {
    error_table_of_abm(
        country,
        ea,
        data,
        quarters,
        horizons,
        model_variant,
        prediction_folder,
        &TRAINING_VAR_SPECS,
        &AGG_BASE_VARIABLES,
        "training",
    )
}

/// `error_table_validation_abm(country, ea, data, quarters, horizons;
/// model_variant, prediction_folder)`: as [`error_table_abm`] with the
/// variables of [`VALIDATION_VAR_SPECS`], saved as
/// `forecast_validation_abm.json`, compared with the VAR benchmark of
/// [`error_table_validation_var`].
pub fn error_table_validation_abm(
    country: &str,
    ea: &Table,
    data: &Table,
    quarters: &[Date],
    horizons: &[usize],
    model_variant: &str,
    prediction_folder: &str,
) -> Result<(), String> {
    error_table_of_abm(
        country,
        ea,
        data,
        quarters,
        horizons,
        model_variant,
        prediction_folder,
        &VALIDATION_VAR_SPECS,
        &AGG_VALIDATION_VARIABLES,
        "validation",
    )
}

// =============================================================================
// Batch runs
// =============================================================================

/// `save_all_params_and_initial_conditions(calibration_object,
/// folder_name; scale, first_calibration_date, last_calibration_date)`:
/// the parameters and initial conditions
/// ([`get_params_and_initial_conditions`]) of every third month from
/// `first_calibration_date` to `last_calibration_date` (in Julia by
/// default 2010-03-31 to 2011-12-31, `scale` 0.0005), written to
/// `<folder_name>/parameters/YYYYQn.json` and
/// `<folder_name>/initial_conditions/YYYYQn.json`.
///
/// **If `folder_name` exists, it is deleted with all its content first**,
/// after a warning, as in the Julia code.
///
/// Returns an error if a calibration fails or a file cannot be written.
pub fn save_all_params_and_initial_conditions(
    calibration_object: &CalibrationData,
    folder_name: impl AsRef<Path>,
    scale: f64,
    first_calibration_date: Date,
    last_calibration_date: Date,
) -> Result<(), String> {
    let folder = folder_name.as_ref();
    if folder.is_dir() {
        eprintln!(
            "Warning: The folder {} already exists and will be deleted with its content.",
            folder.display()
        );
        io(folder, std::fs::remove_dir_all(folder))?;
    }
    for calibration_date in Date::every_months(first_calibration_date, last_calibration_date, 3) {
        let c =
            get_params_and_initial_conditions(calibration_object, calibration_date, scale, false)?;
        for w in &c.warnings {
            eprintln!("Warning: {w}");
        }
        let name = quarter_file(calibration_date.datenum());
        write_file(
            &folder.join("parameters").join(&name),
            &c.parameters.to_json(),
        )?;
        write_file(
            &folder.join("initial_conditions").join(&name),
            &c.initial_conditions.to_json(),
        )?;
    }
    Ok(())
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".into())
}

/// `save_all_simulations(folder_name; T, n_sims, model_constructor,
/// simulation_folder)`: for each quarter that has both
/// `<folder_name>/parameters/YYYYQn.json` and
/// `<folder_name>/initial_conditions/YYYYQn.json`, builds a model with
/// `model_constructor` (for example [`Model::new`](crate::Model::new)),
/// runs `n_sims` copies of it for `T` quarters with [`ensemblerun`] in
/// parallel (Julia's default `parallel = true`), and writes the
/// [`DataVector`] to `<folder_name>/<simulation_folder>/YYYYQn.json`
/// (`simulations` if `None`). Julia's defaults are `T = 12` and `n_sims =
/// 4`. A quarter that fails, with an error or a panic, is skipped with a
/// warning, as Julia catches the exception.
///
/// Returns an error if a folder cannot be read or created, or holds a file
/// with a `Q` in its name that is not `YYYYQn.json` (Julia throws).
pub fn save_all_simulations<X, R, F>(
    folder_name: impl AsRef<Path>,
    T: usize,
    n_sims: usize,
    mut model_constructor: F,
    simulation_folder: Option<&str>,
    rng: &mut R,
) -> Result<(), String>
where
    X: Extension + Send,
    AbstractModel<X>: Clone,
    R: Randomness + ?Sized,
    F: FnMut(&Parameters, &InitialConditions, &mut R) -> Result<AbstractModel<X>, String>,
{
    let folder = folder_name.as_ref();
    let param_dir = folder.join("parameters");
    let init_dir = folder.join("initial_conditions");
    let sim_dir = folder.join(simulation_folder.unwrap_or("simulations"));
    io(&sim_dir, std::fs::create_dir_all(&sim_dir))?;

    let param_yq = extract_yq(&read_dir_names(&param_dir)?, "json")?;
    let init_yq = extract_yq(&read_dir_names(&init_dir)?, "json")?;
    for yq in param_yq.iter().filter(|f| init_yq.contains(f)) {
        let (year, quarter) = (&yq[..4], &yq[5..6]);
        println!("Y: {year} Q: {quarter}");
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            let parameters = Parameters::from_json(&read_file(&param_dir.join(yq))?)?;
            let initial_conditions = InitialConditions::from_json(&read_file(&init_dir.join(yq))?)?;
            let model = model_constructor(&parameters, &initial_conditions, rng)?;
            let mut model_vector = vec![model; n_sims];
            ensemblerun(&mut model_vector, T, rng, true, &NoShock);
            let data_vector = DataVector::from_models(&model_vector);
            write_file(&sim_dir.join(yq), &data_vector.to_json())
        }));
        let error = match result {
            Ok(Ok(())) => continue,
            Ok(Err(e)) => e,
            Err(p) => panic_message(p.as_ref()),
        };
        eprintln!("Warning: Skipping {year}Q{quarter} due to error: {error}");
    }
    Ok(())
}

/// `save_all_predictions_from_sims(folder_name, real_data;
/// simulation_suffix, prediction_suffix)`: for each simulation file
/// `<folder_name>/<simulation_suffix>/YYYYQn.json` (in Julia by default
/// `simulations`), the forecasts of [`get_predictions_from_sims`] from the
/// end of the quarter, written to
/// `<folder_name>/<prediction_suffix>/YYYYQn.json` (by default
/// `abm_predictions`).
///
/// Returns an error where Julia throws: if a folder cannot be read or
/// created, a file name with a `Q` is not `YYYYQn.json`, the quarter is not
/// 0 to 4, or a file or a prediction fails.
pub fn save_all_predictions_from_sims(
    folder_name: impl AsRef<Path>,
    real_data: &Table,
    simulation_suffix: &str,
    prediction_suffix: &str,
) -> Result<(), String> {
    let folder = folder_name.as_ref();
    let sim_folder = folder.join(simulation_suffix);
    let sim_yq = extract_yq(&read_dir_names(&sim_folder)?, "json")?;
    let pred_folder = folder.join(prediction_suffix);
    io(&pred_folder, std::fs::create_dir_all(&pred_folder))?;
    for yq in &sim_yq {
        let y: i32 = yq[..4].parse().expect("four digits");
        let q: u32 = yq[5..6].parse().expect("one digit");
        // The last day of quarter q; for q = 0 the 31 December before.
        let start_date = if q == 4 {
            Date::new(y, 12, 31)
        } else if q < 4 {
            Date::from_datenum(Date::new(y, q * 3 + 1, 1).datenum() - 1.0)
        } else {
            return Err(format!(
                "{yq}: month {} does not exist (Julia throws an ArgumentError)",
                q * 3 + 1
            ));
        };
        let file_name = sim_folder.join(format!("{y}Q{q}.json"));
        let sims = DataVector::from_json(&read_file(&file_name)?)
            .map_err(|e| format!("{}: {e}", file_name.display()))?;
        let predictions = get_predictions_from_sims(&sims, real_data, start_date)?;
        write_file(
            &pred_folder.join(format!("{y}Q{q}.json")),
            &table_to_json(&predictions.to_table()),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stars_and_rmse_improvement() {
        // BeforeIT test/utils/analysis_utils.jl, "stars / rmse_improvement".
        assert_eq!(stars(0.005), "***");
        assert_eq!(stars(0.02), "**");
        assert_eq!(stars(0.07), "*");
        assert_eq!(stars(0.5), "");
        assert_eq!(stars(f64::NAN), "");
        let r = rmse_improvement(&vec![vec![1.0, 2.0]], &vec![vec![2.0, 2.0]]).unwrap();
        assert_eq!(r, vec![vec![50.0, 0.0]]);
        // −round(0) is −0.0, which CSV.jl writes as "-0.0".
        assert!(r[0][1].is_sign_negative());
    }

    #[test]
    fn forecast_errors() {
        let mut forecast = Array3::nan([3, 1, 2]);
        let mut actual = Array3::nan([3, 1, 2]);
        for i in 0..3 {
            forecast[(i, 0, 0)] = i as f64;
            actual[(i, 0, 0)] = 0.5;
        }
        forecast[(0, 0, 1)] = 2.0;
        actual[(0, 0, 1)] = 1.0;
        let (rmse, bias, error) = calculate_forecast_errors(&forecast, &actual).unwrap();
        assert_eq!(error[(2, 0, 0)], 1.5);
        // Errors −0.5, 0.5, 1.5: mean 0.5, mean square 2.75 / 3.
        assert_eq!(bias[0][0], 0.5);
        assert_eq!(rmse[0][0], 100.0 * (2.75f64 / 3.0).sqrt());
        // Only the first quarter is observed for the second variable.
        assert_eq!((rmse[0][1], bias[0][1]), (100.0, 1.0));
        assert!(nanmean(&[f64::NAN]).is_nan());
        assert!(calculate_forecast_errors(&forecast, &Array3::nan([3, 1, 1])).is_err());
    }

    #[test]
    fn csv_cells_and_records() {
        assert_eq!(csv_text("Real GDP (EA)"), "Real GDP (EA)");
        assert_eq!(csv_text(""), "\"\"");
        assert_eq!(csv_text(" a"), "\" a\"");
        assert_eq!(csv_text("a,\"b\""), "\"a,\"\"b\"\"\"");
        let r = csv_records("a,\"b,\"\"c\"\"\"\n\n1,\"\"\r\n2,\n").unwrap();
        assert_eq!(
            r,
            vec![
                vec![("a".into(), false), ("b,\"c\"".into(), true)],
                vec![("1".into(), false), ("".into(), true)],
                vec![("2".into(), false), ("".into(), false)],
            ]
        );
        assert!(csv_records("\"a").is_err());
    }

    #[test]
    fn dates_of_the_forecasts() {
        let q = Date::new(2010, 3, 31).datenum();
        assert_eq!(forecast_quarter(q, 1), Date::new(2010, 6, 30).datenum());
        assert_eq!(forecast_quarter(q, 4), Date::new(2011, 3, 31).datenum());
        let q = Date::new(2010, 6, 30).datenum();
        assert_eq!(forecast_quarter(q, 2), Date::new(2010, 12, 31).datenum());
        assert_eq!(quarter_file(q), "2010Q2.json");
        assert_eq!(
            Date::every_months(Date::new(2010, 3, 31), Date::new(2010, 12, 31), 3),
            vec![
                Date::new(2010, 3, 31),
                Date::new(2010, 6, 30),
                Date::new(2010, 9, 30),
                Date::new(2010, 12, 31),
            ]
        );
    }

    #[test]
    fn levels_from_differences() {
        // Two variables, the first differenced: last level 10 plus the
        // forecast differences 1 and 2.
        let Y0 = vec![vec![Some(8.0), Some(10.0)], vec![Some(0.1), Some(0.2)]];
        let Y = vec![vec![1.0, 0.3], vec![2.0, 0.4]];
        assert_eq!(last_forecast(&Y0, &Y, &[0]).unwrap(), vec![13.0, 0.4]);
        let Y0 = vec![vec![Some(8.0), None]];
        assert!(last_forecast(&Y0, &Y, &[0]).is_err());
        let x = [None, Some(2.0), Some(4.0), Some(7.0)];
        assert_eq!(
            differences(&[x.to_vec()], 1).unwrap(),
            vec![vec![2.0], vec![3.0]]
        );
        // A missing value from the first row differenced on is used, as in
        // Julia, where the differences then hold `missing`.
        assert!(differences(&[x.to_vec()], 0).is_err());
        assert_eq!(
            cumsum(&[Some(1.0), Some(2.0), Some(3.0)]),
            vec![Some(1.0), Some(3.0), Some(6.0)]
        );
        assert_eq!(
            cumsum(&[Some(1.0), None, Some(3.0)]),
            vec![Some(1.0), None, None]
        );
    }
}
