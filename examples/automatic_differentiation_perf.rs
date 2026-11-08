//! Time of a gradient of a GDP objective, BeforeIT
//! `examples/automatic_differentiation_perf.jl`: the gradient of a one-step
//! GDP objective with respect to the first n entries of the model array
//! ([`model_to_array`]) on Austria 2010Q1, by automatic differentiation
//! (Mooncake.jl) and by central finite differences, and the time each takes.
//!
//! ```sh
//! cargo run --release --features plots --example automatic_differentiation_perf
//! cargo run --release --features plots --example automatic_differentiation_perf -- --run
//! ```
//!
//! As in Julia, the example plots by default the saved results in
//! `examples/data/autodiff_performance.csv`, copied from BeforeIT.jl, which
//! its authors measured with Julia: the Mooncake gradient in Float64 and
//! finite differences in Float64 and Float128. The port has no automatic
//! differentiation and computes only in `f64`, so `--run` and
//! `--run float64` time the finite differences in `f64` of the port (Julia's
//! `--run` without names runs all three benchmarks), and `auto` and
//! `float128` are errors. Julia replaces the rows of the rerun
//! in the saved file; the port writes the saved results with the rows of
//! the rerun replaced to `target/example-output/automatic_differentiation_perf/`
//! and plots this copy if it exists. The plot is an SVG file there instead
//! of a PDF in the working directory.
//!
//! Julia seeds its global random number generator with `Random.seed!(42)`
//! before each run of the objective; the port starts a new [`Xoshiro`] with
//! the seed 42 instead, which draws other numbers than Julia's generator.

#![allow(non_snake_case)]

#[path = "common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use beforeit::analysis::{CsvColumn, csv_text, read_csv};
use beforeit::diff_transform::{array_to_model, model_to_array};
use beforeit::julia::{float_string, sum, zero_to_one};
use beforeit::plots::{Color, Legend, LineStyle, Marker, Plot};
use beforeit::{AUSTRIA2010Q1, AbstractModel, Model, NoShock, Xoshiro};

const RESULTS_FILE: &str = "autodiff_performance.csv";
const BENCHMARKS: [&str; 3] = ["auto", "float64", "float128"];

/// The values given after `flag`, up to the next flag, separated by spaces
/// or commas; `None` without the flag.
fn flag_values(args: &[String], flag: &str) -> Option<Vec<String>> {
    let i = args.iter().position(|a| a == flag)?;
    let j = args[i + 1..]
        .iter()
        .position(|a| a.starts_with("--"))
        .map_or(args.len(), |k| i + 1 + k);
    Some(
        args[i + 1..j]
            .iter()
            .flat_map(|a| a.split(','))
            .filter(|v| !v.is_empty())
            .map(str::to_lowercase)
            .collect(),
    )
}

/// The GDP of the model as in the Julia example: value added of the firms,
/// product taxes, and taxes on consumption, investment, government
/// purchases and exports.
fn gdp<X>(m: &AbstractModel<X>) -> f64 {
    let tot_C_h = sum(&m.w_act.C_h) + sum(&m.w_inact.C_h) + sum(&m.firms.C_h) + sum(&m.banks.C_h);
    let tot_I_h = sum(&m.w_act.I_h) + sum(&m.w_inact.I_h) + sum(&m.firms.I_h) + sum(&m.banks.I_h);
    let f = &m.firms;
    let value_added: Vec<f64> = (0..f.Y_i.len())
        .map(|i| f.Y_i[i] * ((1.0 - f.tau_Y_i[i]) - 1.0 / f.beta_i[i]))
        .collect();
    let product_taxes: Vec<f64> = (0..f.Y_i.len()).map(|i| f.tau_Y_i[i] * f.Y_i[i]).collect();
    sum(&value_added)
        + sum(&product_taxes)
        + m.prop.tau_VAT * tot_C_h / zero_to_one(m.agg.P_bar_h)
        + m.prop.tau_CF * tot_I_h / zero_to_one(m.agg.P_bar_CF_h)
        + m.prop.tau_G * m.gov.C_j / zero_to_one(m.gov.P_j)
        + m.prop.tau_EXPORT * m.rotw.C_l / zero_to_one(m.rotw.P_l)
}

/// The objective: GDP growth over one quarter in percent, minus the square
/// of the deviation of the initial GDP from that of the original model in
/// percent; the random numbers of the quarter from a new generator with
/// the seed 42.
fn step_and_reduce(model_arr: &[f64], gdpinit0: f64, start_model: &Model) -> Result<f64, String> {
    let mut m = array_to_model(model_arr, start_model)?;
    let gdpinit = gdp(&m);
    let mut rng = Xoshiro::seed_from_u64(42);
    m.steps(1, &mut rng, false, &NoShock);
    Ok(100.0 * (gdp(&m) - gdpinit) / gdpinit - (100.0 * (gdpinit - gdpinit0) / gdpinit0).powi(2))
}

/// The central finite differences of the objective with respect to the
/// first `n` entries of the model array.
fn numerical_gradient(
    n: usize,
    model_arr: &[f64],
    gdpinit0: f64,
    model: &Model,
) -> Result<Vec<f64>, String> {
    let step_base = f64::EPSILON.powf(1.0 / 3.0);
    let mut x = model_arr.to_vec();
    let mut g = vec![0.0; n];
    for i in 0..n {
        let v = x[i];
        let h = step_base * v.abs().max(1.0e-4);

        x[i] = v + h;
        let f_plus = step_and_reduce(&x, gdpinit0, model)?;

        x[i] = v - h;
        let f_minus = step_and_reduce(&x, gdpinit0, model)?;

        x[i] = v;
        g[i] = (f_plus - f_minus) / (2.0 * h);
    }
    Ok(g)
}

/// A row of the results.
#[derive(Clone, Debug)]
struct Row {
    method: String,
    float_type: String,
    n: usize,
    seconds: f64,
    prepare_seconds: Option<f64>,
}

fn read_results(path: &Path) -> Result<Vec<Row>, String> {
    let t = read_csv(path)?;
    let text = |name: &str| match t.column(name) {
        Some(CsvColumn::Text(v)) => Ok(v.clone()),
        _ => Err(format!("{}: no text column {name}", path.display())),
    };
    let numbers = |name: &str| match t.column(name) {
        Some(CsvColumn::Numbers(v)) => Ok(v.clone()),
        _ => Err(format!("{}: no number column {name}", path.display())),
    };
    let (method, float_type) = (text("method")?, text("float_type")?);
    let (n, seconds, prepare) = (
        numbers("n")?,
        numbers("seconds")?,
        numbers("prepare_seconds")?,
    );
    let missing = |c: &str| format!("{}: a value of {c} is missing", path.display());
    (0..method.len())
        .map(|r| {
            Ok(Row {
                method: method[r].clone().ok_or_else(|| missing("method"))?,
                float_type: float_type[r].clone().ok_or_else(|| missing("float_type"))?,
                n: n[r].ok_or_else(|| missing("n"))? as usize,
                seconds: seconds[r].ok_or_else(|| missing("seconds"))?,
                prepare_seconds: prepare[r],
            })
        })
        .collect()
}

/// The rows as CSV.jl writes the `DataFrame`.
fn results_csv(rows: &[Row]) -> String {
    let mut text = String::from("method,float_type,n,seconds,prepare_seconds\n");
    for r in rows {
        let prepare = r.prepare_seconds.map(float_string).unwrap_or_default();
        text.push_str(&format!(
            "{},{},{},{},{}\n",
            csv_text(&r.method),
            csv_text(&r.float_type),
            r.n,
            float_string(r.seconds),
            prepare
        ));
    }
    text
}

/// Times the finite differences in `f64` and replaces the rows of
/// `numerical`/`Float64` of the saved results in `rerun_file`.
fn run_float64(saved_file: &Path, rerun_file: &Path) -> Result<(), String> {
    let s = &*AUSTRIA2010Q1;
    let mut rng = Xoshiro::seed_from_u64(42);
    let model = Model::new(&s.parameters, &s.initial_conditions, &mut rng)?;
    let gdpinit0 = gdp(&model);
    let model_arr = model_to_array(&model);

    // Two model steps per entry, so only the smaller sizes are timed.
    let ns = [1, 10, 100, 1000, 10000, model_arr.len()];
    let num_max_n = 1000;
    let mut measured = Vec::new();
    for n in ns.into_iter().filter(|&n| n <= num_max_n) {
        numerical_gradient(1, &model_arr, gdpinit0, &model)?;
        let start = Instant::now();
        numerical_gradient(n, &model_arr, gdpinit0, &model)?;
        let t = start.elapsed().as_secs_f64();
        println!("numerical  n={n:<6} Float64 gradient {t:.3} s");
        measured.push(Row {
            method: "numerical".to_string(),
            float_type: "Float64".to_string(),
            n,
            seconds: t,
            prepare_seconds: None,
        });
    }

    // Replace the saved results of the benchmark just run.
    let source = if rerun_file.is_file() {
        rerun_file
    } else {
        saved_file
    };
    let mut rows: Vec<Row> = read_results(source)?
        .into_iter()
        .filter(|r| !(r.method == "numerical" && r.float_type == "Float64"))
        .collect();
    rows.extend(measured);
    rows.sort_by(|a, b| (&a.method, &a.float_type, a.n).cmp(&(&b.method, &b.float_type, b.n)));
    let dir = rerun_file.parent().unwrap_or(Path::new("."));
    let name = rerun_file
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(RESULTS_FILE);
    common::save(dir, name, &results_csv(&rows))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = common::output_dir("automatic_differentiation_perf");
    let saved_file: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("data")
        .join(RESULTS_FILE);
    let rerun_file = dir.join(RESULTS_FILE);

    if let Some(run) = flag_values(&args, "--run") {
        // Julia runs all benchmarks without names; the port has only
        // `float64`.
        let selected: Vec<String> = if run.is_empty() {
            vec!["float64".to_string()]
        } else {
            run
        };
        let unknown: Vec<&String> = selected
            .iter()
            .filter(|b| !BENCHMARKS.contains(&b.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Err(format!(
                "Unknown benchmarks {unknown:?}, choose from {BENCHMARKS:?}"
            ));
        }
        let without: Vec<&String> = selected.iter().filter(|b| *b != "float64").collect();
        if !without.is_empty() {
            return Err(format!(
                "the port has no counterpart of the benchmarks {without:?} (Mooncake.jl's \
                 automatic differentiation, Float128): run --run float64"
            ));
        }
        println!("Running float64");
        run_float64(&saved_file, &rerun_file)?;
    }

    let results_file = if rerun_file.is_file() {
        &rerun_file
    } else {
        &saved_file
    };
    println!("results from {}", results_file.display());
    let results = read_results(results_file)?;
    let select_results = |method: &str, float_type: &str| {
        let mut r: Vec<&Row> = results
            .iter()
            .filter(|row| row.method == method && row.float_type == float_type)
            .collect();
        r.sort_by_key(|row| row.n);
        r
    };
    let automatic = select_results("automatic", "Float64");
    let numerical64 = select_results("numerical", "Float64");
    let numerical128 = select_results("numerical", "Float128");

    // The numerical gradient scales linearly with n: extend it to the full
    // model array.
    let n_full = results.iter().map(|r| r.n).max().unwrap_or(1) as f64;
    let extrapolate = |r: &[&Row]| {
        let last = r[r.len() - 1];
        let n = last.n as f64;
        (
            vec![n, n_full],
            vec![last.seconds, last.seconds * (n_full / n)],
        )
    };

    let xtick_vals = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0];
    let xtick_labels = xtick_vals
        .iter()
        .map(|x: &f64| format!("10^{}", x.log10() as i32))
        .collect();
    let ymaj = (-1..=5).map(|i| 10f64.powi(i)).collect();
    let ymaj_labels = (-1..=5).map(|i| format!("10^{i}")).collect();

    let mut p = Plot::new("");
    p.x_log = true;
    p.y_log = true;
    p.x_ticks = Some((xtick_vals.to_vec(), xtick_labels));
    p.y_ticks = Some((ymaj, ymaj_labels));
    // Minor ticks and grid lines.
    p.y_minor_grid = true;
    p.x_label = "parameters count".to_string();
    p.y_label = "time (s)".to_string();
    p.legend = Legend::TopLeft;

    let columns = |r: &[&Row]| -> (Vec<f64>, Vec<f64>) {
        (
            r.iter().map(|row| row.n as f64).collect(),
            r.iter().map(|row| row.seconds).collect(),
        )
    };

    // Series without saved results are skipped.
    if !automatic.is_empty() {
        let (x, y) = columns(&automatic);
        let s = p.add_line(x, y);
        s.label = Some("Automatic Gradient".to_string());
        s.line_width = 2.0;
        s.marker = Marker::Circle;
        s.color = Color::Palette(0);
    }

    for (r, label, marker, color) in [
        (
            &numerical64,
            "Numerical Gradient (Float64)",
            Marker::Square,
            Color::Palette(1),
        ),
        (
            &numerical128,
            "Numerical Gradient (Float128)",
            Marker::Diamond,
            Color::Palette(2),
        ),
    ] {
        if r.is_empty() {
            continue;
        }
        let (x, y) = extrapolate(r);
        let s = p.add_line(x, y);
        s.label = None;
        s.line_width = 2.0;
        s.line_style = LineStyle::Dash;
        s.color = color;
        let (x, y) = columns(r);
        let s = p.add_line(x, y);
        s.label = Some(label.to_string());
        s.line_width = 2.0;
        s.marker = marker;
        s.color = color;
    }

    common::save(&dir, "autodiff_performance.svg", &p.to_svg((600, 400))?)?;
    Ok(())
}
