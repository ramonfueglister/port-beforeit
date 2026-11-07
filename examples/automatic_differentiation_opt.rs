//! Maximising GDP growth by gradient ascent, BeforeIT
//! `examples/automatic_differentiation_opt.jl`: stochastic gradient ascent
//! of the GDP growth over four quarters with respect to 17 policy and
//! behavioural parameters on Austria 2010Q1, with the gradient by automatic
//! differentiation (Mooncake.jl) or by central finite differences, and the
//! GDP growth reached along each optimisation.
//!
//! ```sh
//! cargo run --release --features plots --example automatic_differentiation_opt
//! cargo run --release --features plots --example automatic_differentiation_opt -- --run
//! ```
//!
//! Each epoch computes the gradient on a new seed; within one gradient every
//! run of the objective uses the same seed, and the parameters of each epoch
//! are scored on held-out seeds that no gradient uses. The finite
//! differences use the two textbook steps eps^(1/3) and eps^(1/2).
//!
//! As in Julia, the example plots by default the saved results in
//! `examples/data/autodiff_optimization.csv`, copied from BeforeIT.jl, which
//! its authors computed with Julia: the Mooncake gradient in Float64 and
//! finite differences in Float64 and Float128. The port has no automatic
//! differentiation and computes only in `f64`, so `--run` and
//! `--run float64` rerun the finite differences in `f64` with the port
//! (Julia's `--run` without names runs all three optimisations), and `auto`
//! and `float128` are errors. Julia replaces the rows of the rerun in the
//! saved file; the port writes the saved results with the rows of the rerun
//! replaced to `target/example-output/automatic_differentiation_opt/` and
//! plots this copy if it exists. The plot is an SVG file there instead of a
//! PDF in the working directory.
//!
//! Julia seeds its global random number generator with `Random.seed!`; the
//! port starts a new [`Xoshiro`] with the same seed instead, which draws
//! other numbers than Julia's generator, so a rerun of the port gives other
//! values than the saved ones.

#![allow(non_snake_case)]

#[path = "common/mod.rs"]
mod common;

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::time::Instant;

use beforeit::analysis::{CsvColumn, csv_text, read_csv};
use beforeit::diff_transform::{array_to_model, model_to_array};
use beforeit::julia::{float_string, round, sum, zero_to_one};
use beforeit::plots::{Color, Legend, LineStyle, Plot, TickFormatter};
use beforeit::{AUSTRIA2010Q1, AbstractModel, Model, NoShock, Properties, Xoshiro};

const RESULTS_FILE: &str = "autodiff_optimization.csv";
const OPTIMISATIONS: [&str; 3] = ["auto", "float64", "float128"];

const HORIZON: usize = 4; // quarters simulated by the objective
const EPOCHS: usize = 50;
const LEARNING_RATE: f64 = 0.1;
const SEED: u64 = 42; // seed of the model initialisation
const EVAL_SEEDS: std::ops::RangeInclusive<u64> = 10_001..=10_010;

/// Seed of the gradient computed at `epoch`.
fn training_seed(epoch: usize) -> u64 {
    SEED + epoch as u64
}

/// The optimised model properties: tax rates, household and banking
/// parameters, all in [0, 1].
const PARAMETERS: [&str; 17] = [
    "tau_INC",
    "tau_FIRM",
    "tau_VAT",
    "tau_SIF",
    "tau_SIW",
    "tau_EXPORT",
    "tau_CF",
    "tau_G",
    "theta_UB",
    "psi",
    "psi_H",
    "mu",
    "theta_DIV",
    "theta",
    "zeta",
    "zeta_LTV",
    "zeta_b",
];

/// Smallest value of `psi` in the optimisation, the largest is 1: with
/// psi = 0 households buy nothing, and the household price index computed
/// in the goods market is 0 / 0, so psi is kept slightly above 0. The other
/// parameters go down to 0.
const PSI_MIN: f64 = 1.0e-6;

/// The property `name` of `prop`, Julia's `getfield(m.prop, parameter)`.
fn property<'a>(prop: &'a mut Properties, name: &str) -> &'a mut f64 {
    match name {
        "tau_INC" => &mut prop.tau_INC,
        "tau_FIRM" => &mut prop.tau_FIRM,
        "tau_VAT" => &mut prop.tau_VAT,
        "tau_SIF" => &mut prop.tau_SIF,
        "tau_SIW" => &mut prop.tau_SIW,
        "tau_EXPORT" => &mut prop.tau_EXPORT,
        "tau_CF" => &mut prop.tau_CF,
        "tau_G" => &mut prop.tau_G,
        "theta_UB" => &mut prop.theta_UB,
        "psi" => &mut prop.psi,
        "psi_H" => &mut prop.psi_H,
        "mu" => &mut prop.mu,
        "theta_DIV" => &mut prop.theta_DIV,
        "theta" => &mut prop.theta,
        "zeta" => &mut prop.zeta,
        "zeta_LTV" => &mut prop.zeta_LTV,
        "zeta_b" => &mut prop.zeta_b,
        _ => unreachable!("{name} is not in PARAMETERS"),
    }
}

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

/// Statistics.jl's `mean` and `std` (with n − 1).
fn mean_std(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = sum(x) / n;
    let d: Vec<f64> = x.iter().map(|v| (v - m) * (v - m)).collect();
    (m, (sum(&d) / (n - 1.0)).sqrt())
}

/// `@sprintf("%.1e", x)`: a two-digit exponent at least, as C's printf.
fn sprintf_e1(x: f64) -> String {
    let s = format!("{x:.1e}");
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let e: i32 = exp.parse().unwrap_or(0);
            let sign = if e < 0 { '-' } else { '+' };
            format!("{mantissa}e{sign}{:02}", e.abs())
        }
        None => s,
    }
}

/// `@sprintf("%.2f", x)`, or `"%+.2f"` with `plus`: Rust's format, except
/// NaN and the infinities, which Julia's Printf writes `NaN`, `Inf` and
/// `-Inf`, with a `+` before NaN and `Inf` for `%+`.
fn printf_2f(x: f64, plus: bool) -> String {
    let sign = if plus { "+" } else { "" };
    if x.is_nan() {
        format!("{sign}NaN")
    } else if x.is_infinite() {
        format!("{}Inf", if x < 0.0 { "-" } else { sign })
    } else if plus {
        format!("{x:+.2}")
    } else {
        format!("{x:.2}")
    }
}

/// `@sprintf("%.1f min", seconds / 60)`.
fn minutes(seconds: f64) -> String {
    format!("{:.1} min", seconds / 60.0)
}

/// The model, its GDP and its array, as the Julia example sets them up.
struct Setup {
    model: Model,
    gdpinit0: f64,
    model_arr: Vec<f64>,
    /// Position of each optimised parameter in the model array: the only
    /// entry that changes when the parameter does.
    indices: Vec<usize>,
    lower_bounds: Vec<f64>,
}

impl Setup {
    fn new() -> Result<Setup, String> {
        let s = &*AUSTRIA2010Q1;
        let mut rng = Xoshiro::seed_from_u64(SEED);
        let model = Model::new(&s.parameters, &s.initial_conditions, &mut rng)?;
        let gdpinit0 = gdp(&model);
        let model_arr = model_to_array(&model);
        let mut indices = Vec::new();
        for parameter in PARAMETERS {
            let mut m = model.clone();
            *property(&mut m.prop, parameter) += 1.0;
            let changed: Vec<usize> = model_to_array(&m)
                .iter()
                .zip(&model_arr)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| i)
                .collect();
            // Julia's `only`.
            match changed[..] {
                [i] => indices.push(i),
                _ => {
                    return Err(format!(
                        "{parameter} changes {} entries of the model array",
                        changed.len()
                    ));
                }
            }
        }
        let lower_bounds = PARAMETERS
            .iter()
            .map(|&p| if p == "psi" { PSI_MIN } else { 0.0 })
            .collect();
        Ok(Setup {
            model,
            gdpinit0,
            model_arr,
            indices,
            lower_bounds,
        })
    }

    /// Rewards GDP growth over `HORIZON` quarters, while keeping the initial
    /// GDP close to that of the original parameters; the random numbers
    /// from a new generator with the seed.
    fn objective(&self, model_arr: &[f64], seed: u64) -> Result<f64, String> {
        let mut m = array_to_model(model_arr, &self.model)?;
        let gdpinit = gdp(&m);
        let mut rng = Xoshiro::seed_from_u64(seed);
        m.steps(HORIZON, &mut rng, false, &NoShock);
        Ok((gdp(&m) - gdpinit) / gdpinit - ((gdpinit - self.gdpinit0) / self.gdpinit0).powi(2))
    }

    /// GDP after `HORIZON` quarters with the given seed.
    fn final_gdp(&self, model_arr: &[f64], seed: u64) -> Result<f64, String> {
        let mut rng = Xoshiro::seed_from_u64(seed);
        let mut m = array_to_model(model_arr, &self.model)?;
        m.steps(HORIZON, &mut rng, false, &NoShock);
        Ok(gdp(&m))
    }

    /// The percentage growth of GDP from the initial state on each held-out
    /// seed. The initial GDP counts no taxes on consumption, investment,
    /// government purchases or exports, whose flows are zero before the
    /// first quarter, so the growth includes those taxes appearing.
    fn eval_growths(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        let initial_gdp = gdp(&array_to_model(x, &self.model)?);
        EVAL_SEEDS
            .map(|seed| Ok(100.0 * (self.final_gdp(x, seed)? / initial_gdp - 1.0)))
            .collect()
    }

    /// The central finite differences of the objective with respect to the
    /// optimised parameters on the given seed, with the relative step
    /// `step`.
    fn numerical_gradient(&self, x: &[f64], seed: u64, step: f64) -> Result<Vec<f64>, String> {
        let mut x = x.to_vec();
        let mut g = vec![0.0; self.indices.len()];
        for (j, &i) in self.indices.iter().enumerate() {
            let v = x[i];
            let h = step * v.abs().max(1.0e-4);

            x[i] = v + h;
            let f_plus = self.objective(&x, seed)?;

            x[i] = v - h;
            let f_minus = self.objective(&x, seed)?;

            x[i] = v;
            g[j] = (f_plus - f_minus) / (2.0 * h);
        }
        Ok(g)
    }

    /// Stochastic gradient ascent from the original parameters with the
    /// finite differences of step `step`, scoring the parameters of each
    /// epoch on the held-out seeds.
    fn optimise(&self, method: &str, step: f64) -> Result<Vec<Row>, String> {
        let mut x = self.model_arr.clone();
        let mut results = Vec::new();
        let name = format!("{method} (h = {})", sprintf_e1(step));
        println!("{name} Float64: starting {EPOCHS} epochs");
        let start = Instant::now();
        for epoch in 1..=EPOCHS {
            let growths = self.eval_growths(&x)?;
            let (mean, sd) = mean_std(&growths);
            results.push(Row {
                method: method.to_string(),
                float_type: "Float64".to_string(),
                horizon: HORIZON,
                step: Some(step),
                epoch,
                gdp_growth_mean: mean,
                gdp_growth_sd: sd,
            });

            // A GDP that is not finite gives a gradient that is not finite
            // either, and the parameters would stay NaN from here on: stop
            // the optimisation.
            let finite = growths.iter().all(|g| g.is_finite());
            if epoch < EPOCHS && finite {
                // Keep the parameters, all rates and shares, within their
                // bounds (Julia's `clamp`, which keeps NaN).
                let g = self.numerical_gradient(&x, training_seed(epoch), step)?;
                for (j, &i) in self.indices.iter().enumerate() {
                    x[i] = (x[i] + LEARNING_RATE * g[j]).clamp(self.lower_bounds[j], 1.0);
                }
            }

            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "{name} Float64 epoch {epoch:2}/{EPOCHS} | GDP growth over {HORIZON} quarters on \
                 held-out seeds {}% (sd {}) | {} elapsed, about {} left",
                printf_2f(mean, true),
                printf_2f(sd, false),
                minutes(elapsed),
                minutes(elapsed / epoch as f64 * (EPOCHS - epoch) as f64)
            );
            if !finite {
                println!("{name} Float64: GDP is not finite, stopping");
                break;
            }
        }
        Ok(results)
    }
}

/// A row of the results.
#[derive(Clone, Debug)]
struct Row {
    method: String,
    float_type: String,
    horizon: usize,
    step: Option<f64>,
    epoch: usize,
    gdp_growth_mean: f64,
    gdp_growth_sd: f64,
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
    let (horizon, step, epoch) = (numbers("horizon")?, numbers("step")?, numbers("epoch")?);
    let (mean, sd) = (numbers("gdp_growth_mean")?, numbers("gdp_growth_sd")?);
    let missing = |c: &str| format!("{}: a value of {c} is missing", path.display());
    (0..method.len())
        .map(|r| {
            Ok(Row {
                method: method[r].clone().ok_or_else(|| missing("method"))?,
                float_type: float_type[r].clone().ok_or_else(|| missing("float_type"))?,
                horizon: horizon[r].ok_or_else(|| missing("horizon"))? as usize,
                step: step[r],
                epoch: epoch[r].ok_or_else(|| missing("epoch"))? as usize,
                gdp_growth_mean: mean[r].ok_or_else(|| missing("gdp_growth_mean"))?,
                gdp_growth_sd: sd[r].ok_or_else(|| missing("gdp_growth_sd"))?,
            })
        })
        .collect()
}

/// Julia's `sort` by `[:method, :float_type, :horizon, :step, :epoch]`:
/// `missing` after every number.
fn row_order(a: &Row, b: &Row) -> Ordering {
    let step = |r: &Row| r.step.map_or((1, 0.0), |s| (0, s));
    let (sa, sb) = (step(a), step(b));
    (&a.method, &a.float_type, a.horizon)
        .cmp(&(&b.method, &b.float_type, b.horizon))
        .then(sa.0.cmp(&sb.0))
        .then(sa.1.total_cmp(&sb.1))
        .then(a.epoch.cmp(&b.epoch))
}

/// The rows as CSV.jl writes the `DataFrame`.
fn results_csv(rows: &[Row]) -> String {
    let mut text =
        String::from("method,float_type,horizon,step,epoch,gdp_growth_mean,gdp_growth_sd\n");
    for r in rows {
        text.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            csv_text(&r.method),
            csv_text(&r.float_type),
            r.horizon,
            r.step.map(float_string).unwrap_or_default(),
            r.epoch,
            float_string(r.gdp_growth_mean),
            float_string(r.gdp_growth_sd)
        ));
    }
    text
}

/// Replaces the saved results of the optimisations run so far, keeping
/// those of other horizons.
fn save_results(measured: &[Row], saved_file: &Path, rerun_file: &Path) -> Result<(), String> {
    let rerun: Vec<(&str, &str, usize)> = measured
        .iter()
        .map(|r| (r.method.as_str(), r.float_type.as_str(), r.horizon))
        .collect();
    let source = if rerun_file.is_file() {
        rerun_file
    } else {
        saved_file
    };
    let mut rows: Vec<Row> = read_results(source)?
        .into_iter()
        .filter(|r| !rerun.contains(&(r.method.as_str(), r.float_type.as_str(), r.horizon)))
        .collect();
    rows.extend(measured.iter().cloned());
    rows.sort_by(row_order);
    let dir = rerun_file.parent().unwrap_or(Path::new("."));
    common::save(dir, RESULTS_FILE, &results_csv(&rows))
}

/// `"$(round(Int, y))%"`, the `yformatter` of the plot.
fn percent(y: f64) -> String {
    format!("{}%", round(y) as i64)
}

/// The finite-difference step named by its rule when it follows one;
/// eps(Float128) = 2^-112.
fn step_label(step: f64, float_type: &str) -> String {
    let e1 = sprintf_e1(step);
    // `replace(@sprintf("%.1e", step), r"e-0*" => "e-")`.
    let value = match e1.split_once("e-") {
        Some((mantissa, exp)) => format!("{mantissa}e-{}", exp.trim_start_matches('0')),
        None => e1,
    };
    let machine_eps = if float_type == "Float128" {
        2f64.powi(-112)
    } else {
        f64::EPSILON
    };
    for (rule, power) in [("ε^(1/3)", 1.0 / 3.0), ("ε^(1/2)", 1.0 / 2.0)] {
        let r = machine_eps.powf(power);
        // `isapprox(step, r; rtol = 1.0e-6)`.
        if step == r || (step - r).abs() <= 1.0e-6 * step.abs().max(r.abs()) {
            return format!("{rule} ≈ {value}");
        }
    }
    value
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = common::output_dir("automatic_differentiation_opt");
    let saved_file: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("data")
        .join(RESULTS_FILE);
    let rerun_file = dir.join(RESULTS_FILE);

    if let Some(run) = flag_values(&args, "--run") {
        // Julia runs all optimisations without names; the port has only
        // `float64`.
        let selected = if run.is_empty() {
            vec!["float64".to_string()]
        } else {
            run
        };
        let unknown: Vec<&String> = selected
            .iter()
            .filter(|o| !OPTIMISATIONS.contains(&o.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Err(format!(
                "Unknown optimisations {unknown:?}, choose from {OPTIMISATIONS:?}"
            ));
        }
        let without: Vec<&String> = selected.iter().filter(|o| *o != "float64").collect();
        if !without.is_empty() {
            return Err(format!(
                "the port has no counterpart of the optimisations {without:?} (Mooncake.jl's \
                 automatic differentiation, Float128): run --run float64"
            ));
        }
        println!("Running float64");
        let setup = Setup::new()?;
        // Relative steps of the finite differences, h = step * max(|x|,
        // 1e-4): eps^(1/3), which minimises the error of central
        // differences of a smooth function, and eps^(1/2), the
        // forward-difference rule.
        let fd_steps = [f64::EPSILON.powf(1.0 / 3.0), f64::EPSILON.sqrt()];
        let mut measured = Vec::new();
        for step in fd_steps {
            measured.extend(setup.optimise("numerical", step)?);
            save_results(&measured, &saved_file, &rerun_file)?;
        }
    }

    let results_file = if rerun_file.is_file() {
        &rerun_file
    } else {
        &saved_file
    };
    println!("results from {}", results_file.display());
    let results: Vec<Row> = read_results(results_file)?
        .into_iter()
        .filter(|r| r.horizon == HORIZON)
        .collect();

    // One colour per gradient method and precision; the finite-difference
    // steps of one precision share it and differ by dash pattern.
    let color = |key: &str| match key {
        "automatic" => Color::Rgb(0x2a, 0x78, 0xd6),
        "Float64" => Color::Rgb(0xeb, 0x68, 0x34),
        _ => Color::Rgb(0x1b, 0xaf, 0x7a),
    };
    let zero_line_color = Color::Rgb(0xc3, 0xc2, 0xb7);

    let mut p = Plot::new("");
    p.x_label = "epoch".to_string();
    p.y_label = format!("GDP growth over {HORIZON} quarters");
    p.y_formatter = Some(TickFormatter(percent));
    let xticks: Vec<f64> = (0..=EPOCHS).step_by(10).map(|e| e as f64).collect();
    let xtick_labels = (0..=EPOCHS).step_by(10).map(|e| e.to_string()).collect();
    p.x_ticks = Some((xticks, xtick_labels));
    p.legend = Legend::BottomRight;
    p.legend_frame = false;
    p.add_hline(0.0, zero_line_color, LineStyle::Solid);

    // Mean percentage growth of GDP from the initial state over the
    // held-out seeds (their standard deviation is saved but not drawn).
    // Series without saved results for this horizon are skipped; a series
    // that ends early stopped at a GDP that is not finite.
    let plot_gdp = |p: &mut Plot, r: &[&Row], label: Option<String>, c: Color, style: LineStyle| {
        let s = p.add_line(
            r.iter().map(|row| row.epoch as f64).collect(),
            r.iter().map(|row| row.gdp_growth_mean).collect(),
        );
        s.line_width = 2.0;
        s.label = label;
        s.color = c;
        s.line_style = style;
    };

    let mut automatic: Vec<&Row> = results.iter().filter(|r| r.method == "automatic").collect();
    automatic.sort_by_key(|r| r.epoch);
    let numerical: Vec<&Row> = results.iter().filter(|r| r.method == "numerical").collect();

    // The automatic gradient comes first in the legend but is drawn last,
    // on top.
    if !automatic.is_empty() {
        plot_gdp(
            &mut p,
            &automatic,
            Some("Automatic Gradient".to_string()),
            color("automatic"),
            LineStyle::Solid,
        );
    }
    for float_type in ["Float64", "Float128"] {
        let r_type: Vec<&Row> = numerical
            .iter()
            .copied()
            .filter(|r| r.float_type == float_type)
            .collect();
        // `sort(unique(r_type.step); rev = true)`.
        let mut steps: Vec<Option<f64>> = Vec::new();
        for r in &r_type {
            if !steps.contains(&r.step) {
                steps.push(r.step);
            }
        }
        let key = |s: &Option<f64>| s.map_or((1, 0.0), |v| (0, v));
        steps.sort_by(|a, b| {
            let (ka, kb) = (key(a), key(b));
            ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).reverse()
        });
        for (i, step) in steps.iter().enumerate() {
            let mut r: Vec<&Row> = r_type.iter().copied().filter(|r| r.step == *step).collect();
            r.sort_by_key(|r| r.epoch);
            let h = step.map_or_else(|| "missing".to_string(), |s| step_label(s, float_type));
            plot_gdp(
                &mut p,
                &r,
                Some(format!("Numerical Gradient ({float_type}, h = {h})")),
                color(float_type),
                [LineStyle::Solid, LineStyle::Dash, LineStyle::Dot][i % 3],
            );
        }
    }
    if !automatic.is_empty() {
        plot_gdp(
            &mut p,
            &automatic,
            None,
            color("automatic"),
            LineStyle::Solid,
        );
    }

    common::save(&dir, "gdp_optimization.svg", &p.to_svg((700, 440))?)?;
    Ok(())
}
