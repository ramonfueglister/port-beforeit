//! Speed-up with the number of threads, BeforeIT
//! `benchmark/benchmark_scalability.sh`: the time of a run of 12 quarters of
//! the model of Austria with parallel goods markets ("intra") and of 24
//! such runs in parallel ("ensemble"), measured with 1 to 6 threads, and the
//! speed-up over one thread.
//!
//! ```sh
//! benchmark/benchmark_scalability.sh
//! ```
//!
//! The shell script `benchmark/benchmark_scalability.sh` of the port, like
//! Julia's, runs the measurement once per number of threads and then draws
//! the plot; it calls this example with `measure <file>`, which appends the
//! two times to the CSV file, and with `plot <file>`, which draws
//! `benchmark_plot.svg` next to it (Julia: `benchmark_plot.png`). The files
//! are in `target/example-output/benchmark_scalability`.
//!
//! Julia starts `i` threads with `julia --threads=<i>`, pins them to cores
//! (ThreadPinning.jl) and runs all its parallel tasks on them. The parallel
//! runs of the port start system threads, as many at once as
//! `std::thread::available_parallelism` gives; the script limits the logical
//! CPUs of each measurement to `i` with `taskset -c 0-<i-1>` (Linux), for `i`
//! up to 6 and at most the number of logical CPUs (`nproc`), so that
//! `available_parallelism` is `i`. `run_model` then runs the goods markets
//! of up to `i` products at once; `ensemble_run_model` runs up to `i` models
//! at once, each with its goods markets in parallel, so up to `i × i`
//! threads share the `i` logical CPUs. Julia's `--gcthreads` and
//! `--check-bounds=no` have no counterpart. The port runs each function
//! once to warm up and then as often as 20 seconds allow, and stores the
//! minimum time, as the Julia script stores the minimum of
//! `@benchmark ... seconds=20` (in the column `MedianTime_ns`).

#![allow(non_snake_case)]

#[path = "common/mod.rs"]
mod common;

use std::path::Path;
use std::time::Instant;

use beforeit::analysis::{CsvColumn, read_csv};
use beforeit::julia::float_string;
use beforeit::plots::{Legend, Marker, Plot};
use beforeit::{AUSTRIA2010Q1, Model, NoShock, Xoshiro, ensemblerun};

/// `run_model`: a new model run for `T` quarters with parallel goods
/// markets.
fn run_model(T: usize, rng: &mut Xoshiro) -> Result<(), String> {
    let s = &*AUSTRIA2010Q1;
    let mut model = Model::new(&s.parameters, &s.initial_conditions, rng)?;
    model.run_with(T, rng, true, &NoShock);
    Ok(())
}

/// `ensemble_run_model`: 24 new models run for `T` quarters in parallel.
/// Julia passes a generator to `ensemblerun!`, which collects it first.
fn ensemble_run_model(T: usize, rng: &mut Xoshiro) -> Result<(), String> {
    let s = &*AUSTRIA2010Q1;
    let mut models = (0..24)
        .map(|_| Model::new(&s.parameters, &s.initial_conditions, rng))
        .collect::<Result<Vec<_>, _>>()?;
    ensemblerun(&mut models, T, rng, true, &NoShock);
    Ok(())
}

/// The minimum time in nanoseconds of the runs of `f` after one run to warm
/// up, running it as long as `seconds` allow (at least once), as
/// BenchmarkTools.jl's `minimum(@benchmark f() seconds=...)` for functions
/// that take milliseconds or more.
fn minimum_time(seconds: f64, mut f: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    f()?;
    let start = Instant::now();
    let mut min = f64::INFINITY;
    let mut samples = 0;
    while samples == 0 || (start.elapsed().as_secs_f64() < seconds && samples < 10_000) {
        let t = Instant::now();
        f()?;
        min = min.min(t.elapsed().as_nanos() as f64);
        samples += 1;
    }
    Ok(min)
}

fn measure(file: &Path) -> Result<(), String> {
    let mut rng = Xoshiro::seed_from_u64(1);
    let T = 12;
    let min_b1_ns = minimum_time(20.0, || run_model(T, &mut rng))?;
    let min_b2_ns = minimum_time(20.0, || ensemble_run_model(T, &mut rng))?;
    let num_threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("{num_threads} threads: run_model {min_b1_ns} ns, ensemble_run_model {min_b2_ns} ns");

    // Append the results to the CSV file.
    let mut text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    for (kind, t) in [("run_model", min_b1_ns), ("ensemble_run_model", min_b2_ns)] {
        text.push_str(&format!("{num_threads},{kind},{}\n", float_string(t)));
    }
    std::fs::write(file, text).map_err(|e| format!("{}: {e}", file.display()))
}

fn plot(file: &Path) -> Result<(), String> {
    let df = read_csv(file)?;
    let numbers = |name: &str| match df.column(name) {
        Some(CsvColumn::Numbers(v)) => v
            .iter()
            .map(|x| x.ok_or_else(|| format!("{}: a value of {name} is missing", file.display())))
            .collect::<Result<Vec<f64>, String>>(),
        _ => Err(format!("{}: no number column {name}", file.display())),
    };
    let threads = numbers("Threads")?;
    let times = numbers("MedianTime_ns")?;
    let kinds: Vec<String> = match df.column("BenchmarkType") {
        Some(CsvColumn::Text(v)) => v.iter().map(|k| k.clone().unwrap_or_default()).collect(),
        _ => return Err(format!("{}: no text column BenchmarkType", file.display())),
    };

    // The time with one thread of each benchmark, Julia's
    // `df[(df.Threads .== 1) .& (df.BenchmarkType .== ...), :MedianTime_ns][1]`.
    let one_thread = |kind: &str| {
        (0..kinds.len())
            .find(|&i| threads[i] == 1.0 && kinds[i] == kind)
            .map(|i| times[i])
            .ok_or_else(|| format!("{}: no time of {kind} with 1 thread", file.display()))
    };
    let run_model_1thread_time = one_thread("run_model")?;
    let ensemble_run_model_1thread_time = one_thread("ensemble_run_model")?;

    let normalized: Vec<f64> = (0..kinds.len())
        .map(|i| match kinds[i].as_str() {
            "run_model" => run_model_1thread_time / times[i],
            "ensemble_run_model" => ensemble_run_model_1thread_time / times[i],
            _ => 0.0,
        })
        .collect();
    let select = |kind: &str| -> (Vec<f64>, Vec<f64>) {
        (0..kinds.len())
            .filter(|&i| kinds[i] == kind)
            .map(|i| (threads[i], normalized[i]))
            .unzip()
    };
    let (x_intra, y_intra) = select("run_model");
    let (x_ensemble, y_ensemble) = select("ensemble_run_model");

    let m = y_ensemble.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let cm = if m.ceil() - m > 0.5 {
        m.ceil() - 0.5
    } else {
        m.ceil()
    };

    let mut p = Plot::new("Performance Scalability");
    p.x_label = "Number of Threads".to_string();
    p.y_label = "Speed-up w.r.t. 1 thread".to_string();
    p.legend = Legend::BottomRight;
    p.y_limits = Some((1.0, cm));
    // `yticks = 1:0.5:cm`, labelled as Plots.jl does.
    let ticks: Vec<f64> = (0..)
        .map(|k| 1.0 + 0.5 * k as f64)
        .take_while(|&v| v <= cm)
        .collect();
    let labels = ticks.iter().map(|v| format!("{v:.1}")).collect();
    p.y_ticks = Some((ticks, labels));

    let s = p.add_line(x_intra, y_intra);
    s.label = Some("intra".to_string());
    s.marker = Marker::Circle;
    s.line_width = 2.0;
    let s = p.add_line(x_ensemble, y_ensemble);
    s.label = Some("ensemble".to_string());
    s.marker = Marker::Square;
    s.line_width = 2.0;

    let dir = file.parent().unwrap_or(Path::new("."));
    common::save(dir, "benchmark_plot.svg", &p.to_svg((600, 400))?)?;
    println!("Plot saved to benchmark_plot.svg");
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: benchmark_scalability measure|plot <file.csv>";
    match (args.first().map(String::as_str), args.get(1)) {
        (Some("measure"), Some(file)) => measure(Path::new(file)),
        (Some("plot"), Some(file)) => plot(Path::new(file)),
        _ => Err(usage.to_string()),
    }
}
