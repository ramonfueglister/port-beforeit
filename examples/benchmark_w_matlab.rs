//! Speed against the MATLAB code, BeforeIT `examples/benchmark_w_matlab.jl`:
//! the time of one quarter of the model of Austria (scale 1:1000) without
//! and with parallel goods markets, as the mean of 5 runs of 12 quarters,
//! in a bar plot next to the times of the MATLAB code, of C code generated
//! from it and, for the model in scale 1:1, of the HPC implementation of
//! Gill et al. (2021) and of BeforeIT.jl.
//!
//! ```sh
//! cargo run --release --features plots --example benchmark_w_matlab
//! ```
//!
//! The plot is written as SVG to `target/example-output/benchmark_w_matlab`
//! (Julia: `benchmark_w_matlab.pdf`).
//!
//! The times of MATLAB, of the generated C code, of the HPC implementation
//! and of BeforeIT.jl in scale 1:1 are the numbers of the Julia example.
//! The example measures the bars that Julia measures, those in scale
//! 1:1000, for the port, and labels them so. Julia runs them with 32-bit
//! numbers (`typeInt = Int32`, `typeFloat = Float32`) and threads pinned to
//! cores (ThreadPinning.jl); the port computes with `f64` and leaves the
//! threads to the system.

#![allow(non_snake_case)]

mod common;

use std::time::Instant;

use beforeit::julia::{float_string, round_digits};
use beforeit::plots::{Color, Legend, Plot, layout_svg};
use beforeit::{AUSTRIA2010Q1, InitialConditions, Model, NoShock, Parameters, Xoshiro};

/// A model run for `T` quarters.
fn run(
    parameters: &Parameters,
    initial_conditions: &InitialConditions,
    T: usize,
    parallel: bool,
    rng: &mut Xoshiro,
) -> Result<Model, String> {
    let mut model = Model::new(parameters, initial_conditions, rng)?;
    model.run_with(T, rng, parallel, &NoShock);
    Ok(model)
}

/// Seconds that `f` takes.
fn elapsed(f: impl FnOnce() -> Result<Model, String>) -> Result<f64, String> {
    let start = Instant::now();
    f()?;
    Ok(start.elapsed().as_secs_f64())
}

/// Statistics.jl's `mean`. Julia also computes the standard deviations of
/// the times, which it does not use.
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn per_step(times: &[f64], T: usize) -> Vec<f64> {
    times.iter().map(|t| t / T as f64).collect()
}

fn main() -> Result<(), String> {
    let dir = common::output_dir("benchmark_w_matlab");
    let mut rng = Xoshiro::seed_from_u64(1);
    let parameters = &AUSTRIA2010Q1.parameters;
    let initial_conditions = &AUSTRIA2010Q1.initial_conditions;
    let T = 12;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());

    // Julia first runs the code once to compile it, timing it with `@time`.
    let t = elapsed(|| run(parameters, initial_conditions, T, false, &mut rng))?;
    println!("first run, parallel = false: {t:.6} seconds");
    let t = elapsed(|| run(parameters, initial_conditions, T, true, &mut rng))?;
    println!("first run, parallel = true: {t:.6} seconds");

    // The times of the MATLAB code and of the C code generated from it with
    // MATLAB Coder (4 threads for the parallel version), measured
    // independently on an AMD Ryzen 5 5600H.
    let matlab_times_small = per_step(&[4.399592, 4.398576, 4.352314, 4.385039, 4.389989], T);
    let matlab_times_big = [
        2000.189556,
        1990.485305,
        2002.295329,
        1994.215843,
        1993.651854,
    ];
    let matlab_mtime_small = mean(&matlab_times_small);
    let matlab_mtime_big = mean(&matlab_times_big);

    let c_times_small = per_step(&[0.952, 0.94, 0.951, 0.942, 0.938], T);
    let c_times_big = [574.692, 576.725, 572.382, 573.921, 575.949];
    let c_mtime_small = mean(&c_times_small);
    let c_mtime_big = mean(&c_times_big);

    let c_times_small_multi = per_step(&[0.305, 0.324, 0.33, 0.334, 0.323], T);
    let c_times_big_multi = [209.603, 210.197, 209.61, 211.759, 209.174];
    let c_mtime_small_multi = mean(&c_times_small_multi);
    let c_mtime_big_multi = mean(&c_times_big_multi);

    // The times of "High-performance computing implementations of
    // agent-based economic models for realizing 1:1 scale simulations of
    // large economies", A. Gill et al. (2021).
    let hpc_mtime_big_multi = 22.92;
    let hpc_mtime_big = hpc_mtime_big_multi * 4.0;

    // The port, as the mean of 5 runs, here; BeforeIT.jl in scale 1:1 on
    // the platform of the MATLAB times, from the Julia example.
    let n_runs = 5;
    let mut port_times_small = Vec::new();
    for _ in 0..n_runs {
        port_times_small.push(elapsed(|| {
            run(parameters, initial_conditions, T, false, &mut rng)
        })?);
    }
    let port_times_small = per_step(&port_times_small, T);
    let julia_times_big = [46.400777, 47.218013, 46.981572, 46.532327, 46.232614];
    let port_mtime_small = mean(&port_times_small);
    let julia_mtime_big = mean(&julia_times_big);

    let mut port_times_small_multi = Vec::new();
    for _ in 0..n_runs {
        port_times_small_multi.push(elapsed(|| {
            run(parameters, initial_conditions, T, true, &mut rng)
        })?);
    }
    let port_times_small_multi = per_step(&port_times_small_multi, T);
    let julia_times_big_multi = [21.683823, 21.517169, 21.923911, 21.53007, 21.283416];
    let port_mtime_small_multi = mean(&port_times_small_multi);
    let julia_mtime_big_multi = mean(&julia_times_big_multi);

    let labels = [
        "MATLAB".to_string(),
        "Gen. C - 1 core".to_string(),
        "Gen. C - 4 cores".to_string(),
        "HPC - 1 core*".to_string(),
        "HPC - 4 cores*".to_string(),
        "BeforeIT.jl - 1 core".to_string(),
        "BeforeIT.jl - 4 cores".to_string(),
    ];
    let port_labels = [
        "Rust port - 1 thread".to_string(),
        format!("Rust port - {threads} threads"),
    ];

    // The bars in blue with black outlines and their heights above them
    // (Makie's `bar_labels = :y`, rounded to 3 digits).
    let bars = |heights: Vec<f64>, title: &str, top: f64, ticks: Vec<String>| {
        let labels = heights
            .iter()
            .map(|&h| float_string(round_digits(h, 3)))
            .collect();
        let max = heights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let n = heights.len();
        let mut p = Plot::bars(heights, Color::Rgb(0, 155, 228), labels);
        p.title = title.to_string();
        // Makie's `titlesize = 18` (pixels) in points.
        p.title_font = 13;
        p.legend = Legend::None;
        p.y_limits = Some((0.0, top * max));
        p.x_ticks = Some(((1..=n).map(|k| k as f64).collect(), ticks));
        // Makie's `xticklabelrotation = π / 5`.
        p.x_rotation = 36.0;
        p.x_grid = false;
        p
    };

    let times_small = vec![
        matlab_mtime_small,
        c_mtime_small,
        c_mtime_small_multi,
        port_mtime_small,
        port_mtime_small_multi,
    ];
    let mut ticks_small = labels[..3].to_vec();
    ticks_small.extend(port_labels);
    let mut ax1 = bars(times_small, "Model in scale 1:1000", 1.25, ticks_small);
    ax1.y_label = "time for one step (s)".to_string();

    let times_big = [
        matlab_mtime_big,
        c_mtime_big,
        c_mtime_big_multi,
        hpc_mtime_big,
        hpc_mtime_big_multi,
        julia_mtime_big,
        julia_mtime_big_multi,
    ];
    let rounded = times_big.iter().map(|&t| round_digits(t, 1)).collect();
    // Julia sets the limits from the times before rounding them.
    let max_big = times_big.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut ax2 = bars(rounded, "Model in scale 1:1", 1.15, labels.to_vec());
    ax2.y_limits = Some((0.0, 1.15 * max_big));

    common::save(
        &dir,
        "benchmark_w_matlab.svg",
        &layout_svg(&[ax1, ax2], (1, 2), (900, 400))?,
    )?;
    Ok(())
}
