//! Threads for one large run, BeforeIT `examples/multithreading_speedup.jl`:
//! the model of Italy from 2010Q1 (scale 1:2000, about 30 000 households)
//! run for 50 quarters with the goods markets one after the other and in
//! parallel threads, and the time each takes.
//!
//! ```sh
//! cargo run --release --example multithreading_speedup
//! ```
//!
//! Julia first runs the model once so that compilation is not timed; Rust
//! compiles ahead of time, so the port times the two runs right away.

#![allow(non_snake_case)]

use std::time::Instant;

use beforeit::{ITALY2010Q1, Model, NoShock, Xoshiro};

fn main() -> Result<(), String> {
    let mut rng = Xoshiro::seed_from_u64(1);
    let parameters = &ITALY2010Q1.parameters;
    let initial_conditions = &ITALY2010Q1.initial_conditions;
    let model = Model::new(parameters, initial_conditions, &mut rng)?;

    // The number of households,
    println!("prop.H = {}", model.prop.H);
    // which is the sum of active and inactive households and of the owners
    // of firms and banks.
    println!(
        "{}",
        model.w_act.len() + model.w_inact.len() + model.firms.len() + model.banks.len()
    );

    // The number of threads available, Julia's `Threads.nthreads()`.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("{threads} threads available");

    // Single-threaded and multi-threaded runs.
    let T = 50;
    let mut model = Model::new(parameters, initial_conditions, &mut rng)?;
    let start = Instant::now();
    model.run_with(T, &mut rng, false, &NoShock);
    println!("parallel = false: {:.2} s", start.elapsed().as_secs_f64());

    let mut model = Model::new(parameters, initial_conditions, &mut rng)?;
    let start = Instant::now();
    model.run_with(T, &mut rng, true, &NoShock);
    println!("parallel = true:  {:.2} s", start.elapsed().as_secs_f64());
    Ok(())
}
