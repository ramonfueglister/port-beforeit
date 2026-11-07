//! The essential use of the model, BeforeIT `examples/basic_example.jl`: a
//! model of Austria from 2010Q1 run for 20 quarters, its GDP and nine of its
//! series plotted, then 10 runs plotted as their mean with a band of one
//! standard deviation.
//!
//! ```sh
//! cargo run --release --features plots --example basic_example
//! ```
//!
//! The plots are written as SVG to `target/example-output/basic_example`.
//! The runs draw their own random numbers (seed 1), so the curves are not
//! those of a Julia session.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{
    DEFAULT_QUANTITIES, DEFAULT_SIZE, DEFAULT_TITLEFONT, Plot, layout_svg, plot_data,
    plot_data_vector,
};
use beforeit::{AUSTRIA2010Q1, Model, NoShock, Xoshiro, ensemblerun};

fn main() -> Result<(), String> {
    let dir = common::output_dir("basic_example");
    let mut rng = Xoshiro::seed_from_u64(1);

    // A precomputed set of parameters and initial conditions.
    let parameters = &AUSTRIA2010Q1.parameters;
    let initial_conditions = &AUSTRIA2010Q1.initial_conditions;

    // The model.
    let mut model = Model::new(parameters, initial_conditions, &mut rng)?;

    // Julia lists the fields with `fieldnames(typeof(model))`; here they are
    // the public fields of `AbstractModel`, and of `Banks` for the banks.
    println!("model: w_act, w_inact, firms, banks, cb, gov, rotw, agg, prop, data, ext");
    println!(
        "banks: del, lastid, id_to_index, ID, E_k, Pi_k, Pi_e_k, D_k, r, Y_h, C_d_h, I_d_h, \
         C_h, I_h, K_h, D_h, firms, w_act, w_inact"
    );

    // Run the model for T quarters, with the goods markets in parallel.
    let T = 20;
    for _ in 0..T {
        model.step_with(&mut rng, true, &NoShock);
        model.collect_data();
    }
    // `model.run(T, &mut rng)` does the same in one call (without parallel
    // goods markets).

    // Plot one series: `plot(model.data.real_gdp, title = "gdp", titlefont =
    // 10)`, whose single series Plots.jl labels "y1".
    let mut p = Plot::new("gdp");
    p.title_font = 10;
    p.add_line(
        common::one_to(model.data.real_gdp.len()),
        model.data.real_gdp.clone(),
    );
    common::save(&dir, "gdp.svg", &p.to_svg(DEFAULT_SIZE)?)?;

    // Or several at once with `plot_data`.
    let quantities = [
        "real_gdp",
        "real_household_consumption",
        "real_government_consumption",
        "real_capitalformation",
        "real_exports",
        "real_imports",
        "wages",
        "euribor",
        "gdp_deflator",
    ];
    let ps = plot_data(&model, DEFAULT_TITLEFONT, &quantities)?;
    common::save(&dir, "data.svg", &layout_svg(&ps, (3, 3), DEFAULT_SIZE)?)?;

    // Monte Carlo repetitions, run in parallel threads.
    let mut models = (0..10)
        .map(|_| Model::new(parameters, initial_conditions, &mut rng))
        .collect::<Result<Vec<_>, _>>()?;
    ensemblerun(&mut models, T, &mut rng, true, &NoShock);

    // The number of threads available, Julia's `Threads.nthreads()`.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("{threads} threads available");

    // The repetitions with `plot_data_vector`.
    let ps = plot_data_vector(&models, DEFAULT_TITLEFONT, &DEFAULT_QUANTITIES, None, None)?;
    common::save(
        &dir,
        "data_vector.svg",
        &layout_svg(&ps, (3, 3), DEFAULT_SIZE)?,
    )?;
    Ok(())
}
