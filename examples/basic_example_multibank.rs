//! A banking sector of several banks, BeforeIT
//! `examples/basic_example_multibank.jl`: the model of Austria with three
//! banks, the equity and the firms of each, and the equity and the loans of
//! each bank over 20 quarters, one line per bank.
//!
//! ```sh
//! cargo run --release --features plots --example basic_example_multibank
//! ```
//!
//! The plot is written as SVG to
//! `target/example-output/basic_example_multibank`.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_SIZE, Legend, Plot, layout_svg};
use beforeit::{AUSTRIA2010Q1, Model, Xoshiro};

fn main() -> Result<(), String> {
    let dir = common::output_dir("basic_example_multibank");
    let mut rng = Xoshiro::seed_from_u64(1);

    // A copy of the parameters, so that the shared preset stays as it is.
    let mut parameters = AUSTRIA2010Q1.parameters.clone();
    let initial_conditions = &AUSTRIA2010Q1.initial_conditions;

    // The number of banks. Each firm and household is assigned to one of
    // them at random, and the equity of the banking sector is split equally.
    parameters.n_banks = 3;
    let mut model = Model::new(&parameters, initial_conditions, &mut rng)?;

    // One entry per bank, for example the equity of each bank
    println!("equity of the banks: {:?}", model.banks.E_k);
    // and the number of firms that are customers of each bank.
    let customers: Vec<usize> = model.banks.firms.iter().map(Vec::len).collect();
    println!("firms per bank: {customers:?}");

    // Run the model as usual.
    let T = 20;
    model.run(T, &mut rng);

    // Some series are also collected per bank (`equity_per_bank`,
    // `credit_stock_per_bank`, `credit_new_per_bank`, `reserves_per_bank`
    // and `roe_per_bank`); one line per bank, as Julia plots the columns of
    // `reduce(hcat, model.data.equity_per_bank)'`.
    let per_bank = |lists: &[Vec<f64>], title: &str, labels: bool| {
        let mut p = Plot::new(title);
        p.title_font = 10;
        for k in 0..parameters.n_banks {
            let y: Vec<f64> = lists.iter().map(|v| v[k]).collect();
            let s = p.add_line(common::one_to(y.len()), y);
            if labels {
                s.label = Some(format!("bank {}", k + 1));
            }
        }
        if !labels {
            p.legend = Legend::None;
        }
        p
    };
    let p1 = per_bank(&model.data.equity_per_bank, "equity", true);
    let p2 = per_bank(&model.data.credit_stock_per_bank, "loans", false);
    common::save(
        &dir,
        "banks.svg",
        &layout_svg(&[p1, p2], (1, 2), DEFAULT_SIZE)?,
    )?;
    Ok(())
}
