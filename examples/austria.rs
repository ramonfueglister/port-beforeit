//! Simulates the Austrian economy from 2010Q1 for 20 quarters, like the
//! example in the README of BeforeIT.jl, and prints real GDP.
//!
//! ```sh
//! cargo run --release --example austria
//! ```

#[path = "../tests/common/mod.rs"]
mod common;

use beforeit::{Model, Xoshiro};

fn main() -> Result<(), String> {
    // Parameters and initial conditions from `tests/data`.
    let (parameters, initial_conditions) = common::austria();
    let mut rng = Xoshiro::seed_from_u64(1);
    let mut model = Model::new(&parameters, &initial_conditions, &mut rng)?;
    model.run(20, &mut rng);
    println!("quarter  real GDP");
    for (t, gdp) in model.data.real_gdp.iter().enumerate() {
        println!("{t:>7}  {gdp:>8.1}");
    }
    Ok(())
}
