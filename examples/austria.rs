//! Simulates the Austrian economy from 2010Q1 for 20 quarters, like the
//! example in the README of BeforeIT.jl, and prints real GDP.
//!
//! ```sh
//! cargo run --release --example austria
//! ```

use beforeit::{AUSTRIA2010Q1, Model, Xoshiro};

fn main() -> Result<(), String> {
    let parameters = &AUSTRIA2010Q1.parameters;
    let initial_conditions = &AUSTRIA2010Q1.initial_conditions;
    let mut rng = Xoshiro::seed_from_u64(1);
    let mut model = Model::new(parameters, initial_conditions, &mut rng)?;
    model.run(20, &mut rng);
    println!("quarter  real GDP");
    for (t, gdp) in model.data.real_gdp.iter().enumerate() {
        println!("{t:>7}  {gdp:>8.1}");
    }
    Ok(())
}
