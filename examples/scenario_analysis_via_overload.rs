//! A scenario by replacing a rule, BeforeIT
//! `examples/scenario_analysis_via_overload.jl`: 4 runs of the model of
//! Austria for 20 quarters as they are, and 4 in which the central bank
//! keeps the policy rate at 1 % before quarter 10 and follows the Taylor
//! rule afterwards; GDP of both as the mean with a band of one standard
//! error.
//!
//! ```sh
//! cargo run --release --features plots --example scenario_analysis_via_overload
//! ```
//!
//! The plot is written as SVG to
//! `target/example-output/scenario_analysis_via_overload`.
//!
//! Julia redefines `Bit.central_bank_rate` for `Bit.Model`, which changes
//! the rule for every model from then on; here the shocked runs are copies
//! of the model with a model type that replaces the rule.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_SIZE, ErrorType, Plot};
use beforeit::{
    AUSTRIA2010Q1, AbstractModel, DataVector, Extension, Model, NoShock, Xoshiro, ensemblerun,
    taylor_rule,
};

/// The shock: a rate of 1 % before quarter 10.
#[derive(Clone, Debug)]
struct LowRate;

impl Extension for LowRate {
    fn central_bank_rate(model: &mut AbstractModel<Self>) -> f64 {
        let cb = &model.cb;
        let gamma_EA = model.rotw.gamma_EA;
        let pi_EA = model.rotw.pi_EA;
        let taylor_rate = taylor_rule(
            cb.rho,
            cb.r_bar,
            cb.r_star,
            cb.pi_star,
            cb.xi_pi,
            cb.xi_gamma,
            gamma_EA,
            pi_EA,
        );
        if model.agg.t < 10 { 0.01 } else { taylor_rate }
    }
}

fn main() -> Result<(), String> {
    let dir = common::output_dir("scenario_analysis_via_overload");
    let mut rng = Xoshiro::seed_from_u64(1);
    let parameters = &AUSTRIA2010Q1.parameters;
    let initial_conditions = &AUSTRIA2010Q1.initial_conditions;

    // Initialise the model.
    let model = Model::new(parameters, initial_conditions, &mut rng)?;

    // Simulate copies of it for T quarters.
    let T = 20;
    let mut model_vec_baseline = vec![model.clone(); 4];
    ensemblerun(&mut model_vec_baseline, T, &mut rng, true, &NoShock);

    // The same with the shock.
    let mut model_vec_shocked = vec![common::with_rules(model, LowRate); 4];
    ensemblerun(&mut model_vec_shocked, T, &mut rng, true, &NoShock);

    // Baseline and shocked simulations.
    let Te = T + 1;
    let mut p = Plot::new("");
    p.title_font = 10;
    (p.x_label, p.y_label) = ("quarters".to_string(), "GDP".to_string());
    let gdp = |dv: DataVector| dv.series(|d| &d.real_gdp);
    p.add_errorline(
        common::one_to(Te),
        gdp(DataVector::from_models(&model_vec_baseline)),
        ErrorType::Sem,
    )?
    .label = Some("baseline".to_string());
    p.add_errorline(
        common::one_to(Te),
        gdp(DataVector::from_models(&model_vec_shocked)),
        ErrorType::Sem,
    )?
    .label = Some("shock".to_string());
    common::save(&dir, "gdp.svg", &p.to_svg(DEFAULT_SIZE)?)?;
    Ok(())
}
