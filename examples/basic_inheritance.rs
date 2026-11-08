//! Extending the model with new model types, BeforeIT
//! `examples/basic_inheritance.jl`.
//!
//! First by specialisation: a model type whose central bank keeps the
//! policy rate fixed at 2 %, run 4 times for 20 quarters next to the
//! standard model; the Euribor and the GDP deflator of both are plotted.
//! Then by invocation: a model type that also records the number of
//! employed workers each quarter, on top of the data of the standard model.
//!
//! ```sh
//! cargo run --release --features plots --example basic_inheritance
//! ```
//!
//! The plots are written as SVG to `target/example-output/basic_inheritance`.
//!
//! In Julia a new model type is declared with `Bit.@object` and new agent
//! types add fields; here a model type is a type that implements
//! `Extension`, and the fields it adds (`fixed_rate`, `N_employed`) are its
//! own. The Julia example builds the standard and the new model from the
//! same agent objects, so the second constructor works on the state the
//! first one left: among other things it collects the first data point a
//! second time and divides the deposits and capital of the firm owners and
//! of the inactive households by the total income once more. Here each
//! model gets its own copy of the agents, built as in the Julia example.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_TITLEFONT, Plot, layout_svg, plot_data_vectors_from};
use beforeit::{
    AUSTRIA2010Q1, AbstractModel, Agents, Aggregates, Banks, CentralBank, Data, DataVector,
    Extension, Firms, Government, Model, NoShock, Properties, RestOfTheWorld, Standard, Workers,
    Xoshiro, base, ensemblerun,
};

/// The model type of the first part, `NewModel` with `NewCentralBank`: the
/// central bank's added field and its rule.
#[derive(Clone, Debug)]
struct FixedRate {
    fixed_rate: f64,
}

impl Extension for FixedRate {
    /// `Bit.central_bank_rate(model::NewModel) = model.cb.fixed_rate`.
    fn central_bank_rate(m: &mut AbstractModel<Self>) -> f64 {
        m.ext.fixed_rate
    }
}

/// The model type of the second part, `NewModel2` with `MoreData`: the
/// number of employed workers at each data collection.
#[derive(Clone, Debug, Default)]
struct MoreData {
    N_employed: Vec<i64>,
}

impl Extension for MoreData {
    /// `collect_data!(m::NewModel2)`: the data of the standard model
    /// (`@invoke`), then the employed workers.
    fn collect_data(m: &mut AbstractModel<Self>) {
        base::collect_data(m);
        let employed = m.firms.N_i.iter().sum();
        m.ext.N_employed.push(employed);
    }
}

/// All model components, initialised one by one as in the Julia example.
fn components(rng: &mut Xoshiro) -> Result<Agents, String> {
    let (p, ic) = (&AUSTRIA2010Q1.parameters, &AUSTRIA2010Q1.initial_conditions);
    let firms = Firms::new(p, ic, rng)?;
    let (w_act, w_inact) = Workers::new(p, ic);
    let cb = CentralBank::new(p, ic);
    let banks = Banks::new(p, ic);
    let government = Government::new(p, ic);
    let rotw = RestOfTheWorld::new(p, ic);
    let agg = Aggregates::new(p, ic);
    let properties = Properties::new(p, ic)?;
    let data = Data::default();
    Ok((
        w_act, w_inact, firms, banks, cb, government, rotw, agg, properties, data,
    ))
}

fn main() -> Result<(), String> {
    let dir = common::output_dir("basic_inheritance");
    let mut rng = Xoshiro::seed_from_u64(1);

    // Extension by specialisation: the standard model and the new one, whose
    // central bank has the fixed rate.
    let agents = components(&mut rng)?;
    let std_model: Model = AbstractModel::from_agents(agents.clone(), Standard, &mut rng)?;
    let new_model = AbstractModel::from_agents(agents, FixedRate { fixed_rate: 0.02 }, &mut rng)?;

    // Four runs of each, from copies of the two models.
    let T = 20;
    let mut model_vec_standard = vec![std_model; 4];
    ensemblerun(&mut model_vec_standard, T, &mut rng, true, &NoShock);
    let mut model_vec_new = vec![new_model; 4];
    ensemblerun(&mut model_vec_new, T, &mut rng, true, &NoShock);

    // Both in one plot per quantity. Julia's `plot_data_vectors` takes a
    // vector that mixes the two model types; here their data do.
    let quantities = ["euribor", "gdp_deflator"];
    let data = [
        DataVector::from_models(&model_vec_standard),
        DataVector::from_models(&model_vec_new),
    ];
    let ps = plot_data_vectors_from(&data, DEFAULT_TITLEFONT, &quantities)?;
    common::save(
        &dir,
        "fixed_rate.svg",
        &layout_svg(&ps, (1, 2), (600, 300))?,
    )?;

    // Extension by invocation: the new data are collected on top of the
    // standard ones.
    let mut new_model =
        AbstractModel::from_agents(components(&mut rng)?, MoreData::default(), &mut rng)?;
    new_model.run(T, &mut rng);

    // `plot(new_model.data.N_employed, label = "Employed Workers")`.
    let y: Vec<f64> = new_model.ext.N_employed.iter().map(|&n| n as f64).collect();
    let mut p = Plot::new("");
    p.add_line(common::one_to(y.len()), y).label = Some("Employed Workers".to_string());
    common::save(&dir, "employed.svg", &p.to_svg((600, 400))?)?;
    Ok(())
}
