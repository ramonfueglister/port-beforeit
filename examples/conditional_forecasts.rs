//! Forecasts conditional on the realised path of government expenditure,
//! exports and imports, BeforeIT `examples/Conditional_forecasts.jl`: a
//! model type in which these three follow the data of the initial
//! conditions instead of their AR(1) rules, 8 runs of the model of Austria
//! for 12 quarters, and three figures of the results (25 series from
//! quarter 5, the six series of figure 3 of Poledna et al. (2023) from
//! quarter 5 and over all quarters).
//!
//! ```sh
//! cargo run --release --features plots --example conditional_forecasts
//! ```
//!
//! The plots are written as SVG to
//! `target/example-output/conditional_forecasts`.
//!
//! The Julia example builds its 8 models from the same agent objects, so
//! they share one state, which the constructor initialises 8 times and the
//! parallel runs then change together. Here each model gets its own copy of
//! the agents.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_TITLEFONT, layout_svg, plot_data_vector};
use beforeit::{
    AUSTRIA2010Q1, AbstractModel, Aggregates, Banks, CentralBank, Data, Extension, Firms,
    Government, NoShock, Properties, Randomness, RestOfTheWorld, Workers, Xoshiro, ensemblerun,
    julia,
};

/// The model type `NewModel` with `ExogenousGovernment` (`C_Gs`) and
/// `ExogenousRestOfTheWorld` (`C_Es`, `Y_Is`): the paths of government
/// expenditure, exports and imports from quarter 1 on.
#[derive(Clone, Debug)]
struct Exogenous {
    C_Gs: Vec<f64>,
    C_Es: Vec<f64>,
    Y_Is: Vec<f64>,
}

/// Julia `sum(a .* b)`.
fn sum_of_products(a: &[f64], b: &[f64]) -> f64 {
    julia::sum(&a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<_>>())
}

impl Extension for Exogenous {
    /// Government expenditure from its path instead of the AR(1) rule.
    fn gov_expenditure<R: Randomness + ?Sized>(
        model: &mut AbstractModel<Self>,
        _rng: &mut R,
    ) -> (f64, Vec<f64>) {
        println!("Using overridden gov_expenditure at t = {}", model.agg.t);
        let c_G_g = &model.prop.c_G_g; // weights of the goods in government consumption
        let P_bar_g = &model.agg.P_bar_g; // prices of the goods
        let pi_e = model.agg.pi_e; // expected inflation

        let C_G = model.ext.C_Gs[model.agg.t - 1]; // government expenditure in quarter t
        let J = model.gov.C_d_j.len(); // number of goods
        // Government demand for each good, adjusted for prices and inflation.
        let C_d_j = vec![C_G / J as f64 * sum_of_products(c_G_g, P_bar_g) * (1.0 + pi_e); J];
        (C_G, C_d_j)
    }

    /// Exports and imports from their paths instead of the AR(1) rules.
    fn rotw_import_export<R: Randomness + ?Sized>(
        model: &mut AbstractModel<Self>,
        _rng: &mut R,
    ) -> (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        println!("Using overridden rotw_import_export at t = {}", model.agg.t);
        let c_E_g = &model.prop.c_E_g; // weights of the goods in exports
        let c_I_g = &model.prop.c_I_g; // weights of the goods in imports
        let P_bar_g = &model.agg.P_bar_g; // prices of the goods
        let pi_e = model.agg.pi_e; // expected inflation

        // Export demand.
        let C_E = model.ext.C_Es[model.agg.t - 1]; // exports in quarter t
        let L = model.rotw.C_d_l.len(); // number of foreign buyers
        let C_d_l = vec![C_E / L as f64 * sum_of_products(c_E_g, P_bar_g) * (1.0 + pi_e); L];

        // Import demand.
        let Y_I = model.ext.Y_Is[model.agg.t - 1]; // imports in quarter t
        let Y_m = c_I_g.iter().map(|c| c * Y_I).collect(); // import demand by good
        let P_m = P_bar_g.iter().map(|p| p * (1.0 + pi_e)).collect(); // import prices

        (C_E, Y_I, C_d_l, Y_m, P_m)
    }
}

fn main() -> Result<(), String> {
    let dir = common::output_dir("conditional_forecasts");
    let mut rng = Xoshiro::seed_from_u64(1);

    // Parameters and initial conditions.
    let (p, ic) = (&AUSTRIA2010Q1.parameters, &AUSTRIA2010Q1.initial_conditions);

    let firms = Firms::new(p, ic, &mut rng)?;
    let (w_act, w_inact) = Workers::new(p, ic);
    let cb = CentralBank::new(p, ic);
    let banks = Banks::new(p, ic);
    let agg = Aggregates::new(p, ic);
    let properties = Properties::new(p, ic)?;
    let data = Data::default();

    // The government and the rest of the world of the model, with the paths.
    let standard_government = Government::new(p, ic);
    let standard_rotw = RestOfTheWorld::new(p, ic);
    let initial_C_G = standard_government.C_G;
    let initial_C_E = standard_rotw.C_E;
    let initial_Y_I = standard_rotw.Y_I;
    // The paths from the initial conditions, from quarter T_prime on.
    let T0 = p.T_prime;
    let C_Gs = ic.C_G[T0 - 1..].to_vec();
    let C_Es = ic.C_E[T0 - 1..].to_vec();
    let Y_Is = ic.Y_I[T0 - 1..].to_vec();
    // The first values are the initial ones.
    println!("Initial C_G match: {}", initial_C_G == C_Gs[0]);
    println!("Initial C_E match: {}", initial_C_E == C_Es[0]);
    println!("Initial Y_I match: {}", initial_Y_I == Y_Is[0]);

    let ext = Exogenous { C_Gs, C_Es, Y_Is };
    let agents = (
        w_act,
        w_inact,
        firms,
        banks,
        cb,
        standard_government,
        standard_rotw,
        agg,
        properties,
        data,
    );
    let mut new_models = (0..8)
        .map(|_| AbstractModel::from_agents(agents.clone(), ext.clone(), &mut rng))
        .collect::<Result<Vec<_>, _>>()?;

    // The conditional forecasts.
    let T = 12; // forecast horizon
    ensemblerun(&mut new_models, T, &mut rng, true, &NoShock);

    // All available variables from quarter 5.
    let all = [
        "nominal_gdp",
        "real_gdp",
        "nominal_gva",
        "real_gva",
        "nominal_household_consumption",
        "real_household_consumption",
        "nominal_government_consumption",
        "real_government_consumption",
        "nominal_capitalformation",
        "real_capitalformation",
        "nominal_fixed_capitalformation",
        "real_fixed_capitalformation",
        "nominal_fixed_capitalformation_dwellings",
        "real_fixed_capitalformation_dwellings",
        "nominal_exports",
        "real_exports",
        "nominal_imports",
        "real_imports",
        "operating_surplus",
        "compensation_employees",
        "wages",
        "taxes_production",
        "gdp_deflator_growth_ea",
        "real_gdp_ea",
        "euribor",
    ];
    let ps = plot_data_vector(&new_models, DEFAULT_TITLEFONT, &all, Some(5), None)?;
    common::save(&dir, "all.svg", &layout_svg(&ps, (5, 5), (1500, 1000))?)?;

    // Figure 3 of Poledna et al. (2023), from quarter 5 (2011).
    let figure_3 = [
        "real_gdp",
        "real_household_consumption",
        "real_fixed_capitalformation",
        "real_government_consumption",
        "real_exports",
        "real_imports",
    ];
    let ps = plot_data_vector(&new_models, DEFAULT_TITLEFONT, &figure_3, Some(5), None)?;
    common::save(
        &dir,
        "figure_3_from_2011.svg",
        &layout_svg(&ps, (2, 3), (1000, 600))?,
    )?;

    // The whole forecast from 2010Q2 for the same six series.
    let ps = plot_data_vector(&new_models, DEFAULT_TITLEFONT, &figure_3, None, None)?;
    common::save(&dir, "figure_3.svg", &layout_svg(&ps, (2, 3), (600, 400))?)?;
    Ok(())
}
