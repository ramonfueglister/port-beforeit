//! The CANVAS model by replacing rules, BeforeIT
//! `examples/CANVAS_extension.jl`.
//!
//! Of the four changes of the CANVAS model (Hommes et al., 2025) with
//! respect to Poledna et al. (2023), the example implements two by
//! replacing rules: firms set prices and quantities by demand pull, and the
//! central bank re-estimates the Taylor rule every quarter (adaptive
//! learning); the other two need data on single households and firms. It
//! runs 8 standard models, 8 of its own CANVAS models and 8 of the CANVAS
//! models of the package (`ModelCANVAS`, which also has the Calvo option
//! `theta_calvo`) for 12 quarters from the calibration of Italy for
//! 2010Q1 at scale 0.001, and plots the three ensembles.
//!
//! ```sh
//! cargo run --release --features plots --example canvas_extension
//! ```
//!
//! The plot is written as SVG to `target/example-output/canvas_extension`.
//!
//! The Julia example builds its 8 CANVAS models from the same agent
//! objects, so they share one state, which the constructor initialises 8
//! times and the parallel runs then change together. Here each model gets
//! its own copy of the agents.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_QUANTITIES, DEFAULT_TITLEFONT, layout_svg, plot_data_vectors_from};
use beforeit::{
    AbstractModel, Aggregates, Banks, CentralBank, Data, DataVector, Date, Extension, Firms,
    FirmsDecisions, Government, ITALY_CALIBRATION, Model, ModelCANVAS, NoShock, Properties,
    Randomness, RestOfTheWorld, Workers, Xoshiro, ensemblerun, estimate::estimate_taylor_rule,
    get_params_and_initial_conditions, julia, taylor_rule,
};

/// The example's model type `ModelCANVAS`, with the fields of
/// `CentralBankCANVAS` (`r_bar_series`) and `RestOfTheWorldCANVAS`
/// (`Y_EA_series`, `pi_EA_series`).
#[derive(Clone, Debug)]
struct Canvas {
    r_bar_series: Vec<f64>,
    Y_EA_series: Vec<f64>,
    pi_EA_series: Vec<f64>,
}

impl Extension for Canvas {
    // The four cases of the Julia code, two of which act alike.
    #[allow(clippy::if_same_then_else)]
    fn firms_expectations_and_decisions<R: Randomness + ?Sized>(
        model: &mut AbstractModel<Self>,
        _rng: &mut R,
    ) -> FirmsDecisions {
        let model = &*model;
        let firms = &model.firms;

        // Non-firm variables.
        let P_bar_g = &model.agg.P_bar_g;
        let gamma_e = model.agg.gamma_e;
        let pi_e = model.agg.pi_e;

        // Quantity and price adjustments of each firm.
        let I = firms.G_i.len();
        let mut gamma_d_i = vec![0.0; I];
        let mut pi_d_i = vec![0.0; I];
        for i in 0..I {
            let (Q_s, Q_d, P, P_g) = (
                firms.Q_s_i[i],
                firms.Q_d_i[i],
                firms.P_i[i],
                P_bar_g[firms.G_i[i]],
            );
            if Q_s <= Q_d && P >= P_g {
                gamma_d_i[i] = Q_d / Q_s - 1.0;
                pi_d_i[i] = 0.0;
            } else if Q_s <= Q_d && P < P_g {
                gamma_d_i[i] = 0.0;
                pi_d_i[i] = Q_d / Q_s - 1.0;
            } else if Q_s > Q_d && P >= P_g {
                gamma_d_i[i] = 0.0;
                pi_d_i[i] = Q_d / Q_s - 1.0;
            } else if Q_s > Q_d && P < P_g {
                gamma_d_i[i] = Q_d / Q_s - 1.0;
                pi_d_i[i] = 0.0;
            }
        }
        let Q_s_i: Vec<f64> = (0..I)
            .map(|i| firms.Q_s_i[i] * (1.0 + gamma_e) * (1.0 + gamma_d_i[i]))
            .collect();

        // Cost-push inflation.
        let pi_c_i = Self::cost_push_inflation(model, firms);
        // Price setting.
        let new_P_i = (0..I)
            .map(|i| firms.P_i[i] * (1.0 + pi_c_i[i]) * (1.0 + pi_e) * (1.0 + pi_d_i[i]))
            .collect();
        // Target investment, intermediate goods and employment.
        let (I_d_i, DM_d_i, N_d_i) = Self::desired_capital_material_employment(firms, &Q_s_i);
        // Expected profits.
        let Pi_e_i: Vec<f64> = firms
            .Pi_i
            .iter()
            .map(|pi| pi * (1.0 + pi_e) * (1.0 + gamma_e))
            .collect();
        // Expected deposits, capital and loans.
        let (DD_e_i, K_e_i, L_e_i) = Self::expected_deposits_capital_loans(model, firms, &Pi_e_i);
        // Target loans.
        let DL_d_i = (0..I)
            .map(|i| julia::max(0.0, -DD_e_i[i] - firms.D_i[i]))
            .collect();
        FirmsDecisions {
            Q_s_i,
            I_d_i,
            DM_d_i,
            N_d_i,
            Pi_e_i,
            DL_d_i,
            K_e_i,
            L_e_i,
            new_P_i,
        }
    }

    fn central_bank_rate(model: &mut AbstractModel<Self>) -> f64 {
        let (gamma_EA, pi_EA) = (model.rotw.gamma_EA, model.rotw.pi_EA);
        let n = model.prop.T_prime + model.agg.t - 1;
        let e = &model.ext;
        let (a1, a2, a3) = (
            &e.r_bar_series[..n],
            &e.Y_EA_series[..n],
            &e.pi_EA_series[..n],
        );

        // Update the parameters of the central bank.
        let cb = &mut model.cb;
        (cb.rho, cb.r_star, cb.xi_pi, cb.xi_gamma, cb.pi_star) = estimate_taylor_rule(a1, a2, a3);
        let r_bar = taylor_rule(
            cb.rho,
            cb.r_bar,
            cb.r_star,
            cb.pi_star,
            cb.xi_pi,
            cb.xi_gamma,
            gamma_EA,
            pi_EA,
        );
        // Append the new value to the series.
        model.ext.r_bar_series.push(r_bar);
        r_bar
    }

    fn growth_inflation_EA<R: Randomness + ?Sized>(
        model: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, f64, f64) {
        let rotw = &model.rotw;
        let epsilon_Y_EA = model.agg.epsilon_Y_EA;

        let Y_EA = (rotw.alpha_Y_EA * rotw.Y_EA.ln() + rotw.beta_Y_EA + epsilon_Y_EA).exp(); // GDP EA
        let gamma_EA = Y_EA / rotw.Y_EA - 1.0; // growth EA
        let epsilon_pi_EA = rng.randn() * rotw.sigma_pi_EA;
        let pi_EA = (rotw.alpha_pi_EA * (1.0 + rotw.pi_EA).ln() + rotw.beta_pi_EA + epsilon_pi_EA)
            .exp()
            - 1.0; // inflation EA
        // Append the new values to the series.
        model.ext.Y_EA_series.push(Y_EA);
        model.ext.pi_EA_series.push(pi_EA);
        (Y_EA, gamma_EA, pi_EA)
    }
}

fn main() -> Result<(), String> {
    let dir = common::output_dir("canvas_extension");
    let mut rng = Xoshiro::seed_from_u64(1);

    // Parameters and initial conditions.
    let calibration_date = Date::new(2010, 3, 31);
    let c = get_params_and_initial_conditions(&ITALY_CALIBRATION, calibration_date, 0.001, false)?;
    let (p, ic) = (&c.parameters, &c.initial_conditions);

    // The historical data only; the series grow during the simulation.
    let ext = Canvas {
        r_bar_series: ic.r_bar_series.clone(),
        Y_EA_series: ic.Y_EA_series.clone(),
        pi_EA_series: ic.pi_EA_series.clone(),
    };

    // The firms, with expected sales equal to demand so that the first
    // price and quantity setting does not divide by 0.
    let mut firms = Firms::new(p, ic, &mut rng)?;
    firms.Q_s_i.clone_from(&firms.Q_d_i);
    let cb = CentralBank::new(p, ic);
    let rotw = RestOfTheWorld::new(p, ic);

    // The standard components.
    let (w_act, w_inact) = Workers::new(p, ic);
    let banks = Banks::new(p, ic);
    let agg = Aggregates::new(p, ic);
    let gov = Government::new(p, ic);
    let prop = Properties::new(p, ic)?;
    let data = Data::default();
    let agents = (w_act, w_inact, firms, banks, cb, gov, rotw, agg, prop, data);

    let n_sims = 8;
    // Standard models.
    let mut model_std = (0..n_sims)
        .map(|_| Model::new(p, ic, &mut rng))
        .collect::<Result<Vec<_>, _>>()?;
    // The example's CANVAS models.
    let mut model_canvas = (0..n_sims)
        .map(|_| AbstractModel::from_agents(agents.clone(), ext.clone(), &mut rng))
        .collect::<Result<Vec<_>, _>>()?;
    // The CANVAS models of the package.
    let mut model_canvas_2 = (0..n_sims)
        .map(|_| ModelCANVAS::new(p, ic, &mut rng))
        .collect::<Result<Vec<_>, _>>()?;

    // Run them.
    let T = 12;
    ensemblerun(&mut model_std, T, &mut rng, true, &NoShock);
    ensemblerun(&mut model_canvas, T, &mut rng, true, &NoShock);
    ensemblerun(&mut model_canvas_2, T, &mut rng, true, &NoShock);

    // The results.
    let data = [
        DataVector::from_models(&model_std),
        DataVector::from_models(&model_canvas),
        DataVector::from_models(&model_canvas_2),
    ];
    let ps = plot_data_vectors_from(&data, DEFAULT_TITLEFONT, &DEFAULT_QUANTITIES)?;
    common::save(&dir, "canvas.svg", &layout_svg(&ps, (3, 3), (600, 400))?)?;
    Ok(())
}
