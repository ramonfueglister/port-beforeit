//! A scenario by a shock, BeforeIT `examples/scenario_analysis_via_shock.jl`:
//! 64 runs of the model of Austria for 16 quarters as they are, and 64 in
//! which the propensity to consume is 2 % higher for the first quarters; the
//! ratio of the mean GDP of the two, with its standard error, is the impulse
//! response.
//!
//! ```sh
//! cargo run --release --features plots --example scenario_analysis_via_shock
//! ```
//!
//! The plot is written as SVG to
//! `target/example-output/scenario_analysis_via_shock`.

#![allow(non_snake_case)]

mod common;

use beforeit::plots::{DEFAULT_SIZE, Plot};
use beforeit::{AUSTRIA2010Q1, DataVector, Model, NoShock, Shock, Standard, Xoshiro, ensemblerun};

/// A shock that multiplies the labour productivity of every firm by
/// `productivity_multiplier`, each quarter in which it is applied (the
/// Julia example calls it permanent).
struct ProductivityShock {
    productivity_multiplier: f64,
}

impl Shock<Standard> for ProductivityShock {
    fn apply(&self, model: &mut Model) {
        for a in &mut model.firms.alpha_bar_i {
            *a *= self.productivity_multiplier;
        }
    }
}

/// A temporary change of the propensity to consume `prop.psi` by the
/// factor `consumption_multiplier`, from quarter 1 to `final_time`.
struct ConsumptionShock {
    consumption_multiplier: f64,
    final_time: usize,
}

impl Shock<Standard> for ConsumptionShock {
    fn apply(&self, model: &mut Model) {
        if model.agg.t == 1 {
            model.prop.psi *= self.consumption_multiplier;
        } else if model.agg.t == self.final_time {
            model.prop.psi /= self.consumption_multiplier;
        }
    }
}

/// The mean and the standard deviation of each row, StatsBase's `mean(x,
/// dims = 2)` and `std(x, dims = 2)`.
fn mean_std(rows: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    rows.iter()
        .map(|r| {
            let n = r.len() as f64;
            let m = r.iter().sum::<f64>() / n;
            let v = r.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
            (m, v.sqrt())
        })
        .unzip()
}

fn main() -> Result<(), String> {
    let dir = common::output_dir("scenario_analysis_via_shock");
    let mut rng = Xoshiro::seed_from_u64(1);
    let parameters = &AUSTRIA2010Q1.parameters;
    let initial_conditions = &AUSTRIA2010Q1.initial_conditions;

    // Initialise the model.
    let model = Model::new(parameters, initial_conditions, &mut rng)?;

    // The baseline: n_sims runs of T quarters from copies of the model.
    let T = 16;
    let n_sims = 64;
    let mut model_vec_baseline = vec![model.clone(); n_sims];
    ensemblerun(&mut model_vec_baseline, T, &mut rng, true, &NoShock);

    // Specific shocks: a 2 % increase in productivity,
    let _productivity_shock = ProductivityShock {
        productivity_multiplier: 1.02,
    };
    // or a 2 % increase in consumption for 4 quarters.
    let consumption_shock = ConsumptionShock {
        consumption_multiplier: 1.02,
        final_time: 4,
    };

    // The runs with the shock.
    let mut model_vec_shocked = vec![model; n_sims];
    ensemblerun(
        &mut model_vec_shocked,
        T,
        &mut rng,
        true,
        &consumption_shock,
    );

    // Mean and standard error of GDP for both.
    let gdp_baseline = DataVector::from_models(&model_vec_baseline).series(|d| &d.real_gdp);
    let gdp_shocked = DataVector::from_models(&model_vec_shocked).series(|d| &d.real_gdp);
    let (mean_gdp_baseline, std_baseline) = mean_std(&gdp_baseline);
    let (mean_gdp_shocked, std_shocked) = mean_std(&gdp_shocked);
    let sqrt_n = (n_sims as f64).sqrt();

    // The ratio of shocked to baseline GDP, and its standard error by the
    // propagation of errors.
    let mut gdp_ratio = Vec::new();
    let mut sem_gdp_ratio = Vec::new();
    for t in 0..=T {
        let ratio = mean_gdp_shocked[t] / mean_gdp_baseline[t];
        let (sem_b, sem_s) = (std_baseline[t] / sqrt_n, std_shocked[t] / sqrt_n);
        let relative =
            (sem_b / mean_gdp_baseline[t]).powi(2) + (sem_s / mean_gdp_shocked[t]).powi(2);
        gdp_ratio.push(ratio);
        sem_gdp_ratio.push(ratio * relative.powf(0.5));
    }

    // The impulse response.
    let mut p = Plot::new("");
    (p.x_label, p.y_label) = ("quarters".to_string(), "GDP change".to_string());
    let s = p.add_line(common::one_to(T + 1), gdp_ratio);
    s.ribbon = Some(sem_gdp_ratio);
    s.fill_alpha = 0.2;
    s.label = None;
    common::save(&dir, "gdp_shock.svg", &p.to_svg(DEFAULT_SIZE)?)?;
    Ok(())
}
