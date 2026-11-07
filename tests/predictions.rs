//! Predictions, BeforeIT `src/utils/get_predictions_from_sims.jl`, against
//! values of Julia: BeforeIT `test/deterministic/prediction_pipeline.jl`
//! calibrates the model on `ITALY_CALIBRATION` for 2010Q1 at scale 0.0001,
//! runs two copies without randomness for 12 quarters with `ensemblerun!`
//! and compares the predictions with `test/deterministic/2010Q1.jld2`
//! (`tests/data/prediction_pipeline_2010Q1.json`, see
//! `tests/data/README.md`). The comparison covers the calibration, the
//! model and the predictions together.

#![allow(non_snake_case)]

mod common;

use beforeit::julia::{max, round};
use beforeit::predictions::Matrix;
use beforeit::{
    AbstractModel, Aggregates, Banks, CentralBank, Data, DataVector, Date, Deterministic,
    Extension, Firms, Government, ITALY_CALIBRATION, Model, NoShock, Predictions, Properties,
    RestOfTheWorld, Workers, base, ensemblerun, get_params_and_initial_conditions,
    get_predictions_from_sims,
};
use common::{approx_list, load, matrix};
use serde_json::Value;

/// Julia `isequal` for floats: NaN equals NaN, −0.0 differs from 0.0.
fn isequal(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// Julia `isless` for floats: NaN is the largest number, −0.0 < 0.0.
fn isless(a: f64, b: f64) -> bool {
    if a.is_nan() {
        false
    } else if b.is_nan() {
        true
    } else if a == b {
        a.is_sign_negative() && b.is_sign_positive()
    } else {
        a < b
    }
}

/// Julia `min(x, y)` of two vectors of the same length:
/// `isless(y, x) ? y : x`, where `isless` compares the vectors
/// lexicographically, so the whole of one vector is the result.
fn julia_vector_min<'a>(x: &'a [f64], y: &'a [f64]) -> &'a [f64] {
    match x.iter().zip(y).find(|&(&a, &b)| !isequal(a, b)) {
        Some((&a, &b)) if isless(b, a) => y,
        _ => x,
    }
}

/// The model type of BeforeIT's `Model` with A.14 and A.22 exactly as in
/// the Julia code, `min(Q_s_i, firms.K_i .* firms.kappa_i)` of the whole
/// vectors, which the port replaces by the minimum firm by firm (README,
/// "Differences from BeforeIT.jl", item 1). Julia's reference values come
/// from this code.
#[derive(Clone)]
struct JuliaMin;

impl Extension for JuliaMin {
    fn desired_capital_material_employment(
        firms: &Firms,
        Q_s_i: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let f = firms;
        let capacity: Vec<f64> = f.K_i.iter().zip(&f.kappa_i).map(|(k, c)| k * c).collect();
        let m = julia_vector_min(Q_s_i, &capacity);
        let I_d_i = (0..f.len())
            .map(|i| f.delta_i[i] / f.kappa_i[i] * m[i])
            .collect();
        // A.18 takes the minimum firm by firm in Julia as well.
        let (_, DM_d_i, _) = base::desired_capital_material_employment(firms, Q_s_i);
        let N_d_i = (0..f.len())
            .map(|i| max(1.0, round(m[i] / f.alpha_bar_i[i])))
            .collect();
        (I_d_i, DM_d_i, N_d_i)
    }
}

const T: usize = 12;
const N_SIMS: usize = 2;

fn calibration_date() -> Date {
    Date::new(2010, 3, 31)
}

/// `get_predictions_from_sims` after the steps of
/// `prediction_pipeline.jl`, for the model type `X` built by `new`.
fn pipeline<X: Extension + Clone + Send>(
    new: impl Fn(&beforeit::Parameters, &beforeit::InitialConditions) -> AbstractModel<X>,
) -> Predictions {
    let mut cal = ITALY_CALIBRATION.clone();
    let c = get_params_and_initial_conditions(&cal, calibration_date(), 0.0001, false).unwrap();
    let model = new(&c.parameters, &c.initial_conditions);
    let mut models = vec![model; N_SIMS];
    ensemblerun(&mut models, T, &mut Deterministic, false, &NoShock);
    // Julia's calibration has stored the GDP deflators in
    // ITALY_CALIBRATION.data, which the predictions read.
    cal.set_gdp_deflators().unwrap();
    get_predictions_from_sims(
        &DataVector::from_models(&models),
        &cal.data,
        calibration_date(),
    )
    .unwrap()
}

fn julia_min_model(
    p: &beforeit::Parameters,
    ic: &beforeit::InitialConditions,
) -> AbstractModel<JuliaMin> {
    let rng = &mut Deterministic;
    let prop = Properties::new(p, ic).unwrap();
    let firms = Firms::new(p, ic, rng).unwrap();
    let (w_act, w_inact) = Workers::new(p, ic);
    let agents = (
        w_act,
        w_inact,
        firms,
        Banks::new(p, ic),
        CentralBank::new(p, ic),
        Government::new(p, ic),
        RestOfTheWorld::new(p, ic),
        Aggregates::new(p, ic),
        prop,
        Data::default(),
    );
    AbstractModel::from_agents(agents, JuliaMin, rng).unwrap()
}

fn standard_model(p: &beforeit::Parameters, ic: &beforeit::InitialConditions) -> Model {
    Model::new(p, ic, &mut Deterministic).unwrap()
}

fn flat(m: &Matrix) -> Vec<f64> {
    m.iter().flatten().copied().collect()
}

/// The reference as matrices, `quarters_num` and `years_num` as columns.
fn reference() -> Vec<(String, Matrix)> {
    let v = load("prediction_pipeline_2010Q1");
    let o = v.as_object().unwrap();
    o.iter()
        .map(|(k, x)| {
            let m = match x {
                Value::Array(a) if a.first().is_some_and(Value::is_number) => {
                    a.iter().map(|y| vec![y.as_f64().unwrap()]).collect()
                }
                _ => matrix(&v, k),
            };
            (k.clone(), m)
        })
        .collect()
}

fn predicted(p: &Predictions, key: &str) -> Matrix {
    match key {
        "quarters_num" => p.quarters_num.iter().map(|&x| vec![x]).collect(),
        "years_num" => p.years_num.iter().map(|&x| vec![x]).collect(),
        _ => p
            .series
            .get(key)
            .unwrap_or_else(|| panic!("the port has no prediction \"{key}\""))
            .clone(),
    }
}

/// The largest deviation of an entry from the reference, relative to the
/// value or, for numbers below 1 such as growth rates, absolute:
/// `|x − r| / max(1, |x|, |r|)`; and its key.
fn largest_deviation(p: &Predictions, reference: &[(String, Matrix)]) -> (f64, String) {
    let mut worst = (0.0, String::new());
    for (key, r) in reference {
        let x = predicted(p, key);
        assert_eq!(x.len(), r.len(), "{key}: number of rows");
        for (a, b) in flat(&x).iter().zip(flat(r)) {
            let d = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
            if d.is_nan() || d > worst.0 {
                worst = (d, key.clone());
            }
        }
    }
    worst
}

fn reference_of(reference: &[(String, Matrix)], key: &str) -> Matrix {
    reference.iter().find(|(k, _)| k == key).unwrap().1.clone()
}

#[test]
fn predictions_agree_with_julia() {
    let reference = reference();
    assert_eq!(reference.len(), 107);
    let p = pipeline(julia_min_model);
    assert_eq!(p.series.len() + 2, reference.len(), "number of predictions");
    for (key, r) in &reference {
        let x = predicted(&p, key);
        assert_eq!(x.len(), r.len(), "{key}: number of rows");
        assert!(
            x.iter().zip(r).all(|(a, b)| a.len() == b.len()),
            "{key}: number of runs"
        );
        // BeforeIT prediction_pipeline.jl: isapprox(atol = 1e-6, rtol = 1e-6).
        assert!(
            approx_list(&flat(&x), &flat(r), 1e-6, 1e-6),
            "{key}: fails Julia's test"
        );
    }
    assert_eq!(
        predicted(&p, "quarters_num"),
        reference_of(&reference, "quarters_num")
    );
    assert_eq!(
        predicted(&p, "years_num"),
        reference_of(&reference, "years_num")
    );
    // Far closer than Julia's test: the largest deviation is 6.3e-12 of a
    // level and 2.9e-12 of a growth rate. Julia and the port round some
    // intermediate results differently, for example the dot products of
    // the calibration's estimates, which Julia leaves to BLAS.
    let (worst, key) = largest_deviation(&p, &reference);
    assert!(worst <= 1e-11, "{key}: deviation {worst:e}");
}

/// With the port's minimum firm by firm in A.14 and A.22, the model differs
/// from the Julia code, and so do the predictions: by up to 3.2e-5 of a
/// value, more than Julia's test allows for 60 of the 107 keys.
#[test]
fn the_minimum_firm_by_firm_changes_the_predictions() {
    let reference = reference();
    let p = pipeline(standard_model);
    let (worst, _) = largest_deviation(&p, &reference);
    assert!(worst > 1e-6 && worst < 1e-4, "deviation {worst:e}");
    let fails = reference
        .iter()
        .filter(|(key, r)| !approx_list(&flat(&predicted(&p, key)), &flat(r), 1e-6, 1e-6))
        .count();
    assert_eq!(fails, 60);
}
