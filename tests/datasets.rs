//! The data sets of BeforeIT.jl and reading parameters and initial
//! conditions from JSON.

#![allow(non_snake_case)]

use beforeit::{
    AUSTRIA2010Q1, ITALY2010Q1, InitialConditions, InitialState, Model, Parameters,
    STEADY_STATE2010Q1, Xoshiro,
};
use serde_json::{Value, json};

const AUSTRIA_PARAMETERS: &str = include_str!("../data/austria/parameters/2010Q1.json");
const AUSTRIA_IC: &str = include_str!("../data/austria/initial_conditions/2010Q1.json");

#[test]
fn every_data_set_runs() {
    for (name, s) in [
        ("AUSTRIA2010Q1", &*AUSTRIA2010Q1),
        ("ITALY2010Q1", &*ITALY2010Q1),
        ("STEADY_STATE2010Q1", &*STEADY_STATE2010Q1),
    ] {
        let rng = &mut Xoshiro::seed_from_u64(1);
        let mut m = Model::new(&s.parameters, &s.initial_conditions, rng)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        m.run(1, rng);
        assert_eq!(m.agg.t, 2, "{name}");
        assert!(m.data.real_gdp[1].is_finite(), "{name}");
    }
}

#[test]
fn data_sets_have_the_values_of_the_files() {
    let a = &AUSTRIA2010Q1.parameters;
    assert_eq!(
        (a.G, a.T_prime, a.H_act, a.H_inact, a.J, a.L),
        (62, 54, 4743, 4130, 156, 312)
    );
    assert_eq!(a.a_sg.len(), 62);
    // The optional parameters are missing in all three files.
    assert_eq!((a.n_banks, a.omega_pi, a.theta_calvo), (1, 0.0, 1.0));
    assert!(!a.overdrafts_in_capital_ratio);
    let i = &ITALY2010Q1;
    assert_eq!((i.parameters.H_act, i.parameters.H_inact), (11702, 18214));
    let ic = &i.initial_conditions;
    for series in [&ic.r_bar_series, &ic.Y_EA_series, &ic.pi_EA_series] {
        assert_eq!(series.len(), 54);
    }
    assert!(AUSTRIA2010Q1.initial_conditions.r_bar_series.is_empty());
    let s = &STEADY_STATE2010Q1.parameters;
    assert_eq!((s.H_act, s.H_inact, s.J, s.L), (23397, 36428, 1166, 2333));
}

fn with(text: &str, change: impl FnOnce(&mut Object)) -> String {
    let mut v: Value = serde_json::from_str(text).unwrap();
    change(v.as_object_mut().unwrap());
    v.to_string()
}

type Object = serde_json::Map<String, Value>;

fn parameters(change: impl FnOnce(&mut Object)) -> Result<Parameters, String> {
    Parameters::from_json(&with(AUSTRIA_PARAMETERS, change))
}

#[test]
fn reading_follows_the_julia_dictionaries() {
    let q = parameters(|o| {
        o.insert("n_banks".into(), json!(3));
        o.insert("omega_pi".into(), json!(0.5));
        o.insert("theta_calvo".into(), json!(0.25));
        o.insert("overdrafts_in_capital_ratio".into(), json!(true));
    })
    .unwrap();
    assert_eq!((q.n_banks, q.omega_pi, q.theta_calvo), (3, 0.5, 0.25));
    assert!(q.overdrafts_in_capital_ratio);
    // Julia `Bool(1.0)`.
    let q = parameters(|o| {
        o.insert("overdrafts_in_capital_ratio".into(), json!(1.0));
    })
    .unwrap();
    assert!(q.overdrafts_in_capital_ratio);

    let error = |key: &str, value: Option<Value>| {
        parameters(|o| {
            match value {
                Some(v) => o.insert(key.into(), v),
                None => o.remove(key),
            };
        })
        .unwrap_err()
    };
    assert_eq!(error("tau_VAT", None), "tau_VAT is missing");
    assert_eq!(error("G", Some(json!(62.5))), "G = 62.5 is not a count");
    assert_eq!(
        error("C", Some(json!([[1.0, 0.0], [0.0, 1.0]]))),
        "C is not a 3 × 3 matrix"
    );
    assert_eq!(
        error("overdrafts_in_capital_ratio", Some(json!(0.5))),
        "overdrafts_in_capital_ratio = 0.5 is not a Bool"
    );
    assert_eq!(error("a_sg", Some(json!("x"))), "a_sg is not a matrix");
    assert_eq!(
        Parameters::from_json("[1, 2]").unwrap_err(),
        "parameters: not a JSON object"
    );

    // Julia `vec`: a column matrix is read as a list.
    let ic = InitialConditions::from_json(&with(AUSTRIA_IC, |o| {
        let y: Vec<Value> = o["Y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| json!([x]))
            .collect();
        o.insert("Y".into(), Value::Array(y));
    }))
    .unwrap();
    assert_eq!(ic.Y, AUSTRIA2010Q1.initial_conditions.Y);

    let s = InitialState::from_json(AUSTRIA_PARAMETERS, AUSTRIA_IC).unwrap();
    assert_eq!(s, *AUSTRIA2010Q1);
}
