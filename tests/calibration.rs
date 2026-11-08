//! Calibration, BeforeIT `src/utils/calibration.jl` and
//! `src/utils/_calibration_steady_state.jl`: the port against an independent
//! NumPy translation of the Julia code (`tests/data/calibration_reference.py`,
//! see `tests/data/README.md`) and against the data set STEADY_STATE2010Q1,
//! and the checks of BeforeIT `test/calibration_script.jl` and
//! `test/utils/zenodo_calibration.jl`.

#![allow(non_snake_case)]

mod common;

use beforeit::calibration::get_params_and_initial_conditions_steady_state;
use beforeit::{
    Calibration, CalibrationData, Date, ITALY_CALIBRATION, InitialConditions, Model, ModelGR,
    Parameters, Xoshiro, get_params_and_initial_conditions, get_valid_calibration_quarters,
};
use common::approx;
use serde_json::Value;

/// The entries of a JSON value, nested lists flattened, `null` as NaN.
fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Null => vec![f64::NAN],
        Value::Bool(b) => vec![f64::from(u8::from(*b))],
        Value::Number(x) => vec![x.as_f64().unwrap()],
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        _ => panic!("unexpected value {v}"),
    }
}

fn json(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

/// Relative tolerance against the NumPy translation. Julia, and so the port
/// and the translation, estimate the AR(1) processes with `rfvar3` and the
/// analytic `svd_mx2`, which works with the dot products of the regressors
/// and so squares their condition number (about 2700 for the log levels).
/// A difference in rounding anywhere, such as the order of the additions in
/// a dot product (BLAS in Julia and NumPy, one after the other here), then
/// changes the estimates in the tenth digit; the intercepts are also small
/// differences of large numbers. All other values agree to about 1e-12.
fn tolerance(key: &str) -> f64 {
    let estimate = key == "C"
        || ["alpha_", "beta_", "sigma_"]
            .iter()
            .any(|p| key.starts_with(p));
    if estimate { 1e-8 } else { 1e-11 }
}

/// Every value of `reference` agrees with the value of the same name in
/// `ours`. `ours` may hold in addition only the options that BeforeIT's
/// calibration does not return.
fn compare(case: &str, ours: &Value, reference: &Value) {
    let (ours, reference) = (ours.as_object().unwrap(), reference.as_object().unwrap());
    for key in ours.keys() {
        assert!(
            reference.contains_key(key)
                || [
                    "n_banks",
                    "omega_pi",
                    "overdrafts_in_capital_ratio",
                    "theta_calvo"
                ]
                .contains(&key.as_str()),
            "{case}: {key} is not in the reference"
        );
    }
    for (key, r) in reference {
        let o = ours
            .get(key)
            .unwrap_or_else(|| panic!("{case}: {key} is missing"));
        let (x, y) = (flat(o), flat(r));
        assert_eq!(x.len(), y.len(), "{case}: length of {key}");
        for (i, (&a, &b)) in x.iter().zip(&y).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || approx(a, b, tolerance(key), 0.0),
                "{case}: {key}[{i}] is {a}, the reference {b}"
            );
        }
    }
}

fn calibrate(date: &str, scale: f64, growth: bool) -> Calibration {
    get_params_and_initial_conditions(
        &ITALY_CALIBRATION,
        Date::parse(date).unwrap(),
        scale,
        growth,
    )
    .unwrap()
}

#[test]
fn italy_agrees_with_the_numpy_translation() {
    // 2019-12-31 is the last quarter of the time series, so the T = 12
    // quarters after it are missing: T_max = 0 and the histories C_G, C_E
    // and Y_I end in NaN.
    for (file, date, growth) in [
        ("calibration_italy_2010-03-31", "2010-03-31", false),
        ("calibration_italy_2019-12-31_growth", "2019-12-31", true),
    ] {
        let reference = common::load(file);
        let c = calibrate(date, 0.001, growth);
        compare(
            file,
            &json(&c.parameters.to_json()),
            &reference["parameters"],
        );
        compare(
            file,
            &json(&c.initial_conditions.to_json()),
            &reference["initial_conditions"],
        );
        let warnings: Vec<&str> = reference["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap())
            .collect();
        assert_eq!(c.warnings, warnings, "{file}");
    }
}

#[test]
fn italy_has_the_quarters_2010_to_2016() {
    let q = get_valid_calibration_quarters(&ITALY_CALIBRATION).unwrap();
    let expected: Vec<(i32, u32)> = (2010..=2016)
        .flat_map(|y| (1..=4).map(move |q| (y, q)))
        .collect();
    assert_eq!(q, expected);
}

#[test]
fn dates_without_data_are_errors() {
    // BeforeIT `test/utils/zenodo_calibration.jl`: a date that is too early.
    let co = &*ITALY_CALIBRATION;
    for date in [Date::new(1900, 1, 1), Date::new(2010, 2, 15)] {
        assert!(get_params_and_initial_conditions(co, date, 0.001, false).is_err());
    }
    assert!(Date::parse("2010-02-29").is_err());
    assert!(Date::parse("2012-02-29").is_ok());
}

#[test]
fn calibrated_models_run() {
    // BeforeIT `test/utils/zenodo_calibration.jl` builds a model from a
    // calibration; here the models also run.
    let rng = &mut Xoshiro::seed_from_u64(1);
    let c = calibrate("2016-12-31", 0.0005, false);
    let mut m = Model::new(&c.parameters, &c.initial_conditions, rng).unwrap();
    m.run(2, rng);
    assert!(m.data.real_gdp.iter().all(|x| x.is_finite()));
    let c = calibrate("2012-09-30", 0.0005, true);
    let mut m = ModelGR::new(&c.parameters, &c.initial_conditions, rng).unwrap();
    m.run(2, rng);
    assert!(m.data.real_gdp.iter().all(|x| x.is_finite()));
}

#[test]
fn calibrations_survive_json() {
    for (date, growth) in [("2010-03-31", false), ("2019-12-31", true)] {
        let c = calibrate(date, 0.001, growth);
        let p = Parameters::from_json(&c.parameters.to_json()).unwrap();
        assert_eq!(p, c.parameters, "{date}");
        let ic = InitialConditions::from_json(&c.initial_conditions.to_json()).unwrap();
        // NaN != NaN, so the 2019 histories are compared as text.
        assert_eq!(ic.to_json(), c.initial_conditions.to_json(), "{date}");
    }
}

/// STEADY_STATE2010Q1 was made with Julia's `round`, which rounds ties to
/// the even number; `_calibration_steady_state.jl` now uses `matlab_round`,
/// which rounds them up. The values agree except where a count of 0.001
/// times the data ends in .5, and the values computed from those counts.
#[test]
fn steady_state_differs_from_the_data_set_only_by_rounding() {
    let co = &*ITALY_CALIBRATION;
    let (p, ic) = get_params_and_initial_conditions_steady_state(
        Date::new(2010, 3, 31),
        &co.calibration,
        &co.figaro,
        &co.data,
        &co.ea,
        co.max_calibration_date,
        co.estimation_date,
        0.001,
    )
    .unwrap();
    let (p, ic) = (json(&p.to_json()), json(&ic.to_json()));
    let rp = json(include_str!("../data/steady_state/parameters/2010Q1.json"));
    let ric = json(include_str!(
        "../data/steady_state/initial_conditions/2010Q1.json"
    ));
    let rounded = ["alpha_s", "w_s", "J", "H_act", "N_s", "sb_other"];
    for (ours, reference) in [(&p, &rp), (&ic, &ric)] {
        for (key, r) in reference.as_object().unwrap() {
            if rounded.contains(&key.as_str()) {
                continue;
            }
            let (x, y) = (flat(&ours[key]), flat(r));
            assert_eq!(x.len(), y.len(), "{key}");
            for (a, b) in x.iter().zip(&y) {
                assert!(approx(*a, *b, 1e-12, 0.0), "{key}: {a} vs {b}");
            }
        }
    }
    // N_s: one more where the data set rounded a tie down to an even number.
    let (N, rN) = (flat(&ic["N_s"]), flat(&ric["N_s"]));
    let ties: Vec<usize> = (0..N.len()).filter(|&s| N[s] != rN[s]).collect();
    assert_eq!(ties, [11, 24, 42]);
    for &s in &ties {
        assert!(N[s] == rN[s] + 1.0 && rN[s] % 2.0 == 0.0, "N_s[{s}]");
    }
    // alpha_s and w_s are divided by N_s.
    for key in ["alpha_s", "w_s"] {
        let (x, y) = (flat(&p[key]), flat(&rp[key]));
        for s in 0..N.len() {
            assert!(approx(x[s] * N[s], y[s] * rN[s], 1e-12, 0.0), "{key}[{s}]");
        }
    }
    // J = round(sum(I_s) / 4), and sum(I_s) / 4 = 1166.5.
    let I_s: f64 = flat(&p["I_s"]).iter().sum();
    assert_eq!(I_s / 4.0, 1166.5);
    assert_eq!(
        (p["J"].as_f64(), rp["J"].as_f64()),
        (Some(1167.0), Some(1166.0))
    );
    // H_act counts the employees; sb_other is divided by H_act + H_inact.
    let H = |p: &Value| p["H_act"].as_f64().unwrap() + p["H_inact"].as_f64().unwrap();
    assert_eq!(H(&p) - H(&rp), ties.len() as f64);
    let sb = |ic: &Value| ic["sb_other"].as_f64().unwrap();
    assert!(approx(sb(&ic) * H(&p), sb(&ric) * H(&rp), 1e-12, 0.0));
}

/// BeforeIT `test/utils/zenodo_calibration.jl` on calibration objects of
/// the Zenodo record [`beforeit::calibration::ZENODO_REPOSITORY_URL`],
/// converted with `data/convert_calibration.py` to `XX.json` (XX the
/// country) in the folder `BEFOREIT_CALIBRATION_DIR`:
///
/// ```sh
/// curl -LO https://zenodo.org/records/18698434/files/AT_calibration_object.jld2
/// python3 data/convert_calibration.py zenodo AT_calibration_object.jld2 dir/AT.json
/// BEFOREIT_CALIBRATION_DIR=dir cargo test --test calibration -- --ignored
/// ```
#[test]
#[ignore = "needs converted Zenodo calibration objects in BEFOREIT_CALIBRATION_DIR"]
fn zenodo_calibration_objects() {
    let dir = std::env::var("BEFOREIT_CALIBRATION_DIR").expect("BEFOREIT_CALIBRATION_DIR");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .json files in {dir}");
    for path in files {
        let country = path.file_stem().unwrap().to_str().unwrap().to_string();
        let co = CalibrationData::from_json(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|e| panic!("{country}: {e}"));
        let quarters = get_valid_calibration_quarters(&co).unwrap();
        eprintln!("{country}: {} valid quarters", quarters.len());
        assert!(!quarters.is_empty(), "{country}");
        assert!(
            quarters
                .iter()
                .all(|&(y, q)| y >= 1995 && (1..=4).contains(&q))
        );
        // The most recent valid quarter.
        let &(y, q) = quarters.iter().max().unwrap();
        let month = 3 * q;
        let last = Date::new(y, month, if month == 3 || month == 12 { 31 } else { 30 });
        get_params_and_initial_conditions(&co, last, 1.0 / 1000.0, false)
            .unwrap_or_else(|e| panic!("{country} {last:?}: {e}"));
        // 2020-03-31, as the test for Austria.
        let c = get_params_and_initial_conditions(&co, Date::new(2020, 3, 31), 1.0 / 1000.0, false)
            .unwrap_or_else(|e| panic!("{country}: {e}"));
        let (p, ic) = (&c.parameters, &c.initial_conditions);
        assert!(p.rho > 0.0 && p.rho < 1.0, "{country}: rho = {}", p.rho);
        assert!(ic.omega > 0.0 && ic.omega < 1.0, "{country}");
        assert!(ic.D_I >= 0.0, "{country}");
        let rng = &mut Xoshiro::seed_from_u64(1);
        Model::new(p, ic, rng).unwrap_or_else(|e| panic!("{country}: {e}"));
        // A date that is too early.
        assert!(
            get_params_and_initial_conditions(&co, Date::new(1900, 1, 1), 0.001, false).is_err()
        );
    }
}

/// The index (from 1) of `date` in `data["quarters_num"]`.
fn quarter_index(co: &CalibrationData, date: Date) -> usize {
    co.data["quarters_num"]
        .data
        .iter()
        .position(|q| *q == Some(date.datenum()))
        .unwrap()
        + 1
}

/// Both calibrations, the full one without and with growth rates.
fn calibrations(co: &CalibrationData, date: Date) -> [Result<(), String>; 3] {
    let steady = get_params_and_initial_conditions_steady_state(
        date,
        &co.calibration,
        &co.figaro,
        &co.data,
        &co.ea,
        co.max_calibration_date,
        co.estimation_date,
        0.001,
    );
    [
        get_params_and_initial_conditions(co, date, 0.001, false).map(|_| ()),
        get_params_and_initial_conditions(co, date, 0.001, true).map(|_| ()),
        steady.map(|_| ()),
    ]
}

/// Where Julia throws in the estimates, also in those that the steady state
/// replaces, the calibrations return errors.
#[test]
fn errors_where_julia_throws_in_the_estimates() {
    let date = Date::new(2010, 3, 31);
    // An estimation date at or after the calibration date leaves fewer than
    // two values for the AR(1) estimates (Julia's svd_mx2 throws); one
    // quarter before, only the growth rates are too few.
    for (estimation_date, ok) in [
        (Date::new(2009, 12, 31), [true, false, true]),
        (date, [false; 3]),
        (Date::new(2011, 6, 30), [false; 3]),
    ] {
        let mut co = (*ITALY_CALIBRATION).clone();
        co.estimation_date = estimation_date;
        for (r, ok) in calibrations(&co, date).into_iter().zip(ok) {
            assert_eq!(r.is_ok(), ok, "{estimation_date:?}: {r:?}");
            if let Err(e) = r {
                assert!(e.contains("svd_mx2"), "{e}");
            }
        }
    }
    // The fourth root of a negative 1 + euribor at the calibration date,
    // and the logarithm of a negative real GDP of the euro area in the
    // quarter before the estimation date, with a negative nominal GDP so
    // that the deflator stays positive (Julia throws DomainErrors).
    let Tx = quarter_index(&ITALY_CALIBRATION, date);
    let Te = quarter_index(&ITALY_CALIBRATION, ITALY_CALIBRATION.estimation_date);
    for (table, key, i) in [
        ("data", "euribor", Tx),
        ("ea", "real_gdp_quarterly", Te - 1),
    ] {
        let mut co = (*ITALY_CALIBRATION).clone();
        let t = if table == "data" {
            &mut co.data
        } else {
            &mut co.ea
        };
        t.get_mut(key).unwrap().data[i - 1] = Some(-2.0);
        if key == "real_gdp_quarterly" {
            co.ea.get_mut("nominal_gdp_quarterly").unwrap().data[i - 1] = Some(-2.0);
        }
        for r in calibrations(&co, date) {
            let e = r.unwrap_err();
            assert!(
                e.contains("DomainError") && e.contains(&format!("[\"{key}\"]")),
                "{e}"
            );
        }
    }
    // Nominal and real GDP of different sizes (a DimensionMismatch).
    let mut co = (*ITALY_CALIBRATION).clone();
    co.ea.get_mut("real_gdp_quarterly").unwrap().data.pop();
    co.ea.get_mut("real_gdp_quarterly").unwrap().size =
        vec![co.ea["real_gdp_quarterly"].data.len()];
    for r in calibrations(&co, date) {
        assert!(r.unwrap_err().contains("DimensionMismatch"));
    }
}

/// A `missing` value that Julia reads only before the steady state replaces
/// it is no error, and the result is unchanged.
#[test]
fn steady_state_ignores_missing_values_it_replaces() {
    let date = Date::new(2010, 3, 31);
    let steady = |co: &CalibrationData| {
        get_params_and_initial_conditions_steady_state(
            date,
            &co.calibration,
            &co.figaro,
            &co.data,
            &co.ea,
            co.max_calibration_date,
            co.estimation_date,
            0.001,
        )
    };
    let reference = steady(&ITALY_CALIBRATION).unwrap();
    let mut co = (*ITALY_CALIBRATION).clone();
    for key in [
        "firm_interest",
        "interest_government_debt",
        "government_deficit",
    ] {
        // The first year, as the steady state takes it for this format.
        co.calibration.get_mut(key).unwrap().data[0] = None;
    }
    let Tx = quarter_index(&co, date);
    co.data.get_mut("euribor").unwrap().data[Tx - 1] = None;
    assert_eq!(steady(&co).unwrap(), reference);
    // The full calibration carries these values into its parameters.
    assert!(get_params_and_initial_conditions(&co, date, 0.001, false).is_err());
}
