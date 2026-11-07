//! The batch runs and the forecast error tables, BeforeIT
//! `src/utils/save_all_predictions.jl`, `src/utils/error_tables.jl` and
//! `src/utils/analysis_utils.jl`, run as in BeforeIT's example
//! `examples/analysis/tabs/simulate_and_create_tables.jl`: parameters,
//! simulations and predictions for the eight quarters 2010Q1 to 2011Q4 of
//! `ITALY_CALIBRATION` (scale 0.0001, 2 runs of 12 quarters) for the base
//! model and the growth-rate model, then all error tables.
//!
//! The tables are compared with those of an independent translation of the
//! Julia code into Python (`tests/data/analysis_reference.py`, see
//! `tests/data/README.md`), which reads the same predictions. The
//! predictions are those of this port; the file of reference values holds
//! the series that the tables use, and the test checks that the run
//! reproduces them.
//!
//! The functions read and write files relative to the working directory,
//! which is shared by the whole process, so this file holds a single test.
//! It works in `target/tmp/analysis`, which it leaves in place for the
//! reference script.

#![allow(non_snake_case)]

use std::collections::BTreeMap;
use std::path::Path;

use beforeit::analysis::{
    FORECAST_HORIZONS, build_12q_matrices, discover_countries, discover_countries_of_variant,
    error_table_abm, error_table_ar, error_table_validation_abm, error_table_validation_var,
    load_country_forecast_data, save_all_params_and_initial_conditions,
    save_all_predictions_from_sims, save_all_simulations,
};
use beforeit::calibration::table_from_json;
use beforeit::{Date, ITALY_CALIBRATION, Model, ModelGR, Xoshiro};
use serde_json::Value;

const VARIANTS: [&str; 2] = ["Model", "ModelGR"];

/// The series of the predictions that the error tables read.
const PREDICTED_KEYS: [&str; 11] = [
    "quarters_num",
    "real_gdp_quarterly",
    "gdp_deflator_growth_quarterly",
    "real_household_consumption_quarterly",
    "real_fixed_capitalformation_quarterly",
    "euribor",
    "real_government_consumption_quarterly",
    "real_exports_quarterly",
    "real_imports_quarterly",
    "real_gdp_ea_quarterly",
    "gdp_deflator_growth_ea_quarterly",
];

fn names(dir: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The numbers of a JSON list of numbers or strings "NaN".
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn same(a: f64, b: f64, rtol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= rtol * a.abs().max(b.abs())
}

/// The cells of a CSV text without quotes, which the tables do not need.
fn cells(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect()
}

/// Compares a table of p-values with its reference: the same labels, and
/// numbers within 1e-8 of each other. The p-values are written with all
/// their digits, and the two computations round differently: the AR
/// benchmark of the Euribor regresses a nearly constant series,
/// `(1 + r)^(1/4)`, on its lag and a constant, an ill-conditioned least
/// squares problem that this port solves with an SVD (BeforeIT's `rfvar3`)
/// and NumPy with LAPACK's `gelsd`. The forecasts agree to 6.2e-13, the
/// p-values to 7.2e-10.
fn compare_pvalues(name: &str, actual: &str, expected: &str) {
    let (a, e) = (cells(actual), cells(expected));
    assert_eq!(a.len(), e.len(), "{name}: number of lines");
    assert_eq!(a[0], e[0], "{name}: header");
    for (ra, re) in a[1..].iter().zip(&e[1..]) {
        assert_eq!(ra[0], re[0], "{name}: horizon");
        assert_eq!(ra.len(), re.len(), "{name}: number of cells");
        for (x, y) in ra[1..].iter().zip(&re[1..]) {
            let (x, y): (f64, f64) = (x.parse().unwrap(), y.parse().unwrap());
            let close = (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-8;
            assert!(close, "{name}: {x} and {y}");
        }
    }
}

#[test]
fn batch_runs_and_error_tables() {
    let reference: Value = serde_json::from_str(&read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/analysis_reference.json"
    )))
    .unwrap();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("analysis");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_current_dir(&dir).unwrap();

    // The data as BeforeIT's calibration leaves them.
    let mut cal = ITALY_CALIBRATION.clone();
    cal.set_gdp_deflators().unwrap();
    let (first, last) = (Date::new(2010, 3, 31), Date::new(2011, 12, 31));
    let quarter_files: Vec<String> = ["2010", "2011"]
        .iter()
        .flat_map(|y| (1..=4).map(move |q| format!("{y}Q{q}.json")))
        .collect();

    // Parameters and initial conditions; a second call starts afresh.
    save_all_params_and_initial_conditions(&cal, "data/it", 0.0001, first, last).unwrap();
    std::fs::write("data/it/stale.txt", "").unwrap();
    save_all_params_and_initial_conditions(&cal, "data/it", 0.0001, first, last).unwrap();
    assert!(!Path::new("data/it/stale.txt").exists());
    assert_eq!(names("data/it/parameters"), quarter_files);
    assert_eq!(names("data/it/initial_conditions"), quarter_files);

    // Simulations and predictions of both models.
    let mut rng = Xoshiro::seed_from_u64(2023);
    save_all_simulations(
        "data/it",
        12,
        2,
        Model::new,
        Some("simulations/Model"),
        &mut rng,
    )
    .unwrap();
    save_all_simulations(
        "data/it",
        12,
        2,
        ModelGR::new,
        Some("simulations/ModelGR"),
        &mut rng,
    )
    .unwrap();
    // Quarters that fail are skipped.
    save_all_simulations(
        "data/it",
        12,
        2,
        |_, _, _| Err::<Model, _>("no model".to_string()),
        Some("simulations/failing"),
        &mut rng,
    )
    .unwrap();
    assert!(names("data/it/simulations/failing").is_empty());
    for variant in VARIANTS {
        assert_eq!(
            names(&format!("data/it/simulations/{variant}")),
            quarter_files
        );
        save_all_predictions_from_sims(
            "data/it",
            &cal.data,
            &format!("simulations/{variant}"),
            &format!("abm_predictions/{variant}"),
        )
        .unwrap();
        assert_eq!(
            names(&format!("data/it/abm_predictions/{variant}")),
            quarter_files
        );
    }
    assert_eq!(
        discover_countries(Path::new("data"), "abm_predictions/Model").unwrap(),
        ["it"]
    );
    assert_eq!(
        discover_countries_of_variant("ModelGR", Path::new("data")).unwrap(),
        ["it"]
    );
    // BeforeIT test/utils/analysis_utils.jl, "discover_countries".
    assert!(
        discover_countries(
            Path::new("/tmp/__beforeit_nonexistent__"),
            "abm_predictions/Model"
        )
        .unwrap()
        .is_empty()
    );

    // The predictions that the reference script read.
    let predictions = reference["predictions"].as_object().unwrap();
    assert_eq!(predictions.len(), 2 * quarter_files.len());
    for (name, expected) in predictions {
        let t = table_from_json(&read(&format!("data/it/abm_predictions/{name}.json"))).unwrap();
        for key in PREDICTED_KEYS {
            let a: Vec<f64> = t[key].data.iter().map(|x| x.unwrap()).collect();
            let e = numbers(&expected[key]["data"]);
            assert_eq!(a.len(), e.len(), "{name} {key}");
            for (x, y) in a.iter().zip(&e) {
                assert!(same(*x, *y, 1e-12), "{name} {key}: {x} and {y}");
            }
        }
    }

    // All error tables, as in the example.
    let quarters = Date::every_months(first, last, 3);
    let (data, ea) = (&cal.data, &cal.ea);
    for variant in VARIANTS {
        let folder = format!("abm_predictions/{variant}");
        error_table_ar("it", ea, data, &quarters, &FORECAST_HORIZONS, variant).unwrap();
        error_table_abm(
            "it",
            ea,
            data,
            &quarters,
            &FORECAST_HORIZONS,
            variant,
            &folder,
        )
        .unwrap();
        error_table_validation_var("it", ea, data, &quarters, &FORECAST_HORIZONS, variant).unwrap();
        error_table_validation_abm(
            "it",
            ea,
            data,
            &quarters,
            &FORECAST_HORIZONS,
            variant,
            &folder,
        )
        .unwrap();
    }

    // The tables against the reference: the same files, the same text
    // except for the unrounded p-values.
    let tables = reference["tables"].as_object().unwrap();
    let mut written = BTreeMap::new();
    for variant in VARIANTS {
        let dir = format!("data/it/analysis/{variant}");
        for f in names(&dir) {
            if f.ends_with(".csv") || f.ends_with(".tex") {
                written.insert(format!("{variant}/{f}"), read(&format!("{dir}/{f}")));
            }
        }
    }
    assert_eq!(
        written.keys().collect::<Vec<_>>(),
        tables.keys().collect::<Vec<_>>()
    );
    for (name, actual) in &written {
        let expected = tables[name].as_str().unwrap();
        if name.contains("/pval_") {
            compare_pvalues(name, actual, expected);
        } else {
            assert_eq!(actual, expected, "{name}");
        }
    }
    // The growth-rate model is compared with the base model.
    assert!(written.contains_key("ModelGR/rmse_validation_abm_vs_base.tex"));

    // Gathering the tables of the countries.
    let r = load_country_forecast_data("Model", "rmse_abm_vs_ar", &["it", "xx"]).unwrap();
    assert_eq!(r.valid_countries, ["it"]);
    assert!(r.has_pvals && r.pval_data.contains_key("it"));
    let (m, p) = build_12q_matrices(&r, "1q").unwrap();
    let row = |file: &str| -> Vec<f64> {
        cells(&read(&format!("data/it/analysis/Model/{file}")))[1][1..]
            .iter()
            .map(|x| x.parse().unwrap())
            .collect()
    };
    let (rm, rp) = (row("rmse_abm_vs_ar.csv"), row("pval_abm_vs_ar.csv"));
    for j in 0..5 {
        assert!(same(m[0][j], rm[j], 0.0) && same(p[0][j], rp[j], 0.0));
    }
    // No row "12q": the forecasts of 12 quarters end after 2011.
    let (m, _) = build_12q_matrices(&r, "12q").unwrap();
    assert!(m[0].iter().all(|x| x.is_nan()));
    // A table of absolute RMSE with a negative value is skipped.
    std::fs::create_dir_all("data/neg/analysis/Model").unwrap();
    std::fs::write(
        "data/neg/analysis/Model/rmse_abm.csv",
        "Horizon,Real GDP\n1q,-1.0\n",
    )
    .unwrap();
    assert!(load_country_forecast_data("Model", "rmse_abm", &["neg"]).is_none());
    let r = load_country_forecast_data("Model", "rmse_abm", &["neg", "it"]).unwrap();
    assert_eq!(r.valid_countries, ["it"]);
    assert!(!r.has_pvals);
}
