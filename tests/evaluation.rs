//! Forecast evaluation: BeforeIT `src/utils/dmtest.jl`, `mztest.jl`,
//! `bias_ttest.jl`, `varx.jl` and `correlation_utils.jl`, against an
//! independent NumPy/SciPy translation (`tests/data/evaluation_reference.py`,
//! see `tests/data/README.md`), and the checks of BeforeIT
//! `test/utils/correlation_utils.jl`.

#![allow(non_snake_case)]

mod common;

use std::collections::BTreeMap;

use beforeit::calibration::{Array, Table, table_to_json};
use beforeit::correlation::{
    auto_correlation, correlation_stats, correlation_stats_folder, cross_correlation, hpfilter,
};
use beforeit::evaluation::{bias_ttest, dmtest_modified, mztest};
use beforeit::varx::{estimate_VARX, forecast_k_steps_VAR, forecast_k_steps_VARX};
use common::{approx, list, load, matrix};
use serde_json::Value;

fn close(name: &str, x: f64, y: f64, rtol: f64, atol: f64) {
    assert!(
        approx(x, y, rtol, atol),
        "{name}: {x} instead of {y} (rtol {rtol}, atol {atol})"
    );
}

fn close_all(name: &str, x: &[f64], y: &[f64], rtol: f64, atol: f64) {
    assert_eq!(x.len(), y.len(), "{name}: length");
    for (i, (a, b)) in x.iter().zip(y).enumerate() {
        close(&format!("{name}[{i}]"), *a, *b, rtol, atol);
    }
}

fn pair(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn tests_of_forecasts_agree_with_scipy() {
    let r = load("evaluation_reference");
    let (e1, e2) = (list(&r["dm"], "e1"), list(&r["dm"], "e2"));
    for (h, want) in r["dm"]["results"].as_object().unwrap() {
        let (dm, p) = dmtest_modified(&e1, &e2, h.parse().unwrap()).unwrap();
        let want = pair(want);
        close(&format!("DM h={h}"), dm, want[0], 1e-12, 0.0);
        close(&format!("DM p h={h}"), p, want[1], 1e-12, 0.0);
    }
    let errors = list(&r["bias"], "errors");
    for (h, want) in r["bias"]["results"].as_object().unwrap() {
        let (t, p) = bias_ttest(&errors, h.parse().unwrap());
        let want = pair(want);
        close(&format!("t h={h}"), t, want[0], 1e-12, 0.0);
        close(&format!("t p h={h}"), p, want[1], 1e-12, 0.0);
    }
    for key in ["mz", "mz_unbiased"] {
        let (c, beta, p) = mztest(&list(&r[key], "y"), &list(&r[key], "x")).unwrap();
        let want = list(&r[key], "result");
        close(&format!("{key} c"), c, want[0], 1e-12, 0.0);
        close(&format!("{key} beta"), beta, want[1], 1e-12, 0.0);
        // 1 − cdf loses the digits of a small p-value in both.
        close(&format!("{key} p"), p, want[2], 1e-12, 1e-15);
    }
}

#[test]
fn errors_where_julia_throws() {
    assert!(dmtest_modified(&[1.0, 2.0], &[1.0], 1).is_err());
    // h > n: an autocovariance of an empty vector.
    assert!(dmtest_modified(&[1.0, 2.0, 3.0], &[0.5, 0.1, 0.2], 5).is_err());
    assert!(mztest(&[1.0, 2.0], &[1.0]).is_err());
    assert!(mztest(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]).is_err());
}

#[test]
fn filters_and_correlations_agree_with_numpy() {
    let r = load("evaluation_reference");
    let (trend, cycle) = hpfilter(&list(&r["hp"], "y"), 1600.0).unwrap();
    close_all("trend", &trend, &list(&r["hp"], "trend"), 1e-12, 0.0);
    close_all("cycle", &cycle, &list(&r["hp"], "cycle"), 0.0, 1e-10);
    let (x, y) = (list(&r["corr"], "x"), list(&r["corr"], "y"));
    close_all(
        "cross",
        &cross_correlation(&x, &y, 5).unwrap(),
        &list(&r["corr"], "cross"),
        1e-13,
        1e-15,
    );
    close_all(
        "auto",
        &auto_correlation(&x, 0..=8),
        &list(&r["corr"], "auto"),
        1e-13,
        1e-15,
    );

    let c = &r["corr_stats"];
    let vector = |key: &str| Array::vector(&list(c, key));
    let data: Table = [
        ("real_gdp_quarterly".to_string(), vector("gdp")),
        ("wages_quarterly".to_string(), vector("wages")),
    ]
    .into();
    let sims: Table = [
        (
            "real_gdp".to_string(),
            Array::matrix(&matrix(c, "sims_gdp")),
        ),
        (
            "wages_quarterly".to_string(),
            Array::matrix(&matrix(c, "sims_wages")),
        ),
    ]
    .into();
    for (table, key) in [(&data, "data"), (&sims, "sims")] {
        let s = correlation_stats(table, &["wages_quarterly"], "real_gdp_quarterly", 5, 8).unwrap();
        let want = &c[key];
        let flat = |m: &Vec<Vec<f64>>| m.iter().flatten().copied().collect::<Vec<_>>();
        close_all(
            &format!("{key} crosscor"),
            &flat(&s.crosscor["wages_quarterly"]),
            &flat(&matrix(want, "crosscor")),
            0.0,
            1e-9,
        );
        close_all(
            &format!("{key} autocor"),
            &flat(&s.autocor["wages_quarterly"]),
            &flat(&matrix(want, "autocor")),
            0.0,
            1e-9,
        );
        close_all(
            &format!("{key} volatility"),
            &s.volatility["wages_quarterly"],
            &list(want, "volatility"),
            1e-13,
            0.0,
        );
    }
}

#[test]
fn var_forecasts_agree_with_numpy() {
    let r = load("evaluation_reference");
    let y = matrix(&r["var"], "y");
    for (key, want) in r["var"]["forecasts"].as_object().unwrap() {
        let (intercept, lags) = key.split_once('_').unwrap();
        let f = forecast_k_steps_VAR(&y, 8, intercept == "True", lags.parse().unwrap(), None);
        let want: Vec<Vec<f64>> = want.as_array().unwrap().iter().map(pair).collect();
        for (t, (a, b)) in f.iter().zip(&want).enumerate() {
            close_all(&format!("VAR {key} step {t}"), a, b, 1e-12, 1e-14);
        }
    }
    let exo = matrix(&r["var"], "exo");
    let f = forecast_k_steps_VARX(&y, &exo, 8, true, 1, None).unwrap();
    let want = matrix(&r["var"], "varx");
    for (t, (a, b)) in f.iter().zip(&want).enumerate() {
        close_all(&format!("VARX step {t}"), a, b, 1e-12, 1e-14);
    }
    // Without intercept, Julia takes the first exogenous coefficient as
    // `beta` and throws in the forecast.
    let (_, beta, gamma, _, _) = estimate_VARX(&y, &exo[1..61], false, 1).unwrap();
    assert_eq!(beta.len(), 2);
    assert!(gamma.iter().all(|r| r.is_empty()));
    assert!(forecast_k_steps_VARX(&y, &exo, 8, false, 1, None).is_err());
}

// BeforeIT test/utils/correlation_utils.jl.

#[test]
fn hpfilter_as_in_the_julia_tests() {
    let y = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0];
    let (trend, cycle) = hpfilter(&y, 1600.0).unwrap();
    let sum: Vec<f64> = trend.iter().zip(&cycle).map(|(t, c)| t + c).collect();
    assert!(common::approx_list(&sum, &y, common::SQRT_EPS, 0.0));
    let linear: Vec<f64> = (1..=20).map(f64::from).collect();
    let (_, linear_cycle) = hpfilter(&linear, 1600.0).unwrap();
    assert!(linear_cycle.iter().all(|c| c.abs() < 1.0e-6));
}

#[test]
fn auto_and_cross_correlation_as_in_the_julia_tests() {
    let x: Vec<f64> = (1..=30)
        .map(|t| f64::from(t) + f64::from(t).sin())
        .collect();
    let y: Vec<f64> = (1..=30).map(f64::from).collect();
    let ac = auto_correlation(&x, 0..=5);
    assert!(approx(ac[0], 1.0, common::SQRT_EPS, 0.0));
    assert_eq!(ac.len(), 6);
    assert!(cross_correlation(&x, &y[..29], 3).is_err());
    assert_eq!(cross_correlation(&x, &y, 5).unwrap().len(), 11);
    let L = 5;
    let xc = cross_correlation(&x, &x, L).unwrap();
    assert!(approx(xc[L], 1.0, common::SQRT_EPS, 0.0));
    assert!(common::approx_list(
        &xc[L..],
        &auto_correlation(&x, 0..=L),
        common::SQRT_EPS,
        0.0
    ));
    let mut rev = xc[L + 1..].to_vec();
    rev.reverse();
    assert!(common::approx_list(&xc[..L], &rev, common::SQRT_EPS, 0.0));
}

fn gdp_and_wages(T: usize) -> (Vec<f64>, Vec<f64>) {
    let gdp: Vec<f64> = (1..=T).map(|t| t as f64 + (t as f64).sin()).collect();
    let wages = (1..=T)
        .map(|t| gdp[t - 1] + 0.3 * (t as f64).cos())
        .collect();
    (gdp, wages)
}

fn repeat(v: &[f64], n: usize) -> Array {
    Array::matrix(&v.iter().map(|&x| vec![x; n]).collect::<Vec<_>>())
}

#[test]
fn correlation_stats_as_in_the_julia_tests() {
    let (L, H, T) = (5, 8, 60);
    let (gdp, wages) = gdp_and_wages(T);
    let real_data: Table = [
        ("real_gdp_quarterly".to_string(), Array::vector(&gdp)),
        ("wages_quarterly".to_string(), Array::vector(&wages)),
    ]
    .into();
    let rs =
        correlation_stats(&real_data, &["wages_quarterly"], "real_gdp_quarterly", L, H).unwrap();
    assert_eq!(rs.variables, ["wages_quarterly"]);
    assert_eq!(rs.crosscor["wages_quarterly"].len(), 2 * L + 1);
    assert!(rs.crosscor["wages_quarterly"].iter().all(|r| r.len() == 1));
    assert_eq!(rs.autocor["wages_quarterly"].len(), H + 1);
    assert_eq!(rs.volatility["wages_quarterly"].len(), 1);
    let m = rs.mean_crosscor("wages_quarterly").unwrap();
    assert_eq!(m.len(), 2 * L + 1);
    let col: Vec<f64> = rs.crosscor["wages_quarterly"]
        .iter()
        .map(|r| r[0])
        .collect();
    assert!(common::approx_list(&m, &col, common::SQRT_EPS, 0.0));
    assert!(rs.std_crosscor("wages_quarterly").unwrap()[0].is_nan());

    let n_sims = 3;
    let sims: Table = [
        ("real_gdp_quarterly".to_string(), repeat(&gdp, n_sims)),
        ("wages_quarterly".to_string(), repeat(&wages, n_sims)),
    ]
    .into();
    let ss = correlation_stats(&sims, &["wages_quarterly"], "real_gdp_quarterly", L, H).unwrap();
    assert!(
        ss.crosscor["wages_quarterly"]
            .iter()
            .all(|r| r.len() == n_sims)
    );
    assert_eq!(ss.volatility["wages_quarterly"].len(), n_sims);
    assert_eq!(ss.std_autocor("wages_quarterly").unwrap()[1], 0.0);

    let gs = correlation_stats(
        &real_data,
        &["real_gdp_quarterly"],
        "real_gdp_quarterly",
        L,
        H,
    )
    .unwrap();
    assert!(approx(
        gs.mean_crosscor("real_gdp_quarterly").unwrap()[L],
        1.0,
        common::SQRT_EPS,
        0.0
    ));

    let mismatched: Table = [
        ("real_gdp_quarterly".to_string(), repeat(&gdp, 2)),
        ("wages_quarterly".to_string(), Array::vector(&gdp)),
    ]
    .into();
    let ms = correlation_stats(
        &mismatched,
        &["wages_quarterly", "missing_var"],
        "real_gdp_quarterly",
        L,
        H,
    )
    .unwrap();
    assert!(ms.variables.is_empty());
    assert!(correlation_stats(&BTreeMap::new(), &["x"], "real_gdp_quarterly", L, H).is_err());
}

#[test]
fn correlation_stats_of_a_folder_as_in_the_julia_tests() {
    let (L, H, T, n_sims) = (5, 8, 60, 2);
    let (gdp, wages) = gdp_and_wages(T);
    let folder = std::env::temp_dir().join(format!("beforeit-corr-{}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    for yq in ["2010Q1", "2010Q2"] {
        let t: Table = [
            ("real_gdp_quarterly".to_string(), repeat(&gdp, n_sims)),
            ("wages_quarterly".to_string(), repeat(&wages, n_sims)),
        ]
        .into();
        std::fs::write(folder.join(format!("{yq}.json")), table_to_json(&t)).unwrap();
    }
    let stats = correlation_stats_folder(&folder, &["wages_quarterly"], "real_gdp_quarterly", L, H)
        .unwrap();
    std::fs::remove_dir_all(&folder).unwrap();
    assert!(
        stats.crosscor["wages_quarterly"]
            .iter()
            .all(|r| r.len() == 2 * n_sims)
    );
    assert_eq!(stats.volatility["wages_quarterly"].len(), 2 * n_sims);
}
