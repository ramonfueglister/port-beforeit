//! Helpers of the tests: the Austrian data set and the values of the
//! authors' MATLAB code from BeforeIT `test/matlab_code`, converted to JSON
//! (see `tests/data/README.md`).

#![allow(dead_code, non_snake_case)]

use beforeit::{InitialConditions, Parameters};
use serde_json::Value;

/// Julia `√eps`, the default relative tolerance of `isapprox`.
pub const SQRT_EPS: f64 = 1.490_116_119_384_765_6e-8;

pub fn load(name: &str) -> Value {
    let path = format!("{}/tests/data/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

pub fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{key} is missing or not a number"))
}

pub fn list(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} is missing or not a list"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

pub fn matrix(v: &Value, key: &str) -> Vec<Vec<f64>> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} is missing or not a matrix"))
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

/// Parameters and initial conditions for Austria, 2010Q1, BeforeIT
/// `AUSTRIA2010Q1`.
pub fn austria() -> (Parameters, InitialConditions) {
    let s = &*beforeit::AUSTRIA2010Q1;
    (s.parameters.clone(), s.initial_conditions.clone())
}

/// Julia `isapprox(x, y; rtol, atol)` for numbers.
pub fn approx(x: f64, y: f64, rtol: f64, atol: f64) -> bool {
    x == y
        || (x.is_finite()
            && y.is_finite()
            && (x - y).abs() <= atol.max(rtol * x.abs().max(y.abs())))
}

fn norm(x: impl Iterator<Item = f64>) -> f64 {
    x.map(|v| v * v).sum::<f64>().sqrt()
}

/// Julia `isapprox(x, y; rtol, atol)` for vectors: compares the norm of the
/// difference, or every entry if that norm is not finite.
pub fn approx_list(x: &[f64], y: &[f64], rtol: f64, atol: f64) -> bool {
    assert_eq!(x.len(), y.len(), "lists of different length");
    if x == y {
        return true;
    }
    let d = norm(x.iter().zip(y).map(|(a, b)| a - b));
    if d.is_finite() {
        let size = norm(x.iter().copied()).max(norm(y.iter().copied()));
        d <= atol.max(rtol * size)
    } else {
        x.iter().zip(y).all(|(&a, &b)| approx(a, b, rtol, atol))
    }
}

/// Checks a number like Julia's `@test isapprox(x, expected; rtol, atol)`.
#[track_caller]
pub fn check(name: &str, x: f64, expected: f64, rtol: f64, atol: f64) {
    assert!(
        approx(x, expected, rtol, atol),
        "{name}: {x} instead of {expected} (rtol {rtol}, atol {atol})"
    );
}

/// Checks a list like Julia's `@test isapprox(x, expected; rtol, atol)`.
#[track_caller]
pub fn check_list(name: &str, x: &[f64], expected: &[f64], rtol: f64, atol: f64) {
    assert!(
        approx_list(x, expected, rtol, atol),
        "{name}: deviation too large (rtol {rtol}, atol {atol}), first entries {:?} instead of {:?}",
        &x[..x.len().min(4)],
        &expected[..expected.len().min(4)]
    );
}

pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn as_f64(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| v as f64).collect()
}
