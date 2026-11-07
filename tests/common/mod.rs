//! Reference data of the authors, used only by the tests: Austria 2010Q1
//! from BeforeIT `data/austria` and values of the MATLAB code from
//! `test/matlab_code`, converted to JSON (see `tests/data/README.md`).

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

fn count(v: &Value, key: &str) -> usize {
    let x = num(v, key);
    assert!(x >= 0.0 && x.fract() == 0.0, "{key} = {x} is not a count");
    x as usize
}

/// Julia `get(parameters, key, default)` for the optional parameters.
fn num_or(v: &Value, key: &str, default: f64) -> f64 {
    if v.get(key).is_some() {
        num(v, key)
    } else {
        default
    }
}

/// Parameters and initial conditions for Austria, 2010Q1.
pub fn austria() -> (Parameters, InitialConditions) {
    let v = load("austria_2010q1_parameters");
    let c = matrix(&v, "C");
    let mut C = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            C[i][j] = c[i][j];
        }
    }
    let p = Parameters {
        G: count(&v, "G"),
        T_prime: count(&v, "T_prime"),
        H_act: count(&v, "H_act"),
        H_inact: count(&v, "H_inact"),
        J: count(&v, "J"),
        L: count(&v, "L"),
        I_s: list(&v, "I_s").iter().map(|&x| x as usize).collect(),
        alpha_s: list(&v, "alpha_s"),
        beta_s: list(&v, "beta_s"),
        kappa_s: list(&v, "kappa_s"),
        delta_s: list(&v, "delta_s"),
        w_s: list(&v, "w_s"),
        tau_Y_s: list(&v, "tau_Y_s"),
        tau_K_s: list(&v, "tau_K_s"),
        tau_INC: num(&v, "tau_INC"),
        tau_FIRM: num(&v, "tau_FIRM"),
        tau_VAT: num(&v, "tau_VAT"),
        tau_SIF: num(&v, "tau_SIF"),
        tau_SIW: num(&v, "tau_SIW"),
        tau_EXPORT: num(&v, "tau_EXPORT"),
        tau_CF: num(&v, "tau_CF"),
        tau_G: num(&v, "tau_G"),
        theta_UB: num(&v, "theta_UB"),
        psi: num(&v, "psi"),
        psi_H: num(&v, "psi_H"),
        mu: num(&v, "mu"),
        theta_DIV: num(&v, "theta_DIV"),
        theta: num(&v, "theta"),
        zeta: num(&v, "zeta"),
        zeta_LTV: num(&v, "zeta_LTV"),
        zeta_b: num(&v, "zeta_b"),
        b_CF_g: list(&v, "b_CF_g"),
        b_CFH_g: list(&v, "b_CFH_g"),
        b_HH_g: list(&v, "b_HH_g"),
        c_G_g: list(&v, "c_G_g"),
        c_E_g: list(&v, "c_E_g"),
        c_I_g: list(&v, "c_I_g"),
        a_sg: matrix(&v, "a_sg"),
        C,
        alpha_G: num(&v, "alpha_G"),
        beta_G: num(&v, "beta_G"),
        sigma_G: num(&v, "sigma_G"),
        alpha_E: num(&v, "alpha_E"),
        beta_E: num(&v, "beta_E"),
        sigma_E: num(&v, "sigma_E"),
        alpha_I: num(&v, "alpha_I"),
        beta_I: num(&v, "beta_I"),
        sigma_I: num(&v, "sigma_I"),
        alpha_pi_EA: num(&v, "alpha_pi_EA"),
        beta_pi_EA: num(&v, "beta_pi_EA"),
        sigma_pi_EA: num(&v, "sigma_pi_EA"),
        alpha_Y_EA: num(&v, "alpha_Y_EA"),
        beta_Y_EA: num(&v, "beta_Y_EA"),
        sigma_Y_EA: num(&v, "sigma_Y_EA"),
        r_G: num(&v, "r_G"),
        rho: num(&v, "rho"),
        r_star: num(&v, "r_star"),
        pi_star: num(&v, "pi_star"),
        xi_pi: num(&v, "xi_pi"),
        xi_gamma: num(&v, "xi_gamma"),
        n_banks: num_or(&v, "n_banks", 1.0) as usize,
        omega_pi: num_or(&v, "omega_pi", 0.0),
        theta_calvo: num_or(&v, "theta_calvo", 1.0),
        overdrafts_in_capital_ratio: v
            .get("overdrafts_in_capital_ratio")
            .is_some_and(|x| x.as_bool().unwrap()),
    };
    let v = load("austria_2010q1_initial_conditions");
    let ic = InitialConditions {
        D_H: num(&v, "D_H"),
        D_I: num(&v, "D_I"),
        D_RoW: num(&v, "D_RoW"),
        E_CB: num(&v, "E_CB"),
        E_k: num(&v, "E_k"),
        K_H: num(&v, "K_H"),
        L_G: num(&v, "L_G"),
        L_I: num(&v, "L_I"),
        N_s: list(&v, "N_s"),
        Y: list(&v, "Y"),
        pi: list(&v, "pi"),
        Y_EA: num(&v, "Y_EA"),
        pi_EA: num(&v, "pi_EA"),
        Y_I: list(&v, "Y_I"),
        C_E: list(&v, "C_E"),
        C_G: list(&v, "C_G"),
        omega: num(&v, "omega"),
        r_bar: num(&v, "r_bar"),
        sb_inact: num(&v, "sb_inact"),
        sb_other: num(&v, "sb_other"),
        w_UB: num(&v, "w_UB"),
    };
    (p, ic)
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
