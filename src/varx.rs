//! Vector autoregressions as forecast benchmarks, BeforeIT
//! `src/utils/varx.jl`: estimation with [`rfvar3`] and forecasts of `k`
//! steps, without or with exogenous variables.
//!
//! Matrices are given by their rows, one row per quarter and one column per
//! variable. The products of matrices and vectors, which Julia leaves to
//! BLAS, are added from left to right.

use crate::estimate::rfvar3;
use crate::evaluation::cov_matrix;
use crate::random::Randomness;

/// A matrix given by its rows.
pub type Matrix = Vec<Vec<f64>>;

/// The result of [`estimate_VAR`], Julia's `(alpha, beta, sigma, u)`:
/// `By` and `Bx` of [`rfvar3`], the covariance matrix of the residuals and
/// the residuals.
pub type VarEstimate = (Vec<Vec<Vec<f64>>>, Matrix, Matrix, Matrix);

/// The result of [`estimate_VARX`], Julia's `(alpha, beta, gamma, sigma,
/// u)`.
pub type VarxEstimate = (Vec<Vec<Vec<f64>>>, Vec<f64>, Matrix, Matrix, Matrix);

/// `estimate_VAR(ydata; intercept, lags)`: [`rfvar3`] with `lags` lags and,
/// if `intercept`, a constant. Returns `alpha = By`, `beta = Bx` (empty
/// without intercept), `sigma = cov(u)` and the residuals `u`.
pub fn estimate_VAR(ydata: &[Vec<f64>], intercept: bool, lags: usize) -> VarEstimate {
    let T = ydata.len();
    let xdata = vec![if intercept { vec![1.0] } else { Vec::new() }; T];
    let var = rfvar3(ydata, lags, &xdata);
    let sigma = cov_matrix(&var.u);
    (var.By, var.Bx, sigma, var.u)
}

/// `estimate_VARX(ydata, xdata; intercept, lags)`: [`rfvar3`] with the
/// exogenous variables `xdata` and, if `intercept`, a constant before them.
/// Returns `alpha = By`, `beta = Bx[:, 1]`, `gamma = Bx[:, 2:end]`,
/// `sigma = cov(u)` and `u`.
///
/// As in the Julia code, `beta` is the first column of `Bx` and `gamma` the
/// others also without intercept, when `beta` then belongs to the first
/// exogenous variable.
///
/// Returns an error if there is neither an intercept nor an exogenous
/// variable (Julia throws a `BoundsError`).
pub fn estimate_VARX(
    ydata: &[Vec<f64>],
    xdata: &[Vec<f64>],
    intercept: bool,
    lags: usize,
) -> Result<VarxEstimate, String> {
    let xdata: Matrix = if intercept {
        xdata
            .iter()
            .map(|r| std::iter::once(1.0).chain(r.iter().copied()).collect())
            .collect()
    } else {
        xdata.to_vec()
    };
    let var = rfvar3(ydata, lags, &xdata);
    if var.Bx.is_empty() || var.Bx[0].is_empty() {
        return Err("estimate_VARX: no exogenous variable and no intercept (Bx[:, 1])".into());
    }
    let beta = var.Bx.iter().map(|r| r[0]).collect();
    let gamma = var.Bx.iter().map(|r| r[1..].to_vec()).collect();
    let sigma = cov_matrix(&var.u);
    Ok((var.By, beta, gamma, sigma, var.u))
}

/// `_prepare_alpha(alpha, n_vars, lags)`: the coefficients of the lags as
/// one matrix, `n_vars × (lags · n_vars)`, column `l · n_vars + j` for
/// variable `j` at lag `l + 1`.
fn prepare_alpha(By: &[Vec<Vec<f64>>], n_vars: usize, lags: usize) -> Matrix {
    (0..n_vars)
        .map(|i| {
            (0..lags * n_vars)
                .map(|c| By[i][c % n_vars][c / n_vars])
                .collect()
        })
        .collect()
}

/// `_get_lagged_data(data, lags)`: the last `lags` rows, newest first, in
/// one vector.
fn lagged_data(data: &[Vec<f64>], lags: usize) -> Vec<f64> {
    (0..lags)
        .flat_map(|l| data[data.len() - 1 - l].iter().copied())
        .collect()
}

/// Julia `A * x`, BLAS `gemv`: `y[i] = Σ_c x[c] A[i][c]` from left to right,
/// starting from 0.
fn mat_vec(A: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    A.iter()
        .map(|row| row.iter().zip(x).fold(0.0, |acc, (a, v)| acc + v * a))
        .collect()
}

/// `_generate_stochastic_epsilon(n_vars, sigma)`: a normal draw with
/// standard deviation `sqrt(sigma[i, i])` for each variable.
fn stochastic_epsilon(sigma: &[Vec<f64>], rng: &mut dyn Randomness) -> Vec<f64> {
    (0..sigma.len())
        .map(|i| 0.0 + sigma[i][i].sqrt() * rng.randn())
        .collect()
}

/// `forecast_k_steps_VAR(data, n_forecasts; intercept, lags, stochastic)`:
/// estimates the VAR on `data` ([`estimate_VAR`]) and iterates it for
/// `n_forecasts` quarters, each forecast becoming the newest data point.
/// With `stochastic = Some(rng)`, each step adds a normal shock with the
/// variance of the residuals of its equation (Julia's `stochastic = true`,
/// which draws from the global generator). Returns the forecasts, one row
/// per quarter.
pub fn forecast_k_steps_VAR(
    data: &[Vec<f64>],
    n_forecasts: usize,
    intercept: bool,
    lags: usize,
    mut stochastic: Option<&mut dyn Randomness>,
) -> Matrix {
    let n_vars = data.first().map_or(0, |r| r.len());
    let mut current_data = data.to_vec();
    let (By, beta, sigma, _) = estimate_VAR(&current_data, intercept, lags);
    let alpha = prepare_alpha(&By, n_vars, lags);
    let mut forecasted_values = Vec::with_capacity(n_forecasts);
    for _ in 0..n_forecasts {
        let epsilon = match stochastic.as_deref_mut() {
            Some(rng) => stochastic_epsilon(&sigma, rng),
            None => vec![0.0; n_vars],
        };
        let lagged = lagged_data(&current_data, lags);
        let mut next_value: Vec<f64> = mat_vec(&alpha, &lagged)
            .iter()
            .zip(&epsilon)
            .map(|(a, e)| a + e)
            .collect();
        if intercept {
            for (v, b) in next_value.iter_mut().zip(&beta) {
                *v += b[0];
            }
        }
        forecasted_values.push(next_value.clone());
        current_data.push(next_value);
    }
    forecasted_values
}

/// `forecast_k_steps_VARX(data, exogenous, n_forecasts; intercept, lags,
/// stochastic)`: estimates the VARX on `data` with the exogenous rows
/// `2:(T + 1)` ([`estimate_VARX`]) and iterates it for `n_forecasts`
/// quarters, step `i` with the exogenous row `end − n_forecasts + i`.
///
/// Returns an error where Julia throws: if `exogenous` has fewer than
/// `T + 1` rows or fewer than `n_forecasts` rows for the forecasts, if there
/// is neither an intercept nor an exogenous variable, and without intercept
/// also otherwise, because `gamma` then lacks the first exogenous variable
/// and does not fit the exogenous rows (a `DimensionMismatch` in Julia).
pub fn forecast_k_steps_VARX(
    data: &[Vec<f64>],
    exogenous: &[Vec<f64>],
    n_forecasts: usize,
    intercept: bool,
    lags: usize,
    mut stochastic: Option<&mut dyn Randomness>,
) -> Result<Matrix, String> {
    let T = data.len();
    let n_vars = data.first().map_or(0, |r| r.len());
    if exogenous.len() < T + 1 {
        return Err(format!(
            "forecast_k_steps_VARX: {} exogenous rows, {} needed (Julia throws a BoundsError)",
            exogenous.len(),
            T + 1
        ));
    }
    if exogenous.len() < n_forecasts {
        return Err("forecast_k_steps_VARX: fewer exogenous rows than forecasts".into());
    }
    let mut current_data = data.to_vec();
    let (By, beta, gamma, sigma, _) =
        estimate_VARX(&current_data, &exogenous[1..T + 1], intercept, lags)?;
    let alpha = prepare_alpha(&By, n_vars, lags);
    let mut forecasted_values = Vec::with_capacity(n_forecasts);
    for i in 1..=n_forecasts {
        let epsilon = match stochastic.as_deref_mut() {
            Some(rng) => stochastic_epsilon(&sigma, rng),
            None => vec![0.0; n_vars],
        };
        let lagged = lagged_data(&current_data, lags);
        let row = &exogenous[exogenous.len() - n_forecasts + i - 1];
        if gamma.first().map_or(0, |r| r.len()) != row.len() {
            return Err(format!(
                "forecast_k_steps_VARX: gamma has {} columns, the exogenous data {} (Julia throws a DimensionMismatch)",
                gamma.first().map_or(0, |r| r.len()),
                row.len()
            ));
        }
        let exogenous_term = mat_vec(&gamma, row);
        let mut next_value: Vec<f64> = mat_vec(&alpha, &lagged)
            .iter()
            .zip(&exogenous_term)
            .zip(&epsilon)
            .map(|((a, x), e)| a + x + e)
            .collect();
        if intercept {
            for (v, b) in next_value.iter_mut().zip(&beta) {
                *v += b;
            }
        }
        forecasted_values.push(next_value.clone());
        current_data.push(next_value);
    }
    Ok(forecasted_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_of_the_lags() {
        // By[e][v][l] for 2 variables and 2 lags.
        let By = vec![
            vec![vec![1.0, 5.0], vec![2.0, 6.0]],
            vec![vec![3.0, 7.0], vec![4.0, 8.0]],
        ];
        assert_eq!(
            prepare_alpha(&By, 2, 2),
            vec![vec![1.0, 2.0, 5.0, 6.0], vec![3.0, 4.0, 7.0, 8.0]]
        );
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        assert_eq!(lagged_data(&data, 2), vec![5.0, 6.0, 3.0, 4.0]);
        assert_eq!(lagged_data(&data, 1), vec![5.0, 6.0]);
    }

    #[test]
    fn an_exact_var_is_continued() {
        // y_t = 0.5 y_(t-1) + 1 without noise: the forecast continues it.
        let mut y = vec![vec![10.0]];
        for _ in 0..30 {
            let last = y.last().unwrap()[0];
            y.push(vec![0.5 * last + 1.0]);
        }
        let f = forecast_k_steps_VAR(&y, 3, true, 1, None);
        let mut last = y.last().unwrap()[0];
        for row in &f {
            last = 0.5 * last + 1.0;
            assert!((row[0] - last).abs() < 1e-12);
        }
    }
}
