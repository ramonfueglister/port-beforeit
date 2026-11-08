//! Vector autoregressions as forecast benchmarks, BeforeIT
//! `src/utils/varx.jl`: estimation with [`rfvar3`] and forecasts of `k`
//! steps, without or with exogenous variables.
//!
//! Matrices are given by their rows, one row per quarter and one column per
//! variable, so a matrix without rows has no columns here (Julia's `0 × n`
//! matrices with `n > 0` have no counterpart). The products of matrices
//! and vectors, which Julia leaves to BLAS, are added from left to right.

use crate::estimate::{nan_in_regressors, rfvar3};
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

/// Returns an error where Julia's `rfvar3` throws in its singular value
/// decomposition of the regressors `X`: in `svd_mx2` if `X` has two
/// columns and no row, that is, no row of data after the first `lags` rows
/// (`zeros(M - 1)` with `M = 0`), for example one series with an intercept
/// and one lag on a single row; and in `svd` if `X` has one column or more
/// than two, at least one row and a NaN entry (LAPACK's `dgesdd`, which
/// Julia's `svd` calls, rejects a NaN, and Julia throws an
/// `ArgumentError`). `nx` is the number of columns of Julia's `xdata`,
/// which an `xdata` without rows here cannot tell.
fn check_svd(
    name: &str,
    ydata: &[Vec<f64>],
    xdata: &[Vec<f64>],
    nx: usize,
    lags: usize,
) -> Result<(), String> {
    let T = ydata.len();
    let nvar = ydata.first().map_or(0, |r| r.len());
    if nvar * lags + nx == 2 && T <= lags {
        return Err(format!(
            "{name}: two regressors and no row of data after the {lags} lags \
             (Julia's svd_mx2 throws)"
        ));
    }
    if nan_in_regressors(ydata, lags, xdata) {
        return Err(format!(
            "{name}: a regressor is NaN (LAPACK's dgesdd rejects it and Julia throws an ArgumentError)"
        ));
    }
    Ok(())
}

/// The error where the data have no columns: `rfvar3` returns, but
/// Julia's `cov` of the residuals then throws a `BoundsError` at
/// `first(C)` of the empty matrix `C = u'u`.
fn no_columns(name: &str) -> String {
    format!("{name}: the data have no columns (Julia's cov of the residuals throws a BoundsError)")
}

/// Julia's `cov(u)` of the residuals of `nvar ≥ 1` equations. Without
/// residuals (no row of data after the lags), Julia's `u'u` is a matrix of
/// zeros, which its factor `1//(0 - 1)` turns into −0.0.
fn residual_cov(u: &[Vec<f64>], nvar: usize) -> Matrix {
    if u.is_empty() {
        vec![vec![-0.0; nvar]; nvar]
    } else {
        cov_matrix(u)
    }
}

/// `estimate_VAR(ydata; intercept, lags)`: [`rfvar3`] with `lags` lags and,
/// if `intercept`, a constant. Returns `alpha = By`, `beta = Bx` (empty
/// without intercept), `sigma = cov(u)` and the residuals `u`.
///
/// Returns an error where Julia throws in `rfvar3`: if the regressors are
/// two columns and the data have no more rows than `lags`, and if the
/// regressors are one column or more than two, `lags` is at least 1, the
/// data have more rows than `lags` and a value of the data in a row before
/// the last is NaN (LAPACK's `dgesdd`, which Julia's `svd` calls, rejects
/// it); and after those, if the data have no columns, which includes data
/// without rows (Julia's `cov` of the residuals throws).
pub fn estimate_VAR(
    ydata: &[Vec<f64>],
    intercept: bool,
    lags: usize,
) -> Result<VarEstimate, String> {
    let T = ydata.len();
    let nvar = ydata.first().map_or(0, |r| r.len());
    let xdata = vec![if intercept { vec![1.0] } else { Vec::new() }; T];
    check_svd("estimate_VAR", ydata, &xdata, usize::from(intercept), lags)?;
    if nvar == 0 {
        return Err(no_columns("estimate_VAR"));
    }
    let var = rfvar3(ydata, lags, &xdata);
    let sigma = residual_cov(&var.u, nvar);
    Ok((var.By, var.Bx, sigma, var.u))
}

/// `estimate_VARX(ydata, xdata; intercept, lags)`: [`rfvar3`] with the
/// exogenous variables `xdata` and, if `intercept`, a constant before them.
/// Returns `alpha = By`, `beta = Bx[:, 1]`, `gamma = Bx[:, 2:end]`,
/// `sigma = cov(u)` and `u`.
///
/// As in the Julia code, `beta` is the first column of `Bx` and `gamma` the
/// others also without intercept, when `beta` then belongs to the first
/// exogenous variable. Without intercept and exogenous variables, `beta`
/// and `gamma` are empty, as Julia's `Bx = []` gives them.
///
/// Returns an error where Julia throws in `rfvar3`: if `xdata` has fewer
/// rows than `ydata` and `ydata` more rows than `lags` (a `BoundsError`),
/// if the regressors are two columns and the data have no more rows than
/// `lags`, and if the regressors are one column or more than two, the data
/// have more rows than `lags` and a regressor is NaN, that is, a value of
/// `xdata` in rows `lags + 1` to `T` (counted from 1, `T` the number of
/// rows of `ydata`) or, if `lags` is at least 1, a value of the data in a
/// row before the last (LAPACK's `dgesdd`, which Julia's `svd` calls,
/// rejects it); and after those, if `ydata` has no columns, which includes
/// data without rows (Julia's `cov` of the residuals throws). Like Julia's
/// `rfvar3`, it first prints "Mismatch of x and y data lengths" if `xdata`
/// and `ydata` have different numbers of rows. With `intercept`, the
/// constant counts as a column of the regressors also if `xdata` has no
/// rows, as in Julia's `hcat(ones(0, 1), xdata)`.
pub fn estimate_VARX(
    ydata: &[Vec<f64>],
    xdata: &[Vec<f64>],
    intercept: bool,
    lags: usize,
) -> Result<VarxEstimate, String> {
    estimate_varx(ydata, xdata, 0, intercept, lags)
}

/// [`estimate_VARX`] with `columns`, the number of columns of Julia's
/// `xdata` before the constant if `xdata` has no rows (here it then has
/// none), as for the exogenous rows `2:1` of [`forecast_k_steps_VARX`]
/// without data, which keep the columns of `exogenous` in Julia.
fn estimate_varx(
    ydata: &[Vec<f64>],
    xdata: &[Vec<f64>],
    columns: usize,
    intercept: bool,
    lags: usize,
) -> Result<VarxEstimate, String> {
    // Julia's `size(xdata, 2)` after the constant.
    let nx = xdata.first().map_or(columns, |r| r.len()) + usize::from(intercept);
    let no_rows = xdata.is_empty();
    let xdata: Matrix = if intercept {
        xdata
            .iter()
            .map(|r| std::iter::once(1.0).chain(r.iter().copied()).collect())
            .collect()
    } else {
        xdata.to_vec()
    };
    let T = ydata.len();
    // The checks that return before `rfvar3` print its message first, as
    // Julia's `rfvar3` prints it before it throws.
    let mismatch = || {
        if xdata.len() != T {
            println!("Mismatch of x and y data lengths");
        }
    };
    if xdata.len() < T && T > lags {
        mismatch();
        return Err(format!(
            "estimate_VARX: {} rows of exogenous data for {T} rows of data \
             (Julia throws a BoundsError)",
            xdata.len()
        ));
    }
    if let Err(e) = check_svd("estimate_VARX", ydata, &xdata, nx, lags) {
        mismatch();
        return Err(e);
    }
    let nvar = ydata.first().map_or(0, |r| r.len());
    if nvar == 0 {
        mismatch();
        return Err(no_columns("estimate_VARX"));
    }
    if no_rows && intercept && T <= lags {
        // Julia's regressors are the lags of at least one equation and the
        // constant without rows: `svd` makes no LAPACK call, the
        // coefficients are 0, and the empty `xdata` gives `Bx = []`.
        // `rfvar3` here would not count the constant.
        mismatch();
        return Ok((
            vec![vec![vec![0.0; lags]; nvar]; nvar],
            Vec::new(),
            Vec::new(),
            residual_cov(&[], nvar),
            Vec::new(),
        ));
    }
    let var = rfvar3(ydata, lags, &xdata);
    let sigma = residual_cov(&var.u, nvar);
    // Julia's `Bx[:, 1]` and `Bx[:, 2:end]` of `Bx = []` are empty.
    let beta = var.Bx.iter().map(|r| r[0]).collect();
    let gamma = var.Bx.iter().map(|r| r[1..].to_vec()).collect();
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

/// The shapes that the first step of a VARX forecast checks: the numbers
/// of columns and rows of `gamma`, the length of the first exogenous row
/// of the forecasts (`None` if `exogenous` has fewer rows than forecasts)
/// and the number of variables.
struct VarxShapes {
    gamma_columns: usize,
    gamma_rows: usize,
    row: Option<usize>,
    n_vars: usize,
}

/// The checks of the first forecast step, in Julia's order, where Julia
/// throws: a variance of the residuals that is NaN with `stochastic`
/// (Julia's `Normal` throws a `DomainError`), fewer rows of data than lags
/// (a `BoundsError` in `_get_lagged_data`), for the VARX fewer exogenous
/// rows than forecasts (a `BoundsError` in `exogenous[end - n_forecasts +
/// 1, :]`) and a `gamma` whose number of columns differs from the length of
/// the exogenous row (a `DimensionMismatch`), no lags (a `MethodError`: Julia multiplies the
/// coefficients, a vector or a three-dimensional array, by the empty vector
/// of lags) and for the VARX a `gamma` without a row per variable, as
/// without intercept and exogenous variables (a `DimensionMismatch` when
/// Julia adds the exogenous term or stacks the forecast).
fn check_first_step(
    name: &str,
    T: usize,
    lags: usize,
    sigma: &[Vec<f64>],
    stochastic: bool,
    varx: Option<VarxShapes>,
) -> Result<(), String> {
    let invalid = (0..sigma.len()).find(|&i| sigma[i][i].is_nan() || sigma[i][i] < 0.0);
    if let (true, Some(i)) = (stochastic, invalid) {
        return Err(format!(
            "{name}: the variance of the residuals of equation {} is {} \
             (Julia's Normal throws a DomainError)",
            i + 1,
            sigma[i][i]
        ));
    }
    if T < lags {
        return Err(format!(
            "{name}: {T} rows of data, fewer than the {lags} lags (Julia throws a BoundsError)"
        ));
    }
    if let Some(s) = &varx {
        let Some(row) = s.row else {
            return Err(format!(
                "{name}: fewer exogenous rows than forecasts (Julia throws a BoundsError)"
            ));
        };
        if s.gamma_columns != row {
            return Err(format!(
                "{name}: gamma has {} columns, the exogenous data {row} (Julia throws a \
                 DimensionMismatch)",
                s.gamma_columns
            ));
        }
    }
    if lags == 0 {
        return Err(format!("{name}: no lags (Julia throws a MethodError)"));
    }
    if let Some(s) = &varx
        && s.gamma_rows != s.n_vars
    {
        return Err(format!(
            "{name}: gamma has {} rows for {} variables (no intercept and no exogenous \
             variable; Julia throws a DimensionMismatch)",
            s.gamma_rows, s.n_vars
        ));
    }
    Ok(())
}

/// `forecast_k_steps_VAR(data, n_forecasts; intercept, lags, stochastic)`:
/// estimates the VAR on `data` ([`estimate_VAR`]) and iterates it for
/// `n_forecasts` quarters, each forecast becoming the newest data point.
/// With `stochastic = Some(rng)`, each step adds a normal shock with the
/// variance of the residuals of its equation (Julia's `stochastic = true`,
/// which draws from the global generator). Returns the forecasts, one row
/// per quarter.
///
/// Returns an error where Julia throws: as [`estimate_VAR`], and, if there
/// is at least one forecast, for fewer rows of data than `lags`, for
/// `lags = 0` and, with `stochastic`, for a variance of the residuals that
/// is NaN (for example with one row of data after the lags, whose variance
/// divides by 0).
pub fn forecast_k_steps_VAR(
    data: &[Vec<f64>],
    n_forecasts: usize,
    intercept: bool,
    lags: usize,
    mut stochastic: Option<&mut dyn Randomness>,
) -> Result<Matrix, String> {
    let n_vars = data.first().map_or(0, |r| r.len());
    let mut current_data = data.to_vec();
    let (By, beta, sigma, _) = estimate_VAR(&current_data, intercept, lags)?;
    if n_forecasts > 0 {
        check_first_step(
            "forecast_k_steps_VAR",
            data.len(),
            lags,
            &sigma,
            stochastic.is_some(),
            None,
        )?;
    }
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
    Ok(forecasted_values)
}

/// `forecast_k_steps_VARX(data, exogenous, n_forecasts; intercept, lags,
/// stochastic)`: estimates the VARX on `data` with the exogenous rows
/// `2:(T + 1)` ([`estimate_VARX`]) and iterates it for `n_forecasts`
/// quarters, step `i` with the exogenous row `end − n_forecasts + i`.
///
/// Returns an error where Julia throws: if `exogenous` has fewer than
/// `T + 1` rows or fewer than `n_forecasts` rows for the forecasts, as
/// [`estimate_VARX`], and, if there is at least one forecast, as
/// [`forecast_k_steps_VAR`] (fewer rows of data than `lags`, `lags = 0`, a
/// variance of the residuals that is NaN with `stochastic`) and without
/// intercept, because `gamma` then lacks the first exogenous variable and
/// does not fit the exogenous rows, or, without exogenous variables, has no
/// rows (a `DimensionMismatch` in Julia).
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
    let mut current_data = data.to_vec();
    // Julia's `exogenous[2:(T + 1), :]` keeps the columns of `exogenous`
    // also without data (`T = 0`).
    let (By, beta, gamma, sigma, _) = estimate_varx(
        &current_data,
        &exogenous[1..T + 1],
        exogenous[0].len(),
        intercept,
        lags,
    )?;
    if n_forecasts > 0 {
        check_first_step(
            "forecast_k_steps_VARX",
            T,
            lags,
            &sigma,
            stochastic.is_some(),
            Some(VarxShapes {
                gamma_columns: gamma.first().map_or(0, |r| r.len()),
                gamma_rows: gamma.len(),
                row: exogenous
                    .len()
                    .checked_sub(n_forecasts)
                    .map(|k| exogenous[k].len()),
                n_vars,
            }),
        )?;
    }
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
        let f = forecast_k_steps_VAR(&y, 3, true, 1, None).unwrap();
        let mut last = y.last().unwrap()[0];
        for row in &f {
            last = 0.5 * last + 1.0;
            assert!((row[0] - last).abs() < 1e-12);
        }
    }

    #[test]
    fn errors_where_julia_throws() {
        let mut rng = crate::Xoshiro::seed_from_u64(1);
        // One series with intercept and one lag on one row: two regressors
        // and no row after the lags, Julia's svd_mx2 throws.
        assert!(estimate_VAR(&[vec![0.5]], true, 1).is_err());
        assert!(forecast_k_steps_VAR(&[vec![0.5]], 2, true, 1, None).is_err());
        assert!(
            forecast_k_steps_VARX(&[vec![0.5]], &[vec![3.0], vec![2.0]], 1, false, 1, None)
                .is_err()
        );
        // Fewer rows than lags: a BoundsError in `_get_lagged_data`, but
        // only once there is a forecast.
        let two_rows = [vec![0.5, 0.1], vec![0.7, 0.2]];
        assert!(forecast_k_steps_VAR(&two_rows, 1, false, 3, None).is_err());
        assert_eq!(
            forecast_k_steps_VAR(&two_rows, 0, false, 3, None),
            Ok(vec![])
        );
        assert!(
            forecast_k_steps_VARX(&[vec![0.5, 0.2]], &[vec![1.0], vec![2.0]], 1, true, 2, None)
                .is_err()
        );
        // Fewer exogenous rows than rows of data: a BoundsError in rfvar3.
        let three_rows = [vec![0.5], vec![0.7], vec![0.9]];
        assert!(estimate_VARX(&three_rows, &[vec![1.0], vec![2.0]], true, 1).is_err());
        // No lags: a MethodError once there is a forecast.
        assert!(forecast_k_steps_VAR(&three_rows, 1, true, 0, None).is_err());
        assert_eq!(
            forecast_k_steps_VAR(&three_rows, 0, false, 0, None),
            Ok(vec![])
        );
        // One row after the lag: the variance of the residuals is NaN, and
        // only the stochastic forecast throws (Normal(0, NaN)).
        let y = [vec![0.5], vec![0.7]];
        assert!(forecast_k_steps_VAR(&y, 2, true, 1, None).is_ok());
        assert!(forecast_k_steps_VAR(&y, 2, true, 1, Some(&mut rng)).is_err());
    }

    #[test]
    fn nan_regressors_as_in_julia() {
        // A NaN among regressors that are not two columns: LAPACK's dgesdd
        // rejects it in Julia.
        let y = [vec![1.0], vec![f64::NAN], vec![2.0], vec![3.0], vec![2.5]];
        assert!(estimate_VAR(&y, true, 2).is_err());
        assert!(estimate_VAR(&y, false, 1).is_err());
        assert!(forecast_k_steps_VAR(&y, 2, true, 2, None).is_err());
        // Two columns go to svd_mx2, which does not throw.
        assert!(estimate_VAR(&y, true, 1).is_ok());
        // A NaN in the last row only enters `y`, not the regressors.
        let last = [vec![1.0], vec![2.0], vec![1.5], vec![f64::NAN]];
        assert!(estimate_VAR(&last, true, 2).is_ok());
        // A NaN in an exogenous row after the first `lags` is a regressor,
        // one in the first row is not.
        let y4 = [vec![2.0], vec![3.0], vec![2.5], vec![2.8]];
        let x = [vec![f64::NAN], vec![1.0], vec![f64::NAN], vec![2.0]];
        assert!(estimate_VARX(&y4, &x, true, 1).is_err());
        let x0 = [vec![f64::NAN], vec![1.0], vec![1.5], vec![2.0]];
        assert!(estimate_VARX(&y4, &x0, true, 1).is_ok());
    }

    #[test]
    fn no_nan_error_where_julia_does_not_throw() {
        // With `lags = 0` the data are no regressors, and rows of `xdata`
        // after those of `ydata` are not used (Julia only prints the
        // mismatch).
        let y = [vec![1.0], vec![f64::NAN], vec![2.0], vec![3.0]];
        assert!(estimate_VAR(&y, true, 0).is_ok());
        let y4 = [vec![1.0], vec![2.0], vec![1.5], vec![3.0]];
        let x6 = [
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![f64::NAN],
        ];
        assert!(estimate_VARX(&y4, &x6, true, 1).is_ok());
    }

    #[test]
    fn constant_over_xdata_without_rows() {
        // Julia's `hcat(ones(0, 1), xdata)` keeps the constant: with one lag
        // of one variable the regressors are two columns without rows
        // (svd_mx2 throws), with two variables three (Julia returns zeros).
        assert!(estimate_VARX(&[vec![0.5]], &[], true, 1).is_err());
        let (By, beta, gamma, sigma, u) = estimate_VARX(&[vec![0.5, 0.3]], &[], true, 1).unwrap();
        assert_eq!(By, vec![vec![vec![0.0]; 2]; 2]);
        assert!(beta.is_empty() && gamma.is_empty() && u.is_empty());
        assert_eq!(sigma.len(), 2);
    }

    #[test]
    fn data_without_columns_as_in_julia() {
        // `rfvar3` returns, and Julia's `cov` of the residuals throws.
        for e in [
            estimate_VAR(&[], true, 1).unwrap_err(),
            estimate_VAR(&vec![vec![]; 3], true, 1).unwrap_err(),
            estimate_VARX(&[], &[], true, 1).unwrap_err(),
            estimate_VARX(&[vec![]], &[], true, 1).unwrap_err(),
            forecast_k_steps_VAR(&[], 0, true, 1, None).unwrap_err(),
        ] {
            assert!(e.contains("no columns"), "{e}");
        }
        // Without data, Julia's exogenous rows `2:1` keep the column of
        // `exogenous`: with the constant two columns, and svd_mx2 throws
        // first.
        for e in [
            forecast_k_steps_VARX(&[], &[vec![1.0]], 0, true, 1, None).unwrap_err(),
            forecast_k_steps_VARX(&[], &[vec![1.0], vec![2.0]], 1, true, 0, None).unwrap_err(),
        ] {
            assert!(e.contains("svd_mx2"), "{e}");
        }
    }

    #[test]
    fn varx_errors_in_julias_order() {
        // A NaN regressor makes the estimation throw before the forecast
        // finds too few exogenous rows.
        let y = [vec![1.0], vec![f64::NAN], vec![2.0], vec![3.0], vec![2.5]];
        let x = vec![vec![1.0]; 6];
        let e = forecast_k_steps_VARX(&y, &x, 7, true, 2, None).unwrap_err();
        assert!(e.contains("ArgumentError"), "{e}");
        let y4 = [vec![1.0], vec![2.0], vec![1.5], vec![3.0]];
        let e = forecast_k_steps_VARX(&y4, &vec![vec![1.0]; 5], 6, true, 1, None).unwrap_err();
        assert!(e.contains("fewer exogenous rows than forecasts"), "{e}");
        assert_eq!(
            forecast_k_steps_VARX(&y4, &vec![vec![1.0]; 5], 5, true, 1, None)
                .unwrap()
                .len(),
            5
        );
    }

    #[test]
    fn varx_without_intercept_and_exogenous_variables() {
        // Julia's `Bx = []`: `beta` and `gamma` are empty, and only a
        // forecast throws (a DimensionMismatch).
        let y = [vec![0.5], vec![0.7], vec![0.6], vec![0.65]];
        let (_, beta, gamma, _, _) = estimate_VARX(&y, &vec![vec![]; 4], false, 1).unwrap();
        assert!(beta.is_empty() && gamma.is_empty());
        assert_eq!(
            forecast_k_steps_VARX(&y, &vec![vec![]; 5], 0, false, 1, None),
            Ok(vec![])
        );
        let e = forecast_k_steps_VARX(&y, &vec![vec![]; 5], 1, false, 1, None).unwrap_err();
        assert!(e.contains("DimensionMismatch"), "{e}");
        // Without intercept and with `lags = 0`, Julia multiplies `gamma` by
        // the exogenous row before the lags: a DimensionMismatch, not the
        // MethodError of the lags.
        let x = [vec![1.0], vec![2.0], vec![1.5], vec![2.5], vec![3.0]];
        let e = forecast_k_steps_VARX(&y, &x, 1, false, 0, None).unwrap_err();
        assert!(e.contains("DimensionMismatch"), "{e}");
        let e = forecast_k_steps_VARX(&y, &x, 1, true, 0, None).unwrap_err();
        assert!(e.contains("MethodError"), "{e}");
    }

    #[test]
    fn no_row_after_the_lags_as_in_julia() {
        // No residuals: Julia's `cov` gives −0.0 and the coefficients are 0,
        // so the forecasts are 0, with and without noise.
        let (_, _, sigma, u) = estimate_VAR(&[vec![0.5, 0.3]], true, 1).unwrap();
        assert!(u.is_empty());
        assert_eq!(sigma.len(), 2);
        assert!(
            sigma
                .iter()
                .flatten()
                .all(|&v| v == 0.0 && v.is_sign_negative())
        );
        let mut rng = crate::Xoshiro::seed_from_u64(1);
        let f = forecast_k_steps_VAR(&[vec![0.5, 0.3]], 2, true, 1, Some(&mut rng)).unwrap();
        assert_eq!(f, vec![vec![0.0, 0.0]; 2]);
    }
}
