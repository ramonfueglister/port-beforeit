//! Statistical tests of forecast errors: BeforeIT `src/utils/dmtest.jl`,
//! `mztest.jl` and `bias_ttest.jl`, with the parts of Statistics.jl,
//! Distributions.jl and SpecialFunctions.jl that they use.
//!
//! The p-values use the standard normal distribution function
//! `erfc(−z/√2)/2` of StatsFuns.jl (`erfc` from `libm`; Julia calls
//! openlibm, both are Sun's fdlibm algorithm) and, for the
//! Mincer-Zarnowitz test, the χ² distribution with 2 degrees of freedom
//! through SpecialFunctions.jl's `gamma_inc`, whose code path for `a = 1`
//! is ported. Dot products, which Julia leaves to BLAS, are added from left
//! to right.

use crate::julia::{dot, max, sum, sum_by};

/// Statistics.jl `mean(x)`: the sum in the order of Julia's `sum`, divided
/// by the length (NaN for an empty list).
pub(crate) fn mean(x: &[f64]) -> f64 {
    sum(x) / x.len() as f64
}

/// Statistics.jl `var(x)`: the squared deviations from the mean, summed in
/// the order of Julia's `sum` and divided by `n − 1`.
pub(crate) fn var(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let m = mean(x);
    sum_by(x.len(), |i| {
        let d = x[i] - m;
        d * d
    }) / (x.len() as f64 - 1.0)
}

/// Statistics.jl `std(x)`: `sqrt(var(x))`.
pub(crate) fn std(x: &[f64]) -> f64 {
    var(x).sqrt()
}

/// Statistics.jl `cov(x, y)` of two vectors: the dot product of the
/// deviations from the means, divided by `n − 1`.
///
/// Returns an error for empty vectors or vectors of different lengths
/// (Julia throws).
pub(crate) fn cov(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.is_empty() || y.is_empty() {
        return Err("covariance only defined for non-empty vectors".into());
    }
    if x.len() != y.len() {
        return Err(format!(
            "dot product of vectors of lengths {} and {}",
            y.len(),
            x.len()
        ));
    }
    let (mx, my) = (mean(x), mean(y));
    let xc: Vec<f64> = x.iter().map(|v| v - mx).collect();
    let yc: Vec<f64> = y.iter().map(|v| v - my).collect();
    Ok(dot(&yc, &xc) / (x.len() as f64 - 1.0))
}

/// Statistics.jl `cov(X)` of the columns of a matrix given by its rows:
/// the column means (Julia's `sum` along the first dimension, divided by
/// the number of rows), then the dot products of the centered columns
/// (BLAS in Julia), multiplied by `1 / (n − 1)` (Julia multiplies by the
/// rational `1//(n − 1)` converted to a float).
pub(crate) fn cov_matrix(X: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = X.len();
    let k = X.first().map_or(0, |r| r.len());
    let columns: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let c: Vec<f64> = X.iter().map(|r| r[j]).collect();
            let m = sum(&c) / n as f64;
            c.iter().map(|v| v - m).collect()
        })
        .collect();
    let b = 1.0 / (n as f64 - 1.0);
    (0..k)
        .map(|i| (0..k).map(|j| dot(&columns[i], &columns[j]) * b).collect())
        .collect()
}

/// StatsFuns.jl `normcdf(z)`: the distribution function of the standard
/// normal distribution, `erfc(−z/√2)/2`.
pub fn normcdf(z: f64) -> f64 {
    libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2) / 2.0
}

/// SpecialFunctions.jl `gamma_inc(1.0, x)`, the regularized incomplete
/// gamma functions `(P(1, x), Q(1, x))` for `x ≥ 0` or NaN, along the code
/// path of `_gamma_inc` for `a = 1` and 14 digits (`ind = 0`): the Taylor
/// series for `x < 1`, the finite sum `Q = exp(−x)` for `1 ≤ x < 31` and
/// the asymptotic expansion above.
fn gamma_inc_a1(x: f64) -> (f64, f64) {
    let a = 1.0;
    if x.is_nan() {
        let ax = a * x;
        return (ax, ax);
    }
    if x.is_infinite() {
        return (1.0, 0.0);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    // a >= 1 and a < big1[1] = 25; 2a is an integer.
    if a > x || x >= 31.0 {
        // _gamma_inc_choose_algorithm
        let r = rgammax(a, x);
        if r == 0.0 {
            return if x <= a { (0.0, 1.0) } else { (1.0, 0.0) };
        }
        if x <= max(a, std::f64::consts::LN_10) {
            gamma_inc_taylor(a, x)
        } else {
            // x < x0[1] = 31 cannot occur here.
            gamma_inc_asym(a, x)
        }
    } else {
        // gamma_inc_fsum for an integer a = 1: sm = exp(-x), no terms.
        let q = libm::exp(-x);
        (1.0 - q, q)
    }
}

/// SpecialFunctions.jl `rgammax(a, x)` for `1 ≤ a ≤ 20`:
/// `e^(−x) x^a / Γ(a)`.
fn rgammax(a: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let t = a * libm::log(x) - x;
    libm::exp(t) / libm::tgamma(a)
}

/// SpecialFunctions.jl `gamma_inc_taylor(a, x, 0)`.
fn gamma_inc_taylor(a: f64, x: f64) -> (f64, f64) {
    let tolerance = 0.5 * 5.0e-15;
    // The first 21 terms, cumprod of x / (a + i).
    let mut ts = [0.0; 21];
    let mut c = 1.0;
    for (i, t) in ts.iter_mut().enumerate() {
        let term = x / (a + (i + 1) as f64);
        c = if i == 0 { term } else { c * term };
        *t = c;
    }
    let first_small_t = ts.iter().position(|&t| t < 1.0e-3).unwrap_or(20);
    let mut t = ts[first_small_t];
    let mut sm = t;
    let mut apn = a + (first_small_t + 1) as f64;
    while t > tolerance {
        apn += 1.0;
        t *= x / apn;
        sm += t;
    }
    for j in (0..first_small_t).rev() {
        sm += ts[j];
    }
    let p = (rgammax(a, x) / a) * (1.0 + sm);
    (p, 1.0 - p)
}

/// SpecialFunctions.jl `gamma_inc_asym(a, x, 0)`.
fn gamma_inc_asym(a: f64, x: f64) -> (f64, f64) {
    let acc = 5.0e-15;
    let mut ts = [0.0; 21];
    let mut c = 1.0;
    for (i, t) in ts.iter_mut().enumerate() {
        let term = (a - (i + 1) as f64) / x;
        c = if i == 0 { term } else { c * term };
        *t = c;
    }
    let first_small_t = ts.iter().position(|t| t.abs() < 1e-3).unwrap_or(20);
    let mut t = ts[first_small_t];
    let mut sm = t;
    let mut amn = a - (first_small_t + 1) as f64;
    while t.abs() >= acc {
        amn -= 1.0;
        t *= amn / x;
        sm += t;
    }
    for j in (0..first_small_t).rev() {
        sm += ts[j];
    }
    let q = (rgammax(a, x) / x) * (1.0 + sm);
    (1.0 - q, q)
}

/// Distributions.jl `cdf(Chisq(2), x)`: StatsFuns.jl
/// `gammacdf(1, 2, x) = first(gamma_inc(1, max(0, x) / 2))`.
pub fn chisq2_cdf(x: f64) -> f64 {
    gamma_inc_a1(max(0.0, x) / 2.0).0
}

/// `dmtest_modified(e1, e2, h)`: the Diebold-Mariano test of equal
/// accuracy of two forecasts (Diebold and Mariano 1995), with the
/// correction of Harvey, Leybourne and Newbold (1997), on the loss
/// differential `d = e1² − e2²`. For `h > 1` the variance of `d` is the
/// Newey-West estimate with `h − 1` autocovariances (each `cov(d[1:end-i],
/// d[1+i:end]) / n`); if that is negative, the sample variance with `h = 1`.
/// Returns the statistic `DM` and the two-sided p-value
/// `2 Φ(−|DM|)`.
///
/// Returns an error if `e1` and `e2` have different lengths or if
/// `h > n` (Julia throws).
pub fn dmtest_modified(e1: &[f64], e2: &[f64], h: i64) -> Result<(f64, f64), String> {
    if e1.len() != e2.len() {
        return Err("Vectors should be of equal length".into());
    }
    let n = e1.len();
    let d: Vec<f64> = e1
        .iter()
        .zip(e2)
        .map(|(a, b)| a.abs() * a.abs() - b.abs() * b.abs())
        .collect();
    let mut h = h;
    let mut varD = if h > 1 {
        let gamma = autocovariances(&d, h as usize - 1)?;
        gamma[0] + 2.0 * sum(&gamma[1..h as usize])
    } else {
        var(&d)
    };
    if varD < 0.0 {
        h = 1;
        varD = var(&d);
    }
    let nf = n as f64;
    let k = (((n as i64 + 1 - 2 * h) as f64 + (h * (h - 1)) as f64 / nf) / nf).sqrt();
    let DM = (mean(&d) / (varD / nf).sqrt()) * k;
    let p_value = 2.0 * normcdf(-DM.abs());
    Ok((DM, p_value))
}

/// `[cov(d[1:(end - i)], d[(1 + i):end]) for i in 0:maxlag] / n`.
fn autocovariances(d: &[f64], maxlag: usize) -> Result<Vec<f64>, String> {
    let n = d.len();
    (0..=maxlag)
        .map(|i| {
            if i >= n {
                return Err("covariance only defined for non-empty vectors".to_string());
            }
            Ok(cov(&d[..n - i], &d[i..])? / n as f64)
        })
        .collect()
}

/// `bias_ttest(errors, h)`: t-test of `E[e] = 0` for forecast errors with
/// the Newey-West variance of [`dmtest_modified`], its lags capped at
/// `n − 2`. Returns the t-statistic and the two-sided p-value from the
/// standard normal distribution; `(NaN, NaN)` for fewer than two errors.
pub fn bias_ttest(errors: &[f64], h: i64) -> (f64, f64) {
    let n = errors.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let d = errors;
    let maxlag = (h - 1).min(n as i64 - 2);
    let mut varD = if maxlag >= 1 {
        let gamma =
            autocovariances(d, maxlag as usize).expect("the lags are shorter than the errors");
        gamma[0] + 2.0 * sum(&gamma[1..])
    } else {
        var(d)
    };
    if varD < 0.0 {
        varD = var(d);
    }
    let t_stat = mean(d) / (varD / n as f64).sqrt();
    let p_value = 2.0 * normcdf(-t_stat.abs());
    (t_stat, p_value)
}

/// Julia `inv(A)` of a 2×2 matrix: the inverse of the diagonal or of the
/// triangle if `A` is diagonal or triangular, otherwise LAPACK's LU
/// factorization with partial pivoting (`getrf`) and inversion (`getri`) in
/// the order of the reference implementation.
///
/// Returns an error if a pivot is exactly 0 (Julia throws a
/// `SingularException`).
fn inv2(A: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], String> {
    let [[a, b], [c, d]] = A;
    if b == 0.0 && c == 0.0 {
        return Ok([[1.0 / a, 0.0], [0.0, 1.0 / d]]);
    }
    if c == 0.0 {
        // trtri, upper triangle.
        let (i11, i22) = (1.0 / a, 1.0 / d);
        return Ok([[i11, (i11 * b) * -i22], [0.0, i22]]);
    }
    if b == 0.0 {
        // trtri, lower triangle.
        let (i11, i22) = (1.0 / a, 1.0 / d);
        return Ok([[i11, 0.0], [(i22 * c) * -i11, i22]]);
    }
    // getrf: pivot in the first column, multiplier by the reciprocal.
    let swap = c.abs() > a.abs();
    let [[a, b], [c, d]] = if swap { [[c, d], [a, b]] } else { A };
    if a == 0.0 {
        return Err("the matrix is singular".into());
    }
    let l = c * (1.0 / a);
    let u22 = d + -b * l;
    if u22 == 0.0 {
        return Err("the matrix is singular".into());
    }
    // trtri on U, then getri: column 1 minus column 2 times l.
    let (i11, i22) = (1.0 / a, 1.0 / u22);
    let i12 = (i11 * b) * -i22;
    let mut inv = [[i11 + -l * i12, i12], [-l * i22, i22]];
    if swap {
        for row in &mut inv {
            row.swap(0, 1);
        }
    }
    Ok(inv)
}

/// `mztest(y, x)`: the Mincer-Zarnowitz test of forecasts `x` of the
/// values `y`: least squares `y = c + β x` with `inv(X'X) X'y`, and the
/// Wald test of `c = 0, β = 1` with `σ² inv(X'X)` as the covariance of the
/// coefficients. Returns `c`, `β` and the p-value of the χ² distribution
/// with 2 degrees of freedom.
///
/// Returns an error if `y` and `x` have different lengths or `X'X` is
/// singular (Julia throws).
pub fn mztest(y: &[f64], x: &[f64]) -> Result<(f64, f64, f64), String> {
    if y.len() != x.len() {
        return Err(format!(
            "{} values and {} forecasts (Julia throws a DimensionMismatch)",
            y.len(),
            x.len()
        ));
    }
    let n = x.len();
    let ones = vec![1.0; n];
    // X'X and X'y, BLAS in Julia.
    let xx = [
        [dot(&ones, &ones), dot(&ones, x)],
        [dot(x, &ones), dot(x, x)],
    ];
    let xy = [dot(&ones, y), dot(x, y)];
    let xxi = inv2(xx)?;
    let beta = [
        xxi[0][0] * xy[0] + xxi[0][1] * xy[1],
        xxi[1][0] * xy[0] + xxi[1][1] * xy[1],
    ];
    let k = 2.0;
    let sigma2 = sum_by(n, |t| {
        let r = y[t] - (beta[0] + beta[1] * x[t]);
        r * r
    }) / (n as f64 - k);
    let var_beta = xxi.map(|row| row.map(|v| sigma2 * v));
    let coef_diff = [beta[0] - 0.0, beta[1] - 1.0];
    // coef_diff' * inv(var_β) * coef_diff = (coef_diff' * B) * coef_diff.
    let B = inv2(var_beta)?;
    let w = [
        B[0][0] * coef_diff[0] + B[1][0] * coef_diff[1],
        B[0][1] * coef_diff[0] + B[1][1] * coef_diff[1],
    ];
    let test_statistic = w[0] * coef_diff[0] + w[1] * coef_diff[1];
    let p_value = 1.0 - chisq2_cdf(test_statistic);
    Ok((beta[0], beta[1], p_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_ttest_as_in_the_julia_tests() {
        // BeforeIT test/utils/analysis_utils.jl, "bias_ttest".
        let errors = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1];
        let (t, p) = bias_ttest(&errors, 1);
        assert!(t.abs() < 1.0e-8);
        assert!((p - 1.0).abs() <= 1.490_116_119_384_765_6e-8);
        let biased: Vec<f64> = (1..=20)
            .map(|i| 1.0 + 0.01 * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!(bias_ttest(&biased, 1).1 < 0.05);
        let _ = bias_ttest(&[1.0, -1.0, 1.0, -1.0], 2);
        let _ = bias_ttest(&[0.5, -0.5], 12);
        let (t, p) = bias_ttest(&[0.3], 1);
        assert!(t.is_nan() && p.is_nan());
    }

    #[test]
    fn normcdf_and_chisq() {
        assert_eq!(normcdf(0.0), 0.5);
        assert_eq!(chisq2_cdf(0.0), 0.0);
        assert_eq!(chisq2_cdf(-1.0), 0.0);
        assert_eq!(chisq2_cdf(f64::INFINITY), 1.0);
        assert!(chisq2_cdf(f64::NAN).is_nan());
        // P(1, x) = 1 − e^(−x), so cdf(Chisq(2), x) = 1 − e^(−x/2), in all
        // three branches.
        for x in [0.01, 0.5, 1.9, 2.0, 10.0, 61.9, 62.0, 100.0] {
            let exact = -libm::expm1(-x / 2.0);
            let got = chisq2_cdf(x);
            assert!((got - exact).abs() <= 1e-14 * exact, "{x}: {got} {exact}");
        }
    }

    #[test]
    fn inv2_inverts() {
        for A in [
            [[4.0, 1.0], [1.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 0.0], [0.0, 5.0]],
            [[2.0, 1.0], [0.0, 5.0]],
            [[2.0, 0.0], [1.0, 5.0]],
        ] {
            let B = inv2(A).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    let e = A[i][0] * B[0][j] + A[i][1] * B[1][j];
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((e - want).abs() < 1e-15, "{A:?}");
                }
            }
        }
        assert!(inv2([[1.0, 2.0], [2.0, 4.0]]).is_err());
    }
}
