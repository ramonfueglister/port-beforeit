//! Estimation for the expectations (A.6, A.10) and the correlated shocks to
//! euro area GDP, exports and imports (A.71, A.76, A.81). BeforeIT
//! `src/utils/estimate.jl`, `src/utils/nfvar3.jl`, `src/utils/julia_svd.jl`
//! and `src/utils/epsilon.jl`.

use crate::julia::{dot, max};
use crate::random::Randomness;

/// Singular value decomposition of a matrix with two columns, Julia
/// `svd_mx2`, computed analytically. The columns in `u` have one entry per
/// row of the matrix; `v` holds the columns of the rotation `[c -s; s c]`.
struct Svd2 {
    u: Vec<Vec<f64>>,
    s: Vec<f64>,
    v: [[f64; 2]; 2],
}

fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn svd_mx2(c1: &[f64], c2: &[f64]) -> Svd2 {
    let m = c1.len();
    let alpha = dot(c1, c1);
    let beta = dot(c2, c2);
    let gamma = dot(c1, c2);
    let theta = 0.5 * libm::atan2(2.0 * gamma, alpha - beta);
    let (c, s) = (libm::cos(theta), libm::sin(theta));
    // Columns of V: (c, s) and (−s, c).
    let v = [[c, s], [-s, c]];
    let w1: Vec<f64> = (0..m).map(|k| c1[k] * c + c2[k] * s).collect();
    let w2: Vec<f64> = (0..m).map(|k| c1[k] * -s + c2[k] * c).collect();
    if m == 1 {
        let s1 = norm(&w1);
        let u1 = if s1 > 0.0 {
            vec![w1[0] / s1]
        } else {
            vec![1.0]
        };
        return Svd2 {
            u: vec![u1],
            s: vec![s1],
            v,
        };
    }
    let s1 = norm(&w1);
    let s2 = norm(&w2);
    let u1: Vec<f64> = if s1 > 1.0e-15 {
        w1.iter().map(|x| x / s1).collect()
    } else {
        let mut e = vec![0.0; m];
        e[0] = 1.0;
        e
    };
    let u2: Vec<f64> = if s2 > 1.0e-15 {
        w2.iter().map(|x| x / s2).collect()
    } else {
        // Gram-Schmidt, so that U stays orthonormal.
        let mut k_min = 0;
        for k in 1..m {
            if u1[k].abs() < u1[k_min].abs() {
                k_min = k;
            }
        }
        let mut e = vec![0.0; m];
        e[k_min] = 1.0;
        let d = dot(&u1, &e);
        for k in 0..m {
            e[k] -= d * u1[k];
        }
        let n = norm(&e);
        e.iter().map(|x| x / n).collect()
    };
    Svd2 {
        u: vec![u1, u2],
        s: vec![s1, s2],
        v,
    }
}

/// Singular value decomposition of a matrix with the columns `a` by
/// one-sided Jacobi rotations (Hestenes), for Julia's `svd`, which calls
/// LAPACK. Returns the singular values in descending order, as LAPACK does,
/// with the matching columns of U and V; a column of U whose singular value
/// is 0 stays 0. The singular values agree with LAPACK's up to rounding.
fn svd_jacobi(a: &[Vec<f64>]) -> Svd {
    let n = a.len();
    let mut u: Vec<Vec<f64>> = a.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let rotate = |x: &mut Vec<Vec<f64>>, p: usize, q: usize, c: f64, s: f64| {
        for k in 0..x[p].len() {
            let (xp, xq) = (x[p][k], x[q][k]);
            x[p][k] = c * xp - s * xq;
            x[q][k] = s * xp + c * xq;
        }
    };
    for _sweep in 0..100 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let alpha = dot(&u[p], &u[p]);
                let beta = dot(&u[q], &u[q]);
                let gamma = dot(&u[p], &u[q]);
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * alpha.sqrt() * beta.sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + libm::hypot(1.0, zeta));
                let c = 1.0 / libm::hypot(1.0, t);
                let s = c * t;
                rotate(&mut u, p, q, c, s);
                rotate(&mut v, p, q, c, s);
            }
        }
        if !rotated {
            break;
        }
    }
    let sigma: Vec<f64> = u.iter().map(|c| norm(c)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| sigma[j].total_cmp(&sigma[i]));
    Svd {
        u: order
            .iter()
            .map(|&j| {
                let s = sigma[j];
                u[j].iter()
                    .map(|x| if s > 0.0 { x / s } else { 0.0 })
                    .collect()
            })
            .collect(),
        s: order.iter().map(|&j| sigma[j]).collect(),
        v: order.iter().map(|&j| v[j].clone()).collect(),
    }
}

/// A singular value decomposition: the columns of U (one entry per row of
/// the matrix), the singular values and the columns of V.
struct Svd {
    u: Vec<Vec<f64>>,
    s: Vec<f64>,
    v: Vec<Vec<f64>>,
}

/// Result of [`rfvar3`], Julia's named tuple `(By, Bx, u, xxi, snglty)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Var {
    /// Coefficients of the lags: `By[e][v][l]` is the coefficient of
    /// variable `v` at lag `l + 1` in equation `e` (Julia `By[e, v, l]`).
    pub By: Vec<Vec<Vec<f64>>>,
    /// Coefficients of the exogenous variables: `Bx[e][x]` (Julia
    /// `Bx[e, x]`); empty without exogenous variables (Julia `[]`).
    pub Bx: Vec<Vec<f64>>,
    /// Residuals: `u[t][e]` for the rows `lags..T` of the data.
    pub u: Vec<Vec<f64>>,
    /// `(X'X)⁻¹` from the pseudoinverse of the regressors `X`.
    pub xxi: Vec<Vec<f64>>,
    /// Number of columns by which `X` falls short of full column rank.
    pub snglty: usize,
}

/// `rfvar3(ydata, lags, xdata)` (BeforeIT `src/utils/nfvar3.jl`): least
/// squares for a vector autoregression of the series in the columns of
/// `ydata` (one row per quarter) with `lags` lags and the exogenous
/// variables in the columns of `xdata`. As in the Julia code, it solves with
/// the pseudoinverse, counting singular values up to `100 · eps` as 0 and
/// using the analytic SVD of Julia's `svd_mx2` when the regressors have two
/// columns. Like the Julia code, it prints "Mismatch of x and y data
/// lengths" if `xdata` and `ydata` have different numbers of rows.
///
/// # Panics
///
/// If `xdata` has fewer rows than `ydata` (Julia throws a `BoundsError`).
pub fn rfvar3(ydata: &[Vec<f64>], lags: usize, xdata: &[Vec<f64>]) -> Var {
    let T = ydata.len();
    let nvar = ydata.first().map_or(0, |r| r.len());
    let nx = xdata.first().map_or(0, |r| r.len());
    let nox = nx == 0;
    if xdata.len() != T {
        println!("Mismatch of x and y data lengths");
    }
    let smpl = lags..T.max(lags);
    let k = nvar * lags + nx;
    // The columns of the regressors: the lags of every variable (variables
    // first, then lags, as Julia's `reshape`), then the exogenous variables.
    let mut X: Vec<Vec<f64>> = Vec::with_capacity(k);
    for l in 0..lags {
        for v in 0..nvar {
            X.push(smpl.clone().map(|t| ydata[t - l - 1][v]).collect());
        }
    }
    for x in 0..nx {
        X.push(smpl.clone().map(|t| xdata[t][x]).collect());
    }
    let y: Vec<Vec<f64>> = (0..nvar)
        .map(|e| smpl.clone().map(|t| ydata[t][e]).collect())
        .collect();
    let svd = if k == 2 {
        let s = svd_mx2(&X[0], &X[1]);
        Svd {
            u: s.u,
            s: s.s,
            v: s.v.iter().map(|c| c.to_vec()).collect(),
        }
    } else {
        svd_jacobi(&X)
    };
    let dfx = svd.s.iter().filter(|&&s| s > 100.0 * f64::EPSILON).count();
    let di: Vec<f64> = svd.s[..dfx].iter().map(|s| 1.0 / s).collect();
    // B = (vr · diagm(di)) · (vl' · y), one column per equation.
    let mut B = vec![vec![0.0; nvar]; k];
    for e in 0..nvar {
        for j in 0..dfx {
            let uy = dot(&svd.u[j], &y[e]);
            for (r, row) in B.iter_mut().enumerate() {
                row[e] += svd.v[j][r] * di[j] * uy;
            }
        }
    }
    let u = (0..smpl.len())
        .map(|t| {
            (0..nvar)
                .map(|e| {
                    let fit = (1..k).fold(X[0][t] * B[0][e], |acc, i| acc + X[i][t] * B[i][e]);
                    y[e][t] - fit
                })
                .collect()
        })
        .collect();
    let xxi = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| {
                    (0..dfx)
                        .map(|m| svd.v[m][i] * di[m] * (svd.v[m][j] * di[m]))
                        .sum()
                })
                .collect()
        })
        .collect();
    let By = (0..nvar)
        .map(|e| {
            (0..nvar)
                .map(|v| (0..lags).map(|l| B[v + nvar * l][e]).collect())
                .collect()
        })
        .collect();
    let Bx = if nox {
        Vec::new()
    } else {
        (0..nvar)
            .map(|e| (0..nx).map(|x| B[nvar * lags + x][e]).collect())
            .collect()
    };
    Var {
        By,
        Bx,
        u,
        xxi,
        snglty: k - dfx,
    }
}

/// `rfvar3(ydata, 1, ones(T, 1))` for one series, Julia's `ydata[:, :]`:
/// the AR(1) with a constant of [`estimate`].
fn ar1(ydata: &[f64]) -> Var {
    let y: Vec<Vec<f64>> = ydata.iter().map(|&x| vec![x]).collect();
    rfvar3(&y, 1, &vec![vec![1.0]; ydata.len()])
}

/// Julia `cov(var.u)` for residuals of one equation: the sample variance,
/// divided by `n − 1`.
fn variance(u: &[Vec<f64>]) -> f64 {
    let n = u.len() as f64;
    let mean = u.iter().map(|r| r[0]).sum::<f64>() / n;
    u.iter().map(|r| (r[0] - mean) * (r[0] - mean)).sum::<f64>() / (n - 1.0)
}

/// `estimate(ydata)`: coefficients of the AR(1) estimate and a shock with
/// mean 0 and the standard deviation of the residuals (0 if that is not
/// positive).
pub fn estimate<R: Randomness + ?Sized>(ydata: &[f64], rng: &mut R) -> (f64, f64, f64) {
    let var = ar1(ydata);
    let sigma = max(0.0, variance(&var.u)).sqrt();
    let epsilon = if sigma > 0.0 {
        sigma * rng.randn()
    } else {
        0.0
    };
    (var.By[0][0][0], var.Bx[0][0], epsilon)
}

/// `estimate_for_calibration_script(ydata)`: coefficients `alpha` and
/// `beta` of the AR(1) estimate, the standard deviation `sigma` of the
/// residuals and the residuals.
pub fn estimate_for_calibration_script(ydata: &[f64]) -> (f64, f64, f64, Vec<f64>) {
    let var = ar1(ydata);
    let sigma = variance(&var.u).sqrt();
    let epsilon = var.u.iter().map(|r| r[0]).collect();
    (var.By[0][0][0], var.Bx[0][0], sigma, epsilon)
}

/// `estimate_taylor_rule(r_bar, pi_EA, gamma_EA)`: the parameters `rho`,
/// `r_star`, `xi_pi`, `xi_gamma` and `pi_star` of the Taylor rule (A.69)
/// from a regression of the policy rate on its last value and on the
/// exogenous series `pi_EA` and `gamma_EA` without a constant. `pi_star` is
/// fixed at 2% a year. Only the first `r_bar.len()` entries of `pi_EA` and
/// `gamma_EA` are used.
///
/// # Panics
///
/// If `pi_EA` or `gamma_EA` is shorter than `r_bar` (Julia throws).
pub fn estimate_taylor_rule(
    r_bar: &[f64],
    pi_EA: &[f64],
    gamma_EA: &[f64],
) -> (f64, f64, f64, f64, f64) {
    let n = r_bar.len();
    let ydata: Vec<Vec<f64>> = r_bar.iter().map(|&x| vec![x]).collect();
    let exo: Vec<Vec<f64>> = (0..n).map(|t| vec![pi_EA[t], gamma_EA[t]]).collect();
    let var = rfvar3(&ydata, 1, &exo);
    let alpha = var.By[0][0][0];
    let (gamma_1, gamma_2) = (var.Bx[0][0], var.Bx[0][1]);
    let rho = alpha;
    let xi_pi = gamma_1 / (1.0 - rho);
    let xi_gamma = gamma_2 / (1.0 - rho);
    let pi_star = libm::pow(0.02 + 1.0, 1.0 / 4.0) - 1.0;
    let r_star = pi_star * (xi_pi - 1.0);
    (rho, r_star, xi_pi, xi_gamma, pi_star)
}

/// `estimate_next_value(data)`: next value of the series by the AR(1) rule.
pub fn estimate_next_value<R: Randomness + ?Sized>(data: &[f64], rng: &mut R) -> f64 {
    let (alpha, beta, epsilon) = estimate(data, rng);
    alpha * data[data.len() - 1] + beta + epsilon
}

/// Lower triangular `L` of the Cholesky decomposition `C = L·Lᵀ`. Like
/// Julia's `cholesky`, this fails for a matrix that is not exactly symmetric
/// or not positive definite.
pub fn cholesky_lower(C: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3], String> {
    for i in 0..3 {
        for j in 0..i {
            if C[i][j] != C[j][i] {
                return Err("covariance matrix C is not symmetric".into());
            }
        }
    }
    let mut L = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut s = C[i][j];
            for k in 0..j {
                s -= L[i][k] * L[j][k];
            }
            if i == j {
                if s.is_nan() || s <= 0.0 {
                    return Err("covariance matrix C is not positive definite".into());
                }
                L[i][i] = s.sqrt();
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    Ok(L)
}

/// The factor `L` used by [`epsilon`]: `None` if the entries of `C` sum to 0
/// (within 1e-8, Julia `isapprox(sum(C), 0, atol = 1e-8)`), otherwise the
/// lower Cholesky factor of `C`.
pub fn shock_factor(C: &[[f64; 3]; 3]) -> Result<Option<[[f64; 3]; 3]>, String> {
    // Column by column, the order of Julia's `sum` over a matrix.
    let sum: f64 = (0..3).flat_map(|j| (0..3).map(move |i| C[i][j])).sum();
    if sum.abs() <= 1.0e-8 {
        return Ok(None);
    }
    cholesky_lower(C).map(Some)
}

/// `epsilon(C)`: three correlated shocks `L·e`, with `e` standard normal and
/// `L` from [`shock_factor`], for euro area GDP, exports and imports (in the
/// order of the rows of `C`). All three are 0 if the entries of `C` sum to 0.
/// As in Julia, the factor is computed on every call.
pub fn epsilon<R: Randomness + ?Sized>(
    C: &[[f64; 3]; 3],
    rng: &mut R,
) -> Result<(f64, f64, f64), String> {
    let Some(L) = shock_factor(C)? else {
        return Ok((0.0, 0.0, 0.0));
    };
    let e = [rng.randn(), rng.randn(), rng.randn()];
    Ok((dot(&L[0], &e), dot(&L[1], &e), dot(&L[2], &e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Deterministic;

    fn close(a: f64, b: f64, rtol: f64) -> bool {
        (a - b).abs() <= rtol * a.abs().max(b.abs())
    }

    /// BeforeIT `test/utils/nfvar3_and_estimate.jl`, "nfvar3".
    #[test]
    fn rfvar3_on_a_straight_line() {
        let y: Vec<Vec<f64>> = [1.0, 2.0, 3.0, 4.0, 5.0].iter().map(|&x| vec![x]).collect();
        let var = rfvar3(&y, 1, &vec![vec![1.0]; 5]);
        assert!(close(var.Bx[0][0], 1.0, 1.5e-8), "{:?}", var.Bx);
        assert!(close(var.By[0][0][0], 1.0, 1.5e-8), "{:?}", var.By);
        let xxi = [[0.2, -0.5], [-0.5, 1.5]];
        let diff: f64 = (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| (var.xxi[i][j] - xxi[i][j]).powi(2))
            .sum::<f64>()
            .sqrt();
        // Julia `isapprox` for matrices: Frobenius norms, rtol = √eps.
        let size = (0.04f64 + 0.25 + 0.25 + 2.25).sqrt();
        assert!(diff <= 1.5e-8 * size, "{:?}", var.xxi);
        assert!(var.u.iter().all(|u| u[0].abs() < 1e-14), "{:?}", var.u);
        assert_eq!(var.snglty, 0);
    }

    /// BeforeIT `test/utils/nfvar3_and_estimate.jl`, values for
    /// `estimate_for_calibration_script(dummy_series)`.
    #[test]
    fn estimate_for_calibration_script_matches_beforeit() {
        let series = [
            9.567778424837963,
            9.574191416943812,
            9.580605059126121,
            9.588931171444848,
            9.600311304740346,
            9.604640931220233,
            9.613107726219265,
            9.618230372010803,
            9.626726085704139,
            9.642208926006326,
            9.641763844732605,
            9.64969091710685,
        ];
        let residuals = [
            -0.001633865231344,
            -0.001530669138557,
            0.000484357409141,
            0.003671515872391,
            -0.003197018579078,
            0.001009382201758,
            -0.002199379953346,
            0.001255600869955,
            0.008378576951367,
            -0.007301768521497,
            0.00106326811918,
        ];
        let (alpha, beta, sigma, epsilon) = estimate_for_calibration_script(&series);
        assert!(close(alpha, 0.984009645632709, 1.5e-8), "{alpha}");
        assert!(close(beta, 0.161039024858073, 1.5e-8), "{beta}");
        assert!(close(sigma, 0.004012824377165, 1.5e-8), "{sigma}");
        let diff: f64 = epsilon
            .iter()
            .zip(residuals)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let size = residuals.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(diff <= 1.5e-8 * size, "{diff}");
    }

    /// BeforeIT `test/utils/estimations.jl`: the policy rate 1, ..., 10 and
    /// the exogenous series 5, ..., 15 and 10, ..., 20, whose regressors have
    /// rank 2, so that the pseudoinverse decides.
    #[test]
    fn estimate_taylor_rule_matches_beforeit() {
        let r_bar: Vec<f64> = (1..=10).map(f64::from).collect();
        let pi_EA: Vec<f64> = (5..=15).map(f64::from).collect();
        let gamma_EA: Vec<f64> = (10..=20).map(f64::from).collect();
        let (rho, r_star, xi_pi, xi_gamma, pi_star) =
            estimate_taylor_rule(&r_bar, &pi_EA, &gamma_EA);
        assert!(close(rho, 0.733333333333333, 1e-5), "{rho}");
        assert!(close(r_star, 0.001240732893301, 1e-5), "{r_star}");
        assert!(close(xi_pi, 1.250000000000001, 1e-5), "{xi_pi}");
        assert!(close(xi_gamma, -0.250000000000001, 1e-5), "{xi_gamma}");
        assert!(close(pi_star, 0.004962931573204, 1e-5), "{pi_star}");
    }

    /// The Jacobi SVD against a matrix with known singular values, and
    /// `rfvar3` with two variables and two lags against exact data.
    #[test]
    fn general_svd_and_rfvar3() {
        // A 4 × 3 matrix with orthogonal columns of norms √3, √3 and √0.75.
        let a = vec![
            vec![1.0, 1.0, 1.0, 0.0],
            vec![1.0, -1.0, 0.0, 1.0],
            vec![0.5, 0.0, -0.5, -0.5],
        ];
        let svd = svd_jacobi(&a);
        let expected = [3f64.sqrt(), 3f64.sqrt(), 0.75f64.sqrt()];
        for (s, e) in svd.s.iter().zip(expected) {
            assert!((s - e).abs() < 1e-14, "{:?}", svd.s);
        }
        // y1(t) = 0.5 y1(t-1) + 0.1 y2(t-1) - 0.2 y1(t-2) + 1,
        // y2(t) = 0.3 y1(t-1) + 0.2 y2(t-2) + 2, from known starting values
        // and a disturbance that keeps the regressors of full rank.
        let mut y = vec![vec![1.0, 2.0], vec![1.5, 2.5]];
        for t in 2..40 {
            let d = if t % 3 == 0 { 0.1 } else { -0.05 };
            let (a, b) = (&y[t - 1], &y[t - 2]);
            let y1 = 0.5 * a[0] + 0.1 * a[1] - 0.2 * b[0] + 1.0 + d;
            let y2 = 0.3 * a[0] + 0.2 * b[1] + 2.0 - d;
            y.push(vec![y1, y2]);
        }
        let x: Vec<Vec<f64>> = (0..40)
            .map(|t| vec![1.0, if t % 3 == 0 { 0.1 } else { -0.05 }])
            .collect();
        let var = rfvar3(&y, 2, &x);
        let close_to = |a: f64, b: f64| (a - b).abs() < 1e-10;
        assert!(close_to(var.By[0][0][0], 0.5) && close_to(var.By[0][1][0], 0.1));
        assert!(close_to(var.By[0][0][1], -0.2) && close_to(var.By[0][1][1], 0.0));
        assert!(close_to(var.By[1][0][0], 0.3) && close_to(var.By[1][1][1], 0.2));
        assert!(close_to(var.Bx[0][0], 1.0) && close_to(var.Bx[0][1], 1.0));
        assert!(close_to(var.Bx[1][0], 2.0) && close_to(var.Bx[1][1], -1.0));
        assert!(var.u.iter().flatten().all(|u| u.abs() < 1e-10));
        assert_eq!((var.u.len(), var.snglty), (38, 0));
    }

    /// Random numbers that are all 1.
    struct Ones;

    impl Randomness for Ones {
        fn randn(&mut self) -> f64 {
            1.0
        }
        fn rand(&mut self) -> f64 {
            1.0
        }
        fn rand_below(&mut self, _: usize) -> usize {
            0
        }
    }

    /// BeforeIT `test/deterministic/epsilon.jl`: the values of the authors'
    /// MATLAB code with all random numbers set to 1.
    #[test]
    #[allow(clippy::approx_constant)] // the first value is √2, as printed there
    fn epsilon_matches_matlab() {
        let C = [[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]];
        let (e_Y_EA, e_E, e_I) = epsilon(&C, &mut Ones).unwrap();
        assert!(close(e_Y_EA, 1.4142135623731, 1.5e-8), "{e_Y_EA}");
        assert!(close(e_E, 1.52615733054913, 1.5e-8), "{e_E}");
        assert!(close(e_I, 1.29014186522609, 1.5e-8), "{e_I}");
        assert_eq!(epsilon(&[[0.0; 3]; 3], &mut Ones).unwrap(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn constant_series_gives_no_shock() {
        // The columns of X are parallel, so one singular value is 0.
        let (alpha, beta, epsilon) = estimate(&[2.0; 10], &mut Deterministic);
        assert!((alpha * 2.0 + beta - 2.0).abs() < 1e-12);
        assert_eq!(epsilon, 0.0);
    }

    #[test]
    fn cholesky_factor_reproduces_c() {
        let C = [[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]];
        let L = cholesky_lower(&C).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let s: f64 = (0..3).map(|k| L[i][k] * L[j][k]).sum();
                assert!((s - C[i][j]).abs() < 1e-14);
            }
        }
        assert!(cholesky_lower(&[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).is_err());
        assert!(cholesky_lower(&[[1.0, 0.1, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]).is_err());
    }
}
