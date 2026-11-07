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

/// Result of `rfvar3(ydata, 1, ones(T, 1))` for one series: least squares
/// for y(t) = By · y(t−1) + Bx, with the residuals u.
pub struct Ar1 {
    pub By: f64,
    pub Bx: f64,
    pub u: Vec<f64>,
}

/// `rfvar3` with one lag and a constant. Solved with the pseudoinverse as in
/// the Julia code: singular values up to `100 · eps` count as 0.
pub fn rfvar3(ydata: &[f64]) -> Ar1 {
    let T = ydata.len();
    let lag = &ydata[..T.saturating_sub(1)];
    let ones = vec![1.0; lag.len()];
    let y = &ydata[T.min(1)..];
    let svd = svd_mx2(lag, &ones);
    let dfx = svd.s.iter().filter(|&&s| s > 100.0 * f64::EPSILON).count();
    let mut B = [0.0, 0.0];
    for k in 0..dfx {
        let uy = dot(&svd.u[k], y);
        let di = 1.0 / svd.s[k];
        for (r, b) in B.iter_mut().enumerate() {
            *b += svd.v[k][r] * di * uy;
        }
    }
    let u = lag
        .iter()
        .zip(y)
        .map(|(&x, &yt)| yt - (x * B[0] + B[1]))
        .collect();
    Ar1 {
        By: B[0],
        Bx: B[1],
        u,
    }
}

/// `estimate(ydata)`: coefficients of the AR(1) estimate and a shock with
/// mean 0 and the standard deviation of the residuals (0 if that is not
/// positive).
pub fn estimate<R: Randomness + ?Sized>(ydata: &[f64], rng: &mut R) -> (f64, f64, f64) {
    let ar = rfvar3(ydata);
    let n = ar.u.len() as f64;
    let mean = ar.u.iter().sum::<f64>() / n;
    let cov = ar.u.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    let sigma = max(0.0, cov).sqrt();
    let epsilon = if sigma > 0.0 {
        sigma * rng.randn()
    } else {
        0.0
    };
    (ar.By, ar.Bx, epsilon)
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

    /// BeforeIT `test/utils/nfvar3_and_estimate.jl`, "nfvar3", without
    /// `xxi`, which this port does not compute.
    #[test]
    fn rfvar3_on_a_straight_line() {
        let ar = rfvar3(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(close(ar.Bx, 1.0, 1e-8), "{}", ar.Bx);
        assert!(close(ar.By, 1.0, 1e-8), "{}", ar.By);
        assert!(ar.u.iter().all(|u| u.abs() < 1e-14), "{:?}", ar.u);
    }

    /// BeforeIT `test/utils/nfvar3_and_estimate.jl`, values for
    /// `estimate_for_calibration_script(dummy_series)`.
    #[test]
    fn rfvar3_matches_beforeit() {
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
        let ar = rfvar3(&series);
        assert!(close(ar.By, 0.984009645632709, 1e-8), "{}", ar.By);
        assert!(close(ar.Bx, 0.161039024858073, 1e-8), "{}", ar.Bx);
        let n = ar.u.len() as f64;
        let mean = ar.u.iter().sum::<f64>() / n;
        let sigma = (ar.u.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt();
        assert!(close(sigma, 0.004012824377165, 1e-8), "{sigma}");
        let diff: f64 =
            ar.u.iter()
                .zip(residuals)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
        let size = residuals.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(diff <= 1.5e-8 * size, "{diff}");
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
