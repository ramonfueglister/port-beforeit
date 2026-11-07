//! Julia semantics that differ from Rust's standard library. The Julia code
//! relies on them, so the port uses the same rules.

/// Julia `min(x, y)` for floats: NaN if either argument is NaN (Rust's
/// `f64::min` returns the other one), and −0 is less than +0.
#[inline]
pub fn min(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x < y {
        x
    } else if y < x {
        y
    } else if x.is_sign_negative() {
        x
    } else {
        y
    }
}

/// Julia `max(x, y)` for floats: NaN if either argument is NaN, and +0 is
/// greater than −0.
#[inline]
pub fn max(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x > y {
        x
    } else if y > x || x.is_sign_negative() {
        y
    } else {
        x
    }
}

/// `pos(x)` (BeforeIT `src/utils/positive.jl`): negative numbers and NaN
/// become 0.
#[inline]
pub fn pos(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 { 0.0 } else { x }
}

/// `neg(x)` (BeforeIT `src/utils/positive.jl`): positive numbers and NaN
/// become 0.
#[inline]
pub fn neg(x: f64) -> f64 {
    if x.is_nan() || x > 0.0 { 0.0 } else { x }
}

/// `pos!(A)` (BeforeIT `src/utils/positive.jl`): NaN becomes 0, every other
/// number `max(0, x)`. Unlike [`pos`], −0 becomes +0.
pub fn pos_in_place(a: &mut [f64]) {
    for x in a {
        *x = if x.is_nan() { 0.0 } else { max(0.0, *x) };
    }
}

/// `neg!(A)` (BeforeIT `src/utils/positive.jl`): NaN becomes 0, every other
/// number `min(0, x)`, which gives the same as [`neg`].
pub fn neg_in_place(a: &mut [f64]) {
    for x in a {
        *x = if x.is_nan() { 0.0 } else { min(0.0, *x) };
    }
}

/// `matlab_round(x)` (BeforeIT `src/utils/positive.jl`), Julia
/// `round(x, RoundNearestTiesUp)`: to the nearest integer, ties towards
/// +∞, with the formula of Julia's `Base` (`floatfuncs.jl`).
#[inline]
pub fn matlab_round(x: f64) -> f64 {
    // eps(0.5) = 2^-53.
    let e = f64::EPSILON / 2.0;
    ((x + (0.25 - e)) + (0.25 + e)).floor().copysign(x)
}

/// `zero_to_one(x)` (BeforeIT `src/utils/positive.jl`): 0 becomes 1.
#[inline]
pub fn zero_to_one(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { x }
}

/// Julia `round(x)`: to the nearest integer, ties to even.
#[inline]
pub fn round(x: f64) -> f64 {
    x.round_ties_even()
}

/// Julia `round(Int, x)`: to the nearest integer, ties to even, and an
/// error (Julia throws an `InexactError`) if the result is not a finite
/// number that fits an `Int`.
pub(crate) fn round_int(x: f64) -> Result<i64, String> {
    let r = round(x);
    // i64 covers [-2^63, 2^63), and both bounds are exact as f64.
    let min = i64::MIN as f64;
    if r >= min && r < -min {
        Ok(r as i64)
    } else {
        Err(format!("cannot round {x} to an integer"))
    }
}

/// Julia `sum` of a list of floats, in the order of Julia's `mapreduce`
/// (Base `reduce.jl`): lists shorter than 16 from left to right, longer ones
/// pairwise, split in halves down to blocks of at most 1024 numbers, which
/// are added from left to right. Within a block Julia lets the compiler
/// reorder the additions (`@simd`), so the last digits can still differ.
pub fn sum(x: &[f64]) -> f64 {
    sum_by(x.len(), |i| x[i])
}

/// Julia `sum` over `f(0), …, f(n − 1)`, for example `sum(a .* b)`, in the
/// order of [`sum`].
pub(crate) fn sum_by(n: usize, f: impl Fn(usize) -> f64) -> f64 {
    match n {
        0 => 0.0,
        1 => f(0),
        n if n < 16 => sequential(&f, 0, n - 1),
        n => pairwise(&f, 0, n - 1),
    }
}

/// `f(first) + f(first + 1) + …` up to `f(last)`, from left to right.
fn sequential(f: &impl Fn(usize) -> f64, first: usize, last: usize) -> f64 {
    let mut v = f(first) + f(first + 1);
    for i in first + 2..=last {
        v += f(i);
    }
    v
}

/// Base `mapreduce_impl` with Julia's block size of 1024.
fn pairwise(f: &impl Fn(usize) -> f64, first: usize, last: usize) -> f64 {
    if first == last {
        f(first)
    } else if last - first < 1024 {
        sequential(f, first, last)
    } else {
        let mid = first + ((last - first) >> 1);
        pairwise(f, first, mid) + pairwise(f, mid + 1, last)
    }
}

/// Dot product, summed from left to right, for Julia's `dot`.
#[inline]
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_and_max_propagate_nan() {
        assert!(min(f64::NAN, 1.0).is_nan());
        assert!(min(1.0, f64::NAN).is_nan());
        assert!(max(f64::NAN, 1.0).is_nan());
        assert!(max(1.0, f64::NAN).is_nan());
        assert_eq!(min(2.0, 1.0), 1.0);
        assert_eq!(max(2.0, 1.0), 2.0);
        assert!(min(0.0, -0.0).is_sign_negative());
        assert!(max(-0.0, 0.0).is_sign_positive());
    }

    #[test]
    fn pos_and_zero_to_one() {
        assert_eq!(pos(-1.0), 0.0);
        assert_eq!(pos(f64::NAN), 0.0);
        assert_eq!(pos(2.5), 2.5);
        assert_eq!(zero_to_one(0.0), 1.0);
        assert_eq!(zero_to_one(-0.0), 1.0);
        assert_eq!(zero_to_one(3.0), 3.0);
    }

    #[test]
    fn neg_and_in_place_versions() {
        assert_eq!(neg(1.0), 0.0);
        assert_eq!(neg(-1.0), -1.0);
        assert_eq!(neg(f64::NAN), 0.0);
        let mut a = [1.0, -0.0, f64::NAN, -1.0];
        pos_in_place(&mut a);
        assert_eq!(a, [1.0, 0.0, 0.0, 0.0]);
        assert!(a[1].is_sign_positive());
        let mut a = [1.0, -0.0, f64::NAN, -1.0];
        neg_in_place(&mut a);
        assert_eq!(a, [0.0, 0.0, 0.0, -1.0]);
        assert!(a[1].is_sign_negative());
        assert!(a[2].is_sign_positive());
        assert!(pos(-0.0).is_sign_negative());
    }

    #[test]
    fn matlab_round_ties_up() {
        assert_eq!(matlab_round(0.5), 1.0);
        assert_eq!(matlab_round(1.5), 2.0);
        assert_eq!(matlab_round(2.5), 3.0);
        assert_eq!(matlab_round(-1.5), -1.0);
        assert_eq!(matlab_round(-2.5), -2.0);
        assert_eq!(matlab_round(-2.6), -3.0);
        let r = matlab_round(-0.5);
        assert!(r == 0.0 && r.is_sign_negative());
        // The largest float below 0.5 and numbers without a fraction.
        assert_eq!(matlab_round(0.49999999999999994), 0.0);
        assert_eq!(matlab_round(4503599627370497.0), 4503599627370497.0);
        assert_eq!(matlab_round(1e300), 1e300);
    }

    #[test]
    fn round_ties_to_even() {
        assert_eq!(round(0.5), 0.0);
        assert_eq!(round(1.5), 2.0);
        assert_eq!(round(2.5), 2.0);
        assert_eq!(round(-0.5), -0.0);
        assert_eq!(round(2.4), 2.0);
        assert_eq!(round(2.6), 3.0);
    }

    #[test]
    fn sum_adds_in_julias_order() {
        assert_eq!(sum(&[]), 0.0);
        assert_eq!(sum(&[2.5]), 2.5);
        // From left to right: (1e16 + 1) + 1 rounds back to 1e16 twice.
        let mut x = vec![1e16, 1.0, 1.0];
        assert_eq!(sum(&x), 1e16);
        // Pairwise: the two halves of 2048 numbers are added separately, so
        // the ones in the second half are not lost.
        x = vec![1.0; 2048];
        x[0] = 1e16;
        assert_eq!(sum(&x), 1e16 + 1024.0);
        assert_eq!(x.iter().sum::<f64>(), 1e16);
        assert_eq!(sum_by(3, |i| [1.0, 2.0, 3.0][i] * 2.0), 12.0);
    }

    #[test]
    fn round_int_fails_like_julia() {
        assert_eq!(round_int(2.5), Ok(2));
        assert_eq!(round_int(-3.5), Ok(-4));
        let two_63 = 2f64.powi(63);
        assert_eq!(round_int(-two_63), Ok(i64::MIN));
        assert!(round_int(two_63).is_err());
        assert!(round_int(f64::NAN).is_err());
        assert!(round_int(f64::INFINITY).is_err());
    }
}
