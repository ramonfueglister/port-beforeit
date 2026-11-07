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

/// Julia `round(x; digits)` for `digits ≥ 0` (Base `floatfuncs.jl`,
/// `_round_digits` and `_round_invstep`): `round(x · 10^digits) /
/// 10^digits` with ties to even, where `10^digits` is Julia's
/// `10.0^digits` ([`powi`]); `x` itself if `x` or the result is not finite.
///
/// # Panics
///
/// If `digits > 308`, where `10^digits` is not finite and Julia takes
/// another formula, which is not ported.
pub fn round_digits(x: f64, digits: u32) -> f64 {
    assert!(
        digits <= 308,
        "round(x; digits) is ported for digits <= 308"
    );
    if !x.is_finite() {
        return x;
    }
    let invstep = powi(10.0, i64::from(digits));
    let y = round(x * invstep) / invstep;
    if y.is_finite() { y } else { x }
}

/// Julia `string(x)` for a `Float64`, which is also how `print`, string
/// interpolation and CSV.jl write it (Base `Ryu.writeshortest` with its
/// default options): the shortest digits that read back as `x`, in fixed
/// notation if the decimal point falls after at most 6 and before at most
/// 3 leading zeros (`123456.0`, `0.0001`), otherwise as `d.ddde±n`
/// (`1.0e6`, `1.0e-5`); always with a decimal point, and `NaN`, `Inf` and
/// `-Inf` for the special values. The shortest digits are those of Rust's
/// formatting, which, like Ryu, takes the shortest digits closest to `x`.
pub fn float_string(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Inf" } else { "-Inf" }.into();
    }
    // `{:e}` writes the shortest digits as d.ddd and the exponent.
    let s = format!("{:e}", x.abs());
    let (mantissa, exp) = s.split_once('e').expect("{:e} writes an exponent");
    let exp: i32 = exp.parse().expect("{:e} writes an integer exponent");
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let olength = digits.len() as i32;
    // Ryu's `pt = nexp + olength`: the number of digits before the decimal
    // point. Ryu also tests a remainder when `pt >= olength`; then x is an
    // integer below 10^6 that the test always passes.
    let pt = exp + 1;
    let mut out = String::with_capacity(digits.len() + 8);
    if x < 0.0 {
        out.push('-');
    }
    if -4 < pt && pt <= 6 {
        if pt <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-pt) as usize));
            out.push_str(&digits);
        } else if pt >= olength {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', (pt - olength) as usize));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..pt as usize]);
            out.push('.');
            out.push_str(&digits[pt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        out.push('.');
        out.push_str(if olength > 1 { &digits[1..] } else { "0" });
        out.push('e');
        out.push_str(&exp.to_string());
    }
    out
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

/// `(x * y, error)` with `x * y = product + error` exactly, Julia
/// `two_mul` on a processor with FMA instructions.
#[inline]
fn two_mul(x: f64, y: f64) -> (f64, f64) {
    let xy = x * y;
    (xy, libm::fma(x, y, -xy))
}

/// Julia `x^n` for a `Float64` and an integer (`^(x::Float64, n::Integer)`
/// of Julia 1.12, which `x^4` with a literal exponent also calls):
/// compensated power by squaring, `pow_body`. Julia's `muladd` is an FMA
/// on processors that have one; here it always is.
///
/// # Panics
///
/// If `n` is outside −4096..=24576, where Julia switches to its
/// floating-point power, which is not ported.
pub fn powi(x: f64, n: i64) -> f64 {
    assert!(
        (-4096..=24576).contains(&n),
        "x^n is ported for -4096 <= n <= 24576 only"
    );
    if n == 0 {
        return 1.0;
    }
    let (mut x, mut n) = (x, n);
    let mut y = 1.0;
    let mut xnlo = -0.0;
    let mut ynlo = 0.0;
    if n == 3 {
        return x * x * x;
    }
    if n < 0 {
        let rx = 1.0 / x;
        if n == -2 {
            return rx * rx;
        }
        if x.is_finite() {
            xnlo = -libm::fma(x, rx, -1.0) * rx;
        }
        x = rx;
        n = -n;
    }
    while n > 1 {
        if n & 1 > 0 {
            let err = libm::fma(y, xnlo, x * ynlo);
            (y, ynlo) = two_mul(x, y);
            ynlo += err;
        }
        let err = x * 2.0 * xnlo;
        (x, xnlo) = two_mul(x, x);
        xnlo += err;
        n >>= 1;
    }
    let err = libm::fma(y, xnlo, x * ynlo);
    if x.is_finite() && err.is_finite() {
        libm::fma(x, y, err)
    } else {
        x * y
    }
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

    #[test]
    fn powi_is_compensated() {
        assert_eq!(powi(2.0, 10), 1024.0);
        assert_eq!(powi(3.0, 0), 1.0);
        assert_eq!(powi(2.0, -2), 0.25);
        assert_eq!(powi(-2.0, 3), -8.0);
        // Correctly rounded where repeated squaring is not: the float
        // nearest to 1.0134^4 is 1.054687016657794, (1.0134^2)^2 rounds
        // to the next one.
        let x = 1.0134;
        assert_eq!(powi(x, 4), 1.054687016657794);
        assert_eq!((x * x) * (x * x), 1.0546870166577942);
    }
    #[test]
    fn floats_are_written_as_julia_writes_them() {
        for (x, s) in [
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (0.1 + 0.2, "0.30000000000000004"),
            (12.3, "12.3"),
            (-12.5, "-12.5"),
            (100.0, "100.0"),
            (1.0e5, "100000.0"),
            (123456.0, "123456.0"),
            (1.0e6, "1.0e6"),
            (1234567.0, "1.234567e6"),
            (0.0001, "0.0001"),
            (2.5e-4, "0.00025"),
            (1.0e-5, "1.0e-5"),
            (1.5e300, "1.5e300"),
            (5.0e-324, "5.0e-324"),
            (f64::MAX, "1.7976931348623157e308"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Inf"),
            (f64::NEG_INFINITY, "-Inf"),
        ] {
            assert_eq!(float_string(x), s);
        }
    }

    #[test]
    fn round_with_digits() {
        // 1.005 * 100 = 100.49999999999999; 2.675 * 100 = 267.5 exactly, a
        // tie, rounded to even.
        assert_eq!(round_digits(1.005, 2), 1.0);
        assert_eq!(round_digits(2.675, 2), 2.68);
        // Ties to even.
        assert_eq!(round_digits(0.125, 2), 0.12);
        assert_eq!(round_digits(12.345678, 1), 12.3);
        let r = round_digits(-0.004, 2);
        assert!(r == 0.0 && r.is_sign_negative());
        assert!(round_digits(f64::NAN, 2).is_nan());
        // x * 100 overflows: x itself.
        assert_eq!(round_digits(1.0e308, 2), 1.0e308);
    }
}
