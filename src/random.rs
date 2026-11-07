//! Randomness. The model draws random numbers only through [`Randomness`].
//!
//! - [`Xoshiro`] for stochastic runs: xoshiro256**, with the same
//!   distributions as the Julia code but not the same stream of numbers.
//! - [`Deterministic`] for the comparison with the authors' reference values:
//!   the replacements of BeforeIT
//!   `test/deterministic/make_model_deterministic.jl`.

use crate::julia::{round, round_int, sum};

/// Source of all random numbers of the model. The first three methods are
/// required; the others have default implementations built on them.
pub trait Randomness {
    /// Standard normal number, Julia `randn()`. The model also uses it for
    /// Julia's `rand(Normal(0, σ))` as `σ · randn()`.
    fn randn(&mut self) -> f64;

    /// Uniform number in [0, 1), Julia `rand()`.
    fn rand(&mut self) -> f64;

    /// Uniform integer in `0..n`, for `n` greater than 0.
    fn rand_below(&mut self, n: usize) -> usize;

    /// Shuffles a list, Julia `fshuffle!` (BeforeIT `src/utils/opt.jl`):
    /// Fisher–Yates from the front.
    fn shuffle(&mut self, v: &mut [usize]) {
        for i in 1..v.len() {
            let j = self.rand_below(i + 1);
            v.swap(i, j);
        }
    }

    /// Draws an index with probability proportional to its weight, Julia
    /// `rand(::FixedSizeWeightVector)`. At least one weight must be
    /// positive.
    fn sample(&mut self, w: &mut WeightVector) -> usize {
        loop {
            let u = self.rand();
            if let Some(i) = w.try_sample(u) {
                return i;
            }
        }
    }

    /// Initial employees of the `n` firms of a sector, `N` in total, drawn
    /// from a power law, Julia `randpl(n, alpha, N)` (BeforeIT
    /// `src/utils/randpl.jl`, online appendix E.1).
    fn randpl(&mut self, n: usize, alpha: f64, N: i64) -> Result<Vec<i64>, String> {
        check_randpl(n, alpha, N)?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let exponent = -1.0 / (alpha - 1.0);
        let x: Vec<f64> = (0..n)
            .map(|_| libm::pow(1.0 - self.rand(), exponent))
            .collect();
        let total = sum(&x);
        let mut x: Vec<i64> = x
            .iter()
            .map(|&v| round_int(v / total * N as f64).map_err(|e| format!("randpl: {e}")))
            .collect::<Result<_, _>>()?;
        for v in &mut x {
            if *v < 1 {
                *v = 1;
            }
        }
        let mut dx = x.iter().sum::<i64>() - N;
        while dx != 0 {
            if dx < 0 {
                let mut id: Vec<usize> = (0..n).collect();
                self.shuffle(&mut id);
                for &i in &id[..dx.unsigned_abs() as usize] {
                    x[i] += 1;
                }
            } else {
                let mut id: Vec<usize> = (0..n).filter(|&i| x[i] > 1).collect();
                self.shuffle(&mut id);
                let k = (dx as usize).min(id.len());
                for &i in &id[..k] {
                    x[i] -= 1;
                }
            }
            dx = x.iter().sum::<i64>() - N;
        }
        Ok(x)
    }
}

/// Inputs on which Julia's `randpl` throws or never terminates.
fn check_randpl(n: usize, alpha: f64, N: i64) -> Result<(), String> {
    if alpha.is_nan() || alpha < 1.0 {
        return Err(format!("randpl: alpha is {alpha}, it must be at least 1"));
    }
    if (n == 0 && N != 0) || N < n as i64 {
        return Err(format!(
            "randpl: {N} employees are not enough for {n} firms (each needs at least 1)"
        ));
    }
    Ok(())
}

/// Weights of the sellers of a product, Julia
/// `WeightVectors.FixedSizeWeightVector`. Index `i` is drawn with probability
/// `w[i] / Σ w`. The model only ever sets weights to 0, when a seller is
/// sold out.
///
/// Draws use the cumulative sums of the weights. A draw that lands on an
/// entry set to 0 since the sums were built is repeated, which gives exactly
/// the probabilities among the remaining entries. The sums are rebuilt once
/// more than half of the total weight has been removed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WeightVector {
    w: Vec<f64>,
    cumulative: Vec<f64>,
    nonzero: usize,
    removed: f64,
}

impl WeightVector {
    pub fn new(w: Vec<f64>) -> Self {
        let nonzero = w.iter().filter(|&&x| x != 0.0).count();
        let mut v = WeightVector {
            w,
            cumulative: Vec::new(),
            nonzero,
            removed: 0.0,
        };
        v.rebuild();
        v
    }

    fn rebuild(&mut self) {
        self.cumulative.clear();
        let mut sum = 0.0;
        for &x in &self.w {
            sum += x;
            self.cumulative.push(sum);
        }
        self.removed = 0.0;
    }

    pub fn len(&self) -> usize {
        self.w.len()
    }

    pub fn is_empty(&self) -> bool {
        self.w.is_empty()
    }

    pub fn get(&self, i: usize) -> f64 {
        self.w[i]
    }

    /// Julia `iszero(w)`: all weights are 0 (also true for an empty list).
    pub fn is_zero(&self) -> bool {
        self.nonzero == 0
    }

    /// Julia `w[i] = 0.0`.
    pub fn set_zero(&mut self, i: usize) {
        if self.w[i] != 0.0 {
            self.nonzero -= 1;
            self.removed += self.w[i];
            self.w[i] = 0.0;
        }
    }

    /// First entry whose weight is not 0.
    pub fn first_nonzero(&self) -> Option<usize> {
        self.w.iter().position(|&x| x != 0.0)
    }

    /// One draw with the uniform number `u` in [0, 1). `None` means that the
    /// draw landed on an entry with weight 0 and has to be repeated.
    pub fn try_sample(&mut self, u: f64) -> Option<usize> {
        let total = *self.cumulative.last()?;
        if !(total.is_finite() && total > 0.0) {
            // Only for invalid weights (infinite or NaN), which Julia's
            // `FixedSizeWeightVector` does not accept.
            return self.first_nonzero();
        }
        if self.removed > 0.5 * total {
            self.rebuild();
            return self.try_sample(u);
        }
        let x = u * total;
        let i = self
            .cumulative
            .partition_point(|&c| c <= x)
            .min(self.w.len() - 1);
        (self.w[i] > 0.0).then_some(i)
    }
}

/// The xoshiro256** generator (Blackman and Vigna), seeded from a single
/// number with SplitMix64. Its state can be saved and restored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut state = seed;
        let mut s = [0; 4];
        for x in &mut s {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *x = z ^ (z >> 31);
        }
        Xoshiro { s }
    }

    /// Restores a saved state. The all-zero state is invalid.
    pub fn from_state(s: [u64; 4]) -> Result<Self, String> {
        if s == [0; 4] {
            return Err("the state of the random number generator is all zeros".into());
        }
        Ok(Xoshiro { s })
    }

    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

impl Randomness for Xoshiro {
    /// Box–Muller transform.
    fn randn(&mut self) -> f64 {
        let u1 = 1.0 - self.rand();
        let u2 = self.rand();
        (-2.0 * libm::log(u1)).sqrt() * libm::cos(2.0 * std::f64::consts::PI * u2)
    }

    fn rand(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Unbiased, with Lemire's multiply-and-reject method.
    fn rand_below(&mut self, n: usize) -> usize {
        let n = n as u64;
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
            }
        }
        (m >> 64) as usize
    }
}

/// No randomness, as in BeforeIT's deterministic test: `randn` returns 0, a
/// draw always picks the first seller with a positive weight, nothing is
/// shuffled, and every firm of a sector gets the same number of employees.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Deterministic;

impl Randomness for Deterministic {
    fn randn(&mut self) -> f64 {
        0.0
    }

    /// The model calls `rand` only from `randpl` and `sample`, which are
    /// replaced here.
    fn rand(&mut self) -> f64 {
        0.0
    }

    /// Julia's test replaces `rand(1:n)` by 1, the first entry.
    fn rand_below(&mut self, _n: usize) -> usize {
        0
    }

    fn shuffle(&mut self, _v: &mut [usize]) {}

    fn sample(&mut self, w: &mut WeightVector) -> usize {
        w.first_nonzero()
            .expect("sample: all weights are 0, the model never draws then")
    }

    /// `[Int(round(N / n + 1e-7)) for _ in 1:n]`.
    fn randpl(&mut self, n: usize, alpha: f64, N: i64) -> Result<Vec<i64>, String> {
        check_randpl(n, alpha, N)?;
        let each = round(N as f64 / n as f64 + 1.0e-7) as i64;
        Ok(vec![each; n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First outputs of the reference implementation for the state
    /// [1, 2, 3, 4].
    #[test]
    fn xoshiro_matches_reference_output() {
        let mut rng = Xoshiro::from_state([1, 2, 3, 4]).unwrap();
        let expected: [u64; 10] = [
            11520,
            0,
            1509978240,
            1215971899390074240,
            1216172134540287360,
            607988272756665600,
            16172922978634559625,
            8476171486693032832,
            10595114339597558777,
            2904607092377533576,
        ];
        for e in expected {
            assert_eq!(rng.next_u64(), e);
        }
    }

    /// SplitMix64 outputs for the seed 0.
    #[test]
    fn splitmix_seeding_matches_reference_output() {
        let rng = Xoshiro::seed_from_u64(0);
        assert_eq!(
            rng.state(),
            [
                16294208416658607535,
                7960286522194355700,
                487617019471545679,
                17909611376780542444
            ]
        );
    }

    #[test]
    fn rand_below_is_uniform() {
        let mut rng = Xoshiro::seed_from_u64(7);
        let mut count = [0usize; 6];
        let n = 60_000;
        for _ in 0..n {
            count[rng.rand_below(6)] += 1;
        }
        for c in count {
            // Expected 10 000, standard deviation about 91.
            assert!((c as f64 - 10_000.0).abs() < 400.0, "{count:?}");
        }
    }

    #[test]
    fn randn_has_mean_0_and_variance_1() {
        let mut rng = Xoshiro::seed_from_u64(11);
        let n = 200_000;
        let x: Vec<f64> = (0..n).map(|_| rng.randn()).collect();
        let mean = x.iter().sum::<f64>() / n as f64;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.01, "{var}");
    }

    #[test]
    fn sample_follows_the_weights() {
        let mut rng = Xoshiro::seed_from_u64(3);
        let mut count = [0usize; 3];
        let n = 90_000;
        for _ in 0..n {
            let mut w = WeightVector::new(vec![1.0, 2.0, 6.0]);
            count[rng.sample(&mut w)] += 1;
        }
        let expected = [10_000.0, 20_000.0, 60_000.0];
        for (c, e) in count.iter().zip(expected) {
            assert!((*c as f64 - e).abs() < 600.0, "{count:?}");
        }
    }

    #[test]
    fn zeroed_weights_are_never_drawn() {
        let mut rng = Xoshiro::seed_from_u64(5);
        let mut w = WeightVector::new(vec![5.0, 1.0, 3.0, 1.0]);
        w.set_zero(0);
        w.set_zero(2);
        let mut count = [0usize; 4];
        for _ in 0..20_000 {
            count[rng.sample(&mut w)] += 1;
        }
        assert_eq!(count[0] + count[2], 0);
        assert!((count[1] as f64 - 10_000.0).abs() < 400.0, "{count:?}");
        w.set_zero(1);
        w.set_zero(3);
        assert!(w.is_zero());
        assert!(WeightVector::new(Vec::new()).is_zero());
    }

    #[test]
    fn deterministic_takes_the_first_positive_weight() {
        let mut w = WeightVector::new(vec![0.5, 0.2]);
        assert_eq!(Deterministic.sample(&mut w), 0);
        w.set_zero(0);
        assert_eq!(Deterministic.sample(&mut w), 1);
        let mut v = vec![3, 1, 2];
        Deterministic.shuffle(&mut v);
        assert_eq!(v, [3, 1, 2]);
        assert_eq!(Deterministic.randpl(4, 2.0, 10).unwrap(), [3, 3, 3, 3]);
    }

    #[test]
    fn randpl_sums_to_n_and_every_firm_has_an_employee() {
        // As BeforeIT `test/utils/randpl.jl`: n = I_s(1), N = N_s(1).
        let mut rng = Xoshiro::seed_from_u64(1);
        for _ in 0..50 {
            let x = rng.randpl(48, 2.0, 123).unwrap();
            assert_eq!(x.iter().sum::<i64>(), 123);
            assert!(x.iter().all(|&v| v >= 1));
        }
        assert!(rng.randpl(5, 2.0, 4).is_err());
        assert!(rng.randpl(0, 2.0, 0).unwrap().is_empty());
    }
}
