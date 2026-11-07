//! The AR(1) estimate on the Austrian GDP series, as in BeforeIT
//! `test/utils/nfvar3_and_estimate.jl`, "estimate". The shock there comes
//! from Julia's random number generator and is not compared.

#![allow(non_snake_case)]

mod common;

use beforeit::Deterministic;
use beforeit::estimate::estimate;
use common::*;

#[test]
fn estimate_on_austrian_gdp() {
    let (_, ic) = austria();
    let log_Y: Vec<f64> = ic.Y.iter().map(|y| y.ln()).collect();
    let (alpha, beta, _) = estimate(&log_Y, &mut Deterministic);
    check("alpha", alpha, 0.971001709000414, SQRT_EPS, 0.0);
    check("beta", beta, 0.344659199612863, SQRT_EPS, 0.0);
}
