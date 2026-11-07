//! The growth-rate model, BeforeIT
//! `src/model_extensions/init_growth_rate_model.jl`: the AR(1) processes of
//! the model of the paper act on logarithms, `log(Y_t) = α·log(Y_{t−1}) + β + ε`;
//! here they act on growth rates, `g_t = α·g_{t−1} + β + ε` and
//! `Y_t = Y_{t−1}·(1 + g_t)`. This applies to the expectations of GDP and
//! inflation, government consumption, exports and imports. Euro area GDP
//! keeps the process of the paper, so that the input of the Taylor rule
//! stays consistent with the calibration.

use crate::estimate::estimate_for_calibration_script;
use crate::julia::sum_by;
use crate::random::Randomness;
use crate::{AbstractModel, Extension, InitialConditions, Parameters};

/// The fields that the growth-rate model adds, BeforeIT's
/// `RestOfTheWorldGR` and `GovernmentGR`: the AR(1) coefficients and the
/// standard deviation of the shock of the growth rates of exports (`E`),
/// imports (`I`) and government consumption (`G`), and the last growth rate
/// of each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GrowthRate {
    pub alpha_E_gr: f64,
    pub beta_E_gr: f64,
    pub sigma_E_gr: f64,
    pub alpha_I_gr: f64,
    pub beta_I_gr: f64,
    pub sigma_I_gr: f64,
    pub g_prev_C_E: f64,
    pub g_prev_Y_I: f64,
    pub alpha_G_gr: f64,
    pub beta_G_gr: f64,
    pub sigma_G_gr: f64,
    pub g_prev_C_G: f64,
}

/// BeforeIT's `ModelGR`.
pub type ModelGR = AbstractModel<GrowthRate>;

/// `estimate_gr_ar1(series)`: AR(1) on the growth rates of a series, 0 for
/// series shorter than 3.
fn estimate_gr_ar1(series: &[f64]) -> (f64, f64, f64) {
    if series.len() < 3 {
        return (0.0, 0.0, 0.0);
    }
    let growth_rates: Vec<f64> = series.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
    if growth_rates.len() < 2 {
        // Not reached after the check above, as in the Julia code; the
        // standard deviation of one number is NaN there.
        return (0.0, growth_rates[0], f64::NAN);
    }
    let (alpha, beta, sigma, _) = estimate_for_calibration_script(&growth_rates);
    (alpha, beta, sigma)
}

/// `last_gr(s)`: the last growth rate of a series, 0 for series shorter
/// than 2.
fn last_gr(s: &[f64]) -> f64 {
    let n = s.len();
    if n < 2 {
        0.0
    } else {
        (s[n - 1] - s[n - 2]) / s[n - 2]
    }
}

impl GrowthRate {
    /// The added fields as in `ModelGR(parameters, initial_conditions)`:
    /// the AR(1) estimates on the growth rates of the first `T_prime`
    /// quarters of `C_G`, `C_E` and `Y_I`, and the last growth rate of each.
    ///
    /// Returns an error if one of the three histories is shorter than
    /// `T_prime` (Julia throws a `BoundsError`).
    pub fn new(p: &Parameters, ic: &InitialConditions) -> Result<GrowthRate, String> {
        let T_prime = p.T_prime;
        let first = |name: &str, s: &[f64]| -> Result<Vec<f64>, String> {
            s.get(..T_prime).map(<[f64]>::to_vec).ok_or_else(|| {
                format!(
                    "{name} has {} quarters, T_prime = {T_prime} are needed",
                    s.len()
                )
            })
        };
        let C_G_series = first("C_G", &ic.C_G)?;
        let C_E_series = first("C_E", &ic.C_E)?;
        let Y_I_series = first("Y_I", &ic.Y_I)?;
        let (alpha_G_gr, beta_G_gr, sigma_G_gr) = estimate_gr_ar1(&C_G_series);
        let (alpha_E_gr, beta_E_gr, sigma_E_gr) = estimate_gr_ar1(&C_E_series);
        let (alpha_I_gr, beta_I_gr, sigma_I_gr) = estimate_gr_ar1(&Y_I_series);
        Ok(GrowthRate {
            alpha_E_gr,
            beta_E_gr,
            sigma_E_gr,
            alpha_I_gr,
            beta_I_gr,
            sigma_I_gr,
            g_prev_C_E: last_gr(&C_E_series),
            g_prev_Y_I: last_gr(&Y_I_series),
            alpha_G_gr,
            beta_G_gr,
            sigma_G_gr,
            g_prev_C_G: last_gr(&C_G_series),
        })
    }
}

impl ModelGR {
    /// `ModelGR(parameters, initial_conditions)`: the model of
    /// [`crate::Model::new`] with the fields of [`GrowthRate::new`].
    pub fn new<R: Randomness + ?Sized>(
        p: &Parameters,
        ic: &InitialConditions,
        rng: &mut R,
    ) -> Result<ModelGR, String> {
        let ext = GrowthRate::new(p, ic)?;
        let agents = crate::agents_of_the_model(p, ic, rng)?;
        AbstractModel::from_agents(agents, ext, rng)
    }
}

impl Extension for GrowthRate {
    /// AR(1) forecasts of the growth rate of GDP and of `1 + π`, on the
    /// first `T_prime + t − 1` quarters. Unlike the model of the paper, the
    /// option `omega_pi` has no effect.
    fn growth_inflation_expectations<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, f64, f64) {
        let n = m.prop.T_prime + m.agg.t - 1;
        let Y_slice = &m.agg.Y[..n];
        let gamma_series: Vec<f64> = (1..n).map(|k| Y_slice[k] / Y_slice[k - 1] - 1.0).collect();
        let gamma_e = Self::estimate_next_value(&gamma_series, rng);
        let Y_e = Y_slice[n - 1] * (1.0 + gamma_e);
        let one_plus_pi: Vec<f64> = m.agg.pi_[..n].iter().map(|p| 1.0 + p).collect();
        let pi_e = Self::estimate_next_value(&one_plus_pi, rng) - 1.0;
        (Y_e, gamma_e, pi_e)
    }

    /// Government consumption grows by an AR(1) growth rate.
    fn gov_expenditure<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, Vec<f64>) {
        let e = &mut m.ext;
        let g_new = e.alpha_G_gr * e.g_prev_C_G + e.beta_G_gr + rng.randn() * e.sigma_G_gr;
        let C_G = m.gov.C_G * (1.0 + g_new);
        e.g_prev_C_G = g_new;
        let (c_G_g, P_bar_g, pi_e) = (&m.prop.c_G_g, &m.agg.P_bar_g, m.agg.pi_e);
        let J = m.gov.C_d_j.len();
        let price = sum_by(c_G_g.len(), |g| c_G_g[g] * P_bar_g[g]);
        let C_d_j = vec![C_G / J as f64 * price * (1.0 + pi_e); J];
        (C_G, C_d_j)
    }

    /// Exports and imports grow by AR(1) growth rates.
    fn rotw_import_export<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (c_E_g, c_I_g, P_bar_g, pi_e) =
            (&m.prop.c_E_g, &m.prop.c_I_g, &m.agg.P_bar_g, m.agg.pi_e);
        let (rotw, e) = (&m.rotw, &mut m.ext);
        let L = rotw.C_d_l.len();

        let g_E = e.alpha_E_gr * e.g_prev_C_E + e.beta_E_gr + rng.randn() * e.sigma_E_gr;
        let C_E = rotw.C_E * (1.0 + g_E);
        e.g_prev_C_E = g_E;
        let price = sum_by(c_E_g.len(), |g| c_E_g[g] * P_bar_g[g]);
        let C_d_l = vec![C_E / L as f64 * price * (1.0 + pi_e); L];

        let g_I = e.alpha_I_gr * e.g_prev_Y_I + e.beta_I_gr + rng.randn() * e.sigma_I_gr;
        let Y_I = rotw.Y_I * (1.0 + g_I);
        e.g_prev_Y_I = g_I;

        let Y_m = c_I_g.iter().map(|c| c * Y_I).collect();
        let P_m = P_bar_g.iter().map(|p| p * (1.0 + pi_e)).collect();
        (C_E, Y_I, C_d_l, Y_m, P_m)
    }
}
