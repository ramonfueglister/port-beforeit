//! Expectations, euro area, price indices and GDP, BeforeIT
//! `src/agent_actions/estimations.jl` and `aggregates.jl`, and `epsilon` and
//! `set_epsilon!` from `src/utils/epsilon.jl`.

use crate::julia::{sum, sum_by};
use crate::random::Randomness;
use crate::{AbstractModel, Extension};

/// `growth_inflation_expectations` (A.6, A.7, A.10): AR(1) rules estimated on
/// the whole history of log real GDP and of inflation. BeforeIT's smoothing
/// option `omega_pi` is not ported; its default 0 leaves `pi_e` as computed
/// here.
pub fn growth_inflation_expectations<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    rng: &mut R,
) -> (f64, f64, f64) {
    let n = m.prop.T_prime + m.agg.t - 1;
    let log_Y: Vec<f64> = m.agg.Y[..n].iter().map(|&y| libm::log(y)).collect();
    let lY_e = X::estimate_next_value(&log_Y, rng);
    let Y_e = libm::exp(lY_e);
    let gamma_e = Y_e / m.agg.Y[n - 1] - 1.0;
    let lpi = X::estimate_next_value(&m.agg.pi_[..n], rng);
    let pi_e = libm::exp(lpi) - 1.0;
    (Y_e, gamma_e, pi_e)
}

/// `epsilon(model)`: correlated shocks to euro area GDP, exports and imports
/// (A.71, A.76, A.81), drawn with [`crate::estimate::epsilon`] from `prop.C`.
///
/// # Panics
///
/// If the entries of `prop.C` do not sum to 0 and `prop.C` is not symmetric
/// positive definite; Julia throws there.
pub fn epsilon<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    rng: &mut R,
) -> (f64, f64, f64) {
    crate::estimate::epsilon(&m.prop.C, rng).unwrap_or_else(|e| panic!("{e}"))
}

/// `growth_inflation_EA` (A.70 to A.72): euro area GDP, growth and
/// inflation.
pub fn growth_inflation_EA<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    rng: &mut R,
) -> (f64, f64, f64) {
    let rotw = &m.rotw;
    let epsilon_Y_EA = m.agg.epsilon_Y_EA;
    let epsilon_pi_EA = rng.randn() * rotw.sigma_pi_EA;
    let Y_EA = libm::exp(rotw.alpha_Y_EA * libm::log(rotw.Y_EA) + rotw.beta_Y_EA + epsilon_Y_EA);
    let gamma_EA = Y_EA / rotw.Y_EA - 1.0;
    let pi_EA =
        libm::exp(rotw.alpha_pi_EA * libm::log(1.0 + rotw.pi_EA) + rotw.beta_pi_EA + epsilon_pi_EA)
            - 1.0;
    (Y_EA, gamma_EA, pi_EA)
}

/// `inflation_priceindex` (A.11, A.12): inflation and the price index. As in
/// the authors' Julia and MATLAB code, the price index weights prices with
/// production `Y_i` and leaves out imports; A.12 in the paper weights with
/// sales including imports.
pub fn inflation_priceindex<X: Extension>(m: &mut AbstractModel<X>) -> (f64, f64) {
    let f = &m.firms;
    let price_index = sum_by(f.P_i.len(), |i| f.P_i[i] * f.Y_i[i]) / sum(&f.Y_i);
    let inflation = libm::log(price_index / m.agg.P_bar);
    (inflation, price_index)
}

/// `sector_specific_priceindex` (A.32): per product, the average price of
/// domestic sales and imports.
pub fn sector_specific_priceindex<X: Extension>(m: &mut AbstractModel<X>) -> Vec<f64> {
    let (f, rotw) = (&m.firms, &m.rotw);
    f.by_sector(m.prop.G)
        .iter()
        .enumerate()
        .map(|(g, firms)| {
            let internal = sum_by(firms.len(), |k| f.P_i[firms[k]] * f.Q_i[firms[k]]);
            let external = rotw.P_m[g] * rotw.Q_m[g];
            let tot_quantity = sum_by(firms.len(), |k| f.Q_i[firms[k]]) + rotw.Q_m[g];
            (internal + external) / tot_quantity
        })
        .collect()
}

/// `capital_formation_priceindex` (A.37).
pub fn capital_formation_priceindex<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let (b_CF_g, P_bar_g) = (&m.prop.b_CF_g, &m.agg.P_bar_g);
    sum_by(m.prop.G, |g| b_CF_g[g] * P_bar_g[g])
}

/// `households_priceindex` (A.31).
pub fn households_priceindex<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let (b_HH_g, P_bar_g) = (&m.prop.b_HH_g, &m.agg.P_bar_g);
    sum_by(m.prop.G, |g| b_HH_g[g] * P_bar_g[g])
}

/// `gross_domestic_product`: the production of all firms as real GDP.
pub fn gross_domestic_product<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    sum(&m.firms.Y_i)
}

/// `set_time!`: the next quarter.
pub fn set_time<X: Extension>(m: &mut AbstractModel<X>) {
    m.agg.t += 1;
}

impl<X: Extension> AbstractModel<X> {
    /// `growth_inflation_expectations(model)`; see
    /// [`crate::base::growth_inflation_expectations`].
    pub fn growth_inflation_expectations<R: Randomness + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> (f64, f64, f64) {
        X::growth_inflation_expectations(self, rng)
    }

    /// `set_growth_inflation_expectations!`: `agg.Y_e`, `agg.gamma_e` and
    /// `agg.pi_e`.
    pub fn set_growth_inflation_expectations<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let (Y_e, gamma_e, pi_e) = self.growth_inflation_expectations(rng);
        self.agg.Y_e = Y_e;
        self.agg.gamma_e = gamma_e;
        self.agg.pi_e = pi_e;
    }

    /// `epsilon(model)`; see [`crate::base::epsilon`].
    pub fn epsilon<R: Randomness + ?Sized>(&mut self, rng: &mut R) -> (f64, f64, f64) {
        X::epsilon(self, rng)
    }

    /// `set_epsilon!`: `agg.epsilon_Y_EA`, `agg.epsilon_E` and
    /// `agg.epsilon_I`.
    ///
    /// # Panics
    ///
    /// As [`crate::base::epsilon`].
    pub fn set_epsilon<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let (e_Y_EA, e_E, e_I) = self.epsilon(rng);
        self.agg.epsilon_Y_EA = e_Y_EA;
        self.agg.epsilon_E = e_E;
        self.agg.epsilon_I = e_I;
    }

    /// `growth_inflation_EA(model)`; see [`crate::base::growth_inflation_EA`].
    pub fn growth_inflation_EA<R: Randomness + ?Sized>(&mut self, rng: &mut R) -> (f64, f64, f64) {
        X::growth_inflation_EA(self, rng)
    }

    /// `set_growth_inflation_EA!`: `rotw.Y_EA`, `rotw.gamma_EA` and
    /// `rotw.pi_EA`.
    pub fn set_growth_inflation_EA<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let (Y_EA, gamma_EA, pi_EA) = self.growth_inflation_EA(rng);
        self.rotw.Y_EA = Y_EA;
        self.rotw.gamma_EA = gamma_EA;
        self.rotw.pi_EA = pi_EA;
    }

    /// `inflation_priceindex(model)`; see
    /// [`crate::base::inflation_priceindex`].
    pub fn inflation_priceindex(&mut self) -> (f64, f64) {
        X::inflation_priceindex(self)
    }

    /// `set_inflation_priceindex!`: appends a 0 to `agg.pi_`, stores the
    /// inflation of the quarter in entry `T_prime + t` (counted from 1) and
    /// the price index in `agg.P_bar`.
    pub fn set_inflation_priceindex(&mut self) {
        self.agg.pi_.push(0.0);
        let (inflation, price_index) = self.inflation_priceindex();
        let k = self.prop.T_prime + self.agg.t - 1;
        self.agg.pi_[k] = inflation;
        self.agg.P_bar = price_index;
    }

    /// `sector_specific_priceindex(model)`; see
    /// [`crate::base::sector_specific_priceindex`].
    pub fn sector_specific_priceindex(&mut self) -> Vec<f64> {
        X::sector_specific_priceindex(self)
    }

    /// `set_sector_specific_priceindex!`: `agg.P_bar_g`.
    pub fn set_sector_specific_priceindex(&mut self) {
        let P_bar_g = self.sector_specific_priceindex();
        self.agg.P_bar_g.copy_from_slice(&P_bar_g);
    }

    /// `capital_formation_priceindex(model)`; see
    /// [`crate::base::capital_formation_priceindex`].
    pub fn capital_formation_priceindex(&mut self) -> f64 {
        X::capital_formation_priceindex(self)
    }

    /// `set_capital_formation_priceindex!`: `agg.P_bar_CF`.
    pub fn set_capital_formation_priceindex(&mut self) {
        self.agg.P_bar_CF = self.capital_formation_priceindex();
    }

    /// `households_priceindex(model)`; see
    /// [`crate::base::households_priceindex`].
    pub fn households_priceindex(&mut self) -> f64 {
        X::households_priceindex(self)
    }

    /// `set_households_priceindex!`: `agg.P_bar_HH`.
    pub fn set_households_priceindex(&mut self) {
        self.agg.P_bar_HH = self.households_priceindex();
    }

    /// `gross_domestic_product(model)`; see
    /// [`crate::base::gross_domestic_product`].
    pub fn gross_domestic_product(&mut self) -> f64 {
        X::gross_domestic_product(self)
    }

    /// `set_gross_domestic_product!`: appends a 0 to `agg.Y` and stores the
    /// real GDP of the quarter in entry `T_prime + t` (counted from 1).
    pub fn set_gross_domestic_product(&mut self) {
        self.agg.Y.push(0.0);
        let Y = self.gross_domestic_product();
        let k = self.prop.T_prime + self.agg.t - 1;
        self.agg.Y[k] = Y;
    }

    /// `set_time!(model)`; see [`crate::base::set_time`].
    pub fn set_time(&mut self) {
        X::set_time(self)
    }
}
