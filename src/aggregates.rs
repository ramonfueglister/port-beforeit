//! Expectations, euro area, price indices and GDP, BeforeIT
//! `src/agent_actions/estimations.jl` and `aggregates.jl`, and
//! `set_epsilon!` from `src/utils/epsilon.jl`.

use crate::Model;
use crate::estimate::{epsilon, estimate_next_value};
use crate::julia::{sum, sum_by};
use crate::random::Randomness;

impl Model {
    /// `growth_inflation_expectations` (A.6, A.7, A.10): AR(1) rules
    /// estimated on the whole history of log real GDP and of inflation.
    /// BeforeIT's smoothing option `omega_pi` is not ported; its default 0
    /// leaves `pi_e` as computed here.
    pub fn set_growth_inflation_expectations<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let n = self.prop.T_prime + self.agg.t - 1;
        let log_Y: Vec<f64> = self.agg.Y[..n].iter().map(|&y| libm::log(y)).collect();
        let lY_e = estimate_next_value(&log_Y, rng);
        let Y_e = libm::exp(lY_e);
        let gamma_e = Y_e / self.agg.Y[n - 1] - 1.0;
        let lpi = estimate_next_value(&self.agg.pi_[..n], rng);
        let pi_e = libm::exp(lpi) - 1.0;
        self.agg.Y_e = Y_e;
        self.agg.gamma_e = gamma_e;
        self.agg.pi_e = pi_e;
    }

    /// `set_epsilon!`: correlated shocks to euro area GDP, exports and
    /// imports (A.71, A.76, A.81).
    ///
    /// # Panics
    ///
    /// If the entries of `prop.C` do not sum to 0 and `prop.C` is not
    /// symmetric positive definite (see [`epsilon`]); Julia throws there.
    pub fn set_epsilon<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let (e_Y_EA, e_E, e_I) = epsilon(&self.prop.C, rng).unwrap_or_else(|e| panic!("{e}"));
        self.agg.epsilon_Y_EA = e_Y_EA;
        self.agg.epsilon_E = e_E;
        self.agg.epsilon_I = e_I;
    }

    /// `growth_inflation_EA` (A.70 to A.72).
    pub fn set_growth_inflation_EA<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let rotw = &mut self.rotw;
        let epsilon_Y_EA = self.agg.epsilon_Y_EA;
        let epsilon_pi_EA = rng.randn() * rotw.sigma_pi_EA;
        let Y_EA =
            libm::exp(rotw.alpha_Y_EA * libm::log(rotw.Y_EA) + rotw.beta_Y_EA + epsilon_Y_EA);
        let gamma_EA = Y_EA / rotw.Y_EA - 1.0;
        let pi_EA = libm::exp(
            rotw.alpha_pi_EA * libm::log(1.0 + rotw.pi_EA) + rotw.beta_pi_EA + epsilon_pi_EA,
        ) - 1.0;
        rotw.Y_EA = Y_EA;
        rotw.gamma_EA = gamma_EA;
        rotw.pi_EA = pi_EA;
    }

    /// `inflation_priceindex` (A.11, A.12). As in the authors' Julia and
    /// MATLAB code, the price index weights prices with production `Y_i` and
    /// leaves out imports; A.12 in the paper weights with sales including
    /// imports.
    pub fn set_inflation_priceindex(&mut self) {
        let f = &self.firms;
        let price_index = sum_by(f.P_i.len(), |i| f.P_i[i] * f.Y_i[i]) / sum(&f.Y_i);
        let inflation = libm::log(price_index / self.agg.P_bar);
        self.agg.pi_.push(0.0);
        let k = self.prop.T_prime + self.agg.t - 1;
        self.agg.pi_[k] = inflation;
        self.agg.P_bar = price_index;
    }

    /// `sector_specific_priceindex` (A.32): per product, the average price of
    /// domestic sales and imports.
    pub fn set_sector_specific_priceindex(&mut self) {
        let f = &self.firms;
        for (g, firms) in f.by_sector(self.prop.G).iter().enumerate() {
            let internal = sum_by(firms.len(), |k| f.P_i[firms[k]] * f.Q_i[firms[k]]);
            let external = self.rotw.P_m[g] * self.rotw.Q_m[g];
            let tot_quantity = sum_by(firms.len(), |k| f.Q_i[firms[k]]) + self.rotw.Q_m[g];
            self.agg.P_bar_g[g] = (internal + external) / tot_quantity;
        }
    }

    /// `capital_formation_priceindex` (A.37).
    pub fn set_capital_formation_priceindex(&mut self) {
        let (b_CF_g, P_bar_g) = (&self.prop.b_CF_g, &self.agg.P_bar_g);
        self.agg.P_bar_CF = sum_by(self.prop.G, |g| b_CF_g[g] * P_bar_g[g]);
    }

    /// `households_priceindex` (A.31).
    pub fn set_households_priceindex(&mut self) {
        let (b_HH_g, P_bar_g) = (&self.prop.b_HH_g, &self.agg.P_bar_g);
        self.agg.P_bar_HH = sum_by(self.prop.G, |g| b_HH_g[g] * P_bar_g[g]);
    }

    /// `set_gross_domestic_product!`: the production of all firms as real
    /// GDP of the quarter, entry `T_prime + t` (counted from 1) of the
    /// series. As in the Julia code, a 0 is appended first.
    pub fn set_gross_domestic_product(&mut self) {
        self.agg.Y.push(0.0);
        let k = self.prop.T_prime + self.agg.t - 1;
        self.agg.Y[k] = sum(&self.firms.Y_i);
    }

    /// `set_time!`.
    pub fn set_time(&mut self) {
        self.agg.t += 1;
    }
}
