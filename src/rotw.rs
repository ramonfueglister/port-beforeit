//! Rest of the world, BeforeIT `src/agent_actions/rotw.jl` (appendix A.6).

use crate::julia::sum_by;
use crate::random::Randomness;
use crate::{AbstractModel, Extension};

/// `rotw_import_export` (A.76 to A.78, A.81, A.82): demand for exports and
/// supply of imports by AR(1) rules with the shocks of `set_epsilon!`;
/// import prices from last quarter's price indices and expected inflation.
/// Returns `C_E`, `Y_I`, `C_d_l`, `Y_m` and `P_m`. The standard rule draws
/// no random numbers.
#[allow(clippy::type_complexity)]
pub fn rotw_import_export<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    _rng: &mut R,
) -> (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>) {
    let rotw = &m.rotw;
    let (c_E_g, c_I_g, P_bar_g, pi_e) = (&m.prop.c_E_g, &m.prop.c_I_g, &m.agg.P_bar_g, m.agg.pi_e);
    let (epsilon_E, epsilon_I) = (m.agg.epsilon_E, m.agg.epsilon_I);
    let L = rotw.C_d_l.len();
    let C_E = libm::exp(rotw.alpha_E * libm::log(rotw.C_E) + rotw.beta_E + epsilon_E);
    let price = sum_by(c_E_g.len(), |g| c_E_g[g] * P_bar_g[g]);
    let C_d_l = vec![C_E / L as f64 * price * (1.0 + pi_e); L];
    let Y_I = libm::exp(rotw.alpha_I * libm::log(rotw.Y_I) + rotw.beta_I + epsilon_I);
    let Y_m = c_I_g.iter().map(|c| c * Y_I).collect();
    let P_m = P_bar_g.iter().map(|p| p * (1.0 + pi_e)).collect();
    (C_E, Y_I, C_d_l, Y_m, P_m)
}

/// `rotw_deposits` (A.75): the net position of the rest of the world grows
/// with imports and falls with exports.
pub fn rotw_deposits<X: Extension>(m: &mut AbstractModel<X>) -> f64 {
    let rotw = &m.rotw;
    let imports = sum_by(rotw.P_m.len(), |g| rotw.P_m[g] * rotw.Q_m[g]);
    let DD_RoW = imports - (1.0 + m.prop.tau_EXPORT) * rotw.C_l;
    rotw.D_RoW + DD_RoW
}

impl<X: Extension> AbstractModel<X> {
    /// `rotw_import_export(model)`; see [`crate::base::rotw_import_export`].
    #[allow(clippy::type_complexity)]
    pub fn rotw_import_export<R: Randomness + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        X::rotw_import_export(self, rng)
    }

    /// `set_rotw_import_export!`: `rotw.C_E`, `Y_I`, `C_d_l`, `Y_m` and
    /// `P_m`.
    pub fn set_rotw_import_export<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let (C_E, Y_I, C_d_l, Y_m, P_m) = self.rotw_import_export(rng);
        let rotw = &mut self.rotw;
        rotw.C_E = C_E;
        rotw.Y_I = Y_I;
        rotw.C_d_l.copy_from_slice(&C_d_l);
        rotw.Y_m.copy_from_slice(&Y_m);
        rotw.P_m.copy_from_slice(&P_m);
    }

    /// `rotw_deposits(model)`; see [`crate::base::rotw_deposits`].
    pub fn rotw_deposits(&mut self) -> f64 {
        X::rotw_deposits(self)
    }

    /// `set_rotw_deposits!`: `rotw.D_RoW`.
    pub fn set_rotw_deposits(&mut self) {
        self.rotw.D_RoW = self.rotw_deposits();
    }
}
