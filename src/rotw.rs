//! Rest of the world, BeforeIT `src/agent_actions/rotw.jl` (appendix A.6).

use crate::Model;
use crate::julia::sum_by;

impl Model {
    /// `rotw_import_export` (A.76 to A.78, A.81, A.82): demand for exports
    /// and supply of imports by AR(1) rules; import prices from last
    /// quarter's price indices and expected inflation.
    pub fn set_rotw_import_export(&mut self) {
        let (pi_e, epsilon_E, epsilon_I) = (self.agg.pi_e, self.agg.epsilon_E, self.agg.epsilon_I);
        let (c_E_g, P_bar_g) = (&self.prop.c_E_g, &self.agg.P_bar_g);
        let price = sum_by(c_E_g.len(), |g| c_E_g[g] * P_bar_g[g]);
        let rotw = &mut self.rotw;
        let C_E = libm::exp(rotw.alpha_E * libm::log(rotw.C_E) + rotw.beta_E + epsilon_E);
        let C_d_l = C_E / rotw.C_d_l.len() as f64 * price * (1.0 + pi_e);
        let Y_I = libm::exp(rotw.alpha_I * libm::log(rotw.Y_I) + rotw.beta_I + epsilon_I);
        rotw.C_E = C_E;
        rotw.Y_I = Y_I;
        rotw.C_d_l.fill(C_d_l);
        for g in 0..self.prop.G {
            rotw.Y_m[g] = self.prop.c_I_g[g] * Y_I;
            rotw.P_m[g] = self.agg.P_bar_g[g] * (1.0 + pi_e);
        }
    }

    /// `rotw_deposits` (A.75).
    pub fn set_rotw_deposits(&mut self) {
        let rotw = &mut self.rotw;
        let imports = sum_by(rotw.P_m.len(), |g| rotw.P_m[g] * rotw.Q_m[g]);
        let DD_RoW = imports - (1.0 + self.prop.tau_EXPORT) * rotw.C_l;
        rotw.D_RoW += DD_RoW;
    }
}
