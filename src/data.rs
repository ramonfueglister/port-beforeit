//! Time series of national accounts variables, BeforeIT `src/utils/data.jl`.
//! The per-bank series of BeforeIT's multi-bank extension
//! (`credit_new_per_bank` and the like) are not collected.

use crate::julia::{sum, sum_by};
use crate::{AbstractModel, Extension};

/// One list per variable and one entry per collection: first the initial
/// state, then one entry after each quarter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Data {
    /// Quarter of the collection, 1 for the initial state.
    pub collection_time: Vec<usize>,
    pub nominal_gdp: Vec<f64>,
    pub real_gdp: Vec<f64>,
    pub nominal_gva: Vec<f64>,
    pub real_gva: Vec<f64>,
    pub nominal_household_consumption: Vec<f64>,
    pub real_household_consumption: Vec<f64>,
    pub nominal_government_consumption: Vec<f64>,
    pub real_government_consumption: Vec<f64>,
    pub nominal_capitalformation: Vec<f64>,
    pub real_capitalformation: Vec<f64>,
    pub nominal_fixed_capitalformation: Vec<f64>,
    pub real_fixed_capitalformation: Vec<f64>,
    pub nominal_fixed_capitalformation_dwellings: Vec<f64>,
    pub real_fixed_capitalformation_dwellings: Vec<f64>,
    pub nominal_exports: Vec<f64>,
    pub real_exports: Vec<f64>,
    pub nominal_imports: Vec<f64>,
    pub real_imports: Vec<f64>,
    pub operating_surplus: Vec<f64>,
    pub compensation_employees: Vec<f64>,
    pub wages: Vec<f64>,
    pub taxes_production: Vec<f64>,
    pub gdp_deflator_growth_ea: Vec<f64>,
    pub real_gdp_ea: Vec<f64>,
    pub euribor: Vec<f64>,
    /// Gross value added per sector, nominal and real.
    pub nominal_sector_gva: Vec<Vec<f64>>,
    pub real_sector_gva: Vec<Vec<f64>>,
}

/// `collect_data!(m)`: appends the values of the current state; the first
/// call records the initial state. As in the Julia code, it calls
/// `allocate_new_data!` and then `update_data_init!` or
/// `update_data_step!` through the model type.
pub fn collect_data<X: Extension>(m: &mut AbstractModel<X>) {
    X::allocate_new_data(m);
    let t = m.data.collection_time.len();
    if t == 1 {
        X::update_data_init(m);
    } else {
        X::update_data_step(m);
    }
}

/// `allocate_new_data!`: appends a 0 to every series and a list of zeros to
/// the series per sector.
pub fn allocate_new_data<X: Extension>(m: &mut AbstractModel<X>) {
    let d = &mut m.data;
    d.collection_time.push(0);
    for series in [
        &mut d.nominal_gdp,
        &mut d.real_gdp,
        &mut d.nominal_gva,
        &mut d.real_gva,
        &mut d.nominal_household_consumption,
        &mut d.real_household_consumption,
        &mut d.nominal_government_consumption,
        &mut d.real_government_consumption,
        &mut d.nominal_capitalformation,
        &mut d.real_capitalformation,
        &mut d.nominal_fixed_capitalformation,
        &mut d.real_fixed_capitalformation,
        &mut d.nominal_fixed_capitalformation_dwellings,
        &mut d.real_fixed_capitalformation_dwellings,
        &mut d.nominal_exports,
        &mut d.real_exports,
        &mut d.nominal_imports,
        &mut d.real_imports,
        &mut d.operating_surplus,
        &mut d.compensation_employees,
        &mut d.wages,
        &mut d.taxes_production,
        &mut d.gdp_deflator_growth_ea,
        &mut d.real_gdp_ea,
        &mut d.euribor,
    ] {
        series.push(0.0);
    }
    d.nominal_sector_gva.push(vec![0.0; m.prop.G]);
    d.real_sector_gva.push(vec![0.0; m.prop.G]);
}

/// Sum of `f(i)` over the firms of each sector.
fn by_sector<X>(m: &AbstractModel<X>, f: impl Fn(usize) -> f64) -> Vec<f64> {
    let sectors = m.firms.by_sector(m.prop.G);
    sectors
        .iter()
        .map(|firms| sum_by(firms.len(), |k| f(firms[k])))
        .collect()
}

/// `update_data_init!`: the initial state, at prices of 1, in the first
/// entry.
pub fn update_data_init<X: Extension>(m: &mut AbstractModel<X>) {
    let (p, f) = (&m.prop, &m.firms);
    let I = f.len();
    let tot_Y_h = sum(&m.w_act.Y_h) + sum(&m.w_inact.Y_h) + sum(&f.Y_h) + m.bank.Y_h;
    let (C_G, C_E) = (m.gov.C_G, m.rotw.C_E);
    let gva = |i: usize| f.Y_i[i] * ((1.0 - f.tau_Y_i[i]) - 1.0 / f.beta_i[i]);
    let sector_gva = by_sector(m, gva);
    let nominal_gdp = sum_by(I, |i| f.Y_i[i] * (1.0 - 1.0 / f.beta_i[i]))
        + tot_Y_h * p.psi / (1.0 / p.tau_VAT + 1.0)
        + p.tau_G * C_G
        + tot_Y_h * p.psi_H / (1.0 / p.tau_CF + 1.0)
        + p.tau_EXPORT * C_E;
    let nominal_gva = sum_by(I, gva);
    let capitalformation =
        sum_by(I, |i| f.Y_i[i] * f.delta_i[i] / f.kappa_i[i]) + tot_Y_h * p.psi_H;
    let operating_surplus = sum_by(I, |i| {
        f.Y_i[i] * (1.0 - ((1.0 + p.tau_SIF) * f.w_bar_i[i] / f.alpha_bar_i[i] + 1.0 / f.beta_i[i]))
            - f.tau_K_i[i] * f.Y_i[i]
            - f.tau_Y_i[i] * f.Y_i[i]
    });
    let wages = sum_by(I, |i| f.w_bar_i[i] * f.N_i[i] as f64);
    let taxes_production = sum_by(I, |i| f.tau_K_i[i] * f.Y_i[i]);
    let (pi_EA, Y_EA, Y_I, r_bar) = (m.rotw.pi_EA, m.rotw.Y_EA, m.rotw.Y_I, m.cb.r_bar);
    let (psi, psi_H, tau_G, tau_SIF, tau_EXPORT) =
        (p.psi, p.psi_H, p.tau_G, p.tau_SIF, p.tau_EXPORT);

    let d = &mut m.data;
    d.nominal_gdp[0] = nominal_gdp;
    d.real_gdp[0] = d.nominal_gdp[0];
    d.nominal_gva[0] = nominal_gva;
    d.real_gva[0] = d.nominal_gva[0];
    d.nominal_household_consumption[0] = tot_Y_h * psi;
    d.real_household_consumption[0] = d.nominal_household_consumption[0];
    d.nominal_government_consumption[0] = (1.0 + tau_G) * C_G;
    d.real_government_consumption[0] = d.nominal_government_consumption[0];
    d.nominal_capitalformation[0] = capitalformation;
    d.real_capitalformation[0] = d.nominal_capitalformation[0];
    d.nominal_fixed_capitalformation[0] = d.nominal_capitalformation[0];
    d.real_fixed_capitalformation[0] = d.nominal_capitalformation[0];
    d.nominal_fixed_capitalformation_dwellings[0] = tot_Y_h * psi_H;
    d.real_fixed_capitalformation_dwellings[0] = d.nominal_fixed_capitalformation_dwellings[0];
    d.nominal_exports[0] = (1.0 + tau_EXPORT) * C_E;
    d.real_exports[0] = d.nominal_exports[0];
    d.nominal_imports[0] = Y_I;
    d.real_imports[0] = d.nominal_imports[0];
    d.operating_surplus[0] = operating_surplus;
    d.compensation_employees[0] = (1.0 + tau_SIF) * wages;
    d.wages[0] = wages;
    d.taxes_production[0] = taxes_production;
    d.nominal_sector_gva[0].copy_from_slice(&sector_gva);
    d.real_sector_gva[0].copy_from_slice(&sector_gva);
    d.euribor[0] = r_bar;
    d.gdp_deflator_growth_ea[0] = pi_EA;
    d.real_gdp_ea[0] = Y_EA;
    d.collection_time[0] = 1;
}

/// `update_data_step!`: the values after a quarter, in the last entry.
pub fn update_data_step<X: Extension>(m: &mut AbstractModel<X>) {
    let (p, f, agg) = (&m.prop, &m.firms, &m.agg);
    let (gov, rotw) = (&m.gov, &m.rotw);
    let I = f.len();
    let tot_C_h = sum(&m.w_act.C_h) + sum(&m.w_inact.C_h) + sum(&f.C_h) + m.bank.C_h;
    let tot_I_h = sum(&m.w_act.I_h) + sum(&m.w_inact.I_h) + sum(&f.I_h) + m.bank.I_h;
    let sales_net = |i: usize| (1.0 - f.tau_Y_i[i]) * f.P_i[i] * f.Y_i[i];
    let materials = |i: usize| 1.0 / f.beta_i[i] * f.P_bar_i[i] * f.Y_i[i];
    let real_gva_i = |i: usize| f.Y_i[i] * ((1.0 - f.tau_Y_i[i]) - 1.0 / f.beta_i[i]);
    let nominal_gdp = sum_by(I, |i| f.tau_Y_i[i] * f.Y_i[i] * f.P_i[i])
        + p.tau_VAT * tot_C_h
        + p.tau_CF * tot_I_h
        + p.tau_G * gov.C_j
        + p.tau_EXPORT * rotw.C_l
        + sum_by(I, sales_net)
        - sum_by(I, materials);
    let real_gdp = sum_by(I, real_gva_i)
        + sum_by(I, |i| f.tau_Y_i[i] * f.Y_i[i])
        + p.tau_VAT * tot_C_h / agg.P_bar_h
        + p.tau_CF * tot_I_h / agg.P_bar_CF_h
        + p.tau_G * gov.C_j / gov.P_j
        + p.tau_EXPORT * rotw.C_l / rotw.P_l;
    let nominal_gva = sum_by(I, sales_net) - sum_by(I, materials);
    let real_gva = sum_by(I, real_gva_i);
    let nominal_capitalformation = sum_by(I, |i| f.P_CF_i[i] * f.I_i[i])
        + (1.0 + p.tau_CF) * tot_I_h
        + sum_by(I, |i| f.DS_i[i] * f.P_i[i])
        + sum_by(I, |i| f.DM_i[i] * f.P_bar_i[i] - materials(i));
    let real_capitalformation = sum(&f.I_i)
        + (1.0 + p.tau_CF) * tot_I_h / agg.P_bar_CF_h
        + sum_by(I, |i| f.DM_i[i] - f.Y_i[i] / f.beta_i[i])
        + sum(&f.DS_i);
    let nominal_fixed_capitalformation =
        sum_by(I, |i| f.P_CF_i[i] * f.I_i[i]) + (1.0 + p.tau_CF) * tot_I_h;
    let real_fixed_capitalformation = sum(&f.I_i) + (1.0 + p.tau_CF) * tot_I_h / agg.P_bar_CF_h;
    let operating_surplus = sum_by(I, |i| {
        f.P_i[i] * f.Q_i[i] + f.P_i[i] * f.DS_i[i]
            - (1.0 + p.tau_SIF) * f.w_i[i] * f.N_i[i] as f64 * agg.P_bar_HH
            - materials(i)
            - f.tau_Y_i[i] * f.P_i[i] * f.Y_i[i]
            - f.tau_K_i[i] * f.P_i[i] * f.Y_i[i]
    });
    let compensation_employees =
        (1.0 + p.tau_SIF) * sum_by(I, |i| f.w_i[i] * f.N_i[i] as f64) * agg.P_bar_HH;
    let wages = sum_by(I, |i| f.w_i[i] * f.N_i[i] as f64) * agg.P_bar_HH;
    let taxes_production = sum_by(I, |i| f.tau_K_i[i] * f.Y_i[i] * f.P_i[i]);
    let nominal_sector_gva: Vec<f64> = by_sector(m, sales_net)
        .iter()
        .zip(by_sector(m, materials))
        .map(|(a, b)| a - b)
        .collect();
    let real_sector_gva = by_sector(m, real_gva_i);
    let nominal_imports = sum_by(rotw.P_m.len(), |g| rotw.P_m[g] * rotw.Q_m[g]);
    let real_imports = sum(&rotw.Q_m);
    let (tau_VAT, tau_G, tau_CF, tau_EXPORT) = (p.tau_VAT, p.tau_G, p.tau_CF, p.tau_EXPORT);
    let (P_bar_h, P_bar_CF_h, t_now) = (agg.P_bar_h, agg.P_bar_CF_h, agg.t);
    let (C_j, P_j, C_l, P_l) = (gov.C_j, gov.P_j, rotw.C_l, rotw.P_l);
    let (pi_EA, Y_EA, r_bar) = (rotw.pi_EA, rotw.Y_EA, m.cb.r_bar);

    let d = &mut m.data;
    let t = d.collection_time.len() - 1;
    d.nominal_gdp[t] = nominal_gdp;
    d.real_gdp[t] = real_gdp;
    d.nominal_gva[t] = nominal_gva;
    d.real_gva[t] = real_gva;
    d.nominal_household_consumption[t] = (1.0 + tau_VAT) * tot_C_h;
    d.real_household_consumption[t] = (1.0 + tau_VAT) * tot_C_h / P_bar_h;
    d.nominal_government_consumption[t] = (1.0 + tau_G) * C_j;
    d.real_government_consumption[t] = (1.0 + tau_G) * C_j / P_j;
    d.nominal_capitalformation[t] = nominal_capitalformation;
    d.real_capitalformation[t] = real_capitalformation;
    d.nominal_fixed_capitalformation[t] = nominal_fixed_capitalformation;
    d.real_fixed_capitalformation[t] = real_fixed_capitalformation;
    d.nominal_fixed_capitalformation_dwellings[t] = (1.0 + tau_CF) * tot_I_h;
    d.real_fixed_capitalformation_dwellings[t] = (1.0 + tau_CF) * tot_I_h / P_bar_CF_h;
    d.nominal_exports[t] = (1.0 + tau_EXPORT) * C_l;
    d.real_exports[t] = (1.0 + tau_EXPORT) * C_l / P_l;
    d.nominal_imports[t] = nominal_imports;
    d.real_imports[t] = real_imports;
    d.operating_surplus[t] = operating_surplus;
    d.compensation_employees[t] = compensation_employees;
    d.wages[t] = wages;
    d.taxes_production[t] = taxes_production;
    d.nominal_sector_gva[t].copy_from_slice(&nominal_sector_gva);
    d.real_sector_gva[t].copy_from_slice(&real_sector_gva);
    d.euribor[t] = r_bar;
    d.gdp_deflator_growth_ea[t] = pi_EA;
    d.real_gdp_ea[t] = Y_EA;
    d.collection_time[t] = t_now;
}

impl<X: Extension> AbstractModel<X> {
    /// `collect_data!(model)`; see [`crate::base::collect_data`].
    pub fn collect_data(&mut self) {
        X::collect_data(self)
    }
}
