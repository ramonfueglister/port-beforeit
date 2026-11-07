//! Time series of national accounts variables, BeforeIT `src/utils/data.jl`.
//! The per-bank series of BeforeIT's multi-bank extension
//! (`credit_new_per_bank` and the like) are not collected.

use crate::Model;
use crate::julia::{sum, sum_by};

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

/// The values of one collection.
struct Entry {
    collection_time: usize,
    nominal_gdp: f64,
    real_gdp: f64,
    nominal_gva: f64,
    real_gva: f64,
    nominal_household_consumption: f64,
    real_household_consumption: f64,
    nominal_government_consumption: f64,
    real_government_consumption: f64,
    nominal_capitalformation: f64,
    real_capitalformation: f64,
    nominal_fixed_capitalformation: f64,
    real_fixed_capitalformation: f64,
    nominal_fixed_capitalformation_dwellings: f64,
    real_fixed_capitalformation_dwellings: f64,
    nominal_exports: f64,
    real_exports: f64,
    nominal_imports: f64,
    real_imports: f64,
    operating_surplus: f64,
    compensation_employees: f64,
    wages: f64,
    taxes_production: f64,
    gdp_deflator_growth_ea: f64,
    real_gdp_ea: f64,
    euribor: f64,
    nominal_sector_gva: Vec<f64>,
    real_sector_gva: Vec<f64>,
}

impl Data {
    fn push(&mut self, e: Entry) {
        self.collection_time.push(e.collection_time);
        self.nominal_gdp.push(e.nominal_gdp);
        self.real_gdp.push(e.real_gdp);
        self.nominal_gva.push(e.nominal_gva);
        self.real_gva.push(e.real_gva);
        self.nominal_household_consumption
            .push(e.nominal_household_consumption);
        self.real_household_consumption
            .push(e.real_household_consumption);
        self.nominal_government_consumption
            .push(e.nominal_government_consumption);
        self.real_government_consumption
            .push(e.real_government_consumption);
        self.nominal_capitalformation
            .push(e.nominal_capitalformation);
        self.real_capitalformation.push(e.real_capitalformation);
        self.nominal_fixed_capitalformation
            .push(e.nominal_fixed_capitalformation);
        self.real_fixed_capitalformation
            .push(e.real_fixed_capitalformation);
        self.nominal_fixed_capitalformation_dwellings
            .push(e.nominal_fixed_capitalformation_dwellings);
        self.real_fixed_capitalformation_dwellings
            .push(e.real_fixed_capitalformation_dwellings);
        self.nominal_exports.push(e.nominal_exports);
        self.real_exports.push(e.real_exports);
        self.nominal_imports.push(e.nominal_imports);
        self.real_imports.push(e.real_imports);
        self.operating_surplus.push(e.operating_surplus);
        self.compensation_employees.push(e.compensation_employees);
        self.wages.push(e.wages);
        self.taxes_production.push(e.taxes_production);
        self.gdp_deflator_growth_ea.push(e.gdp_deflator_growth_ea);
        self.real_gdp_ea.push(e.real_gdp_ea);
        self.euribor.push(e.euribor);
        self.nominal_sector_gva.push(e.nominal_sector_gva);
        self.real_sector_gva.push(e.real_sector_gva);
    }
}

impl Model {
    /// `collect_data!`: appends the values of the current state; the first
    /// call records the initial state.
    pub fn collect_data(&mut self) {
        let e = if self.data.collection_time.is_empty() {
            self.data_init()
        } else {
            self.data_step()
        };
        self.data.push(e);
    }

    /// Sum of `f(i)` over the firms of each sector.
    fn by_sector(&self, f: impl Fn(usize) -> f64) -> Vec<f64> {
        let sectors = self.firms.by_sector(self.prop.G);
        sectors
            .iter()
            .map(|firms| sum_by(firms.len(), |k| f(firms[k])))
            .collect()
    }

    /// `update_data_init!`: the initial state, at prices of 1.
    fn data_init(&self) -> Entry {
        let (p, f) = (&self.prop, &self.firms);
        let I = f.len();
        let tot_Y_h = sum(&self.w_act.Y_h) + sum(&self.w_inact.Y_h) + sum(&f.Y_h) + self.bank.Y_h;
        let (C_G, C_E) = (self.gov.C_G, self.rotw.C_E);
        let nominal_gdp = sum_by(I, |i| f.Y_i[i] * (1.0 - 1.0 / f.beta_i[i]))
            + tot_Y_h * p.psi / (1.0 / p.tau_VAT + 1.0)
            + p.tau_G * C_G
            + tot_Y_h * p.psi_H / (1.0 / p.tau_CF + 1.0)
            + p.tau_EXPORT * C_E;
        let gva = |i: usize| f.Y_i[i] * ((1.0 - f.tau_Y_i[i]) - 1.0 / f.beta_i[i]);
        let nominal_gva = sum_by(I, gva);
        let household_consumption = tot_Y_h * p.psi;
        let government_consumption = (1.0 + p.tau_G) * C_G;
        let dwellings = tot_Y_h * p.psi_H;
        let capitalformation =
            sum_by(I, |i| f.Y_i[i] * f.delta_i[i] / f.kappa_i[i]) + tot_Y_h * p.psi_H;
        let exports = (1.0 + p.tau_EXPORT) * C_E;
        let imports = self.rotw.Y_I;
        let operating_surplus = sum_by(I, |i| {
            f.Y_i[i]
                * (1.0 - ((1.0 + p.tau_SIF) * f.w_bar_i[i] / f.alpha_bar_i[i] + 1.0 / f.beta_i[i]))
                - f.tau_K_i[i] * f.Y_i[i]
                - f.tau_Y_i[i] * f.Y_i[i]
        });
        let wages = sum_by(I, |i| f.w_bar_i[i] * f.N_i[i] as f64);
        let sector_gva = self.by_sector(gva);
        Entry {
            collection_time: 1,
            nominal_gdp,
            real_gdp: nominal_gdp,
            nominal_gva,
            real_gva: nominal_gva,
            nominal_household_consumption: household_consumption,
            real_household_consumption: household_consumption,
            nominal_government_consumption: government_consumption,
            real_government_consumption: government_consumption,
            nominal_capitalformation: capitalformation,
            real_capitalformation: capitalformation,
            nominal_fixed_capitalformation: capitalformation,
            real_fixed_capitalformation: capitalformation,
            nominal_fixed_capitalformation_dwellings: dwellings,
            real_fixed_capitalformation_dwellings: dwellings,
            nominal_exports: exports,
            real_exports: exports,
            nominal_imports: imports,
            real_imports: imports,
            operating_surplus,
            compensation_employees: (1.0 + p.tau_SIF) * wages,
            wages,
            taxes_production: sum_by(I, |i| f.tau_K_i[i] * f.Y_i[i]),
            gdp_deflator_growth_ea: self.rotw.pi_EA,
            real_gdp_ea: self.rotw.Y_EA,
            euribor: self.cb.r_bar,
            nominal_sector_gva: sector_gva.clone(),
            real_sector_gva: sector_gva,
        }
    }

    /// `update_data_step!`: the values after a quarter.
    fn data_step(&self) -> Entry {
        let (p, f, agg) = (&self.prop, &self.firms, &self.agg);
        let (gov, rotw) = (&self.gov, &self.rotw);
        let I = f.len();
        let tot_C_h = sum(&self.w_act.C_h) + sum(&self.w_inact.C_h) + sum(&f.C_h) + self.bank.C_h;
        let tot_I_h = sum(&self.w_act.I_h) + sum(&self.w_inact.I_h) + sum(&f.I_h) + self.bank.I_h;
        let sales_net = |i: usize| (1.0 - f.tau_Y_i[i]) * f.P_i[i] * f.Y_i[i];
        let materials = |i: usize| 1.0 / f.beta_i[i] * f.P_bar_i[i] * f.Y_i[i];
        let real_gva_i = |i: usize| f.Y_i[i] * ((1.0 - f.tau_Y_i[i]) - 1.0 / f.beta_i[i]);
        let nominal_gva = sum_by(I, sales_net) - sum_by(I, materials);
        let real_gva = sum_by(I, real_gva_i);
        let nominal_gdp = sum_by(I, |i| f.tau_Y_i[i] * f.Y_i[i] * f.P_i[i])
            + p.tau_VAT * tot_C_h
            + p.tau_CF * tot_I_h
            + p.tau_G * gov.C_j
            + p.tau_EXPORT * rotw.C_l
            + sum_by(I, sales_net)
            - sum_by(I, materials);
        let real_gdp = real_gva
            + sum_by(I, |i| f.tau_Y_i[i] * f.Y_i[i])
            + p.tau_VAT * tot_C_h / agg.P_bar_h
            + p.tau_CF * tot_I_h / agg.P_bar_CF_h
            + p.tau_G * gov.C_j / gov.P_j
            + p.tau_EXPORT * rotw.C_l / rotw.P_l;
        let investment_nominal = sum_by(I, |i| f.P_CF_i[i] * f.I_i[i]);
        let investment_real = sum(&f.I_i);
        let dwellings = (1.0 + p.tau_CF) * tot_I_h;
        let wages = sum_by(I, |i| f.w_i[i] * f.N_i[i] as f64) * agg.P_bar_HH;
        Entry {
            collection_time: agg.t,
            nominal_gdp,
            real_gdp,
            nominal_gva,
            real_gva,
            nominal_household_consumption: (1.0 + p.tau_VAT) * tot_C_h,
            real_household_consumption: (1.0 + p.tau_VAT) * tot_C_h / agg.P_bar_h,
            nominal_government_consumption: (1.0 + p.tau_G) * gov.C_j,
            real_government_consumption: (1.0 + p.tau_G) * gov.C_j / gov.P_j,
            nominal_capitalformation: investment_nominal
                + dwellings
                + sum_by(I, |i| f.DS_i[i] * f.P_i[i])
                + sum_by(I, |i| f.DM_i[i] * f.P_bar_i[i] - materials(i)),
            real_capitalformation: investment_real
                + dwellings / agg.P_bar_CF_h
                + sum_by(I, |i| f.DM_i[i] - f.Y_i[i] / f.beta_i[i])
                + sum(&f.DS_i),
            nominal_fixed_capitalformation: investment_nominal + dwellings,
            real_fixed_capitalformation: investment_real + dwellings / agg.P_bar_CF_h,
            nominal_fixed_capitalformation_dwellings: dwellings,
            real_fixed_capitalformation_dwellings: dwellings / agg.P_bar_CF_h,
            nominal_exports: (1.0 + p.tau_EXPORT) * rotw.C_l,
            real_exports: (1.0 + p.tau_EXPORT) * rotw.C_l / rotw.P_l,
            nominal_imports: sum_by(rotw.P_m.len(), |g| rotw.P_m[g] * rotw.Q_m[g]),
            real_imports: sum(&rotw.Q_m),
            operating_surplus: sum_by(I, |i| {
                f.P_i[i] * f.Q_i[i] + f.P_i[i] * f.DS_i[i]
                    - (1.0 + p.tau_SIF) * f.w_i[i] * f.N_i[i] as f64 * agg.P_bar_HH
                    - materials(i)
                    - f.tau_Y_i[i] * f.P_i[i] * f.Y_i[i]
                    - f.tau_K_i[i] * f.P_i[i] * f.Y_i[i]
            }),
            compensation_employees: (1.0 + p.tau_SIF)
                * sum_by(I, |i| f.w_i[i] * f.N_i[i] as f64)
                * agg.P_bar_HH,
            wages,
            taxes_production: sum_by(I, |i| f.tau_K_i[i] * f.Y_i[i] * f.P_i[i]),
            gdp_deflator_growth_ea: rotw.pi_EA,
            real_gdp_ea: rotw.Y_EA,
            euribor: self.cb.r_bar,
            nominal_sector_gva: self
                .by_sector(sales_net)
                .iter()
                .zip(self.by_sector(materials))
                .map(|(a, b)| a - b)
                .collect(),
            real_sector_gva: self.by_sector(real_gva_i),
        }
    }
}
