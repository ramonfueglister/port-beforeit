//! Time series of national accounts variables and of each bank, BeforeIT
//! `src/utils/data.jl`.

use crate::calibration::{json_float, json_number};
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
    /// New loans of each bank's firms.
    pub credit_new_per_bank: Vec<Vec<f64>>,
    /// Loans of each bank's firms.
    pub credit_stock_per_bank: Vec<Vec<f64>>,
    /// Equity of each bank.
    pub equity_per_bank: Vec<Vec<f64>>,
    /// Net position of each bank at the central bank.
    pub reserves_per_bank: Vec<Vec<f64>>,
    /// Profits over equity of each bank, 0 for a bank without equity.
    pub roe_per_bank: Vec<Vec<f64>>,
}

/// The fields of [`Data`] with one number per collection, and those with a
/// list per collection: accessors by name and the JSON of
/// [`DataVector::to_json`].
macro_rules! fields {
    (series: $($s:ident),* ; lists: $($l:ident),* $(,)?) => {
        impl Data {
            /// The national accounts series called `name`, Julia
            /// `getproperty(data, Symbol(name))` for the fields with one
            /// number per collection, or `None` if there is no such series.
            pub fn series_by_name(&self, name: &str) -> Option<&[f64]> {
                match name {
                    $(stringify!($s) => Some(&self.$s),)*
                    _ => None,
                }
            }

            fn to_json_value(&self) -> serde_json::Value {
                let numbers = |v: &[f64]| -> serde_json::Value {
                    v.iter().map(|&x| json_number(x)).collect()
                };
                let mut o = serde_json::Map::new();
                o.insert("collection_time".into(), self.collection_time.clone().into());
                $(o.insert(stringify!($s).into(), numbers(&self.$s));)*
                $(o.insert(
                    stringify!($l).into(),
                    self.$l.iter().map(|r| numbers(r)).collect(),
                );)*
                serde_json::Value::Object(o)
            }

            fn from_json_value(v: &serde_json::Value, what: &str) -> Result<Data, String> {
                let o = v.as_object().ok_or_else(|| format!("{what} is not an object"))?;
                let get = |key: &str| {
                    o.get(key)
                        .and_then(|x| x.as_array())
                        .ok_or_else(|| format!("{what}.{key} is missing or not a list"))
                };
                let numbers = |key: &str, x: &[serde_json::Value]| -> Result<Vec<f64>, String> {
                    x.iter().map(|x| json_float(x, &format!("{what}.{key}"))).collect()
                };
                let collection_time = get("collection_time")?
                    .iter()
                    .map(|x| x.as_u64().map(|t| t as usize))
                    .collect::<Option<_>>()
                    .ok_or_else(|| format!("{what}.collection_time: not a list of counts"))?;
                Ok(Data {
                    collection_time,
                    $($s: numbers(stringify!($s), get(stringify!($s))?)?,)*
                    $($l: get(stringify!($l))?
                        .iter()
                        .map(|r| {
                            let r = r.as_array().ok_or_else(|| {
                                format!("{what}.{}: not a list of lists", stringify!($l))
                            })?;
                            numbers(stringify!($l), r)
                        })
                        .collect::<Result<_, _>>()?,)*
                })
            }
        }
    };
}

fields!(
    series: nominal_gdp, real_gdp, nominal_gva, real_gva, nominal_household_consumption,
    real_household_consumption, nominal_government_consumption, real_government_consumption,
    nominal_capitalformation, real_capitalformation, nominal_fixed_capitalformation,
    real_fixed_capitalformation, nominal_fixed_capitalformation_dwellings,
    real_fixed_capitalformation_dwellings, nominal_exports, real_exports, nominal_imports,
    real_imports, operating_surplus, compensation_employees, wages, taxes_production,
    gdp_deflator_growth_ea, real_gdp_ea, euribor;
    lists: nominal_sector_gva, real_sector_gva, credit_new_per_bank, credit_stock_per_bank,
    equity_per_bank, reserves_per_bank, roe_per_bank,
);

/// The data of several runs, BeforeIT `DataVector`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DataVector {
    pub vector: Vec<Data>,
}

impl DataVector {
    /// `DataVector(model)`: the data of one model.
    pub fn from_model<X>(model: &AbstractModel<X>) -> DataVector {
        DataVector {
            vector: vec![model.data.clone()],
        }
    }

    /// `DataVector(model_vec)`: the data of several models.
    pub fn from_models<'a, X: 'a>(
        models: impl IntoIterator<Item = &'a AbstractModel<X>>,
    ) -> DataVector {
        DataVector {
            vector: models.into_iter().map(|m| m.data.clone()).collect(),
        }
    }

    /// One series of every run side by side, Julia `data_vector.name`
    /// (`hcat` of the series): row `t` holds the value of each run at
    /// collection `t`. For example, `dv.series(|d| &d.real_gdp)`.
    ///
    /// # Panics
    ///
    /// If the runs have series of different lengths (Julia throws a
    /// `DimensionMismatch`).
    pub fn series<T: Clone>(&self, field: impl Fn(&Data) -> &[T]) -> Vec<Vec<T>> {
        let columns: Vec<&[T]> = self.vector.iter().map(field).collect();
        let T = columns.first().map_or(0, |c| c.len());
        assert!(
            columns.iter().all(|c| c.len() == T),
            "the runs have series of different lengths"
        );
        (0..T)
            .map(|t| columns.iter().map(|c| c[t].clone()).collect())
            .collect()
    }

    /// Number of runs.
    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vector.is_empty()
    }

    /// The data of each run.
    pub fn iter(&self) -> std::slice::Iter<'_, Data> {
        self.vector.iter()
    }

    /// Writes the data of all runs as JSON, the port's format for the
    /// `data_vector` that BeforeIT saves in JLD2 files: `{"vector": [...]}`
    /// with one object per run, whose keys are the fields of [`Data`]:
    /// lists of numbers, and lists of lists for the fields per sector and
    /// per bank. NaN and infinities are written as the strings "NaN",
    /// "Inf" and "-Inf". [`DataVector::from_json`] reads it back exactly.
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "vector": self.vector.iter().map(Data::to_json_value).collect::<Vec<_>>()
        })
        .to_string()
    }

    /// Reads the JSON of [`DataVector::to_json`].
    pub fn from_json(text: &str) -> Result<DataVector, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("data vector: {e}"))?;
        let runs = v
            .get("vector")
            .and_then(|x| x.as_array())
            .ok_or("data vector: no list \"vector\"")?;
        let vector = runs
            .iter()
            .enumerate()
            .map(|(i, r)| Data::from_json_value(r, &format!("vector[{i}]")))
            .collect::<Result<_, _>>()?;
        Ok(DataVector { vector })
    }
}

impl std::ops::Index<usize> for DataVector {
    type Output = Data;

    /// The data of run `i`, from 0 (Julia `data_vector[i]`, from 1).
    fn index(&self, i: usize) -> &Data {
        &self.vector[i]
    }
}

impl<'a> IntoIterator for &'a DataVector {
    type Item = &'a Data;
    type IntoIter = std::slice::Iter<'a, Data>;

    fn into_iter(self) -> Self::IntoIter {
        self.vector.iter()
    }
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
/// the series per sector and per bank.
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
    let n_banks = m.prop.n_banks;
    for series in [
        &mut d.credit_new_per_bank,
        &mut d.credit_stock_per_bank,
        &mut d.equity_per_bank,
        &mut d.reserves_per_bank,
        &mut d.roe_per_bank,
    ] {
        series.push(vec![0.0; n_banks]);
    }
}

/// Sum of `f(i)` over the firms of each sector.
fn by_sector<X>(m: &AbstractModel<X>, f: impl Fn(usize) -> f64) -> Vec<f64> {
    let sectors = m.firms.by_sector(m.prop.G);
    sectors
        .iter()
        .map(|firms| sum_by(firms.len(), |k| f(firms[k])))
        .collect()
}

/// New loans, loans, equity, reserves and return on equity of each bank, as
/// `update_data_init!` and `update_data_step!` record them.
fn per_bank<X>(m: &AbstractModel<X>) -> [Vec<f64>; 5] {
    let (f, b, n) = (&m.firms, &m.banks, m.prop.n_banks);
    let mut v: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; n]);
    for k in 0..n {
        // From left to right, as Julia's `sum` with `init`.
        v[0][k] = b.firms[k]
            .iter()
            .fold(0.0, |s, &id| s + f.DL_i[f.index(id)]);
        v[1][k] = b.firms[k].iter().fold(0.0, |s, &id| s + f.L_i[f.index(id)]);
        v[2][k] = b.E_k[k];
        v[3][k] = b.D_k[k];
        v[4][k] = if b.E_k[k] != 0.0 {
            b.Pi_k[k] / b.E_k[k]
        } else {
            0.0
        };
    }
    v
}

/// Stores the series per bank in entry `t`.
fn set_per_bank(d: &mut Data, t: usize, v: [Vec<f64>; 5]) {
    let [credit_new, credit_stock, equity, reserves, roe] = v;
    d.credit_new_per_bank[t].copy_from_slice(&credit_new);
    d.credit_stock_per_bank[t].copy_from_slice(&credit_stock);
    d.equity_per_bank[t].copy_from_slice(&equity);
    d.reserves_per_bank[t].copy_from_slice(&reserves);
    d.roe_per_bank[t].copy_from_slice(&roe);
}

/// `update_data_init!`: the initial state, at prices of 1, in the first
/// entry.
pub fn update_data_init<X: Extension>(m: &mut AbstractModel<X>) {
    let (p, f) = (&m.prop, &m.firms);
    let I = f.len();
    let tot_Y_h = sum(&m.w_act.Y_h) + sum(&m.w_inact.Y_h) + sum(&f.Y_h) + sum(&m.banks.Y_h);
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
    let banks = per_bank(m);

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
    set_per_bank(d, 0, banks);
    d.collection_time[0] = 1;
}

/// `update_data_step!`: the values after a quarter, in the last entry.
pub fn update_data_step<X: Extension>(m: &mut AbstractModel<X>) {
    let (p, f, agg) = (&m.prop, &m.firms, &m.agg);
    let (gov, rotw) = (&m.gov, &m.rotw);
    let I = f.len();
    let tot_C_h = sum(&m.w_act.C_h) + sum(&m.w_inact.C_h) + sum(&f.C_h) + sum(&m.banks.C_h);
    let tot_I_h = sum(&m.w_act.I_h) + sum(&m.w_inact.I_h) + sum(&f.I_h) + sum(&m.banks.I_h);
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
    let banks = per_bank(m);

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
    set_per_bank(d, t, banks);
    d.collection_time[t] = t_now;
}

impl<X: Extension> AbstractModel<X> {
    /// `collect_data!(model)`; see [`crate::base::collect_data`].
    pub fn collect_data(&mut self) {
        X::collect_data(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_vector_json_round_trip() {
        let mut d = Data {
            collection_time: vec![1, 2],
            real_gdp: vec![1.0, 0.1 + 0.2],
            euribor: vec![f64::INFINITY, -0.0],
            equity_per_bank: vec![vec![1.5, 2.5], vec![f64::NEG_INFINITY, 3.0]],
            ..Data::default()
        };
        let dv = DataVector {
            vector: vec![d.clone(), Data::default()],
        };
        assert_eq!(DataVector::from_json(&dv.to_json()).unwrap(), dv);
        d.wages = vec![f64::NAN];
        let back = DataVector::from_json(&DataVector { vector: vec![d] }.to_json()).unwrap();
        assert!(back[0].wages[0].is_nan());
        assert!(back[0].euribor[1].is_sign_negative());
        assert!(DataVector::from_json("{\"vector\": [{}]}").is_err());
    }
}
