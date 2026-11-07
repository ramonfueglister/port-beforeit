//! Accounting identities, BeforeIT `src/utils/get_accounting_identities.jl`.

use crate::julia::{sum, sum_by};
use crate::{AbstractModel, Data};

/// `get_accounting_identities(data)`: summed over all collections of data,
/// value added minus compensation of employees, operating surplus and taxes
/// on production; nominal GDP minus its uses; and real GDP minus its uses.
/// Each is 0 up to rounding.
pub fn get_accounting_identities(data: &Data) -> (f64, f64, f64) {
    let d = data;
    let n = d.nominal_gva.len();
    let income_and_production = sum_by(n, |t| {
        d.nominal_gva[t]
            - d.compensation_employees[t]
            - d.operating_surplus[t]
            - d.taxes_production[t]
    });
    let gdp_and_expenditure = sum_by(n, |t| {
        d.nominal_gdp[t]
            - d.nominal_household_consumption[t]
            - d.nominal_government_consumption[t]
            - d.nominal_capitalformation[t]
            - d.nominal_exports[t]
            + d.nominal_imports[t]
    });
    let gdp_and_expenditure_real = sum_by(n, |t| {
        d.real_gdp[t]
            - d.real_household_consumption[t]
            - d.real_government_consumption[t]
            - d.real_capitalformation[t]
            - d.real_exports[t]
            + d.real_imports[t]
    });
    (
        income_and_production,
        gdp_and_expenditure,
        gdp_and_expenditure_real,
    )
}

/// `get_accounting_identity_banks(model)`: the balance sheet of the central
/// bank and the combined balance sheet of the banks, each 0 up to rounding.
/// The Julia code adds the vector of the banks' reserves `D_k` to a number,
/// which throws; here the reserves are summed, as in BeforeIT
/// `test/accounting_identities.jl`.
pub fn get_accounting_identity_banks<X>(m: &AbstractModel<X>) -> (f64, f64) {
    let cb_balance = m.cb.E_CB + m.rotw.D_RoW - m.gov.L_G + sum(&m.banks.D_k);
    let tot_D_h = sum(&m.w_act.D_h) + sum(&m.w_inact.D_h) + sum(&m.firms.D_h) + sum(&m.banks.D_h);
    let bank_balance =
        sum(&m.firms.D_i) + tot_D_h + sum(&m.banks.E_k) - sum(&m.firms.L_i) - sum(&m.banks.D_k);
    (cb_balance, bank_balance)
}
