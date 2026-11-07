//! Model types: the Rust counterpart of BeforeIT's model types, which are
//! created with `Bit.@object` and get their own methods by multiple
//! dispatch.
//!
//! In BeforeIT.jl every rule is a function of the model, for example
//! `central_bank_rate(model)`, and a new model type changes the behaviour by
//! defining a method of that function for its type; a new agent type adds
//! fields. Here a model type is a type `X` that implements [`Extension`]:
//! [`AbstractModel<X>`](crate::AbstractModel) stores a value of it in the
//! field `ext`, which holds the fields the model type adds, and each method
//! of [`Extension`] is a rule that the model type can replace. The default
//! of each method is BeforeIT's method for `AbstractModel`, which is also
//! available in [`crate::base`], so a replacement can call it as Julia's
//! `@invoke` does.
//!
//! The rules run through these methods wherever the Julia code calls the
//! function, also from inside other rules. Rules that draw random numbers in
//! BeforeIT.jl or in one of its model extensions receive the generator;
//! a replacement of another rule that needs random numbers can keep its own
//! generator in `ext`.

use crate::AbstractModel;
use crate::agents::{Firms, Households};
use crate::base;
use crate::diff_transform::Part;
use crate::firms::FirmsDecisions;
use crate::random::Randomness;

/// The model type of BeforeIT's `Model`: no added fields, every rule as in
/// BeforeIT.jl.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Standard;

impl Extension for Standard {}

/// The rules that a model type can replace, one method per function of
/// BeforeIT.jl. See the [module documentation](self).
///
/// A model type with a fixed policy rate, as in BeforeIT
/// `examples/basic_inheritance.jl`:
///
/// ```
/// use beforeit::{base, AbstractModel, Extension};
///
/// struct FixedRate {
///     fixed_rate: f64,
/// }
///
/// impl Extension for FixedRate {
///     fn central_bank_rate(m: &mut AbstractModel<Self>) -> f64 {
///         m.ext.fixed_rate
///     }
/// }
///
/// /// A model type that keeps the Taylor rule but never goes below 1%.
/// struct Floor;
///
/// impl Extension for Floor {
///     fn central_bank_rate(m: &mut AbstractModel<Self>) -> f64 {
///         base::central_bank_rate(m).max(0.01)
///     }
/// }
/// ```
#[allow(unused_variables)]
pub trait Extension: Sized {
    // ---- src/utils/estimate.jl and epsilon.jl ----

    /// `estimate_next_value(data)`: AR(1) forecast of the next value of a
    /// series, with a random shock. BeforeIT `examples/change_expectations.jl`
    /// redefines it.
    fn estimate_next_value<R: Randomness + ?Sized>(data: &[f64], rng: &mut R) -> f64 {
        crate::estimate::estimate_next_value(data, rng)
    }

    /// `epsilon(model)`: the shocks to euro area GDP, exports and imports.
    fn epsilon<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, f64, f64) {
        base::epsilon(m, rng)
    }

    // ---- src/agent_actions/estimations.jl ----

    /// `growth_inflation_expectations(model)`: expected GDP, growth and
    /// inflation.
    fn growth_inflation_expectations<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, f64, f64) {
        base::growth_inflation_expectations(m, rng)
    }

    /// `growth_inflation_EA(model)`: euro area GDP, growth and inflation.
    fn growth_inflation_EA<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, f64, f64) {
        base::growth_inflation_EA(m, rng)
    }

    /// `inflation_priceindex(model)`: inflation and price index.
    fn inflation_priceindex(m: &mut AbstractModel<Self>) -> (f64, f64) {
        base::inflation_priceindex(m)
    }

    /// `sector_specific_priceindex(model)`.
    fn sector_specific_priceindex(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::sector_specific_priceindex(m)
    }

    /// `capital_formation_priceindex(model)`.
    fn capital_formation_priceindex(m: &mut AbstractModel<Self>) -> f64 {
        base::capital_formation_priceindex(m)
    }

    /// `households_priceindex(model)`.
    fn households_priceindex(m: &mut AbstractModel<Self>) -> f64 {
        base::households_priceindex(m)
    }

    // ---- src/agent_actions/central_bank.jl ----

    /// `central_bank_rate(model)`: the policy rate.
    fn central_bank_rate(m: &mut AbstractModel<Self>) -> f64 {
        base::central_bank_rate(m)
    }

    /// `central_bank_profits(cb, model)`.
    fn central_bank_profits(m: &mut AbstractModel<Self>) -> f64 {
        base::central_bank_profits(m)
    }

    /// `central_bank_equity(model)`.
    fn central_bank_equity(m: &mut AbstractModel<Self>) -> f64 {
        base::central_bank_equity(m)
    }

    // ---- src/agent_actions/bank.jl ----

    /// `bank_exposure(model, bank_id; expected, overdrafts)`.
    fn bank_exposure(
        m: &AbstractModel<Self>,
        bank_id: usize,
        expected: bool,
        overdrafts: bool,
    ) -> f64 {
        base::bank_exposure(m, bank_id, expected, overdrafts)
    }

    /// `banks_rate(model)`: the lending rate, the same for every bank.
    fn banks_rate(m: &mut AbstractModel<Self>) -> f64 {
        base::banks_rate(m)
    }

    /// `banks_expected_profits(model)`, one entry per bank.
    fn banks_expected_profits(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::banks_expected_profits(m)
    }

    /// `banks_profits(model)`.
    fn banks_profits(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::banks_profits(m)
    }

    /// `banks_equity(model)`.
    fn banks_equity(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::banks_equity(m)
    }

    /// `banks_deposits(model)`.
    fn banks_deposits(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::banks_deposits(m)
    }

    /// `finance_insolvent_firms!(model)`.
    fn finance_insolvent_firms(m: &mut AbstractModel<Self>) {
        base::finance_insolvent_firms(m)
    }

    // ---- src/agent_actions/firms.jl ----

    /// `cost_push_inflation(firms, model)`.
    fn cost_push_inflation(m: &AbstractModel<Self>, firms: &Firms) -> Vec<f64> {
        base::cost_push_inflation(m, firms)
    }

    /// `desired_capital_material_employment(firms, Q_s_i)`: desired
    /// investment, intermediate inputs and employees.
    fn desired_capital_material_employment(
        firms: &Firms,
        Q_s_i: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        base::desired_capital_material_employment(firms, Q_s_i)
    }

    /// `expected_deposits_capital_loans(firms, model, Pi_e_i)`.
    fn expected_deposits_capital_loans(
        m: &AbstractModel<Self>,
        firms: &Firms,
        Pi_e_i: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        base::expected_deposits_capital_loans(m, firms, Pi_e_i)
    }

    /// `firms_expectations_and_decisions(model)`.
    fn firms_expectations_and_decisions<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> FirmsDecisions {
        base::firms_expectations_and_decisions(m, rng)
    }

    /// `firms_wages(model)`.
    fn firms_wages(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::firms_wages(m)
    }

    /// `firms_production(model)`.
    fn firms_production(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::firms_production(m)
    }

    /// `leontief_production(Q_s_i, N_i, alpha_i, K_i, kappa_i, M_i, beta_i)`.
    #[allow(clippy::too_many_arguments)]
    fn leontief_production(
        Q_s_i: &[f64],
        N_i: &[i64],
        alpha_i: &[f64],
        K_i: &[f64],
        kappa_i: &[f64],
        M_i: &[f64],
        beta_i: &[f64],
    ) -> Vec<f64> {
        base::leontief_production(Q_s_i, N_i, alpha_i, K_i, kappa_i, M_i, beta_i)
    }

    /// `firms_stocks(model)`: capital, intermediate inputs, change in
    /// inventories and inventories.
    fn firms_stocks(m: &mut AbstractModel<Self>) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        base::firms_stocks(m)
    }

    /// `firms_profits(model)`.
    fn firms_profits(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::firms_profits(m)
    }

    /// `firms_deposits(model)`.
    fn firms_deposits(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::firms_deposits(m)
    }

    /// `firms_loans(model)`.
    fn firms_loans(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::firms_loans(m)
    }

    /// `firms_equity(model)`.
    fn firms_equity(m: &mut AbstractModel<Self>) -> Vec<f64> {
        base::firms_equity(m)
    }

    // ---- src/agent_actions/households.jl ----

    /// `update_workers_wages!(model)`.
    fn update_workers_wages(m: &mut AbstractModel<Self>) {
        base::update_workers_wages(m)
    }

    /// `households_income_act(model; expected)`.
    fn households_income_act(m: &mut AbstractModel<Self>, expected: bool) -> Vec<f64> {
        base::households_income_act(m, expected)
    }

    /// `households_income_inact(model; expected)`.
    fn households_income_inact(m: &mut AbstractModel<Self>, expected: bool) -> Vec<f64> {
        base::households_income_inact(m, expected)
    }

    /// `households_income_firms(model; expected)`.
    fn households_income_firms(m: &mut AbstractModel<Self>, expected: bool) -> Vec<f64> {
        base::households_income_firms(m, expected)
    }

    /// `households_income_banks(model; expected)`, one entry per bank.
    fn households_income_banks(m: &mut AbstractModel<Self>, expected: bool) -> Vec<f64> {
        base::households_income_banks(m, expected)
    }

    /// `households_budget_act(model)`: consumption and investment budgets.
    fn households_budget_act(m: &mut AbstractModel<Self>) -> (Vec<f64>, Vec<f64>) {
        base::households_budget_act(m)
    }

    /// `households_budget_inact(model)`.
    fn households_budget_inact(m: &mut AbstractModel<Self>) -> (Vec<f64>, Vec<f64>) {
        base::households_budget_inact(m)
    }

    /// `households_budget_firms(model)`.
    fn households_budget_firms(m: &mut AbstractModel<Self>) -> (Vec<f64>, Vec<f64>) {
        base::households_budget_firms(m)
    }

    /// `households_budget_banks(model)`.
    fn households_budget_banks(m: &mut AbstractModel<Self>) -> (Vec<f64>, Vec<f64>) {
        base::households_budget_banks(m)
    }

    /// `households_deposits(households, r, model)`.
    fn households_deposits<H: Households + ?Sized>(
        m: &AbstractModel<Self>,
        households: &H,
        r: &[f64],
    ) -> Vec<f64> {
        base::households_deposits(m, households, r)
    }

    // ---- src/agent_actions/government.jl ----

    /// `gov_social_benefits(model)`: `sb_other` and `sb_inact`.
    fn gov_social_benefits(m: &mut AbstractModel<Self>) -> (f64, f64) {
        base::gov_social_benefits(m)
    }

    /// `gov_expenditure(model)`: `C_G` and `C_d_j`.
    fn gov_expenditure<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, Vec<f64>) {
        base::gov_expenditure(m, rng)
    }

    /// `gov_revenues(model)`.
    fn gov_revenues(m: &mut AbstractModel<Self>) -> f64 {
        base::gov_revenues(m)
    }

    /// `gov_loans(model)`.
    fn gov_loans(m: &mut AbstractModel<Self>) -> f64 {
        base::gov_loans(m)
    }

    // ---- src/agent_actions/rotw.jl ----

    /// `rotw_import_export(model)`: `C_E`, `Y_I`, `C_d_l`, `Y_m` and `P_m`.
    #[allow(clippy::type_complexity)]
    fn rotw_import_export<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) -> (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        base::rotw_import_export(m, rng)
    }

    /// `rotw_deposits(model)`.
    fn rotw_deposits(m: &mut AbstractModel<Self>) -> f64 {
        base::rotw_deposits(m)
    }

    // ---- src/agent_actions/aggregates.jl ----

    /// `gross_domestic_product(model)`.
    fn gross_domestic_product(m: &mut AbstractModel<Self>) -> f64 {
        base::gross_domestic_product(m)
    }

    /// `set_time!(model)`.
    fn set_time(m: &mut AbstractModel<Self>) {
        base::set_time(m)
    }

    // ---- src/markets/ ----

    /// `search_and_matching_credit!(model)`.
    fn search_and_matching_credit<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) {
        base::search_and_matching_credit(m, rng)
    }

    /// `search_and_matching_labour!(model)`.
    fn search_and_matching_labour<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
    ) {
        base::search_and_matching_labour(m, rng)
    }

    /// `search_and_matching!(model; parallel)`.
    fn search_and_matching<R: Randomness + ?Sized>(
        m: &mut AbstractModel<Self>,
        rng: &mut R,
        parallel: bool,
    ) {
        base::search_and_matching(m, rng, parallel)
    }

    // ---- src/utils/data.jl ----

    /// `collect_data!(model)`.
    fn collect_data(m: &mut AbstractModel<Self>) {
        base::collect_data(m)
    }

    /// `allocate_new_data!(model)`.
    fn allocate_new_data(m: &mut AbstractModel<Self>) {
        base::allocate_new_data(m)
    }

    /// `update_data_init!(model)`.
    fn update_data_init(m: &mut AbstractModel<Self>) {
        base::update_data_init(m)
    }

    /// `update_data_step!(model)`.
    fn update_data_step(m: &mut AbstractModel<Self>) {
        base::update_data_step(m)
    }

    // ---- src/utils/diff_transform.jl ----

    /// Appends the numbers of the fields that the model type adds to the
    /// agent type of `part`, for
    /// [`model_to_array`](crate::diff_transform::model_to_array). In
    /// BeforeIT.jl such an agent type is made with
    /// `Bit.@object NewType(BaseType)`, which puts its own fields after those
    /// of the base type, and `model_to_array` takes those that hold a number
    /// or a vector or matrix of numbers. The default adds nothing.
    fn push_fields(&self, part: Part, v: &mut Vec<f64>) {}

    /// Sets the fields of [`push_fields`](Self::push_fields) from the start
    /// of `numbers`, for
    /// [`array_to_model`](crate::diff_transform::array_to_model), and returns
    /// how many numbers it took; vectors keep their length. Returns an error
    /// where Julia throws, for example if `numbers` is too short
    /// ([`set_floats`](crate::diff_transform::set_floats) checks that). The
    /// default takes nothing.
    fn set_fields(&mut self, part: Part, numbers: &[f64]) -> Result<usize, String> {
        Ok(0)
    }
}
