//! Unofficial Rust port of [BeforeIT.jl](https://github.com/bancaditalia/BeforeIT.jl),
//! the Julia implementation of the agent-based macroeconomic model of
//! Poledna, Miess, Hommes and Rabitsch (2023), "Economic forecasting with an
//! agent-based model", *European Economic Review* 151, 104306,
//! <https://doi.org/10.1016/j.euroecorev.2022.104306>.
//!
//! The port follows BeforeIT.jl at commit
//! `09263989b56bd2d58c28657bb8e940142d995cf4` (branch `main`, 29 October
//! 2026), which is newer than the release v0.6.0. Functions and variables
//! keep the names of the Julia code so that each line can be compared with
//! the original; equation numbers A.1 to A.84 refer to the online appendix of
//! the paper. One step is one quarter, and [`AbstractModel::step`] calls the
//! rules in the order of BeforeIT `src/one_step.jl`. All random draws go
//! through [`Randomness`]: [`Xoshiro`] for stochastic runs, [`Deterministic`]
//! for the comparison with the authors' reference values.
//!
//! As in BeforeIT.jl, every rule is a function of the model (`banks_rate`,
//! `firms_profits`, ...) whose result a `set_` function stores (`set_banks_rate`,
//! `set_firms_profits`, ...), and a model type can replace any rule; see
//! [`Extension`]. [`Model`] is BeforeIT's `Model`. [`AUSTRIA2010Q1`],
//! [`ITALY2010Q1`] and [`STEADY_STATE2010Q1`] are BeforeIT's data sets;
//! [`get_params_and_initial_conditions`] computes parameters and initial
//! conditions from calibration data such as [`ITALY_CALIBRATION`].
//!
//! Indices start at 0: sector `g` and firm `i` are Julia's `g + 1` and `i + 1`.
//! Where this port deliberately differs from the Julia code is listed in
//! `README.md`, section "Differences from BeforeIT.jl".

#![forbid(unsafe_code)]
#![allow(non_snake_case, clippy::needless_range_loop)]

pub mod agents;
pub mod analysis;
pub mod calibration;
pub mod correlation;
pub mod data;
pub mod datasets;
pub mod diff_transform;
pub mod estimate;
pub mod evaluation;
pub mod extension;
pub mod julia;
pub mod model_extensions;
pub mod params;
#[cfg(feature = "plots")]
pub mod plots;
pub mod predictions;
pub mod random;
pub mod shocks;
pub mod varx;

mod accounting;
mod aggregates;
mod bank;
mod firms;
mod government;
mod households;
mod markets;
mod rotw;

pub use accounting::{get_accounting_identities, get_accounting_identity_banks};
pub use agents::{
    Aggregates, Banks, CentralBank, Firm, Firms, Government, Households, RestOfTheWorld, Worker,
    Workers,
};
pub use bank::taylor_rule;
pub use calibration::{
    Calibration, CalibrationData, Date, ITALY_CALIBRATION, get_params_and_initial_conditions,
    get_valid_calibration_quarters,
};
pub use data::{Data, DataVector};
pub use datasets::{AUSTRIA2010Q1, ITALY2010Q1, InitialState, STEADY_STATE2010Q1};
pub use extension::{Extension, Standard};
pub use firms::FirmsDecisions;
pub use model_extensions::canvas::{Canvas, ModelCANVAS};
pub use model_extensions::growth_rate::{GrowthRate, ModelGR};
pub use params::{InitialConditions, Parameters, Properties};
pub use predictions::{Predictions, get_predictions_from_sims};
pub use random::{Deterministic, Randomness, WeightVector, Xoshiro};
pub use shocks::{
    ConsumptionShock, InterestRateShock, NoShock, ProductivityShock, Shock, ensemblerun,
};

/// BeforeIT's methods for `AbstractModel`: the rules as BeforeIT.jl defines
/// them, before any model type replaces them. They are the defaults of the
/// methods of [`Extension`]; calling one from a replacement is Julia's
/// `@invoke`. Inside, they call the other rules through the model type, as
/// the Julia code does.
pub mod base {
    pub use crate::aggregates::{
        capital_formation_priceindex, epsilon, gross_domestic_product, growth_inflation_EA,
        growth_inflation_expectations, households_priceindex, inflation_priceindex,
        sector_specific_priceindex, set_time,
    };
    pub use crate::bank::{
        bank_exposure, banks_deposits, banks_equity, banks_expected_profits, banks_profits,
        banks_rate, central_bank_equity, central_bank_profits, central_bank_rate,
        finance_insolvent_firms,
    };
    pub use crate::data::{allocate_new_data, collect_data, update_data_init, update_data_step};
    pub use crate::firms::{
        cost_push_inflation, desired_capital_material_employment, expected_deposits_capital_loans,
        firms_deposits, firms_equity, firms_expectations_and_decisions, firms_loans,
        firms_production, firms_profits, firms_stocks, firms_wages, leontief_production,
    };
    pub use crate::government::{gov_expenditure, gov_loans, gov_revenues, gov_social_benefits};
    pub use crate::households::{
        households_budget_act, households_budget_banks, households_budget_firms,
        households_budget_inact, households_deposits, households_income_act,
        households_income_banks, households_income_firms, households_income_inact,
        update_workers_wages,
    };
    pub use crate::markets::{
        search_and_matching, search_and_matching_credit, search_and_matching_labour,
    };
    pub use crate::rotw::{rotw_deposits, rotw_import_export};
}

/// A model of any model type `X`, BeforeIT's `AbstractModel`: the agents,
/// the fixed properties, the collected data and, in `ext`, the fields that
/// the model type adds (see [`Extension`]).
#[derive(Clone, Debug, PartialEq)]
pub struct AbstractModel<X = Standard> {
    /// Economically active persons without the firm and bank owners: employed
    /// and unemployed workers.
    pub w_act: Workers,
    /// Economically inactive persons.
    pub w_inact: Workers,
    /// Firms, each together with its owner (investor) household.
    pub firms: Firms,
    /// The banks, each together with its owner household.
    pub banks: Banks,
    pub cb: CentralBank,
    pub gov: Government,
    pub rotw: RestOfTheWorld,
    pub agg: Aggregates,
    pub prop: Properties,
    /// Time series, one entry per call of [`AbstractModel::collect_data`].
    pub data: Data,
    /// The fields that the model type adds.
    pub ext: X,
}

/// BeforeIT's `Model`: the model of the paper, with every rule as in
/// BeforeIT.jl.
pub type Model = AbstractModel<Standard>;

/// The parts of a model in the order of BeforeIT's constructor
/// `Model((workers_act, workers_inact, firms, banks, central_bank,
/// government, rotw, agg, properties, data))`.
pub type Agents = (
    Workers,
    Workers,
    Firms,
    Banks,
    CentralBank,
    Government,
    RestOfTheWorld,
    Aggregates,
    Properties,
    Data,
);

/// The agents of `Model(parameters, initial_conditions)`, built with their
/// constructors `Workers(p, ic)`, `Firms(p, ic)` and so on; the model
/// extensions start from them as well.
pub(crate) fn agents_of_the_model<R: Randomness + ?Sized>(
    p: &Parameters,
    ic: &InitialConditions,
    rng: &mut R,
) -> Result<Agents, String> {
    let prop = Properties::new(p, ic)?;
    let firms = Firms::new(p, ic, rng)?;
    let (w_act, w_inact) = Workers::new(p, ic);
    Ok((
        w_act,
        w_inact,
        firms,
        Banks::new(p, ic),
        CentralBank::new(p, ic),
        Government::new(p, ic),
        RestOfTheWorld::new(p, ic),
        Aggregates::new(p, ic),
        prop,
        Data::default(),
    ))
}

impl Model {
    /// `Model(parameters, initial_conditions)` (online appendix E): initial
    /// state and the first entry of [`AbstractModel::data`].
    ///
    /// Returns an error if the inputs fail the checks of
    /// [`Properties::new`], if the employees of a sector cannot be drawn
    /// (Julia's `randpl` throws or never terminates), or if the firms need
    /// more employees than there are workers (the Julia code throws).
    pub fn new<R: Randomness + ?Sized>(
        p: &Parameters,
        ic: &InitialConditions,
        rng: &mut R,
    ) -> Result<Model, String> {
        Model::from_agents(agents_of_the_model(p, ic, rng)?, Standard, rng)
    }
}

impl<X: Extension> AbstractModel<X> {
    /// The model constructor of BeforeIT `src/model_init/agents.jl`,
    /// `T((w_act, w_inact, firms, banks, cb, gov, rotw, agg, prop, data))`
    /// for a model type `T`: assigns the firms and persons to the banks,
    /// fills the jobs of the firms, sets up the banks and the deposits of the
    /// households, and collects the first data.
    ///
    /// Returns an error if `banks` does not hold `prop.n_banks` banks (a
    /// check the Julia code does not make), if there is no bank, if the
    /// firms need more employees than there are workers, or if a bank gets
    /// no firms (in the last three cases the Julia code throws).
    pub fn from_agents<R: Randomness + ?Sized>(
        agents: Agents,
        ext: X,
        rng: &mut R,
    ) -> Result<Self, String> {
        let (w_act, w_inact, firms, banks, cb, gov, rotw, agg, prop, data) = agents;
        if banks.len() != prop.n_banks {
            return Err(format!(
                "there are {} banks, but prop.n_banks is {}",
                banks.len(),
                prop.n_banks
            ));
        }
        if banks.is_empty() {
            return Err("there is no bank".into());
        }
        let mut model = AbstractModel {
            w_act,
            w_inact,
            firms,
            banks,
            cb,
            gov,
            rotw,
            agg,
            prop,
            data,
            ext,
        };
        model.assign_banks(rng);
        model.assign_workers()?;
        model.init_banks()?;
        model.update_variables_with_totals();
        model.collect_data();
        Ok(model)
    }

    /// The bank of each firm, worker and inactive person, which they keep:
    /// bank 0 if there is only one, otherwise drawn at random. Part of the
    /// model constructor in BeforeIT `src/model_init/agents.jl`.
    fn assign_banks<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        let n_banks = self.banks.len();
        for B in [
            &mut self.firms.B_i,
            &mut self.w_act.B_h,
            &mut self.w_inact.B_h,
        ] {
            for b in B.iter_mut() {
                *b = if n_banks == 1 {
                    0
                } else {
                    rng.rand_below(n_banks)
                };
            }
        }
    }

    /// Fills the jobs of each firm, in firm order, with the next workers and
    /// pays them the firm's average wage; then the workers' income (A.53) and
    /// their deposits and capital in proportion to income (appendix E.2). Part
    /// of the model constructor in BeforeIT `src/model_init/agents.jl`.
    fn assign_workers(&mut self) -> Result<(), String> {
        let H_W = self.w_act.len();
        let needed: i64 = self.firms.V_i.iter().sum();
        if needed > H_W as i64 {
            return Err(format!(
                "the firms need {needed} employees, but there are only {H_W} workers \
                 (H_act minus the firm and bank owners)"
            ));
        }
        let mut h = 0;
        for i in 0..self.prop.I {
            while self.firms.V_i[i] > 0 {
                self.w_act.O_h[h] = Some(i);
                self.w_act.w_h[h] = self.firms.w_bar_i[i];
                self.firms.V_i[i] -= 1;
                h += 1;
            }
        }
        let P_bar_HH = 1.0;
        let prop = &self.prop;
        for h in 0..H_W {
            let w = self.w_act.w_h[h];
            self.w_act.Y_h[h] = if self.w_act.O_h[h].is_some() {
                (w * (1.0 - prop.tau_SIW - prop.tau_INC * (1.0 - prop.tau_SIW)) + prop.sb_other)
                    * P_bar_HH
            } else {
                (prop.theta_UB * w + prop.sb_other) * P_bar_HH
            };
        }
        for h in 0..H_W {
            self.w_act.D_h[h] = prop.D_H * self.w_act.Y_h[h];
            self.w_act.K_h[h] = prop.K_H * self.w_act.Y_h[h];
        }
        Ok(())
    }

    /// The customers of each bank, its initial profit and reserves, and the
    /// income, deposits and capital of its owner. Part of the model
    /// constructor in BeforeIT `src/model_init/agents.jl`.
    fn init_banks(&mut self) -> Result<(), String> {
        let prop = &self.prop;
        let (f, b) = (&self.firms, &mut self.banks);
        let P_bar_HH = 1.0;
        for k in 0..prop.n_banks {
            b.firms[k] = (0..prop.I).filter(|&i| f.B_i[i] == k).collect();
            b.w_act[k] = (0..self.w_act.D_h.len())
                .filter(|&h| self.w_act.B_h[h] == k)
                .collect();
            b.w_inact[k] = (0..self.w_inact.D_h.len())
                .filter(|&h| self.w_inact.B_h[h] == k)
                .collect();

            // Julia's `sum` over a generator: from left to right, and an
            // exception for a bank without firms.
            let no_firms = || format!("bank {k} has no firms");
            let L_base = b.firms[k]
                .iter()
                .map(|&id| f.L_i[f.index(id)])
                .reduce(|x, y| x + y)
                .ok_or_else(no_firms)?;
            let D_base = b.firms[k]
                .iter()
                .map(|&id| f.D_i[f.index(id)])
                .reduce(|x, y| x + y)
                .ok_or_else(no_firms)?;
            // Each bank's equity is its share of the sector's equity prop.E_k.
            let E_k_base = b.E_k[k];
            b.Pi_k[k] = prop.mu * L_base + prop.r_bar * E_k_base;
            b.D_k[k] = D_base + E_k_base - L_base;
            b.Y_h[k] = prop.theta_DIV
                * (1.0 - prop.tau_INC)
                * (1.0 - prop.tau_FIRM)
                * julia::max(0.0, b.Pi_k[k])
                + prop.sb_other * P_bar_HH;
            b.D_h[k] = prop.D_H * b.Y_h[k];
            b.K_h[k] = prop.K_H * b.Y_h[k];
        }
        Ok(())
    }

    /// `update_variables_with_totals!`: household deposits and capital as
    /// shares of total household income; each bank adds the deposits of its
    /// customers and its owner to its reserves.
    pub fn update_variables_with_totals(&mut self) {
        let tot_Y_h = julia::sum(&self.firms.Y_h)
            + julia::sum(&self.w_act.Y_h)
            + julia::sum(&self.w_inact.Y_h)
            + julia::sum(&self.banks.Y_h);
        for v in [
            &mut self.firms.K_h,
            &mut self.firms.D_h,
            &mut self.w_act.K_h,
            &mut self.w_act.D_h,
            &mut self.w_inact.K_h,
            &mut self.w_inact.D_h,
            &mut self.banks.K_h,
            &mut self.banks.D_h,
        ] {
            for x in v.iter_mut() {
                *x /= tot_Y_h;
            }
        }
        let (f, w_act, w_inact) = (&self.firms, &self.w_act, &self.w_inact);
        let b = &mut self.banks;
        for k in 0..self.prop.n_banks {
            let D_h = agents::gather(&b.firms[k], &f.D_h, |id| f.index(id));
            let D_h_act = agents::gather(&b.w_act[k], &w_act.D_h, |id| w_act.index(id));
            let D_h_inact = agents::gather(&b.w_inact[k], &w_inact.D_h, |id| w_inact.index(id));
            let tot_D_h =
                julia::sum(&D_h) + julia::sum(&D_h_act) + julia::sum(&D_h_inact) + b.D_h[k];
            b.D_k[k] += tot_D_h;
        }
    }

    /// One quarter, `step!(model)`.
    ///
    /// # Panics
    ///
    /// Where the Julia code throws: if `prop.C` was changed to a matrix from
    /// which no shocks can be drawn (see [`base::epsilon`]), if the
    /// desired employment of a firm is not a finite number (see
    /// [`AbstractModel::set_firms_expectations_and_decisions`]), or, in
    /// `ModelCANVAS`, if a regressor of the re-estimated Taylor rule is NaN
    /// (see [`estimate::estimate_taylor_rule`]).
    pub fn step<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        self.step_with(rng, false, &NoShock);
    }

    /// `T` quarters without collecting data, `step!(model, T; parallel,
    /// shock!)`.
    ///
    /// # Panics
    ///
    /// As [`AbstractModel::step`].
    pub fn steps<R, S>(&mut self, T: usize, rng: &mut R, parallel: bool, shock: &S)
    where
        R: Randomness + ?Sized,
        S: Shock<X> + ?Sized,
    {
        for _ in 0..T {
            self.step_with(rng, parallel, shock);
        }
    }

    /// One quarter, `step!(model; parallel, shock!)`: `shock` is applied
    /// right after the central bank has set the policy rate, and with
    /// `parallel` the goods markets of the products run in parallel threads
    /// (see [`base::search_and_matching`]).
    ///
    /// # Panics
    ///
    /// As [`AbstractModel::step`].
    pub fn step_with<R, S>(&mut self, rng: &mut R, parallel: bool, shock: &S)
    where
        R: Randomness + ?Sized,
        S: Shock<X> + ?Sized,
    {
        self.finance_insolvent_firms();

        // General estimations: expectations, euro area, policy rate.
        self.set_growth_inflation_expectations(rng);
        self.set_epsilon(rng);
        self.set_growth_inflation_EA(rng);
        self.set_central_bank_rate();
        shock.apply(self);
        self.set_banks_rate();

        // Firm expectations and decisions.
        self.set_firms_expectations_and_decisions(rng);

        // Credit market, labour market and production.
        self.search_and_matching_credit(rng);
        self.search_and_matching_labour(rng);
        self.set_firms_wages();
        self.set_firms_production();
        self.update_workers_wages();

        // Consumption and investment budgets.
        self.set_gov_social_benefits();
        self.set_banks_expected_profits();
        self.set_households_budget_act();
        self.set_households_budget_inact();
        self.set_households_budget_firms();
        self.set_households_budget_banks();

        // Government spending, imports and exports.
        self.set_gov_expenditure(rng);
        self.set_rotw_import_export(rng);

        // Goods markets.
        self.search_and_matching(rng, parallel);

        // Final accounting.
        self.set_inflation_priceindex();
        self.set_sector_specific_priceindex();
        self.set_capital_formation_priceindex();
        self.set_households_priceindex();
        self.set_firms_stocks();
        self.set_firms_profits();
        self.set_banks_profits();
        self.set_banks_equity();
        self.set_households_income_act(false);
        self.set_households_income_inact(false);
        self.set_households_income_firms(false);
        self.set_households_income_banks(false);
        self.set_households_deposits_act();
        self.set_households_deposits_inact();
        self.set_households_deposits_firms();
        self.set_households_deposits_banks();
        self.set_central_bank_equity();
        self.set_gov_revenues();
        self.set_gov_loans();
        self.set_firms_deposits();
        self.set_firms_loans();
        self.set_firms_equity();
        self.set_rotw_deposits();
        self.set_banks_deposits();
        self.set_gross_domestic_product();
        self.set_time();
    }

    /// `run!(model, T)`: `T` quarters, collecting data after each.
    ///
    /// # Panics
    ///
    /// As [`AbstractModel::step`].
    pub fn run<R: Randomness + ?Sized>(&mut self, T: usize, rng: &mut R) {
        self.run_with(T, rng, false, &NoShock);
    }

    /// `run!(model, T; parallel, shock!)`: `T` quarters as
    /// [`AbstractModel::step_with`], collecting data after each.
    ///
    /// # Panics
    ///
    /// As [`AbstractModel::step`].
    pub fn run_with<R, S>(&mut self, T: usize, rng: &mut R, parallel: bool, shock: &S)
    where
        R: Randomness + ?Sized,
        S: Shock<X> + ?Sized,
    {
        for _ in 0..T {
            self.step_with(rng, parallel, shock);
            self.collect_data();
        }
    }
}

// The Rust examples in README.md are compiled as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
