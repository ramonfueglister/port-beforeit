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
//! the paper. One step is one quarter, and [`Model::step`] calls the rules in
//! the order of BeforeIT `src/one_step.jl`. All random draws go through
//! [`Randomness`]: [`Xoshiro`] for stochastic runs, [`Deterministic`] for the
//! comparison with the authors' reference values.
//!
//! Indices start at 0: sector `g` and firm `i` are Julia's `g + 1` and `i + 1`.
//! Where this port deliberately differs from the Julia code is listed in
//! `README.md`, section "Differences from BeforeIT.jl".

#![forbid(unsafe_code)]
#![allow(non_snake_case, clippy::needless_range_loop)]

pub mod agents;
pub mod data;
pub mod estimate;
pub mod julia;
pub mod params;
pub mod random;

mod aggregates;
mod bank;
mod firms;
mod government;
mod households;
mod markets;
mod rotw;

pub use agents::{Aggregates, Bank, CentralBank, Firms, Government, RestOfTheWorld, Workers};
pub use data::Data;
pub use params::{InitialConditions, Parameters, Properties};
pub use random::{Deterministic, Randomness, WeightVector, Xoshiro};

/// The whole model, BeforeIT `Model`.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// Economically active persons without the firm and bank owners: employed
    /// and unemployed workers.
    pub w_act: Workers,
    /// Economically inactive persons.
    pub w_inact: Workers,
    /// Firms, each together with its owner (investor) household.
    pub firms: Firms,
    /// The bank and its owner household.
    pub bank: Bank,
    pub cb: CentralBank,
    pub gov: Government,
    pub rotw: RestOfTheWorld,
    pub agg: Aggregates,
    pub prop: Properties,
    /// Time series, one entry per call of [`Model::collect_data`].
    pub data: Data,
}

impl Model {
    /// `Model(parameters, initial_conditions)` (online appendix E): initial
    /// state and the first entry of [`Model::data`].
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
        let prop = Properties::new(p, ic)?;
        let firms = Firms::new(p, ic, rng)?;
        let (w_act, w_inact) = Workers::new(p, ic);
        let mut model = Model {
            w_act,
            w_inact,
            firms,
            bank: Bank::new(p, ic),
            cb: CentralBank::new(p, ic),
            gov: Government::new(p, ic),
            rotw: RestOfTheWorld::new(p, ic),
            agg: Aggregates::new(p, ic),
            prop,
            data: Data::default(),
        };
        model.assign_workers()?;
        model.init_bank();
        model.update_variables_with_totals();
        model.collect_data();
        Ok(model)
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

    /// Initial profit and reserves of the bank and income, deposits and
    /// capital of its owner. Part of the model constructor in BeforeIT
    /// `src/model_init/agents.jl`, with one bank.
    fn init_bank(&mut self) {
        let prop = &self.prop;
        let P_bar_HH = 1.0;
        // From left to right, as Julia's `sum` over a generator.
        let L_base: f64 = self.firms.L_i.iter().sum();
        let D_base: f64 = self.firms.D_i.iter().sum();
        let E_k_base = self.bank.E_k;
        self.bank.Pi_k = prop.mu * L_base + prop.r_bar * E_k_base;
        self.bank.D_k = D_base + E_k_base - L_base;
        self.bank.Y_h = prop.theta_DIV
            * (1.0 - prop.tau_INC)
            * (1.0 - prop.tau_FIRM)
            * julia::max(0.0, self.bank.Pi_k)
            + prop.sb_other * P_bar_HH;
        self.bank.D_h = prop.D_H * self.bank.Y_h;
        self.bank.K_h = prop.K_H * self.bank.Y_h;
    }

    /// `update_variables_with_totals!`: household deposits and capital as
    /// shares of total household income; the deposits are added to the
    /// bank's reserves.
    fn update_variables_with_totals(&mut self) {
        let tot_Y_h = julia::sum(&self.firms.Y_h)
            + julia::sum(&self.w_act.Y_h)
            + julia::sum(&self.w_inact.Y_h)
            + self.bank.Y_h;
        for v in [
            &mut self.firms.K_h,
            &mut self.firms.D_h,
            &mut self.w_act.K_h,
            &mut self.w_act.D_h,
            &mut self.w_inact.K_h,
            &mut self.w_inact.D_h,
        ] {
            for x in v.iter_mut() {
                *x /= tot_Y_h;
            }
        }
        self.bank.K_h /= tot_Y_h;
        self.bank.D_h /= tot_Y_h;
        let tot_D_h = julia::sum(&self.firms.D_h)
            + julia::sum(&self.w_act.D_h)
            + julia::sum(&self.w_inact.D_h)
            + self.bank.D_h;
        self.bank.D_k += tot_D_h;
    }

    /// One quarter, `step!(model)`.
    ///
    /// # Panics
    ///
    /// Where the Julia code throws: if `prop.C` was changed to a matrix from
    /// which no shocks can be drawn (see [`Model::set_epsilon`]), or if the
    /// desired employment of a firm is not a finite number (see
    /// [`Model::set_firms_expectations_and_decisions`]).
    pub fn step<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        self.step_with_shock(rng, |_| {});
    }

    /// One quarter with a shock applied right after the policy rate is set,
    /// `step!(model; shock!)`.
    ///
    /// # Panics
    ///
    /// As [`Model::step`].
    pub fn step_with_shock<R, S>(&mut self, rng: &mut R, shock: S)
    where
        R: Randomness + ?Sized,
        S: FnOnce(&mut Model),
    {
        self.finance_insolvent_firms();

        // General estimations: expectations, euro area, policy rate.
        self.set_growth_inflation_expectations(rng);
        self.set_epsilon(rng);
        self.set_growth_inflation_EA(rng);
        self.set_central_bank_rate();
        shock(self);
        self.set_banks_rate();

        // Firm expectations and decisions.
        self.set_firms_expectations_and_decisions();

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
        self.set_rotw_import_export();

        // Goods markets.
        self.search_and_matching(rng);

        // Final accounting.
        self.set_inflation_priceindex();
        self.set_sector_specific_priceindex();
        self.set_capital_formation_priceindex();
        self.set_households_priceindex();
        self.set_firms_stocks();
        self.set_firms_profits();
        self.set_banks_profits();
        self.set_banks_equity();
        self.set_households_income_act();
        self.set_households_income_inact();
        self.set_households_income_firms();
        self.set_households_income_banks();
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
    /// As [`Model::step`].
    pub fn run<R: Randomness + ?Sized>(&mut self, T: usize, rng: &mut R) {
        for _ in 0..T {
            self.step(rng);
            self.collect_data();
        }
    }
}

// The Rust examples in README.md are compiled as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
