# port-beforeit

[![CI](https://github.com/ramonfueglister/port-beforeit/actions/workflows/ci.yml/badge.svg)](https://github.com/ramonfueglister/port-beforeit/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV: 1.88](https://img.shields.io/badge/MSRV-1.88-orange.svg)](Cargo.toml)

An unofficial Rust port of [BeforeIT.jl](https://github.com/bancaditalia/BeforeIT.jl),
the Julia implementation of the agent-based macroeconomic model of Poledna,
Miess, Hommes and Rabitsch (2023),
[*Economic forecasting with an agent-based model*](https://doi.org/10.1016/j.euroecorev.2022.104306),
European Economic Review 151, 104306.

> [!IMPORTANT]
> This is an independent port. It is not affiliated with or endorsed by the
> authors of the paper, the developers of BeforeIT.jl, Banca d'Italia or the
> Bank of England. The reference implementation is BeforeIT.jl; errors in this
> port are its own.

## The model

The model simulates an economy quarter by quarter. Its agents are households
(employed and unemployed workers, inactive persons, and the owners of the
firms and of the banks), firms, banks (one in the paper; BeforeIT.jl allows
several), a central bank, the government and the rest of the world. Every quarter the agents form expectations with AR(1)
models that are estimated anew on the past data, firms set prices and plan
production, and the credit, labour and goods markets match buyers and sellers
by search and matching. After each quarter the model records national
accounts series such as GDP, consumption, investment, exports and imports.

This crate translates the model of BeforeIT.jl at commit
[`0926398`](https://github.com/bancaditalia/BeforeIT.jl/tree/09263989b56bd2d58c28657bb8e940142d995cf4)
(29 October 2026, newer than release v0.6.0) function by function. Functions
and variables keep their Julia names, so every rule can be read side by side
with the original. The documentation of each rule names the Julia function it
translates and the equations it implements, numbered as in the online
appendix of the paper (A.1 to A.84,
[supplementary material](https://doi.org/10.5281/zenodo.7271552)).

## Status

| | |
|---|---|
| Ported | Model setup (`src/model_init/`); adding, deleting and looking up agents by ID (`src/utils/modify.jl`); model types (`@object` and methods per model type) as the trait `Extension`; all agent rules (`src/agent_actions/`) with one or several banks (`n_banks`) and the options `omega_pi` and `overdrafts_in_capital_ratio`; the credit, labour and goods markets (`src/markets/`), also with the goods markets of the products in parallel threads; `step!`, `run!` and `ensemblerun!` with shocks and the option `parallel` (`one_step.jl`, `one_simulation.jl`) and the shock types (`src/shocks/`); data collection with the series per bank and `DataVector` (`src/utils/data.jl`); `get_accounting_identities.jl`; the data sets `AUSTRIA2010Q1`, `ITALY2010Q1` and `STEADY_STATE2010Q1` (`standard_params_initial_conditions.jl`); the model extensions CANVAS (`init_CANVAS.jl`, with the option `theta_calvo`) and growth-rate model (`init_growth_rate_model.jl`); `estimate.jl` with `estimate_taylor_rule`, `nfvar3.jl`, `julia_svd.jl`, `epsilon.jl` and `positive.jl`; the calibration (`calibration.jl`, `_calibration_steady_state.jl`) with its data `ITALY_CALIBRATION` and the calibration objects of the Zenodo record (`standard_calibration_data.jl`); predictions from simulations (`get_predictions_from_sims.jl`, `toannual.jl`); the tests of forecasts (`dmtest.jl`, `mztest.jl`, `bias_ttest.jl`), the VAR benchmarks (`varx.jl`) and the business-cycle statistics (`correlation_utils.jl`); the forecast error tables (`analysis_utils.jl`, `error_tables.jl`) and the batch runs of calibrations, simulations and predictions (`save_all_predictions.jl`); the model as one list of numbers and back (`diff_transform.jl`); the plots of the package extension `ext/BeforeITPlots` (feature `plots`); the parts of `randpl.jl` and `opt.jl` that these use; and the examples of `examples/`, the script `main.jl` and `benchmark/benchmark_scalability.sh` (see [Examples](#examples)) |
| No Rust counterpart | the package extension `ext/BeforeITMooncake`, rules for the automatic differentiation of Mooncake.jl, which the port does not have; `ext/BeforeITFormat`, `format.jl` and `pre_commit.sh`, which format the Julia sources (the port uses `cargo fmt`); `src/precompile.jl`, a precompilation workload (Rust compiles ahead of time); the documentation in `docs/`, which the rustdoc documentation replaces |

## Installation

The crate is not on crates.io. Add it as a Git dependency:

```toml
[dependencies]
beforeit = { git = "https://github.com/ramonfueglister/port-beforeit" }
```

It needs Rust 1.88 or newer. It depends on
[`libm`](https://crates.io/crates/libm) and, to read the data sets,
[`serde_json`](https://crates.io/crates/serde_json) with the feature
`float_roundtrip`, so that every number is read as the nearest `f64`, as
Julia reads it; it contains no `unsafe` code. The plots, BeforeIT.jl's
package extension for Plots.jl, need the feature `plots`, which adds
[`plotters`](https://crates.io/crates/plotters) to draw them as SVG:

```toml
[dependencies]
beforeit = { git = "https://github.com/ramonfueglister/port-beforeit", features = ["plots"] }
```

## Usage

```rust
use beforeit::{AUSTRIA2010Q1, Model, Xoshiro};

/// Real GDP of the initial state and of 20 simulated quarters, Austria
/// from 2010Q1.
fn real_gdp() -> Result<Vec<f64>, String> {
    let s = &*AUSTRIA2010Q1;
    let mut rng = Xoshiro::seed_from_u64(1);
    let mut model = Model::new(&s.parameters, &s.initial_conditions, &mut rng)?;
    model.run(20, &mut rng);
    Ok(model.data.real_gdp)
}
```

`Parameters` and `InitialConditions` are plain structs whose fields have the
names of the keys of BeforeIT.jl's `parameters` and `initial_conditions`;
tables D.2, D.3, E.1 and E.2 of the online appendix explain them. The crate
contains BeforeIT.jl's data sets `AUSTRIA2010Q1`, `ITALY2010Q1` and
`STEADY_STATE2010Q1`, converted to JSON (see [`data`](data/README.md));
`Parameters::from_json` and `InitialConditions::from_json` read other
calibrations with the same keys. The example simulates 20 quarters of
`AUSTRIA2010Q1` like the example in the README of BeforeIT.jl and prints real
GDP:

```sh
cargo run --release --example austria
```

| Rust | BeforeIT.jl |
|---|---|
| `AUSTRIA2010Q1.parameters`, `AUSTRIA2010Q1.initial_conditions` | `Bit.AUSTRIA2010Q1.parameters`, `Bit.AUSTRIA2010Q1.initial_conditions` |
| `Model::new(&p, &ic, &mut rng)` | `Model(parameters, initial_conditions)` |
| `ModelCANVAS::new(&p, &ic, &mut rng)`, `ModelGR::new(&p, &ic, &mut rng)` | `ModelCANVAS(parameters, initial_conditions)`, `ModelGR(parameters, initial_conditions)` |
| `p.n_banks = 3` before `Model::new` | `parameters["n_banks"] = 3` |
| `model.step(&mut rng)` | `step!(model)` |
| `model.step_with(&mut rng, parallel, &shock)` | `step!(model; parallel, shock!)` |
| `model.steps(T, &mut rng, parallel, &shock)` | `step!(model, T; parallel, shock!)` |
| `model.collect_data()` | `collect_data!(model)` |
| `model.run(T, &mut rng)` | `run!(model, T)` |
| `model.run_with(T, &mut rng, parallel, &shock)` | `run!(model, T; parallel, shock!)` |
| `ensemblerun(&mut models, T, &mut rng, parallel, &shock)` | `ensemblerun!(models, T; parallel, shock!)` |
| `InterestRateShock { rate: 0.01, final_time: 4 }` | `InterestRateShock(0.01, 4)` |
| `model.firms.Y_i[i]`, `model.cb.r_bar`, ... | `model.firms.Y_i[i]`, `model.cb.r_bar`, ... |
| `model.data.real_gdp`, ... | `model.data.real_gdp`, ... |
| `let i = model.w_act.index(id); model.w_act.Y_h[i]` | `model.w_act[id].Y_h` |
| `model.w_act.getfields(id)`, `model.w_act.delete(id)`, `model.w_act.push(worker)` | `getfields(model.w_act[id])`, `delete!(model.w_act, id)`, `push!(model.w_act, t)` |
| `DataVector::from_models(&models).series(\|d\| &d.real_gdp)` | `DataVector(models).real_gdp` |
| `get_params_and_initial_conditions(&ITALY_CALIBRATION, Date::new(2010, 3, 31), 0.001, false)?` | `get_params_and_initial_conditions(Bit.ITALY_CALIBRATION, DateTime(2010, 3, 31); scale = 0.001)` |
| `get_valid_calibration_quarters(&co)?` | `get_valid_calibration_quarters(co)` |
| `zenodo_calibration_url("AT")?`, then `data/convert_calibration.py` and `CalibrationData::from_json(&text)?` | `download_zenodo_calibration_object("AT")` |
| `diff_transform::model_to_array(&model)`, `diff_transform::array_to_model(&arr, &model)?` | `Bit.model_to_array(model)`, `Bit.array_to_model(arr, model)` |
| `plots::plot_data(&model, DEFAULT_TITLEFONT, &DEFAULT_QUANTITIES)?` | `Bit.plot_data(model)` |
| `plots::plot_data_vector(&models, DEFAULT_TITLEFONT, &DEFAULT_QUANTITIES, None, None)?` | `Bit.plot_data_vector(models)` |
| `plots::plot_data_vectors(&[models_a, models_b], DEFAULT_TITLEFONT, &DEFAULT_QUANTITIES)?` | `Bit.plot_data_vectors([models_a, models_b])` |
| `plots::plot_model_vs_real(&model, &real, "real_gdp_quarterly", true)?` | `Bit.plot_model_vs_real(model, real, "real_gdp_quarterly")` |
| `plots::layout_svg(&ps, (3, 3), DEFAULT_SIZE)?`, `p.to_svg(DEFAULT_SIZE)?` | `plot(ps..., layout = (3, 3))`, `savefig(p, "p.svg")` |
| `let mut p = plots::Plot::new("gdp"); p.add_line(x, y).label = Some("baseline".into());` | `p = plot(x, y, title = "gdp", label = "baseline")` |
| `p.add_errorline(x, runs, plots::ErrorType::Sem)?;` | `errorline!(p, x, runs, errortype = :sem)` |

`step` calls the rules in the order of BeforeIT `src/one_step.jl`. Each rule
is a public method named after its Julia function without the `!`, for
example `set_firms_production` for `set_firms_production!`, so a quarter can
also be run rule by rule. The state of all agents is public, one list per
variable as in the Julia code: firm `i` is entry `i` of every firm list.
Indices start at 0. Firms and persons also have IDs, as in BeforeIT
`src/utils/modify.jl`: until an agent is deleted, its ID is its index, and
afterwards `index(id)` gives its position.

`ModelCANVAS` and `ModelGR` are BeforeIT.jl's model extensions
(`src/model_extensions/`). In the CANVAS model (Hommes et al., 2025), firms
set prices and quantities by demand pull, each firm applies its demand-pull
and cost-push price changes in a quarter only with probability
`theta_calvo` (all prices still move with expected inflation), and the
central bank re-estimates the Taylor
rule every quarter; its histories are in `model.ext` (BeforeIT.jl keeps
them in `model.cb` and `model.rotw`). As in BeforeIT.jl, this estimate
receives euro area GDP in levels where `estimate_taylor_rule` expects
inflation, and inflation where it expects growth. The growth-rate model
replaces the AR(1) processes on logarithms by AR(1) processes on growth
rates for the expectations, government consumption, exports and imports.

A shock changes the model in every quarter right after the central bank has
set the policy rate, where `step!` calls `shock!`. BeforeIT.jl's shock types
`NoShock`, `InterestRateShock`, `ProductivityShock` and `ConsumptionShock`
implement the trait `Shock`, and so does every closure that implements
`Fn(&mut AbstractModel<X>)`, for example every closure written
`|m: &mut Model| { ... }` that returns `()` and neither moves out of the
variables it captures nor needs `&mut` access to them:

```rust
use beforeit::{AUSTRIA2010Q1, InterestRateShock, Model, Xoshiro, ensemblerun};

/// Ten runs of 12 quarters with the policy rate at 1% while the model time
/// `agg.t` is at most 4, in parallel threads.
fn rate_scenario() -> Result<Vec<Model>, String> {
    let s = &*AUSTRIA2010Q1;
    let mut rng = Xoshiro::seed_from_u64(1);
    let model = Model::new(&s.parameters, &s.initial_conditions, &mut rng)?;
    let mut models = vec![model; 10];
    let shock = InterestRateShock { rate: 0.01, final_time: 4 };
    ensemblerun(&mut models, 12, &mut rng, true, &shock);
    Ok(models)
}

/// The same shock written as a closure, for one quarter of one model.
fn one_quarter(model: &mut Model, rng: &mut Xoshiro) {
    model.step_with(rng, false, &|m: &mut Model| {
        if m.agg.t <= 4 {
            m.cb.r_bar = 0.01;
        }
    });
}
```

With `parallel`, the goods markets of the products run in parallel threads,
and `ensemblerun` runs the models in parallel threads. Each product's market
and each model of the ensemble draws from its own generator, split off from
the given one in a fixed order (`Randomness::split`), so a parallel run is
reproducible and does not depend on the number of threads; it differs from
the serial run with the same seed, as in BeforeIT.jl.

The calibration computes parameters and initial conditions from national
accounts, input-output tables and time series, as BeforeIT
`src/utils/calibration.jl`:

```rust
use beforeit::{Date, ITALY_CALIBRATION, Model, Xoshiro, get_params_and_initial_conditions};

/// A model of Italy calibrated on 2012Q3, with 0.001 agents per person or
/// firm of the data.
fn italy_2012q3() -> Result<Model, String> {
    let c = get_params_and_initial_conditions(&ITALY_CALIBRATION, Date::new(2012, 9, 30), 0.001, false)?;
    for w in &c.warnings {
        eprintln!("{w}");
    }
    let mut rng = Xoshiro::seed_from_u64(1);
    Model::new(&c.parameters, &c.initial_conditions, &mut rng)
}
```

`ITALY_CALIBRATION` holds the calibration data of BeforeIT.jl for Italy, for
the quarters 2010Q1 to 2016Q4 (`get_valid_calibration_quarters`); the last
argument is BeforeIT's option `use_growth_rate_ar1`. The messages that
BeforeIT.jl logs with `@warn` are returned in `warnings`. For the 27 member
states of the EU, BeforeIT.jl downloads calibration objects (JLD2 files) from
the Zenodo record [*EuroStat Tables for BeforeIT Calibration*](https://doi.org/10.5281/zenodo.18698434)
(Reiter, Hoekstra, Meligrana and Glielmo, 2026, CC BY 4.0).
The port reads neither HTTP nor JLD2: `zenodo_calibration_url` gives the
address of a country's file, [`data/convert_calibration.py`](data/convert_calibration.py)
converts it to JSON (with `h5py`), and `CalibrationData::from_json` reads the
result. `beforeit::calibration::get_params_and_initial_conditions_steady_state`
is the variant of `_calibration_steady_state.jl` with interest, the government
deficit and the AR(1) and Taylor rule coefficients set for a steady state;
its result for 2010Q1 is `STEADY_STATE2010Q1` up to the rounding described
below.

`get_predictions_from_sims` turns the runs of an ensemble into forecasts in
the units of the data, as BeforeIT `src/utils/get_predictions_from_sims.jl`:
the simulated growth rates applied to the last observed values, quarterly
and as calendar years, with growth rates and deflators:

```rust
use beforeit::{
    DataVector, Date, ITALY_CALIBRATION, Model, NoShock, Predictions, Xoshiro, ensemblerun,
    get_params_and_initial_conditions, get_predictions_from_sims,
};

/// Forecasts of 100 runs of 12 quarters from 2010Q1.
fn forecast() -> Result<Predictions, String> {
    let start = Date::new(2010, 3, 31);
    let c = get_params_and_initial_conditions(&ITALY_CALIBRATION, start, 0.001, false)?;
    let mut rng = Xoshiro::seed_from_u64(1);
    let model = Model::new(&c.parameters, &c.initial_conditions, &mut rng)?;
    let mut models = vec![model; 100];
    ensemblerun(&mut models, 12, &mut rng, true, &NoShock);
    let mut data = ITALY_CALIBRATION.clone();
    data.set_gdp_deflators()?;
    let p = get_predictions_from_sims(&DataVector::from_models(&models), &data.data, start)?;
    // Real GDP in 2011, one value per run.
    println!("{:?}", p.series["real_gdp"][1]);
    Ok(p)
}
```

Row 0 of each series is the data at the start date (quarterly) or in its
year (yearly). `set_gdp_deflators` stores the GDP deflators in the data as
BeforeIT's calibration does in the dictionaries it is given; the
predictions read them.

The module `evaluation` has the Diebold-Mariano test `dmtest_modified`, the
Mincer-Zarnowitz test `mztest` and `bias_ttest`; `varx` has the VAR and VARX
benchmarks (`estimate_VAR`, `forecast_k_steps_VAR`, ...); `correlation` has
`hpfilter`, `cross_correlation`, `auto_correlation` and `correlation_stats`
for data or for `Predictions::to_table()`. Where BeforeIT.jl reads and
writes JLD2 files, the port uses JSON (`calibration::table_to_json` and
`table_from_json`).

The module `analysis` has the batch runs and the forecast error tables of
BeforeIT's example `examples/analysis/tabs/simulate_and_create_tables.jl`:

```rust
use beforeit::analysis::{
    FORECAST_HORIZONS, error_table_abm, error_table_ar, save_all_params_and_initial_conditions,
    save_all_predictions_from_sims, save_all_simulations,
};
use beforeit::{Date, ITALY_CALIBRATION, Model, Xoshiro};

/// Forecasts from 2010Q1 to 2011Q4 and their error tables, in the folder
/// `data/it` of the working directory, which is deleted first if it exists.
fn tables() -> Result<(), String> {
    let mut cal = ITALY_CALIBRATION.clone();
    cal.set_gdp_deflators()?;
    let (first, last) = (Date::new(2010, 3, 31), Date::new(2011, 12, 31));
    save_all_params_and_initial_conditions(&cal, "data/it", 0.0005, first, last)?;
    let mut rng = Xoshiro::seed_from_u64(1);
    let sims = Some("simulations/Model");
    save_all_simulations("data/it", 12, 4, Model::new, sims, &mut rng)?;
    let predictions = "abm_predictions/Model";
    save_all_predictions_from_sims("data/it", &cal.data, "simulations/Model", predictions)?;
    let quarters = Date::every_months(first, last, 3);
    let (data, ea) = (&cal.data, &cal.ea);
    error_table_ar("it", ea, data, &quarters, &FORECAST_HORIZONS, "Model")?;
    error_table_abm("it", ea, data, &quarters, &FORECAST_HORIZONS, "Model", predictions)
}
```

The tables are in `data/it/analysis/Model/`, as CSV and LaTeX files with
the names of BeforeIT.jl: the root mean squared errors of the ABM in percent
(`rmse_abm`), their improvement over the AR benchmarks with Diebold-Mariano
tests (`rmse_abm_vs_ar`), the bias with t-tests (`bias_abm`), and the same
for the benchmarks. `error_table_validation_var` and
`error_table_validation_abm` add the tables of the validation variables;
`load_country_forecast_data` and `build_12q_matrices` gather the tables of
several countries. Parameters, simulations, predictions and forecasts are
stored as JSON files, with the names of BeforeIT's JLD2 files.

The module `diff_transform` writes the numbers of a model into one list
(`model_to_array`) and makes a model from such a list (`array_to_model`),
in the order of BeforeIT.jl: the agent types and then the properties, each
with the fields of its Julia struct in their order, integers as floats and
indices counted from 1 as in Julia. BeforeIT's examples
`examples/automatic_differentiation_*.jl` compute gradients of a simulation
with respect to these numbers, with Mooncake.jl (reverse mode) and with
central finite differences; the port has no automatic differentiation, and
its versions of the two examples compute the finite differences only.

With the feature `plots`, the module `plots` has the functions of BeforeIT's
package extension for Plots.jl: `plot_data` draws the series of one model,
`plot_data_vector` the mean over several runs with a band of one standard
deviation (StatsPlots' `errorline`), `plot_data_vectors` several such
ensembles in one plot (`plot_data_vectors_from` takes their data, so that
they may come from models of different types), and `plot_model_vs_real`
predictions against the data. They return `Plot` values with the series, colours, titles, limits
and ticks that BeforeIT.jl hands to Plots.jl; `layout_svg` and
`Plot::to_svg` draw them as SVG. For the plots of BeforeIT's examples,
`Plot::new` starts an empty plot with Plots.jl's defaults, to which
`add_line`, `add_errorline` (standard deviation or standard error),
`add_hline` and `annotate` add elements; its fields set the axis labels,
limits, ticks (`x_rotation` rotates the labels of the x axis, `y_formatter`
writes those of the y axis), logarithmic axes, minor grid lines and the
legend (`legend`, `legend_frame`). `Plot::heatmap` and `Plot::bars`
make heatmaps and bar plots, and `layout_svg_with_title` puts a title over
a layout.

`cargo doc --open` shows the documentation of every type and rule.

## Examples

The folder `examples/` has the examples of BeforeIT.jl's `examples/`, one
Rust file for each Julia file, with its name in lower case; those of
`examples/analysis/` keep their folders, and the two files
`aggregate_forecast_results.jl` are the examples
`tabs_aggregate_forecast_results` and `figs_aggregate_forecast_results`.
`main` is the script `main.jl` at the top of BeforeIT.jl, and
`benchmark_scalability` the measurement of its
`benchmark/benchmark_scalability.sh`. The examples with plots need the
feature `plots` and write SVG files instead of PNG or PDF files, to
`target/example-output/<example>` or, for those that work in folders of
the working directory as in Julia, to `target/example-output`:

```sh
cargo run --release --features plots --example basic_example
```

| Rust | BeforeIT.jl | Notes |
|---|---|---|
| `austria` | the example in the README | prints real GDP |
| `basic_example` | `basic_example.jl` | the field names that Julia lists with `fieldnames` are printed from a fixed list |
| `basic_example_multibank` | `basic_example_multibank.jl` | |
| `basic_inheritance` | `basic_inheritance.jl` | each model has its own agents (see below) |
| `change_expectations` | `change_expectations.jl` | the second model is of a model type with the new rule (see below) |
| `scenario_analysis_via_overload` | `scenario_analysis_via_overload.jl` | the shocked runs are of a model type with the new rule (see below) |
| `scenario_analysis_via_shock` | `scenario_analysis_via_shock.jl` | |
| `single_agents` | `single_agents.jl` | IDs count from 0; a consumer loan holds the ID of its debtor, where Julia holds the agent |
| `multithreading_speedup` | `multithreading_speedup.jl` | no first run to compile, which Rust does ahead of time |
| `canvas_extension` | `CANVAS_extension.jl` | each model has its own agents (see below) |
| `conditional_forecasts` | `Conditional_forecasts.jl` | each model has its own agents (see below) |
| `main` | `main.jl` at the top of BeforeIT.jl | |
| `prediction_pipeline` | `prediction_pipeline.jl` | the field names that Julia lists with `fieldnames` are printed from a fixed list |
| `prediction_pipeline_multiple` | `prediction_pipeline_multiple.jl` | works in `target/example-output` (see below) |
| `compare_model_vs_real` | `compare_model_vs_real.jl` | reads a file of predictions that BeforeIT.jl does not contain (see below) |
| `benchmark_w_matlab` | `benchmark_w_matlab.jl` | measures the port at scale 1:1000 in `f64`; the other bars are the numbers of the Julia example |
| `simulate_and_create_tables` | `analysis/tabs/simulate_and_create_tables.jl` | needs a calibration object of the Zenodo record (see below) |
| `simulate_and_create_tables_multicountry` | `analysis/tabs/simulate_and_create_tables_multicountry.jl` | needs calibration objects of the Zenodo record (see below) |
| `tabs_aggregate_forecast_results` | `analysis/tabs/aggregate_forecast_results.jl` | reads the tables of `simulate_and_create_tables` and `simulate_and_create_tables_multicountry` (see below) |
| `figs_aggregate_forecast_results` | `analysis/figs/aggregate_forecast_results.jl` | reads the tables of `simulate_and_create_tables` and `simulate_and_create_tables_multicountry` (see below) |
| `crosscorrelations` | `analysis/figs/crosscorrelations.jl` | needs a calibration object of the Zenodo record (see below) |
| `automatic_differentiation_perf` | `automatic_differentiation_perf.jl` | plots the times saved by BeforeIT.jl; `--run` times the finite differences of the port in `f64` (see below) |
| `automatic_differentiation_opt` | `automatic_differentiation_opt.jl` | plots the results saved by BeforeIT.jl; `--run` reruns the finite differences of the port in `f64` (see below) |
| `benchmark_scalability` | `benchmark/benchmark_scalability.sh` | started by the port's `benchmark/benchmark_scalability.sh`, which runs 1 to 6 threads, at most the number of logical CPUs (`nproc`), by limiting the logical CPUs with `taskset` (Linux); the ensemble starts up to the square of that number of threads (see the example) |

A model type of BeforeIT.jl (`Bit.@object struct NewModel(Bit.Model)`) with
its own methods is a type that implements `Extension`, and the fields that
its agent types add are fields of that type. Where a Julia example redefines
a method for `Bit.Model`, which changes the rule for every model from then
on, the Rust example uses a model type with the new rule for the models
that follow it.

`basic_inheritance.jl`, `CANVAS_extension.jl` and
`Conditional_forecasts.jl` build several models from the same agent
objects. Each constructor then works on the state the previous one left: it
collects the first data point again and divides the deposits and capital
of the firm owners and the inactive households by the total income once
more. In the last two examples the models also share one state while they
run. In the Rust examples each model has its own copy of the agents.

The Julia examples draw their random numbers from Julia's global
generator: `change_expectations.jl` with the seed 1234, the two
`automatic_differentiation_*.jl` with the seeds 42, 43, … and 10 001 to
10 010, the others without a seed. The Rust examples seed `Xoshiro` with
1, or with the seeds of Julia where it has them (a new generator for each
`Random.seed!`), so each run gives the same numbers; they are not those of
a Julia run, whose generator differs.

The examples `prediction_pipeline_multiple`, `compare_model_vs_real` and
those of `analysis/` read and write folders of the working directory, as
in Julia, such as `data/italy` or `data/AT`; the port makes
`target/example-output` the working directory and writes JSON files
instead of JLD2 files. Like the Julia examples,
`prediction_pipeline_multiple`, `simulate_and_create_tables` and
`crosscorrelations` **delete the folder `data/<country>` there first** if
it exists, and `simulate_and_create_tables_multicountry` deletes it if it
has no folder `parameters`. `compare_model_vs_real.jl` reads the
predictions from 2015Q1 in `data/italy/abm_predictions/2015Q1.jld2`, a
file that BeforeIT.jl does not contain; the Rust example reads the JSON
file of the same name, or the file given as argument, such as one that
`prediction_pipeline_multiple` writes for 2014Q1 to 2014Q4.

`simulate_and_create_tables`, `simulate_and_create_tables_multicountry`
and `crosscorrelations` download in Julia the calibration objects of
Austria, Italy and Spain from the Zenodo record
(`download_zenodo_calibration_object`). The port reads neither HTTP nor
JLD2: download the files `<country>_calibration_object.jld2` of the record,
convert them with `data/convert_calibration.py` (see `data/README.md`) to
`<country>.json` and set `BEFOREIT_CALIBRATION_DIR` to their folder (a
relative folder is taken from the directory where the example is started):

```sh
curl -LO https://zenodo.org/records/18698434/files/AT_calibration_object.jld2
python3 data/convert_calibration.py zenodo AT_calibration_object.jld2 AT.json
BEFOREIT_CALIBRATION_DIR=. cargo run --release --example simulate_and_create_tables
```

BeforeIT's calibration stores the GDP deflators in the calibration object
it is given, and the later steps of these examples read them; the Rust
examples call `set_gdp_deflators` where Julia's later steps see the changed
object. With their settings, the Julia analysis examples do not fit
together: the two `simulate_and_create_tables*.jl` write the tables of
`ModelCANVAS`, the two `aggregate_forecast_results.jl` read those of
`Model`, and the error tables count a forecast only if its quarter lies
within the year of the last quarter of the list, so with the quarters of
`simulate_and_create_tables_multicountry.jl` (2010Q1 to 2011Q1) they have
no value at 12 quarters, the horizon that `aggregate_forecast_results.jl`
gathers. The Rust examples keep these settings. With a later end, such as
the commented 2019Q4 of `simulate_and_create_tables.jl`, and the variant
`"ModelCANVAS"` in the aggregating examples, the chain runs through.
With a list of quarters that leaves at least two forecasts at a horizon
but fewer than the horizon has quarters, such as 2010Q1 to 2013Q1 in
`simulate_and_create_tables_multicountry.jl` (four forecasts at 12
quarters), the Diebold-Mariano test throws in Julia, and the port returns
an error; with no forecast or one, the p-value is NaN.

`automatic_differentiation_perf` and `automatic_differentiation_opt` plot,
as in Julia, the results that BeforeIT.jl saved in
`examples/data/autodiff_performance.csv` and
`examples/data/autodiff_optimization.csv` (copied unchanged): times and
results of the Mooncake gradient in Float64 and of finite differences in
Float64 and Float128. The port has no automatic differentiation and
computes only in `f64`: `--run` (or `--run float64`) reruns the finite
differences in `f64` with the port, where Julia's `--run` reruns all three,
and `auto` and `float128` return an error. Julia replaces the rows of the
rerun in its saved file; the port writes the saved results with the rows
of the rerun replaced to `target/example-output/<example>/` and plots this
copy if it exists, so that the plot then shows the port's finite
differences next to Julia's other results.

## Randomness and reproducibility

The model draws all random numbers through the trait `Randomness`. The crate
has two implementations:

- `Xoshiro`, the generator xoshiro256\*\* seeded with SplitMix64. The same
  seed gives the same run. `Model` implements `Clone` and `PartialEq`, and
  `Xoshiro::state` and `Xoshiro::from_state` save and restore the generator,
  so a run can be continued or branched exactly.
- `Deterministic`, the replacements of BeforeIT
  `test/deterministic/make_model_deterministic.jl`: no randomness, for the
  comparison with the reference values of the authors' MATLAB code.

Another generator can be used by implementing the three methods `randn`,
`rand` and `rand_below`. Outside the plots, the library takes
mathematical functions such as `exp` and `log` from the `libm` crate, not
from the system's math library. The plots (feature `plots`), including the
code of `plotters` that they call, and some examples use the standard
library's functions, which call the system's math library: for example
`log10` on logarithmic axes, `sin` and `cos` for rotated labels, and `ln`
and `exp` in the example `canvas_extension`.

## Differences from BeforeIT.jl

1. **Minimum per firm in desired investment and employment (A.14, A.22).**
   BeforeIT `src/agent_actions/firms.jl` computes
   `min(Q_s_i, firms.K_i .* firms.kappa_i)` without the dot in lines 18 and
   24. Julia's `min` of two vectors compares them as a whole, in
   lexicographic order, and returns one of them, so either every firm uses
   its expected sales or every firm uses its capacity, depending on the first
   firm in which the two differ. The paper and the authors' MATLAB code
   (`test/matlab_code/detabm.m`, lines 116 and 126) take the minimum firm by
   firm, as BeforeIT.jl does itself for intermediate inputs (`min.` in line
   21, A.18). The port takes the minimum firm by firm in all three places.
   Its forecasts therefore differ from the reference values of BeforeIT's
   prediction test by up to 3.2e-5 of a value (see Verification).
2. **Reserves in `get_accounting_identity_banks`.** BeforeIT
   `src/utils/get_accounting_identities.jl` adds the vector of the banks'
   reserves `model.banks.D_k` to a number, which throws in Julia. The port
   adds their sum, as BeforeIT's own test `test/accounting_identities.jl`
   does.
3. **Random numbers.** The distributions are those of the Julia code, the
   numbers are not: Julia's default generator is Xoshiro256++ and draws
   normal numbers with the ziggurat method, the port uses xoshiro256\*\*,
   Box–Muller and Lemire's method for integers. Sellers are drawn from
   cumulative sums of their weights instead of with WeightVectors.jl, and
   `randpl` shuffles where Julia calls `randperm`. Where Julia removes
   entries from a list with `ufilter!`, which moves the last entry into the
   gap, the port keeps the order, like `filter!` in BeforeIT's deterministic
   test. These lists are either shuffled before every pass or only map the
   draws of a weighted sampler to the sellers, so the distribution of the
   outcomes is the same. Runs with randomness therefore agree with
   BeforeIT.jl in distribution, not number by number.
4. **Floating point.** The port computes in `f64`, BeforeIT.jl's default
   `typeFloat`. Sums over lists use the order of Julia's `sum`: lists longer
   than 1024 numbers are split in halves until blocks of at most 1024
   numbers remain, and each block is added from left to right. Julia may
   still reorder additions inside a block (`@simd`), and it calls BLAS for
   dot products and norms, which the port adds from left to right, and
   LAPACK for the singular value decomposition in `rfvar3` when the
   regressors have one column or more than two columns and at least one
   row, which the port computes with one-sided Jacobi rotations. Results
   can therefore differ from BeforeIT.jl in the last digits. The port has
   not been run side by side with Julia; it is checked against the reference
   values that the tests of BeforeIT.jl use (see [Verification](#verification)).
5. **Errors.** `Model::new` checks its inputs and returns an error message
   where the Julia code would throw or `randpl` would not terminate, for
   example for lists of the wrong length, a matrix `C` that is not positive
   definite, more jobs than workers, or a bank without firms. `Model::step` panics where the Julia
   code throws during a step; its documentation lists the cases.
6. **Parallel runs.** In BeforeIT.jl, each task spawned for a product or a
   model draws from its own generator forked from the task-local one, and
   the parallel goods markets add the households' consumption and
   investment under a lock in the order in which the tasks finish, so the
   last digits can change from run to run. The port splits one generator
   per product or model off the given one with the jump function of
   xoshiro256\*\* (2^128 draws ahead), runs them on system threads and adds
   the products' results in the order of the products, as the serial
   version does. Without randomness, parallel and serial runs are
   therefore identical. Julia's tasks share the threads Julia was started
   with; the port starts as many system threads at once as
   `std::thread::available_parallelism` gives, for the products of one
   model and for the models of `ensemblerun`, whose models each run their
   goods markets in parallel, so that an ensemble can have the square of
   that number of product threads at once, besides one thread per group of
   models.
7. **Calibration.** Where BeforeIT.jl would carry a `missing` value of
   the data into a parameter, or would throw for a date that is not in the
   data, an estimation date too close to the calibration date, the
   logarithm of a negative number or a NaN regressor of the Taylor rule,
   the port returns an error message, also where nominal and real GDP
   differ in size other than by trailing dimensions of length 1 (Julia
   throws or broadcasts one over the other). Series by industry whose
   number of rows differs from that of the input-output table, for which
   BeforeIT.jl throws a `DimensionMismatch`, are not checked: the port then
   panics or returns a result. BeforeIT.jl stores
   the GDP deflators in the caller's dictionaries `data` and `ea`; the port
   does not change the calibration data. Dates are calendar days: Julia's
   `num2date` also keeps a time of day, which the calibration never uses.
8. **Predictions and evaluation.** `get_predictions_from_sims` returns an
   error where BeforeIT.jl would throw and also where a value of the data
   at the start date is `missing` or not unique, which Julia would carry
   into the forecasts as `missing` or broadcast over several values. The
   result is a struct with the series as `time × runs` matrices given by
   their rows (`Predictions`) instead of a dictionary. The tests, VAR
   estimates and forecasts and correlations return errors where Julia
   throws (for the VAR, for example, data with too few rows for the lags,
   or a NaN among regressors that LAPACK's `dgesdd` rejects in Julia); the
   stochastic VAR forecasts draw from a given generator instead of Julia's
   global one. Files are JSON instead of JLD2. The batch runs and error
   tables print what BeforeIT.jl logs with `@warn` to standard error and
   return errors where Julia throws, for example for the logarithm of a
   negative number, for which the transformations of the variables
   (`VarSpec`) return errors; `save_all_simulations` skips a quarter
   whose model returns an error or panics, as Julia catches the exception.
   The AR and VAR benchmarks of the error tables accept a `missing` value
   of the data in rows that their estimation does not use, as Julia does
   (its broadcast narrows the type of the differences), and return an
   error for one in the rows used, where Julia's `estimate_VAR` throws.
   The CSV reader of `load_country_forecast_data` knows numbers and text
   only (CSV.jl also recognises dates and booleans, which the tables do not
   contain), and returns an error for a line with too few or too many
   cells, where CSV.jl warns and fills or widens the table.
9. **The model as a list of numbers.** `array_to_model` returns an error
   where Julia throws, and also for an integer that Julia would store but
   the port's types cannot hold: a negative count, a sector or bank below
   1, or an employer `O_h` below 1 other than 0 (unemployed, `w_act`) and −1
   (inactive, `w_inact`). Julia finds the fields by reflection, so its
   functions also take the fields that a model type adds to an agent type;
   in the port, a model type provides them through
   `Extension::push_fields` and `Extension::set_fields`, as `ModelCANVAS`
   and `ModelGR` do.
10. **Plots.** The plot functions return the content of the plots (series,
    bands, colours of Plots.jl's default palette, titles, font sizes,
    limits, ticks, legend) instead of Plots.jl objects, and `plotters`
    draws them as SVG. Automatic axis limits are widened by 6 % and the
    legend goes to the least crowded corner, as in Plots.jl; fonts,
    margins, axes and dash patterns are those of `plotters`, and so is the
    format of the tick numbers, except that the port writes magnitudes from
    10^5 on and nonzero magnitudes below 10^-4 with a power of ten (such as
    `3.95e11`); rotated
    tick labels are SVG text next to the drawing of
    `plotters`, a line that leaves the x limits ends at its last point
    inside them, and heatmaps have no colour bar. The functions return an error where Julia throws, also for a
    `missing` value of the data within the period of the predictions in
    `plot_model_vs_real`.

## Where BeforeIT.jl differs from the paper

In these places the code of BeforeIT.jl differs from the online appendix, and
the port follows the code. They are the ones we found; the list may not be
complete.

| | Online appendix | BeforeIT.jl and the port |
|---|---|---|
| Price index (A.12) | Weighted by the sales of the domestic firms and of the foreign producers | Weighted by the production of the domestic firms, as in the authors' MATLAB code (`detabm.m`, lines 301 and 302) |
| Policy rate (A.69) | Taylor rule without a lower bound | Not below 0 (`pos` in `taylor_rule`); the MATLAB code has no lower bound (`detabm.m`, line 98) |
| Choice of a seller (A.1.1) | The size of a firm is its production | The size of a firm is its production plus its inventories, as in the MATLAB code (`search_and_matching_det.m`, line 10) |

The calibration code of BeforeIT.jl has these properties, which the port
keeps:

- `get_params_and_initial_conditions_steady_state` finds the year of the
  calibration with `findall(...)[1][1]`, the row of the first match. In
  `ITALY_CALIBRATION` the years are a row, so it always uses the first year,
  2010, whatever the date.
- `_calibration_steady_state.jl` rounds the numbers of agents with
  `matlab_round` (ties up). `STEADY_STATE2010Q1` was made with Julia's
  `round` (ties to even): three sectors have one employee more in the port's
  result, and `J`, `H_act` and the values divided by them differ
  accordingly (see `tests/calibration.rs`).
- `ITALY2010Q1` and the MATLAB values in
  `test/matlab_code/italy_calibration` were made with earlier versions of
  the calibration; the current `calibration.jl` gives other values for
  2010Q1 (for example `tau_Y_s = 0`, since it sets `taxes_products` to
  zero). BeforeIT's `test/calibration_script.jl` no longer compares them: the
  comparison with MATLAB is commented out, and the remaining check compares
  each initial condition with itself.
- With `use_growth_rate_ar1`, the calibration returns `use_growth_rate_ar1`
  and the growth rates `g_G`, `g_E` and `g_I`, which no model of BeforeIT.jl
  reads; `ModelGR` estimates its growth-rate processes itself.

The error tables of BeforeIT.jl have these properties, which the port also
keeps:

- They evaluate forecasts only up to the end of the year of the last
  quarter in `quarters`. With the quarters of BeforeIT's example, 2010Q1 to
  2010Q4, the rows for 4, 8 and 12 quarters are therefore NaN.
- `error_table_ar` estimates its AR models on the differences of the
  logarithm of the GDP deflator, so it forecasts the deflator's growth,
  which the tables compare with `log(1 + g)` of the growth `g` in the data.
  `error_table_validation_var` estimates its VAR on the differences of
  `log(1 + g)` itself, for the country and for the euro area, and compares
  the forecast of that difference with `log(1 + g)`.

## Verification

`cargo test --release` runs these tests:

- [`tests/matlab_reference.rs`](tests/matlab_reference.rs), as BeforeIT.jl's
  `test/deterministic`: without randomness, the initial state, every rule of
  one quarter (with serial and with parallel goods markets, as there), and
  the aggregates after quarters 1 and 5 against the values of the authors'
  MATLAB code, with the tolerances of BeforeIT.jl's tests. A
  stricter test requires every compared quantity of the initial state and
  of the aggregates after quarters 1 and 5 to lie within 1e-10 (relative)
  of the MATLAB values; the largest deviation is 8.0e-13, for the
  series of nominal capital formation over five quarters.
- [`tests/accounting_identities.rs`](tests/accounting_identities.rs), as
  `test/accounting_identities.jl`: in runs of three quarters with three
  seeds, with one bank and with three banks, income equals production and
  nominal and real GDP equal the sum of their uses, summed over the initial
  state and the three quarters within the authors' tolerance of 1e-8 (with
  three banks the GDP identities skip the initial state, as there); the
  balance sheets of the central bank and of each bank add up within 1e-8;
  and the loans per bank in the data equal the loans of each bank's firms.
  Splitting the banking sector into one, two or five banks leaves total
  equity, initial profits and reserves unchanged within 1e-10 (relative).
  With one bank, each of the three identities also holds in every single
  quarter within 1e-13 of nominal GDP, and `get_accounting_identities` and
  `get_accounting_identity_banks` return the same zeros.
- [`tests/stochastic.rs`](tests/stochastic.rs): after one goods market, ten
  quantities against the means and standard deviations of the original
  implementation given in `test/markets/search_and_matching.jl`: one run
  within three standard deviations, as there, and 50 runs whose mean lies
  within three standard errors and whose standard deviation lies between 0.6
  and 1.4 times the original's, each with serial and with parallel goods
  markets. Also: the same seed gives the same run, and a saved state
  continues exactly like the original run.
- [`tests/parallel.rs`](tests/parallel.rs): without randomness, parallel
  goods markets give the same run as serial ones (as
  `test/deterministic/one_run_deterministic.jl`, which allows 1e-4); parallel
  runs are reproducible and keep the accounting identities; `ensemblerun`
  runs the models one after the other without `parallel`, and with
  `parallel` gives each model the generator split off for it, also with
  more models than threads; a parallel ensemble of `ModelCANVAS` runs too,
  with different results per model.
- [`tests/shocks.rs`](tests/shocks.rs), as `test/shocks/shocks.jl`: the
  shock types, and that `run_with` applies a shock in every quarter before
  the data are collected.
- [`tests/agent_actions.rs`](tests/agent_actions.rs) and
  [`tests/estimate.rs`](tests/estimate.rs): single rules and the AR(1)
  estimate with the numbers of `test/agent_actions` and
  `test/utils/nfvar3_and_estimate.jl`, and `bank_exposure` and the option
  `omega_pi`, for which BeforeIT.jl has no test.
- [`tests/utils.rs`](tests/utils.rs): adding and deleting agents and `pos`
  and `neg` as in `test/utils/modify.jl` and `test/utils/positive.jl`; that
  every ID still finds its agent after several deletions; and `DataVector`.
- [`tests/model_extensions.rs`](tests/model_extensions.rs), as
  `test/model_extensions.jl`: `ModelGR` with `AUSTRIA2010Q1` and
  `ModelCANVAS` with `ITALY2010Q1` run; also, in both the accounting
  identities hold, the CANVAS central bank stores the re-estimated Taylor
  rule and extends its histories every quarter, `theta_calvo = 0` leaves
  only expected inflation in the prices, and without shocks the growth
  rates of the growth-rate model follow their AR(1) processes exactly.
- [`tests/steady_state.rs`](tests/steady_state.rs), as
  `test/steady_state.jl`: with `STEADY_STATE2010Q1`, every national
  accounts series stays constant over five quarters within the tolerances
  of the Julia test.
- [`tests/calibration.rs`](tests/calibration.rs): the calibration of
  `ITALY_CALIBRATION` for 2010Q1, and for 2019Q4 with `use_growth_rate_ar1`
  (the last quarter of the time series, so that the last 12 entries of the
  histories `C_G`, `C_E` and `Y_I` are NaN; it lies after 2016Q4, the last
  quarter that
  `get_valid_calibration_quarters` returns), against an
  independent NumPy translation of `calibration.jl`
  ([`tests/data/calibration_reference.py`](tests/data/calibration_reference.py)):
  the estimates of the AR(1) processes within 1e-8 (relative; see the test
  for why), all other values within 1e-11, and the same warnings. The
  steady-state calibration against `STEADY_STATE2010Q1`: within 1e-12
  except for the rounding described above, which the test checks entry by
  entry. Also the valid quarters, errors for dates without data, models
  built from calibrations run, and calibrations survive JSON. As
  `test/utils/zenodo_calibration.jl`, an ignored test calibrates converted
  Zenodo calibration objects (`cargo test --test calibration -- --ignored`,
  see the test); it passes for AT, DE and FR, and for these three the port
  also agrees with the NumPy translation for 2010Q1, 2020Q1 and 2023Q4 with
  and without `use_growth_rate_ar1` within the same tolerances.
- [`tests/predictions.rs`](tests/predictions.rs), as
  `test/deterministic/prediction_pipeline.jl`: calibration of
  `ITALY_CALIBRATION` for 2010Q1 at scale 0.0001, two runs of 12 quarters
  without randomness and `get_predictions_from_sims`, against Julia's
  values in `test/deterministic/2010Q1.jld2`. With the minimum of A.14 and
  A.22 as in the Julia code (difference 1 above, as a model type in the
  test), all 105 series and the dates agree within 6.3e-12 of a level and
  2.9e-12 of a growth rate, far within Julia's test (`atol` and `rtol`
  1e-6). With the port's minimum per firm, the predictions differ by up to
  3.2e-5 of a value, and 60 of the 107 keys of the result fail Julia's
  test; the test checks this too.
- [`tests/evaluation.rs`](tests/evaluation.rs): the tests of forecasts, the
  VAR and VARX forecasts, the HP filter, the correlations and
  `correlation_stats` against an independent NumPy/SciPy translation
  ([`tests/data/evaluation_reference.py`](tests/data/evaluation_reference.py)):
  within 1e-12 (relative; values near 0 within 1e-14 absolute), except the
  HP filter's cycle, within 1e-10 (absolute, for series around 100), and
  the auto- and cross-correlations of `correlation_stats`, which are
  computed from HP cycles, within 1e-9 (absolute): LAPACK and the port
  order the operations of the LU factorization differently. Also the checks of
  `test/utils/correlation_utils.jl`, and errors where Julia throws.
- [`tests/analysis.rs`](tests/analysis.rs): the batch runs and all error
  tables as in BeforeIT's example, for 2010Q1 to 2011Q4 of
  `ITALY_CALIBRATION` with the base model and the growth-rate model. The 82
  tables against an independent NumPy/SciPy translation of
  `error_tables.jl` and `analysis_utils.jl`
  ([`tests/data/analysis_reference.py`](tests/data/analysis_reference.py))
  applied to the same predictions: the same text, except for the 22 tables
  of p-values, which are written with all their digits and agree within
  7.2e-10. Also the checks of `test/utils/analysis_utils.jl` for
  `discover_countries`, the gathering of tables across countries, and that
  failing quarters are skipped.
- Unit tests of `diff_transform` (BeforeIT.jl has none): the fields in the
  order of the Julia structs, whose names the test lists as they are in
  `src/model_init/agents.jl` and `init_properties.jl`, with the length of
  each; the added fields of `ModelGR` and `ModelCANVAS` after those of their
  agent types, as `Bit.@object` puts them; that each of the 17 properties of
  `examples/automatic_differentiation_opt.jl` changes exactly one number at
  its position; the round trip from one model into another of the same
  shape; Julia's rounding of integers and `Bool`s; and the length of the
  list, which for a new model of Austria 2010Q1 is 115 829 numbers, the
  largest `n` in BeforeIT's saved results
  `examples/data/autodiff_performance.csv`.
- Unit tests of `plots`, with `--features plots`: the mean and standard
  deviation of StatsPlots' `errorline` (NaN ignored, as NaNMath.jl does)
  and its handling of the shape of `y`; the titles, series, colours and
  time ranges of the four functions; the limits and year ticks of
  `plot_model_vs_real` against values computed by hand, with and without
  `crop`; the colours of the gradients RdBu and YlOrRd and of heatmap
  cells; the corner of the legend; the ticks and minor ticks of linear and
  logarithmic axes and the widening of their limits; and that the SVG
  drawings contain the titles, bands, labels, ticks (also rotated and
  written by a formatter), annotations and markers, a legend with or
  without frame, and horizontal lines between the series added before and
  after them. The drawings were also checked by eye in a browser.
- [`tests/datasets.rs`](tests/datasets.rs): each data set runs, has the
  values of its file, and is read like BeforeIT's dictionaries (defaults of
  the optional parameters, counts stored as floats, column matrices as
  lists, errors for missing keys).
- [`tests/invalid_inputs.rs`](tests/invalid_inputs.rs): inputs that
  `Model::new` rejects, and borderline inputs that it accepts like the Julia
  code.
- Unit tests in `src/`: the Julia semantics of `min`, `max`, `round`,
  `round(x; digits)`, `matlab_round`, `pos`, `neg`, `pos!`, `neg!`, the
  order of `sum` and how Julia prints floats; the checks of `stars` and
  `rmse_improvement` in `test/utils/analysis_utils.jl`;
  xoshiro256\*\* and SplitMix64 against reference outputs, and its jump
  function against the 2^128-th power of the generator's transition
  matrix over GF(2); the
  distributions of the draws; the Cholesky factor; `rfvar3`,
  `estimate_for_calibration_script` and `estimate_taylor_rule` against the
  values of BeforeIT.jl's tests (the last with regressors of rank 2, where
  the pseudoinverse decides); the Jacobi SVD and `rfvar3` with two
  variables, two lags and two exogenous series against exact data; the
  shocks `epsilon` against the MATLAB values in
  `test/deterministic/epsilon.jl`.

The data sets, the MATLAB values in `tests/data/matlab_*.json` and Julia's
predictions in `tests/data/prediction_pipeline_2010Q1.json` are converted
from files of BeforeIT.jl by [`data/convert.py`](data/convert.py),
[`data/convert_calibration.py`](data/convert_calibration.py) and
[`tests/data/convert.py`](tests/data/convert.py), which reproduce them byte
for byte. The other files of reference values in `tests/data` are written
by the independent translations `calibration_reference.py`,
`evaluation_reference.py` and `analysis_reference.py` there (see
[`tests/data/README.md`](tests/data/README.md)). CI checks the formatting
and runs `cargo clippy` (with and without the feature `plots`) and the
tests with all features on stable Rust, and the tests with all features on
Rust 1.88.

Each example was run (those of `analysis/` with the calibration objects of
Austria, Italy and Spain of the Zenodo record, the two
`automatic_differentiation_*` also with `--run`, `benchmark_scalability`
with 1 to 4 threads) and its plots were looked at in a browser. CI
compiles the examples (`cargo clippy --all-targets`, `cargo test`) but does
not run them, and their output is not compared with that of Julia.

## Layout

| Rust | BeforeIT.jl |
|---|---|
| `src/lib.rs`: `Model`, `step`, `run` | `src/model_init/init.jl`, `agents.jl`, `src/one_step.jl`, `run!` in `src/one_simulation.jl` |
| `src/shocks.rs` | `src/shocks/shocks.jl`, `ensemblerun!` in `src/one_simulation.jl` |
| `src/extension.rs` | model types: `src/model_init/object_macro.jl` and methods per model type |
| `src/params.rs` | the inputs and `src/model_init/init_properties.jl` |
| `src/agents.rs` | `src/model_init/`, `src/utils/modify.jl` |
| `src/firms.rs` | `src/agent_actions/firms.jl` |
| `src/households.rs` | `src/agent_actions/households.jl` |
| `src/bank.rs` | `src/agent_actions/bank.jl`, `central_bank.jl` |
| `src/government.rs` | `src/agent_actions/government.jl` |
| `src/rotw.rs` | `src/agent_actions/rotw.jl` |
| `src/aggregates.rs` | `src/agent_actions/estimations.jl`, `aggregates.jl`, `set_epsilon!` |
| `src/markets.rs` | `src/markets/` |
| `src/data.rs` | `src/utils/data.jl` |
| `src/accounting.rs` | `src/utils/get_accounting_identities.jl` |
| `src/datasets.rs`, `data/` | `src/utils/standard_params_initial_conditions.jl`, `data/` |
| `src/predictions.rs` | `src/utils/get_predictions_from_sims.jl`, `toannual.jl` |
| `src/evaluation.rs` | `src/utils/dmtest.jl`, `mztest.jl`, `bias_ttest.jl`; `mean`, `var`, `cov` of Statistics.jl, `normcdf` of StatsFuns.jl, `gamma_inc` of SpecialFunctions.jl for `a = 1` |
| `src/varx.rs` | `src/utils/varx.jl` |
| `src/correlation.rs` | `src/utils/correlation_utils.jl` |
| `src/analysis.rs` | `src/utils/analysis_utils.jl`, `error_tables.jl`, `save_all_predictions.jl` |
| `src/diff_transform.rs` | `src/utils/diff_transform.jl` |
| `src/plots.rs` | `ext/BeforeITPlots/BeforeITPlots.jl`; `errorline` of StatsPlots.jl; `mean` and `std` of NaNMath.jl; the default palette, attributes, minor ticks, axis widening and legend placement of Plots.jl; the gradients of PlotUtils.jl and ColorSchemes.jl; named colours of Colors.jl; `barplot` of Makie.jl |
| `src/calibration.rs`, `data/italy/calibration.json` | `src/utils/calibration.jl`, `_calibration_steady_state.jl`, `standard_calibration_data.jl`, `data/italy/calibration/` |
| `src/estimate.rs` | `src/utils/estimate.jl`, `nfvar3.jl`, `julia_svd.jl`, `epsilon.jl` |
| `src/model_extensions/` | `src/model_extensions/` |
| `src/random.rs` | `src/utils/randpl.jl`, `fshuffle!`, `FixedSizeWeightVector` of WeightVectors.jl, `test/deterministic/make_model_deterministic.jl` |
| `src/julia.rs` | Julia's `min`, `max`, `round`, `sum`, `x^n` and `string(x)` of a float; `src/utils/positive.jl` |
| `examples/`, `examples/data/`, `benchmark/` | `examples/` with the saved results in `examples/data/` (copied unchanged), `main.jl`, `benchmark/benchmark_scalability.sh`; the example in the README |

## Citation

This is an unofficial port. If you use it, please cite the paper on the model
and BeforeIT.jl:

```bibtex
@article{poledna2023economic,
  title   = {Economic forecasting with an agent-based model},
  author  = {Poledna, Sebastian and Miess, Michael Gregor and Hommes, Cars and Rabitsch, Katrin},
  journal = {European Economic Review},
  volume  = {151},
  pages   = {104306},
  year    = {2023},
  doi     = {10.1016/j.euroecorev.2022.104306}
}

@article{meligrana2026beforeit,
  title={BeforeIT.jl: High-Performance Agent-Based Macroeconomics Made Easy},
  author={Meligrana, Adriano and Devetak, Mitja and Hoekstra, Steven and Reiter, Oliver and Minnella, Enrico and Poledna, Sebastian and Bardoscia, Marco and Glielmo, Aldo},
  journal={arXiv preprint arXiv:2502.13267},
  year={2026}
}
```

The second entry is the citation that BeforeIT.jl asks for in its README.
[`CITATION.cff`](CITATION.cff) has the same references in machine-readable
form.

## License

Apache License 2.0, like BeforeIT.jl; see [`LICENSE`](LICENSE). The port is a
derivative work of BeforeIT.jl, Copyright 2024 Banca d'Italia and the authors;
see [`NOTICE`](NOTICE). The data sets in `data` and the MATLAB and
prediction-pipeline reference values in `tests/data` are converted from
files of BeforeIT.jl. The reference values in `tests/data` for the
calibration, forecast-evaluation and error-table tests are written by
Python scripts there: `calibration_reference.py` from
`data/italy/calibration.json`,
`evaluation_reference.py` from pseudo-random inputs it generates, and
`analysis_reference.py` from the predictions the port writes in
`tests/analysis.rs` and from `data/italy/calibration.json`; see
`tests/data/README.md`. The online appendix of the paper (CC BY-NC-ND 4.0)
is cited by its equation and table numbers; its text is not reproduced.

The disclaimer of BeforeIT.jl applies to the model: "All views expressed are
personal views, and do not necessarily reflect the official policy or
position of Banca d’Italia or those of the Bank of England or its
committees."

## Acknowledgements

The model is the work of Sebastian Poledna, Michael Gregor Miess, Cars Hommes
and Katrin Rabitsch; BeforeIT.jl is the work of its authors listed in its
README. This port was written with the help of Claude Code, an AI coding
assistant. Every rule was translated from the Julia source. The tests
described under [Verification](#verification) check the port against the
reference values in BeforeIT.jl's tests (from the authors' MATLAB code and
from Julia) and against independent re-computations. The rules that the
model extensions CANVAS and growth-rate model add are checked only for
consistency, because BeforeIT.jl's tests have no reference values for them
(`test/model_extensions.jl` is only a smoke test).
