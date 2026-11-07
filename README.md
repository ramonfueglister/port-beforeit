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
| Ported | Model setup (`src/model_init/`); adding, deleting and looking up agents by ID (`src/utils/modify.jl`); model types (`@object` and methods per model type) as the trait `Extension`; all agent rules (`src/agent_actions/`) with one or several banks (`n_banks`) and the options `omega_pi` and `overdrafts_in_capital_ratio`; the credit, labour and goods markets (`src/markets/`), also with the goods markets of the products in parallel threads; `step!`, `run!` and `ensemblerun!` with shocks and the option `parallel` (`one_step.jl`, `one_simulation.jl`) and the shock types (`src/shocks/`); data collection with the series per bank and `DataVector` (`src/utils/data.jl`); `get_accounting_identities.jl`; the data sets `AUSTRIA2010Q1`, `ITALY2010Q1` and `STEADY_STATE2010Q1` (`standard_params_initial_conditions.jl`); the model extensions CANVAS (`init_CANVAS.jl`, with the option `theta_calvo`) and growth-rate model (`init_growth_rate_model.jl`); `estimate.jl` with `estimate_taylor_rule`, `nfvar3.jl`, `julia_svd.jl`, `epsilon.jl` and `positive.jl`; and the parts of `randpl.jl` and `opt.jl` that these use |
| Not yet ported | Calibration with its data `ITALY_CALIBRATION`, prediction pipelines and forecast evaluation; the package extensions in `ext/` |

## Installation

The crate is not on crates.io. Add it as a Git dependency:

```toml
[dependencies]
beforeit = { git = "https://github.com/ramonfueglister/port-beforeit" }
```

It needs Rust 1.88 or newer. It depends on
[`libm`](https://crates.io/crates/libm) and, to read the data sets,
[`serde_json`](https://crates.io/crates/serde_json); it contains no `unsafe`
code.

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
set prices and quantities by demand pull, only a share `theta_calvo` of them
adjusts prices in a quarter, and the central bank re-estimates the Taylor
rule every quarter; its histories are in `model.ext` (BeforeIT.jl keeps
them in `model.cb` and `model.rotw`). As in BeforeIT.jl, this estimate
receives euro area GDP in levels where `estimate_taylor_rule` expects
inflation, and inflation where it expects growth. The growth-rate model
replaces the AR(1) processes on logarithms by AR(1) processes on growth
rates for the expectations, government consumption, exports and imports.

A shock changes the model in every quarter right after the central bank has
set the policy rate, where `step!` calls `shock!`. BeforeIT.jl's shock types
`NoShock`, `InterestRateShock`, `ProductivityShock` and `ConsumptionShock`
implement the trait `Shock`, and so does every closure that takes the model:

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

`cargo doc --open` shows the documentation of every type and rule.

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
`rand` and `rand_below`. Mathematical functions such as `exp` and `log` come
from the `libm` crate, not from the system's math library.

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
   regressors have more than two columns, which the port computes with
   one-sided Jacobi rotations. Results
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
   therefore identical.

## Where BeforeIT.jl differs from the paper

In these places the code of BeforeIT.jl differs from the online appendix, and
the port follows the code. They are the ones we found; the list may not be
complete.

| | Online appendix | BeforeIT.jl and the port |
|---|---|---|
| Price index (A.12) | Weighted by the sales of the domestic firms and of the foreign producers | Weighted by the production of the domestic firms, as in the authors' MATLAB code (`detabm.m`, lines 301 and 302) |
| Policy rate (A.69) | Taylor rule without a lower bound | Not below 0 (`pos` in `taylor_rule`); the MATLAB code has no lower bound (`detabm.m`, line 98) |
| Choice of a seller (A.1.1) | The size of a firm is its production | The size of a firm is its production plus its inventories, as in the MATLAB code (`search_and_matching_det.m`, line 10) |

## Verification

`cargo test --release` runs these tests:

- [`tests/matlab_reference.rs`](tests/matlab_reference.rs), as BeforeIT.jl's
  `test/deterministic`: without randomness, the initial state, every rule of
  one quarter (with serial and with parallel goods markets, as there), and
  the aggregates after quarters 1 and 5 against the values of the authors'
  MATLAB code, with the tolerances of BeforeIT.jl's tests. A
  stricter test requires every compared quantity to lie within 1e-10
  (relative) of the MATLAB values; the largest deviation is 6.5e-13, for the
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
  more models than threads and for `ModelCANVAS`.
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
- [`tests/datasets.rs`](tests/datasets.rs): each data set runs, has the
  values of its file, and is read like BeforeIT's dictionaries (defaults of
  the optional parameters, counts stored as floats, column matrices as
  lists, errors for missing keys).
- [`tests/invalid_inputs.rs`](tests/invalid_inputs.rs): inputs that
  `Model::new` rejects, and borderline inputs that it accepts like the Julia
  code.
- Unit tests in `src/`: the Julia semantics of `min`, `max`, `round`,
  `matlab_round`, `pos`, `neg`, `pos!`, `neg!` and the order of `sum`;
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

The data sets and the reference values are converted from files of
BeforeIT.jl by [`data/convert.py`](data/convert.py) and
[`tests/data/convert.py`](tests/data/convert.py), which reproduce them byte
for byte. CI checks the formatting and runs `cargo clippy` and the
tests on stable Rust, and the tests on Rust 1.88.

## Layout

| Rust | BeforeIT.jl |
|---|---|
| `src/lib.rs`: `Model`, `step`, `run` | `src/model_init/init.jl`, `agents.jl`, `src/one_step.jl`, `run!` in `src/one_simulation.jl` |
| `src/shocks.rs` | `src/shocks/shocks.jl`, `ensemblerun!` in `src/one_simulation.jl` |
| `src/extension.rs` | model types: `src/model_init/object_macro.jl` and methods per model type |
| `src/params.rs` | the inputs and `src/model_init/init_properties.jl` |
| `src/agents.rs` | `src/model_init/`, `src/utils/modify.jl` |
| `src/firms.rs` | `src/agent_actions/firms.jl`, `finance_insolvent_firms!` |
| `src/households.rs` | `src/agent_actions/households.jl` |
| `src/bank.rs` | `src/agent_actions/bank.jl`, `central_bank.jl` |
| `src/government.rs` | `src/agent_actions/government.jl` |
| `src/rotw.rs` | `src/agent_actions/rotw.jl` |
| `src/aggregates.rs` | `src/agent_actions/estimations.jl`, `aggregates.jl`, `set_epsilon!` |
| `src/markets.rs` | `src/markets/` |
| `src/data.rs` | `src/utils/data.jl` |
| `src/accounting.rs` | `src/utils/get_accounting_identities.jl` |
| `src/datasets.rs`, `data/` | `src/utils/standard_params_initial_conditions.jl`, `data/` |
| `src/estimate.rs` | `src/utils/estimate.jl`, `nfvar3.jl`, `julia_svd.jl`, `epsilon.jl` |
| `src/model_extensions/` | `src/model_extensions/` |
| `src/random.rs` | `src/utils/randpl.jl`, `fshuffle!`, `FixedSizeWeightVector` of WeightVectors.jl, `test/deterministic/make_model_deterministic.jl` |
| `src/julia.rs` | Julia's `min`, `max`, `round` and `sum`; `src/utils/positive.jl` |

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
see [`NOTICE`](NOTICE). The files in `data` and `tests/data` are converted
from files of BeforeIT.jl. The online appendix of the paper (CC BY-NC-ND 4.0) is cited by
its equation and table numbers; its text is not reproduced.

The disclaimer of BeforeIT.jl applies to the model: "All views expressed are
personal views, and do not necessarily reflect the official policy or
position of Banca d’Italia or those of the Bank of England or its
committees."

## Acknowledgements

The model is the work of Sebastian Poledna, Michael Gregor Miess, Cars Hommes
and Katrin Rabitsch; BeforeIT.jl is the work of its authors listed in its
README. This port was written with the help of Claude Code, an AI coding
assistant. Every rule was translated from the Julia source and is checked
against the reference values described under [Verification](#verification).
