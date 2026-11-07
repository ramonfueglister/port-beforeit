//! The data sets of BeforeIT.jl (`src/utils/standard_params_initial_conditions.jl`),
//! converted from its folder `data` to JSON in the crate's folder `data` and
//! embedded in the crate.

use std::sync::LazyLock;

use crate::params::{InitialConditions, Parameters};

/// Parameters and initial conditions of one calibration, BeforeIT
/// `InitialState`.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialState {
    pub parameters: Parameters,
    pub initial_conditions: InitialConditions,
}

impl InitialState {
    /// Reads a calibration from two JSON objects with BeforeIT's keys, see
    /// [`Parameters::from_json`] and [`InitialConditions::from_json`].
    pub fn from_json(parameters: &str, initial_conditions: &str) -> Result<InitialState, String> {
        Ok(InitialState {
            parameters: Parameters::from_json(parameters)?,
            initial_conditions: InitialConditions::from_json(initial_conditions)?,
        })
    }
}

fn embedded(parameters: &str, initial_conditions: &str) -> InitialState {
    InitialState::from_json(parameters, initial_conditions)
        .unwrap_or_else(|e| panic!("embedded data set: {e}"))
}

/// Austria, 2010Q1, BeforeIT `AUSTRIA2010Q1` (`data/austria`).
pub static AUSTRIA2010Q1: LazyLock<InitialState> = LazyLock::new(|| {
    embedded(
        include_str!("../data/austria/parameters/2010Q1.json"),
        include_str!("../data/austria/initial_conditions/2010Q1.json"),
    )
});

/// Italy, 2010Q1, BeforeIT `ITALY2010Q1` (`data/italy`), with the series
/// of the CANVAS model.
pub static ITALY2010Q1: LazyLock<InitialState> = LazyLock::new(|| {
    embedded(
        include_str!("../data/italy/parameters/2010Q1.json"),
        include_str!("../data/italy/initial_conditions/2010Q1.json"),
    )
});

/// BeforeIT `STEADY_STATE2010Q1` (`data/steady_state`): a calibration in
/// which every series of the data stays constant, as BeforeIT's
/// `test/steady_state.jl` checks.
pub static STEADY_STATE2010Q1: LazyLock<InitialState> = LazyLock::new(|| {
    embedded(
        include_str!("../data/steady_state/parameters/2010Q1.json"),
        include_str!("../data/steady_state/initial_conditions/2010Q1.json"),
    )
});
