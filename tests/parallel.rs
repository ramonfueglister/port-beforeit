//! Parallel runs, `parallel = true` in BeforeIT: the goods markets of the
//! products in parallel threads (`search_and_matching!`), and ensembles of
//! runs (`ensemblerun!`).

#![allow(non_snake_case)]

mod common;

use beforeit::{
    Deterministic, ITALY2010Q1, InterestRateShock, Model, ModelCANVAS, NoShock, Randomness,
    Xoshiro, ensemblerun, get_accounting_identities, get_accounting_identity_banks,
};
use common::*;

fn model(rng: &mut dyn Randomness) -> Model {
    let (p, ic) = austria();
    Model::new(&p, &ic, rng).unwrap()
}

/// `one_run_deterministic.jl`: without randomness, serial and parallel goods
/// markets give the same run. The products' results are added in the order
/// of the products, so the runs are identical; the authors' test allows a
/// relative difference of 1e-4, because BeforeIT.jl adds the households'
/// consumption in the order in which the threads finish.
#[test]
fn without_randomness_parallel_equals_serial() {
    let run = |parallel| {
        let mut m = model(&mut Deterministic);
        m.run_with(2, &mut Deterministic, parallel, &NoShock);
        m
    };
    assert_eq!(run(false), run(true));
}

/// With randomness, a parallel run is reproducible with the same seed and
/// keeps the accounting identities.
#[test]
fn parallel_runs_are_reproducible_and_consistent() {
    let run = |seed| {
        let rng = &mut Xoshiro::seed_from_u64(seed);
        let mut m = model(rng);
        m.run_with(3, rng, true, &NoShock);
        m
    };
    let m = run(5);
    assert_eq!(m, run(5));
    assert_ne!(m.data.nominal_gdp, run(6).data.nominal_gdp);
    let (a, b, c) = get_accounting_identities(&m.data);
    let (d, e) = get_accounting_identity_banks(&m);
    for r in [a, b, c, d, e] {
        assert!(r.abs() <= 1e-8, "{r:e}");
    }
}

/// Without `parallel`, `ensemblerun` runs the models one after the other on
/// the same generator.
#[test]
fn serial_ensemble_runs_one_model_after_the_other() {
    let models = vec![model(&mut Deterministic); 3];
    let shock = InterestRateShock {
        rate: 0.01,
        final_time: 1,
    };
    let mut ensemble = models.clone();
    let rng = &mut Xoshiro::seed_from_u64(9);
    ensemblerun(&mut ensemble, 2, rng, false, &shock);
    let mut one_by_one = models;
    let rng_b = &mut Xoshiro::seed_from_u64(9);
    for m in &mut one_by_one {
        m.run_with(2, rng_b, false, &shock);
    }
    assert_eq!(ensemble, one_by_one);
    assert_eq!(rng, rng_b);
}

/// With `parallel`, each model has its own generator, split off in the
/// order of the models, so the result does not depend on the number of
/// threads. More models than processors put several models on a thread.
#[test]
fn parallel_ensemble_gives_each_model_its_own_generator() {
    let n = 2 * std::thread::available_parallelism().map_or(1, |n| n.get()) + 1;
    let models = vec![model(&mut Deterministic); n];
    let mut ensemble = models.clone();
    let rng = &mut Xoshiro::seed_from_u64(9);
    ensemblerun(&mut ensemble, 1, rng, true, &NoShock);
    let rng_b = &mut Xoshiro::seed_from_u64(9);
    let mut rngs: Vec<_> = (0..n).map(|_| rng_b.split()).collect();
    let mut expected = models;
    for (m, r) in expected.iter_mut().zip(&mut rngs) {
        m.run_with(1, &mut **r, true, &NoShock);
    }
    assert_eq!(ensemble, expected);
    assert_eq!(rng, rng_b);
    assert_ne!(ensemble[0].data, ensemble[1].data);
}

/// Without randomness, the parallel ensemble equals the serial one.
#[test]
fn without_randomness_parallel_ensemble_equals_serial() {
    let models = vec![model(&mut Deterministic); 3];
    let mut serial = models.clone();
    ensemblerun(&mut serial, 2, &mut Deterministic, false, &NoShock);
    let mut parallel = models;
    ensemblerun(&mut parallel, 2, &mut Deterministic, true, &NoShock);
    assert_eq!(serial, parallel);
}

/// The model extensions run in ensembles too, as in BeforeIT
/// `examples/CANVAS_extension.jl`.
#[test]
fn canvas_ensemble() {
    let s = &*ITALY2010Q1;
    let rng = &mut Xoshiro::seed_from_u64(2);
    let m = ModelCANVAS::new(&s.parameters, &s.initial_conditions, rng).unwrap();
    let mut models = vec![m; 2];
    ensemblerun(&mut models, 1, rng, true, &NoShock);
    assert!(models.iter().all(|m| m.agg.t == 2));
    assert_ne!(models[0].data, models[1].data);
}
