//! Tools around the model, as in BeforeIT `test/utils`.

#![allow(non_snake_case)]

mod common;

use beforeit::julia::{neg, pos};
use beforeit::{DataVector, Deterministic, Model, Worker, Xoshiro};
use common::*;

fn model() -> Model {
    let (p, ic) = austria();
    Model::new(&p, &ic, &mut Deterministic).unwrap()
}

/// `Retrieve/Add/Remove Agents` (`test/utils/modify.jl`). IDs, employers
/// and banks count from 0 here, from 1 in Julia.
#[test]
fn retrieve_add_remove_agents() {
    let mut model = model();
    let actw = &mut model.w_act;
    let id = 0;
    let i = actw.index(id);
    actw.Y_h[i] += 1.0;
    let agentf = actw.getfields(id);
    actw.delete(id);
    assert!(!actw.allids().contains(&id));

    // A new worker of the first bank.
    actw.push(Worker {
        Y_h: 1.0,
        D_h: 1.0,
        K_h: 1.0,
        w_h: 1.0,
        O_h: Some(0),
        C_d_h: 1.0,
        I_d_h: 1.0,
        C_h: 1.0,
        I_h: 2.0,
        B_h: 0,
    });
    let id = actw.lastid().unwrap();
    let i = actw.index(id);
    assert_eq!(actw.Y_h[i], 1.0);
    assert_eq!(actw.I_h[i], 2.0);
    assert_eq!(actw.B_h[i], 0);

    actw.push(agentf);
    let id = model.w_act.lastid().unwrap();
    assert_eq!(model.w_act.getfields(id), agentf);
}

/// The bookkeeping behind `delete!` and `push!`: the last agent moves to
/// the free position, and every remaining ID still finds its values.
#[test]
fn ids_find_their_agents_after_deletions() {
    let mut firms = model().firms;
    let n = firms.len();
    let before: Vec<_> = (0..n).map(|id| firms.getfields(id)).collect();
    assert_eq!(firms.lastid(), Some(n - 1));
    assert!(firms.id_to_index.is_empty());

    firms.delete(3);
    assert!(firms.del);
    assert_eq!(firms.len(), n - 1);
    assert_eq!(firms.ID[3], n - 1);
    assert_eq!(firms.index(n - 1), 3);
    firms.delete(n - 1);
    firms.delete(0);
    let mut extra = before[5];
    extra.Y_i = 123.0;
    firms.push(extra);
    assert_eq!(firms.lastid(), Some(n));
    assert_eq!(firms.len(), n - 2);

    for id in [3, 0, n - 1] {
        assert!(!firms.allids().contains(&id));
    }
    for &id in firms.allids() {
        let expected = if id == n { extra } else { before[id] };
        assert_eq!(firms.getfields(id), expected, "ID {id}");
    }
    assert_eq!(firms.id_to_index.len(), firms.len());
}

#[test]
#[should_panic(expected = "there is no agent with ID 7")]
fn deleting_a_missing_id_panics() {
    let mut w = model().w_inact;
    w.delete(7);
    w.delete(7);
}

/// `test pos` and `test neg` (`test/utils/positive.jl`).
#[test]
fn pos_and_neg() {
    let a = [1.0, 2.0, 3.0, f64::NAN, -1.0];
    let p: Vec<f64> = a.iter().map(|&x| pos(x)).collect();
    assert_eq!(p, [1.0, 2.0, 3.0, 0.0, 0.0]);
    assert_eq!(pos(1.0), 1.0);
    assert_eq!(pos(-1.0), 0.0);
    assert_eq!(pos(f64::NAN), 0.0);

    let n: Vec<f64> = a.iter().map(|&x| neg(x)).collect();
    assert_eq!(n, [0.0, 0.0, 0.0, 0.0, -1.0]);
    assert_eq!(neg(1.0), 0.0);
    assert_eq!(neg(-1.0), -1.0);
    assert_eq!(neg(f64::NAN), 0.0);
}

/// `DataVector`: row `t` of a series holds the value of each run.
#[test]
fn data_vector_puts_runs_side_by_side() {
    let (p, ic) = austria();
    let models: Vec<Model> = (1..=3)
        .map(|seed| {
            let rng = &mut Xoshiro::seed_from_u64(seed);
            let mut m = Model::new(&p, &ic, rng).unwrap();
            m.run(2, rng);
            m
        })
        .collect();
    let dv = DataVector::from_models(&models);
    assert_eq!(dv.len(), 3);
    let gdp = dv.series(|d| &d.real_gdp);
    assert_eq!(gdp.len(), 3);
    for (t, row) in gdp.iter().enumerate() {
        for (k, m) in models.iter().enumerate() {
            assert_eq!(row[k], m.data.real_gdp[t]);
        }
    }
    assert_eq!(dv[1], models[1].data);
    assert_eq!(dv.iter().count(), 3);
    let one = DataVector::from_model(&models[0]);
    assert_eq!(one.series(|d| &d.equity_per_bank).len(), 3);
}
