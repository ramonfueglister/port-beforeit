//! Inputs that `Model::new` rejects with an error, and borderline inputs it
//! accepts.

#![allow(non_snake_case)]

mod common;

use beforeit::{Deterministic, InitialConditions, Model, Parameters, Xoshiro};
use common::*;

fn error(p: &Parameters, ic: &InitialConditions) -> String {
    Model::new(p, ic, &mut Deterministic).unwrap_err()
}

#[test]
fn invalid_inputs_give_an_error() {
    let (p, ic) = austria();
    let I: usize = p.I_s.iter().sum();

    let mut q = p.clone();
    q.H_act = I;
    let e = error(&q, &ic);
    assert!(e.contains("firm owners plus the bank owner"), "{e}");

    let mut q = p.clone();
    q.H_act = I + 100;
    let e = error(&q, &ic);
    assert!(
        e.contains("employees, but there are only 99 workers"),
        "{e}"
    );

    let mut q = p.clone();
    q.b_HH_g.pop();
    let e = error(&q, &ic);
    assert!(e.contains("b_HH_g has 61 entries instead of G = 62"), "{e}");

    let mut q = p.clone();
    q.a_sg[5].pop();
    let e = error(&q, &ic);
    assert!(e.contains("a_sg has a row with 61 entries"), "{e}");

    let mut q = p.clone();
    q.T_prime = 1;
    let e = error(&q, &ic);
    assert!(e.contains("needs at least 2 quarters"), "{e}");

    let mut j = ic.clone();
    j.Y.truncate(10);
    let e = error(&p, &j);
    assert!(
        e.contains("Y has 10 quarters, T_prime = 54 are needed"),
        "{e}"
    );

    let mut q = p.clone();
    q.C[0][0] = -1.0;
    let e = error(&q, &ic);
    assert!(e.contains("not positive definite"), "{e}");

    let mut q = p.clone();
    q.C[0][1] += 0.1;
    let e = error(&q, &ic);
    assert!(e.contains("not symmetric"), "{e}");

    let mut j = ic.clone();
    j.N_s[0] = f64::NAN;
    let e = error(&p, &j);
    assert!(e.contains("N_s: cannot round NaN"), "{e}");
}

/// Accepted as by the Julia code: two quarters of history, where the
/// variance of the single residual is undefined and the shock is 0, and a
/// matrix `C` of zeros, which turns the shocks off.
#[test]
fn borderline_inputs_are_accepted() {
    let (p, ic) = austria();

    let mut q = p.clone();
    q.T_prime = 2;
    let mut m = Model::new(&q, &ic, &mut Xoshiro::seed_from_u64(1)).unwrap();
    m.run(1, &mut Xoshiro::seed_from_u64(2));
    assert!(m.agg.Y_e.is_finite());

    let mut q = p.clone();
    q.C = [[0.0; 3]; 3];
    let mut m = Model::new(&q, &ic, &mut Xoshiro::seed_from_u64(1)).unwrap();
    m.run(1, &mut Xoshiro::seed_from_u64(2));
    assert_eq!(
        (m.agg.epsilon_Y_EA, m.agg.epsilon_E, m.agg.epsilon_I),
        (0.0, 0.0, 0.0)
    );
}
