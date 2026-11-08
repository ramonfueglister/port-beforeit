//! Shocks, BeforeIT `src/shocks/shocks.jl`, and ensembles of runs,
//! `ensemblerun!` in BeforeIT `src/one_simulation.jl`.
//!
//! A shock changes the model in every quarter right after the central bank
//! has set the policy rate, where `step!` calls `shock!(model)`. A Julia
//! shock is a callable struct; here it is a type that implements [`Shock`].
//! Closures `|m: &mut Model| { ... }` that implement `Fn(&mut Model)`, that
//! is, return `()` and neither move out of the variables they capture nor
//! need `&mut` access to them, are shocks as well.

use crate::random::Randomness;
use crate::{AbstractModel, Extension, Model};

/// A change to the model applied in every quarter right after the central
/// bank has set the policy rate (`shock!(model)` in BeforeIT
/// `src/one_step.jl`).
pub trait Shock<X> {
    fn apply(&self, model: &mut AbstractModel<X>);
}

impl<X, F: Fn(&mut AbstractModel<X>)> Shock<X> for F {
    fn apply(&self, model: &mut AbstractModel<X>) {
        self(model)
    }
}

/// `NoShock`: changes nothing; the default of `step!` and `run!`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoShock;

impl<X> Shock<X> for NoShock {
    fn apply(&self, _model: &mut AbstractModel<X>) {}
}

/// `InterestRateShock(rate, final_time)`: the policy rate `cb.r_bar` is
/// `rate` as long as the model time `agg.t` is at most `final_time`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterestRateShock {
    pub rate: f64,
    pub final_time: usize,
}

impl<X> Shock<X> for InterestRateShock {
    fn apply(&self, model: &mut AbstractModel<X>) {
        if model.agg.t <= self.final_time {
            model.cb.r_bar = self.rate;
        }
    }
}

/// `ProductivityShock(productivity_multiplier)`: multiplies the labour
/// productivity `alpha_bar_i` of every firm by `productivity_multiplier`.
/// As in BeforeIT.jl, it does so in every quarter in which it is applied,
/// and only for [`Model`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProductivityShock {
    pub productivity_multiplier: f64,
}

impl Shock<crate::Standard> for ProductivityShock {
    fn apply(&self, model: &mut Model) {
        for a in &mut model.firms.alpha_bar_i {
            *a *= self.productivity_multiplier;
        }
    }
}

/// `ConsumptionShock(consumption_multiplier, final_time)`: multiplies the
/// propensity to consume `prop.psi` by `consumption_multiplier` when the
/// model time `agg.t` is 1, and divides it by `consumption_multiplier`
/// again when `agg.t` is `final_time`. Only for [`Model`], as in
/// BeforeIT.jl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsumptionShock {
    pub consumption_multiplier: f64,
    pub final_time: usize,
}

impl Shock<crate::Standard> for ConsumptionShock {
    fn apply(&self, model: &mut Model) {
        if model.agg.t == 1 {
            model.prop.psi *= self.consumption_multiplier;
        } else if model.agg.t == self.final_time {
            model.prop.psi /= self.consumption_multiplier;
        }
    }
}

/// `ensemblerun!(models, T; parallel, shock!)`: runs each model `T`
/// quarters as [`AbstractModel::run_with`].
///
/// Without `parallel`, the models run one after the other and all draw from
/// `rng`. With `parallel`, they run in parallel threads: each model draws
/// from its own generator, split off from `rng` in the order of the models
/// (see [`Randomness::split`]), so the results do not depend on the number
/// of threads; as in BeforeIT.jl, `parallel` is passed on to the goods
/// markets of each model. BeforeIT.jl also accepts a generator of models;
/// it collects it into a vector first, which here is the caller's
/// `Vec`.
///
/// # Panics
///
/// As [`AbstractModel::step`]; a panic in a thread is passed on.
pub fn ensemblerun<X, R, S>(
    models: &mut [AbstractModel<X>],
    T: usize,
    rng: &mut R,
    parallel: bool,
    shock: &S,
) where
    X: Extension + Send,
    R: Randomness + ?Sized,
    S: Shock<X> + Sync + ?Sized,
{
    if !parallel {
        for model in models.iter_mut() {
            model.run_with(T, rng, false, shock);
        }
        return;
    }
    let mut rngs: Vec<_> = models.iter().map(|_| rng.split()).collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = models.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = models
            .chunks_mut(chunk)
            .zip(rngs.chunks_mut(chunk))
            .map(|(models, rngs)| {
                scope.spawn(move || {
                    for (model, rng) in models.iter_mut().zip(rngs) {
                        model.run_with(T, &mut **rng, true, shock);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap_or_else(|e| std::panic::resume_unwind(e));
        }
    });
}
