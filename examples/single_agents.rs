//! Working with single agents, BeforeIT `examples/single_agents.jl`: an
//! agent is looked up by its ID, changed, removed and added again; then a
//! model type in which workers take out consumer loans before the credit
//! market opens.
//!
//! ```sh
//! cargo run --release --example single_agents
//! ```
//!
//! In Julia `model.w_act[id]` is a view of one agent whose fields can be
//! read and set; here the agent's position is `model.w_act.index(id)`, and
//! its fields are the entries at that position. IDs count from 0, so the
//! first worker has ID 0 (1 in Julia). The fields that the new model type
//! adds to the workers and the properties (`sum_money_to_be_paid`,
//! `contracts`) are fields of the model type, and a contract holds the ID
//! of its debtor where Julia holds the agent view.

#![allow(non_snake_case)]

use beforeit::{
    AUSTRIA2010Q1, AbstractModel, Aggregates, Banks, CentralBank, Data, Extension, Firms,
    Government, Model, Properties, Randomness, RestOfTheWorld, Workers, Xoshiro, base,
};

/// A consumer loan: `principal` paid out at issuance, `money_to_be_paid`
/// due in quarter `period` from the worker with ID `debtor`.
#[derive(Clone, Debug)]
struct ConsumerLoanContract {
    #[allow(dead_code)]
    principal: f64,
    money_to_be_paid: f64,
    period: i32,
    debtor: usize,
}

/// The model type `NewModel` with `NewWorkers` (the money each active worker
/// still has to pay, by position) and `NewProperties` (the open contracts).
#[derive(Clone, Debug)]
struct ConsumerLoans {
    sum_money_to_be_paid: Vec<f64>,
    contracts: Vec<ConsumerLoanContract>,
}

impl Extension for ConsumerLoans {
    /// The loans are signed and repaid right before the credit market,
    /// which then runs as in the standard model (`@invoke`).
    fn search_and_matching_credit<R: Randomness + ?Sized>(
        model: &mut AbstractModel<Self>,
        rng: &mut R,
    ) {
        sign_and_repay_contracts(model, rng);
        base::search_and_matching_credit(model, rng);
    }
}

/// Each active worker signs a new contract with probability 0.3 if the money
/// it has to pay is at most its income `Y_h`: a principal of 20 % of `Y_h`,
/// to be repaid with a margin of 10 % five quarters later. Then the
/// contracts due in this quarter are repaid, as the Julia example does it:
/// if the debtor's income is at most the money to be paid.
fn sign_and_repay_contracts<R: Randomness + ?Sized>(
    model: &mut AbstractModel<ConsumerLoans>,
    rng: &mut R,
) {
    let workers = &mut model.w_act;
    let ext = &mut model.ext;
    for id in workers.allids().to_vec() {
        let i = workers.index(id);
        if rng.rand() < 0.3 && ext.sum_money_to_be_paid[i] <= workers.Y_h[i] {
            let principal = 0.2 * workers.Y_h[i];
            workers.Y_h[i] += principal;
            let money_to_be_paid = 1.1 * principal;
            let period = model.agg.t as i32 + 5;
            ext.contracts.push(ConsumerLoanContract {
                principal,
                money_to_be_paid,
                period,
                debtor: id,
            });
            ext.sum_money_to_be_paid[i] += money_to_be_paid;
        }
    }
    let mut repaid_contracts_indices = Vec::new();
    for (k, contract) in ext.contracts.iter().enumerate() {
        if contract.period == model.agg.t as i32 {
            let i = workers.index(contract.debtor);
            if workers.Y_h[i] <= contract.money_to_be_paid {
                workers.Y_h[i] -= contract.money_to_be_paid;
                ext.sum_money_to_be_paid[i] -= contract.money_to_be_paid;
                repaid_contracts_indices.push(k);
            }
        }
    }
    // As in Julia: swap with the last contract and drop it, in the order of
    // the indices (out of range panics where Julia throws).
    for k in repaid_contracts_indices {
        let last = ext.contracts.len() - 1;
        ext.contracts.swap(k, last);
        ext.contracts.pop();
    }
}

fn main() -> Result<(), String> {
    let mut rng = Xoshiro::seed_from_u64(1);
    let parameters = &AUSTRIA2010Q1.parameters;
    let initial_conditions = &AUSTRIA2010Q1.initial_conditions;
    let mut model = Model::new(parameters, initial_conditions, &mut rng)?;

    // Firms, active and inactive workers hold IDs; at initialisation the ID
    // of an agent is its position, which no longer holds after a deletion.
    let id = 0;
    let i = model.w_act.index(id);

    // Read and change a field of the agent.
    println!("Y_h of worker {id}: {:?}", model.w_act.Y_h[i]);
    model.w_act.Y_h[i] = 1.0;

    // Remove the agent and add it again with the same fields. Read the
    // fields before removing the agent.
    let agentfields = model.w_act.getfields(id);
    model.w_act.delete(id);
    model.w_act.push(agentfields);

    // The ID of the last agent added,
    let id = model.w_act.lastid().ok_or("the workers have no last ID")?;
    // whose Y_h is 1.0, as it should be.
    let i = model.w_act.index(id);
    println!("Y_h of worker {id}: {:?}", model.w_act.Y_h[i]);

    // The model with consumer loans, from components initialised one by one.
    let (p, ic) = (parameters, initial_conditions);
    let firms = Firms::new(p, ic, &mut rng)?;
    let (w_act, w_inact) = Workers::new(p, ic);
    let cb = CentralBank::new(p, ic);
    let banks = Banks::new(p, ic);
    let gov = Government::new(p, ic);
    let rotw = RestOfTheWorld::new(p, ic);
    let agg = Aggregates::new(p, ic);
    let prop = Properties::new(p, ic)?;
    let data = Data::default();
    let ext = ConsumerLoans {
        sum_money_to_be_paid: vec![0.0; w_act.Y_h.len()],
        contracts: Vec::new(),
    };
    let mut model = AbstractModel::from_agents(
        (w_act, w_inact, firms, banks, cb, gov, rotw, agg, prop, data),
        ext,
        &mut rng,
    )?;

    // One quarter.
    model.step(&mut rng);
    println!(
        "after one quarter: {} open consumer loans, {:.1} to be paid",
        model.ext.contracts.len(),
        model.ext.sum_money_to_be_paid.iter().sum::<f64>()
    );
    Ok(())
}
