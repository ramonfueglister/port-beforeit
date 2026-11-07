//! The model as one list of numbers and back, BeforeIT
//! `src/utils/diff_transform.jl`: [`model_to_array`] writes the numbers of a
//! model into a list, [`array_to_model`] makes a model from such a list.
//! BeforeIT's examples `examples/automatic_differentiation_*.jl` use the two
//! functions to compute gradients of a simulation with respect to these
//! numbers.
//!
//! The order is BeforeIT's: `w_act`, `w_inact`, `firms`, `banks`, `cb`,
//! `gov`, `rotw`, `agg` and `prop`, and in each of them the fields in the
//! order of the Julia struct. Of each struct, Julia takes the fields that
//! hold a number, a vector of numbers or a matrix of numbers (column by
//! column), and skips `del`, `lastid`, `id_to_index`, `ID` and every other
//! field, such as the lists of customers of the banks; `data` is not
//! included. A model type whose agent types have more fields adds them after
//! the fields of each agent type, through [`Extension::push_fields`] and
//! [`Extension::set_fields`].
//!
//! The numbers are those of the Julia model: integers and `Bool`s as floats,
//! sectors (`G_i`), banks (`B_i`, `B_h`) and employers (`O_h`) counted from 1,
//! an unemployed worker with employer 0 and an inactive person with −1.
//!
//! ```
//! use beforeit::diff_transform::{array_to_model, model_to_array};
//! use beforeit::{AUSTRIA2010Q1, Model, Xoshiro};
//!
//! # fn main() -> Result<(), String> {
//! let s = &*AUSTRIA2010Q1;
//! let mut rng = Xoshiro::seed_from_u64(1);
//! let model = Model::new(&s.parameters, &s.initial_conditions, &mut rng)?;
//! let mut arr = model_to_array(&model);
//! // The last number is `prop.n_banks`, the one before `prop.r_bar`.
//! let n = arr.len();
//! arr[n - 2] += 0.01;
//! let changed = array_to_model(&arr, &model)?;
//! assert_eq!(changed.prop.r_bar, model.prop.r_bar + 0.01);
//! # Ok(())
//! # }
//! ```

use crate::julia::{round, round_int};
use crate::{AbstractModel, Extension};

/// A part of the model, in the order of [`model_to_array`]: the agent types
/// whose added fields [`Extension::push_fields`] and
/// [`Extension::set_fields`] handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// `w_act`.
    WorkersActive,
    /// `w_inact`.
    WorkersInactive,
    /// `firms`.
    Firms,
    /// `banks`.
    Banks,
    /// `cb`.
    CentralBank,
    /// `gov`.
    Government,
    /// `rotw`.
    RestOfTheWorld,
    /// `agg`.
    Aggregates,
    /// `prop`.
    Properties,
}

/// Employer `O_h` of an unemployed worker in BeforeIT.jl.
const UNEMPLOYED: i64 = 0;
/// Employer `O_h` of an economically inactive person in BeforeIT.jl.
const INACTIVE: i64 = -1;

/// Calls `$f!(kind field)` for every field that BeforeIT's
/// `_flatten_struct!` takes, in the order of `model_to_array`, and
/// `$f!(added Part)` after the fields of each part. The kinds say how the
/// port stores the field: `float(s)` as `f64`, `ints` as `i64`, `count(s)` as
/// `usize`, `indices` as `usize` counted from 0 (Julia from 1),
/// `occupations` as `Option<usize>` (Julia: employer from 1, or the given
/// value), `bool` and `matrix` as rows.
macro_rules! julia_fields {
    ($f:ident, $m:ident) => {
        julia_fields!(@workers $f, $m.w_act, UNEMPLOYED);
        $f!(added WorkersActive);
        julia_fields!(@workers $f, $m.w_inact, INACTIVE);
        $f!(added WorkersInactive);
        julia_fields!(@each $f, $m.firms;
            indices G_i, floats alpha_bar_i, floats beta_i, floats kappa_i, floats w_i,
            floats w_bar_i, floats delta_i, floats tau_Y_i, floats tau_K_i, ints N_i,
            floats Y_i, floats Q_i, floats Q_d_i, floats P_i, floats S_i, floats K_i,
            floats M_i, floats L_i, floats pi_bar_i, floats D_i, floats Pi_i, ints V_i,
            floats I_i, floats E_i, floats P_bar_i, floats P_CF_i, floats DS_i, floats DM_i,
            floats DL_i, floats DL_d_i, floats K_e_i, floats L_e_i, floats Q_s_i,
            floats I_d_i, floats DM_d_i, ints N_d_i, floats Pi_e_i, indices B_i,
            floats Y_h, floats C_d_h, floats I_d_h, floats C_h, floats I_h, floats K_h,
            floats D_h);
        $f!(added Firms);
        // `firms`, `w_act` and `w_inact` hold the customers, which Julia skips.
        julia_fields!(@each $f, $m.banks;
            floats E_k, floats Pi_k, floats Pi_e_k, floats D_k, floats r, floats Y_h,
            floats C_d_h, floats I_d_h, floats C_h, floats I_h, floats K_h, floats D_h);
        $f!(added Banks);
        julia_fields!(@each $f, $m.cb;
            float r_bar, float r_G, float rho, float r_star, float pi_star, float xi_pi,
            float xi_gamma, float E_CB);
        $f!(added CentralBank);
        julia_fields!(@each $f, $m.gov;
            float alpha_G, float beta_G, float sigma_G, float Y_G, float C_G, float L_G,
            float sb_inact, float sb_other, floats C_d_j, float C_j, float P_j);
        $f!(added Government);
        julia_fields!(@each $f, $m.rotw;
            float alpha_E, float beta_E, float sigma_E, float alpha_I, float beta_I,
            float sigma_I, float Y_EA, float gamma_EA, float pi_EA, float alpha_pi_EA,
            float beta_pi_EA, float sigma_pi_EA, float alpha_Y_EA, float beta_Y_EA,
            float sigma_Y_EA, float D_RoW, float Y_I, float C_E, floats C_d_l, float C_l,
            floats Y_m, floats Q_m, floats Q_d_m, floats P_m, float P_l);
        $f!(added RestOfTheWorld);
        julia_fields!(@each $f, $m.agg;
            floats Y, floats pi_, float P_bar, floats P_bar_g, float P_bar_HH,
            float P_bar_CF, float P_bar_h, float P_bar_CF_h, float Y_e, float gamma_e,
            float pi_e, float epsilon_Y_EA, float epsilon_E, float epsilon_I, count t);
        $f!(added Aggregates);
        julia_fields!(@each $f, $m.prop;
            count G, count T_prime, count H_act, count H_inact, count J, count L,
            counts I_s, count I, count H, float tau_INC, float tau_FIRM, float tau_VAT,
            float tau_SIF, float tau_SIW, float tau_EXPORT, float tau_CF, float tau_G,
            float theta_UB, float psi, float psi_H, float mu, float theta_DIV, float theta,
            float zeta, float zeta_LTV, float zeta_b, float omega_pi, float theta_calvo,
            bool overdrafts_in_capital_ratio, floats b_CF_g, floats b_CFH_g, floats b_HH_g,
            floats c_G_g, floats c_E_g, floats c_I_g, matrix a_sg, matrix C, float D_H,
            float K_H, float sb_other, float E_k, float r_bar, count n_banks);
        $f!(added Properties);
    };
    (@workers $f:ident, $w:expr, $none:expr) => {
        julia_fields!(@each $f, $w; floats Y_h, floats D_h, floats K_h, floats w_h);
        $f!(occupations $w.O_h, $none);
        julia_fields!(@each $f, $w;
            floats C_d_h, floats I_d_h, floats C_h, floats I_h, indices B_h);
    };
    (@each $f:ident, $s:expr; $($kind:ident $field:ident),* $(,)?) => {
        $( $f!($kind $s.$field); )*
    };
}

/// `model_to_array(model)`: the numbers of the model in the order described
/// in the [module documentation](self).
pub fn model_to_array<X: Extension>(model: &AbstractModel<X>) -> Vec<f64> {
    let mut v = Vec::new();
    macro_rules! out {
        (float $e:expr) => {
            v.push($e)
        };
        (floats $e:expr) => {
            v.extend_from_slice(&$e)
        };
        (ints $e:expr) => {
            v.extend($e.iter().map(|&x| x as f64))
        };
        (count $e:expr) => {
            v.push($e as f64)
        };
        (counts $e:expr) => {
            v.extend($e.iter().map(|&x| x as f64))
        };
        (indices $e:expr) => {
            v.extend($e.iter().map(|&x| (x + 1) as f64))
        };
        (occupations $e:expr, $none:expr) => {
            v.extend(
                $e.iter()
                    .map(|o| o.map_or($none as f64, |i| (i + 1) as f64)),
            )
        };
        (bool $e:expr) => {
            v.push(if $e { 1.0 } else { 0.0 })
        };
        (matrix $e:expr) => {
            push_columns(&mut v, &$e[..])
        };
        (added $part:ident) => {
            model.ext.push_fields(Part::$part, &mut v)
        };
    }
    julia_fields!(out, model);
    v
}

/// `array_to_model(arr, original_model)`: a copy of `original_model` with
/// the numbers of `arr`, in the order of [`model_to_array`]. Vectors and
/// matrices keep the lengths of `original_model`; `data` and the fields that
/// Julia skips are those of `original_model`.
///
/// Integer fields take the nearest integer, ties to even, Julia
/// `round(Int, x)`, and the `Bool` field `overdrafts_in_capital_ratio` the
/// nearest of 0 and 1, Julia `round(Bool, x)`. As in Julia, numbers after
/// those that the model takes are ignored.
///
/// Returns an error where Julia throws: if `arr` is shorter than the model
/// needs, or if a number for an integer field is not finite or its nearest
/// integer does not fit in 64 bits (for `Bool`: is neither 0 nor 1). It also
/// returns an error for an integer that Julia stores but this port cannot:
/// a negative count, a sector or bank below 1, or an employer `O_h` below 1
/// other than 0 for `w_act` and −1 for `w_inact`.
pub fn array_to_model<X: Extension + Clone>(
    arr: &[f64],
    original_model: &AbstractModel<X>,
) -> Result<AbstractModel<X>, String> {
    let mut m = original_model.clone();
    let mut r = Numbers { arr, pos: 0 };
    macro_rules! inp {
        (float $e:expr) => {
            $e = r.take(1, stringify!($e))?[0]
        };
        (floats $e:expr) => {{
            let n = $e.len();
            $e.copy_from_slice(r.take(n, stringify!($e))?)
        }};
        (ints $e:expr) => {{
            let n = $e.len();
            for (x, &y) in $e.iter_mut().zip(r.take(n, stringify!($e))?) {
                *x = round_int(y).map_err(|e| field_error(stringify!($e), e))?;
            }
        }};
        (count $e:expr) => {
            $e = count(r.take(1, stringify!($e))?[0]).map_err(|e| field_error(stringify!($e), e))?
        };
        (counts $e:expr) => {{
            let n = $e.len();
            for (x, &y) in $e.iter_mut().zip(r.take(n, stringify!($e))?) {
                *x = count(y).map_err(|e| field_error(stringify!($e), e))?;
            }
        }};
        (indices $e:expr) => {{
            let n = $e.len();
            for (x, &y) in $e.iter_mut().zip(r.take(n, stringify!($e))?) {
                *x = index(y).map_err(|e| field_error(stringify!($e), e))?;
            }
        }};
        (occupations $e:expr, $none:expr) => {{
            let n = $e.len();
            for (x, &y) in $e.iter_mut().zip(r.take(n, stringify!($e))?) {
                *x = occupation(y, $none).map_err(|e| field_error(stringify!($e), e))?;
            }
        }};
        (bool $e:expr) => {
            $e = boolean(r.take(1, stringify!($e))?[0])
                .map_err(|e| field_error(stringify!($e), e))?
        };
        (matrix $e:expr) => {
            set_columns(&mut $e[..], &mut r, stringify!($e))?
        };
        (added $part:ident) => {{
            let n = m.ext.set_fields(Part::$part, &r.arr[r.pos..])?;
            r.take(n, stringify!($part))?;
        }};
    }
    julia_fields!(inp, m);
    Ok(m)
}

/// Sets the floats of `fields`, in order, from the start of `numbers`, and
/// returns how many it took: a helper for [`Extension::set_fields`].
///
/// Returns an error if `numbers` is too short.
pub fn set_floats<'a>(
    fields: impl IntoIterator<Item = &'a mut [f64]>,
    numbers: &[f64],
) -> Result<usize, String> {
    let mut r = Numbers {
        arr: numbers,
        pos: 0,
    };
    for field in fields {
        let n = field.len();
        field.copy_from_slice(r.take(n, "a field of the model type")?);
    }
    Ok(r.pos)
}

/// The numbers of `arr` from `pos` on.
struct Numbers<'a> {
    arr: &'a [f64],
    pos: usize,
}

impl<'a> Numbers<'a> {
    /// The next `n` numbers; an error if there are fewer (Julia throws a
    /// `BoundsError`).
    fn take(&mut self, n: usize, field: &str) -> Result<&'a [f64], String> {
        let end = self.pos + n;
        if end > self.arr.len() {
            return Err(format!(
                "{}: the array has {} numbers, the model needs at least {end}",
                field_name(field),
                self.arr.len()
            ));
        }
        let x = &self.arr[self.pos..end];
        self.pos = end;
        Ok(x)
    }
}

/// The field without the name of the model, for error messages.
fn field_name(field: &str) -> &str {
    field.split_once('.').map_or(field, |(_, f)| f)
}

fn field_error(field: &str, e: String) -> String {
    format!("{}: {e}", field_name(field))
}

/// A count, Julia `round(Int, x)`, which this port stores as `usize`.
fn count(x: f64) -> Result<usize, String> {
    let n = round_int(x)?;
    usize::try_from(n).map_err(|_| format!("{x} rounds to {n}, which is not a count"))
}

/// A sector or bank counted from 1 in Julia, `round(Int, x)`, as an index
/// from 0.
fn index(x: f64) -> Result<usize, String> {
    let n = round_int(x)?;
    match usize::try_from(n) {
        Ok(i) if i >= 1 => Ok(i - 1),
        _ => Err(format!("{x} rounds to {n}, which is not a number from 1")),
    }
}

/// An employer `O_h`, Julia `round(Int, x)`: a firm counted from 1 or `none`.
fn occupation(x: f64, none: i64) -> Result<Option<usize>, String> {
    let n = round_int(x)?;
    match usize::try_from(n) {
        Ok(i) if i >= 1 => Ok(Some(i - 1)),
        _ if n == none => Ok(None),
        _ => Err(format!(
            "{x} rounds to {n}, which is neither a firm from 1 nor {none}"
        )),
    }
}

/// Julia `round(Bool, x)`: `convert(Bool, round(x))`, an error unless the
/// rounded value is 0 or 1.
fn boolean(x: f64) -> Result<bool, String> {
    let r = round(x);
    if r == 0.0 {
        Ok(false)
    } else if r == 1.0 {
        Ok(true)
    } else {
        Err(format!("cannot round {x} to a Bool"))
    }
}

/// Appends a matrix, given by its rows, column by column, Julia `vec`.
fn push_columns<R: AsRef<[f64]>>(v: &mut Vec<f64>, rows: &[R]) {
    let ncols = rows.first().map_or(0, |row| row.as_ref().len());
    for j in 0..ncols {
        v.extend(rows.iter().map(|row| row.as_ref()[j]));
    }
}

/// Sets a matrix, given by its rows, column by column from the next numbers.
fn set_columns<R: AsMut<[f64]>>(
    rows: &mut [R],
    r: &mut Numbers,
    field: &str,
) -> Result<(), String> {
    let nrows = rows.len();
    let ncols = rows.first_mut().map_or(0, |row| row.as_mut().len());
    let x = r.take(nrows * ncols, field)?;
    for j in 0..ncols {
        for (i, row) in rows.iter_mut().enumerate() {
            row.as_mut()[j] = x[j * nrows + i];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{AUSTRIA2010Q1, Canvas, Model, ModelCANVAS, ModelGR, Standard, Xoshiro};

    /// The fields of the Julia structs that `_flatten_struct!` takes, in
    /// their order, from BeforeIT `src/model_init/agents.jl`,
    /// `init_properties.jl` and `src/model_extensions/`.
    const WORKERS: &str = "Y_h D_h K_h w_h O_h C_d_h I_d_h C_h I_h B_h";
    const FIRMS: &str = "G_i alpha_bar_i beta_i kappa_i w_i w_bar_i delta_i tau_Y_i tau_K_i N_i \
        Y_i Q_i Q_d_i P_i S_i K_i M_i L_i pi_bar_i D_i Pi_i V_i I_i E_i P_bar_i P_CF_i DS_i DM_i \
        DL_i DL_d_i K_e_i L_e_i Q_s_i I_d_i DM_d_i N_d_i Pi_e_i B_i Y_h C_d_h I_d_h C_h I_h K_h \
        D_h";
    const BANKS: &str = "E_k Pi_k Pi_e_k D_k r Y_h C_d_h I_d_h C_h I_h K_h D_h";
    const CENTRAL_BANK: &str = "r_bar r_G rho r_star pi_star xi_pi xi_gamma E_CB";
    const GOVERNMENT: &str = "alpha_G beta_G sigma_G Y_G C_G L_G sb_inact sb_other C_d_j C_j P_j";
    const REST_OF_THE_WORLD: &str = "alpha_E beta_E sigma_E alpha_I beta_I sigma_I Y_EA \
        gamma_EA pi_EA alpha_pi_EA beta_pi_EA sigma_pi_EA alpha_Y_EA beta_Y_EA sigma_Y_EA D_RoW \
        Y_I C_E C_d_l C_l Y_m Q_m Q_d_m P_m P_l";
    const AGGREGATES: &str = "Y pi_ P_bar P_bar_g P_bar_HH P_bar_CF P_bar_h P_bar_CF_h Y_e \
        gamma_e pi_e epsilon_Y_EA epsilon_E epsilon_I t";
    const PROPERTIES: &str = "G T_prime H_act H_inact J L I_s I H tau_INC tau_FIRM tau_VAT \
        tau_SIF tau_SIW tau_EXPORT tau_CF tau_G theta_UB psi psi_H mu theta_DIV theta zeta \
        zeta_LTV zeta_b omega_pi theta_calvo overdrafts_in_capital_ratio b_CF_g b_CFH_g b_HH_g \
        c_G_g c_E_g c_I_g a_sg C D_H K_H sb_other E_k r_bar n_banks";

    /// Each field of `model_to_array` with its first position and its
    /// number of entries, from the same list of fields.
    fn layout<X: Extension>(model: &AbstractModel<X>) -> Vec<(String, usize, usize)> {
        let mut fields = Vec::new();
        let mut pos = 0;
        macro_rules! entry {
            (added $part:ident) => {{
                let mut v = Vec::new();
                model.ext.push_fields(Part::$part, &mut v);
                fields.push((format!("added {}", stringify!($part)), pos, v.len()));
                pos += v.len();
            }};
            (occupations $e:expr, $none:expr) => {
                entry!(floats $e)
            };
            (matrix $e:expr) => {{
                let n = $e.len() * $e.first().map_or(0, |row| row.len());
                fields.push((field_name(stringify!($e)).to_string(), pos, n));
                pos += n;
            }};
            ($kind:ident $e:expr) => {{
                let n = Entries::entries(&$e);
                fields.push((field_name(stringify!($e)).to_string(), pos, n));
                pos += n;
            }};
        }
        julia_fields!(entry, model);
        assert_eq!(pos, fields.iter().map(|(_, _, n)| n).sum::<usize>());
        fields
    }

    /// Number of entries of a field.
    trait Entries {
        fn entries(&self) -> usize;
    }
    impl<T> Entries for Vec<T> {
        fn entries(&self) -> usize {
            self.len()
        }
    }
    macro_rules! scalar {
        ($($t:ty),*) => { $(impl Entries for $t { fn entries(&self) -> usize { 1 } })* };
    }
    scalar!(f64, usize, bool);

    fn positions<X: Extension>(model: &AbstractModel<X>) -> HashMap<String, (usize, usize)> {
        layout(model)
            .into_iter()
            .map(|(name, pos, n)| (name, (pos, n)))
            .collect()
    }

    fn model(seed: u64) -> Model {
        let s = &*AUSTRIA2010Q1;
        let mut rng = Xoshiro::seed_from_u64(seed);
        Model::new(&s.parameters, &s.initial_conditions, &mut rng).unwrap()
    }

    fn stepped<X: Extension>(mut m: AbstractModel<X>, seed: u64) -> AbstractModel<X> {
        let mut rng = Xoshiro::seed_from_u64(seed);
        m.step(&mut rng);
        m
    }

    fn model_gr(seed: u64) -> ModelGR {
        let s = &*AUSTRIA2010Q1;
        let mut rng = Xoshiro::seed_from_u64(seed);
        ModelGR::new(&s.parameters, &s.initial_conditions, &mut rng).unwrap()
    }

    /// A CANVAS model: the Austrian data set has no series of the policy
    /// rate and of euro area GDP and inflation, so they are made up here.
    fn model_canvas(seed: u64) -> ModelCANVAS {
        let s = &*AUSTRIA2010Q1;
        let mut ic = s.initial_conditions.clone();
        let n = s.parameters.T_prime + 4;
        ic.r_bar_series = (0..n).map(|k| 0.01 + 1e-4 * k as f64).collect();
        ic.Y_EA_series = (0..n).map(|k| 1000.0 * (1.0 + 0.004 * k as f64)).collect();
        ic.pi_EA_series = (0..n).map(|k| 0.004 + 1e-5 * k as f64).collect();
        let mut rng = Xoshiro::seed_from_u64(seed);
        ModelCANVAS::new(&s.parameters, &ic, &mut rng).unwrap()
    }

    /// The same model as model type `Y`.
    fn with_ext<X, Y>(m: &AbstractModel<X>, ext: Y) -> AbstractModel<Y> {
        AbstractModel {
            w_act: m.w_act.clone(),
            w_inact: m.w_inact.clone(),
            firms: m.firms.clone(),
            banks: m.banks.clone(),
            cb: m.cb.clone(),
            gov: m.gov.clone(),
            rotw: m.rotw.clone(),
            agg: m.agg.clone(),
            prop: m.prop.clone(),
            data: m.data.clone(),
            ext,
        }
    }

    fn same_bits(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn fields_in_the_order_of_julia() {
        let m = stepped(model(1), 2);
        let parts = [
            ("w_act", WORKERS, "WorkersActive"),
            ("w_inact", WORKERS, "WorkersInactive"),
            ("firms", FIRMS, "Firms"),
            ("banks", BANKS, "Banks"),
            ("cb", CENTRAL_BANK, "CentralBank"),
            ("gov", GOVERNMENT, "Government"),
            ("rotw", REST_OF_THE_WORLD, "RestOfTheWorld"),
            ("agg", AGGREGATES, "Aggregates"),
            ("prop", PROPERTIES, "Properties"),
        ];
        let mut expected = Vec::new();
        for (part, fields, added) in parts {
            expected.extend(fields.split_whitespace().map(|f| format!("{part}.{f}")));
            expected.push(format!("added {added}"));
        }
        let fields = layout(&m);
        let names: Vec<&String> = fields.iter().map(|(name, _, _)| name).collect();
        assert_eq!(names, expected.iter().collect::<Vec<_>>());

        // The lengths: vectors of agents, of sectors and so on.
        let (I, H_W, H_inact, G) = (m.firms.len(), m.w_act.len(), m.w_inact.len(), m.prop.G);
        let pos = positions(&m);
        assert_eq!(pos["w_act.O_h"].1, H_W);
        assert_eq!(pos["w_inact.B_h"].1, H_inact);
        assert_eq!(pos["firms.N_i"].1, I);
        assert_eq!(pos["banks.r"].1, m.prop.n_banks);
        assert_eq!(pos["gov.C_d_j"].1, m.prop.J);
        assert_eq!(pos["rotw.C_d_l"].1, m.prop.L);
        assert_eq!(pos["rotw.P_m"].1, G);
        assert_eq!(pos["agg.Y"].1, m.agg.Y.len());
        assert_eq!(pos["prop.a_sg"].1, G * G);
        assert_eq!(pos["prop.C"].1, 9);
        let n = 10 * (H_W + H_inact)
            + 45 * I
            + 12 * m.prop.n_banks
            + 8
            + (10 + m.prop.J)
            + (20 + m.prop.L + 4 * G)
            + (12 + m.agg.Y.len() + m.agg.pi_.len() + G)
            + (34 + 7 * G + G * G + 9);
        assert_eq!(model_to_array(&m).len(), n);
        let (last, p, k) = fields.last().unwrap();
        assert_eq!((last.as_str(), p + k), ("added Properties", n));
    }

    #[test]
    fn numbers_as_in_julia() {
        let m = stepped(model(1), 2);
        let arr = model_to_array(&m);
        let pos = positions(&m);
        let field = |name: &str| {
            let (p, n) = pos[name];
            &arr[p..p + n]
        };
        // Sectors and banks from 1.
        let G_i: Vec<f64> = m.firms.G_i.iter().map(|&g| (g + 1) as f64).collect();
        assert_eq!(field("firms.G_i"), G_i.as_slice());
        assert!(field("firms.G_i").iter().all(|&g| g >= 1.0));
        assert!(field("firms.B_i").iter().all(|&b| b == 1.0));
        assert!(field("w_act.B_h").iter().all(|&b| b == 1.0));
        // Employers from 1, unemployed 0, inactive −1.
        let I = m.firms.len() as f64;
        let O_h = field("w_act.O_h");
        assert!(O_h.iter().all(|&o| o == 0.0 || (1.0..=I).contains(&o)));
        assert!(O_h.contains(&0.0) && O_h.iter().any(|&o| o >= 1.0));
        for (o, x) in m.w_act.O_h.iter().zip(O_h) {
            assert_eq!(o.map_or(0.0, |i| (i + 1) as f64), *x);
        }
        assert!(field("w_inact.O_h").iter().all(|&o| o == -1.0));
        assert_eq!(field("agg.t"), [2.0]);
        assert_eq!(field("prop.overdrafts_in_capital_ratio"), [0.0]);
        // Matrices column by column.
        let G = m.prop.G;
        let a_sg = field("prop.a_sg");
        assert_eq!(a_sg[1], m.prop.a_sg[1][0]);
        assert_eq!(a_sg[G], m.prop.a_sg[0][1]);
        let C = field("prop.C");
        assert_eq!(C[1], m.prop.C[1][0]);
        assert_eq!(C[3], m.prop.C[0][1]);
    }

    #[test]
    fn round_trip() {
        // Into a model of the same shape with other numbers: the result has
        // the numbers of the first model and the data of the second.
        let a = stepped(model(1), 2);
        let b = stepped(model(3), 4);
        let arr = model_to_array(&a);
        let c = array_to_model(&arr, &b).unwrap();
        assert!(same_bits(&model_to_array(&c), &arr));
        assert_eq!(c.data, b.data);
        assert_eq!(
            AbstractModel {
                data: a.data.clone(),
                ..c
            },
            a
        );
        assert_eq!(array_to_model(&arr, &a).unwrap(), a);

        let a = stepped(model_gr(1), 2);
        let b = stepped(model_gr(3), 4);
        let arr = model_to_array(&a);
        let c = array_to_model(&arr, &b).unwrap();
        assert_eq!(c.ext, a.ext);
        assert_eq!(
            AbstractModel {
                data: a.data.clone(),
                ..c
            },
            a
        );

        let a = stepped(model_canvas(1), 2);
        let b = stepped(model_canvas(3), 4);
        let arr = model_to_array(&a);
        let c = array_to_model(&arr, &b).unwrap();
        assert_eq!(c.ext, a.ext);
        assert_eq!(
            AbstractModel {
                data: a.data.clone(),
                ..c
            },
            a
        );
    }

    #[test]
    fn added_fields_after_the_agent_types() {
        let m = stepped(model_gr(1), 2);
        let base = model_to_array(&with_ext(&m, Standard));
        let pos = positions(&with_ext(&m, Standard));
        let end = |name: &str| pos[name].0 + pos[name].1;
        let (gov, rotw) = (end("gov.P_j"), end("rotw.P_l"));
        let e = &m.ext;
        let mut expected = base[..gov].to_vec();
        expected.extend([e.alpha_G_gr, e.beta_G_gr, e.sigma_G_gr, e.g_prev_C_G]);
        expected.extend_from_slice(&base[gov..rotw]);
        expected.extend([
            e.alpha_E_gr,
            e.beta_E_gr,
            e.sigma_E_gr,
            e.alpha_I_gr,
            e.beta_I_gr,
            e.sigma_I_gr,
            e.g_prev_C_E,
            e.g_prev_Y_I,
        ]);
        expected.extend_from_slice(&base[rotw..]);
        assert!(same_bits(&model_to_array(&m), &expected));

        let m = stepped(model_canvas(1), 2);
        let base = model_to_array(&with_ext(&m, Standard));
        let (cb, rotw) = (end("cb.E_CB"), end("rotw.P_l"));
        let e: &Canvas = &m.ext;
        // One entry per quarter is added to each series.
        assert_eq!(e.Y_EA_series.len(), m.prop.T_prime + 4 + 1);
        let mut expected = base[..cb].to_vec();
        expected.extend_from_slice(&e.r_bar_series);
        expected.extend_from_slice(&base[cb..rotw]);
        expected.extend_from_slice(&e.Y_EA_series);
        expected.extend_from_slice(&e.pi_EA_series);
        expected.extend_from_slice(&base[rotw..]);
        assert!(same_bits(&model_to_array(&m), &expected));
    }

    #[test]
    fn one_property_one_number() {
        // As BeforeIT `examples/automatic_differentiation_opt.jl` finds the
        // position of each optimised property: the only number that changes
        // when the property does.
        let m = model(1);
        let arr = model_to_array(&m);
        let pos = positions(&m);
        type Set = fn(&mut crate::Properties) -> &mut f64;
        let properties: [(&str, Set); 17] = [
            ("tau_INC", |p| &mut p.tau_INC),
            ("tau_FIRM", |p| &mut p.tau_FIRM),
            ("tau_VAT", |p| &mut p.tau_VAT),
            ("tau_SIF", |p| &mut p.tau_SIF),
            ("tau_SIW", |p| &mut p.tau_SIW),
            ("tau_EXPORT", |p| &mut p.tau_EXPORT),
            ("tau_CF", |p| &mut p.tau_CF),
            ("tau_G", |p| &mut p.tau_G),
            ("theta_UB", |p| &mut p.theta_UB),
            ("psi", |p| &mut p.psi),
            ("psi_H", |p| &mut p.psi_H),
            ("mu", |p| &mut p.mu),
            ("theta_DIV", |p| &mut p.theta_DIV),
            ("theta", |p| &mut p.theta),
            ("zeta", |p| &mut p.zeta),
            ("zeta_LTV", |p| &mut p.zeta_LTV),
            ("zeta_b", |p| &mut p.zeta_b),
        ];
        for (name, set) in properties {
            let mut changed = m.clone();
            *set(&mut changed.prop) += 1.0;
            let a = model_to_array(&changed);
            let differ: Vec<usize> = (0..arr.len()).filter(|&k| a[k] != arr[k]).collect();
            assert_eq!(differ, [pos[&format!("prop.{name}")].0], "{name}");
            // And back.
            let mut x = arr.clone();
            x[differ[0]] += 1.0;
            assert_eq!(array_to_model(&x, &m).unwrap(), changed, "{name}");
        }
    }

    #[test]
    fn integers_rounded_as_julia() {
        let m = stepped(model(1), 2);
        let arr = model_to_array(&m);
        let pos = positions(&m);
        let with = |name: &str, k: usize, x: f64| {
            let mut a = arr.clone();
            a[pos[name].0 + k] = x;
            array_to_model(&a, &m)
        };
        // round(Int, x): ties to even.
        assert_eq!(with("firms.N_i", 0, 2.5).unwrap().firms.N_i[0], 2);
        assert_eq!(with("firms.N_i", 0, 3.5).unwrap().firms.N_i[0], 4);
        assert_eq!(with("firms.N_i", 0, -2.5).unwrap().firms.N_i[0], -2);
        assert_eq!(with("firms.V_i", 1, 6.7).unwrap().firms.V_i[1], 7);
        assert_eq!(with("agg.t", 0, 4.4).unwrap().agg.t, 4);
        assert_eq!(with("prop.I_s", 2, 12.5).unwrap().prop.I_s[2], 12);
        // Sectors, banks and employers from 1.
        assert_eq!(with("firms.G_i", 0, 1.5).unwrap().firms.G_i[0], 1);
        assert_eq!(with("w_act.B_h", 0, 0.6).unwrap().w_act.B_h[0], 0);
        assert_eq!(with("w_act.O_h", 0, 3.0).unwrap().w_act.O_h[0], Some(2));
        assert_eq!(with("w_act.O_h", 0, 0.4).unwrap().w_act.O_h[0], None);
        assert_eq!(with("w_inact.O_h", 0, -1.2).unwrap().w_inact.O_h[0], None);
        assert_eq!(with("w_inact.O_h", 0, 5.0).unwrap().w_inact.O_h[0], Some(4));
        // round(Bool, x).
        let flag = "prop.overdrafts_in_capital_ratio";
        assert!(!with(flag, 0, 0.5).unwrap().prop.overdrafts_in_capital_ratio);
        assert!(
            !with(flag, 0, -0.4)
                .unwrap()
                .prop
                .overdrafts_in_capital_ratio
        );
        assert!(with(flag, 0, 0.6).unwrap().prop.overdrafts_in_capital_ratio);
        assert!(with(flag, 0, 1.5).is_err());
        // Julia throws an InexactError.
        assert!(with("firms.N_i", 0, f64::NAN).is_err());
        assert!(with("firms.N_i", 0, f64::INFINITY).is_err());
        assert!(with("firms.N_i", 0, 1e19).is_err());
        // Julia stores these; the port cannot.
        let e = with("firms.G_i", 0, 0.4).unwrap_err();
        assert!(e.starts_with("firms.G_i: "), "{e}");
        assert!(with("prop.G", 0, -1.0).is_err());
        assert!(with("w_act.O_h", 0, -1.0).is_err());
        assert!(with("w_inact.O_h", 0, 0.0).is_err());
    }

    #[test]
    fn length_of_the_array() {
        let m = model(1);
        let arr = model_to_array(&m);
        let e = array_to_model(&arr[..arr.len() - 1], &m).unwrap_err();
        assert!(e.starts_with("prop.n_banks: "), "{e}");
        assert!(array_to_model(&[], &m).is_err());
        // As in Julia, numbers after those of the model are ignored.
        let mut longer = arr.clone();
        longer.extend([1.0, 2.0]);
        assert_eq!(array_to_model(&longer, &m).unwrap(), m);
        // The fields of the model type are checked too.
        let c = model_canvas(1);
        let arr = model_to_array(&c);
        let rotw = positions(&with_ext(&c, Standard))["rotw.P_l"].0 + 1;
        let cut = rotw + c.ext.r_bar_series.len() + 3;
        let e = array_to_model(&arr[..cut], &c).unwrap_err();
        assert!(e.contains("a field of the model type"), "{e}");
    }
}
