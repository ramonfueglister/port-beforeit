//! Predictions from simulations, BeforeIT
//! `src/utils/get_predictions_from_sims.jl` and `src/utils/toannual.jl`.
//!
//! [`get_predictions_from_sims`] turns the simulated paths of an ensemble
//! into forecasts in the units of the data: each simulated quarter-on-quarter
//! growth rate is applied to the last observed value, quarterly paths are
//! summed (or, for deflators, averaged) to calendar years, and growth rates
//! and deflators are derived from the levels.

use std::collections::BTreeMap;

use crate::calibration::{Array, Date, Table};
use crate::data::DataVector;
use crate::julia::{powi, sum};

/// A matrix given by its rows. For series of an ensemble, row `t` holds the
/// value of each run at time `t` (Julia's `time × runs` matrices).
pub type Matrix = Vec<Vec<f64>>;

/// BeforeIT's `predictions_dict` of [`get_predictions_from_sims`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Predictions {
    /// The series by their keys in `predictions_dict`, each `time × runs`.
    pub series: BTreeMap<String, Matrix>,
    /// `predictions_dict["quarters_num"]`: the MATLAB date numbers of the
    /// quarters of the quarterly series.
    pub quarters_num: Vec<f64>,
    /// `predictions_dict["years_num"]`: the MATLAB date numbers of the years
    /// of the yearly series (31 December).
    pub years_num: Vec<f64>,
}

impl Predictions {
    /// The predictions as a table like BeforeIT's `predictions_dict`: each
    /// series a `time × runs` matrix, `quarters_num` and `years_num`
    /// vectors.
    pub fn to_table(&self) -> Table {
        let mut t: Table = self
            .series
            .iter()
            .map(|(k, m)| (k.clone(), Array::matrix(m)))
            .collect();
        t.insert("quarters_num".into(), Array::vector(&self.quarters_num));
        t.insert("years_num".into(), Array::vector(&self.years_num));
        t
    }

    /// The predictions from a table written by [`Predictions::to_table`].
    ///
    /// Returns an error if `quarters_num` or `years_num` is missing, a
    /// series is not a matrix, or an entry is `missing`.
    pub fn from_table(t: &Table) -> Result<Predictions, String> {
        let numbers = |key: &str, a: &Array| -> Result<Vec<f64>, String> {
            a.data
                .iter()
                .map(|x| x.ok_or_else(|| format!("\"{key}\" has a missing entry")))
                .collect()
        };
        let mut p = Predictions::default();
        for (key, a) in t {
            match key.as_str() {
                "quarters_num" => p.quarters_num = numbers(key, a)?,
                "years_num" => p.years_num = numbers(key, a)?,
                _ => {
                    if a.size.len() != 2 {
                        return Err(format!("\"{key}\" is not a matrix"));
                    }
                    let (rows, cols) = (a.size[0], a.size[1]);
                    let d = numbers(key, a)?;
                    let m = (0..rows)
                        .map(|r| (0..cols).map(|c| d[c * rows + r]).collect())
                        .collect();
                    p.series.insert(key.clone(), m);
                }
            }
        }
        for key in ["quarters_num", "years_num"] {
            if !t.contains_key(key) {
                return Err(format!("the predictions have no \"{key}\""));
            }
        }
        Ok(p)
    }
}

/// The variables with real and nominal series, levels and growth rates, and
/// a deflator.
pub const PREDICTED_VARIABLES: [&str; 8] = [
    "gdp",
    "gva",
    "household_consumption",
    "government_consumption",
    "capitalformation",
    "fixed_capitalformation",
    "exports",
    "imports",
];

/// The variables predicted as levels only.
pub const PREDICTED_LEVELS_ONLY: [&str; 3] =
    ["operating_surplus", "compensation_employees", "wages"];

/// Shared part of `toannual` and `toannual_mean`: `f` of each block of four
/// columns of each row.
fn by_year(ftsa: &Matrix, f: impl Fn(&[f64]) -> f64, name: &str) -> Result<Matrix, String> {
    let cols = ftsa.first().map_or(0, |r| r.len());
    if ftsa.iter().any(|r| r.len() != cols) {
        return Err(format!("{name}: the rows have different lengths"));
    }
    if !cols.is_multiple_of(4) {
        // Julia throws a BoundsError for the last, incomplete year.
        return Err(format!(
            "{name}: {cols} columns are not whole years of four quarters"
        ));
    }
    Ok(ftsa.iter().map(|r| r.chunks(4).map(&f).collect()).collect())
}

/// `toannual(ftsa)`: the sums of blocks of four columns of each row, the
/// yearly totals of quarterly series that run along the rows.
///
/// Returns an error if the number of columns is not a multiple of 4 (Julia
/// throws a `BoundsError`).
pub fn toannual(ftsa: &Matrix) -> Result<Matrix, String> {
    by_year(ftsa, sum, "toannual")
}

/// `toannual_mean(ftsa)`: the means of blocks of four columns of each row.
///
/// Returns an error if the number of columns is not a multiple of 4 (Julia
/// throws a `BoundsError`).
pub fn toannual_mean(ftsa: &Matrix) -> Result<Matrix, String> {
    by_year(ftsa, |b| sum(b) / b.len() as f64, "toannual_mean")
}

/// `growth_rate(x; dims = 1)`: `x[t + 1] / x[t] − 1` down each column,
/// with an exact 0.0 in the denominator replaced by `eps()` (Julia's
/// `replace(x, 0.0 => eps())`, which leaves −0.0 alone).
pub fn growth_rate(x: &Matrix) -> Matrix {
    x.windows(2)
        .map(|w| {
            w[1].iter()
                .zip(&w[0])
                .map(|(&next, &prev)| {
                    let safe = if prev.to_bits() == 0.0_f64.to_bits() {
                        f64::EPSILON
                    } else {
                        prev
                    };
                    next / safe - 1.0
                })
                .collect()
        })
        .collect()
}

/// `compound_quarterly(base, growth)`: `base .* cumprod(1 .+ growth,
/// dims = 1)`, the levels reached from `base` by the growth rates in each
/// column.
pub fn compound_quarterly(base: f64, growth: &Matrix) -> Matrix {
    let mut out: Matrix = Vec::with_capacity(growth.len());
    for (t, g) in growth.iter().enumerate() {
        let row: Vec<f64> = if t == 0 {
            g.iter().map(|x| 1.0 + x).collect()
        } else {
            out[t - 1]
                .iter()
                .zip(g)
                .map(|(c, x)| c * (1.0 + x))
                .collect()
        };
        out.push(row);
    }
    for row in &mut out {
        for c in row.iter_mut() {
            *c *= base;
        }
    }
    out
}

/// Julia `d[key][d[nums] .== num]` for one value: the entry of `key` at the
/// date number `num` of `nums` (`quarters_num` or `years_num`).
fn value_at(d: &Table, key: &str, nums: &str, num: f64) -> Result<f64, String> {
    let a = d
        .get(key)
        .ok_or_else(|| format!("the data have no series \"{key}\""))?;
    let n = d
        .get(nums)
        .ok_or_else(|| format!("the data have no \"{nums}\""))?;
    if a.data.len() != n.data.len() {
        return Err(format!(
            "\"{key}\" has {} entries, \"{nums}\" {} (Julia throws a BoundsError)",
            a.data.len(),
            n.data.len()
        ));
    }
    let hits: Vec<Option<f64>> = (0..a.data.len())
        .filter(|&i| n.data[i] == Some(num))
        .map(|i| a.data[i])
        .collect();
    match hits[..] {
        [Some(x)] => Ok(x),
        [None] => Err(format!("\"{key}\" is missing at date number {num}")),
        _ => Err(format!(
            "\"{nums}\" has {} entries equal to {num}, one is needed",
            hits.len()
        )),
    }
}

/// `[repeat(value, 1, runs); rest]`.
fn with_first_row(value: f64, runs: usize, rest: Matrix) -> Matrix {
    let mut m = vec![vec![value; runs]];
    m.extend(rest);
    m
}

fn transpose(m: &Matrix, cols: usize) -> Matrix {
    (0..cols)
        .map(|j| m.iter().map(|r| r[j]).collect())
        .collect()
}

/// `quarterly_forecast[(5 - q):(end - mod(q, 4)), :]`: the quarters from
/// the first quarter of the next year to the end of the last whole year.
fn whole_years(m: &Matrix, q: u32) -> Matrix {
    let first = 4 - q as usize;
    let end = m.len().saturating_sub(q as usize % 4);
    if end > first {
        m[first..end].to_vec()
    } else {
        Vec::new()
    }
}

/// `toannual(m')'` or `toannual_mean(m')'` for a `time × runs` matrix.
fn yearly(
    m: &Matrix,
    runs: usize,
    f: fn(&Matrix) -> Result<Matrix, String>,
) -> Result<Matrix, String> {
    let years = f(&transpose(m, runs))?;
    let n = years.first().map_or(0, |r| r.len());
    Ok(transpose(&years, n))
}

/// The common inputs of the `prepare_*` functions.
struct Dates<'a> {
    data: &'a Table,
    quarter_num: f64,
    year_num: f64,
    runs: usize,
    q: u32,
}

impl Dates<'_> {
    fn quarter(&self, key: &str) -> Result<f64, String> {
        value_at(self.data, key, "quarters_num", self.quarter_num)
    }

    fn year(&self, key: &str) -> Result<f64, String> {
        value_at(self.data, key, "years_num", self.year_num)
    }
}

/// `prepare_quarterly_annual_level`: quarterly levels (the data at the
/// start date, then the forecasts) and yearly levels (the data of the
/// year of the start date, then the sums of the whole forecast years).
fn prepare_quarterly_annual_level(
    d: &Dates,
    sims: &DataVector,
    varname: &str,
) -> Result<(Matrix, Matrix), String> {
    if sims.vector[0].series_by_name(varname).is_none() {
        return Err(format!("the simulations have no series \"{varname}\""));
    }
    let var_sim = sims.series(|data| data.series_by_name(varname).unwrap_or_default());
    let growth_quarterly = growth_rate(&var_sim);
    let base = d.quarter(&format!("{varname}_quarterly"))?;
    let quarterly_forecast = compound_quarterly(base, &growth_quarterly);
    let annual = yearly(&whole_years(&quarterly_forecast, d.q), d.runs, toannual)?;
    let quarterly_full = with_first_row(base, d.runs, quarterly_forecast);
    let annual_full = with_first_row(d.year(varname)?, d.runs, annual);
    Ok((quarterly_full, annual_full))
}

/// `prepare_quarterly_annual_level_growth`: also the yearly and quarterly
/// growth rates, each after the data's growth rate at the start.
fn prepare_quarterly_annual_level_growth(
    d: &Dates,
    sims: &DataVector,
    varname: &str,
) -> Result<[Matrix; 4], String> {
    let (quarterly_full, annual_full) = prepare_quarterly_annual_level(d, sims, varname)?;
    let growth_annual_full = with_first_row(
        d.year(&format!("{varname}_growth"))?,
        d.runs,
        growth_rate(&annual_full),
    );
    let growth_quarterly_full = with_first_row(
        d.quarter(&format!("{varname}_growth_quarterly"))?,
        d.runs,
        growth_rate(&quarterly_full),
    );
    Ok([
        quarterly_full,
        annual_full,
        growth_annual_full,
        growth_quarterly_full,
    ])
}

/// `prepare_quarterly_annual_level_growth_deflator`: deflators as nominal
/// over real forecasts, yearly as the mean of the quarters, and their
/// growth rates.
fn prepare_quarterly_annual_level_growth_deflator(
    d: &Dates,
    nominal: &[Vec<f64>],
    real: &[Vec<f64>],
    varname: &str,
) -> Result<[Matrix; 4], String> {
    let deflator_quarterly: Matrix = nominal
        .iter()
        .zip(real)
        .map(|(n, r)| n.iter().zip(r).map(|(a, b)| a / b).collect())
        .collect();
    let deflator_annual = with_first_row(
        d.year(&format!("{varname}_deflator"))?,
        d.runs,
        yearly(
            &whole_years(&deflator_quarterly, d.q),
            d.runs,
            toannual_mean,
        )?,
    );
    let deflator_quarterly_full = with_first_row(
        d.quarter(&format!("{varname}_deflator_quarterly"))?,
        d.runs,
        deflator_quarterly,
    );
    let deflator_growth_annual_full = with_first_row(
        d.year(&format!("{varname}_deflator_growth"))?,
        d.runs,
        growth_rate(&deflator_annual),
    );
    let deflator_growth_quarterly_full = with_first_row(
        d.quarter(&format!("{varname}_deflator_growth_quarterly"))?,
        d.runs,
        growth_rate(&deflator_quarterly_full),
    );
    Ok([
        deflator_quarterly_full,
        deflator_annual,
        deflator_growth_annual_full,
        deflator_growth_quarterly_full,
    ])
}

/// The last day of month `m` (counted from 0 = January of year `y`, may
/// exceed 11).
fn last_day_of_month(y: i32, m: i64) -> Date {
    let y = y + m.div_euclid(12) as i32;
    let month = m.rem_euclid(12) as u32 + 1;
    let next = if month == 12 {
        Date::new(y + 1, 1, 1)
    } else {
        Date::new(y, month + 1, 1)
    };
    Date::from_datenum(next.datenum() - 1.0)
}

/// `get_predictions_from_sims(sims, real_data, start_date)`: forecasts in
/// the units of `real_data` (the calibration's `data`, e.g.
/// `ITALY_CALIBRATION.data`) from the runs in `sims`, which start at
/// `start_date`, the end of a quarter.
///
/// For every variable of [`PREDICTED_VARIABLES`], real and nominal, the
/// series `{var}_quarterly`, `{var}`, `{var}_growth` and
/// `{var}_growth_quarterly` (with `var` = `real_gdp`, `nominal_gdp`, …);
/// for [`PREDICTED_LEVELS_ONLY`] the first two; for every variable of
/// [`PREDICTED_VARIABLES`] the deflator series `{name}_deflator_quarterly`,
/// `{name}_deflator`, `{name}_deflator_growth` and
/// `{name}_deflator_growth_quarterly`; and `euribor` (yearly rate),
/// `gdp_deflator_growth_ea_quarterly` and `real_gdp_ea_quarterly` as
/// simulated. The first row of each series holds the data at the start
/// date (quarterly) or in its year (yearly). Yearly series sum or average
/// the whole calendar years after the start date.
///
/// BeforeIT's calibration overwrites `gdp_deflator_quarterly` in the data
/// it is given, and this function reads it; to obtain Julia's values after
/// a calibration, call
/// [`CalibrationData::set_gdp_deflators`](crate::CalibrationData::set_gdp_deflators)
/// on the data first.
///
/// Returns an error if a series is missing in `real_data` or in the
/// simulations, or if the forecast quarters of whole years are not a
/// multiple of four (Julia throws in these cases), and also unless exactly
/// one non-missing value of a series is found at the start date (quarterly
/// series, `quarters_num`) or at 31 December of its year (yearly series,
/// `years_num`). In that case Julia either throws (a `missing` value of
/// `{var}_quarterly` makes `toannual` fail when the forecasts contain a
/// whole calendar year; with a horizon of two or more quarters, a number of
/// `quarters_num` matches other than one or the horizon makes the broadcast
/// in `compound_quarterly` fail) or puts `missing`, several values or no
/// value at the start of the series.
pub fn get_predictions_from_sims(
    sims: &DataVector,
    real_data: &Table,
    start_date: Date,
) -> Result<Predictions, String> {
    if sims.is_empty() {
        return Err("there are no simulations".into());
    }
    let year_end = Date::new(start_date.year(), 12, 31);
    let q = (start_date.month() - 1) / 3 + 1;
    let d = Dates {
        data: real_data,
        quarter_num: start_date.datenum(),
        year_num: year_end.datenum(),
        runs: sims.len(),
        q,
    };
    let horizon = sims.vector[0].real_gdp.len().saturating_sub(1);
    let mut series = BTreeMap::new();

    for name in PREDICTED_VARIABLES {
        for version in ["real", "nominal"] {
            let var = format!("{version}_{name}");
            let [quarterly, annual, growth_annual, growth_quarterly] =
                prepare_quarterly_annual_level_growth(&d, sims, &var)?;
            series.insert(format!("{var}_quarterly"), quarterly);
            series.insert(var.clone(), annual);
            series.insert(format!("{var}_growth"), growth_annual);
            series.insert(format!("{var}_growth_quarterly"), growth_quarterly);
        }
    }

    for var in PREDICTED_LEVELS_ONLY {
        let (quarterly, annual) = prepare_quarterly_annual_level(&d, sims, var)?;
        series.insert(format!("{var}_quarterly"), quarterly);
        series.insert(var.to_string(), annual);
    }

    for name in PREDICTED_VARIABLES {
        let nominal = &series[&format!("nominal_{name}_quarterly")][1..];
        let real = &series[&format!("real_{name}_quarterly")][1..];
        let [q_, y_, gy, gq] =
            prepare_quarterly_annual_level_growth_deflator(&d, nominal, real, name)?;
        series.insert(format!("{name}_deflator_quarterly"), q_);
        series.insert(format!("{name}_deflator"), y_);
        series.insert(format!("{name}_deflator_growth"), gy);
        series.insert(format!("{name}_deflator_growth_quarterly"), gq);
    }

    // DateTime(year, month, 1) + Month(m + 1) - Day(1) for m in 0:3:3T.
    let month0 = i64::from(start_date.month()) - 1;
    let quarters_num = (0..=horizon)
        .map(|k| last_day_of_month(start_date.year(), month0 + 3 * k as i64).datenum())
        .collect();
    // DateTime(year + 1, 1, 1) + Month(month - 1) - Day(1) for month in
    // 1:12:(T / 4 * 12 + floor(q / 4)): the 31 December of the start year
    // and of the following years.
    let stop = horizon as f64 / 4.0 * 12.0 + f64::from(q / 4);
    let n_years = if stop >= 1.0 {
        ((stop - 1.0) / 12.0).floor() as i32 + 1
    } else {
        0
    };
    let years_num = (0..n_years)
        .map(|k| Date::new(start_date.year() + k, 12, 31).datenum())
        .collect();

    let euribor = sims.series(|data| &data.euribor);
    series.insert(
        "euribor".into(),
        euribor
            .iter()
            .map(|r| r.iter().map(|x| powi(1.0 + x, 4) - 1.0).collect())
            .collect(),
    );
    series.insert(
        "gdp_deflator_growth_ea_quarterly".into(),
        sims.series(|data| &data.gdp_deflator_growth_ea),
    );
    series.insert(
        "real_gdp_ea_quarterly".into(),
        sims.series(|data| &data.real_gdp_ea),
    );

    Ok(Predictions {
        series,
        quarters_num,
        years_num,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toannual_as_in_its_docstring() {
        // BeforeIT: toannual([1 2 3 4 5 6 7 8]) == [10 26].
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];
        assert_eq!(toannual(&x).unwrap(), vec![vec![10.0, 26.0]]);
        assert_eq!(toannual_mean(&x).unwrap(), vec![vec![2.5, 6.5]]);
        assert!(toannual(&vec![vec![1.0; 6]]).is_err());
    }

    #[test]
    fn growth_rates_as_in_the_julia_tests() {
        // BeforeIT test/utils/analysis_utils.jl, "growth_rate".
        let col = |v: &[f64]| v.iter().map(|&x| vec![x]).collect::<Matrix>();
        assert_eq!(growth_rate(&col(&[1.0, 2.0, 4.0])), col(&[1.0, 1.0]));
        let m = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![4.0, 40.0]];
        assert_eq!(growth_rate(&m), vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(growth_rate(&col(&[-1.0, -2.0])), col(&[1.0]));
        assert!(growth_rate(&col(&[0.0, 1.0]))[0][0].is_finite());
        // −0.0 is not replaced, as in Julia.
        assert_eq!(growth_rate(&col(&[-0.0, 1.0]))[0][0], f64::NEG_INFINITY);
    }

    #[test]
    fn last_days_of_months() {
        assert_eq!(last_day_of_month(2010, 2), Date::new(2010, 3, 31));
        assert_eq!(last_day_of_month(2010, 5), Date::new(2010, 6, 30));
        assert_eq!(last_day_of_month(2010, 13), Date::new(2011, 2, 28));
        assert_eq!(last_day_of_month(2012, 13), Date::new(2012 + 1, 2, 28));
        assert_eq!(last_day_of_month(2011, 13), Date::new(2012, 2, 29));
    }
}
