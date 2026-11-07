//! Calibration from national accounts, BeforeIT `src/utils/calibration.jl`,
//! `src/utils/_calibration_steady_state.jl` and
//! `src/utils/standard_calibration_data.jl`.
//!
//! A calibration turns tables of national accounts, input-output tables
//! (FIGARO) and time series of the country and of the euro area into
//! [`Parameters`] and [`InitialConditions`] for one quarter. The tables keep
//! the keys and the array shapes of BeforeIT's dictionaries, and the code
//! keeps Julia's 1-based time indices (`T_calibration`, ...) so that it can
//! be compared with the original line by line.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::estimate::{estimate_for_calibration_script, estimate_taylor_rule};
use crate::julia::{matlab_round, max, min, neg, pos, sum};
use crate::params::{InitialConditions, Parameters};

/// A Julia array of calibration data: its size and its entries in Julia's
/// column-major order. `None` is Julia's `missing`. A scalar has the size
/// `[]` and one entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    pub size: Vec<usize>,
    pub data: Vec<Option<f64>>,
}

/// A table of calibration data, one of BeforeIT's dictionaries
/// `Dict{String, Any}`.
pub type Table = BTreeMap<String, Array>;

/// A date, Julia's `DateTime` at midnight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    year: i32,
    month: u32,
    day: u32,
}

/// The calibration data of a country, BeforeIT `CalibrationData`.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationData {
    /// National accounts: sector accounts, balance sheets, employment and
    /// firms by industry, yearly (`years_num`) or quarterly (`quarters_num`).
    pub calibration: Table,
    /// Input-output tables (FIGARO), one per year.
    pub figaro: Table,
    /// Time series of the country.
    pub data: Table,
    /// Time series of the euro area.
    pub ea: Table,
    /// The last year whose tables a calibration uses.
    pub max_calibration_date: Date,
    /// The first quarter of the time series on which the AR(1) processes
    /// and the Taylor rule are estimated.
    pub estimation_date: Date,
}

/// The result of a calibration: BeforeIT's dictionaries `params` and
/// `initial_conditions`, and the messages that BeforeIT.jl logs with
/// `@warn`.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub parameters: Parameters,
    pub initial_conditions: InitialConditions,
    pub warnings: Vec<String>,
}

/// Lower bound of the sectoral productivities `alpha_s` and `kappa_s`,
/// `MIN_PRODUCTIVITY`: inactive sectors of small economies would otherwise
/// give NaN when the model is set up.
pub const MIN_PRODUCTIVITY: f64 = 1.0e-6;

/// The Zenodo record with BeforeIT's calibration data for the countries of
/// the euro area, `ZENODO_REPOSITORY_URL`.
pub const ZENODO_REPOSITORY_URL: &str = "https://zenodo.org/records/18698434/files/";

/// The countries in [`ZENODO_REPOSITORY_URL`], `AVAILABLE_COUNTRIES`.
pub const AVAILABLE_COUNTRIES: [&str; 27] = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT",
    "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
];

/// The address of the calibration object of `country` (a two-letter code
/// of [`AVAILABLE_COUNTRIES`]), as `download_zenodo_calibration_object`
/// builds it. The file is in Julia's JLD2 format; `data/convert_calibration.py`
/// converts it to the JSON that [`CalibrationData::from_json`] reads.
pub fn zenodo_calibration_url(country: &str) -> Result<String, String> {
    if !AVAILABLE_COUNTRIES.contains(&country) {
        return Err(format!(
            "Country {country} is not in the list of available countries. See AVAILABLE_COUNTRIES."
        ));
    }
    Ok(format!(
        "{ZENODO_REPOSITORY_URL}{country}_calibration_object.jld2"
    ))
}

/// Italy: the tables of 2010Q1 and the time series from 1996, with
/// calibrations up to 2016, BeforeIT `ITALY_CALIBRATION`
/// (`data/italy/calibration`).
pub static ITALY_CALIBRATION: LazyLock<CalibrationData> = LazyLock::new(|| {
    CalibrationData::from_json(include_str!("../data/italy/calibration.json"))
        .unwrap_or_else(|e| panic!("embedded calibration data: {e}"))
});

impl Date {
    /// The date `year`-`month`-`day` of the proleptic Gregorian calendar,
    /// Julia `DateTime(year, month, day)`.
    ///
    /// # Panics
    ///
    /// If the month or the day does not exist (Julia throws an
    /// `ArgumentError`).
    pub fn new(year: i32, month: u32, day: u32) -> Date {
        assert!(
            is_valid(year, month, day),
            "{year:04}-{month:02}-{day:02} is not a date"
        );
        Date { year, month, day }
    }

    /// Reads `YYYY-MM-DD`.
    pub fn parse(text: &str) -> Result<Date, String> {
        let bad = || format!("{text} is not a date YYYY-MM-DD");
        let mut parts = text.splitn(3, '-');
        let mut next = || parts.next().ok_or_else(bad);
        let (y, m, d) = (next()?, next()?, next()?);
        let (year, month, day) = (
            y.parse().map_err(|_| bad())?,
            m.parse().map_err(|_| bad())?,
            d.parse().map_err(|_| bad())?,
        );
        if !is_valid(year, month, day) {
            return Err(bad());
        }
        Ok(Date { year, month, day })
    }

    pub fn year(self) -> i32 {
        self.year
    }

    pub fn month(self) -> u32 {
        self.month
    }

    pub fn day(self) -> u32 {
        self.day
    }

    /// The MATLAB date number, `date2num`: days since the year 0, with
    /// 0000-01-01 as day 1.
    pub fn datenum(self) -> f64 {
        (days_from_civil(self.year, self.month, self.day) + 719_529) as f64
    }

    /// The date of a MATLAB date number, `num2date`. Julia rounds to
    /// milliseconds; the time of day is dropped here.
    pub fn from_datenum(n: f64) -> Date {
        let ms = (n * 86_400_000.0).round();
        let days = (ms / 86_400_000.0).floor() as i64 - 719_529;
        let (year, month, day) = civil_from_days(days);
        Date { year, month, day }
    }
}

/// Whether the month and the day exist in the proleptic Gregorian calendar.
fn is_valid(year: i32, month: u32, day: u32) -> bool {
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days).contains(&day)
}

/// Days since 1970-01-01 in the proleptic Gregorian calendar (H. Hinnant's
/// algorithm `days_from_civil`).
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = i64::from(y) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of [`days_from_civil`].
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y as i32, m, d)
}

impl Array {
    /// Julia `size(A, d)` for `d` counted from 1: 1 beyond the dimensions.
    pub fn size(&self, d: usize) -> usize {
        self.size.get(d - 1).copied().unwrap_or(1)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl CalibrationData {
    /// Reads calibration data in the format of `data/convert_calibration.py`:
    /// a JSON object with the tables `calibration`, `figaro`, `data` and
    /// `ea` and the dates `max_calibration_date` and `estimation_date`
    /// (`YYYY-MM-DD`). A value of a table is a number or
    /// `{"size": [...], "data": [...]}` with the entries in Julia's
    /// column-major order: numbers, `null` for `missing`, or "NaN", "Inf"
    /// and "-Inf".
    pub fn from_json(text: &str) -> Result<CalibrationData, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("calibration data: {e}"))?;
        let o = v.as_object().ok_or("calibration data: not a JSON object")?;
        let get = |key: &str| o.get(key).ok_or_else(|| format!("{key} is missing"));
        let table = |key: &str| -> Result<Table, String> {
            let t = get(key)?
                .as_object()
                .ok_or_else(|| format!("{key} is not a table"))?;
            t.iter()
                .map(|(k, v)| Ok((k.clone(), array(v, &format!("{key}[\"{k}\"]"))?)))
                .collect()
        };
        let date = |key: &str| -> Result<Date, String> {
            Date::parse(
                get(key)?
                    .as_str()
                    .ok_or_else(|| format!("{key} is not a date"))?,
            )
        };
        Ok(CalibrationData {
            calibration: table("calibration")?,
            figaro: table("figaro")?,
            data: table("data")?,
            ea: table("ea")?,
            max_calibration_date: date("max_calibration_date")?,
            estimation_date: date("estimation_date")?,
        })
    }
}

fn array(v: &serde_json::Value, what: &str) -> Result<Array, String> {
    let entry = |x: &serde_json::Value| -> Result<Option<f64>, String> {
        match x {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::String(s) => match s.as_str() {
                "NaN" => Ok(Some(f64::NAN)),
                "Inf" => Ok(Some(f64::INFINITY)),
                "-Inf" => Ok(Some(f64::NEG_INFINITY)),
                _ => Err(format!("{what}: {s} is not a number")),
            },
            _ => x
                .as_f64()
                .map(Some)
                .ok_or_else(|| format!("{what}: {x} is not a number")),
        }
    };
    if let serde_json::Value::Object(o) = v {
        let size: Vec<usize> = o
            .get("size")
            .and_then(|s| s.as_array())
            .ok_or_else(|| format!("{what} has no size"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("{what}: invalid size"))?;
        let data: Vec<Option<f64>> = o
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| format!("{what} has no data"))?
            .iter()
            .map(entry)
            .collect::<Result<_, _>>()?;
        if size.iter().product::<usize>() != data.len() {
            return Err(format!("{what}: size and number of entries differ"));
        }
        Ok(Array { size, data })
    } else {
        Ok(Array {
            size: Vec::new(),
            data: vec![entry(v)?],
        })
    }
}

/// A table with its name, for error messages; indices count from 1 as in
/// Julia.
#[derive(Clone, Copy)]
struct Tab<'a> {
    name: &'static str,
    t: &'a Table,
}

impl<'a> Tab<'a> {
    fn has(&self, key: &str) -> bool {
        self.t.contains_key(key)
    }

    /// Julia `d[key]`, a `KeyError` for a missing key.
    fn get(&self, key: &str) -> Result<&'a Array, String> {
        self.t
            .get(key)
            .ok_or_else(|| format!("{}[\"{key}\"] is missing", self.name))
    }

    /// Entry `i` (linear index from 1). Where Julia would compute with
    /// `missing` and fail later, this is an error.
    fn num(&self, key: &str, i: usize) -> Result<f64, String> {
        let a = self.get(key)?;
        match i.checked_sub(1).and_then(|k| a.data.get(k)) {
            None => Err(format!(
                "{}[\"{key}\"] has {} entries, index {i}",
                self.name,
                a.len()
            )),
            Some(None) => Err(format!("{}[\"{key}\"][{i}] is missing", self.name)),
            Some(Some(x)) => Ok(*x),
        }
    }

    /// Julia `d[key][a:b]`.
    fn range(&self, key: &str, a: usize, b: usize) -> Result<Vec<f64>, String> {
        (a..=b).map(|i| self.num(key, i)).collect()
    }

    /// Julia `d[key][:, t]`.
    fn column(&self, key: &str, t: usize) -> Result<Vec<f64>, String> {
        let rows = self.get(key)?.size(1);
        let start = (t - 1) * rows;
        self.range(key, start + 1, start + rows)
    }

    /// Julia `d[key][:, :, t]`, as rows.
    fn slab(&self, key: &str, t: usize) -> Result<Vec<Vec<f64>>, String> {
        let a = self.get(key)?;
        let (r, c) = (a.size(1), a.size(2));
        let start = (t - 1) * r * c;
        (0..r)
            .map(|i| {
                (0..c)
                    .map(|j| self.num(key, start + j * r + i + 1))
                    .collect()
            })
            .collect()
    }

    /// Julia `findfirst(==(x), vec(d[key]))`.
    fn find(&self, key: &str, x: f64, what: Date) -> Result<usize, String> {
        self.get(key)?
            .data
            .iter()
            .position(|&v| v == Some(x))
            .map(|k| k + 1)
            .ok_or_else(|| {
                format!(
                    "{}-{:02}-{:02} is not in {}[\"{key}\"]",
                    what.year, what.month, what.day, self.name
                )
            })
    }

    /// Julia `findall(d[key] .== x)[1]`: the Cartesian index of the first
    /// match, one index per dimension.
    fn find_cartesian(&self, key: &str, x: f64, what: Date) -> Result<Vec<usize>, String> {
        let a = self.get(key)?;
        let mut k = self.find(key, x, what)? - 1;
        let mut index = Vec::new();
        for &n in &a.size {
            index.push(k % n + 1);
            k /= n;
        }
        Ok(index)
    }

    /// `has_quarterly(d, key, idx)`: `d[key]` exists, has an entry `idx`,
    /// and that entry is not `missing`.
    fn has_quarterly(&self, key: &str, idx: usize) -> bool {
        self.t
            .get(key)
            .is_some_and(|a| idx >= 1 && a.data.get(idx - 1).is_some_and(|x| x.is_some()))
    }

    /// A scalar value, `d[key]` for a number.
    fn scalar(&self, key: &str) -> Result<f64, String> {
        if self.get(key)?.len() != 1 {
            return Err(format!("{}[\"{key}\"] is not a number", self.name));
        }
        self.num(key, 1)
    }
}

/// The GDP deflator of the series of a table, `nominal_gdp_quarterly ./
/// real_gdp_quarterly`, at index `i`. BeforeIT.jl stores the whole ratio as
/// `gdp_deflator_quarterly` in the table; the port computes it where it is
/// needed and leaves the data unchanged.
fn deflator(t: Tab, i: usize) -> Result<f64, String> {
    Ok(t.num("nominal_gdp_quarterly", i)? / t.num("real_gdp_quarterly", i)?)
}

/// `diff(log.(x))`.
fn diff_log(x: &[f64]) -> Vec<f64> {
    x.windows(2)
        .map(|w| libm::log(w[1]) - libm::log(w[0]))
        .collect()
}

/// Julia `replace!(v, NaN => 0.0)`.
fn nan_to_zero(v: &mut [f64]) {
    for x in v.iter_mut() {
        if x.is_nan() {
            *x = 0.0;
        }
    }
}

/// `sum(A, dims = 1)'` of a matrix given as rows: the column sums.
fn column_sums(A: &[Vec<f64>]) -> Vec<f64> {
    let c = A.first().map_or(0, |r| r.len());
    (0..c)
        .map(|j| sum(&A.iter().map(|r| r[j]).collect::<Vec<_>>()))
        .collect()
}

/// `sum(A, dims = 2)`: the row sums, added from left to right as Julia
/// reduces along the second dimension.
fn row_sums(A: &[Vec<f64>]) -> Vec<f64> {
    A.iter().map(|r| r.iter().fold(0.0, |s, x| s + x)).collect()
}

/// Statistics.jl `cov([a b c])`: the covariance matrix of three series with
/// the divisor `n - 1`.
fn cov3(a: &[f64], b: &[f64], c: &[f64]) -> [[f64; 3]; 3] {
    let n = a.len();
    let cols = [a, b, c];
    let centred: Vec<Vec<f64>> = cols
        .iter()
        .map(|x| {
            let m = sum(x) / n as f64;
            x.iter().map(|v| v - m).collect()
        })
        .collect();
    let mut C = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let s = (0..n).fold(0.0, |s, k| s + centred[i][k] * centred[j][k]);
            C[i][j] = s / (n - 1) as f64;
        }
    }
    C
}

/// A count stored as a float, Julia `Int(x)` when the model is set up.
fn count(x: f64, what: &str) -> Result<usize, String> {
    if x >= 0.0 && x.fract() == 0.0 && x < 2f64.powi(53) {
        Ok(x as usize)
    } else {
        Err(format!("{what} = {x} is not a count"))
    }
}

fn counts(v: &[f64], what: &str) -> Result<Vec<usize>, String> {
    v.iter().map(|&x| count(x, what)).collect()
}

/// `(x .+ 1.0) .^ (1.0 / 4.0) .- 1`: a yearly rate as a quarterly one.
fn quarterly_rate(x: f64) -> f64 {
    libm::pow(x + 1.0, 1.0 / 4.0) - 1.0
}

/// The calendar date of quarter `q` of year `y`, as BeforeIT forms it:
/// the last day of the quarter.
fn quarter_end(y: i32, q: u32) -> Date {
    let month = q * 3;
    Date::new(y, month, if month == 3 || month == 12 { 31 } else { 30 })
}

/// `get_valid_calibration_quarters`: the quarters `(year, quarter)` for
/// which `calibration_object` has the data that a calibration needs: after
/// the estimation date, within the quarterly and yearly series, with the
/// key quarterly and yearly variables not `missing`, and in the time series
/// of the country.
pub fn get_valid_calibration_quarters(
    calibration_object: &CalibrationData,
) -> Result<Vec<(i32, u32)>, String> {
    let cd = Tab {
        name: "calibration",
        t: &calibration_object.calibration,
    };
    let data = Tab {
        name: "data",
        t: &calibration_object.data,
    };
    let numbers = |key: &str| -> Result<Vec<f64>, String> {
        Ok(cd.get(key)?.data.iter().flatten().copied().collect())
    };
    let (years_num, quarters_num) = (numbers("years_num")?, numbers("quarters_num")?);
    let minimum = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, min);
    let maximum = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, max);
    let min_figaro_year = Date::from_datenum(minimum(&years_num)).year;
    let max_figaro_year = Date::from_datenum(maximum(&years_num)).year;
    let (min_quarter, max_quarter) = (minimum(&quarters_num), maximum(&quarters_num));
    let mut valid = Vec::new();
    for year in min_figaro_year..=max_figaro_year {
        for quarter in 1..=4 {
            let calibration_date = quarter_end(year, quarter);
            if calibration_date <= calibration_object.estimation_date {
                continue;
            }
            let n = calibration_date.datenum();
            if n < min_quarter || n > max_quarter {
                continue;
            }
            let Some(T_calibration_quarterly) =
                quarters_num.iter().position(|&x| x == n).map(|k| k + 1)
            else {
                continue;
            };
            let capped = calibration_date.min(calibration_object.max_calibration_date);
            let year_end = Date::new(capped.year, 12, 31).datenum();
            let Some(T_calibration) = years_num.iter().position(|&x| x == year_end).map(|k| k + 1)
            else {
                continue;
            };
            let present = |key: &str, idx: usize| cd.has_quarterly(key, idx);
            let quarterly = [
                "household_cash_quarterly",
                "firm_cash_quarterly",
                "firm_debt_quarterly",
                "government_debt_quarterly",
                "bank_equity_quarterly",
            ];
            let annual = ["property_income", "mixed_income", "corporate_tax"];
            if !quarterly
                .iter()
                .all(|key| present(key, T_calibration_quarterly))
                || !annual.iter().all(|key| present(key, T_calibration))
            {
                continue;
            }
            let in_data = data.get("quarters_num")?.data.contains(&Some(n));
            if in_data {
                valid.push((year, quarter));
            }
        }
    }
    Ok(valid)
}

/// `get_params_and_initial_conditions(calibration_object, calibration_date;
/// scale, use_growth_rate_ar1)`: parameters and initial conditions for the
/// quarter that ends on `calibration_date`. `scale` is the number of agents
/// per firm and per person of the country (BeforeIT's default 0.001). With
/// `use_growth_rate_ar1` (default false), the AR(1) processes of government
/// consumption, exports and imports are estimated on growth rates, and the
/// last growth rates are stored as `g_G`, `g_E` and `g_I`; the processes of
/// the euro area always use logarithms.
///
/// Both formats of BeforeIT.jl's calibration data are read: that of
/// [`ITALY_CALIBRATION`] and that of the Zenodo record (wages, fixed assets
/// and capital consumption by industry, quarterly interest and deficit
/// series, census counts of unemployed and inactive persons). Where Julia
/// would carry a `missing` value into the result, or fails because a date
/// is not in the data, this returns an error.
pub fn get_params_and_initial_conditions(
    calibration_object: &CalibrationData,
    calibration_date: Date,
    scale: f64,
    use_growth_rate_ar1: bool,
) -> Result<Calibration, String> {
    let co = calibration_object;
    let cd = Tab {
        name: "calibration",
        t: &co.calibration,
    };
    let figaro = Tab {
        name: "figaro",
        t: &co.figaro,
    };
    let data = Tab {
        name: "data",
        t: &co.data,
    };
    let ea = Tab {
        name: "ea",
        t: &co.ea,
    };
    let mut warnings = Vec::new();

    // ========================= Resolve time indices =========================
    let capped = calibration_date.min(co.max_calibration_date);
    let year_end = Date::new(capped.year, 12, 31);
    let T_calibration = cd.find("years_num", year_end.datenum(), year_end)?;
    let T_calibration_quarterly =
        cd.find("quarters_num", calibration_date.datenum(), calibration_date)?;
    let T_estimation_exo = data.find(
        "quarters_num",
        co.estimation_date.datenum(),
        co.estimation_date,
    )?;
    let T_calibration_exo =
        data.find("quarters_num", calibration_date.datenum(), calibration_date)?;
    let T_calibration_exo_max = data.get("quarters_num")?.len();
    let (Tc, Tq, Te, Tx) = (
        T_calibration,
        T_calibration_quarterly,
        T_estimation_exo,
        T_calibration_exo,
    );
    if Te < 2 {
        return Err("estimation_date is the first quarter of data[\"quarters_num\"]".into());
    }

    // ================= Extract raw FIGARO / calibration data =================
    let intermediate_consumption = figaro.slab("intermediate_consumption", Tc)?;
    let G = intermediate_consumption.len();
    let S = G;
    let household_consumption = figaro.column("household_consumption", Tc)?;
    let fixed_capitalformation = figaro.column("fixed_capitalformation", Tc)?;
    let exports = figaro.column("exports", Tc)?;
    let compensation_employees = figaro.column("compensation_employees", Tc)?;
    let household_cash_quarterly = cd.num("household_cash_quarterly", Tq)?;
    let property_income = cd.num("property_income", Tc)?;
    let mixed_income = cd.num("mixed_income", Tc)?;
    let operating_surplus = figaro.column("operating_surplus", Tc)?;
    let firm_cash_quarterly = cd.num("firm_cash_quarterly", Tq)?;
    let mut firm_debt_quarterly = cd.num("firm_debt_quarterly", Tq)?;

    // Consolidation ratio: intra-group lending is not firm debt.
    if cd.has_quarterly("firm_debt_consolidation_ratio_quarterly", Tq) {
        firm_debt_quarterly *= cd.num("firm_debt_consolidation_ratio_quarterly", Tq)?;
    }

    // Quarterly interest series where they exist, otherwise yearly ones.
    let has_quarterly_firm_interest = cd.has_quarterly("firm_interest_quarterly", Tq);
    let has_quarterly_govt_interest = cd.has_quarterly("interest_government_debt_quarterly", Tq);
    let mut firm_interest_quarterly = if has_quarterly_firm_interest {
        cd.num("firm_interest_quarterly", Tq)?
    } else {
        cd.num("firm_interest", Tc)?
    };
    let government_debt_quarterly = cd.num("government_debt_quarterly", Tq)?;
    let mut interest_government_debt_quarterly = if has_quarterly_govt_interest {
        cd.num("interest_government_debt_quarterly", Tq)?
    } else {
        warnings.push(
            "Using annual 'interest_government_debt' - will apply timescale conversion".into(),
        );
        cd.num("interest_government_debt", Tc)?
    };

    let government_consumption = figaro.column("government_consumption", Tc)?;
    let social_benefits = cd.num("social_benefits", Tc)?;
    let unemployment_benefits = cd.num("unemployment_benefits", Tc)?;
    let pension_benefits = cd.num("pension_benefits", Tc)?;
    let corporate_tax = cd.num("corporate_tax", Tc)?;

    // Wages by industry where they exist, otherwise the total wages.
    let has_sectoral_wages = cd.has("wages_by_sector") && cd.get("wages_by_sector")?.size(2) >= Tc;
    let (wages_by_sector, wages_scalar) = if has_sectoral_wages {
        (cd.column("wages_by_sector", Tc)?, f64::NAN)
    } else {
        warnings.push("wages_by_sector not available, estimating from aggregate wages".into());
        // BeforeIT.jl also spreads the total over the industries here; that
        // estimate is not used further.
        (Vec::new(), cd.num("wages", Tc)?)
    };

    let taxes_products_household = figaro.num("taxes_products_household", Tc)?;
    let social_contributions = cd.num("social_contributions", Tc)?;
    let income_tax = cd.num("income_tax", Tc)?;
    let capital_taxes = cd.num("capital_taxes", Tc)?;
    let taxes_products_fixed_capitalformation =
        figaro.num("taxes_products_capitalformation", Tc)?;
    let taxes_production = figaro.column("taxes_production", Tc)?;
    let taxes_products_government = figaro.num("taxes_products_government", Tc)?;
    let bank_equity_quarterly = cd.num("bank_equity_quarterly", Tq)?;
    figaro.column("taxes_products", Tc)?;
    // As the MATLAB code, BeforeIT.jl sets the taxes on products to zero, so
    // that GDP equals the sum of its uses in the model.
    let taxes_products = vec![0.0; G];

    let has_quarterly_govt_deficit = cd.has_quarterly("government_deficit_quarterly", Tq);
    let government_deficit_quarterly = if has_quarterly_govt_deficit {
        cd.num("government_deficit_quarterly", Tq)?
    } else {
        warnings
            .push("Using annual 'government_deficit' - will NOT apply timescale conversion".into());
        f64::NAN
    };
    let government_deficit_annual = cd.num("government_deficit", Tc)?;

    let mut firms = cd.column("firms", Tc)?;
    let mut employees = cd.column("employees", Tc)?;
    let population = cd.num("population", Tc)?;
    let r_bar = quarterly_rate(data.num("euribor", Tx)?);

    let omega = 0.85;

    // ===================== Derive accounting identities ======================
    let intermediate_consumption: Vec<Vec<f64>> = intermediate_consumption
        .iter()
        .map(|r| r.iter().map(|&x| max(0.0, x)).collect())
        .collect();
    let ic_columns = column_sums(&intermediate_consumption);
    // Output = intermediate consumption + value added.
    let output: Vec<f64> = (0..G)
        .map(|i| {
            ic_columns[i]
                + taxes_products[i]
                + taxes_production[i]
                + compensation_employees[i]
                + operating_surplus[i]
        })
        .collect();

    // Julia parses `size(fa)[1] == G & size(dw)[1] == G` as the chained
    // comparison `size(fa)[1] == (G & size(dw)[1]) == G`.
    let (fa1, dw1) = (
        cd.get("fixed_assets")?.size(1),
        cd.get("dwellings")?.size(1),
    );
    let (fixed_assets_other_than_dwellings, dwellings) = if fa1 == (G & dw1) && (G & dw1) == G {
        let fixed_assets = cd.column("fixed_assets", Tc)?;
        let dwellings = cd.column("dwellings", Tc)?;
        let other: Vec<f64> = (0..G).map(|i| fixed_assets[i] - dwellings[i]).collect();
        (other, dwellings)
    } else {
        let fixed_assets = cd.num("fixed_assets", Tc)?;
        let dwellings = cd.num("dwellings", Tc)?;
        let fixed_assets_eu7 = cd.column("fixed_assets_eu7", Tc)?;
        let dwellings_eu7 = cd.column("dwellings_eu7", Tc)?;
        let nominal_nace64_output_eu7 = cd.column("nominal_nace64_output_eu7", Tc)?;
        let w: Vec<f64> = (0..G)
            .map(|i| {
                (fixed_assets_eu7[i] - dwellings_eu7[i]) / nominal_nace64_output_eu7[i] * output[i]
            })
            .collect();
        let total = sum(&w);
        let other = w
            .iter()
            .map(|wi| (fixed_assets - dwellings) * wi / total)
            .collect();
        (other, vec![dwellings])
    };

    let capital_consumption: Vec<f64> =
        if cd.get("capital_consumption")?.size(1) == compensation_employees.len() {
            cd.column("capital_consumption", Tc)?
        } else {
            let nace64_capital_consumption = cd.column("nace64_capital_consumption", Tc)?;
            let nominal_nace64_output = cd.column("nominal_nace64_output", Tc)?;
            (0..G)
                .map(|i| nace64_capital_consumption[i] / nominal_nace64_output[i] * output[i])
                .collect()
        };

    let unemployment_rate_quarterly = data.num("unemployment_rate_quarterly", Tx)?;
    let taxes_products_export = 0.0;
    // Employers' social contributions = compensation - wages.
    let employers_social_contributions: Vec<f64> = if has_sectoral_wages {
        (0..G)
            .map(|i| compensation_employees[i] - wages_by_sector[i])
            .collect()
    } else {
        let total_compensation = sum(&compensation_employees);
        let scalar = min(social_contributions, total_compensation - wages_scalar);
        compensation_employees
            .iter()
            .map(|c| scalar * (c / total_compensation))
            .collect()
    };
    let fixed_capitalformation: Vec<f64> = fixed_capitalformation.iter().map(|&x| pos(x)).collect();
    let gross_capitalformation_dwellings = cd.num("gross_capitalformation_dwellings", Tc)?;
    let taxes_products_capitalformation_dwellings = gross_capitalformation_dwellings
        * (1.0
            - 1.0 / (1.0 + taxes_products_fixed_capitalformation / sum(&fixed_capitalformation)));
    // Timescale: quarterly GDP over the yearly income components of GDP.
    let income: Vec<f64> = (0..G)
        .map(|i| {
            compensation_employees[i]
                + operating_surplus[i]
                + taxes_production[i]
                + taxes_products[i]
        })
        .collect();
    let timescale = data.num("nominal_gdp_quarterly", Tx)?
        / (sum(&income)
            + taxes_products_household
            + taxes_products_capitalformation_dwellings
            + taxes_products_government
            + taxes_products_export);

    if !has_quarterly_firm_interest {
        firm_interest_quarterly *= timescale;
    }
    if !has_quarterly_govt_interest {
        interest_government_debt_quarterly *= timescale;
    }

    let fcf_total = sum(&fixed_capitalformation);
    let capitalformation_dwellings: Vec<f64> = fixed_capitalformation
        .iter()
        .map(|f| {
            (gross_capitalformation_dwellings - taxes_products_capitalformation_dwellings) * f
                / fcf_total
        })
        .collect();
    let fixed_capital_formation_other_than_dwellings: Vec<f64> = (0..G)
        .map(|i| fixed_capitalformation[i] - capitalformation_dwellings[i])
        .collect();
    let exports: Vec<f64> = exports.iter().map(|&x| pos(x)).collect();
    // Imports as the residual of the goods balance; its negative part is
    // re-exports.
    let ic_rows = row_sums(&intermediate_consumption);
    let (cc_total, fcfo_total) = (
        sum(&capital_consumption),
        sum(&fixed_capital_formation_other_than_dwellings),
    );
    let net_imports: Vec<f64> = (0..G)
        .map(|i| {
            ic_rows[i]
                + household_consumption[i]
                + government_consumption[i]
                + fixed_capital_formation_other_than_dwellings[i] * cc_total / fcfo_total
                + capitalformation_dwellings[i]
                + exports[i]
                - output[i]
        })
        .collect();
    let imports: Vec<f64> = net_imports.iter().map(|&x| pos(x)).collect();
    let reexports: Vec<f64> = net_imports.iter().map(|&x| neg(x)).collect();
    let esc_total = sum(&employers_social_contributions);
    let household_social_contributions = social_contributions - esc_total;
    let wages: Vec<f64> = if has_sectoral_wages {
        wages_by_sector
    } else {
        let total_compensation = sum(&compensation_employees);
        compensation_employees
            .iter()
            .map(|c| c * (1.0 - esc_total / total_compensation))
            .collect()
    };
    let household_income_tax = income_tax - corporate_tax;
    let govt_deficit_term = if has_quarterly_govt_deficit {
        government_deficit_quarterly / timescale
    } else {
        government_deficit_annual
    };
    let other_net_transfers = taxes_products_household
        + taxes_products_capitalformation_dwellings
        + taxes_products_export
        + sum(&taxes_products)
        + sum(&taxes_production)
        + esc_total
        + household_social_contributions
        + household_income_tax
        + corporate_tax
        + capital_taxes
        - social_benefits
        - sum(&government_consumption)
        - interest_government_debt_quarterly / timescale
        - govt_deficit_term;
    let wages_total = sum(&wages);
    let disposable_income =
        wages_total + mixed_income + property_income + social_benefits + other_net_transfers
            - household_social_contributions
            - household_income_tax
            - capital_taxes;
    let unemployed = if cd.has("unemployed_census") {
        cd.scalar("unemployed_census")?
    } else {
        matlab_round(
            (unemployment_rate_quarterly * sum(&employees)) / (1.0 - unemployment_rate_quarterly),
        )
    };
    let inactive = if cd.has("inactive_census") {
        cd.scalar("inactive_census")?
    } else {
        let at_least_one: Vec<f64> = firms.iter().map(|&f| max(1.0, f)).collect();
        let persons: Vec<f64> = (0..G).map(|i| max(at_least_one[i], employees[i])).collect();
        population - sum(&persons) - unemployed - sum(&at_least_one) - 1.0
    };

    // Scale the numbers of firms and persons.
    for f in firms.iter_mut() {
        *f = max(1.0, matlab_round(scale * *f));
    }
    for (e, f) in employees.iter_mut().zip(&firms) {
        *e = max(*f, matlab_round(scale * *e));
    }
    let inactive = max(1.0, matlab_round(scale * inactive));
    let unemployed = max(1.0, matlab_round(scale * unemployed));

    // ================= Sector parameters & tax/policy rates ==================
    let I_s = firms.clone();
    let by_sector = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { (0..G).map(f).collect() };
    let mut alpha_s = by_sector(&|i| timescale * output[i] / employees[i]);
    let mut beta_s = by_sector(&|i| output[i] / ic_columns[i]);
    let mut kappa_s =
        by_sector(&|i| timescale * output[i] / fixed_assets_other_than_dwellings[i] / omega);
    let mut delta_s = by_sector(&|i| {
        timescale * capital_consumption[i] / fixed_assets_other_than_dwellings[i] / omega
    });
    nan_to_zero(&mut delta_s);
    let mut w_s = by_sector(&|i| timescale * wages[i] / employees[i]);
    nan_to_zero(&mut alpha_s);
    nan_to_zero(&mut kappa_s);
    nan_to_zero(&mut w_s);
    nan_to_zero(&mut beta_s);
    for x in alpha_s.iter_mut().chain(kappa_s.iter_mut()) {
        *x = max(*x, MIN_PRODUCTIVITY);
    }
    let mut tau_Y_s = by_sector(&|i| taxes_products[i] / output[i]);
    let mut tau_K_s = by_sector(&|i| taxes_production[i] / output[i]);
    nan_to_zero(&mut tau_Y_s);
    nan_to_zero(&mut tau_K_s);
    let shares = |v: &[f64]| -> Vec<f64> {
        let total = sum(v);
        v.iter().map(|x| x / total).collect()
    };
    let b_CF_g = shares(&fixed_capital_formation_other_than_dwellings);
    let b_CFH_g = shares(&capitalformation_dwellings);
    let b_HH_g = shares(&household_consumption);
    let mut a_sg: Vec<Vec<f64>> = intermediate_consumption
        .iter()
        .map(|r| (0..G).map(|j| r[j] / ic_columns[j]).collect())
        .collect();
    for r in a_sg.iter_mut() {
        nan_to_zero(r);
    }
    let c_G_g = shares(&government_consumption);
    let net_exports: Vec<f64> = (0..G).map(|i| exports[i] - reexports[i]).collect();
    let c_E_g = shares(&net_exports);
    let c_I_g = shares(&imports);

    // Parameters.
    let T_prime = Tx - Te + 1;
    let T = 12;
    let T_max = T as i64 - 0.max(Tx as i64 + T as i64 - T_calibration_exo_max as i64);
    let firms_total = sum(&firms);
    let H_act = sum(&employees) + unemployed + firms_total + 1.0;
    let H_inact = inactive;
    let J = matlab_round(firms_total / 4.0);
    let L = matlab_round(firms_total / 2.0);
    let mu = firm_interest_quarterly / firm_debt_quarterly - r_bar;
    let tau_INC = (household_income_tax + capital_taxes)
        / (wages_total + property_income + mixed_income - household_social_contributions);
    let tau_SIF = esc_total / wages_total;

    // Profits by industry as the model computes them (pi_bar_s × Y_s), which
    // allocate the firms' deposits.
    let Y_s = by_sector(&|i| timescale * output[i]);
    let mut pi_bar_s = by_sector(&|i| {
        1.0 - (1.0 + tau_SIF) * w_s[i] / alpha_s[i]
            - delta_s[i] / kappa_s[i]
            - 1.0 / beta_s[i]
            - tau_K_s[i]
            - tau_Y_s[i]
    });
    for x in pi_bar_s.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    let model_profit_s = by_sector(&|i| pi_bar_s[i] * Y_s[i]);
    let positive_profit: Vec<f64> = model_profit_s.iter().map(|&x| pos(x)).collect();
    let pos_model_profit_sum = sum(&positive_profit);
    let D_s_alloc: Vec<f64> = if pos_model_profit_sum > 0.0 {
        positive_profit
            .iter()
            .map(|p| p / pos_model_profit_sum)
            .collect()
    } else {
        vec![0.0; G]
    };
    let L_s_alloc = shares(&fixed_assets_other_than_dwellings);
    let profit_after_interest_s: Vec<f64> = by_sector(&|i| {
        model_profit_s[i] - (r_bar + mu) * firm_debt_quarterly * L_s_alloc[i]
            + r_bar * firm_cash_quarterly * D_s_alloc[i]
    });
    let positive_after_interest = sum(&profit_after_interest_s
        .iter()
        .map(|&x| pos(x))
        .collect::<Vec<_>>());
    let bank_lending =
        firm_interest_quarterly - r_bar * (firm_debt_quarterly - bank_equity_quarterly);
    let tau_FIRM = timescale * corporate_tax / (positive_after_interest + bank_lending);
    let hc_total = sum(&household_consumption);
    let tau_VAT = taxes_products_household / hc_total;
    let tau_SIW = household_social_contributions / wages_total;
    let tau_EXPORT = taxes_products_export / sum(&net_exports);
    let cfd_total = sum(&capitalformation_dwellings);
    let tau_CF = taxes_products_capitalformation_dwellings / cfd_total;
    let gc_total = sum(&government_consumption);
    let tau_G = taxes_products_government / gc_total;
    let psi = (hc_total + taxes_products_household) / disposable_income;
    let psi_H = (cfd_total + taxes_products_capitalformation_dwellings) / disposable_income;
    let theta_DIV = timescale * (mixed_income + property_income)
        / (positive_after_interest + bank_lending - timescale * corporate_tax);
    let r_G = interest_government_debt_quarterly / government_debt_quarterly;
    let theta_UB = 0.55 * (1.0 - tau_INC) * (1.0 - tau_SIW);
    let (theta, zeta, zeta_LTV, zeta_b) = (0.05, 0.03, 0.6, 0.5);

    // ============ Estimate exogenous processes (AR1 / Taylor rule) ===========
    let ea_deflator = (Te - 1..=Tx)
        .map(|i| deflator(ea, i))
        .collect::<Result<Vec<_>, _>>()?;
    let ea_inflation = diff_log(&ea_deflator);
    let (alpha_pi_EA, beta_pi_EA, sigma_pi_EA, _) = estimate_for_calibration_script(&ea_inflation);
    let a1: Vec<f64> = data
        .range("euribor", Te, Tx)?
        .into_iter()
        .map(quarterly_rate)
        .collect();
    let a2: Vec<f64> = ea_inflation.iter().map(|&x| libm::exp(x) - 1.0).collect();
    let a3: Vec<f64> = diff_log(&ea.range("real_gdp_quarterly", Te - 1, Tx)?)
        .iter()
        .map(|&x| libm::exp(x) - 1.0)
        .collect();
    let (rho, r_star, xi_pi, xi_gamma, pi_star) = estimate_taylor_rule(&a1, &a2, &a3);

    // Level series of government consumption, exports and imports.
    let levels = |total: f64, key: &str| -> Result<Vec<f64>, String> {
        let last = data.num(key, Tx)?;
        Ok(data
            .range(key, Te, Tx)?
            .iter()
            .map(|x| timescale * total * x / last)
            .collect())
    };
    let G_est_levels = levels(gc_total, "real_government_consumption_quarterly")?;
    let E_est_levels = levels(sum(&net_exports), "real_exports_quarterly")?;
    let I_est_levels = levels(sum(&imports), "real_imports_quarterly")?;
    let Y_EA_series_raw = ea.range("real_gdp_quarterly", Te, Tx)?;
    let log = |v: &[f64]| -> Vec<f64> { v.iter().map(|&x| libm::log(x)).collect() };
    // The euro area always uses logarithms, consistent with the Taylor rule.
    let (alpha_Y_EA, beta_Y_EA, sigma_Y_EA, epsilon_Y_EA) =
        estimate_for_calibration_script(&log(&Y_EA_series_raw));
    let estimate = |v: &[f64]| {
        if use_growth_rate_ar1 {
            let growth = diff_log(v);
            let last = growth.last().copied().unwrap_or(f64::NAN);
            (estimate_for_calibration_script(&growth), Some(last))
        } else {
            (estimate_for_calibration_script(&log(v)), None)
        }
    };
    let ((alpha_G, beta_G, sigma_G, _), g_G) = estimate(&G_est_levels);
    let ((alpha_E, beta_E, sigma_E, epsilon_E), g_E) = estimate(&E_est_levels);
    let ((alpha_I, beta_I, sigma_I, epsilon_I), g_I) = estimate(&I_est_levels);
    // On growth rates the residuals are one shorter.
    let epsilon_Y_EA = if use_growth_rate_ar1 {
        &epsilon_Y_EA[1..]
    } else {
        &epsilon_Y_EA[..]
    };
    let C = cov3(epsilon_Y_EA, &epsilon_E, &epsilon_I);

    // ================ Pack params & initial_conditions output ================
    let parameters = Parameters {
        G,
        T_prime,
        H_act: count(H_act, "H_act")?,
        H_inact: count(H_inact, "H_inact")?,
        J: count(J, "J")?,
        L: count(L, "L")?,
        I_s: counts(&I_s, "I_s")?,
        alpha_s,
        beta_s,
        kappa_s,
        delta_s,
        w_s,
        tau_Y_s,
        tau_K_s,
        tau_INC,
        tau_FIRM,
        tau_VAT,
        tau_SIF,
        tau_SIW,
        tau_EXPORT,
        tau_CF,
        tau_G,
        theta_UB,
        psi,
        psi_H,
        mu,
        theta_DIV,
        theta,
        zeta,
        zeta_LTV,
        zeta_b,
        b_CF_g,
        b_CFH_g,
        b_HH_g,
        c_G_g,
        c_E_g,
        c_I_g,
        a_sg,
        C,
        alpha_G,
        beta_G,
        sigma_G,
        alpha_E,
        beta_E,
        sigma_E,
        alpha_I,
        beta_I,
        sigma_I,
        alpha_pi_EA,
        beta_pi_EA,
        sigma_pi_EA,
        alpha_Y_EA,
        beta_Y_EA,
        sigma_Y_EA,
        r_G,
        rho,
        r_star,
        pi_star,
        xi_pi,
        xi_gamma,
        n_banks: 1,
        omega_pi: 0.0,
        theta_calvo: 1.0,
        overdrafts_in_capital_ratio: false,
        S: Some(S),
        T: Some(T),
        T_max: Some(count(T_max as f64, "T_max")?),
        use_growth_rate_ar1: Some(use_growth_rate_ar1),
    };

    let employees_total = sum(&employees);
    let sb_other = timescale
        * (social_benefits + other_net_transfers - unemployment_benefits - pension_benefits)
        / (employees_total + unemployed + inactive + firms_total + 1.0);
    let L_G = government_debt_quarterly;
    let (D_I, L_I, D_H, E_k) = (
        firm_cash_quarterly,
        firm_debt_quarterly,
        household_cash_quarterly,
        bank_equity_quarterly,
    );
    let output_total = sum(&output);
    let gdp_last = data.num("real_gdp_quarterly", Tx)?;
    let Y = data
        .range("real_gdp_quarterly", Te, Tx)?
        .iter()
        .map(|x| timescale * output_total * x / gdp_last)
        .collect();
    let data_deflator = (Te - 1..=Tx)
        .map(|i| deflator(data, i))
        .collect::<Result<Vec<_>, _>>()?;
    let pi = diff_log(&data_deflator);
    let Y_EA = ea.num("real_gdp_quarterly", Tx)?;
    if Tx < 2 {
        return Err("calibration_date is the first quarter of data[\"quarters_num\"]".into());
    }
    let pi_EA = deflator(ea, Tx)? / deflator(ea, Tx - 1)? - 1.0;
    // Histories up to T quarters after the calibration date, NaN beyond the
    // data.
    let history = |total: f64, key: &str| -> Result<Vec<f64>, String> {
        let last = data.num(key, Tx)?;
        let end = (Tx + T).min(T_calibration_exo_max);
        let mut v: Vec<f64> = data
            .range(key, Te, end)?
            .iter()
            .map(|x| timescale * total * x / last)
            .collect();
        v.extend(std::iter::repeat_n(
            f64::NAN,
            (Tx + T).saturating_sub(T_calibration_exo_max),
        ));
        Ok(v)
    };
    let ea_last = ea.num("real_gdp_quarterly", Tx)?;
    let initial_conditions = InitialConditions {
        D_I,
        L_I,
        omega,
        w_UB: timescale * unemployment_benefits / unemployed,
        sb_inact: timescale * pension_benefits / inactive,
        sb_other,
        D_H,
        K_H: sum(&dwellings),
        L_G,
        E_k,
        E_CB: L_G + L_I - D_I - D_H - E_k,
        D_RoW: 0.0,
        N_s: employees,
        Y,
        pi,
        Y_EA,
        pi_EA,
        r_bar,
        C_G: history(gc_total, "real_government_consumption_quarterly")?,
        C_E: history(sum(&net_exports), "real_exports_quarterly")?,
        Y_I: history(sum(&imports), "real_imports_quarterly")?,
        Y_EA_series: Y_EA_series_raw
            .iter()
            .map(|x| timescale * output_total * x / ea_last)
            .collect(),
        pi_EA_series: ea_inflation,
        r_bar_series: a1,
        g_G,
        g_E,
        g_I,
    };
    Ok(Calibration {
        parameters,
        initial_conditions,
        warnings,
    })
}

/// `get_params_and_initial_conditions_steady_state`, which BeforeIT.jl
/// keeps for tests: the calibration of the older version (as the MATLAB
/// code) with zero interest rates, government deficit and inflation, AR(1)
/// processes that keep every variable at its last value, and a constant
/// history of GDP. BeforeIT's data set `STEADY_STATE2010Q1` is its result
/// for Italy in 2010Q1 with `scale = 0.001`.
///
/// It needs the format of [`ITALY_CALIBRATION`]. As in BeforeIT.jl, it
/// takes the year as the row index of the first match in `years_num`,
/// which for that format is always 1, the first year of the tables.
#[allow(clippy::too_many_arguments)]
pub fn get_params_and_initial_conditions_steady_state(
    calibration_date: Date,
    calibration_data: &Table,
    figaro: &Table,
    data: &Table,
    ea: &Table,
    max_calibration_date: Date,
    estimation_date: Date,
    scale: f64,
) -> Result<(Parameters, InitialConditions), String> {
    let cd = Tab {
        name: "calibration",
        t: calibration_data,
    };
    let figaro = Tab {
        name: "figaro",
        t: figaro,
    };
    let data = Tab {
        name: "data",
        t: data,
    };
    let ea = Tab { name: "ea", t: ea };
    let index = |v: Vec<usize>, k: usize, what: &str| -> Result<usize, String> {
        v.get(k - 1).copied().ok_or_else(|| {
            format!(
                "{what}: BoundsError, index {k} of a {}-dimensional index",
                v.len()
            )
        })
    };
    let capped = calibration_date.min(max_calibration_date);
    let year_end = Date::new(capped.year, 12, 31);
    let Tc = index(
        cd.find_cartesian("years_num", year_end.datenum(), year_end)?,
        1,
        "years_num",
    )?;
    let Tq = index(
        cd.find_cartesian("quarters_num", calibration_date.datenum(), calibration_date)?,
        2,
        "quarters_num",
    )?;
    let Te = index(
        data.find_cartesian("quarters_num", estimation_date.datenum(), estimation_date)?,
        1,
        "quarters_num",
    )?;
    let Tx = index(
        data.find_cartesian("quarters_num", calibration_date.datenum(), calibration_date)?,
        1,
        "quarters_num",
    )?;
    let T_calibration_exo_max = data.get("quarters_num")?.len();
    if Te < 2 || Tx < 2 {
        return Err("the dates are at the first quarter of data[\"quarters_num\"]".into());
    }
    let intermediate_consumption = figaro.slab("intermediate_consumption", Tc)?;
    let G = intermediate_consumption.len();
    let household_consumption = figaro.column("household_consumption", Tc)?;
    let fixed_capitalformation = figaro.column("fixed_capitalformation", Tc)?;
    let exports = figaro.column("exports", Tc)?;
    let compensation_employees = figaro.column("compensation_employees", Tc)?;
    let household_cash_quarterly = cd.num("household_cash_quarterly", Tq)?;
    let property_income = cd.num("property_income", Tc)?;
    let mixed_income = cd.num("mixed_income", Tc)?;
    let operating_surplus = figaro.column("operating_surplus", Tc)?;
    let firm_cash_quarterly = cd.num("firm_cash_quarterly", Tq)?;
    let firm_debt_quarterly = cd.num("firm_debt_quarterly", Tq)?;
    cd.num("firm_interest", Tc)?;
    let firm_interest = 0.0; // STEADY-STATE
    let government_debt_quarterly = cd.num("government_debt_quarterly", Tq)?;
    cd.num("interest_government_debt", Tc)?;
    let interest_government_debt = 0.0; // STEADY-STATE
    let government_consumption = figaro.column("government_consumption", Tc)?;
    let social_benefits = cd.num("social_benefits", Tc)?;
    let unemployment_benefits = cd.num("unemployment_benefits", Tc)?;
    let pension_benefits = cd.num("pension_benefits", Tc)?;
    let corporate_tax = cd.num("corporate_tax", Tc)?;
    let wages_scalar = cd.num("wages", Tc)?;
    let taxes_products_household = figaro.num("taxes_products_household", Tc)?;
    let social_contributions = cd.num("social_contributions", Tc)?;
    let income_tax = cd.num("income_tax", Tc)?;
    let capital_taxes = cd.num("capital_taxes", Tc)?;
    let taxes_products_fixed_capitalformation =
        figaro.num("taxes_products_capitalformation", Tc)?;
    let taxes_production = figaro.column("taxes_production", Tc)?;
    let taxes_products_government = figaro.num("taxes_products_government", Tc)?;
    let bank_equity_quarterly = cd.num("bank_equity_quarterly", Tq)?;
    let taxes_products = figaro.column("taxes_products", Tc)?;
    cd.num("government_deficit", Tc)?;
    let government_deficit = 0.0; // STEADY-STATE
    let mut firms = cd.column("firms", Tc)?;
    let mut employees = cd.column("employees", Tc)?;
    let population = cd.num("population", Tc)?;
    data.num("euribor", Tx)?;
    let r_bar = 0.0; // STEADY-STATE
    let omega = 0.85;
    let fixed_assets = cd.num("fixed_assets", Tc)?;
    let dwellings = cd.num("dwellings", Tc)?;
    let fixed_assets_eu7 = cd.column("fixed_assets_eu7", Tc)?;
    let dwellings_eu7 = cd.column("dwellings_eu7", Tc)?;
    let nominal_nace64_output_eu7 = cd.column("nominal_nace64_output_eu7", Tc)?;
    let gross_capitalformation_dwellings = cd.num("gross_capitalformation_dwellings", Tc)?;
    let nace64_capital_consumption = cd.column("nace64_capital_consumption", Tc)?;
    let nominal_nace64_output = cd.column("nominal_nace64_output", Tc)?;
    let unemployment_rate_quarterly = data.num("unemployment_rate_quarterly", Tx)?;

    let ic_columns = column_sums(&intermediate_consumption);
    let output: Vec<f64> = (0..G)
        .map(|i| {
            ic_columns[i]
                + taxes_products[i]
                + taxes_production[i]
                + compensation_employees[i]
                + operating_surplus[i]
        })
        .collect();
    let capital_consumption: Vec<f64> = (0..G)
        .map(|i| nace64_capital_consumption[i] / nominal_nace64_output[i] * output[i])
        .collect();
    let operating_surplus: Vec<f64> = (0..G)
        .map(|i| operating_surplus[i] - capital_consumption[i])
        .collect();
    let taxes_products_export = 0.0;
    let compensation_total = sum(&compensation_employees);
    let employers_social_contributions =
        min(social_contributions, compensation_total - wages_scalar);
    let fixed_capitalformation: Vec<f64> = fixed_capitalformation.iter().map(|&x| pos(x)).collect();
    let fcf_total = sum(&fixed_capitalformation);
    let taxes_products_capitalformation_dwellings = gross_capitalformation_dwellings
        * (1.0 - 1.0 / (1.0 + taxes_products_fixed_capitalformation / fcf_total));
    let income: Vec<f64> = (0..G)
        .map(|i| {
            compensation_employees[i]
                + operating_surplus[i]
                + capital_consumption[i]
                + taxes_production[i]
                + taxes_products[i]
        })
        .collect();
    let timescale = data.num("nominal_gdp_quarterly", Tx)?
        / (sum(&income)
            + taxes_products_household
            + taxes_products_capitalformation_dwellings
            + taxes_products_government
            + taxes_products_export);
    let w: Vec<f64> = (0..G)
        .map(|i| {
            (fixed_assets_eu7[i] - dwellings_eu7[i]) / nominal_nace64_output_eu7[i] * output[i]
        })
        .collect();
    let w_total = sum(&w);
    let fixed_assets_other_than_dwellings: Vec<f64> = w
        .iter()
        .map(|wi| (fixed_assets - dwellings) * wi / w_total)
        .collect();
    let capitalformation_dwellings: Vec<f64> = fixed_capitalformation
        .iter()
        .map(|f| {
            (gross_capitalformation_dwellings - taxes_products_capitalformation_dwellings) * f
                / fcf_total
        })
        .collect();
    let fixed_capital_formation_other_than_dwellings: Vec<f64> = (0..G)
        .map(|i| fixed_capitalformation[i] - capitalformation_dwellings[i])
        .collect();
    let exports: Vec<f64> = exports.iter().map(|&x| pos(x)).collect();
    let ic_rows = row_sums(&intermediate_consumption);
    let (cc_total, fcfo_total) = (
        sum(&capital_consumption),
        sum(&fixed_capital_formation_other_than_dwellings),
    );
    let net_imports: Vec<f64> = (0..G)
        .map(|i| {
            ic_rows[i]
                + household_consumption[i]
                + government_consumption[i]
                + fixed_capital_formation_other_than_dwellings[i] * cc_total / fcfo_total
                + capitalformation_dwellings[i]
                + exports[i]
                - output[i]
        })
        .collect();
    let imports: Vec<f64> = net_imports.iter().map(|&x| pos(x)).collect();
    let reexports: Vec<f64> = net_imports.iter().map(|&x| neg(x)).collect();
    let household_social_contributions = social_contributions - employers_social_contributions;
    let wages: Vec<f64> = compensation_employees
        .iter()
        .map(|c| c * (1.0 - employers_social_contributions / compensation_total))
        .collect();
    let household_income_tax = income_tax - corporate_tax;
    let other_net_transfers = pos(taxes_products_household
        + taxes_products_capitalformation_dwellings
        + taxes_products_export
        + sum(&taxes_products)
        + sum(&taxes_production)
        + employers_social_contributions
        + household_social_contributions
        + household_income_tax
        + corporate_tax
        + capital_taxes
        - social_benefits
        - sum(&government_consumption)
        - interest_government_debt
        - government_deficit);
    let wages_total = sum(&wages);
    let disposable_income =
        wages_total + mixed_income + property_income + social_benefits + other_net_transfers
            - household_social_contributions
            - household_income_tax
            - capital_taxes;
    let unemployed = matlab_round(unemployment_rate_quarterly * sum(&employees));
    let at_least_one: Vec<f64> = firms.iter().map(|&f| max(1.0, f)).collect();
    let persons: Vec<f64> = (0..G).map(|i| max(at_least_one[i], employees[i])).collect();
    let inactive = population - sum(&persons) - unemployed - sum(&at_least_one) - 1.0;

    for f in firms.iter_mut() {
        *f = max(1.0, matlab_round(scale * *f));
    }
    for (e, f) in employees.iter_mut().zip(&firms) {
        *e = max(*f, matlab_round(scale * *e));
    }
    let inactive = max(1.0, matlab_round(scale * inactive));
    let unemployed = max(1.0, matlab_round(scale * unemployed));

    let by_sector = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { (0..G).map(f).collect() };
    let alpha_s = by_sector(&|i| timescale * output[i] / employees[i]);
    let beta_s = by_sector(&|i| output[i] / ic_columns[i]);
    let kappa_s =
        by_sector(&|i| timescale * output[i] / fixed_assets_other_than_dwellings[i] / omega);
    let mut delta_s = by_sector(&|i| {
        timescale * capital_consumption[i] / fixed_assets_other_than_dwellings[i] / omega
    });
    nan_to_zero(&mut delta_s);
    let w_s = by_sector(&|i| timescale * wages[i] / employees[i]);
    let tau_Y_s = by_sector(&|i| taxes_products[i] / output[i]);
    let tau_K_s = by_sector(&|i| taxes_production[i] / output[i]);
    let shares = |v: &[f64]| -> Vec<f64> {
        let total = sum(v);
        v.iter().map(|x| x / total).collect()
    };
    let b_CF_g = shares(&fixed_capital_formation_other_than_dwellings);
    let b_CFH_g = shares(&capitalformation_dwellings);
    let b_HH_g = shares(&household_consumption);
    let mut a_sg: Vec<Vec<f64>> = intermediate_consumption
        .iter()
        .map(|r| (0..G).map(|j| r[j] / ic_columns[j]).collect())
        .collect();
    for r in a_sg.iter_mut() {
        nan_to_zero(r);
    }
    let c_G_g = shares(&government_consumption);
    let net_exports: Vec<f64> = (0..G).map(|i| exports[i] - reexports[i]).collect();
    let c_E_g = shares(&net_exports);
    let c_I_g = shares(&imports);

    let T_prime = Tx - Te + 1;
    let T = 12;
    let T_max = T as i64 - 0.max(Tx as i64 + T as i64 - T_calibration_exo_max as i64);
    let firms_total = sum(&firms);
    let employees_total = sum(&employees);
    let H_act = employees_total + unemployed + firms_total + 1.0;
    let J = matlab_round(firms_total / 4.0);
    let L = matlab_round(firms_total / 2.0);
    let mu = timescale * firm_interest / firm_debt_quarterly - r_bar;
    let tau_INC = (household_income_tax + capital_taxes)
        / (wages_total + property_income + mixed_income - household_social_contributions);
    let fao_total = sum(&fixed_assets_other_than_dwellings);
    let positive_surplus: Vec<f64> = operating_surplus.iter().map(|&x| pos(x)).collect();
    let positive_surplus_total = sum(&positive_surplus);
    let profits: Vec<f64> = (0..G)
        .map(|i| {
            pos(timescale * operating_surplus[i]
                - timescale * firm_interest * fixed_assets_other_than_dwellings[i] / fao_total
                + r_bar * firm_cash_quarterly * positive_surplus[i] / positive_surplus_total)
        })
        .collect();
    let profits_total = sum(&profits);
    let bank_lending =
        timescale * firm_interest - r_bar * (firm_debt_quarterly - bank_equity_quarterly);
    let tau_FIRM = timescale * corporate_tax / (profits_total + bank_lending);
    let hc_total = sum(&household_consumption);
    let tau_VAT = taxes_products_household / hc_total;
    let tau_SIF = employers_social_contributions / wages_total;
    let tau_SIW = household_social_contributions / wages_total;
    let tau_EXPORT = taxes_products_export / sum(&net_exports);
    let cfd_total = sum(&capitalformation_dwellings);
    let tau_CF = taxes_products_capitalformation_dwellings / cfd_total;
    let gc_total = sum(&government_consumption);
    let tau_G = taxes_products_government / gc_total;
    let psi = (hc_total + taxes_products_household) / disposable_income;
    let psi_H = (cfd_total + taxes_products_capitalformation_dwellings) / disposable_income;
    let theta_DIV = timescale * (mixed_income + property_income)
        / (profits_total + bank_lending - timescale * corporate_tax);
    let r_G = timescale * interest_government_debt / government_debt_quarterly;
    let theta_UB = 0.55 * (1.0 - tau_INC) * (1.0 - tau_SIW);

    // The estimates of BeforeIT.jl, which are then replaced by the steady
    // state; computed for the errors that missing data give.
    let ea_deflator = (Te - 1..=Tx)
        .map(|i| deflator(ea, i))
        .collect::<Result<Vec<_>, _>>()?;
    ea.range("real_gdp_quarterly", Te - 1, Tx)?;
    data.range("euribor", Te, Tx)?;
    for key in [
        "real_government_consumption_quarterly",
        "real_exports_quarterly",
        "real_imports_quarterly",
    ] {
        data.range(key, Te, Tx)?;
    }
    drop(ea_deflator);

    let parameters = Parameters {
        G,
        T_prime,
        H_act: count(H_act, "H_act")?,
        H_inact: count(inactive, "H_inact")?,
        J: count(J, "J")?,
        L: count(L, "L")?,
        I_s: counts(&firms, "I_s")?,
        alpha_s,
        beta_s,
        kappa_s,
        delta_s,
        w_s,
        tau_Y_s,
        tau_K_s,
        tau_INC,
        tau_FIRM,
        tau_VAT,
        tau_SIF,
        tau_SIW,
        tau_EXPORT,
        tau_CF,
        tau_G,
        theta_UB,
        psi,
        psi_H,
        mu,
        theta_DIV,
        theta: 0.05,
        zeta: 0.03,
        zeta_LTV: 0.6,
        zeta_b: 0.5,
        b_CF_g,
        b_CFH_g,
        b_HH_g,
        c_G_g,
        c_E_g,
        c_I_g,
        a_sg,
        // STEADY-STATE: no shocks, every AR(1) process keeps its last value.
        C: [[0.0; 3]; 3],
        alpha_G: 1.0,
        beta_G: 0.0,
        sigma_G: 0.0,
        alpha_E: 1.0,
        beta_E: 0.0,
        sigma_E: 0.0,
        alpha_I: 1.0,
        beta_I: 0.0,
        sigma_I: 0.0,
        alpha_pi_EA: 1.0,
        beta_pi_EA: 0.0,
        sigma_pi_EA: 0.0,
        alpha_Y_EA: 1.0,
        beta_Y_EA: 0.0,
        sigma_Y_EA: 0.0,
        r_G,
        rho: 1.0,
        r_star: 0.0,
        pi_star: 0.0,
        xi_pi: 0.0,
        xi_gamma: 0.0,
        n_banks: 1,
        omega_pi: 0.0,
        theta_calvo: 1.0,
        overdrafts_in_capital_ratio: false,
        S: Some(G),
        T: Some(T),
        T_max: Some(count(T_max as f64, "T_max")?),
        use_growth_rate_ar1: None,
    };

    let sb_other = timescale
        * (social_benefits + other_net_transfers - unemployment_benefits - pension_benefits)
        / (employees_total + unemployed + inactive + firms_total + 1.0);
    let L_G = government_debt_quarterly;
    let (D_I, L_I, D_H, E_k) = (
        firm_cash_quarterly,
        firm_debt_quarterly,
        household_cash_quarterly,
        bank_equity_quarterly,
    );
    let output_total = sum(&output);
    let history = |total: f64, key: &str| -> Result<Vec<f64>, String> {
        let last = data.num(key, Tx)?;
        let end = (Tx + T).min(T_calibration_exo_max);
        let mut v: Vec<f64> = data
            .range(key, Te, end)?
            .iter()
            .map(|x| timescale * total * x / last)
            .collect();
        v.extend(std::iter::repeat_n(
            f64::NAN,
            (Tx + T).saturating_sub(T_calibration_exo_max),
        ));
        Ok(v)
    };
    let initial_conditions = InitialConditions {
        D_I,
        L_I,
        omega,
        w_UB: timescale * unemployment_benefits / unemployed,
        sb_inact: timescale * pension_benefits / inactive,
        sb_other,
        D_H,
        K_H: dwellings,
        L_G,
        E_k,
        E_CB: L_G + L_I - D_I - D_H - E_k,
        D_RoW: 0.0,
        N_s: employees,
        // STEADY-STATE: constant GDP and no inflation.
        Y: vec![timescale * output_total; T_prime],
        pi: vec![0.0; T_prime],
        Y_EA: ea.num("real_gdp_quarterly", Tx)?,
        pi_EA: 0.0,
        r_bar,
        C_G: history(gc_total, "real_government_consumption_quarterly")?,
        C_E: history(sum(&net_exports), "real_exports_quarterly")?,
        Y_I: history(sum(&imports), "real_imports_quarterly")?,
        r_bar_series: Vec::new(),
        Y_EA_series: Vec::new(),
        pi_EA_series: Vec::new(),
        g_G: None,
        g_E: None,
        g_I: None,
    };
    Ok((parameters, initial_conditions))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MATLAB `datenum` values of the data of BeforeIT.jl.
    #[test]
    fn date_numbers() {
        assert_eq!(Date::new(2010, 3, 31).datenum(), 734228.0);
        assert_eq!(Date::new(2010, 12, 31).datenum(), 734503.0);
        assert_eq!(Date::new(1, 1, 1).datenum(), 367.0);
        assert_eq!(Date::new(0, 1, 1).datenum(), 1.0);
        for n in [1.0, 367.0, 729024.0, 734228.0, 737790.0, 800000.0] {
            assert_eq!(Date::from_datenum(n).datenum(), n);
        }
        assert_eq!(Date::from_datenum(734228.0), Date::new(2010, 3, 31));
        assert_eq!(Date::parse("1996-12-31"), Ok(Date::new(1996, 12, 31)));
        assert!(Date::parse("1996-13-31").is_err());
    }

    #[test]
    fn zenodo_urls() {
        assert_eq!(
            zenodo_calibration_url("AT").unwrap(),
            "https://zenodo.org/records/18698434/files/AT_calibration_object.jld2"
        );
        assert!(zenodo_calibration_url("XX").is_err());
    }
}
