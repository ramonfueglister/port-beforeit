//! Business-cycle statistics, BeforeIT `src/utils/correlation_utils.jl`:
//! the Hodrick-Prescott filter, cross- and autocorrelations of the cyclical
//! components, and [`CorrelationStats`] of data or simulations.

use std::collections::BTreeMap;
use std::path::Path;

use crate::calibration::{Array, CalibrationData, Table, table_from_json};
use crate::evaluation::{mean, std};
use crate::julia::sum_by;

/// A matrix given by its rows.
pub type Matrix = Vec<Vec<f64>>;

/// Solves `A x = b` by LU factorization with partial pivoting, as Julia's
/// `lu(A) \ b` (LAPACK `getrf` and `getrs`): row exchanges by the largest
/// entry of the column, multipliers with the reciprocal of the pivot,
/// forward and back substitution by columns. LAPACK orders the operations
/// of the factorization differently, so the last digits can differ.
///
/// Returns an error if a pivot is exactly 0 (Julia throws a
/// `SingularException`).
fn lu_solve(mut A: Matrix, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = A.len();
    for j in 0..n {
        let mut p = j;
        for i in j + 1..n {
            if A[i][j].abs() > A[p][j].abs() {
                p = i;
            }
        }
        if A[p][j] == 0.0 {
            return Err("the matrix is singular".into());
        }
        A.swap(j, p);
        b.swap(j, p);
        let r = 1.0 / A[j][j];
        for i in j + 1..n {
            A[i][j] *= r;
        }
        for i in j + 1..n {
            let l = A[i][j];
            if l != 0.0 {
                for k in j + 1..n {
                    A[i][k] -= l * A[j][k];
                }
            }
        }
    }
    // L y = b (unit lower), then U x = y, column by column.
    for k in 0..n {
        if b[k] != 0.0 {
            for i in k + 1..n {
                b[i] -= b[k] * A[i][k];
            }
        }
    }
    for k in (0..n).rev() {
        if b[k] != 0.0 {
            b[k] /= A[k][k];
            for i in 0..k {
                b[i] -= b[k] * A[i][k];
            }
        }
    }
    Ok(b)
}

/// `hpfilter(y; λ)`: the Hodrick-Prescott filter, the trend
/// `(I + λ D'D) \ y` with the second-difference matrix `D`, and the cycle
/// `y − trend`. BeforeIT's default is `λ = 1600` for quarterly data.
///
/// Returns an error for fewer than two values (Julia throws).
pub fn hpfilter(y: &[f64], lambda: f64) -> Result<(Vec<f64>, Vec<f64>), String> {
    let T = y.len();
    if T < 2 {
        return Err(format!("hpfilter: {T} values, at least 2 are needed"));
    }
    // I + λ D'D; D'D has entries of at most 6, so every entry is exact.
    let mut A = vec![vec![0.0; T]; T];
    let row = [1.0, -2.0, 1.0];
    for i in 0..T - 2 {
        for (a, &da) in row.iter().enumerate() {
            for (c, &dc) in row.iter().enumerate() {
                A[i + a][i + c] += da * dc;
            }
        }
    }
    for (i, r) in A.iter_mut().enumerate() {
        for (j, v) in r.iter_mut().enumerate() {
            *v = f64::from(u8::from(i == j)) + lambda * *v;
        }
    }
    let trend = if T == 2 {
        // D is empty: A = I is diagonal, Julia solves `Diagonal(A) \ y`.
        y.iter().map(|v| v / 1.0).collect()
    } else {
        lu_solve(A, y.to_vec())?
    };
    let cycle = y.iter().zip(&trend).map(|(a, t)| a - t).collect();
    Ok((trend, cycle))
}

/// `x .- mean(x)`.
fn centered(x: &[f64]) -> Vec<f64> {
    let m = mean(x);
    x.iter().map(|v| v - m).collect()
}

/// `sum(a .* b)` in the order of Julia's `sum`.
fn sum_of_products(a: &[f64], b: &[f64]) -> f64 {
    sum_by(a.len(), |i| a[i] * b[i])
}

/// `cross_correlation(x, y, maxlag)`: the correlation of `x[t]` with
/// `y[t + lag]` for the lags `−maxlag..=maxlag`, with the covariance and
/// the standard deviations divided by `n` (as MATLAB's `xcorr` with
/// normalisation); 0 for lags of at least `n`.
///
/// Returns an error if `x` and `y` have different lengths (Julia throws).
pub fn cross_correlation(x: &[f64], y: &[f64], maxlag: usize) -> Result<Vec<f64>, String> {
    let nx = x.len();
    if nx != y.len() {
        return Err("Inputs must be of the same length".into());
    }
    let n = nx as f64;
    let (xc, yc) = (centered(x), centered(y));
    let x_std_dev = (sum_of_products(&xc, &xc) / n).sqrt();
    let y_std_dev = (sum_of_products(&yc, &yc) / n).sqrt();
    let maxlag = maxlag as i64;
    Ok((-maxlag..=maxlag)
        .map(|lag| {
            let shift = lag.unsigned_abs() as usize;
            if shift >= nx {
                0.0
            } else if lag >= 0 {
                let covariance = sum_of_products(&xc[..nx - shift], &yc[shift..]) / n;
                covariance / (x_std_dev * y_std_dev)
            } else {
                let covariance = sum_of_products(&xc[shift..], &yc[..nx - shift]) / n;
                covariance / (x_std_dev * y_std_dev)
            }
        })
        .collect())
}

/// `auto_correlation(x, lags)`: the autocorrelation of `x` at each lag,
/// with the autocovariance and the variance divided by `n` (as MATLAB's
/// `autocorr`); 1 at lag 0 and 0 for lags of at least `n`. BeforeIT's
/// default is `lags = 0:20`.
pub fn auto_correlation(x: &[f64], lags: impl IntoIterator<Item = usize>) -> Vec<f64> {
    let nx = x.len();
    let n = nx as f64;
    let xc = centered(x);
    let x_var = sum_of_products(&xc, &xc) / n;
    lags.into_iter()
        .map(|lag| {
            if lag == 0 {
                1.0
            } else if lag >= nx {
                0.0
            } else {
                sum_of_products(&xc[..nx - lag], &xc[lag..]) / n / x_var
            }
        })
        .collect()
}

/// `CorrelationStats`: cyclical statistics of variables, computed by
/// [`correlation_stats`]. Each variable is HP-filtered and its cycle is
/// correlated with the GDP cycle (`crosscor`) and with itself (`autocor`);
/// `volatility` is the standard deviation of the series itself. Each
/// column of a matrix is a sample: one for data, one per run for
/// simulations.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelationStats {
    /// The variables that were processed.
    pub variables: Vec<String>,
    /// Largest lag of the cross-correlations (rows for `−L..=L`).
    pub correlation_lags: usize,
    /// Largest lag of the autocorrelations (rows for `0..=H`).
    pub autocorr_lags: usize,
    /// `(2 L + 1) × samples` per variable.
    pub crosscor: BTreeMap<String, Matrix>,
    /// `(H + 1) × samples` per variable.
    pub autocor: BTreeMap<String, Matrix>,
    /// One standard deviation per sample.
    pub volatility: BTreeMap<String, Vec<f64>>,
}

/// `_as_columns(v)`: the samples of a series. A vector is one sample
/// without its `missing` entries (`strip_missing`); each column of a matrix
/// is a sample. `None` for anything else (Julia: not an `AbstractVecOrMat`).
fn as_columns(a: &Array) -> Option<Vec<Vec<Option<f64>>>> {
    match a.size.len() {
        1 => Some(vec![
            a.data.iter().filter(|x| x.is_some()).copied().collect(),
        ]),
        2 => Some((0..a.size[1]).map(|c| a.column(c).to_vec()).collect()),
        _ => None,
    }
}

/// The numbers of a sample, or an error if one is `missing` (Julia cannot
/// filter a series with `missing` entries in a matrix and throws).
fn numbers(sample: &[Option<f64>], name: &str) -> Result<Vec<f64>, String> {
    sample
        .iter()
        .map(|x| x.ok_or_else(|| format!("\"{name}\" has a missing value")))
        .collect()
}

/// `correlation_stats(data, variables; gdp_var, correlation_lags,
/// autocorr_lags)`: [`CorrelationStats`] of the series in `data`, either
/// data (vectors, such as the table `data` of [`CalibrationData`]) or
/// simulations (`time × runs` matrices, such as
/// [`Predictions::to_table`](crate::Predictions::to_table)). The GDP series
/// is `gdp_var` or, if that is absent, `real_gdp`. Variables that are
/// absent, not vectors or matrices, or have another number of samples than
/// GDP are skipped. The cycles are correlated over their last
/// `min(length)` entries. BeforeIT's defaults are `gdp_var =
/// "real_gdp_quarterly"`, `correlation_lags = 15` and `autocorr_lags = 20`.
///
/// Returns an error if there is no GDP series, if a series used has a
/// `missing` value in a matrix or fewer than two values (Julia throws).
pub fn correlation_stats(
    data: &Table,
    variables: &[&str],
    gdp_var: &str,
    correlation_lags: usize,
    autocorr_lags: usize,
) -> Result<CorrelationStats, String> {
    let gdp_ref = [gdp_var, "real_gdp"]
        .into_iter()
        .find(|v| data.contains_key(*v))
        .ok_or_else(|| format!("No GDP reference ('{gdp_var}' or 'real_gdp') found in data"))?;
    let gdp_cols = as_columns(&data[gdp_ref])
        .ok_or_else(|| format!("\"{gdp_ref}\" is not a vector or matrix"))?;
    let mut stats = CorrelationStats {
        variables: Vec::new(),
        correlation_lags,
        autocorr_lags,
        crosscor: BTreeMap::new(),
        autocor: BTreeMap::new(),
        volatility: BTreeMap::new(),
    };
    for &name in variables {
        let Some(cols) = data.get(name).and_then(as_columns) else {
            continue;
        };
        let n_samples = cols.len();
        if gdp_cols.len() != n_samples {
            continue;
        }
        let mut xc = vec![vec![0.0; n_samples]; 2 * correlation_lags + 1];
        let mut ac = vec![vec![0.0; n_samples]; autocorr_lags + 1];
        let mut vol = vec![0.0; n_samples];
        for n in 0..n_samples {
            let gdp = numbers(&gdp_cols[n], gdp_ref)?;
            let var = numbers(&cols[n], name)?;
            let (_, gdp_cycle) = hpfilter(&gdp, 1600.0)?;
            let (_, var_cycle) = hpfilter(&var, 1600.0)?;
            let m = gdp_cycle.len().min(var_cycle.len());
            let x = cross_correlation(
                &gdp_cycle[gdp_cycle.len() - m..],
                &var_cycle[var_cycle.len() - m..],
                correlation_lags,
            )?;
            let a = auto_correlation(&var_cycle, 0..=autocorr_lags);
            for (r, v) in x.into_iter().enumerate() {
                xc[r][n] = v;
            }
            for (r, v) in a.into_iter().enumerate() {
                ac[r][n] = v;
            }
            vol[n] = std(&var);
        }
        stats.crosscor.insert(name.to_string(), xc);
        stats.autocor.insert(name.to_string(), ac);
        stats.volatility.insert(name.to_string(), vol);
        stats.variables.push(name.to_string());
    }
    Ok(stats)
}

/// `extract_yq(files)`: the names `YYYYQn.<extension>` among `files`, as a
/// sorted list. Returns an error for a name with a `Q` that does not have
/// this form (Julia throws).
pub(crate) fn extract_yq(files: &[String], extension: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for f in files {
        if !f.contains('Q') {
            continue;
        }
        let b = f.as_bytes();
        let ok = f.len() == 6 + 1 + extension.len()
            && b[..4].iter().all(u8::is_ascii_digit)
            && b[4] == b'Q'
            && b[5].is_ascii_digit()
            && f[6..] == format!(".{extension}");
        if !ok {
            return Err(format!("\"{f}\" is not of the form YYYYQn.{extension}"));
        }
        out.push(f.clone());
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// The names of the files in `folder`.
pub(crate) fn read_dir_names(folder: &Path) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    for e in std::fs::read_dir(folder).map_err(|e| format!("{}: {e}", folder.display()))? {
        let e = e.map_err(|e| format!("{}: {e}", folder.display()))?;
        names.push(e.file_name().to_string_lossy().into_owned());
    }
    Ok(names)
}

/// `correlation_stats(folder, variables; kwargs...)`: the
/// [`CorrelationStats`] of every prediction file `YYYYQn.json` in `folder`
/// (the port's JSON of [`table_to_json`](crate::calibration::table_to_json)
/// with the table of [`Predictions::to_table`](crate::Predictions::to_table)
/// in place of BeforeIT's `YYYYQn.jld2`), with their samples side by side. Prints the number of files found, as
/// BeforeIT logs it with `@info`.
///
/// Returns an error if there is no such file or a file cannot be read, or
/// as [`correlation_stats`].
pub fn correlation_stats_folder(
    folder: &Path,
    variables: &[&str],
    gdp_var: &str,
    correlation_lags: usize,
    autocorr_lags: usize,
) -> Result<CorrelationStats, String> {
    let files = extract_yq(&read_dir_names(folder)?, "json")?;
    if files.is_empty() {
        return Err(format!(
            "No valid prediction files (YYYYQn.json) found in {}",
            folder.display()
        ));
    }
    eprintln!(
        "Found {} prediction files in {}",
        files.len(),
        folder.display()
    );
    let mut stats = Vec::with_capacity(files.len());
    for f in &files {
        let path = folder.join(f);
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let t = table_from_json(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        stats.push(correlation_stats(
            &t,
            variables,
            gdp_var,
            correlation_lags,
            autocorr_lags,
        )?);
    }
    let mut it = stats.into_iter();
    let first = it.next().expect("at least one file");
    Ok(it.fold(first, merge_samples))
}

/// `_merge_samples(a, b)`: the samples of `b` after those of `a`, for the
/// variables in both; `variables` and the lags of `a`.
fn merge_samples(a: CorrelationStats, b: CorrelationStats) -> CorrelationStats {
    let hcat = |x: &Matrix, y: &Matrix| -> Matrix {
        x.iter()
            .zip(y)
            .map(|(r, s)| r.iter().chain(s).copied().collect())
            .collect()
    };
    CorrelationStats {
        crosscor: a
            .crosscor
            .iter()
            .filter_map(|(k, m)| b.crosscor.get(k).map(|n| (k.clone(), hcat(m, n))))
            .collect(),
        autocor: a
            .autocor
            .iter()
            .filter_map(|(k, m)| b.autocor.get(k).map(|n| (k.clone(), hcat(m, n))))
            .collect(),
        volatility: a
            .volatility
            .iter()
            .filter_map(|(k, v)| {
                b.volatility
                    .get(k)
                    .map(|w| (k.clone(), v.iter().chain(w).copied().collect()))
            })
            .collect(),
        variables: a.variables,
        correlation_lags: a.correlation_lags,
        autocorr_lags: a.autocorr_lags,
    }
}

/// `mean(A; dims = 2)`: the mean of each row, the sum from left to right
/// divided by the number of columns.
fn row_means(m: &Matrix) -> Vec<f64> {
    m.iter()
        .map(|r| r.iter().fold(0.0, |acc, v| acc + v) / r.len() as f64)
        .collect()
}

/// `std(A; dims = 2)`: the standard deviation of each row, the squared
/// deviations summed from left to right and multiplied by `1 / (n − 1)`
/// (Julia multiplies by the rational `1//(n − 1)` converted to a float).
fn row_stds(m: &Matrix) -> Vec<f64> {
    m.iter()
        .zip(row_means(m))
        .map(|(r, mu)| {
            let s = r.iter().fold(0.0, |acc, v| acc + (v - mu) * (v - mu));
            (s * (1.0 / (r.len() as f64 - 1.0))).sqrt()
        })
        .collect()
}

impl CorrelationStats {
    fn get<'a>(m: &'a BTreeMap<String, Matrix>, var: &str) -> Result<&'a Matrix, String> {
        m.get(var)
            .ok_or_else(|| format!("no statistics for \"{var}\" (Julia throws a KeyError)"))
    }

    /// `mean_crosscor(stats, var)`: the mean over the samples at each lag.
    pub fn mean_crosscor(&self, var: &str) -> Result<Vec<f64>, String> {
        Ok(row_means(Self::get(&self.crosscor, var)?))
    }

    /// `std_crosscor(stats, var)`: the standard deviation over the samples
    /// at each lag (NaN for one sample).
    pub fn std_crosscor(&self, var: &str) -> Result<Vec<f64>, String> {
        Ok(row_stds(Self::get(&self.crosscor, var)?))
    }

    /// `mean_autocor(stats, var)`.
    pub fn mean_autocor(&self, var: &str) -> Result<Vec<f64>, String> {
        Ok(row_means(Self::get(&self.autocor, var)?))
    }

    /// `std_autocor(stats, var)`.
    pub fn std_autocor(&self, var: &str) -> Result<Vec<f64>, String> {
        Ok(row_stds(Self::get(&self.autocor, var)?))
    }
}

/// `load_calibration_data(country_code)`: the calibration data in
/// `data/<country code in lower case>/` relative to the working
/// directory, from `calibration_object.json` or else
/// `calibration_data.json` (the port's JSON of
/// [`CalibrationData::from_json`] in place of BeforeIT's JLD2 files).
/// Prints where it loads from, as BeforeIT logs it with `@info`.
///
/// Returns an error if the folder or both files are missing or a file
/// cannot be read.
pub fn load_calibration_data(country_code: &str) -> Result<CalibrationData, String> {
    eprintln!("Loading calibration data for {country_code} from migrated data");
    let country_dir = format!("data/{}", country_code.to_lowercase());
    if !Path::new(&country_dir).is_dir() {
        return Err(format!(
            "No migrated data found for {country_code} at {country_dir}"
        ));
    }
    for name in ["calibration_object.json", "calibration_data.json"] {
        let path = Path::new(&country_dir).join(name);
        if path.is_file() {
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            return CalibrationData::from_json(&text);
        }
    }
    Err(format!(
        "No calibration data found for {country_code} in {country_dir}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solve_pivots() {
        let A = vec![
            vec![0.0, 2.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![3.0, 0.0, 1.0],
        ];
        let x = lu_solve(A, vec![5.0, 3.0, 6.0]).unwrap();
        for (a, b) in x.iter().zip([1.4, 1.6, 1.8]) {
            assert!((a - b).abs() < 1e-15, "{x:?}");
        }
        assert!(lu_solve(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![1.0, 1.0]).is_err());
    }

    #[test]
    fn year_quarter_names() {
        let f = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            extract_yq(&f(&["2010Q2.json", "a.txt", "2010Q1.json"]), "json").unwrap(),
            f(&["2010Q1.json", "2010Q2.json"])
        );
        assert!(extract_yq(&f(&["Quarters.json"]), "json").is_err());
    }
}
