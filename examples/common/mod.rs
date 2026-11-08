//! Helpers shared by the examples.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use beforeit::calibration::zenodo_calibration_url;
use beforeit::{AbstractModel, CalibrationData, Date, Model};

/// The folder for the files that an example writes,
/// `target/example-output/<example>` in the crate, created if needed.
pub fn output_dir(example: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("example-output")
        .join(example);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

/// Writes `text` to `dir/file` and prints the path.
pub fn save(dir: &Path, file: &str, text: &str) -> Result<(), String> {
    let path = dir.join(file);
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// The model with the rules of the model type `X`: the counterpart of
/// redefining a method for `Bit.Model` in Julia, after which the same model
/// follows the new rule.
pub fn with_rules<X>(m: Model, ext: X) -> AbstractModel<X> {
    AbstractModel {
        w_act: m.w_act,
        w_inact: m.w_inact,
        firms: m.firms,
        banks: m.banks,
        cb: m.cb,
        gov: m.gov,
        rotw: m.rotw,
        agg: m.agg,
        prop: m.prop,
        data: m.data,
        ext,
    }
}

/// `1.0, 2.0, …, n`, the x values of Plots.jl's `plot(y)`.
pub fn one_to(n: usize) -> Vec<f64> {
    (1..=n).map(|t| t as f64).collect()
}

/// Makes `target/example-output` the working directory, so that the paths
/// of an example that reads and writes files relative to the working
/// directory, as the Julia example does (such as `data/italy`), are the
/// same for all examples.
pub fn enter_work_dir() -> Result<PathBuf, String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("example-output");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if let Ok(start) = std::env::current_dir() {
        START_DIR.get_or_init(|| start);
    }
    std::env::set_current_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    println!("working directory {}", dir.display());
    Ok(dir)
}

/// The working directory in which the example started, before
/// [`enter_work_dir`] changed it.
static START_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The calibration object of `country` from the Zenodo record, in place of
/// BeforeIT's `download_zenodo_calibration_object(country)`, which
/// downloads it: the file `<country>.json` in the folder
/// `BEFOREIT_CALIBRATION_DIR`, converted from the JLD2 file of the record
/// by `data/convert_calibration.py`, as for the ignored test of
/// `tests/calibration.rs`. A relative folder is taken from the directory in
/// which the example was started, also after [`enter_work_dir`].
pub fn zenodo_calibration_object(country: &str) -> Result<CalibrationData, String> {
    let url = zenodo_calibration_url(country)?;
    let dir = std::env::var("BEFOREIT_CALIBRATION_DIR").map_err(|_| {
        format!(
            "set BEFOREIT_CALIBRATION_DIR to a folder with {country}.json: download {url} and \
             convert it with python3 data/convert_calibration.py zenodo \
             {country}_calibration_object.jld2 {country}.json"
        )
    })?;
    let dir = PathBuf::from(dir);
    let dir = match START_DIR.get() {
        Some(start) if dir.is_relative() => start.join(dir),
        _ => dir,
    };
    let path = dir.join(format!("{country}.json"));
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e} (download {url} and convert it with python3 \
             data/convert_calibration.py zenodo {country}_calibration_object.jld2 {})",
            path.display(),
            path.display()
        )
    })?;
    CalibrationData::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The date as Julia prints a `DateTime` at midnight, `2016-12-31T00:00:00`.
pub fn datetime(d: Date) -> String {
    format!("{:04}-{:02}-{:02}T00:00:00", d.year(), d.month(), d.day())
}
