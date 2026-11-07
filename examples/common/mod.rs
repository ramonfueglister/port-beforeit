//! Helpers shared by the examples.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use beforeit::{AbstractModel, Model};

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
