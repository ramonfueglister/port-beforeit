//! Predictions against the data, BeforeIT `examples/compare_model_vs_real.jl`:
//! the predictions of one quarter for Italy against the data, for six
//! series by calendar year and by quarter.
//!
//! ```sh
//! cargo run --release --features plots --example compare_model_vs_real [file]
//! ```
//!
//! The Julia example reads the predictions from 2015Q1 in
//! `data/italy/abm_predictions/2015Q1.jld2` of the working directory, a
//! file that BeforeIT.jl does not contain. The example reads the port's
//! JSON file of the same name in `target/example-output`, or the file given
//! as argument; `prediction_pipeline_multiple` writes such files for 2014Q1
//! to 2014Q4, for example
//! `target/example-output/data/italy/abm_predictions/2014Q2.json`. The plots
//! are written as SVG to `target/example-output/compare_model_vs_real`.

mod common;

use std::path::PathBuf;

use beforeit::ITALY_CALIBRATION;
use beforeit::calibration::table_from_json;
use beforeit::plots::{DEFAULT_SIZE, Legend, layout_svg, plot_model_vs_real};

fn main() -> Result<(), String> {
    let dir = common::output_dir("compare_model_vs_real");

    // The data from 1996.
    let real_data = &ITALY_CALIBRATION.data;

    // The predictions from 2015Q1.
    let file = match std::env::args().nth(1) {
        Some(file) => PathBuf::from(file),
        None => common::enter_work_dir()?.join("data/italy/abm_predictions/2015Q1.json"),
    };
    let text = std::fs::read_to_string(&file).map_err(|e| {
        format!(
            "{}: {e} (prediction_pipeline_multiple writes files of predictions)",
            file.display()
        )
    })?;
    let model = table_from_json(&text)?;

    // Real GDP, household consumption, fixed capital formation, government
    // consumption, exports and imports by calendar year, one plot each.
    for v in [
        "real_gdp",
        "real_household_consumption",
        "real_fixed_capitalformation",
        "real_government_consumption",
        "real_exports",
        "real_imports",
    ] {
        let p = plot_model_vs_real(&model, real_data, v, true)?;
        common::save(&dir, &format!("{v}.svg"), &p.to_svg(DEFAULT_SIZE)?)?;
    }

    // The same by quarter.
    let ps = [
        "real_gdp_quarterly",
        "real_household_consumption_quarterly",
        "real_fixed_capitalformation_quarterly",
        "real_government_consumption_quarterly",
        "real_exports_quarterly",
        "real_imports_quarterly",
    ]
    .iter()
    .map(|v| {
        let mut p = plot_model_vs_real(&model, real_data, v, true)?;
        p.legend = Legend::None;
        Ok(p)
    })
    .collect::<Result<Vec<_>, String>>()?;
    common::save(
        &dir,
        "quarterly.svg",
        &layout_svg(&ps, (3, 2), DEFAULT_SIZE)?,
    )?;
    Ok(())
}
