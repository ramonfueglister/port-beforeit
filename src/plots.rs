//! Plots of the data of models, BeforeIT's package extension
//! `ext/BeforeITPlots/BeforeITPlots.jl`, with the feature `plots`.
//!
//! In BeforeIT.jl the extension is loaded together with Plots.jl and
//! StatsPlots.jl, and its functions return Plots.jl plot objects. Here they
//! return [`Plot`]s, which hold what the Julia code hands to Plots.jl: the
//! series with their colours, labels and ribbons, the title and its font
//! size, the axis limits and the ticks. [`Plot::to_svg`] and [`layout_svg`]
//! draw them as SVG with the `plotters` crate, the counterpart of
//! `savefig` and of Julia's `plot(ps..., layout = (3, 3))`. Fonts, margins and
//! axes of the drawing are those of `plotters`, not those of Plots.jl.
//!
//! ```
//! use beforeit::plots::{DEFAULT_QUANTITIES, DEFAULT_SIZE, DEFAULT_TITLEFONT, layout_svg, plot_data};
//! use beforeit::{AUSTRIA2010Q1, Model, Xoshiro};
//!
//! # fn main() -> Result<(), String> {
//! let s = &*AUSTRIA2010Q1;
//! let mut rng = Xoshiro::seed_from_u64(1);
//! let mut model = Model::new(&s.parameters, &s.initial_conditions, &mut rng)?;
//! model.run(4, &mut rng);
//! let ps = plot_data(&model, DEFAULT_TITLEFONT, &DEFAULT_QUANTITIES)?;
//! let svg = layout_svg(&ps, (3, 3), DEFAULT_SIZE)?;
//! assert!(svg.starts_with("<svg"));
//! # Ok(())
//! # }
//! ```

use std::collections::HashSet;

use plotters::coord::Shift;
use plotters::coord::ranged1d::{KeyPointHint, NoDefaultFormatting, Ranged, ValueFormatter};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

use crate::AbstractModel;
use crate::calibration::{Array, Date, Table};
use crate::data::{Data, DataVector};
use crate::julia;

/// The quantities that the functions plot by default, BeforeIT's
/// `DEFAULT_QUANTITIES`.
pub const DEFAULT_QUANTITIES: [&str; 9] = [
    "real_gdp",
    "real_household_consumption",
    "real_government_consumption",
    "real_capitalformation",
    "real_exports",
    "real_imports",
    "wages",
    "euribor",
    "gdp_deflator",
];

/// Font size of the titles in points, BeforeIT's `DEFAULT_TITLEFONT`.
pub const DEFAULT_TITLEFONT: u32 = 9;

/// Font size of the ticks in points, BeforeIT's `DEFAULT_TICKFONT`.
pub const DEFAULT_TICKFONT: u32 = 7;

/// Factor from millions to values, BeforeIT's `SCALE_FACTOR`.
pub const SCALE_FACTOR: f64 = 1.0e6;

/// Size of a figure in pixels, Plots.jl's default `size = (600, 400)`.
pub const DEFAULT_SIZE: (u32, u32) = (600, 400);

/// Pixels per inch, Plots.jl's `PX_PER_INCH`; font sizes are in points of
/// 1/72 inch.
const PX_PER_INCH: f64 = 100.0;

/// The series colours of Plots.jl, `palette(:default)`, as RGB fractions:
/// the 16 colours that PlotUtils.jl's `generate_colorscheme` computes, as
/// Julia prints them.
pub const PALETTE: [(f64, f64, f64); 16] = [
    (0.0, 0.6056031611752245, 0.9786801175696073),
    (0.8888735002725198, 0.43564919034818994, 0.2781229361419438),
    (0.2422242978521988, 0.6432750931576305, 0.3044486515341153),
    (0.7644401754934356, 0.4441117794687767, 0.8242975359232758),
    (0.6755439572114057, 0.5556623322045815, 0.09423433626639477),
    (4.821181644776295e-7, 0.6657589812923561, 0.6809969518707945),
    (0.930767491919665, 0.3674771896571412, 0.5757699667547829),
    (0.7769816661712932, 0.5097431319944513, 0.1464252569555497),
    (
        3.8077343661790943e-7,
        0.6642678029460116,
        0.5529508754522481,
    ),
    (0.558464964115081, 0.5934846564332882, 0.11748125233232104),
    (5.947623898072685e-7, 0.6608785231434254, 0.7981787608414297),
    (0.6096707676128648, 0.49918492100827777, 0.9117812665042642),
    (0.3800016049820351, 0.5510532724353506, 0.9665056985227146),
    (0.942181647954218, 0.37516423354097583, 0.4518168202944593),
    (0.8684020893043971, 0.3959893639954845, 0.7135147524811879),
    (0.42314674364630817, 0.6224954944199981, 0.19877060252130468),
];

/// One series of a plot: a line through `(x[k], y[k])`, NaN leaving a gap
/// as in Plots.jl.
#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// Half width of a band around the line, Plots.jl's `ribbon`: from
    /// `y[k] − ribbon[k]` to `y[k] + ribbon[k]`.
    pub ribbon: Option<Vec<f64>>,
    /// Opacity of the band, Plots.jl's `fillalpha`.
    pub fill_alpha: f64,
    /// Colour, an index into [`PALETTE`] (from 0).
    pub color: usize,
    /// Label in the legend.
    pub label: Option<String>,
}

/// One plot, the attributes that BeforeIT.jl passes to Plots.jl.
#[derive(Clone, Debug, PartialEq)]
pub struct Plot {
    pub title: String,
    /// Font size of the title in points.
    pub title_font: u32,
    /// Font size of the ticks in points; `None`: the default of the
    /// drawing.
    pub tick_font: Option<u32>,
    /// Whether the legend is shown.
    pub legend: bool,
    /// Axis limits; `None`: from the data.
    pub x_limits: Option<(f64, f64)>,
    pub y_limits: Option<(f64, f64)>,
    /// Positions and labels of the ticks on the x axis; `None`: automatic.
    pub x_ticks: Option<(Vec<f64>, Vec<String>)>,
    /// Rotation of the tick labels on the x axis in degrees, Plots.jl's
    /// `xrotation`. The SVG drawing does not rotate them: `plotters` turns
    /// text by multiples of 90 degrees only.
    pub x_rotation: f64,
    pub series: Vec<Series>,
}

impl Plot {
    fn new(title: String, title_font: u32, legend: bool) -> Plot {
        Plot {
            title,
            title_font,
            tick_font: None,
            legend,
            x_limits: None,
            y_limits: None,
            x_ticks: None,
            x_rotation: 0.0,
            series: Vec::new(),
        }
    }

    /// Plots.jl's colour for the next series that a recipe adds without a
    /// colour of its own: the one after the series already in the plot,
    /// wrapping around the palette (StatsPlots' `errorline`).
    fn next_color(&self) -> usize {
        self.series.len() % PALETTE.len()
    }

    /// The plot as an SVG document of `size` pixels.
    ///
    /// Returns an error if `plotters` cannot draw it.
    pub fn to_svg(&self, size: (u32, u32)) -> Result<String, String> {
        layout_svg(std::slice::from_ref(self), (1, 1), size)
    }
}

/// The plots side by side in a grid of `(rows, columns)`, filled row by
/// row, as an SVG document of `size` pixels: Julia's
/// `plot(ps..., layout = (rows, columns))`.
///
/// Returns an error if there are more plots than cells, or if `plotters`
/// cannot draw them.
pub fn layout_svg(
    plots: &[Plot],
    layout: (usize, usize),
    size: (u32, u32),
) -> Result<String, String> {
    let (rows, cols) = layout;
    if rows == 0 || cols == 0 {
        return Err("a layout needs at least one row and one column".to_string());
    }
    if plots.len() > rows * cols {
        return Err(format!(
            "{} plots do not fit in a layout of {rows} × {cols}",
            plots.len()
        ));
    }
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, size).into_drawing_area();
        root.fill(&WHITE).map_err(|e| e.to_string())?;
        let cells = root.split_evenly((rows, cols));
        for (p, cell) in plots.iter().zip(&cells) {
            draw(p, cell)?;
        }
        root.present().map_err(|e| e.to_string())?;
    }
    Ok(svg)
}

/// `get_plot_title(quantity)`.
pub fn get_plot_title(quantity: &str) -> String {
    match quantity {
        "real_gdp" => "GDP",
        "real_household_consumption" => "Household Cons.",
        "real_government_consumption" => "Gov. Cons.",
        "real_capitalformation" => "Capital Form.",
        "real_exports" => "Exports",
        "real_imports" => "Imports",
        "gdp_deflator" => "GDP Deflator",
        _ => quantity,
    }
    .to_string()
}

/// The series of `quantity` in the data of one run: the GDP deflator as
/// nominal over real GDP, `collection_time` as floats, or a series of
/// [`Data`]. `None` for the lists per sector or per bank.
fn quantity_series(data: &Data, quantity: &str) -> Result<Option<Vec<f64>>, String> {
    if quantity == "gdp_deflator" {
        let (n, r) = (&data.nominal_gdp, &data.real_gdp);
        if n.len() != r.len() {
            return Err(format!(
                "nominal_gdp has {} entries and real_gdp {}",
                n.len(),
                r.len()
            ));
        }
        return Ok(Some(n.iter().zip(r).map(|(a, b)| a / b).collect()));
    }
    if quantity == "collection_time" {
        return Ok(Some(
            data.collection_time.iter().map(|&t| t as f64).collect(),
        ));
    }
    if let Some(s) = data.series_by_name(quantity) {
        return Ok(Some(s.to_vec()));
    }
    if data.lists_by_name(quantity).is_some() {
        return Ok(None);
    }
    Err(format!("Data has no field {quantity}"))
}

/// `compute_quantity_data(data_vector, quantity)`: the series of every run
/// side by side, row `t` the values at collection `t + 1`.
fn compute_quantity_data(dv: &DataVector, quantity: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut columns = Vec::with_capacity(dv.len());
    for d in dv.iter() {
        match quantity_series(d, quantity)? {
            Some(s) => columns.push(s),
            // NaNMath.mean of a matrix of vectors throws a MethodError.
            None => {
                return Err(format!(
                    "{quantity} has a list per collection, which errorline cannot plot"
                ));
            }
        }
    }
    let T = columns.first().map_or(0, Vec::len);
    if columns.iter().any(|c| c.len() != T) {
        return Err(format!(
            "the runs have series {quantity} of different lengths"
        ));
    }
    Ok((0..T)
        .map(|t| columns.iter().map(|c| c[t]).collect())
        .collect())
}

/// NaNMath.jl's `mean` and `std` of the numbers that are not NaN (both NaN
/// if there are none, `std` also if there is one).
fn nan_mean_std(x: &[f64]) -> (f64, f64) {
    let (mut sum, mut count) = (0.0, 0usize);
    for &v in x {
        if !v.is_nan() {
            sum += v;
            count += 1;
        }
    }
    let mean = sum / count as f64;
    if mean.is_nan() {
        return (mean, mean);
    }
    let mut sum_square = 0.0;
    for &v in x {
        if !v.is_nan() {
            sum_square += (v - mean) * (v - mean);
        }
    }
    (mean, (sum_square / (count as f64 - 1.0)).sqrt())
}

/// StatsPlots' `errorline(x, y; errorstyle = :ribbon)` with its default
/// `centertype = :mean` and `errortype = :std`: a line through the mean of
/// each row of `y` with a band of one standard deviation, both ignoring NaN,
/// and `fillalpha = 0.1`. `y` is given by its rows; if their number differs
/// from the length of `x` and the number of columns equals it, `y` is
/// transposed, as StatsPlots does.
///
/// Returns an error where StatsPlots throws: if neither dimension of `y`
/// has the length of `x`.
fn errorline(
    x: Vec<f64>,
    y: Vec<Vec<f64>>,
    color: usize,
    label: Option<String>,
) -> Result<Series, String> {
    let ncols = y.first().map_or(0, Vec::len);
    if y.iter().any(|r| r.len() != ncols) {
        return Err("the rows of y have different lengths".to_string());
    }
    let y = if y.len() == x.len() {
        y
    } else if ncols == x.len() {
        (0..ncols)
            .map(|j| y.iter().map(|r| r[j]).collect())
            .collect()
    } else {
        return Err("Size of x and y do not match".to_string());
    };
    let (center, error): (Vec<f64>, Vec<f64>) = y.iter().map(|r| nan_mean_std(r)).unzip();
    Ok(Series {
        x,
        y: center,
        ribbon: Some(error),
        fill_alpha: 0.1,
        color,
        label,
    })
}

/// A line without band in the colour of its position, Plots.jl's
/// `plot(y)`.
fn line(x: Vec<f64>, y: Vec<f64>, color: usize, label: Option<String>) -> Series {
    Series {
        x,
        y,
        ribbon: None,
        fill_alpha: 1.0,
        color,
        label,
    }
}

/// `plot_data(model; titlefont, quantities)`: one plot per quantity with
/// the series of the model's data, without legend. A list per sector or
/// per bank gives one line per collection, as Plots.jl draws a vector of
/// vectors.
///
/// Returns an error for a quantity that is not a field of [`Data`] or
/// `gdp_deflator` (Julia throws).
pub fn plot_data<X>(
    model: &AbstractModel<X>,
    titlefont: u32,
    quantities: &[&str],
) -> Result<Vec<Plot>, String> {
    let data = &model.data;
    let mut plots = Vec::with_capacity(quantities.len());
    for &quantity in quantities {
        let mut p = Plot::new(get_plot_title(quantity), titlefont, false);
        match quantity_series(data, quantity)? {
            Some(y) => p
                .series
                .push(line((1..=y.len()).map(|t| t as f64).collect(), y, 0, None)),
            None => {
                for (k, y) in data
                    .lists_by_name(quantity)
                    .unwrap_or_default()
                    .iter()
                    .enumerate()
                {
                    let x = (1..=y.len()).map(|t| t as f64).collect();
                    p.series.push(line(x, y.clone(), k % PALETTE.len(), None));
                }
            }
        }
        plots.push(p);
    }
    Ok(plots)
}

/// `plot_data_vector(model_vector; titlefont, quantities, t_start, t_end)`:
/// one plot per quantity with the mean over the runs and a band of one
/// standard deviation, from collection `t_start` to `t_end`, counted from 1
/// as in Julia, which are also the x values; `None` for the first and the
/// last collection. Without legend.
///
/// Returns an error for an unknown quantity, a list per sector or per bank
/// (Julia throws in StatsPlots), series of different lengths, or no models
/// (Julia throws).
pub fn plot_data_vector<X>(
    model_vector: &[AbstractModel<X>],
    titlefont: u32,
    quantities: &[&str],
    t_start: Option<i64>,
    t_end: Option<i64>,
) -> Result<Vec<Plot>, String> {
    let data_vector = DataVector::from_models(model_vector);
    let time_length = data_vector
        .iter()
        .next()
        .ok_or("there are no models")?
        .wages
        .len() as i64;
    let start_idx = t_start.map_or(1, |t| t.max(1));
    let end_idx = t_end.map_or(time_length, |t| t.min(time_length));
    let mut plots = Vec::with_capacity(quantities.len());
    for &quantity in quantities {
        let full_data = compute_quantity_data(&data_vector, quantity)?;
        let data = (start_idx..=end_idx)
            .map(|t| full_data.get((t - 1) as usize).cloned())
            .collect::<Option<Vec<Vec<f64>>>>()
            .ok_or_else(|| format!("{quantity} has fewer collections than wages"))?;
        let x = (start_idx..=end_idx).map(|t| t as f64).collect();
        let mut p = Plot::new(get_plot_title(quantity), titlefont, false);
        p.series.push(errorline(x, data, p.next_color(), None)?);
        plots.push(p);
    }
    Ok(plots)
}

/// `plot_data_vectors(model_vectors; titlefont, quantities)`: as
/// [`plot_data_vector`] for each vector of models, all in the same plot,
/// each in the next colour, over the collections of the first.
///
/// Returns the errors of [`plot_data_vector`], and an error if a vector
/// has series of another length than the first and not as many runs as
/// the first has collections (StatsPlots throws).
pub fn plot_data_vectors<X>(
    model_vectors: &[Vec<AbstractModel<X>>],
    titlefont: u32,
    quantities: &[&str],
) -> Result<Vec<Plot>, String> {
    let data_vectors: Vec<DataVector> = model_vectors
        .iter()
        .map(|m| DataVector::from_models(m))
        .collect();
    let time_length = data_vectors
        .first()
        .and_then(|dv| dv.iter().next())
        .ok_or("there are no models")?
        .wages
        .len();
    let mut plots = Vec::with_capacity(quantities.len());
    for &quantity in quantities {
        let mut p = Plot::new(get_plot_title(quantity), titlefont, false);
        for dv in &data_vectors {
            let data = compute_quantity_data(dv, quantity)?;
            let x = (1..=time_length).map(|t| t as f64).collect();
            p.series.push(errorline(x, data, p.next_color(), None)?);
        }
        plots.push(p);
    }
    Ok(plots)
}

/// The numbers of an entry of a table, column by column; an error if it is
/// missing.
fn numbers<'a>(t: &'a Table, key: &str, what: &str) -> Result<&'a Array, String> {
    t.get(key).ok_or_else(|| format!("{what} has no \"{key}\""))
}

/// Julia's `minimum` and `maximum`: NaN if there is a NaN; an error for a
/// `missing` value (Plots.jl cannot use the result) or no values.
fn extrema(values: impl Iterator<Item = Option<f64>>, what: &str) -> Result<(f64, f64), String> {
    let mut m: Option<(f64, f64)> = None;
    for v in values {
        let v = v.ok_or_else(|| format!("{what} has a missing value"))?;
        m = Some(match m {
            None => (v, v),
            Some((lo, hi)) => (julia::min(lo, v), julia::max(hi, v)),
        });
    }
    m.ok_or_else(|| format!("{what} has no values"))
}

/// `parse_variable_name(varname)`: the time axis and the title.
fn parse_variable_name(varname: &str) -> (&'static str, String) {
    if varname.chars().count() > 9 && varname.ends_with("quarterly") {
        let n = varname.chars().count();
        ("quarters_num", varname.chars().take(n - 10).collect())
    } else {
        ("years_num", varname.to_string())
    }
}

/// `generate_year_ticks(time_numbers)`: the first date number of each year
/// and the year.
///
/// Returns an error for a date number that is not an integer (Julia stores
/// the positions as `Int` and throws an `InexactError`) or missing.
fn generate_year_ticks(
    time_numbers: &[Option<f64>],
    what: &str,
) -> Result<(Vec<f64>, Vec<String>), String> {
    let mut num_ticks = Vec::new();
    let mut year_ticks: Vec<i32> = Vec::new();
    let mut seen = HashSet::new();
    for &t in time_numbers {
        let t = t.ok_or_else(|| format!("{what} has a missing date"))?;
        if !t.is_finite() {
            return Err(format!("{what}: the date number {t} is not finite"));
        }
        let y = Date::from_datenum(t).year();
        if seen.insert(y) {
            if t.fract() != 0.0 {
                return Err(format!("{what}: the date number {t} is not an integer"));
            }
            num_ticks.push(t);
            year_ticks.push(y);
        }
    }
    Ok((num_ticks, year_ticks.iter().map(i32::to_string).collect()))
}

/// `plot_model_vs_real(model, real, varname; crop)`: the data `real` as a
/// line ("Real") and the predictions `model` as their mean with a band of
/// one standard deviation ("Model"), both times [`SCALE_FACTOR`], against
/// the date numbers of `quarters_num` for a name that ends in `quarterly`
/// and of `years_num` otherwise, with the years as ticks. With `crop`, the
/// axes show the period of the predictions and the values of both in it;
/// otherwise the limits come from the data, and the ticks from `real`.
///
/// `model` is a table of predictions ([`crate::Predictions::to_table`]),
/// `real` one of data, such as the `data` of [`crate::ITALY_CALIBRATION`].
///
/// Returns an error if an entry is missing or has a missing value where
/// Julia uses it, or if the predictions do not have one row per date
/// (StatsPlots throws).
pub fn plot_model_vs_real(
    model: &Table,
    real: &Table,
    varname: &str,
    crop: bool,
) -> Result<Plot, String> {
    let (x_nums, title) = parse_variable_name(varname);
    let model_x = numbers(model, x_nums, "model")?;
    let model_y = numbers(model, varname, "model")?;
    let real_x = numbers(real, x_nums, "real")?;
    let real_y = numbers(real, varname, "real")?;

    let mut p = Plot::new(title, DEFAULT_TITLEFONT, true);
    if crop {
        // compute_axis_limits
        let (min_x, max_x) = extrema(model_x.data.iter().copied(), "model x")?;
        let mut in_range = Vec::new();
        for (x, y) in real_x.data.iter().zip(&real_y.data) {
            let x = x.ok_or("real has a missing date")?;
            if min_x <= x && x <= max_x {
                in_range.push(*y);
            }
        }
        let (min_y, max_y) = extrema(in_range.into_iter(), &format!("real {varname}"))?;
        let (lo, hi) = extrema(model_y.data.iter().copied(), &format!("model {varname}"))?;
        let (min_y, max_y) = (julia::min(min_y, lo), julia::max(max_y, hi));
        p.x_limits = Some((min_x, max_x));
        p.y_limits = Some((min_y * SCALE_FACTOR, max_y * SCALE_FACTOR));
    }
    let tick_source = if crop { model_x } else { real_x };
    p.x_ticks = Some(generate_year_ticks(&tick_source.data, x_nums)?);
    p.x_rotation = 20.0;
    p.tick_font = Some(DEFAULT_TICKFONT);

    let scaled = |a: &Array| -> Vec<f64> {
        a.data
            .iter()
            .map(|v| v.map_or(f64::NAN, |v| SCALE_FACTOR * v))
            .collect()
    };
    let real_x: Vec<f64> = real_x.data.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let real_y = scaled(real_y);
    if real_x.len() != real_y.len() {
        return Err(format!(
            "real has {} dates and {} values of {varname}",
            real_x.len(),
            real_y.len()
        ));
    }
    p.series.push(line(
        real_x,
        real_y,
        p.next_color(),
        Some("Real".to_string()),
    ));

    let x: Vec<f64> = model_x.data.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let y = scaled(model_y);
    let rows = model_y.size.first().copied().unwrap_or(0);
    let cols = y.len().checked_div(rows).unwrap_or(0);
    let y_rows: Vec<Vec<f64>> = (0..rows)
        .map(|t| (0..cols).map(|j| y[t + rows * j]).collect())
        .collect();
    let color = p.next_color();
    p.series
        .push(errorline(x, y_rows, color, Some("Model".to_string()))?);
    Ok(p)
}

/// RGB colour of a palette entry.
fn rgb(color: usize) -> RGBColor {
    let (r, g, b) = PALETTE[color % PALETTE.len()];
    let byte = |c: f64| (c * 255.0).round() as u8;
    RGBColor(byte(r), byte(g), byte(b))
}

fn font_px(points: u32) -> f64 {
    f64::from(points) * PX_PER_INCH / 72.0
}

/// The range of the finite values, widened if it is a single value.
fn auto_range(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in values.filter(|v| v.is_finite()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo > hi {
        (0.0, 1.0)
    } else if lo == hi {
        let d = if lo == 0.0 { 1.0 } else { lo.abs() * 0.01 };
        (lo - d, hi + d)
    } else {
        (lo, hi)
    }
}

/// The runs of consecutive points for which `keep` holds.
fn runs(n: usize, keep: impl Fn(usize) -> bool) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = None;
    for k in 0..=n {
        match (k < n && keep(k), start) {
            (true, None) => start = Some(k),
            (false, Some(s)) => {
                out.push(s..k);
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// A tick label of the y axis: as `plotters` writes it, or with a power of
/// ten (`3.95e11`) for magnitudes from 10^5 on and below 10^-4, whose
/// digits would not fit beside the axis.
fn short_number(v: &f64) -> String {
    let a = v.abs();
    if a == 0.0 || (1e-4..1e5).contains(&a) {
        return RangedCoordf64::format(v);
    }
    let text = format!("{v:.3e}");
    match text.split_once('e') {
        Some((mantissa, exp)) if mantissa.contains('.') => {
            let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
            format!("{mantissa}e{exp}")
        }
        _ => text,
    }
}

/// An x axis with the ticks of a [`Plot`]: given positions and labels, or
/// those of `plotters`.
struct Axis {
    range: RangedCoordf64,
    ticks: Option<(Vec<f64>, Vec<String>)>,
}

impl Ranged for Axis {
    type FormatOption = NoDefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        self.range.map(value, limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<f64> {
        match &self.ticks {
            Some((positions, _)) => {
                let r = self.range.range();
                positions
                    .iter()
                    .copied()
                    .filter(|v| r.contains(v) || *v == r.end)
                    .collect()
            }
            None => self.range.key_points(hint),
        }
    }

    fn range(&self) -> std::ops::Range<f64> {
        self.range.range()
    }
}

impl ValueFormatter<f64> for Axis {
    fn format_ext(&self, value: &f64) -> String {
        match &self.ticks {
            Some((positions, labels)) => positions
                .iter()
                .position(|q| q == value)
                .and_then(|k| labels.get(k).cloned())
                .unwrap_or_default(),
            None => RangedCoordf64::format(value),
        }
    }
}

fn draw<DB: DrawingBackend>(p: &Plot, area: &DrawingArea<DB, Shift>) -> Result<(), String> {
    let err = |e: DrawingAreaErrorKind<DB::ErrorType>| e.to_string();
    let (x0, x1) = p
        .x_limits
        .unwrap_or_else(|| auto_range(p.series.iter().flat_map(|s| s.x.iter().copied())));
    let (y0, y1) = p.y_limits.unwrap_or_else(|| {
        auto_range(p.series.iter().flat_map(|s| {
            let band = s
                .ribbon
                .iter()
                .flat_map(move |r| s.y.iter().zip(r).flat_map(|(y, e)| [y - e, y + e]));
            s.y.iter().copied().chain(band)
        }))
    });
    let x_axis = Axis {
        range: RangedCoordf64::from(x0..x1),
        ticks: p.x_ticks.clone(),
    };
    let label_font = p.tick_font.map(font_px);
    let mut chart = ChartBuilder::on(area)
        .caption(&p.title, ("sans-serif", font_px(p.title_font)))
        .margin(5)
        .x_label_area_size(label_font.map_or(20.0, |f| 2.0 * f) as u32)
        .y_label_area_size(label_font.map_or(45.0, |f| 5.0 * f) as u32)
        .build_cartesian_2d(x_axis, y0..y1)
        .map_err(err)?;
    let mut mesh = chart.configure_mesh();
    mesh.light_line_style(TRANSPARENT)
        .bold_line_style(BLACK.mix(0.1))
        .x_labels(6)
        .y_labels(6)
        .y_label_formatter(&short_number);
    if let Some(f) = label_font {
        mesh.label_style(("sans-serif", f));
    }
    mesh.draw().map_err(err)?;

    let inside = |x: f64| x >= x0.min(x1) && x <= x0.max(x1);
    for s in &p.series {
        let color = rgb(s.color);
        if let Some(r) = &s.ribbon {
            let ok = |k: usize| inside(s.x[k]) && s.y[k].is_finite() && r[k].is_finite();
            for run in runs(s.x.len(), ok) {
                let upper = run.clone().map(|k| (s.x[k], s.y[k] + r[k]));
                let lower = run.clone().rev().map(|k| (s.x[k], s.y[k] - r[k]));
                let band: Vec<(f64, f64)> = upper.chain(lower).collect();
                chart
                    .draw_series(std::iter::once(Polygon::new(
                        band,
                        color.mix(s.fill_alpha).filled(),
                    )))
                    .map_err(err)?;
            }
        }
        let ok = |k: usize| inside(s.x[k]) && s.y[k].is_finite();
        for (n, run) in runs(s.x.len(), ok).into_iter().enumerate() {
            let points: Vec<(f64, f64)> = run.map(|k| (s.x[k], s.y[k])).collect();
            let drawn = chart
                .draw_series(std::iter::once(PathElement::new(points, color)))
                .map_err(err)?;
            if let (true, Some(label), 0) = (p.legend, &s.label, n) {
                drawn
                    .label(label.as_str())
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], color));
            }
        }
    }
    if p.legend && p.series.iter().any(|s| s.label.is_some()) {
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .draw()
            .map_err(err)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AUSTRIA2010Q1, Model, Xoshiro};

    fn models(n: usize, T: usize, seed: u64) -> Vec<Model> {
        let s = &*AUSTRIA2010Q1;
        let mut rng = Xoshiro::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut m = Model::new(&s.parameters, &s.initial_conditions, &mut rng).unwrap();
                m.run(T, &mut rng);
                m
            })
            .collect()
    }

    #[test]
    fn palette_and_titles() {
        // Plots.jl's first colours, #009AFA, #E36F47 and #3EA44E.
        assert_eq!(rgb(0), RGBColor(0x00, 0x9A, 0xFA));
        assert_eq!(rgb(1), RGBColor(0xE3, 0x6F, 0x47));
        assert_eq!(rgb(2), RGBColor(0x3E, 0xA4, 0x4E));
        assert_eq!(rgb(16), rgb(0));
        assert_eq!(get_plot_title("real_gdp"), "GDP");
        assert_eq!(get_plot_title("gdp_deflator"), "GDP Deflator");
        assert_eq!(get_plot_title("wages"), "wages");
        assert_eq!(font_px(9), 12.5);
    }

    #[test]
    fn errorline_as_statsplots() {
        let y = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, f64::NAN, 5.0],
            vec![f64::NAN; 3],
        ];
        let s = errorline(vec![1.0, 2.0, 3.0], y.clone(), 4, None).unwrap();
        assert_eq!(s.y[..2], [2.0, 3.0]);
        assert!(s.y[2].is_nan());
        let r = s.ribbon.unwrap();
        assert_eq!(r[..2], [1.0, 8.0f64.sqrt()]);
        assert!(r[2].is_nan());
        assert_eq!((s.color, s.fill_alpha), (4, 0.1));
        // One run: no band.
        let s = errorline(vec![1.0, 2.0], vec![vec![3.0], vec![4.0]], 0, None).unwrap();
        assert!(s.ribbon.unwrap().iter().all(|e| e.is_nan()));
        // y is transposed if its columns match x, else an error.
        let s = errorline(
            vec![1.0, 2.0, 3.0],
            vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]],
            0,
            None,
        )
        .unwrap();
        assert_eq!(s.y, [2.0, 3.0, 4.0]);
        assert!(errorline(vec![1.0, 2.0], y, 0, None).is_err());
    }

    #[test]
    fn data_of_one_model() {
        let m = &models(1, 3, 1)[0];
        let ps = plot_data(m, DEFAULT_TITLEFONT, &DEFAULT_QUANTITIES).unwrap();
        assert_eq!(ps.len(), 9);
        let titles: Vec<&str> = ps.iter().map(|p| p.title.as_str()).collect();
        assert_eq!(
            titles,
            [
                "GDP",
                "Household Cons.",
                "Gov. Cons.",
                "Capital Form.",
                "Exports",
                "Imports",
                "wages",
                "euribor",
                "GDP Deflator"
            ]
        );
        let gdp = &ps[0];
        assert_eq!(
            (gdp.title_font, gdp.legend, gdp.series.len()),
            (9, false, 1)
        );
        assert_eq!(gdp.series[0].y, m.data.real_gdp);
        assert_eq!(gdp.series[0].x, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gdp.series[0].ribbon, None);
        let deflator: Vec<f64> = (0..4)
            .map(|t| m.data.nominal_gdp[t] / m.data.real_gdp[t])
            .collect();
        assert_eq!(ps[8].series[0].y, deflator);
        // A list per bank: one line per collection.
        let p = &plot_data(m, 10, &["equity_per_bank", "collection_time"]).unwrap();
        assert_eq!(p[0].series.len(), 4);
        assert_eq!(p[0].series[1].y, m.data.equity_per_bank[1]);
        assert_eq!(p[0].series[1].color, 1);
        assert_eq!(p[1].series[0].y, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p[1].title_font, 10);
        assert!(plot_data(m, 9, &["gdp"]).is_err());
    }

    #[test]
    fn data_of_several_models() {
        let ms = models(3, 4, 2);
        let ps =
            plot_data_vector(&ms, DEFAULT_TITLEFONT, &["real_gdp", "euribor"], None, None).unwrap();
        let s = &ps[0].series[0];
        assert_eq!(s.x, [1.0, 2.0, 3.0, 4.0, 5.0]);
        for t in 0..5 {
            let v: Vec<f64> = ms.iter().map(|m| m.data.real_gdp[t]).collect();
            let mean = v.iter().sum::<f64>() / 3.0;
            let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / 2.0;
            assert!((s.y[t] - mean).abs() <= 1e-12 * mean);
            assert!((s.ribbon.as_ref().unwrap()[t] - var.sqrt()).abs() <= 1e-9 * mean);
        }
        assert_eq!((s.color, ps[1].title.as_str()), (0, "euribor"));
        // From collection 2 to 4, and limits beyond the data.
        let ps = plot_data_vector(&ms, 9, &["wages"], Some(2), Some(4)).unwrap();
        assert_eq!(ps[0].series[0].x, [2.0, 3.0, 4.0]);
        let ps = plot_data_vector(&ms, 9, &["wages"], Some(-3), Some(99)).unwrap();
        assert_eq!(ps[0].series[0].x.len(), 5);
        assert!(
            plot_data_vector(&ms, 9, &["wages"], Some(4), Some(2)).unwrap()[0].series[0]
                .x
                .is_empty()
        );
        assert!(plot_data_vector(&ms, 9, &["roe_per_bank"], None, None).is_err());
        assert!(plot_data_vector::<crate::Standard>(&[], 9, &["wages"], None, None).is_err());

        // Several vectors of models in one plot, each in the next colour.
        let other = models(2, 4, 3);
        let ps = plot_data_vectors(&[ms.clone(), other], 9, &DEFAULT_QUANTITIES).unwrap();
        assert_eq!(ps.len(), 9);
        let colors: Vec<usize> = ps[0].series.iter().map(|s| s.color).collect();
        assert_eq!(colors, [0, 1]);
        assert_eq!(
            ps[0].series[0],
            plot_data_vector(&ms, 9, &["real_gdp"], None, None).unwrap()[0].series[0]
        );
        // Series of another length.
        assert!(plot_data_vectors(&[ms, models(2, 2, 4)], 9, &["wages"]).is_err());
    }

    fn quarter(y: i32, q: u32) -> f64 {
        Date::new(y, 3 * q, if q == 1 || q == 4 { 31 } else { 30 }).datenum()
    }

    #[test]
    fn model_against_data() {
        // Data from 2009Q1 to 2011Q4, predictions from 2010Q2 to 2011Q1.
        let real_q: Vec<f64> = (2009..=2011)
            .flat_map(|y| (1..=4).map(move |q| quarter(y, q)))
            .collect();
        let real_v: Vec<f64> = (0..12).map(|k| 1.0 + 0.1 * k as f64).collect();
        let mut real = Table::new();
        real.insert("quarters_num".into(), Array::vector(&real_q));
        real.insert(
            "real_gdp_quarterly".into(),
            Array::matrix(&real_v.iter().map(|&v| vec![v]).collect::<Vec<_>>()),
        );
        let model_q = vec![
            quarter(2010, 2),
            quarter(2010, 3),
            quarter(2010, 4),
            quarter(2011, 1),
        ];
        let runs = vec![
            vec![1.5, 1.7],
            vec![1.6, 2.4],
            vec![1.7, 1.9],
            vec![1.8, 2.0],
        ];
        let mut model = Table::new();
        model.insert("quarters_num".into(), Array::vector(&model_q));
        model.insert("real_gdp_quarterly".into(), Array::matrix(&runs));

        let p = plot_model_vs_real(&model, &real, "real_gdp_quarterly", true).unwrap();
        assert_eq!(p.title, "real_gdp");
        assert_eq!(
            (p.title_font, p.tick_font, p.x_rotation, p.legend),
            (9, Some(7), 20.0, true)
        );
        assert_eq!(p.x_limits, Some((model_q[0], model_q[3])));
        // The data from 2010Q2 to 2011Q1 lie between 1.5 and 1.8, the runs
        // between 1.5 and 2.4.
        let (lo, hi) = p.y_limits.unwrap();
        assert_eq!((lo, hi), (1.5 * SCALE_FACTOR, 2.4 * SCALE_FACTOR));
        assert_eq!(
            p.x_ticks,
            Some((
                vec![model_q[0], model_q[3]],
                vec!["2010".to_string(), "2011".to_string()]
            ))
        );
        let (data, pred) = (&p.series[0], &p.series[1]);
        assert_eq!(
            (data.label.as_deref(), data.color, &data.ribbon),
            (Some("Real"), 0, &None)
        );
        assert_eq!(data.x, real_q);
        assert_eq!(data.y[3], real_v[3] * SCALE_FACTOR);
        assert_eq!((pred.label.as_deref(), pred.color), (Some("Model"), 1));
        assert_eq!(pred.x, model_q);
        assert!((pred.y[1] - 2.0 * SCALE_FACTOR).abs() < 1e-6);
        let band = pred.ribbon.as_ref().unwrap();
        assert!((band[1] - 0.32f64.sqrt() * SCALE_FACTOR).abs() < 1e-6);

        // Without crop: limits from the data, the years of the data as ticks.
        let p = plot_model_vs_real(&model, &real, "real_gdp_quarterly", false).unwrap();
        assert_eq!((p.x_limits, p.y_limits), (None, None));
        let (positions, years) = p.x_ticks.unwrap();
        assert_eq!(positions, [real_q[0], real_q[4], real_q[8]]);
        assert_eq!(years, ["2009", "2010", "2011"]);

        // A name without "quarterly" uses years_num.
        let e = plot_model_vs_real(&model, &real, "real_gdp", true).unwrap_err();
        assert_eq!(e, "model has no \"years_num\"");
        // A missing value of the data in the period of the predictions.
        let mut gap = real.clone();
        gap.get_mut("real_gdp_quarterly").unwrap().data[5] = None;
        assert!(plot_model_vs_real(&model, &gap, "real_gdp_quarterly", true).is_err());
        // Outside, the line has a gap.
        let mut gap = real.clone();
        gap.get_mut("real_gdp_quarterly").unwrap().data[0] = None;
        let p = plot_model_vs_real(&model, &gap, "real_gdp_quarterly", true).unwrap();
        assert!(p.series[0].y[0].is_nan());
    }

    #[test]
    fn drawn_as_svg() {
        let ms = models(2, 3, 5);
        let ps = plot_data_vector(&ms, DEFAULT_TITLEFONT, &DEFAULT_QUANTITIES, None, None).unwrap();
        let svg = layout_svg(&ps, (3, 3), DEFAULT_SIZE).unwrap();
        assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
        for p in &ps {
            assert!(svg.contains(&format!("\n{}\n", p.title)), "{}", p.title);
        }
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("width=\"600\"") && svg.contains("height=\"400\""));
        let one = ps[0].to_svg((300, 200)).unwrap();
        assert!(one.contains("\nGDP\n") && one.contains("width=\"300\""));
        assert!(layout_svg(&ps, (2, 4), DEFAULT_SIZE).is_err());
        assert!(layout_svg(&[], (0, 1), DEFAULT_SIZE).is_err());
        assert!(layout_svg(&[], (1, 1), DEFAULT_SIZE).is_ok());
        // Legend and explicit ticks.
        let mut p = Plot::new("t".into(), 9, true);
        p.series.push(line(
            vec![1.0, 2.0, f64::NAN, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
            0,
            Some("Real".into()),
        ));
        p.x_ticks = Some((vec![1.0, 3.0], vec!["one".into(), "three".into()]));
        let svg = p.to_svg(DEFAULT_SIZE).unwrap();
        for text in ["Real", "one", "three"] {
            assert!(svg.contains(&format!("\n{text}\n")), "{text}");
        }
        assert_eq!(short_number(&395e9), "3.95e11");
        assert_eq!(short_number(&2.5e-5), "2.5e-5");
    }
}
