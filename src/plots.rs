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
//! The examples of BeforeIT.jl also draw plots of their own with Plots.jl,
//! StatsPlots.jl and, in one case, CairoMakie. [`Plot::new`] and its methods
//! build those: lines ([`Plot::add_line`]) with their colour, width, style
//! and markers, `errorline` with standard deviation or standard error
//! ([`Plot::add_errorline`]), horizontal lines ([`Plot::add_hline`]),
//! heatmaps ([`Plot::heatmap`]) with annotations ([`Plot::annotate`]), bars
//! ([`Plot::bars`]), axis labels, logarithmic axes and a title over a layout
//! ([`layout_svg_with_title`]).
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
use plotters::element::DashedPathElement;
use plotters::prelude::*;
use plotters::style::Color as _;
use plotters::style::text_anchor::{HPos, Pos, VPos};

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

/// Plots.jl's default font sizes in points (Plots 1.40, `args.jl` and
/// `components.jl`): `titlefontsize`, `tickfontsize`, `guidefontsize`,
/// the legend font and `plot_titlefontsize`.
const TITLE_FONT: u32 = 14;
const TICK_FONT: u32 = 8;
const GUIDE_FONT: u32 = 11;
const LEGEND_FONT: u32 = 8;
const PLOT_TITLE_FONT: u32 = 16;

/// Plots.jl's `plot_titlevspan`: the share of the height that the title
/// over a layout takes.
const PLOT_TITLE_VSPAN: f64 = 0.05;

/// Plots.jl's `fillalpha` of a ribbon without one of its own
/// (`make_fillrange_from_ribbon`).
const RIBBON_ALPHA: f64 = 0.5;

/// Plots.jl's `markersize` and `markerstrokewidth`.
const MARKER_SIZE: f64 = 4.0;
const MARKER_STROKE: u32 = 1;

/// Opacity of the grid lines, Plots.jl's `gridalpha` and `minorgridalpha`.
const GRID_ALPHA: f64 = 0.1;
const MINOR_GRID_ALPHA: f64 = 0.05;

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

/// A colour: an entry of Plots.jl's palette or RGB bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    /// The entry of [`PALETTE`], counted from 0 and wrapping around;
    /// Julia's `color = k` is `Palette(k - 1)`.
    Palette(usize),
    /// Red, green and blue from 0 to 255.
    Rgb(u8, u8, u8),
}

impl Color {
    /// The named colours of Colors.jl that the examples use.
    pub const BLACK: Color = Color::Rgb(0, 0, 0);
    pub const WHITE: Color = Color::Rgb(255, 255, 255);
    pub const GRAY: Color = Color::Rgb(128, 128, 128);
    pub const STEELBLUE: Color = Color::Rgb(70, 130, 180);
    pub const CRIMSON: Color = Color::Rgb(220, 20, 60);

    fn rgb(self) -> RGBColor {
        match self {
            Color::Palette(k) => rgb_of(PALETTE[k % PALETTE.len()]),
            Color::Rgb(r, g, b) => RGBColor(r, g, b),
        }
    }
}

/// RGB bytes of RGB fractions.
fn rgb_of((r, g, b): (f64, f64, f64)) -> RGBColor {
    let byte = |c: f64| (c * 255.0).round() as u8;
    RGBColor(byte(r), byte(g), byte(b))
}

/// Line style, Plots.jl's `linestyle`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LineStyle {
    #[default]
    Solid,
    Dash,
    Dot,
}

/// Marker at each point, Plots.jl's `markershape`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Marker {
    #[default]
    None,
    Circle,
    Square,
    Diamond,
}

/// The error band of [`Plot::add_errorline`], StatsPlots' `errortype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorType {
    /// One standard deviation of the runs, `:std`.
    Std,
    /// The standard deviation divided by the square root of the number of
    /// runs, `:sem`.
    Sem,
}

/// Whether and where the legend is shown, Plots.jl's `legend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Legend {
    /// No legend, `legend = false`.
    None,
    /// The corner chosen as Plots.jl's `_guess_best_legend_position` does,
    /// `legend = true` and the default.
    Best,
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
}

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
    pub color: Color,
    /// Label in the legend; `None` for none, Plots.jl's `label = false`.
    pub label: Option<String>,
    /// Width of the line in pixels, Plots.jl's `linewidth`.
    pub line_width: f64,
    pub line_style: LineStyle,
    pub marker: Marker,
}

/// A horizontal line across the plot, Plots.jl's `hline!`.
#[derive(Clone, Debug, PartialEq)]
pub struct HLine {
    pub y: f64,
    pub color: Color,
    pub line_style: LineStyle,
    pub line_width: f64,
}

/// A colour gradient of ColorSchemes.jl.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gradient {
    /// `:RdBu`, which PlotUtils.jl takes as ColorBrewer's `RdBu_11`.
    RdBu,
    /// `:YlOrRd`, which PlotUtils.jl takes as ColorBrewer's `YlOrRd_8`.
    YlOrRd,
}

/// ColorSchemes.jl's `RdBu_11` (`data/colorbrewerschemes.jl`).
const RDBU_11: [(f64, f64, f64); 11] = [
    (0.404, 0.0, 0.122),
    (0.698, 0.094, 0.169),
    (0.839, 0.376, 0.302),
    (0.957, 0.647, 0.51),
    (0.992, 0.859, 0.78),
    (0.969, 0.969, 0.969),
    (0.82, 0.898, 0.941),
    (0.573, 0.773, 0.871),
    (0.263, 0.576, 0.765),
    (0.129, 0.4, 0.675),
    (0.02, 0.188, 0.38),
];

/// ColorSchemes.jl's `YlOrRd_8` (`data/colorbrewerschemes.jl`).
const YLORRD_8: [(f64, f64, f64); 8] = [
    (1.0, 1.0, 0.8),
    (1.0, 0.929, 0.627),
    (0.996, 0.851, 0.463),
    (0.996, 0.698, 0.298),
    (0.992, 0.553, 0.235),
    (0.988, 0.306, 0.165),
    (0.89, 0.102, 0.11),
    (0.694, 0.0, 0.149),
];

impl Gradient {
    /// The colours of the scheme as RGB fractions.
    pub fn colors(self) -> &'static [(f64, f64, f64)] {
        match self {
            Gradient::RdBu => &RDBU_11,
            Gradient::YlOrRd => &YLORRD_8,
        }
    }

    /// The colour at `t` from 0 to 1 (clamped): the colours of the scheme
    /// at equal distances and linear interpolation of red, green and blue
    /// between them, as ColorSchemes.jl's `get` computes it.
    pub fn get(self, t: f64) -> (f64, f64, f64) {
        let c = self.colors();
        let before_fp = t.clamp(0.0, 1.0) * (c.len() - 1) as f64 + 1.0;
        let before = before_fp.floor() as usize;
        let after = (before + 1).min(c.len());
        let w = before_fp - before as f64;
        let (a, b) = (c[before - 1], c[after - 1]);
        (
            (1.0 - w) * a.0 + w * b.0,
            (1.0 - w) * a.1 + w * b.1,
            (1.0 - w) * a.2 + w * b.2,
        )
    }
}

/// A heatmap, Plots.jl's `heatmap(z)`: the cell of `z[i][j]` is centred on
/// `x = j + 1`, `y = i + 1` (row 1 at the bottom), coloured by the gradient
/// between the colour limits; NaN leaves it empty. No colour bar is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Heatmap {
    pub z: Vec<Vec<f64>>,
    pub gradient: Gradient,
    /// Colour limits, Plots.jl's `clims`; `None`: the smallest and the
    /// largest value that is not NaN.
    pub clims: Option<(f64, f64)>,
}

impl Heatmap {
    fn ncols(&self) -> usize {
        self.z.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// The colour limits.
    fn limits(&self) -> (f64, f64) {
        self.clims.unwrap_or_else(|| {
            let values = self.z.iter().flatten().filter(|v| v.is_finite());
            values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
        })
    }

    /// The colour of a value as PlotUtils.jl's `get(gradient, z, clims)`
    /// gives it: none for a value that is not finite, the first colour if
    /// the limits are equal.
    pub fn color(&self, z: f64) -> Option<(f64, f64, f64)> {
        if !z.is_finite() {
            return None;
        }
        let (lo, hi) = self.limits();
        if lo == hi {
            return Some(self.gradient.colors()[0]);
        }
        Some(self.gradient.get((z.clamp(lo, hi) - lo) / (hi - lo)))
    }
}

/// A text in the plot, Plots.jl's `annotate!(x, y, text(text, font,
/// color, :center))`.
#[derive(Clone, Debug, PartialEq)]
pub struct Annotation {
    pub x: f64,
    pub y: f64,
    pub text: String,
    /// Font size in points.
    pub font: u32,
    pub color: Color,
}

/// Bars from 0 to `heights[k]` at `x[k]`, 0.8 wide, with a black outline
/// of one pixel and `labels[k]` above each bar: Makie's `barplot` with its
/// default `gap = 0.2`, `strokecolor = :black`, `strokewidth = 1` and
/// `bar_labels`, as BeforeIT's example `benchmark_w_matlab.jl` draws them.
#[derive(Clone, Debug, PartialEq)]
pub struct Bars {
    pub x: Vec<f64>,
    pub heights: Vec<f64>,
    pub color: Color,
    pub labels: Vec<String>,
}

/// One plot, the attributes that BeforeIT.jl passes to Plots.jl.
#[derive(Clone, Debug, PartialEq)]
pub struct Plot {
    pub title: String,
    /// Font size of the title in points.
    pub title_font: u32,
    /// Font size of the ticks in points.
    pub tick_font: u32,
    pub legend: Legend,
    /// Axis labels, Plots.jl's `xlabel` and `ylabel`; empty for none.
    pub x_label: String,
    pub y_label: String,
    /// Axis limits; `None`: from the data.
    pub x_limits: Option<(f64, f64)>,
    pub y_limits: Option<(f64, f64)>,
    /// Positions and labels of the ticks; `None`: automatic.
    pub x_ticks: Option<(Vec<f64>, Vec<String>)>,
    pub y_ticks: Option<(Vec<f64>, Vec<String>)>,
    /// Rotation of the tick labels on the x axis in degrees, Plots.jl's
    /// `xrotation`. The SVG drawing does not rotate them: `plotters` turns
    /// text by multiples of 90 degrees only.
    pub x_rotation: f64,
    /// Logarithmic axes, Plots.jl's `xscale = :log10` and `yscale =
    /// :log10`; values that are not positive leave a gap.
    pub x_log: bool,
    pub y_log: bool,
    /// Grid lines at the ticks of the x axis, Plots.jl's `xgrid`.
    pub x_grid: bool,
    /// Grid lines between the ticks of the y axis, Plots.jl's `yminorgrid`:
    /// four between two ticks of a linear axis, at 2, 3, …, 9 times the
    /// ticks of a logarithmic one (Plots.jl's `get_minor_ticks`).
    pub y_minor_grid: bool,
    pub series: Vec<Series>,
    pub hlines: Vec<HLine>,
    pub heatmap: Option<Heatmap>,
    pub bars: Option<Bars>,
    pub annotations: Vec<Annotation>,
}

impl Plot {
    /// An empty plot with the title and Plots.jl's defaults: title font of
    /// 14 points, ticks of 8 points, legend in the best corner, grid lines
    /// at the ticks.
    pub fn new(title: impl Into<String>) -> Plot {
        Plot {
            title: title.into(),
            title_font: TITLE_FONT,
            tick_font: TICK_FONT,
            legend: Legend::Best,
            x_label: String::new(),
            y_label: String::new(),
            x_limits: None,
            y_limits: None,
            x_ticks: None,
            y_ticks: None,
            x_rotation: 0.0,
            x_log: false,
            y_log: false,
            x_grid: true,
            y_minor_grid: false,
            series: Vec::new(),
            hlines: Vec::new(),
            heatmap: None,
            bars: None,
            annotations: Vec::new(),
        }
    }

    /// A heatmap of `z` ([`Heatmap`]) with the gradient and the colour limits
    /// from the data.
    pub fn heatmap(z: Vec<Vec<f64>>, gradient: Gradient) -> Plot {
        let mut p = Plot::new("");
        p.heatmap = Some(Heatmap {
            z,
            gradient,
            clims: None,
        });
        p
    }

    /// Bars ([`Bars`]) of the heights at 1, 2, …, with the labels.
    pub fn bars(heights: Vec<f64>, color: Color, labels: Vec<String>) -> Plot {
        let mut p = Plot::new("");
        p.bars = Some(Bars {
            x: (1..=heights.len()).map(|k| k as f64).collect(),
            heights,
            color,
            labels,
        });
        p
    }

    /// The number of series in the plot, which Plots.jl counts to pick the
    /// colour and the label of the next one.
    fn count(&self) -> usize {
        self.series.len() + self.hlines.len() + usize::from(self.heatmap.is_some())
    }

    /// Plots.jl's colour for the next series added without a colour of its
    /// own: the palette entry after the series already in the plot, wrapping
    /// around.
    fn next_color(&self) -> Color {
        Color::Palette(self.count() % PALETTE.len())
    }

    /// Adds a line through `(x[k], y[k])`, Plots.jl's `plot!(p, x, y)`, in the
    /// next colour of the palette, one pixel wide, with Plots.jl's label
    /// `y1`, `y2`, … after the number of series; returns it to set other
    /// attributes.
    pub fn add_line(&mut self, x: Vec<f64>, y: Vec<f64>) -> &mut Series {
        let s = Series {
            x,
            y,
            ribbon: None,
            fill_alpha: RIBBON_ALPHA,
            color: self.next_color(),
            label: Some(format!("y{}", self.count() + 1)),
            line_width: 1.0,
            line_style: LineStyle::Solid,
            marker: Marker::None,
        };
        self.series.push(s);
        self.series.last_mut().expect("a series was just added")
    }

    /// Adds StatsPlots' `errorline!(x, y; errorstyle = :ribbon, errortype)`
    /// with its default `centertype = :mean`: a line through the mean of
    /// each row of `y` with a band of one standard deviation or standard
    /// error, both ignoring NaN, with `fillalpha = 0.1`, as
    /// [`Plot::add_line`]. `y` is given by its rows; if their number differs
    /// from the length of `x` and the number of columns equals it, `y` is
    /// transposed, as StatsPlots does. The standard error divides by the
    /// square root of the number of columns, NaN included.
    ///
    /// Returns an error where StatsPlots throws: if neither dimension of `y`
    /// has the length of `x`.
    pub fn add_errorline(
        &mut self,
        x: Vec<f64>,
        y: Vec<Vec<f64>>,
        errortype: ErrorType,
    ) -> Result<&mut Series, String> {
        let (center, error) = errorline(x.len(), y, errortype)?;
        let s = self.add_line(x, center);
        s.ribbon = Some(error);
        s.fill_alpha = 0.1;
        Ok(s)
    }

    /// Adds a horizontal line at `y` across the plot, Plots.jl's `hline!(p,
    /// [y]; color, linestyle, label = false)`, one pixel wide.
    pub fn add_hline(&mut self, y: f64, color: Color, line_style: LineStyle) {
        self.hlines.push(HLine {
            y,
            color,
            line_style,
            line_width: 1.0,
        });
    }

    /// Adds a centred text at `(x, y)`, Plots.jl's `annotate!(x, y,
    /// text(text, font, color, :center))`.
    pub fn annotate(&mut self, x: f64, y: f64, text: impl Into<String>, font: u32, color: Color) {
        self.annotations.push(Annotation {
            x,
            y,
            text: text.into(),
            font,
            color,
        });
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
    layout_with(plots, layout, size, None)
}

/// As [`layout_svg`], with a title over the layout in Plots.jl's
/// `plot_titlefontsize` of 16 points: Julia's `plot(ps..., layout =
/// (rows, columns), plot_title = title)`.
///
/// Returns the errors of [`layout_svg`].
pub fn layout_svg_with_title(
    plots: &[Plot],
    layout: (usize, usize),
    size: (u32, u32),
    title: &str,
) -> Result<String, String> {
    layout_with(plots, layout, size, Some(title))
}

fn layout_with(
    plots: &[Plot],
    layout: (usize, usize),
    size: (u32, u32),
    title: Option<&str>,
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
        let area = match title {
            Some(t) => {
                let font = font_px(PLOT_TITLE_FONT);
                let height = (f64::from(size.1) * PLOT_TITLE_VSPAN).max(font + 4.0);
                let (top, rest) = root.split_vertically(height as u32);
                let style = ("sans-serif", font)
                    .into_font()
                    .into_text_style(&top)
                    .pos(Pos::new(HPos::Center, VPos::Center));
                let (w, h) = top.dim_in_pixel();
                top.draw_text(t, &style, (w as i32 / 2, h as i32 / 2))
                    .map_err(|e| e.to_string())?;
                rest
            }
            None => root.clone(),
        };
        let cells = area.split_evenly((rows, cols));
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

/// StatsPlots' `compute_error` for `centertype = :mean` after its check of
/// the sizes: the mean of each row and its error. See
/// [`Plot::add_errorline`].
fn errorline(
    nx: usize,
    y: Vec<Vec<f64>>,
    errortype: ErrorType,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let ncols = y.first().map_or(0, Vec::len);
    if y.iter().any(|r| r.len() != ncols) {
        return Err("the rows of y have different lengths".to_string());
    }
    let y = if y.len() == nx {
        y
    } else if ncols == nx {
        (0..ncols)
            .map(|j| y.iter().map(|r| r[j]).collect())
            .collect()
    } else {
        return Err("Size of x and y do not match".to_string());
    };
    let runs = y.first().map_or(0, Vec::len) as f64;
    Ok(y.iter()
        .map(|r| {
            let (mean, std) = nan_mean_std(r);
            match errortype {
                ErrorType::Std => (mean, std),
                ErrorType::Sem => (mean, std / runs.sqrt()),
            }
        })
        .unzip())
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
        let mut p = Plot::new(get_plot_title(quantity));
        p.title_font = titlefont;
        p.legend = Legend::None;
        match quantity_series(data, quantity)? {
            Some(y) => {
                p.add_line(steps(1, y.len()), y);
            }
            None => {
                for y in data.lists_by_name(quantity).unwrap_or_default() {
                    p.add_line(steps(1, y.len()), y.clone());
                }
            }
        }
        plots.push(p);
    }
    Ok(plots)
}

/// `from`, `from + 1`, … as floats, `n` of them.
fn steps(from: usize, n: usize) -> Vec<f64> {
    (from..from + n).map(|t| t as f64).collect()
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
        let mut p = Plot::new(get_plot_title(quantity));
        p.title_font = titlefont;
        p.legend = Legend::None;
        p.add_errorline(x, data, ErrorType::Std)?;
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
    plot_data_vectors_from(&data_vectors, titlefont, quantities)
}

/// [`plot_data_vectors`] from the data of the runs, which may come from
/// models of different types, as a Julia vector of model vectors may mix
/// them.
///
/// Returns the errors of [`plot_data_vectors`].
pub fn plot_data_vectors_from(
    data_vectors: &[DataVector],
    titlefont: u32,
    quantities: &[&str],
) -> Result<Vec<Plot>, String> {
    let time_length = data_vectors
        .first()
        .and_then(|dv| dv.iter().next())
        .ok_or("there are no models")?
        .wages
        .len();
    let mut plots = Vec::with_capacity(quantities.len());
    for &quantity in quantities {
        let mut p = Plot::new(get_plot_title(quantity));
        p.title_font = titlefont;
        p.legend = Legend::None;
        for dv in data_vectors {
            let data = compute_quantity_data(dv, quantity)?;
            p.add_errorline(steps(1, time_length), data, ErrorType::Std)?;
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

    let mut p = Plot::new(title);
    p.title_font = DEFAULT_TITLEFONT;
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
    p.tick_font = DEFAULT_TICKFONT;

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
    p.add_line(real_x, real_y).label = Some("Real".to_string());

    let x: Vec<f64> = model_x.data.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let y = scaled(model_y);
    let rows = model_y.size.first().copied().unwrap_or(0);
    let cols = y.len().checked_div(rows).unwrap_or(0);
    let y_rows: Vec<Vec<f64>> = (0..rows)
        .map(|t| (0..cols).map(|j| y[t + rows * j]).collect())
        .collect();
    p.add_errorline(x, y_rows, ErrorType::Std)?.label = Some("Model".to_string());
    Ok(p)
}

// =============================================================================
// Drawing
// =============================================================================

fn font_px(points: u32) -> f64 {
    f64::from(points) * PX_PER_INCH / 72.0
}

/// Whether a value can be drawn on an axis: finite, and positive on a
/// logarithmic one.
fn drawable(v: f64, log: bool) -> bool {
    v.is_finite() && (!log || v > 0.0)
}

/// Plots.jl's `default_widen_factor`: automatic limits of an axis with
/// lines are widened by 6 %.
const WIDEN_FACTOR: f64 = 1.06;

/// Plots.jl's `axis_limits` for automatic limits: from the smallest to the
/// largest value that can be drawn, `(0, 1)` without such values (`(1, 10)`
/// on a logarithmic axis), the largest raised to the smallest plus 1 if it
/// is not larger; then with `widen` the span around its middle times
/// [`WIDEN_FACTOR`], of the logarithms on a logarithmic axis
/// (`scale_lims`).
fn auto_range(values: impl Iterator<Item = f64>, log: bool, widen: bool) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in values.filter(|&v| drawable(v, log)) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo > hi {
        return if log { (1.0, 10.0) } else { (0.0, 1.0) };
    }
    if hi <= lo {
        hi = lo + 1.0;
    }
    if !widen {
        return (lo, hi);
    }
    let scale = |a: f64, b: f64| {
        let (mid, span) = ((a + b) / 2.0, (b - a) / 2.0);
        (mid - span * WIDEN_FACTOR, mid + span * WIDEN_FACTOR)
    };
    if log {
        let (a, b) = scale(lo.log10(), hi.log10());
        (10f64.powf(a), 10f64.powf(b))
    } else {
        scale(lo, hi)
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

/// A tick label: as `plotters` writes it, or with a power of ten
/// (`3.95e11`) for magnitudes from 10^5 on and below 10^-4, whose digits
/// would not fit beside the axis.
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

/// An axis of a [`Plot`]: linear or logarithmic, with given ticks or those
/// of `plotters` (powers of ten on a logarithmic axis), and minor grid
/// lines.
#[derive(Clone)]
struct Axis {
    lo: f64,
    hi: f64,
    log: bool,
    ticks: Option<(Vec<f64>, Vec<String>)>,
    minor: bool,
}

impl Axis {
    /// The axis in the coordinates that are drawn linearly.
    fn linear(&self) -> RangedCoordf64 {
        if self.log {
            RangedCoordf64::from(self.lo.log10()..self.hi.log10())
        } else {
            RangedCoordf64::from(self.lo..self.hi)
        }
    }

    fn contains(&self, v: f64) -> bool {
        v >= self.lo.min(self.hi) && v <= self.lo.max(self.hi)
    }

    /// The positions of the ticks.
    fn ticks(&self, n: usize) -> Vec<f64> {
        match &self.ticks {
            Some((positions, _)) => positions
                .iter()
                .copied()
                .filter(|&v| self.contains(v))
                .collect(),
            None if self.log => {
                let (a, b) = (self.lo.min(self.hi), self.lo.max(self.hi));
                let (first, last) = (a.log10().ceil() as i32, b.log10().floor() as i32);
                (first..=last).map(|k| 10f64.powi(k)).collect()
            }
            None => self.linear().key_points(n),
        }
    }

    /// Plots.jl's `get_minor_ticks` from the ticks: four between two ticks of
    /// a linear axis, 2, 3, …, 9 times each power of ten between two ticks of
    /// a logarithmic one, also beyond the first and the last tick.
    fn minor_ticks(&self, n: usize) -> Vec<f64> {
        let ticks = self.ticks(n);
        if !self.minor || ticks.len() < 2 {
            return Vec::new();
        }
        let mut phantom = Vec::with_capacity(ticks.len() + 2);
        let sub;
        let intervals;
        if self.log {
            sub = julia::round((ticks[1] / ticks[0]).log10()).max(1.0) as i32;
            intervals = 9.0;
            phantom.push(ticks[0] / 10.0);
            phantom.extend(&ticks);
            phantom.push(ticks[ticks.len() - 1] * 10.0);
        } else {
            sub = 1;
            intervals = 5.0;
            let k = ticks.len();
            let ratio = if k > 2 {
                (ticks[2] - ticks[1]) / (ticks[1] - ticks[0])
            } else {
                1.0
            };
            phantom.push(ticks[0] - (ticks[1] - ticks[0]) / ratio);
            phantom.extend(&ticks);
            phantom.push(ticks[k - 1] + (ticks[k - 1] - ticks[k - 2]) * ratio);
        }
        let mut minor = Vec::new();
        for w in phantom.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            if !(lo.is_finite() && hi.is_finite() && hi > lo) {
                continue;
            }
            if self.log {
                for e in 1..=sub {
                    let lo_ = lo * 10f64.powi(e - 1);
                    let hi_ = lo_ * 10.0;
                    let step = (hi_ - lo_) / intervals;
                    let start = lo_ + if e > 1 { 0.0 } else { step };
                    let stop = hi_ - if e < sub { 0.0 } else { step / 2.0 };
                    minor.extend(julia_range(start, step, stop));
                }
            } else {
                let step = (hi - lo) / intervals;
                minor.extend(julia_range(lo + step, step, hi - step / 2.0));
            }
        }
        minor.retain(|&v| self.contains(v));
        minor
    }

    fn label(&self, v: f64) -> String {
        match &self.ticks {
            Some((positions, labels)) => positions
                .iter()
                .position(|&q| q == v)
                .and_then(|k| labels.get(k).cloned())
                .unwrap_or_default(),
            None => short_number(&v),
        }
    }
}

/// `start + k * step` for `k = 0, 1, …` up to `stop`, Julia's
/// `start:step:stop` without its extra precision (the minor grid lines
/// differ from Plots.jl's by rounding at most).
fn julia_range(start: f64, step: f64, stop: f64) -> Vec<f64> {
    let n = ((stop - start) / step).floor();
    if n.is_nan() || n < 0.0 {
        return Vec::new();
    }
    (0..=n as usize).map(|k| start + k as f64 * step).collect()
}

impl Ranged for Axis {
    type FormatOption = NoDefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        let v = if self.log { value.log10() } else { *value };
        self.linear().map(&v, limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<f64> {
        if hint.weight().allow_light_points() {
            self.minor_ticks(hint.bold_points())
        } else {
            self.ticks(hint.max_num_points())
        }
    }

    fn range(&self) -> std::ops::Range<f64> {
        self.lo..self.hi
    }
}

impl ValueFormatter<f64> for Axis {
    fn format_ext(&self, value: &f64) -> String {
        self.label(*value)
    }
}

/// Julia's `isapprox(a, b)` with its default relative tolerance.
fn isapprox(a: f64, b: f64) -> bool {
    a == b
        || (a.is_finite()
            && b.is_finite()
            && (a - b).abs() <= f64::EPSILON.sqrt() * a.abs().max(b.abs()))
}

/// Plots.jl's `_guess_best_legend_position(xlims, ylims, sp)`: the corner
/// least crowded by the points of the series, top right on a tie, then top
/// left and bottom right.
fn best_corner(series: &[Series], xl: (f64, f64), yl: (f64, f64)) -> SeriesLabelPosition {
    let scale = (
        xl.0.max(xl.1) - xl.0.min(xl.1),
        yl.0.max(yl.1) - yl.0.min(yl.1),
    );
    let quadrants = [
        ((0.0, 0.25), (0.0, 0.25)),
        ((0.75, 1.0), (0.0, 0.25)),
        ((0.0, 0.25), (0.75, 1.0)),
        ((0.75, 1.0), (0.75, 1.0)),
    ];
    let corners = [(xl.0, yl.0), (xl.1, yl.0), (xl.0, yl.1), (xl.1, yl.1)];
    let mut u = [0.0f64; 4];
    for s in series.iter().filter(|s| !s.y.is_empty()) {
        for i in 0..4 {
            let lim = (corners[i].0 / scale.0, corners[i].1 / scale.1);
            let ((qx0, qx1), (qy0, qy1)) = quadrants[i];
            for (k, &xi) in s.x.iter().enumerate() {
                let yi = s.y[k % s.y.len()];
                if xi < xl.0 + qx0 * (xl.1 - xl.0)
                    || xi > xl.0 + qx1 * (xl.1 - xl.0)
                    || yi < yl.0 + qy0 * (yl.1 - yl.0)
                    || yi > yl.0 + qy1 * (yl.1 - yl.0)
                {
                    continue;
                }
                let d = (lim.0 - xi / scale.0).powi(2) + (lim.1 - yi / scale.1).powi(2);
                let d = if d.is_nan() { 0.0 } else { d };
                u[i] += 1.0 / (1.0 + 100.0 * d);
            }
        }
    }
    let mut best = 0;
    for i in 1..4 {
        if u[i] < u[best] {
            best = i;
        }
    }
    if isapprox(u[best], u[3]) {
        SeriesLabelPosition::UpperRight
    } else if isapprox(u[best], u[2]) {
        SeriesLabelPosition::UpperLeft
    } else if isapprox(u[best], u[1]) {
        SeriesLabelPosition::LowerRight
    } else {
        SeriesLabelPosition::LowerLeft
    }
}

/// The limits of the x axis.
fn x_range(p: &Plot) -> (f64, f64) {
    if let Some(l) = p.x_limits {
        return l;
    }
    if let Some(h) = &p.heatmap {
        return (0.5, h.ncols() as f64 + 0.5);
    }
    if let Some(b) = &p.bars {
        let (lo, hi) = auto_range(b.x.iter().copied(), false, false);
        return (lo - 0.5, hi + 0.5);
    }
    let widen = !p.series.is_empty();
    auto_range(
        p.series.iter().flat_map(|s| s.x.iter().copied()),
        p.x_log,
        widen,
    )
}

/// The limits of the y axis.
fn y_range(p: &Plot) -> (f64, f64) {
    if let Some(l) = p.y_limits {
        return l;
    }
    if let Some(h) = &p.heatmap {
        return (0.5, h.z.len() as f64 + 0.5);
    }
    let lines = p.series.iter().flat_map(|s| {
        let band = s
            .ribbon
            .iter()
            .flat_map(move |r| s.y.iter().zip(r).flat_map(|(y, e)| [y - e, y + e]));
        s.y.iter().copied().chain(band)
    });
    let bars = p
        .bars
        .iter()
        .flat_map(|b| b.heights.iter().copied().chain([0.0]));
    let hlines = p.hlines.iter().map(|h| h.y);
    let widen = !p.series.is_empty();
    auto_range(lines.chain(bars).chain(hlines), p.y_log, widen)
}

/// The style of a line of `width` pixels.
fn line_style(color: RGBColor, width: f64) -> ShapeStyle {
    color.stroke_width(width.round().max(1.0) as u32)
}

/// Draws a polyline in a line style.
fn draw_path<DB: DrawingBackend, CT: CoordTranslate>(
    area: &DrawingArea<DB, CT>,
    points: Vec<CT::From>,
    style: ShapeStyle,
    line_style: LineStyle,
    width: f64,
) -> Result<(), String>
where
    CT::From: Clone,
{
    let w = width.max(1.0);
    match line_style {
        LineStyle::Solid => area.draw(&PathElement::new(points, style)),
        LineStyle::Dash => area.draw(&DashedPathElement::new(points, 6.0 * w, 4.0 * w, style)),
        LineStyle::Dot => area.draw(&DashedPathElement::new(points, w, 2.0 * w, style)),
    }
    .map_err(|e| e.to_string())
}

fn draw<DB: DrawingBackend>(p: &Plot, area: &DrawingArea<DB, Shift>) -> Result<(), String> {
    let err = |e: DrawingAreaErrorKind<DB::ErrorType>| e.to_string();
    let (x0, x1) = x_range(p);
    let (y0, y1) = y_range(p);
    let x_axis = Axis {
        lo: x0,
        hi: x1,
        log: p.x_log,
        ticks: p.x_ticks.clone(),
        minor: false,
    };
    let y_axis = Axis {
        lo: y0,
        hi: y1,
        log: p.y_log,
        ticks: p.y_ticks.clone(),
        minor: p.y_minor_grid,
    };
    const N_TICKS: usize = 6;
    let tick_px = font_px(p.tick_font);
    let guide_px = font_px(GUIDE_FONT);
    let x_labels: Vec<String> = x_axis
        .ticks(N_TICKS)
        .iter()
        .map(|&v| x_axis.label(v))
        .collect();
    let y_chars = y_axis
        .ticks(N_TICKS)
        .iter()
        .map(|&v| y_axis.label(v).chars().count())
        .max()
        .unwrap_or(1);
    let x_lines = x_labels
        .iter()
        .map(|l| l.split('\n').count())
        .max()
        .unwrap_or(1);
    let multi_line = x_lines > 1;
    let guide = |label: &str| {
        if label.is_empty() {
            0.0
        } else {
            1.5 * guide_px
        }
    };
    let x_area = tick_px * (1.25 * x_lines as f64 + 0.75) + guide(&p.x_label);
    let y_area = tick_px * (0.6 * y_chars as f64 + 1.0) + guide(&p.y_label);

    let mut builder = ChartBuilder::on(area);
    if !p.title.is_empty() {
        builder.caption(&p.title, ("sans-serif", font_px(p.title_font)));
    }
    let mut chart = builder
        .margin(5)
        .x_label_area_size(x_area as u32)
        .y_label_area_size(y_area as u32)
        .build_cartesian_2d(x_axis.clone(), y_axis)
        .map_err(err)?;
    let no_label = |_: &f64| String::new();
    let mut mesh = chart.configure_mesh();
    mesh.light_line_style(BLACK.mix(MINOR_GRID_ALPHA))
        .bold_line_style(BLACK.mix(GRID_ALPHA))
        .x_labels(N_TICKS)
        .y_labels(N_TICKS)
        .label_style(("sans-serif", tick_px))
        .axis_desc_style(("sans-serif", guide_px));
    if !p.x_label.is_empty() {
        mesh.x_desc(&p.x_label);
    }
    if !p.y_label.is_empty() {
        mesh.y_desc(&p.y_label);
    }
    if !p.x_grid {
        mesh.disable_x_mesh();
    }
    if multi_line {
        mesh.x_label_formatter(&no_label);
    }
    mesh.draw().map_err(err)?;

    // Tick labels of several lines, which `plotters` writes on one.
    if multi_line {
        let base = area.get_base_pixel();
        let bottom = if y0 <= y1 { y0 } else { y1 };
        let style = ("sans-serif", tick_px)
            .into_font()
            .into_text_style(area)
            .pos(Pos::new(HPos::Center, VPos::Top));
        for (v, label) in x_axis.ticks(N_TICKS).iter().zip(&x_labels) {
            let (px, py) = chart.backend_coord(&(*v, bottom));
            for (k, line) in label.split('\n').enumerate() {
                let y = py - base.1 + 6 + (k as f64 * 1.2 * tick_px) as i32;
                area.draw_text(line, &style, (px - base.0, y))
                    .map_err(err)?;
            }
        }
    }

    let plotting = chart.plotting_area().clone();
    if let Some(h) = &p.heatmap {
        for (i, row) in h.z.iter().enumerate() {
            for (j, &z) in row.iter().enumerate() {
                if let Some(c) = h.color(z) {
                    let (x, y) = ((j + 1) as f64, (i + 1) as f64);
                    plotting
                        .draw(&Rectangle::new(
                            [(x - 0.5, y - 0.5), (x + 0.5, y + 0.5)],
                            rgb_of(c).filled(),
                        ))
                        .map_err(err)?;
                }
            }
        }
    }
    if let Some(b) = &p.bars {
        let color = b.color.rgb();
        let label_style = ("sans-serif", tick_px)
            .into_font()
            .into_text_style(area)
            .pos(Pos::new(HPos::Center, VPos::Bottom));
        for (k, (&x, &h)) in b.x.iter().zip(&b.heights).enumerate() {
            let corners = [(x - 0.4, 0.0), (x + 0.4, h)];
            plotting
                .draw(&Rectangle::new(corners, color.filled()))
                .map_err(err)?;
            plotting
                .draw(&Rectangle::new(corners, BLACK.stroke_width(1)))
                .map_err(err)?;
            if let Some(label) = b.labels.get(k) {
                let (px, py) = chart.backend_coord(&(x, h));
                let base = area.get_base_pixel();
                area.draw_text(label, &label_style, (px - base.0, py - base.1 - 5))
                    .map_err(err)?;
            }
        }
    }

    let inside = |x: f64| x_axis.contains(x);
    for s in &p.series {
        let color = s.color.rgb();
        if let Some(r) = &s.ribbon {
            let ok = |k: usize| {
                inside(s.x[k])
                    && drawable(s.x[k], p.x_log)
                    && drawable(s.y[k] - r[k], p.y_log)
                    && drawable(s.y[k] + r[k], p.y_log)
            };
            for run in runs(s.x.len().min(r.len()).min(s.y.len()), ok) {
                let upper = run.clone().map(|k| (s.x[k], s.y[k] + r[k]));
                let lower = run.clone().rev().map(|k| (s.x[k], s.y[k] - r[k]));
                let band: Vec<(f64, f64)> = upper.chain(lower).collect();
                plotting
                    .draw(&Polygon::new(band, color.mix(s.fill_alpha).filled()))
                    .map_err(err)?;
            }
        }
        let style = line_style(color, s.line_width);
        let n = s.x.len().min(s.y.len());
        let ok =
            |k: usize| inside(s.x[k]) && drawable(s.x[k], p.x_log) && drawable(s.y[k], p.y_log);
        for run in runs(n, ok) {
            let points: Vec<(f64, f64)> = run.clone().map(|k| (s.x[k], s.y[k])).collect();
            draw_path(&plotting, points.clone(), style, s.line_style, s.line_width)?;
            for &(x, y) in &points {
                draw_marker(&plotting, (x, y), s.marker, color)?;
            }
        }
    }
    for h in &p.hlines {
        if drawable(h.y, p.y_log) && y_axis_contains(y0, y1, h.y) {
            let style = line_style(h.color.rgb(), h.line_width);
            draw_path(
                &plotting,
                vec![(x0, h.y), (x1, h.y)],
                style,
                h.line_style,
                h.line_width,
            )?;
        }
    }
    for a in &p.annotations {
        let style = ("sans-serif", font_px(a.font))
            .into_font()
            .color(&a.color.rgb())
            .pos(Pos::new(HPos::Center, VPos::Center));
        plotting
            .draw(&Text::new(a.text.clone(), (a.x, a.y), style))
            .map_err(err)?;
    }

    let labelled: Vec<&Series> = p.series.iter().filter(|s| s.label.is_some()).collect();
    let position = match p.legend {
        Legend::None => None,
        Legend::Best => Some(best_corner(&p.series, (x0, x1), (y0, y1))),
        Legend::TopRight => Some(SeriesLabelPosition::UpperRight),
        Legend::TopLeft => Some(SeriesLabelPosition::UpperLeft),
        Legend::BottomRight => Some(SeriesLabelPosition::LowerRight),
        Legend::BottomLeft => Some(SeriesLabelPosition::LowerLeft),
    };
    if let (Some(position), false) = (position, labelled.is_empty()) {
        for s in labelled {
            let color = s.color.rgb();
            let style = line_style(color, s.line_width);
            let marker = s.marker;
            chart
                .draw_series(std::iter::empty::<PathElement<(f64, f64)>>())
                .map_err(err)?
                .label(s.label.clone().unwrap_or_default())
                .legend(move |(x, y)| {
                    EmptyElement::at((x, y))
                        + PathElement::new(vec![(0, 0), (15, 0)], style)
                        + legend_marker(marker, color)
                });
        }
        chart
            .configure_series_labels()
            .position(position)
            .label_font(("sans-serif", font_px(LEGEND_FONT)))
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .draw()
            .map_err(err)?;
    }
    Ok(())
}

fn y_axis_contains(y0: f64, y1: f64, y: f64) -> bool {
    y >= y0.min(y1) && y <= y0.max(y1)
}

/// The points of a marker around `(0, 0)` in pixels.
fn marker_polygon(marker: Marker, size: f64) -> Vec<(i32, i32)> {
    let s = size.round() as i32;
    match marker {
        Marker::None | Marker::Circle => Vec::new(),
        Marker::Square => vec![(-s, -s), (s, -s), (s, s), (-s, s)],
        Marker::Diamond => {
            let d = (size * std::f64::consts::SQRT_2).round() as i32;
            vec![(0, -d), (d, 0), (0, d), (-d, 0)]
        }
    }
}

/// Draws a marker filled with the series colour and outlined in black,
/// Plots.jl's `markerstrokecolor = :match` (the foreground colour).
fn draw_marker<DB: DrawingBackend>(
    area: &DrawingArea<DB, Cartesian2d<Axis, Axis>>,
    at: (f64, f64),
    marker: Marker,
    color: RGBColor,
) -> Result<(), String> {
    let err = |e: DrawingAreaErrorKind<DB::ErrorType>| e.to_string();
    let r = MARKER_SIZE.round() as i32;
    match marker {
        Marker::None => Ok(()),
        Marker::Circle => area
            .draw(
                &(EmptyElement::at(at)
                    + Circle::new((0, 0), r, color.filled())
                    + Circle::new((0, 0), r, BLACK.stroke_width(MARKER_STROKE))),
            )
            .map_err(err),
        Marker::Square | Marker::Diamond => {
            let points = marker_polygon(marker, MARKER_SIZE);
            let mut outline = points.clone();
            outline.push(points[0]);
            area.draw(
                &(EmptyElement::at(at)
                    + Polygon::new(points, color.filled())
                    + PathElement::new(outline, BLACK.stroke_width(MARKER_STROKE))),
            )
            .map_err(err)
        }
    }
}

/// The marker of a series in the legend, smaller than in the plot.
fn legend_marker(marker: Marker, color: RGBColor) -> Polygon<(i32, i32)> {
    let points = match marker {
        Marker::None => Vec::new(),
        Marker::Circle => (0..12)
            .map(|k| {
                let a = k as f64 * std::f64::consts::PI / 6.0;
                (
                    (7.5 + 3.0 * a.cos()).round() as i32,
                    (3.0 * a.sin()).round() as i32,
                )
            })
            .collect(),
        _ => marker_polygon(marker, 3.0)
            .into_iter()
            .map(|(x, y)| (x + 7, y))
            .collect(),
    };
    Polygon::new(points, color.filled())
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
        assert_eq!(Color::Palette(0).rgb(), RGBColor(0x00, 0x9A, 0xFA));
        assert_eq!(Color::Palette(1).rgb(), RGBColor(0xE3, 0x6F, 0x47));
        assert_eq!(Color::Palette(2).rgb(), RGBColor(0x3E, 0xA4, 0x4E));
        assert_eq!(Color::Palette(16).rgb(), Color::Palette(0).rgb());
        assert_eq!(Color::STEELBLUE.rgb(), RGBColor(70, 130, 180));
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
        let mut p = Plot::new("");
        for _ in 0..4 {
            p.add_line(vec![], vec![]);
        }
        let s = p
            .add_errorline(vec![1.0, 2.0, 3.0], y.clone(), ErrorType::Std)
            .unwrap()
            .clone();
        assert_eq!(s.y[..2], [2.0, 3.0]);
        assert!(s.y[2].is_nan());
        let r = s.ribbon.unwrap();
        assert_eq!(r[..2], [1.0, 8.0f64.sqrt()]);
        assert!(r[2].is_nan());
        assert_eq!((s.color, s.fill_alpha), (Color::Palette(4), 0.1));
        assert_eq!(s.label.as_deref(), Some("y5"));
        // The standard error divides by the square root of the number of
        // runs, NaN included.
        let (_, sem) = errorline(3, y.clone(), ErrorType::Sem).unwrap();
        assert_eq!(sem[..2], [1.0 / 3f64.sqrt(), 8.0f64.sqrt() / 3f64.sqrt()]);
        // One run: no band.
        let (_, e) = errorline(2, vec![vec![3.0], vec![4.0]], ErrorType::Std).unwrap();
        assert!(e.iter().all(|e| e.is_nan()));
        // y is transposed if its columns match x, else an error.
        let (c, _) = errorline(
            3,
            vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]],
            ErrorType::Std,
        )
        .unwrap();
        assert_eq!(c, [2.0, 3.0, 4.0]);
        assert!(errorline(2, y, ErrorType::Std).is_err());
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
            (9, Legend::None, 1)
        );
        assert_eq!(gdp.series[0].y, m.data.real_gdp);
        assert_eq!(gdp.series[0].x, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gdp.series[0].ribbon, None);
        assert_eq!(gdp.series[0].color, Color::Palette(0));
        let deflator: Vec<f64> = (0..4)
            .map(|t| m.data.nominal_gdp[t] / m.data.real_gdp[t])
            .collect();
        assert_eq!(ps[8].series[0].y, deflator);
        // A list per bank: one line per collection.
        let p = &plot_data(m, 10, &["equity_per_bank", "collection_time"]).unwrap();
        assert_eq!(p[0].series.len(), 4);
        assert_eq!(p[0].series[1].y, m.data.equity_per_bank[1]);
        assert_eq!(p[0].series[1].color, Color::Palette(1));
        assert_eq!(p[0].series[1].label.as_deref(), Some("y2"));
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
        assert_eq!(
            (s.color, ps[1].title.as_str()),
            (Color::Palette(0), "euribor")
        );
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
        let colors: Vec<Color> = ps[0].series.iter().map(|s| s.color).collect();
        assert_eq!(colors, [Color::Palette(0), Color::Palette(1)]);
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
            (9, 7, 20.0, Legend::Best)
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
            (Some("Real"), Color::Palette(0), &None)
        );
        assert_eq!(data.x, real_q);
        assert_eq!(data.y[3], real_v[3] * SCALE_FACTOR);
        assert_eq!(
            (pred.label.as_deref(), pred.color),
            (Some("Model"), Color::Palette(1))
        );
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
    fn colours_of_the_gradients() {
        // The ends and the middle of RdBu_11, and halfway between the first
        // two colours of YlOrRd_8 (ColorSchemes.jl's linear interpolation).
        assert_eq!(Gradient::RdBu.get(0.0), RDBU_11[0]);
        assert_eq!(Gradient::RdBu.get(1.0), RDBU_11[10]);
        assert_eq!(Gradient::RdBu.get(0.5), RDBU_11[5]);
        assert_eq!(Gradient::RdBu.get(-3.0), RDBU_11[0]);
        let (r, g, b) = Gradient::YlOrRd.get(0.5 / 7.0);
        assert!((r - 1.0).abs() < 1e-15);
        assert!((g - (1.0 + 0.929) / 2.0).abs() < 1e-15);
        assert!((b - (0.8 + 0.627) / 2.0).abs() < 1e-15);
        // Limits from the data, NaN empty, values beyond the limits clamped.
        let mut h = Heatmap {
            z: vec![vec![1.0, f64::NAN], vec![3.0, 2.0]],
            gradient: Gradient::YlOrRd,
            clims: None,
        };
        assert_eq!(h.color(1.0), Some(YLORRD_8[0]));
        assert_eq!(h.color(3.0), Some(YLORRD_8[7]));
        assert_eq!(h.color(f64::NAN), None);
        h.clims = Some((-4.0, 4.0));
        h.gradient = Gradient::RdBu;
        assert_eq!(h.color(9.0), Some(RDBU_11[10]));
        assert_eq!(h.color(0.0), Some(RDBU_11[5]));
        h.clims = Some((2.0, 2.0));
        assert_eq!(h.color(5.0), Some(RDBU_11[0]));
    }

    #[test]
    fn legend_in_the_best_corner() {
        let line = |x: Vec<f64>, y: Vec<f64>| {
            let mut p = Plot::new("");
            p.add_line(x, y);
            p.series
        };
        let lims = ((0.0, 10.0), (0.0, 10.0));
        // A line through the top right corner: the legend goes elsewhere,
        // to the empty corner preferred on a tie, top left.
        let s = line(vec![9.0, 10.0], vec![9.0, 10.0]);
        assert_eq!(
            best_corner(&s, lims.0, lims.1),
            SeriesLabelPosition::UpperLeft
        );
        // Points in all corners but the bottom left.
        let s = line(vec![10.0, 0.0, 10.0, 5.0], vec![10.0, 10.0, 0.0, 5.0]);
        assert_eq!(
            best_corner(&s, lims.0, lims.1),
            SeriesLabelPosition::LowerLeft
        );
        // No points: top right.
        assert_eq!(
            best_corner(&[], lims.0, lims.1),
            SeriesLabelPosition::UpperRight
        );
    }

    #[test]
    fn ticks_of_the_axes() {
        let axis = |lo, hi, log, minor| Axis {
            lo,
            hi,
            log,
            ticks: None,
            minor,
        };
        // Powers of ten on a logarithmic axis, with 2, …, 9 times them as
        // minor ticks, also before the first tick.
        let a = axis(0.5, 200.0, true, true);
        assert_eq!(a.ticks(6), [1.0, 10.0, 100.0]);
        let minor = a.minor_ticks(6);
        let expected = [
            0.5, 0.6, 0.7, 0.8, 0.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0,
        ];
        for (m, e) in minor.iter().zip(expected) {
            assert!((m - e).abs() < 1e-12 * e, "{m} {e}");
        }
        assert_eq!(minor.len(), 5 + 8 + 8 + 1);
        assert!((minor[minor.len() - 1] - 200.0).abs() < 1e-9);
        // Four minor ticks between two ticks of a linear axis.
        let a = Axis {
            ticks: Some((
                vec![0.0, 1.0, 2.0],
                vec!["a".into(), "b".into(), "c".into()],
            )),
            ..axis(0.0, 2.0, false, true)
        };
        let minor = a.minor_ticks(6);
        assert_eq!(minor.len(), 8);
        assert!((minor[0] - 0.2).abs() < 1e-12 && (minor[4] - 1.2).abs() < 1e-12);
        assert_eq!(
            (a.label(1.0), a.label(0.5)),
            ("b".to_string(), String::new())
        );
        assert!(axis(0.0, 1.0, false, false).minor_ticks(6).is_empty());
        assert_eq!(axis(1.0, 100.0, true, false).map(&10.0, (0, 100)), 50);
        assert_eq!(
            auto_range([0.0, -1.0, 10.0].into_iter(), true, false),
            (10.0, 11.0)
        );
        assert_eq!(auto_range([f64::NAN].into_iter(), false, true), (0.0, 1.0));
        let (lo, hi) = auto_range([1.0, 3.0].into_iter(), false, true);
        assert!((lo - 0.94).abs() < 1e-12 && (hi - 3.06).abs() < 1e-12);
        let (lo, hi) = auto_range([1.0, 100.0].into_iter(), true, true);
        assert!((lo - 10f64.powf(-0.06)).abs() < 1e-12 && (hi - 10f64.powf(2.06)).abs() < 1e-9);
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
        let titled = layout_svg_with_title(&ps, (3, 3), DEFAULT_SIZE, "All").unwrap();
        assert!(titled.contains("\nAll\n"));
        // Legend, explicit ticks, axis labels, styles and markers.
        let mut p = Plot::new("t");
        p.add_line(vec![1.0, 2.0, f64::NAN, 4.0], vec![1.0, 2.0, 3.0, 4.0])
            .label = Some("Real".into());
        let s = p.add_line(vec![1.0, 2.0, 3.0], vec![2.0, 1.0, 2.0]);
        (s.line_style, s.marker, s.line_width) = (LineStyle::Dash, Marker::Diamond, 2.0);
        p.add_hline(1.5, Color::GRAY, LineStyle::Dot);
        p.x_ticks = Some((vec![1.0, 3.0], vec!["one".into(), "three".into()]));
        (p.x_label, p.y_label) = ("quarters".into(), "GDP".into());
        p.annotate(2.0, 1.5, "note", 6, Color::CRIMSON);
        let svg = p.to_svg(DEFAULT_SIZE).unwrap();
        for text in ["Real", "y2", "one", "three", "quarters", "GDP", "note"] {
            assert!(svg.contains(&format!("\n{text}\n")), "{text}");
        }
        assert!(svg.contains("#DC143C"), "crimson annotation");
        // A heatmap with labels of two lines, bars, logarithmic axes.
        let mut h = Plot::heatmap(vec![vec![1.0, 2.0], vec![3.0, f64::NAN]], Gradient::RdBu);
        h.x_ticks = Some((vec![1.0, 2.0], vec!["Real\nGDP".into(), "Euribor".into()]));
        let svg = h.to_svg(DEFAULT_SIZE).unwrap();
        assert!(svg.contains("\nReal\n") && svg.contains("\nGDP\n"));
        let b = Plot::bars(
            vec![1.0, 2.5],
            Color::Rgb(0, 155, 228),
            vec!["1.0".into(), "2.5".into()],
        );
        assert!(b.to_svg(DEFAULT_SIZE).unwrap().contains("\n2.5\n"));
        let mut l = Plot::new("log");
        l.add_line(vec![1.0, 10.0, 100.0], vec![0.1, 1.0, 1000.0])
            .marker = Marker::Circle;
        (l.x_log, l.y_log, l.y_minor_grid) = (true, true, true);
        assert!(l.to_svg(DEFAULT_SIZE).unwrap().contains("<circle"));
        assert_eq!(short_number(&395e9), "3.95e11");
        assert_eq!(short_number(&2.5e-5), "2.5e-5");
    }
}
