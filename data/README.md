# Data sets

The data sets of [BeforeIT.jl](https://github.com/bancaditalia/BeforeIT.jl)
at commit `09263989b56bd2d58c28657bb8e940142d995cf4`, converted to JSON. The
crate embeds them as `AUSTRIA2010Q1`, `ITALY2010Q1` and `STEADY_STATE2010Q1`,
the names of the constants in BeforeIT
`src/utils/standard_params_initial_conditions.jl`.

| File here | Source in BeforeIT.jl |
|---|---|
| `austria/parameters/2010Q1.json` | `data/austria/parameters/2010Q1.jld2` |
| `austria/initial_conditions/2010Q1.json` | `data/austria/initial_conditions/2010Q1.jld2` |
| `italy/parameters/2010Q1.json` | `data/italy/parameters/2010Q1.jld2` |
| `italy/initial_conditions/2010Q1.json` | `data/italy/initial_conditions/2010Q1.jld2` |
| `steady_state/parameters/2010Q1.json` | `data/steady_state/parameters/2010Q1.jld2` |
| `steady_state/initial_conditions/2010Q1.json` | `data/steady_state/initial_conditions/2010Q1.jld2` |

BeforeIT.jl is licensed under the Apache License 2.0, Copyright 2024 Banca
d'Italia and the authors.

`convert.py` does the conversion; run on a checkout of BeforeIT.jl at the
commit above, it reproduces these files byte for byte:

```sh
python3 data/convert.py path/to/BeforeIT.jl data
```

Matrices are stored in Julia orientation (row `r`, column `c` is `A[r, c]`),
row vectors and vectors as lists, and the keys keep the names of the
original files. The folder `data/austria` of BeforeIT.jl also has MATLAB
copies of its files with the same values; those in `data/steady_state`
differ from the JLD2 files, which are the ones BeforeIT.jl loads.

## Calibration data

`italy/calibration.json` is BeforeIT's `ITALY_CALIBRATION` (constant in
`src/utils/standard_calibration_data.jl`), which the crate embeds under the
same name: the four JLD2 files of the folder `data/italy/calibration` of
BeforeIT.jl at the commit above with the dates of
`standard_calibration_data.jl`.

| Key here | Source in BeforeIT.jl |
|---|---|
| `calibration` | `data/italy/calibration/calibration/2010Q1.jld2` |
| `figaro` | `data/italy/calibration/figaro/2010.jld2` |
| `data` | `data/italy/calibration/data/1996.jld2` |
| `ea` | `data/italy/calibration/ea/1996.jld2` |
| `max_calibration_date`, `estimation_date` | `2016-12-31` and `1996-12-31`, as in `standard_calibration_data.jl` |

`convert_calibration.py` (needs `h5py` and `numpy`) does the conversion and
reproduces the file byte for byte:

```sh
python3 data/convert_calibration.py italy path/to/BeforeIT.jl data/italy/calibration.json
```

It also converts a calibration object of the Zenodo record that BeforeIT's
`download_zenodo_calibration_object` downloads,
[*EuroStat Tables for BeforeIT Calibration*](https://doi.org/10.5281/zenodo.18698434)
by Oliver Reiter, Steven Hoekstra, Adriano Meligrana and Aldo Glielmo
(CC BY 4.0), which this repository does not contain:

```sh
curl -LO https://zenodo.org/records/18698434/files/AT_calibration_object.jld2
python3 data/convert_calibration.py zenodo AT_calibration_object.jld2 AT.json
```

A value of a table is a number or `{"size": [...], "data": [...]}` with the
size of the Julia array and its entries in Julia's column-major order; an
entry is a number, `null` for Julia's `missing`, or `"NaN"`, `"Inf"` or
`"-Inf"`.
