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
