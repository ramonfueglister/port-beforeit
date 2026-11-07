# Test data

Reference data for the tests, converted to JSON from files of
[BeforeIT.jl](https://github.com/bancaditalia/BeforeIT.jl) at commit
`09263989b56bd2d58c28657bb8e940142d995cf4`:

| File here | Source in BeforeIT.jl | Content |
|---|---|---|
| `matlab_init_vars_firms.json` | `test/matlab_code/init_vars_firms.mat` | Initial state of the firms, MATLAB code |
| `matlab_init_vars_bank.json` | `test/matlab_code/init_vars_bank.mat` | Initial state of the bank, MATLAB code |
| `matlab_init_vars_households.json` | `test/matlab_code/init_vars_households.mat` | Initial state of the households, MATLAB code |
| `matlab_output_t1.json` | `test/matlab_code/output_t1.mat` | Aggregates after one quarter without randomness, MATLAB code |
| `matlab_output_t5.json` | `test/matlab_code/output_t5.mat` | Aggregates after five quarters without randomness, MATLAB code |

BeforeIT.jl uses the `.mat` files in its own tests as the values of the
original MATLAB implementation of the model. BeforeIT.jl is licensed under
the Apache License 2.0, Copyright 2024 Banca d'Italia and the authors.

`convert.py` does the conversion; run on a checkout of BeforeIT.jl at the
commit above, it reproduces these files byte for byte:

```sh
python3 tests/data/convert.py path/to/BeforeIT.jl tests/data
```

Matrices are stored in Julia orientation (row `r`, column `c` is `A[r, c]`),
and the keys keep the names of the original files.

## Reference values of the calibration

`calibration_italy_2010-03-31.json` and
`calibration_italy_2019-12-31_growth.json` are not converted from
BeforeIT.jl. `calibration_reference.py`, an independent NumPy translation of
`get_params_and_initial_conditions` in BeforeIT `src/utils/calibration.jl`,
computes them from `data/italy/calibration.json` (scale 0.001, the second
with `use_growth_rate_ar1`):

```sh
python3 tests/data/calibration_reference.py data/italy/calibration.json 2010-03-31 0.001 0 tests/data/calibration_italy_2010-03-31.json
python3 tests/data/calibration_reference.py data/italy/calibration.json 2019-12-31 0.001 1 tests/data/calibration_italy_2019-12-31_growth.json
```

The translation shares no code with the port, so the comparison in
`tests/calibration.rs` checks that both read the Julia code the same way. It
does not check the Julia code itself, and no values of Julia were available
for this check.
