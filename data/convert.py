"""Converts the built-in data sets of BeforeIT.jl (JLD2 files in its folder
`data`) to the JSON files in this folder, which the crate embeds.

Usage: python3 convert.py <BeforeIT.jl checkout> <output folder>
Needs h5py and numpy. Matrices are written in Julia orientation: row r,
column c is A[r, c]; row vectors and vectors become lists.
"""
import json
import os
import sys

import h5py
import numpy as np

src, dst = sys.argv[1], sys.argv[2]


def jld2_to_dict(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if k.startswith("_"):
                continue
            v = f[k][()]
            if isinstance(v, np.ndarray):
                if v.ndim == 2 and v.shape[0] == 1:
                    out[k] = [float(x) for x in v.ravel()]
                elif v.ndim == 2:
                    # h5py shows Julia matrices transposed.
                    out[k] = [[float(x) for x in row] for row in v.T]
                else:
                    out[k] = [float(x) for x in v.ravel()]
            else:
                out[k] = float(v)
    return out


for country in ["austria", "italy", "steady_state"]:
    for kind in ["parameters", "initial_conditions"]:
        rel = os.path.join(country, kind, "2010Q1")
        d = jld2_to_dict(os.path.join(src, "data", rel + ".jld2"))
        os.makedirs(os.path.join(dst, country, kind), exist_ok=True)
        with open(os.path.join(dst, rel + ".json"), "w") as fh:
            json.dump(d, fh, separators=(",", ":"), sort_keys=True)
            fh.write("\n")
