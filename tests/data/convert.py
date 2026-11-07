"""Converts the reference data of BeforeIT.jl (JLD2 and MATLAB files) to the
JSON files in this folder.

Usage: python3 convert.py <BeforeIT.jl checkout> <output folder>
Needs h5py, numpy and scipy. Matrices are written in Julia orientation:
row r, column c is A[r, c].
"""
import json
import os
import sys

import h5py
import numpy as np
import scipy.io as sio

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


def mat_to_dict(path):
    m = sio.loadmat(path)
    out = {}
    for k, v in m.items():
        if k.startswith("__"):
            continue
        v = np.asarray(v, dtype=float)
        if v.ndim == 2 and v.shape[0] == 1:
            out[k] = [float(x) for x in v.ravel()]
        elif v.ndim == 2:
            out[k] = [[float(x) for x in row] for row in v]
        else:
            out[k] = [float(x) for x in v.ravel()]
    return out


def write(name, d):
    with open(os.path.join(dst, name), "w") as fh:
        json.dump(d, fh, separators=(",", ":"), sort_keys=True)
        fh.write("\n")


write(
    "austria_2010q1_parameters.json",
    jld2_to_dict(os.path.join(src, "data/austria/parameters/2010Q1.jld2")),
)
write(
    "austria_2010q1_initial_conditions.json",
    jld2_to_dict(os.path.join(src, "data/austria/initial_conditions/2010Q1.jld2")),
)
for n in ["init_vars_firms", "init_vars_bank", "init_vars_households", "output_t1", "output_t5"]:
    write(f"matlab_{n}.json", mat_to_dict(os.path.join(src, f"test/matlab_code/{n}.mat")))
