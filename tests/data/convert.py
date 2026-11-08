"""Converts the reference values of BeforeIT.jl (MATLAB and JLD2 files) to
the JSON files in this folder.

Usage: python3 convert.py <BeforeIT.jl checkout> <output folder>
Needs numpy, scipy and h5py. Matrices are written in Julia orientation:
row r, column c is A[r, c].
"""
import json
import os
import sys

import h5py
import numpy as np
import scipy.io as sio

src, dst = sys.argv[1], sys.argv[2]


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


def jld2_dict(path, name):
    """The Dict{String, Array{Float64}} `name` of a JLD2 file."""
    out = {}
    with h5py.File(path, "r") as f:
        for pair in f[f[name][()]["kvvec"]][()]:
            p = f[pair][()]
            key = p["first"]
            key = f[key][()] if isinstance(key, h5py.Reference) else key
            # h5py shows a Julia array with its dimensions reversed.
            v = np.asarray(f[p["second"]][()], dtype=float).T
            out[key.decode()] = v.tolist()
    return out


def write(name, d):
    with open(os.path.join(dst, name), "w") as fh:
        json.dump(d, fh, separators=(",", ":"), sort_keys=True)
        fh.write("\n")


for n in ["init_vars_firms", "init_vars_bank", "init_vars_households", "output_t1", "output_t5"]:
    write(f"matlab_{n}.json", mat_to_dict(os.path.join(src, f"test/matlab_code/{n}.mat")))
write(
    "prediction_pipeline_2010Q1.json",
    jld2_dict(os.path.join(src, "test/deterministic/2010Q1.jld2"), "model_dict"),
)
