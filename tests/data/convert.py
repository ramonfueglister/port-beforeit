"""Converts the reference values of BeforeIT.jl (MATLAB files) to the JSON
files in this folder.

Usage: python3 convert.py <BeforeIT.jl checkout> <output folder>
Needs numpy and scipy. Matrices are written in Julia orientation:
row r, column c is A[r, c].
"""
import json
import os
import sys

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


def write(name, d):
    with open(os.path.join(dst, name), "w") as fh:
        json.dump(d, fh, separators=(",", ":"), sort_keys=True)
        fh.write("\n")


for n in ["init_vars_firms", "init_vars_bank", "init_vars_households", "output_t1", "output_t5"]:
    write(f"matlab_{n}.json", mat_to_dict(os.path.join(src, f"test/matlab_code/{n}.mat")))
