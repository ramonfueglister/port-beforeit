"""Converts calibration data of BeforeIT.jl (JLD2 files) to the JSON format
that `CalibrationData::from_json` reads.

Usage:
  python3 convert_calibration.py italy <BeforeIT.jl checkout> <output file>
      BeforeIT's ITALY_CALIBRATION, the four files of its folder
      data/italy/calibration with the dates of
      src/utils/standard_calibration_data.jl.
  python3 convert_calibration.py zenodo <XX_calibration_object.jld2> <output file>
      a calibration object of the Zenodo record that BeforeIT's
      download_zenodo_calibration_object downloads.

Needs h5py and numpy. Each value of the tables `calibration`, `figaro`,
`data` and `ea` becomes a number if it is a scalar, otherwise
{"size": [...], "data": [...]} with the size and the entries of the Julia
array in Julia's column-major order. An entry is a number, null for Julia's
`missing`, or one of the strings "NaN", "Inf" and "-Inf". Dates are
written as "YYYY-MM-DD".
"""
import datetime
import json
import math
import sys

import h5py
import numpy as np

# Julia's DateTime counts milliseconds from 0000-12-31, Python's ordinals
# count days from 0001-01-01 = 1, so both count days the same way.
MS_PER_DAY = 86_400_000


def entry(x, missing=False):
    if missing:
        return None
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == int(x) and abs(x) < 2**53:
        return int(x)
    return x


def value(f, ref):
    obj = f[ref]
    v = obj[()]
    if obj.dtype.names == ("mask", "t2"):
        # JLD2 stores Union{Missing, Float64} as a mask byte (0 for missing)
        # and the number.
        flat = [entry(x, m == 0) for m, x in zip(v["mask"].ravel(), v["t2"].ravel())]
        shape = v.shape
    elif obj.dtype.names is None and obj.dtype.kind in "fiu":
        if np.ndim(v) == 0:
            return entry(v)
        flat = [entry(x) for x in np.asarray(v).ravel()]
        shape = v.shape
    else:
        raise ValueError(f"unsupported value of type {obj.dtype}")
    # h5py shows a Julia array with its dimensions reversed, so its row-major
    # order is Julia's column-major order.
    return {"size": list(reversed(shape)), "data": flat}


def table(f, ref):
    out = {}
    for pair in f[f[ref][()]["kvvec"]][()]:
        p = f[pair][()]
        key = p["first"]
        key = f[key][()] if isinstance(key, h5py.Reference) else key
        out[key.decode()] = value(f, p["second"])
    return out


def date_from_julia(d):
    ms = int(d["instant"]["periods"]["value"])
    assert ms % MS_PER_DAY == 0, "dates with a time of day are not supported"
    return datetime.date.fromordinal(ms // MS_PER_DAY).isoformat()


def italy(src):
    base = src + "/data/italy/calibration/"
    out = {}
    for name, rel, top in [
        ("calibration", "calibration/2010Q1.jld2", "calibration_data"),
        ("figaro", "figaro/2010.jld2", "figaro"),
        ("data", "data/1996.jld2", "data"),
        ("ea", "ea/1996.jld2", "ea"),
    ]:
        with h5py.File(base + rel, "r") as f:
            out[name] = table(f, top)
    out["max_calibration_date"] = "2016-12-31"
    out["estimation_date"] = "1996-12-31"
    return out


def zenodo(path):
    with h5py.File(path, "r") as f:
        co = f["calibration_object"][()]
        out = {name: table(f, co[name]) for name in ["calibration", "figaro", "data", "ea"]}
        out["max_calibration_date"] = date_from_julia(co["max_calibration_date"])
        out["estimation_date"] = date_from_julia(co["estimation_date"])
    return out


def main():
    kind, src, dst = sys.argv[1:4]
    out = italy(src) if kind == "italy" else zenodo(src)
    with open(dst, "w") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=True, allow_nan=False)
        fh.write("\n")


main()
