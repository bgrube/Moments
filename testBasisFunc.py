#!/usr/bin/env python3

import functools
import numpy as np

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# C++ implementation of (complex conjugated) Wigner D function and spherical harmonics
# also provides complexT typedef for std::complex<double>
ROOT.gROOT.LoadMacro("./wignerD.C++")  # type: ignore


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore

  # see https://root.cern/doc/master/pyroot001__arrayInterface_8py.html
  # and https://root-forum.cern.ch/t/stl-vector-and-numpy-types-in-pyroot/54073/6
  # and https://root-forum.cern.ch/t/cant-append-array-to-vector-in-pyroot-if-it-is-created-with-processline/43803/5
  # cppyy does not work see https://github.com/root-project/root/issues/12635
  # https://cppyy.readthedocs.io/en/latest/numba.html
  # https://cppyy.readthedocs.io/en/latest/lowlevel.html#numpy-casts
  vals = np.random.rand(10)
  theta = ROOT.std.vector["double"](vals)  # type: ignore
  phi   = ROOT.std.vector["double"](vals)  # type: ignore
  Phi   = ROOT.std.vector["double"](vals)  # type: ignore
  fcnResults = ROOT.f_meas(1, 3, 2, theta, phi, Phi, 1.0)
  arr = np.asarray(fcnResults)
  print(f"{type(arr)} {arr.dtype} {arr}")
