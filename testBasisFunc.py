#!/usr/bin/env python3

import OpenMp

import functools
import numpy as np
import timeit

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)

  OpenMp.enableRootACLiCOpenMp()
  OpenMp.printRootACLiCSettings()
  # C++ implementation of (complex conjugated) Wigner D function and spherical harmonics
  # also provides complexT typedef for std::complex<double>
  # OpenMp.setNmbOpenMpThreads(1)
  OpenMp.setNmbOpenMpThreads(5)
  ROOT.gROOT.LoadMacro("./wignerD.C++")
  print(f"Using {ROOT.getNmbOpenMpThreads()} OpenMP threads")
  ROOT.testOpenMp()

  # see https://root.cern/doc/master/pyroot001__arrayInterface_8py.html
  # and https://root-forum.cern.ch/t/stl-vector-and-numpy-types-in-pyroot/54073/6
  # and https://root-forum.cern.ch/t/cant-append-array-to-vector-in-pyroot-if-it-is-created-with-processline/43803/5
  # cppyy does not work see https://github.com/root-project/root/issues/12635
  # https://cppyy.readthedocs.io/en/latest/numba.html
  # https://cppyy.readthedocs.io/en/latest/lowlevel.html#numpy-casts
  vals  = np.random.rand(100000000)
  theta = ROOT.std.vector["double"](vals)
  phi   = ROOT.std.vector["double"](vals)
  Phi   = ROOT.std.vector["double"](vals)
  start = timeit.default_timer()
  fcnResults = np.asarray(ROOT.f_phys(1, 3, 2, theta, phi, Phi, 1.0))
  stop = timeit.default_timer()
  time = stop -start
  print(f"{ROOT.getNmbOpenMpThreads()} threads: {str(time)} sec")
  OpenMp.restoreNmbOpenMpThreads()
