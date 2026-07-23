#!/usr/bin/env python3

import functools
import numpy as np
import threadpoolctl
import timeit

import ROOT

import OpenMpUtilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)

  OpenMpUtilities.enableRootACLiCOpenMp()
  OpenMpUtilities.printRootACLiCSettings()
  # C++ implementation of (complex conjugated) Wigner D function and spherical harmonics
  # also provides complexT typedef for std::complex<double>
  assert ROOT.gROOT.LoadMacro("./basisFunctions.C++") == 0, "Error loading './basisFunctions.C++'"
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 256):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    print(f"Using {ROOT.getNmbOpenMpThreads()} OpenMP threads")
    ROOT.testOpenMp()

  start = timeit.default_timer()
  rng   = np.random.default_rng(1)
  vals  = rng.random(100000000, dtype = np.float64)
  stop  = timeit.default_timer()
  print(f"!!! needed {str(stop - start)} sec to generate random data")
  thetas = ROOT.std.vector["double"](vals)
  phis   = ROOT.std.vector["double"](vals)
  Phis   = ROOT.std.vector["double"](vals)
  for nmbThreads in (1, 2, 4, 8, 16, 32, 64, 128):
    with threadController.limit(limits = nmbThreads):
      print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
      # see https://root.cern/doc/master/pyroot001__arrayInterface_8py.html
      # and https://root-forum.cern.ch/t/stl-vector-and-numpy-types-in-pyroot/54073/6
      # and https://root-forum.cern.ch/t/cant-append-array-to-vector-in-pyroot-if-it-is-created-with-processline/43803/5
      # cppyy does not work see https://github.com/root-project/root/issues/12635
      # https://cppyy.readthedocs.io/en/latest/numba.html
      # https://cppyy.readthedocs.io/en/latest/lowlevel.html#numpy-casts
      start = timeit.default_timer()
      fcnResults = ROOT.f_basis(1, 3, 2, thetas, phis, Phis, 1.0)  # 15 sec for 1 thread, 2.1 sec for 8 threads
      # fcnResults = ROOT.f_phys(1, 3, 2, thetas, phis, Phis, 1.0)  # 16 sec for 1 thread, 3.0 sec for 8 threads
      stop = timeit.default_timer()
      time = stop - start
      print(f"!!! {ROOT.getNmbOpenMpThreads()} threads: total time = {str(time)} sec; total time * number of threads = {str(time * nmbThreads)} sec")
      start = timeit.default_timer()
      print(f"!!! {type(fcnResults)=}")
      foo = np.asarray(fcnResults)  #TODO 0.05 sec for std::vector<double> but 6!!! sec for std::vector<complex<double>>
      stop = timeit.default_timer()
      time = stop - start
      print(f"!!! needed {str(time)} sec for conversion to numpy array")
