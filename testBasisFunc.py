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
  ROOT.gROOT.LoadMacro("./basisFunctions.C++")
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 5):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    print(f"Using {ROOT.getNmbOpenMpThreads()} OpenMP threads")
    ROOT.testOpenMp()

  for nmbThreads in (1, 2, 4, 8):
    with threadController.limit(limits = nmbThreads):
      print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
      # see https://root.cern/doc/master/pyroot001__arrayInterface_8py.html
      # and https://root-forum.cern.ch/t/stl-vector-and-numpy-types-in-pyroot/54073/6
      # and https://root-forum.cern.ch/t/cant-append-array-to-vector-in-pyroot-if-it-is-created-with-processline/43803/5
      # cppyy does not work see https://github.com/root-project/root/issues/12635
      # https://cppyy.readthedocs.io/en/latest/numba.html
      # https://cppyy.readthedocs.io/en/latest/lowlevel.html#numpy-casts
      vals  = np.random.rand(10000000)
      theta = ROOT.std.vector["double"](vals)
      phi   = ROOT.std.vector["double"](vals)
      Phi   = ROOT.std.vector["double"](vals)
      start = timeit.default_timer()
      fcnResults = np.asarray(ROOT.f_phys(1, 3, 2, theta, phi, Phi, 1.0))
      stop = timeit.default_timer()
      time = stop -start
      print(f"{ROOT.getNmbOpenMpThreads()} threads: {str(time)} sec")
