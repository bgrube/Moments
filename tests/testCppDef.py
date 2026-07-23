#!/usr/bin/env python3

from __future__ import annotations

import functools

import ROOT

# import quaternionic
import spherical
# need to import one of the above modules, otherwise loading of `cpp/basisFunctions.C` fails with error:
# cling::DynamicLibraryManager::loadLibrary(): libopenblas64_p-r0-0cf96a72.3.23.dev.so: cannot open shared object file: No such file or directory
# Traceback (most recent call last):
#   File "/w/halld-scshelf2101/bgrube/Moments/./tests/testCppDef.py", line 27, in <module>
#     RootUtilities.loadBasisFunctionsLibrary(enableOpenMp = False)  # initializes OpenMP and loads `cpp/basisFunctions.C`
#   File "/w/halld-scshelf2101/bgrube/Moments/moments/RootUtilities.py", line 31, in wrapper
#     funcResult = func(*args, **kwargs)
#   File "/w/halld-scshelf2101/bgrube/Moments/moments/RootUtilities.py", line 97, in loadBasisFunctionsLibrary
#     assert ROOT.gROOT.LoadMacro(f"{cppSourceFilePath}+{'+' if forceRecompilation else ''}") == 0, f"Error loading '{cppSourceFilePath}'"
# AssertionError: Error loading './cpp/basisFunctions.C'
from moments import RootUtilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  RootUtilities.loadBasisFunctionsLibrary(enableOpenMp = True)  # initializes OpenMP and loads `cpp/basisFunctions.C`
  ROOT.gROOT.SetBatch(True)
  print("FIRST")
  hist = ROOT.TH1D("hist", "", 100, 0, 1)
  RootUtilities.declareInCpp(hist = hist)
  ROOT.gInterpreter.ProcessLine("std::cout << PyVars::hist.GetName() << std::endl;")
  print("SECOND")
  hist2 = ROOT.TH1D("hist2", "", 100, 0, 1)
  RootUtilities.declareInCpp(hist = hist2)
  ROOT.gInterpreter.ProcessLine("std::cout << PyVars::hist.GetName() << std::endl;")
