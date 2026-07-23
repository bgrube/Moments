#!/usr/bin/env python3

from __future__ import annotations

import ROOT

import spherical  #TODO weird behavior: without this import I get
#   File "/w/halld-scshelf2101/bgrube/Moments/./testCpp.py", line 8, in <module>
#     import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
#   File "/group/halld/Software/builds/Linux_Alma9-x86_64-gcc11.5.0/root/root-6.24.04/lib/ROOT/_facade.py", line 150, in _importhook
#     return _orig_ihook(name, *args, **kwds)
#   File "/w/halld-scshelf2101/bgrube/Moments/RootUtilities.py", line 20, in <module>
#     assert ROOT.gROOT.LoadMacro("basisFunctions.C+") == 0, "Error loading 'basisFunctions.C'"
# AssertionError: Error loading 'basisFunctions.C'
# But calling the corresponding LoadMacro() line in __main__ works w/o problems
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  # ROOT.gROOT.LoadMacro("./basisFunctions.C+") == 0, "Error loading './basisFunctions.C'"
  print("FIRST")
  hist = ROOT.TH1D("hist", "", 100, 0, 1)
  RootUtilities.declareInCpp(hist = hist)
  ROOT.gInterpreter.ProcessLine("std::cout << PyVars::hist.GetName() << std::endl;")
  print("SECOND")
  hist2 = ROOT.TH1D("hist2", "", 100, 0, 1)
  RootUtilities.declareInCpp(hist = hist2)
  ROOT.gInterpreter.ProcessLine("std::cout << PyVars::hist.GetName() << std::endl;")
