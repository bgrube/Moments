"""Module that provides functions for using ROOT code"""

import functools
from typing import Any

import ROOT

import OpenMpUtilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# C++ implementation of (complex conjugated) Wigner D function, spherical harmonics, and basis functions for polarized photoproduction moments
# also provides complexT typedef for std::complex<double>
OpenMpUtilities.enableRootACLiCOpenMp()
# OpenMpUtilities.printRootACLiCSettings()
assert ROOT.gROOT.LoadMacro("./basisFunctions.C+") == 0, "Error loading './basisFunctions.C'"


# see https://root-forum.cern.ch/t/tf1-eval-as-a-function-in-rdataframe/50699/3
def declareInCpp(**kwargs: Any) -> None:
  """Creates C++ variables (names = keys of kwargs) for PyROOT objects (values of kwargs) in PyVars:: namespace"""
  for key, value in kwargs.items():
    ROOT.gInterpreter.Declare(
f"""
namespace PyVars
{{
  auto& {key} = *reinterpret_cast<{type(value).__cpp_name__}*>({ROOT.addressof(value)});
}}
""")
