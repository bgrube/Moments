"""Module that provides functions for using ROOT code"""

from typing import Any

import ROOT

import OpenMp


# C++ implementation of (complex conjugated) Wigner D function and spherical harmonics
# also provides complexT typedef for std::complex<double>
OpenMp.enableRootACLiCOpenMp()
# OpenMp.printRootACLiCSettings()
ROOT.gROOT.LoadMacro("./wignerD.C++")


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
