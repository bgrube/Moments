"""Module that provides functions for using ROOT code"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Callable

import ROOT

from . import OpenMpUtilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def runOnce(func) -> Callable[..., Any | None]:
  """Decorator that ensures that the decorated function is executed only once; subsequent calls return the result of the first call"""
  threadLock = threading.Lock()
  firstCall  = True
  funcResult = None  # store result of first function call to return it for all subsequent calls

  def wrapper(*args, **kwargs) -> Any | None:
    nonlocal firstCall, funcResult
    if not firstCall:
      return funcResult
    with threadLock:
      if firstCall:
        funcResult = func(*args, **kwargs)
        firstCall = False
    return funcResult

  return wrapper


@runOnce
def loadFSROOTLibraries() -> None:
  """Loads FSROOT libraries"""
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"


#TODO put this into a function to make loading more explicit
# C++ implementation of (complex conjugated) Wigner D function, spherical harmonics, and basis functions for polarized photoproduction moments
# also provides complexT typedef for std::complex<double>
OpenMpUtilities.enableRootACLiCOpenMp()
# OpenMpUtilities.printRootACLiCSettings()
assert ROOT.gROOT.LoadMacro("./cpp/basisFunctions.C+") == 0, "Error loading './cpp/basisFunctions.C'"


# see https://root-forum.cern.ch/t/tf1-eval-as-a-function-in-rdataframe/50699/3
def declareInCpp(**kwargs: Any) -> None:
  """Creates C++ variables (names = keys of kwargs) for PyROOT objects (values of kwargs) in PyVars:: namespace"""
  for key, value in kwargs.items():
    if hasattr(ROOT, "PyVars") and key in dir(ROOT.PyVars):
      # variable was already defined before; assign new value
      ROOT.gInterpreter.ProcessLine(  # cannot use Declare() here; this also prevents us from using include-guard logic
        f"PyVars::{key} = *reinterpret_cast<{type(value).__cpp_name__}*>({ROOT.addressof(value)});")
    else:
      # define new Python variable in C++
      ROOT.gInterpreter.Declare(
f"""
namespace PyVars
{{
  auto& {key} = *reinterpret_cast<{type(value).__cpp_name__}*>({ROOT.addressof(value)});
}};
""")
