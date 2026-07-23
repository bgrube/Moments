"""Module that provides functions for using ROOT code"""

from __future__ import annotations

import functools
import os
import socket
import subprocess
import threading
from typing import Any, Callable

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def runOnce(func) -> Callable[..., Any | None]:
  """Decorator that ensures that the decorated function is executed only once; subsequent calls only return the result of the first call"""
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


def printRootACLiCSettings() -> None:
  """Prints ROOT settings that affect ACLiC compilation"""
  print(f" GetBuildArch()               = {ROOT.gSystem.GetBuildArch()}")
  print(f" GetBuildCompiler()           = {ROOT.gSystem.GetBuildCompiler()}")
  print(f" GetBuildCompilerVersion()    = {ROOT.gSystem.GetBuildCompilerVersion()}")
  print(f" GetBuildCompilerVersionStr() = {ROOT.gSystem.GetBuildCompilerVersionStr()}")
  print(f" GetBuildDir()                = {ROOT.gSystem.GetBuildDir()}")
  print(f" GetFlagsDebug()              = {ROOT.gSystem.GetFlagsDebug()}")
  print(f" GetFlagsOpt()                = {ROOT.gSystem.GetFlagsOpt()}")
  print(f" GetIncludePath()             = {ROOT.gSystem.GetIncludePath()}")
  print(f" GetDynamicPath()             = {ROOT.gSystem.GetDynamicPath()}")
  print(f" GetLinkedLibs()              = {ROOT.gSystem.GetLinkedLibs()}")
  print(f" GetLinkdefSuffix()           = {ROOT.gSystem.GetLinkdefSuffix()}")
  print(f" GetLibraries()               = {ROOT.gSystem.GetLibraries()}")
  print(f" GetMakeExe()                 = {ROOT.gSystem.GetMakeExe()}")
  print(f" GetMakeSharedLib()           = {ROOT.gSystem.GetMakeSharedLib()}")


@runOnce
def enableRootACLiCOpenMp(verbose: bool = False) -> None:
  """Enables OpenMP support for ROOT macros compiled via ACLiC"""
  arch = ROOT.gSystem.GetBuildArch()
  if "macos" in arch.lower():
    #!NOTE! MacOS (Apple does not ship libomp; needs to be installed via MacPorts or Homebrew see testOpenMp.c)
    print(f"Enabling ACLiC compilation with OpenMP for MacOS")
    ROOT.gSystem.SetFlagsOpt("-Xpreprocessor -fopenmp")  # compiler flags for optimized mode
    ROOT.gSystem.AddIncludePath("-I/opt/local/include/libomp")
    ROOT.gSystem.AddDynamicPath("/opt/local/lib/libomp")
    ROOT.gSystem.AddLinkedLibs("-L/opt/local/lib/libomp -lomp")
  elif "linux" in arch.lower():
    print(f"Enabling ACLiC compilation with OpenMP for Linux")
    ROOT.gSystem.SetFlagsOpt("-fopenmp")  # compiler flags for optimized mode
    ROOT.gSystem.AddLinkedLibs("-lgomp")
  if verbose:
    printRootACLiCSettings()


@runOnce
def loadBasisFunctionsLibrary(
  enableOpenMp:       bool = True,
  forceRecompilation: bool = False,
) -> None:
  """Loads C++ implementation of Wigner D function, spherical harmonics, and basis functions for moments; also provides complexT typedef for std::complex<double>"""
  if enableOpenMp:
    enableRootACLiCOpenMp()
  cppSourceFilePath = "./cpp/basisFunctions.C"
  assert ROOT.gROOT.LoadMacro(f"{cppSourceFilePath}+{'+' if forceRecompilation else ''}") == 0, f"Error loading '{cppSourceFilePath}'"


@runOnce
def loadFSROOTLibraries() -> None:
  """Loads FSROOT libraries"""
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  macroFilePath = f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro(macroFilePath) == 0, f"Error loading '{macroFilePath}'"


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
