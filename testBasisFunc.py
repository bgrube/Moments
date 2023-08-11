#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "5"  # limit number of OpenMP threads; see also threadpoolctl

import functools
import numpy as np
import subprocess

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def printRootACLiCSettings():
  '''Prints ROOT settings that affect ACLiC compilation'''
  print(f" GetBuildArch()               = {ROOT.gSystem.GetBuildArch()}")  # type: ignore
  print(f" GetBuildCompiler()           = {ROOT.gSystem.GetBuildCompiler()}")  # type: ignore
  print(f" GetBuildCompilerVersion()    = {ROOT.gSystem.GetBuildCompilerVersion()}")  # type: ignore
  print(f" GetBuildCompilerVersionStr() = {ROOT.gSystem.GetBuildCompilerVersionStr()}")  # type: ignore
  print(f" GetBuildDir()                = {ROOT.gSystem.GetBuildDir()}")  # type: ignore
  print(f" GetFlagsDebug()              = {ROOT.gSystem.GetFlagsDebug()}")  # type: ignore
  print(f" GetFlagsOpt()                = {ROOT.gSystem.GetFlagsOpt()}")  # type: ignore
  print(f" GetIncludePath()             = {ROOT.gSystem.GetIncludePath()}")  # type: ignore
  print(f" GetDynamicPath()             = {ROOT.gSystem.GetDynamicPath()}")  # type: ignore
  print(f" GetLinkedLibs()              = {ROOT.gSystem.GetLinkedLibs()}")  # type: ignore
  print(f" GetLinkdefSuffix()           = {ROOT.gSystem.GetLinkdefSuffix()}")  # type: ignore
  print(f" GetLibraries()               = {ROOT.gSystem.GetLibraries()}")  # type: ignore
  print(f" GetMakeExe()                 = {ROOT.gSystem.GetMakeExe()}")  # type: ignore
  print(f" GetMakeSharedLib()           = {ROOT.gSystem.GetMakeSharedLib()}")  # type: ignore


def enableRootACLiCOpenMp():
  '''Enables openMP support for ROOT macros compiled via ACLiC'''
  arch = ROOT.gSystem.GetBuildArch()  # type: ignore
  if "macos" in arch.lower():
    # !Note! MacOS (Apple does not ship libomp; needs to be installed via MacPorts or Homebrew see testOpenMp.c)
    print(f"Enabling ACLiC compilation with OpenMP for MacOS")
    ROOT.gSystem.SetFlagsOpt("-Xpreprocessor -fopenmp")  # compiler flags for optimized mode  # type: ignore
    ROOT.gSystem.AddIncludePath("-I/opt/local/include/libomp")  # type: ignore
    ROOT.gSystem.AddDynamicPath("/opt/local/lib/libomp")  # type: ignore
    ROOT.gSystem.AddLinkedLibs("-L/opt/local/lib/libomp -lomp")  # type: ignore
  elif "linux" in arch.lower():
    print(f"Enabling ACLiC compilation with OpenMP for Linux")
    ROOT.gSystem.SetFlagsOpt("-fopenmp")  # compiler flags for optimized mode  # type: ignore
    ROOT.gSystem.AddLinkedLibs("-lgomp")  # type: ignore
    # For usual setups, the above should be sufficient.
    # On ifarm, however, ROOT seems to be compiled with a weird gcc setup.
    # ACLiC does not search the gcc include directory at `/usr/lib/gcc/x86_64-redhat-linux/4.8.5/include`.`
    # Instead, it seems to go to /usr/lib/gcc/x86_64-redhat-linux/4.8.5/../../../../include/c++/4.8.5/,
    # which is /usr/include/c++/4.8.5/ but does not contain omp.h.
    # There seems to be no way to fix this without changing the source code.
    # ACLiC always complaints that it cannot find `omp.h`.`
    # # get gcc include directory
    # result = subprocess.run("gcc -print-search-dirs",
    #                         shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, universal_newlines = True)
    # lines= result.stdout.split("\n")
    # gccIncludePath = None
    # for line in lines:
    #   if "install:" in line:
    #     gccIncludePath = line.split(": ")[1] + "include"
    # ROOT.gSystem.AddIncludePath(f"-I \"{gccIncludePath}\"")  # this causes compile errors because it clashes with headers in `/usr/include/c++/4.8.5/`, which weirdly does not contain `omp.h``  # type: ignore
    # ROOT.gSystem.AddIncludePath(f"-idirafter \"{gccIncludePath}\"")  # this has no effect # type: ignore
    # ROOT.gSystem.AddIncludePath(f"-isystem \"{gccIncludePath}\"")  # this has no effect # type: ignore
    #
    # The only way to make it work is to copy/link `omp.h`` from the
    # gcc include dir here and turn the system include statement in
    # `wigner.C`` into the quoted form.
enableRootACLiCOpenMp()


# C++ implementation of (complex conjugated) Wigner D function and spherical harmonics
# also provides complexT typedef for std::complex<double>
ROOT.gROOT.LoadMacro("./wignerD.C++")  # type: ignore


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore

  printRootACLiCSettings()
  ROOT.testOpenMp()  # type: ignore

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
  fcnResults = ROOT.f_meas(1, 3, 2, theta, phi, Phi, 1.0)  # type: ignore
  arr = np.asarray(fcnResults)
  print(f"{type(arr)} {arr.dtype} {arr}")
