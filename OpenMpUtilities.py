"""Module that provides helper function to use OpenMp in ACLiC compiled code"""

import functools

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


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


def enableRootACLiCOpenMp() -> None:
  """Enables OpenMP support for ROOT macros compiled via ACLiC"""
  arch = ROOT.gSystem.GetBuildArch()
  if "macos" in arch.lower():
    # !Note! MacOS (Apple does not ship libomp; needs to be installed via MacPorts or Homebrew see testOpenMp.c)
    print(f"Enabling ACLiC compilation with OpenMP for MacOS")
    ROOT.gSystem.SetFlagsOpt("-Xpreprocessor -fopenmp")  # compiler flags for optimized mode
    ROOT.gSystem.AddIncludePath("-I/opt/local/include/libomp")
    ROOT.gSystem.AddDynamicPath("/opt/local/lib/libomp")
    ROOT.gSystem.AddLinkedLibs("-L/opt/local/lib/libomp -lomp")
  elif "linux" in arch.lower():
    print(f"Enabling ACLiC compilation with OpenMP for Linux")
    ROOT.gSystem.SetFlagsOpt("-fopenmp")  # compiler flags for optimized mode
    ROOT.gSystem.AddLinkedLibs("-lgomp")
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
    # ROOT.gSystem.AddIncludePath(f"-I \"{gccIncludePath}\"")  # this causes compile errors because it clashes with headers in `/usr/include/c++/4.8.5/`, which weirdly does not contain `omp.h`
    # ROOT.gSystem.AddIncludePath(f"-idirafter \"{gccIncludePath}\"")  # this has no effect # type: ignore
    # ROOT.gSystem.AddIncludePath(f"-isystem \"{gccIncludePath}\"")  # this has no effect # type: ignore
    #
    # The only way to make it work is to copy/link `omp.h` from the
    # gcc include dir here and turn the system include statement in
    # `wigner.C` into the quoted form.
