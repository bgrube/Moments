"""Module that provides utility functions of general scope"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import contextlib
from dataclasses import (
  dataclass,
  field,
)
import functools
import os
import subprocess
import sys
import time

import ROOT

# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def printGitInfo() -> None:
  """Prints directory of this file and git hash in this directory"""
  repoDir = os.path.dirname(os.path.abspath(__file__))
  gitInfo = subprocess.check_output(["git", "describe", "--always"], cwd = repoDir).strip().decode()
  print(f"Running code in '{repoDir}', git version '{gitInfo}'")


def print_command_line_arguments(args: argparse.Namespace) -> None:
  """Prints all command-line arguments and their values and the git hash."""
  launched_script_file_name = os.path.basename(sys.argv[0])  # get file name of script that was launched
  print("-------------------------------------------------------------------------------")
  print(f"Running script {launched_script_file_name} with arguments:")
  max_arg_name_length = max(len(arg_name) for arg_name in vars(args).keys())
  for arg_name, arg_value in sorted(vars(args).items()):  # sort keys for stable, tidy output
    print(f"{arg_name:>{max_arg_name_length + 2}} : {arg_value}")
  printGitInfo()
  print("-------------------------------------------------------------------------------")


#TODO is this really needed?
def makeDirPath(dirPath: str) -> str:
  """Create path to directory and return directory path as given by argument"""
  try:
    os.makedirs(dirPath, exist_ok = False)
  except FileExistsError:
    pass  # directory already exists; do nothing
  except Exception:
    raise  # something went wrong
  else:
    print(f"Created directory '{dirPath}'")
  return dirPath


@dataclass
class TimeData:
  """Holds start and stop times for wall and cpu timer"""
  wallTimeStart: float
  cpuTimeStart:  float
  wallTimeStop:  float | None = None
  cpuTimeStop:   float | None = None

  def stop(self) -> None:
    """Sets stop times"""
    self.wallTimeStop = time.monotonic()
    self.cpuTimeStop  = time.process_time()

  @property
  def wallTime(self) -> float | None:
    """Returns elapsed wall time"""
    return None if self.wallTimeStop is None else self.wallTimeStop - self.wallTimeStart

  @property
  def cpuTime(self) -> float | None:
    """Returns elapsed CPU time"""
    return None if self.cpuTimeStop  is None else self.cpuTimeStop  - self.cpuTimeStart

  @property
  def summary(self) -> str | None:
    """Returns string that summarizes wall and CPU time"""
    strings = []
    if self.wallTime is not None:
      strings.append(f"wall time = {self.wallTime:.4g} sec")
    if self.cpuTime is not None:
      strings.append(f"CPU time = {self.cpuTime:.4g} sec")
    summary = ", ".join(strings)
    return summary if summary else None


@dataclass
class Timer:
  """Measures time differences"""
  _times: dict[str, TimeData] = field(default_factory = lambda: {})  # stores start and stop times for wall time and CPU time indexed by name

  def start(
    self,
    name: str,
  ) -> TimeData:
    """Creates or updates the timer associated with the given name"""
    t = TimeData(wallTimeStart = time.monotonic(), cpuTimeStart = time.process_time())
    self._times[name] = t
    return t

  def stop(
    self,
    name: str,
  ) -> TimeData | None:
    """Stops the timer associated with given name"""
    if name not in self._times:
      # gracefully ignore unknown timers
      return None
    t = self._times[name]
    t.stop()
    return t

  @contextlib.contextmanager
  def timeThis(
    self,
    name: str
  ):
    """Context manager that measures time of enclosed code block"""
    try:
      t = self.start(name)
      yield
    finally:
      t.stop()

  @property
  def summary(self) -> str | None:
    """Returns string with summary of all timers"""
    strings = []
    for name, timeData in self._times.items():
      if timeData.summary is not None:
        strings.append(f"{name}: {timeData.summary}")
    summary = "\n".join(strings)
    return summary if summary else None


DATA_TCHAINS: list[ROOT.TChain] = []  # use global variable to avoid garbage collection
def getDataFrameWithCorrectEventWeights(
  dataSigRegionFilePaths:    Sequence[str],  # file paths of input data files for signal region
  dataBkgRegionFilePaths:    Sequence[str],  # file paths of input data files for background region
  treeName:                  str,            # name of tree in input files
  sigRegionWeightFormula:    str  = "Weight",   # formula for calculating event weight for signal events
  bkgRegionWeightFormula:    str  = "-Weight",  # formula for calculating event weight for background events
  friendSigRegionFilePath:   str  = "./data_sig.root.weights",  # file path for friend tree that contains event weights for signal region
  friendBkgRegionFilePath:   str  = "./data_bkg.root.weights",  # file path for friend tree that contains event weights for background region
  forceOverwriteFriendFiles: bool = True,  # if False existing friend files will be used and assumed to contain the correct event weights
  weightColNameOutput:       str  = "eventWeight",  # name of column in friend trees that contains event weights
) -> ROOT.RDataFrame:
  """Creates friend trees with correct event weights and attaches them to data tree; must not be used in multi-threaded mode"""
  if ROOT.IsImplicitMTEnabled():
    raise RuntimeError("getDataFrameWithCorrectEventWeights() must not be used in multi-threaded mode")
  # write corrected weights into friend trees
  for dataFilePath, weightFormula, friendFilePath in (
    (dataSigRegionFilePaths, sigRegionWeightFormula, friendSigRegionFilePath),
    (dataBkgRegionFilePaths, bkgRegionWeightFormula, friendBkgRegionFilePath),
  ):
    print(f"Processing file(s) {dataFilePath}")
    if not forceOverwriteFriendFiles and os.path.exists(friendFilePath):
      print(f"File '{friendFilePath}' already exists, skipping creation of event-weight friend tree")
      continue
    print(f"Writing friend tree '{treeName}' with '{weightColNameOutput}' = '{weightFormula}' column to file '{friendFilePath}'")
    ROOT.RDataFrame(treeName, dataFilePath) \
        .Define(weightColNameOutput, weightFormula) \
        .Snapshot(treeName, friendFilePath, [weightColNameOutput])  #!NOTE! when multi-threading is enabled, the order of entries is not guaranteed to be preserved
  # chain trees for signal and background regions and add friend trees with weights
  dataTChain   = ROOT.TChain(treeName)
  weightTChain = ROOT.TChain(treeName)
  for dataFilePath, friendFilePath in (
    (dataSigRegionFilePaths, friendSigRegionFilePath),
    (dataBkgRegionFilePaths, friendBkgRegionFilePath),
  ):
    for dataFilePath in dataFilePath:
      dataTChain.Add(dataFilePath)
    weightTChain.Add(friendFilePath)
  dataTChain.AddFriend(weightTChain)
  #TODO have a look at <https://root.cern/doc/v632/classROOT_1_1RDataFrame.html#rdf-from-spec> to build data frame.
  DATA_TCHAINS.append(dataTChain)  # avoid garbage collection of TChain
  return ROOT.RDataFrame(dataTChain)
