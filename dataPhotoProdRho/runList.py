#!/usr/bin/env python2
# RCDB is still a Python 2 module

from __future__ import absolute_import, division, print_function

import glob
import os
import re
import sys

import rcdb


RCDB_CONNECTION_STRING = os.environ["RCDB_CONNECTION"] if os.environ.get("RCDB_CONNECTION") else "mysql://rcdb@hallddb.jlab.org/rcdb"


# assumed file-name patterns
FILE_NAME_PATTERNS = (
  re.compile(r"(?P<begin>.*)(?P<runNmb>_[0-9]{6})(?P<end>\.root.*)"),                        # for real data: <anything>_<6-digit run number>.root<anything>
  re.compile(r"(?P<begin>.*)(?P<runNmb>_[0-9]{6})(?P<chunkNmb>_[0-9]{3})(?P<end>\.root.*)")  # for MC data:   <anything>_<6-digit run number>_<3-digit chunk number>.root<anything>
)


# see https://halldweb.jlab.org/wiki-private/index.php/GlueX_Phase-I_Dataset_Summary
# and https://halldweb.jlab.org/wiki-private/index.php/GlueX_Phase-II_Dataset_Summary
# and https://halldweb.jlab.org/wiki/index.php/Run_Periods
# as of Feb 02, 2023
RUN_PERIOD_INFO = {
  "2017_01" : {  # Spring 2017
    "runNmbRange"       : (30274, 31057),
    "rcdbSearchString"  : "@is_production and @status_approved",
    "cohPeakEbeamRange" : (8.2, 8.8),  # [GeV]
    "cohPeakLuminosity" : 21.8         # [pb^{-1}]
  },
  "2018_01" : {  # Spring 2018
    "runNmbRange"       : (40856, 42559),
    "rcdbSearchString"  : "@is_2018production and @status_approved",
    "cohPeakEbeamRange" : (8.2, 8.8),  # [GeV]
    "cohPeakLuminosity" : 63.0         # [pb^{-1}]
  },
  "2018_08" : {  # Fall 2018
    "runNmbRange"       : (50685, 51768),
    "rcdbSearchString"  : "@is_2018production and @status_approved and beam_on_current>49",
    "cohPeakEbeamRange" : (8.2, 8.8),  # [GeV]
    "cohPeakLuminosity" : 40.1         # [pb^{-1}]
  },
  "2019_11" : {  # Spring 2020
    "runNmbRange"       : (71350, 73266),
    "rcdbSearchString"  : "@is_dirc_production and @status_approved",
    "cohPeakEbeamRange" : (8.0, 8.6),  # [GeV]
    "cohPeakLuminosity" : 132.4        # [pb^{-1}]
  }
}


# assumes that run-period name starts with <4-digit year>_<2-digit month> and returns only this part
def getRunPeriod(runPeriodName):
  # ignore everything after the first two fields separated by '_'
  field = runPeriodName.split("_")
  return "_".join( (field[0], field[1][:2]) )


def getRunNmbRange(runPeriodName):
  runPeriod = getRunPeriod(runPeriodName)
  if not runPeriod in RUN_PERIOD_INFO:
    raise KeyError("Unknown run period '" + runPeriod + "'")
  return RUN_PERIOD_INFO[runPeriod]["runNmbRange"]


def getCohPeakEbeamRange(runPeriodName):
  runPeriod = getRunPeriod(runPeriodName)
  if not runPeriod in RUN_PERIOD_INFO:
    raise KeyError("Unknown run period '" + runPeriod + "'")
  return RUN_PERIOD_INFO[runPeriod]["cohPeakEbeamRange"]


def getCohPeakLuminosity(runPeriodName):
  runPeriod = getRunPeriod(runPeriodName)
  if not runPeriod in RUN_PERIOD_INFO:
    raise KeyError("Unknown run period '" + runPeriod + "'")
  return RUN_PERIOD_INFO[runPeriod]["cohPeakLuminosity"]


# returns list of run numbers for given run range that fulfill condition in searchString
#!NOTE! this function won't work with Python 3
def getRcdbRunNmbListForRange(
  runNmbMin    = 0,
  runNmbMax    = sys.maxsize,
  searchString = "@is_production and @status_approved"
):
  rcDatabase = rcdb.RCDBProvider(RCDB_CONNECTION_STRING)
  runList = rcDatabase.select_runs(searchString, runNmbMin, runNmbMax).get_values(condition_names = [], insert_run_number = True)
  runList = [int(run[0]) for run in runList]
  return runList


# returns list of run numbers for given run-period name
#!NOTE! this function won't work with Python 3
def getRcdbRunNmbListForRunPeriod(runPeriodName):
  runPeriod = getRunPeriod(runPeriodName)
  if not runPeriod in RUN_PERIOD_INFO:
    raise KeyError("Unknown run period '" + runPeriod + "'")
  info = RUN_PERIOD_INFO[runPeriod]
  return getRcdbRunNmbListForRange(info["runNmbRange"][0], info["runNmbRange"][1], info["rcdbSearchString"])


# returns string
def getRunNmbFromFileNameAsStr(fileName):
  for pattern in FILE_NAME_PATTERNS:
    match = pattern.search(fileName)
    if match:
      return match.group('runNmb')[1:]  # remove leading "_"
  raise ValueError("File name '{}' does not match a known pattern; cannot extract run number".format(fileName))

# returns int
def getRunNmbFromFileNameAsInt(fileName):
  return int(getRunNmbFromFileNameAsStr(fileName))


# removes "_<run number>" (and for MC also "_<chunk number>") from file name
def removeRunNmbFromFileName(fileName):
  for pattern in FILE_NAME_PATTERNS:
    result = pattern.subn(r"\g<begin>\g<end>", fileName)
    if result[1] == 1:
      # pattern matched exactly once and replaced
      return result[0]
  raise ValueError("File name '{}' does not match a known pattern; cannot remove run number".format(fileName))


# returns list of strings
def getRunNmbsFromFileNamesStr(fileNames):
  return [getRunNmbFromFileNameAsStr(fileName) for fileName in fileNames]

# returns list of ints
def getRunNmbsFromFileNamesInt(fileNames):
  return [getRunNmbFromFileNameAsInt(fileName) for fileName in fileNames]


# returns list of run numbers in found files that belong to given period
def getRunNmbsFromFileNamePattern(
  fileNamePattern,
  runPeriod = None
):
  runNmbsInDir    = getRunNmbsFromFileNamesInt(glob.glob(fileNamePattern))
  runNmbRange     = getRunNmbRange(runPeriod) if runPeriod else (0, sys.maxsize)
  runNmbsInPeriod = [runNmb for runNmb in runNmbsInDir if runNmbRange[0] <= runNmb <= runNmbRange[1]]
  print("Found {} runs for period '{}' using '{}'".format(len(runNmbsInPeriod), runPeriod if runPeriod else "All", fileNamePattern))
  return runNmbsInPeriod


def compareRunLists(
  runLists,            # iterable with two iterables to compare
  labels = ("A", "B")  # iterable with labels
):
  assert len(runLists) == 2, "The `runList` argument hat to have length 2; provided argument has length {}".format(len(runLists))
  assert len(labels)   == 2, "the `labels` argument hat to have length 2; provided argument has length {}".format(len(labels))
  runSets = (set(runLists[0]), set(runLists[1]))
  if runSets[0] == runSets[1]:
    print("Run lists '{}' and '{}' are identical; both consist of {} runs".format(labels[0], labels[1], len(runSets[0])))
  else:
    print("Run lists '{}' and '{}' differ".format(*labels))
    if len(runSets[0]) != len(runSets[1]):
      print("    Run lists have different number of runs: {} runs for '{}' and {} runs for '{}'".format(len(runSets[0]), labels[0], len(runSets[1]), labels[1]))
    diff = runSets[0] - runSets[1]
    if diff:
      print("    {} run(s) in '{}' but not not in '{}': ".format(len(diff), *labels), end = "")
      print(*diff, sep = ", ")
    diff = runSets[1] - runSets[0]
    if diff:
      print("    {} run(s) in '{}' but not not in '{}': ".format(len(diff), *labels[::-1]), end = "")
      print(*diff, sep = ", ")


if __name__ == "__main__":
  # runPeriod       = "2017_01-ver20"
  # runPeriod       = "2018_01-ver02"
  runPeriod       = "2018_08-ver02"
  # runPeriod       = "2019_11-ver06"
  fileNamePattern = "tree_pippimetapr__B4_M35_*.root"
  # runPeriod       = "2018_08-ver10"
  # fileNamePattern = "tree_pippippimpimeta__T1_S2_*.root"
  dirName         = "/w/halld-scshelf2101/bgrube/pippimetapr/Submit/" + runPeriod + "/merged"  # part files
  # dirName         = "/volatile/halld/home/bgrube/pippimetapr/FSRoot_RD/" + runPeriod  # flat-tree files

  runList = getRunNmbsFromFileNamePattern(dirName + "/" + fileNamePattern, runPeriod)
  compareRunLists((runList, getRunNmbsFromFileNamesInt(glob.glob(dirName + "/" + fileNamePattern))), ("period", "glob"))
  compareRunLists((runList, getRcdbRunNmbListForRunPeriod(runPeriod)), ("period", "rcdb"))

  fileNames  = [
    "/volatile/halld/home/bgrube/pippimetapr/FSRoot_RD/2019_11-ver06/tree_pippimetapr__B4_M35_073266.root.Chi2Rank",
    "../Submit/MCPhaseSpace/2018-08/thrown/tree_thrown_genr8_FSROOTEXAMPLE_051290_002.root",
    "../Submit/MCPhaseSpace/2018-08/trees/tree_pippimetapr__etapr_pippimeta__M35_genr8_FSROOTEXAMPLE_051768_002.root"
  ]
  for name in fileNames:
    print(removeRunNmbFromFileName(name))
  print()
