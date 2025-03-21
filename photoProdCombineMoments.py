#!/usr/bin/env python3
"""
This module combines the moment values from independent data samples.

Usage: Run this module as a script to perform the moment calculations
and to generate the output files.
"""


from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import functools
import os

from scipy.stats import moment
from wurlitzer import pipes, STDOUT

from MomentCalculator import (
  MomentResult,
  MomentResultsKinematicBinning,
)
from photoProdCalcMoments import (
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_PWA,
)
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def combineMomentResults(results: Sequence[MomentResult]) -> MomentResult:
  """Combines several `MomentResults` into one `MomentResult` by summing moment values and (co)variances"""
  assert len(results) > 0, "No `MomentResults` to combine"
  # ensure that all `MomentResults` have the same indices and bin centers
  assert all(results[0].hasSameMomentIndicesAndBinCenters(result) for result in results[1:]), "Moment results must have the same moment indices and bin centers"
  if any(result.hasBootstrapSamples for result in results):
    print(f"Warning: bootstrap samples are not combined.")
  # initialize combined MomentResult
  combinedMomentResult = MomentResult(
      indices    = results[0].indices,
      binCenters = results[0].binCenters,
      label      = "_".join((result.label for result in results)),
  )
  # sum moment values and (co)variances; see Eq. (220)
  # relies on arrays being initialized with zeros
  for result in results:
      combinedMomentResult._valsFlatIndex   += result._valsFlatIndex
      combinedMomentResult._V_ReReFlatIndex += result._V_ReReFlatIndex
      combinedMomentResult._V_ImImFlatIndex += result._V_ImImFlatIndex
      combinedMomentResult._V_ReImFlatIndex += result._V_ReImFlatIndex
  return combinedMomentResult


def combineMomentResultsKinematicBinning(results: Sequence[MomentResultsKinematicBinning]) -> MomentResultsKinematicBinning:
  """Combines several `MomentResultsKinematicBinning` into one `MomentResultsKinematicBinning` by summing moment values and (co)variances"""
  assert len(results) > 0, "No `MomentResultsKinematicBinning` to combine"
  # ensure that all `MomentResultsKinematicBinning` have the same bin centers
  assert all(results[0].binCenters == result.binCenters for result in results[1:]), "Moment results must have the same bin centers"
  # combine moments in each kinematic bin
  combinedMomentResults = MomentResultsKinematicBinning(
    moments = [combineMomentResults([result[binIndex] for result in results]) for binIndex in range(len(results[0]))]
  )
  return combinedMomentResults


if __name__ == "__main__":
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data

  tBinLabels = (
    # "tbin_0.1_0.2",
    "tbin_0.1_0.2.trackDistFdc",
    # "tbin_0.2_0.3",
  )
  dataSetsToCombine = {
    # "0_90"      : ("PARA_0",   "PERP_90"),
    # "-45_45"    : ("PARA_135", "PERP_45"),
    # "0_-45"     : ("PARA_0",   "PARA_135"),
    # "45_90"     : ("PERP_45",  "PERP_90"),
    "allOrient" : ("PARA_0", "PARA_135", "PERP_45", "PERP_90"),
  }
  maxLs = (
    4,
    # 5,
    6,
    # 7,
    8,
  )
  momentsFileName = "_moments_phys.pkl"

  thisSourceFileName = os.path.basename(__file__)
  for tBinLabel in tBinLabels:
    for maxL in maxLs:
      cfg.maxL = maxL
      for labelCombined, beamPolLabels in dataSetsToCombine.items():
        print(f"Combining moments for data sets '{beamPolLabels}' for t bin '{tBinLabel}' and L_max = {maxL}")
        # constructing input file names
        momentResultsFileNames = []
        for beamPolLabel in beamPolLabels:
          cfg.outFileDirBaseName = f"./plotsPhotoProdPiPiPol.{tBinLabel}/{beamPolLabel}"
          momentResultsFileNames.append(f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}{momentsFileName}")
        cfg.outFileDirBaseName = f"./plotsPhotoProdPiPiPol.{tBinLabel}/{labelCombined}"
        logFileName = f"{cfg.outFileDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
        print(f"Writing output to log file '{logFileName}'")
        with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
          Utilities.printGitInfo()
          timer = Utilities.Timer()
          timer.start("Total execution time")
          print(f"Using configuration:\n{cfg}")

          print(f"Combining moments from {momentResultsFileNames}")
          momentResultsToCombine = tuple(MomentResultsKinematicBinning.load(momentResultsFileName) for momentResultsFileName in momentResultsFileNames)
          momentResultsCombined = combineMomentResultsKinematicBinning(momentResultsToCombine)
          momentResultsCombinedFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}{momentsFileName}"
          print(f"Writing PWA moments to file '{momentResultsCombinedFileName}'")
          momentResultsCombined.save(momentResultsCombinedFileName)

          timer.stop("Total execution time")
          print(timer.summary)
