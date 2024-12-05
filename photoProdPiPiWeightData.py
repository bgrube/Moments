#!/usr/bin/env python3
"""
This module weights accepted phase-space data with the intensity
calculated from the results of the moment analysis of unpolarized and
polarized pi+ pi- photoproduction data. The moment values are read
from files produced by the script `photoProdPiPiCalcMoments.py` that
calculates the moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import os
import subprocess

import ROOT
from wurlitzer import pipes, STDOUT

from photoProdPiPiCalcMoments import (
  AnalysisConfig,
  CFG_POLARIZED,
  CFG_UNPOLARIZED,
)
from MomentCalculator import MomentResultsKinematicBinning
from PlottingUtilities import setupPlotStyle
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def weightAccPhaseSpaceWithIntensity(
  intensityFormula:   str,  # formula for intensity calculation
  kinematicBinFilter: str,  # filter string that selects kinematic bin
  outFileName:        str,  # ROOT file to which weighted events are written
  cfg:                AnalysisConfig,
  seed:               int = 123456789,
) -> None:
  """Weight accepted phase-space MC data with given intensity formula"""
  # read accepted phase-space MC data
  psAccData = ROOT.RDataFrame(cfg.treeName, cfg.psAccFileName).Filter(kinematicBinFilter)
  nmbPsAccEvents = psAccData.Count().GetValue()
  print(f"File '{cfg.psAccFileName}' contains {nmbPsAccEvents} events in bin '{kinematicBinFilter}'")
  # calculate intensity weight and random number in [0, 1] for each event
  print(f"Calculating weights using formula '{intensityFormula}'")
  ROOT.gRandom.SetSeed(seed)
  psAccData = psAccData.Define("intensityWeight", f"(Double32_t){intensityFormula}") \
                       .Define("rndNmb",           "(Double32_t)gRandom->Rndm()")
  # determine maximum weight
  maxIntensityWeight = psAccData.Max("intensityWeight").GetValue()
  print(f"Maximum intensity is {maxIntensityWeight}")
  # accept each event with probability intensityWeight / maxIntensityWeight
  weightedPsAccData = psAccData.Define("acceptEvent", f"(bool)(rndNmb < (intensityWeight / {maxIntensityWeight}))") \
                               .Filter("acceptEvent == true")
  nmbWeightedEvents = weightedPsAccData.Count().GetValue()
  print(f"After intensity weighting sample contains {nmbWeightedEvents} events; efficiency is {nmbWeightedEvents / nmbPsAccEvents}")
  # write weighted data to file
  print(f"Writing data weighted with intensity function to file '{outFileName}'")
  weightedPsAccData.Snapshot(cfg.treeName, outFileName)


if __name__ == "__main__":
  # cfg = deepcopy(CFG_UNPOLARIZED)  # perform unpolarized analysis
  cfg = deepcopy(CFG_POLARIZED)    # perform polarized analysis

  # for maxL in (2, 4, 5, 8, 10, 12, 20):
  for maxL in (4, ):
    print(f"Generating weighted MC for L_max = {maxL}")
    cfg.maxL = maxL
    thisSourceFileName = os.path.basename(__file__)
    logFileName = f"{cfg.outFileDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
    print(f"Writing output to log file '{logFileName}'")
    with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
      Utilities.printGitInfo()
      timer = Utilities.Timer()
      ROOT.gROOT.SetBatch(True)
      setupPlotStyle()

      timer.start("Total execution time")

      momentResultsFileBaseName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments"
      # momentResults = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_phys.pkl")
      momentResults = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_pwa_SPD.pkl")
      for momentResult in momentResults:
        intensityFormula = momentResult.intensityFormula(
          polarization = cfg.polarization,
          thetaFormula = "theta",
          phiFormula   = "phi",
          PhiFormula   = "Phi",
          printFormula = False,
        )
        massBinCenter = momentResult.binCenters[cfg.massBinning.var]
        massBinIndex  = cfg.massBinning.findBin(massBinCenter)
        assert massBinIndex is not None, f"Could not find bin for mass value of {massBinCenter} GeV"
        # outFileBaseName = f"{cfg.outFileDirName}/psAccData_weighted_flat"
        outFileBaseName = f"{cfg.outFileDirName}/psAccData_weighted_pwa_SPD_flat"
        outFileName     = f"{outFileBaseName}_{massBinIndex}.root"
        print(f"Weighting accepted phase-space events for bin {massBinIndex} at {massBinCenter:.2f} {cfg.massBinning.var.unit}")
        weightAccPhaseSpaceWithIntensity(
          intensityFormula   = intensityFormula,
          kinematicBinFilter = cfg.massBinning.binFilter(massBinIndex),
          outFileName        = outFileName,
          cfg                = cfg,
        )

      # merge trees with weighted MC data
      outFileName = f"{outFileBaseName}.maxL_{cfg.maxL}.root"
      cmd = f"hadd {outFileName} {outFileBaseName}_*.root"
      print(f"Merging files: '{cmd}'")
      gitInfo = subprocess.run(cmd, shell = True)

      timer.stop("Total execution time")
      print(timer.summary)
