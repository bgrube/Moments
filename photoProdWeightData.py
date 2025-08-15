#!/usr/bin/env python3
"""
This module weights accepted phase-space data with the intensity
calculated from the results of the moment analysis of unpolarized and
polarized pi+ pi- photoproduction data. The moment values are read
from files produced by the script `photoProdCalcMoments.py` that
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

from AnalysisConfig import (
  AnalysisConfig,
  CFG_KEVIN,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_JPAC,
  CFG_UNPOLARIZED_PIPI_PWA,
)
from MomentCalculator import MomentResultsKinematicBinning
from PlottingUtilities import setupPlotStyle
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def weightDataWithIntensity(
  intensityFormula: str,  # formula for intensity function
  massBinIndex:     int,  # index of mass bin to generate data for
  outFileName:      str,  # ROOT file to which weighted events are written
  cfg:              AnalysisConfig,
  inputDataType:    AnalysisConfig.DataType | None = None,    # if `None`, phase-space distribution in angles is generated
  nmbGenPsEvents:   int                            = 100000,  # number phase-space events to generate
  seed:             int                            = 123456789,
) -> None:
  """Weight accepted phase-space MC data with given intensity formula"""
  kinematicBinFilter: str = cfg.massBinning.binFilter(massBinIndex)
  kinematicBinRange: tuple[float, float] = cfg.massBinning.binValueRange(massBinIndex)
  # get input data
  dataToWeight: ROOT.RDataFrame | None = None
  if inputDataType is None:
    print(f"Generating phase-space distribution with {nmbGenPsEvents} events")
    ROOT.gRandom.SetSeed(seed)
    dataToWeight = (
      ROOT.RDataFrame(nmbGenPsEvents)
          .Define("cosTheta", "(Double32_t)gRandom->Uniform(-1, +1)")
          .Define("theta",    "(Double32_t)std::acos(cosTheta)")
          .Define("phiDeg",   "(Double32_t)gRandom->Uniform(-180, +180)")
          .Define("phi",      "(Double32_t)phiDeg * TMath::DegToRad()")
          .Define("mass",    f"(Double32_t)gRandom->Uniform({kinematicBinRange[0]}, {kinematicBinRange[1]})")
          .Filter('if (rdfentry_ == 0) { cout << "Running event loop in weightAccPhaseSpaceWithIntensity()" << endl; } return true;')  # noop filter that just logs when event loop is running
    )
    if cfg.polarization is not None:
      # polarized case: add Phi and polarization columns
      dataToWeight = (
        dataToWeight.Define("PhiDeg",   "(Double32_t)gRandom->Uniform(-180, +180)")
                    .Define("Phi",      "(Double32_t)PhiDeg * TMath::DegToRad()")
      )
      if isinstance(cfg.polarization, float):
        dataToWeight = dataToWeight.Define("beamPol", f"(Double32_t){cfg.polarization}")
      elif isinstance(cfg.polarization, str):
        raise ValueError(f"Cannot read polarization from column '{cfg.polarization}'")
  else:
    print(f"Loading data of type '{inputDataType}'")
    dataToWeight = cfg.loadData(inputDataType)
    assert dataToWeight is not None, f"Could not load data of type '{inputDataType}'"
    dataToWeight = dataToWeight.Filter(kinematicBinFilter)
    nmbInputEvents = dataToWeight.Count().GetValue()
    print(f"Input data contain {nmbInputEvents} events in bin '{kinematicBinFilter}'")
  # calculate intensity weight and random number in [0, 1] for each event
  print(f"Calculating weights using formula '{intensityFormula}'")
  # ROOT.gRandom.SetSeed(seed)
  dataToWeight = (
    dataToWeight.Define("intensityWeight", f"(Double32_t){intensityFormula}")
                .Define("rndNmb",           "(Double32_t)gRandom->Rndm()")  # random number in [0, 1] for each event
  )
  tmpFileName = f"{outFileName}.unweighted.root"
  dataToWeight.Snapshot(cfg.treeName, tmpFileName)  # write unweighted data to file to ensure that random columns are filled only once
  dataToWeight = ROOT.RDataFrame(cfg.treeName, tmpFileName)  # read data back
  # determine maximum weight
  maxIntensityWeight = dataToWeight.Max("intensityWeight").GetValue()
  print(f"Maximum intensity is {maxIntensityWeight}")
  # accept each event with probability intensityWeight / maxIntensityWeight
  weightedData = (
    dataToWeight.Define("acceptEvent", f"(bool)(rndNmb < (intensityWeight / {maxIntensityWeight}))")
                .Filter("acceptEvent == true")
  )
  nmbWeightedEvents = weightedData.Count().GetValue()
  print(f"After weighting with the intensity function, the sample contains {nmbWeightedEvents} accepted events; generator efficiency is {nmbWeightedEvents / nmbGenPsEvents}")
  # write weighted data to file
  print(f"Writing data weighted with intensity function to file '{outFileName}'")
  weightedData.Snapshot(cfg.treeName, outFileName)
  subprocess.run(f"rm -f {tmpFileName}", shell = True)


if __name__ == "__main__":
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarized K- K_S Delta++ data

  tBinLabels = (
    # "tbin_0.1_0.5",
    "tbin_0.4_0.5",
  )
  beamPolLabels = (
    # "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
    "Unpol",
  )
  maxLs = (
    4,
    # 5,
    # 6,
    # 7,
    # 8,
  )

  outFileDirBaseNameCommon = cfg.outFileDirBaseName
  for tBinLabel in tBinLabels:
    for beamPolLabel in beamPolLabels:
      cfg.outFileDirBaseName = f"{outFileDirBaseNameCommon}.{tBinLabel}/{beamPolLabel}"
      for maxL in maxLs:
        print(f"Generating weighted MC for t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
        cfg.maxL = maxL
        cfg.init()
        thisSourceFileName = os.path.basename(__file__)
        logFileName = f"{cfg.outFileDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
        print(f"Writing output to log file '{logFileName}'")
        with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
          Utilities.printGitInfo()
          timer = Utilities.Timer()
          ROOT.gROOT.SetBatch(True)
          setupPlotStyle()

          print(f"Using configuration:\n{cfg}")
          timer.start("Total execution time")
          momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_phys.pkl"
          # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_pwa_SPD.pkl"
          print(f"Reading physical moments from file '{momentResultsFileName}'")
          momentResults = MomentResultsKinematicBinning.loadPickle(momentResultsFileName)
          for momentResultsForBin in momentResults:
            massBinCenter = momentResultsForBin.binCenters[cfg.massBinning.var]
            massBinIndex  = cfg.massBinning.findBin(massBinCenter)
            assert massBinIndex is not None, f"Could not find bin for mass value of {massBinCenter} GeV"
            outFileBaseName = f"{cfg.outFileDirName}/data_weighted_flat"
            # outFileBaseName = f"{cfg.outFileDirName}/data_weighted_pwa_SPD_flat"
            outFileName     = f"{outFileBaseName}_{massBinIndex}.root"
            print(f"Writing accepted phase-space events for bin {massBinIndex} at {massBinCenter:.2f} {cfg.massBinning.var.unit} weighted by intensity function into file '{outFileName}'")
            weightDataWithIntensity(
              intensityFormula = momentResultsForBin.intensityFormula(  #TODO include imaginary parts into intensity formula
                polarization = cfg.polarization,
                thetaFormula = "theta",
                phiFormula   = "phi",
                PhiFormula   = "Phi",
                printFormula = False,
              ),
              massBinIndex     = massBinIndex,
              outFileName      = outFileName,
              cfg              = cfg,
              # inputDataType    = AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE,
              inputDataType    = None,  # generate phase-space distribution in angles
              nmbGenPsEvents   = 100000,
              seed             = 12345 + massBinIndex,  # ensure random data in mass bins are independent
            )

          # merge trees with weighted MC data from different mass bins into single file
          nmbParallelJobs = 10
          with timer.timeThis(f"Time to merge ROOT files from all mass bins using hadd with {nmbParallelJobs} parallel jobs"):
            outFileName = f"{outFileBaseName}.root"
            cmd = f"hadd -j {nmbParallelJobs} {outFileName} {outFileBaseName}_*.root"
            print(f"Merging ROOT files from all mass bins: '{cmd}'")
            subprocess.run(cmd, shell = True)

          timer.stop("Total execution time")
          print(timer.summary)
