#!/usr/bin/env python3
"""
This module performs the moment analysis of unpolarized and polarized
pi+ pi- photoproduction data. The calculated moments are written to
files to be read by the plotting script `photoProdPlotMoments.py`.

Usage: Run this module as a script to perform the moment calculations
and to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import numpy as np
import os
import threadpoolctl

import ROOT
from wurlitzer import pipes, STDOUT

from AnalysisConfig import (
  AnalysisConfig,
  CFG_KEVIN,
  CFG_NIZAR,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_PWA,
  CFG_UNPOLARIZED_PIPP,
)
from MomentCalculator import (
  DataSet,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def calculateAllMoments(
  cfg:                   AnalysisConfig,
  timer:                 Utilities.Timer        = Utilities.Timer(),
  limitToDataEntryRange: tuple[int, int] | None = None,  # for debugging: limit analysis to entry range [begin, end) of real-data data tree
) -> None:
  """Performs the moment analysis for the given configuration"""
  # setup MomentCalculators for all mass bins
  momentCalculators = MomentCalculatorsKinematicBinning([])
  assert len(cfg.massBinning) > 0, f"Need at least one mass bin, but found {len(cfg.massBinning)}"
  with timer.timeThis(f"Time to load data and setup MomentCalculators for {len(cfg.massBinning)} bins"):
    print(f"Loading real data from tree '{cfg.treeName}' in file '{cfg.dataFileName}'")
    data = ROOT.RDataFrame(cfg.treeName, cfg.dataFileName)
    if limitToDataEntryRange is not None:
      print(f"Limiting analysis to entry range [{limitToDataEntryRange[0]}, {limitToDataEntryRange[1]}) of real data")
      data = data.Range(*limitToDataEntryRange)
    print(f"Loaded {data.Count().GetValue()} real-data events")
    print(f"Loading accepted phase-space data from tree '{cfg.treeName}' in file '{cfg.psAccFileName}'")
    dataPsAcc = ROOT.RDataFrame(cfg.treeName, cfg.psAccFileName)
    if cfg.limitNmbPsAccEvents > 0:
      dataPsAcc = dataPsAcc.Range(cfg.limitNmbPsAccEvents)  #!Caution! Range() switches to single-threaded mode
    dataPsGen = None
    if cfg.psGenFileName is not None:
      print(f"Loading generated phase-space data from tree '{cfg.treeName}' in file '{cfg.psGenFileName}'")
      dataPsGen = ROOT.RDataFrame(cfg.treeName, cfg.psGenFileName)
    else:
      print("??? Warning: File name for generated phase-space data was not provided. Cannot calculate acceptance correctly.")
    for massBinIndex, massBinCenter in enumerate(cfg.massBinning):
      massBinRange = cfg.massBinning.binValueRange(massBinIndex)
      print(f"Preparing {cfg.binVarMass.name} bin [{massBinIndex + 1} of {len(cfg.massBinning)}] at {massBinCenter} {cfg.binVarMass.unit} with range {massBinRange} {cfg.binVarMass.unit}")
      # load data for mass bin
      massBinFilter = cfg.massBinning.binFilter(massBinIndex)
      print(f"Applying filter '{massBinFilter}' to select kinematic bin")
      dataInBin = data.Filter(massBinFilter)
      print(f"Loaded {dataInBin.Count().GetValue()} data events; {dataInBin.Sum('eventWeight').GetValue()} background subtracted events")
      dataPsAccInBin = dataPsAcc.Filter(massBinFilter)
      nmbPsAccEvents = dataPsAccInBin.Count().GetValue()
      nmbPsGenEvents = None
      if dataPsGen is not None:
        dataPsGenInBin = dataPsGen.Filter(massBinFilter)
        nmbPsGenEvents = dataPsGenInBin.Count().GetValue()
        print(f"Loaded phase-space events: number generated = {nmbPsGenEvents}; "
              f"number accepted = {nmbPsAccEvents}, "
              f" -> efficiency = {nmbPsAccEvents / nmbPsGenEvents:.3f}")
      else:
        print(f"Loaded phase-space events: number accepted = {nmbPsAccEvents}")
      # setup moment calculators for data
      dataSet = DataSet(
        data           = dataInBin,
        phaseSpaceData = dataPsAccInBin,
        nmbGenEvents   = nmbPsGenEvents or nmbPsAccEvents,
        polarization   = cfg.polarization,
      )
      momentCalculators.append(
        MomentCalculator(
          indices              = MomentIndices(maxL = cfg.maxL, polarized = (cfg.polarization is not None)),
          dataSet              = dataSet,
          binCenters           = {cfg.binVarMass : massBinCenter},
          integralFileBaseName = f"{cfg.outFileDirName}/integralMatrix",
        )
      )

  # calculate integral matrix for all mass bins
  nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()
  with timer.timeThis(f"Time to calculate integral matrices for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads"):
    print(f"Calculating acceptance integral matrices for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads")
    momentCalculators.calculateIntegralMatrices(forceCalculation = True)
    print(f"Acceptance integral matrix for first bin at {cfg.massBinning[0]} {cfg.binVarMass.unit}:\n{momentCalculators[0].integralMatrix}")
    eigenVals, _ = momentCalculators[0].integralMatrix.eigenDecomp
    print(f"Sorted eigenvalues of acceptance integral matrix for first bin at {cfg.massBinning[0]} {cfg.binVarMass.unit}:\n{np.sort(eigenVals)}")

  # calculate moments of accepted phase-space data
  momentResultsFileBaseName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments"
  if cfg.calcAccPsMoments:
    with timer.timeThis(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating moments of phase-space MC data for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads")
      momentCalculators.calculateMoments(dataSource = MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE, normalize = cfg.normalizeMoments)
      momentCalculators.momentResultsMeas.savePickle(f"{momentResultsFileBaseName}_accPs_meas.pkl")
      momentCalculators.momentResultsPhys.savePickle(f"{momentResultsFileBaseName}_accPs_phys.pkl")

  # calculate moments of real data and write them to files
  #TODO calculate normalized and unnormalized moments
  with timer.timeThis(f"Time to calculate moments of real data for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads"):
    print(f"Calculating moments of real data for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads")
    momentCalculators.calculateMoments(normalize = cfg.normalizeMoments, nmbBootstrapSamples = cfg.nmbBootstrapSamples)
    momentCalculators.momentResultsMeas.savePickle(f"{momentResultsFileBaseName}_meas.pkl")
    momentCalculators.momentResultsPhys.savePickle(f"{momentResultsFileBaseName}_phys.pkl")
    with open(f"{momentResultsFileBaseName}_phys.json", "w") as jsonFile:  # write results to also to JSON file
      jsonFile.write(momentCalculators.momentResultsPhys.toJsonStr())


if __name__ == "__main__":
  cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarizedK- K_S Delta++ data
  # cfg = deepcopy(CFG_NIZAR)  # perform analysis of Nizar's polarized eta pi0 data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPP)  # perform analysis of unpolarized pi+ p data

  tBinLabels = (
    # "tbin_0.1_0.2",
    # "tbin_0.1_0.2.trackDistFdc",
    # "tbin_0.2_0.3",
    "tbin_0.1_0.5",
  )
  beamPolLabels = (
    "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
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
      # cfg.dataFileName       = f"./dataPhotoProdPiPiPol/{tBinLabel}/data_flat_{beamPolLabel}.root"
      # cfg.psAccFileName      = f"./dataPhotoProdPiPiPol/{tBinLabel}/phaseSpace_acc_flat_{beamPolLabel}.root"
      # cfg.psGenFileName      = f"./dataPhotoProdPiPiPol/{tBinLabel}/phaseSpace_gen_flat_{beamPolLabel}.root"
      cfg.outFileDirBaseName = f"{outFileDirBaseNameCommon}.{tBinLabel}/{beamPolLabel}"
      for maxL in maxLs:
        print(f"Performing moment analysis for t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
        cfg.maxL = maxL
        cfg.init(createOutFileDir = True)
        thisSourceFileName = os.path.basename(__file__)
        logFileName = f"{cfg.outFileDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
        print(f"Writing output to log file '{logFileName}'")
        with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
          Utilities.printGitInfo()
          timer = Utilities.Timer()
          ROOT.gROOT.SetBatch(True)
          threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
          print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
          with threadController.limit(limits = 4):
            print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")
            print(f"Using configuration:\n{cfg}")
            timer.start("Total execution time")
            calculateAllMoments(cfg, timer)
            timer.stop("Total execution time")
            print(timer.summary)
