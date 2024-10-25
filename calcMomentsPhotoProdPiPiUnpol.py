#!/usr/bin/env python3
"""
This module performs the moment analysis of unpolarized pi+ pi- photoproduction data in the CLAS energy range.
The calculated moments are written to files to be read by the plotting script.

Usage:
Run this module as a script to perform the moment calculations and to generate the output files.
"""


from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
)
import functools
import numpy as np
import threadpoolctl

import ROOT
from wurlitzer import pipes, STDOUT

from MomentCalculator import (
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
)
from PlottingUtilities import (
  HistAxisBinning,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads basisFunctions.C
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class AnalysisConfig:
  """Stores configuration parameters for the moment analysis"""
  treeName:                 str                      = "PiPi"
  dataFileName:             str                      = f"./dataPhotoProdPiPiUnpol/data_flat.root"
  psAccFileName:            str                      = f"./dataPhotoProdPiPiUnpol/phaseSpace_acc_flat.root"
  psGenFileName:            str                      = f"./dataPhotoProdPiPiUnpol/phaseSpace_gen_flat.root"
  maxL:                     int                      = 8
  outFileDirName:           str                      = field(init = False)
  outFileNamePrefix:        str                      = field(init = False)
  normalizeMoments:         bool                     = False
  nmbBootstrapSamples:      int                      = 0
  # nmbBootstrapSamples:      int                      = 10000
  # plotAngularDistributions: bool                     = True
  plotAngularDistributions: bool                     = False
  # plotAccIntegralMatrices:  bool                     = True
  plotAccIntegralMatrices:  bool                     = False
  calcAccPsMoments:         bool                     = True
  # calcAccPsMoments:         bool                     = False
  # plotAccPsMoments:         bool                     = True
  plotAccPsMoments:         bool                     = False
  limitNmbPsAccEvents:      int                      = 0
  # limitNmbPsAccEvents:      int                      = 100000
  binVarMass:               KinematicBinningVariable = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}", unit = "GeV/#it{c}^{2}", nmbDigits = 3)
  massBinning:              HistAxisBinning          = HistAxisBinning(nmbBins = 100, minVal = 0.4, maxVal = 1.4, _var = binVarMass)  # same binning as used by CLAS
  # massBinning:              HistAxisBinning          = HistAxisBinning(nmbBins = 1, minVal = 1.25, maxVal = 1.29, _var = binVarMass)  # f_2(1270) region

  def __post_init__(self) -> None:
    """Creates output directory and initializes member variables"""
    self.outFileDirName    = Utilities.makeDirPath(f"./plotsPhotoProdPiPiUnpol.maxL_{self.maxL}")
    self.outFileNamePrefix = "norm" if self.normalizeMoments else "unnorm"

CFG = AnalysisConfig()


def calculateAllMoments(cfg: AnalysisConfig) -> None:
  """Performs the moment analysis for the given configuration"""
  timer.start("Total execution time")

  nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

  # setup MomentCalculators for all mass bins
  momentIndices = MomentIndices(cfg.maxL)
  momentCalculators = MomentCalculatorsKinematicBinning([])
  assert len(cfg.massBinning) > 0, f"Need at least one mass bin, but found {len(cfg.massBinning)}"
  with timer.timeThis(f"Time to load data and setup MomentCalculators for {len(cfg.massBinning)} bins"):
    print(f"Loading real data from tree '{cfg.treeName}' in file '{cfg.dataFileName}'")
    data = ROOT.RDataFrame(cfg.treeName, cfg.dataFileName)
    print(f"Loading accepted phase-space data from tree '{cfg.treeName}' in file '{cfg.psAccFileName}'")
    dataPsAcc = ROOT.RDataFrame(cfg.treeName, cfg.psAccFileName)
    if cfg.limitNmbPsAccEvents > 0:
      dataPsAcc = dataPsAcc.Range(cfg.limitNmbPsAccEvents)  #!Caution! .Range switches to single-threaded mode
    print(f"Loading generated phase-space data from tree '{cfg.treeName}' in file '{cfg.psGenFileName}'")
    dataPsGen = ROOT.RDataFrame(cfg.treeName, cfg.psGenFileName)
    for massBinIndex, massBinCenter in enumerate(cfg.massBinning):
      massBinRange = cfg.massBinning.binValueRange(massBinIndex)
      print(f"Preparing {cfg.binVarMass.name} bin [{massBinIndex} of {len(cfg.massBinning)}] at {massBinCenter} {cfg.binVarMass.unit} with range {massBinRange} {cfg.binVarMass.unit}")
      # load data for mass bin
      binMassRangeFilter = f"(({massBinRange[0]} < {cfg.binVarMass.name}) && ({cfg.binVarMass.name} < {massBinRange[1]}))"
      print(f"Applying filter '{binMassRangeFilter}' to select kinematic bin")
      dataInBin = data.Filter(binMassRangeFilter)
      print(f"Loaded {dataInBin.Count().GetValue()} data events; {dataInBin.Sum('eventWeight').GetValue()} background subtracted events")
      dataPsAccInBin = dataPsAcc.Filter(binMassRangeFilter)
      dataPsGenInBin = dataPsGen.Filter(binMassRangeFilter)
      nmbPsGenEvents = dataPsGenInBin.Count().GetValue()
      nmbPsAccEvents = dataPsAccInBin.Count().GetValue()
      print(f"Loaded phase-space events: number generated = {nmbPsGenEvents}; "
            f"number accepted = {nmbPsAccEvents}, "
            f" -> efficiency = {nmbPsAccEvents / nmbPsGenEvents:.3f}")
      # setup moment calculators for data
      dataSet = DataSet(
        data           = dataInBin,
        phaseSpaceData = dataPsAccInBin,
        nmbGenEvents   = nmbPsGenEvents,
        polarization   = None,
      )
      momentCalculators.append(
        MomentCalculator(
          indices              = momentIndices,
          dataSet              = dataSet,
          binCenters           = {cfg.binVarMass : massBinCenter},
          integralFileBaseName = f"{cfg.outFileDirName}/integralMatrix",
        )
      )

  # calculate and plot integral matrix for all mass bins
  with timer.timeThis(f"Time to calculate integral matrices for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads"):
    print(f"Calculating acceptance integral matrices for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads")
    momentCalculators.calculateIntegralMatrices(forceCalculation = True)
    print(f"Acceptance integral matrix for first bin at {cfg.massBinning[0]} {cfg.binVarMass.unit}:\n{momentCalculators[0].integralMatrix}")
    eigenVals, _ = momentCalculators[0].integralMatrix.eigenDecomp
    print(f"Sorted eigenvalues of acceptance integral matrix for first bin at {cfg.massBinning[0]} {cfg.binVarMass.unit}:\n{np.sort(eigenVals)}")

  momentResultsFileBaseName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments"
  if cfg.calcAccPsMoments:
    # calculate moments of accepted phase-space data
    with timer.timeThis(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating moments of phase-space MC data for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads")
      momentCalculators.calculateMoments(dataSource = MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE, normalize = cfg.normalizeMoments)
      momentCalculators.momentResultsMeas.save(f"{momentResultsFileBaseName}_accPs_meas.pkl")
      momentCalculators.momentResultsPhys.save(f"{momentResultsFileBaseName}_accPs_phys.pkl")

  # calculate moments of real data and write them to files
  #TODO calculate normalized and unnormalized moments
  with timer.timeThis(f"Time to calculate moments of real data for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads"):
    print(f"Calculating moments of real data for {len(momentCalculators)} bins using {nmbOpenMpThreads} OpenMP threads")
    momentCalculators.calculateMoments(normalize = cfg.normalizeMoments, nmbBootstrapSamples = cfg.nmbBootstrapSamples)
    momentCalculators.momentResultsMeas.save(f"{momentResultsFileBaseName}_meas.pkl")
    momentCalculators.momentResultsPhys.save(f"{momentResultsFileBaseName}_phys.pkl")

  timer.stop("Total execution time")
  print(timer.summary)


if __name__ == "__main__":
  logFileName = f"{CFG.outFileDirName}/calcMomentsPhotoProdPiPiUnpol.log"
  print(f"Writing output to log file '{logFileName}'")
  with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
    Utilities.printGitInfo()
    timer = Utilities.Timer()
    ROOT.gROOT.SetBatch(True)
    setupPlotStyle()
    threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
    print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
    with threadController.limit(limits = 4):
      print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")

      calculateAllMoments(CFG)
