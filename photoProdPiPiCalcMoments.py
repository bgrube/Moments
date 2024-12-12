#!/usr/bin/env python3
"""
This module performs the moment analysis of unpolarized and polarized
pi+ pi- photoproduction data. The calculated moments are written to
files to be read by the plotting script `photoProdPiPiPlotMoments.py`.

Usage: Run this module as a script to perform the moment calculations
and to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
from dataclasses import (
  dataclass,
  field,
)
import functools
import numpy as np
import os
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
  """Stores configuration parameters for the moment analysis; defaults are for unpolarized production"""
  treeName:                 str                      = "PiPi"
  dataFileName:             str                      = "./dataPhotoProdPiPiUnpol/data_flat.PiPi.root"
  psAccFileName:            str                      = "./dataPhotoProdPiPiUnpol/phaseSpace_acc_flat.PiPi.root"
  psGenFileName:            str | None               = "./dataPhotoProdPiPiUnpol/phaseSpace_gen_flat.PiPi.root"
  polarization:             float | None             = None  # unpolarized data
  _maxL:                    int                      = 8
  # outFileDirBaseName:       str                      = "./plotsPhotoProdPiPiUnpol"
  outFileDirBaseName:       str                      = "./plotsPhotoProdPiPiUnpolPwa"
  outFileDirName:           str                      = field(init = False)
  outFileNamePrefix:        str                      = field(init = False)
  # _normalizeMoments:        bool                     = True
  _normalizeMoments:        bool                     = False
  nmbBootstrapSamples:      int                      = 0
  # nmbBootstrapSamples:      int                      = 10000
  # plotAngularDistributions: bool                     = True
  plotAngularDistributions: bool                     = False
  # plotAccIntegralMatrices:  bool                     = True
  plotAccIntegralMatrices:  bool                     = False
  # calcAccPsMoments:         bool                     = True
  calcAccPsMoments:         bool                     = False
  # plotAccPsMoments:         bool                     = True
  plotAccPsMoments:         bool                     = False
  # plotMeasuredMoments:      bool                     = True
  plotMeasuredMoments:      bool                     = False
  # plotCovarianceMatrices:   bool                     = True
  plotCovarianceMatrices:   bool                     = False
  limitNmbPsAccEvents:      int                      = 0
  # limitNmbPsAccEvents:      int                      = 100000
  binVarMass:               KinematicBinningVariable = field(default_factory=lambda: KinematicBinningVariable(
    name  = "mass",
    label = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}",
    unit = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ))
  # massBinning:              HistAxisBinning          = field(default_factory=lambda: HistAxisBinning(nmbBins = 100, minVal = 0.4, maxVal = 1.4))  # same binning as used by CLAS
  # massBinning:              HistAxisBinning          = field(default_factory=lambda: HistAxisBinning(nmbBins = 1, minVal = 1.25, maxVal = 1.29))  # f_2(1270) region
  massBinning:              HistAxisBinning          = field(default_factory=lambda: HistAxisBinning(nmbBins = 56, minVal = 0.28, maxVal = 1.40))  # binning used in PWA of unpolarized data

  def __post_init__(self) -> None:
    """Creates output directory and initializes member variables"""
    self.outFileDirName    = Utilities.makeDirPath(f"{self.outFileDirBaseName}.maxL_{self.maxL}")
    self.outFileNamePrefix = "norm" if self.normalizeMoments else "unnorm"
    self.massBinning.var   = self.binVarMass

  @property
  def maxL(self) -> int:
    return self._maxL

  @maxL.setter
  def maxL(
    self,
    value: int
  ) -> None:
    assert value > 0, f"maxL must be > 0, but is {value}"
    self._maxL = value
    self.__post_init__()

  @property
  def normalizeMoments(self) -> bool:
    return self._normalizeMoments

  @normalizeMoments.setter
  def normalizeMoments(
    self,
    value: bool
  ) -> None:
    self._normalizeMoments = value
    self.__post_init__()


# configuration for unpolarized pi+ pi- data
CFG_UNPOLARIZED_PIPI = AnalysisConfig()
# configuration for polarized pi+ pi- data
CFG_POLARIZED_PIPI = AnalysisConfig(
  dataFileName       = "./dataPhotoProdPiPiPol/data_flat.root",
  # dataFileName       = "./dataPhotoProdPiPiPol/data_flat_downsampled_0.1.root",
  psAccFileName      = "./dataPhotoProdPiPiPol/phaseSpace_acc_flat.root",
  psGenFileName      = "./dataPhotoProdPiPiPol/phaseSpace_gen_flat.root",
  polarization       = 0.3519,
  _maxL              = 4,
  outFileDirBaseName = "./plotsPhotoProdPiPiPol",
  # outFileDirBaseName = "./plotsPhotoProdPiPiPol_downsampled_0.1",
  massBinning        = HistAxisBinning(nmbBins = 50, minVal = 0.28, maxVal = 2.28),  # binning used in PWA of polarized data
)


def calculateAllMoments(
  cfg:   AnalysisConfig,
  timer: Utilities.Timer = Utilities.Timer(),
) -> None:
  """Performs the moment analysis for the given configuration"""
  # setup MomentCalculators for all mass bins
  momentCalculators = MomentCalculatorsKinematicBinning([])
  assert len(cfg.massBinning) > 0, f"Need at least one mass bin, but found {len(cfg.massBinning)}"
  with timer.timeThis(f"Time to load data and setup MomentCalculators for {len(cfg.massBinning)} bins"):
    print(f"Loading real data from tree '{cfg.treeName}' in file '{cfg.dataFileName}'")
    data = ROOT.RDataFrame(cfg.treeName, cfg.dataFileName)
    print(f"Loading accepted phase-space data from tree '{cfg.treeName}' in file '{cfg.psAccFileName}'")
    dataPsAcc = ROOT.RDataFrame(cfg.treeName, cfg.psAccFileName)
    if cfg.limitNmbPsAccEvents > 0:
      dataPsAcc = dataPsAcc.Range(cfg.limitNmbPsAccEvents)  #!Caution! Range() switches to single-threaded mode
    print(f"Loading generated phase-space data from tree '{cfg.treeName}' in file '{cfg.psGenFileName}'")
    dataPsGen = ROOT.RDataFrame(cfg.treeName, cfg.psGenFileName)
    for massBinIndex, massBinCenter in enumerate(cfg.massBinning):
      massBinRange = cfg.massBinning.binValueRange(massBinIndex)
      print(f"Preparing {cfg.binVarMass.name} bin [{massBinIndex + 1} of {len(cfg.massBinning)}] at {massBinCenter} {cfg.binVarMass.unit} with range {massBinRange} {cfg.binVarMass.unit}")
      # load data for mass bin
      massBinFilter = cfg.massBinning.binFilter(massBinIndex)
      print(f"Applying filter '{massBinFilter}' to select kinematic bin")
      dataInBin = data.Filter(massBinFilter)
      print(f"Loaded {dataInBin.Count().GetValue()} data events; {dataInBin.Sum('eventWeight').GetValue()} background subtracted events")
      dataPsAccInBin = dataPsAcc.Filter(massBinFilter)
      dataPsGenInBin = dataPsGen.Filter(massBinFilter)
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

  # calculate and plot integral matrix for all mass bins
  nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()
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


if __name__ == "__main__":
  cfg = deepcopy(CFG_UNPOLARIZED_PIPI)  # perform unpolarized analysis
  # cfg = deepcopy(CFG_POLARIZED_PIPI)    # perform polarized analysis

  # for maxL in (2, 4, 5, 8, 10, 12, 20):
  for maxL in (8, ):
    print(f"Performing moment analysis for L_max = {maxL}")
    cfg.maxL = maxL
    thisSourceFileName = os.path.basename(__file__)
    logFileName = f"{cfg.outFileDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
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

        timer.start("Total execution time")

        calculateAllMoments(cfg, timer)

        timer.stop("Total execution time")
        print(timer.summary)
