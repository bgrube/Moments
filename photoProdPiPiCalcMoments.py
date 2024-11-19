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
import pandas as pd
import threadpoolctl

import ROOT
from wurlitzer import pipes, STDOUT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
  MomentResult,
  MomentResultsKinematicBinning,
  QnWaveIndex,
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
  dataFileName:             str                      = f"./dataPhotoProdPiPiUnpol/data_flat.root"
  psAccFileName:            str                      = f"./dataPhotoProdPiPiUnpol/phaseSpace_acc_flat.root"
  psGenFileName:            str                      = f"./dataPhotoProdPiPiUnpol/phaseSpace_gen_flat.root"
  polarization:             float | None             = None  # unpolarized data
  _maxL:                    int                      = 8
  # outFileDirBaseName:       str                      = "./plotsPhotoProdPiPiUnpol"
  outFileDirBaseName:       str                      = "./plotsPhotoProdPiPiUnpolPwa"
  outFileDirName:           str                      = field(init = False)
  outFileNamePrefix:        str                      = field(init = False)
  normalizeMoments:         bool                     = False
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

  def init(self) -> None:
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
    self.init()

CFG_UNPOLARIZED = AnalysisConfig()
CFG_POLARIZED = AnalysisConfig(
  # dataFileName       = f"./dataPhotoProdPiPiPol/data_flat.root",
  dataFileName       = f"./dataPhotoProdPiPiPol/data_flat_downsampled_0.1.root",
  psAccFileName      = f"./dataPhotoProdPiPiPol/phaseSpace_acc_flat.root",
  psGenFileName      = f"./dataPhotoProdPiPiPol/phaseSpace_gen_flat.root",
  polarization       = 0.0,  # read polarization value from input data
  _maxL              = 6,
  # outFileDirBaseName = "./plotsPhotoProdPiPiPol",
  outFileDirBaseName = "./plotsPhotoProdPiPiPol_downsampled_0.1",
  massBinning        = HistAxisBinning(nmbBins = 50, minVal = 0.28, maxVal = 2.28),  # binning used in PWA of polarized data
)


def readMomentResultsPwa(
  dataFileName:          str,
  maxL:                  int,  # maximum L quantum number of moments
  waves:                 list[tuple[str, QnWaveIndex]],  # wave labels and quantum numbers
  binVarMass:            KinematicBinningVariable,       # binning variable for mass bins
  momentResultsFileName: str | None = None,   # if not `None`, moments are written to this file; if file already exists moments are read from this file instead of recalculating them
  overwriteExistingFile: bool       = False,  # False: read values from existing file; True: recalculate moments and write new file
) -> MomentResultsKinematicBinning:
  """Reads the partial-amplitude values from the PWA fit and calculates the corresponding moments"""
  if momentResultsFileName and not overwriteExistingFile:
    try:
      print(f"Reading PWA moment values from file '{momentResultsFileName}'")
      return MomentResultsKinematicBinning.load(momentResultsFileName)
    except FileNotFoundError:
      print(f"Could not find file '{momentResultsFileName}'. Calculating PWA moments.")
  print(f"Reading partial-wave amplitude values from file '{dataFileName}'")
  waveLabels = [wave[0] for wave in waves]
  amplitudesDf = pd.read_csv(
    dataFileName,
    sep   = r"\s+",  # values are whitespace separated
    names = ["mass", ] + waveLabels,
  )
  # converts columns to correct types
  def strToComplex(s: str) -> complex:
    """Converts string of form '(float,float)' to complex"""
    real, imag = s.strip("()").split(",")
    return complex(float(real), float(imag))
  for wave in waves:
    amplitudesDf[wave[0]] = amplitudesDf[wave[0]].apply(strToComplex)
  # convert dataframe to MomentResultsKinematicBinning
  momentResults: list[MomentResult] = []
  for amplitudesRow in amplitudesDf.to_dict(orient = "records"):  # iterate over list of dictionaries
    massBinCenter = amplitudesRow["mass"]
    print(f"Reading partial-wave amplitudes for mass bin at {massBinCenter} GeV")
    amplitudeSet = AmplitudeSet(
      amps      = [AmplitudeValue(qn = wave[1], val = amplitudesRow[wave[0]]) for wave in waves],
      tolerance = 1e-8,
    )
    momentResults.append(
      amplitudeSet.photoProdMomentSet(
        maxL                = maxL,
        normalize           = False,
        printMomentFormulas = False,
        binCenters          = {binVarMass: massBinCenter},
      )
    )
  momentResultsPwa = MomentResultsKinematicBinning(momentResults)
  if momentResultsFileName:
    print(f"Writing moments to file '{momentResultsFileName}'")
    momentResultsPwa.save(momentResultsFileName)
  return momentResultsPwa


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
  # cfg = deepcopy(CFG_UNPOLARIZED)
  cfg = deepcopy(CFG_POLARIZED)

  # for maxL in (2, 4, 5, 8, 10, 12, 20):
  for maxL in (8, ):
    print(f"Performing moment analysis for L_max = {maxL}")
    cfg.maxL = maxL
    thisSourceFileName = os.path.basename(__file__)
    logFileName = f"{cfg.outFileDirName}/{os.path.splitext(thisSourceFileName)[0]}.log"
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

        if True:
          with timer.timeThis(f"Time to load moments from partial-wave analysis"):
            # # unpolarized data
            # # pwaAmplitudesFileName = "./dataPhotoProdPiPiUnpol/PWA_S_P_D/amplitudes_range_tbin.txt"
            # pwaAmplitudesFileName = "./dataPhotoProdPiPiUnpol/PWA_S_P_D_F/amplitudes_new_SPDF.txt"
            # waves: list[tuple[str, QnWaveIndex]] = [  # order must match columns in file with partial-wave amplitudes
            #   ("S_0",  QnWaveIndex(refl = None, l = 0, m =  0)),
            #   # P-waves
            #   ("P_0",  QnWaveIndex(refl = None, l = 1, m =  0)),
            #   ("P_+1", QnWaveIndex(refl = None, l = 1, m = +1)),
            #   ("P_-1", QnWaveIndex(refl = None, l = 1, m = -1)),
            #   # D-waves
            #   ("D_0",  QnWaveIndex(refl = None, l = 2, m =  0)),
            #   ("D_+1", QnWaveIndex(refl = None, l = 2, m = +1)),
            #   ("D_+2", QnWaveIndex(refl = None, l = 2, m = +2)),
            #   ("D_-1", QnWaveIndex(refl = None, l = 2, m = -1)),
            #   ("D_-2", QnWaveIndex(refl = None, l = 2, m = -2)),
            #   # # F-waves
            #   # ("F_0",  QnWaveIndex(refl = None, l = 3, m =  0)),
            #   # ("F_+1", QnWaveIndex(refl = None, l = 3, m = +1)),
            #   # ("F_+2", QnWaveIndex(refl = None, l = 3, m = +2)),
            #   # ("F_+3", QnWaveIndex(refl = None, l = 3, m = +3)),
            #   # ("F_-1", QnWaveIndex(refl = None, l = 3, m = -1)),
            #   # ("F_-2", QnWaveIndex(refl = None, l = 3, m = -2)),
            #   # ("F_-3", QnWaveIndex(refl = None, l = 3, m = -3)),
            # ]
            # polarized data
            pwaAmplitudesFileName = "./dataPhotoProdPiPiPol/PWA_S_P_D/amplitudes_SPD.txt"
            waves: list[tuple[str, QnWaveIndex]] = [  # order must match columns in file with partial-wave amplitudes
              # S-waves
              ("S_0+",  QnWaveIndex(refl = +1, l = 0, m =  0)),
              ("S_0-",  QnWaveIndex(refl = -1, l = 0, m =  0)),
              # P-waves
              ("P_0+",  QnWaveIndex(refl = +1, l = 1, m =  0)),
              ("P_+1+", QnWaveIndex(refl = +1, l = 1, m = +1)),
              ("P_-1+", QnWaveIndex(refl = +1, l = 1, m = -1)),
              ("P_0-",  QnWaveIndex(refl = -1, l = 1, m =  0)),
              ("P_+1-", QnWaveIndex(refl = -1, l = 1, m = +1)),
              ("P_-1-", QnWaveIndex(refl = -1, l = 1, m = -1)),
              # D-waves
              ("D_0+",  QnWaveIndex(refl = +1, l = 2, m =  0)),
              ("D_+1+", QnWaveIndex(refl = +1, l = 2, m = +1)),
              ("D_+2+", QnWaveIndex(refl = +1, l = 2, m = +2)),
              ("D_-1+", QnWaveIndex(refl = +1, l = 2, m = -1)),
              ("D_-2+", QnWaveIndex(refl = +1, l = 2, m = -2)),
              ("D_0-",  QnWaveIndex(refl = -1, l = 2, m =  0)),
              ("D_+1-", QnWaveIndex(refl = -1, l = 2, m = +1)),
              ("D_+2-", QnWaveIndex(refl = -1, l = 2, m = +2)),
              ("D_-1-", QnWaveIndex(refl = -1, l = 2, m = -1)),
              ("D_-2-", QnWaveIndex(refl = -1, l = 2, m = -2)),
            ]
            readMomentResultsPwa(
              dataFileName          = pwaAmplitudesFileName,
              maxL                  = cfg.maxL,
              waves                 = waves,
              binVarMass            = cfg.binVarMass,
              momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_pwa_SPD.pkl",
              # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_pwa_SPDF.pkl",
              # overwriteExistingFile = True,
              overwriteExistingFile = False,
            )

        calculateAllMoments(cfg, timer)

        timer.stop("Total execution time")
        print(timer.summary)
