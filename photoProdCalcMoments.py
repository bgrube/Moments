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
  # setup MomentCalculators for all data samples and mass bins
  momentCalculators: dict[str | None, MomentCalculatorsKinematicBinning] = {  # dict: key = data sample label
    None : MomentCalculatorsKinematicBinning([]),
  } if cfg.method == AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_NEG_WEIGHTS else {
    "Sig" : MomentCalculatorsKinematicBinning([]),
    "Bkg" : MomentCalculatorsKinematicBinning([]),
  }
  assert len(cfg.massBinning) > 0, f"Need at least one mass bin, but found {len(cfg.massBinning)}"
  with timer.timeThis(f"Time to load data and setup MomentCalculators for {len(cfg.massBinning)} bins"):
    dataSamples: dict[str | None, ROOT.RDataFrame] = {  # dict: key = data sample label
      None : cfg.loadData(AnalysisConfig.DataType.REAL_DATA),
    } if cfg.method == AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_NEG_WEIGHTS else {
      "Sig" : cfg.loadData(AnalysisConfig.DataType.REAL_DATA_SIGNAL),    # select events in signal region (mixture of signal and background)
      "Bkg" : cfg.loadData(AnalysisConfig.DataType.REAL_DATA_SIDEBAND),  # select events in sideband regions (ideally, pure background)
    }
    #TODO: implement event range limit
    # if limitToDataEntryRange is not None:
    #   print(f"Limiting analysis to entry range [{limitToDataEntryRange[0]}, {limitToDataEntryRange[1]}) of real data")
    #   data = data.Range(*limitToDataEntryRange)
    # use same MC events for all real-data samples
    dataPsAcc = cfg.loadData(AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE)
    if cfg.limitNmbPsAccEvents > 0 and dataPsAcc is not None:
      dataPsAcc = dataPsAcc.Range(cfg.limitNmbPsAccEvents)  #!Caution! Range() switches to single-threaded mode
    dataPsGen = cfg.loadData(AnalysisConfig.DataType.GENERATED_PHASE_SPACE)
    for labelDataSample, dataSample in dataSamples.items():
      print(f"Loaded {dataSample.Count().GetValue()} real-data events" + (f" of type '{labelDataSample}'" if labelDataSample is not None else ""))
      for massBinIndex, massBinCenter in enumerate(cfg.massBinning):
        massBinRange = cfg.massBinning.binValueRange(massBinIndex)
        print(f"Preparing {cfg.binVarMass.name} bin [{massBinIndex + 1} of {len(cfg.massBinning)}] at {massBinCenter} {cfg.binVarMass.unit} with range {massBinRange} {cfg.binVarMass.unit}")
        massBinFilter = cfg.massBinning.binFilter(massBinIndex)
        print(f"Applying filter '{massBinFilter}' to select kinematic bin")
        dataInBin = dataSample.Filter(massBinFilter)
        print(f"Loaded {dataInBin.Count().GetValue()} data events; {dataInBin.Sum('eventWeight').GetValue()} after applying event weights")
        dataPsAccInBin = None if dataPsAcc is None else dataPsAcc.Filter(massBinFilter)
        nmbPsAccEvents = None if dataPsAcc is None else dataPsAccInBin.Count().GetValue()
        nmbPsGenEvents = None
        if nmbPsAccEvents is not None:
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
          nmbGenEvents   = nmbPsGenEvents or nmbPsAccEvents or 0,
          polarization   = cfg.polarization,
        )
        momentCalculators[labelDataSample].append(
          MomentCalculator(
            indices              = MomentIndices(maxL = cfg.maxL, polarized = (cfg.polarization is not None)),
            dataSet              = dataSet,
            binCenters           = {cfg.binVarMass : massBinCenter},
            integralFileBaseName = f"{cfg.outFileDirName}/integralMatrix",
            flipSignOfWeights    = (labelDataSample == "Bkg"),  # flip sign of weights for background events
          )
        )

  if cfg.method == AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_NEG_WEIGHTS or AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_MOMENTS:
    # calculate integral matrix for all mass bins in all data sets
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()
    # since all data sets use identical MC events, calculate integral matrix only for the first real-data sample
    labelFirstSample, momentCalculatorsFirstSample = next(iter(momentCalculators.items()))  # get first entry in momentCalculators
    with timer.timeThis(f"Time to calculate integral matrices for {len(momentCalculatorsFirstSample)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating acceptance integral matrices for {len(momentCalculatorsFirstSample)} bins using {nmbOpenMpThreads} OpenMP threads")
      momentCalculatorsFirstSample.calculateIntegralMatrices(forceCalculation = True)
      print(f"Acceptance integral matrix for first bin at {cfg.massBinning[0]} {cfg.binVarMass.unit}:\n{momentCalculatorsFirstSample[0].integralMatrix}")
      eigenVals, _ = momentCalculatorsFirstSample[0].integralMatrix.eigenDecomp
      print(f"Sorted eigenvalues of acceptance integral matrix for first bin at {cfg.massBinning[0]} {cfg.binVarMass.unit}:\n{np.sort(eigenVals)}")
    # assign integral matrices to `MomentCalculators` for other data samples
    labelsOtherSamples = (label for label in momentCalculators.keys() if label is not labelFirstSample)
    for labelDataSample in labelsOtherSamples:
      for massBinIndex in range(len(cfg.massBinning)):
        momentCalculators[labelDataSample][massBinIndex]._integralMatrix = momentCalculatorsFirstSample[massBinIndex].integralMatrix

  momentResultsFileBaseName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments"
  if cfg.calcAccPsMoments:
    # calculate moments of accepted phase-space data
    if cfg.method == AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_NEG_WEIGHTS or AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_MOMENTS:
      with timer.timeThis(f"Time to calculate moments of phase-space MC data using the linear-algebra method with {nmbOpenMpThreads} OpenMP threads"):
        # since all data samples use identical MC events, calculate moments only for first accepted phase-space sample
        momentCalculatorsFirstSample = next(iter(momentCalculators.values()))  # get first entry in momentCalculators
        print(f"Calculating moments of phase-space MC data for {len(momentCalculatorsFirstSample)} bins using {nmbOpenMpThreads} OpenMP threads")
        momentCalculatorsFirstSample.calculateMoments(dataSourceType = MomentCalculator.DataSourceType.ACCEPTED_PHASE_SPACE, normalize = cfg.normalizeMoments)
        momentCalculatorsFirstSample.momentResultsMeas.savePickle(f"{momentResultsFileBaseName}_accPs_meas.pkl")
        momentCalculatorsFirstSample.momentResultsPhys.savePickle(f"{momentResultsFileBaseName}_accPs_phys.pkl")
    else:
      #TODO calculate also using FIT method
      raise NotImplementedError(f"Calculation of accepted phase-space moments using method {cfg.method} is not implemented yet")

  if cfg.method == AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_NEG_WEIGHTS:
    #TODO calculate normalized and unnormalized moments
    with timer.timeThis(f"Time to calculate moments of real data for {len(momentCalculators[None])} bins using the linear-algebra method with {nmbOpenMpThreads} OpenMP threads and subtracting background using negative weights"):
      print(f"Calculating moments of real data for {len(momentCalculators[None])} bins using the linear-algebra method with {nmbOpenMpThreads} OpenMP threads and subtracting background using negative weights")
      momentCalculators[None].calculateMoments(normalize = cfg.normalizeMoments, nmbBootstrapSamples = cfg.nmbBootstrapSamples)
      momentCalculators[None].momentResultsMeas.savePickle(f"{momentResultsFileBaseName}_meas.pkl")
      momentCalculators[None].momentResultsPhys.savePickle(f"{momentResultsFileBaseName}_phys.pkl")

  elif cfg.method == AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_MOMENTS:
    for labelDataSample, momentCalculatorsDataSample in momentCalculators.items():
      with timer.timeThis(f"Time to calculate moments of '{labelDataSample}' real data with {len(momentCalculatorsDataSample)} bins using the linear-algebra method with {nmbOpenMpThreads} OpenMP threads and subtracting background at moment level"):
        print(f"Calculating moments of '{labelDataSample}' real data with {len(momentCalculatorsDataSample)} bins using the linear-algebra method with {nmbOpenMpThreads} OpenMP threads and subtracting background at moment level")
        momentCalculatorsDataSample.calculateMoments(normalize = cfg.normalizeMoments, nmbBootstrapSamples = cfg.nmbBootstrapSamples)
        momentCalculatorsDataSample.momentResultsMeas.savePickle(f"{momentResultsFileBaseName}_meas.pkl")
        momentCalculatorsDataSample.momentResultsPhys.savePickle(f"{momentResultsFileBaseName}_phys.pkl")
    # perform background subtraction at moment level
    #!NOTE! This gives wrong estimates for the uncertainty (and covariances) of
    # H_0(0, 0) because the number of signal and background events are treated
    # as being constant. The uncertainty estimates for the other moments seem
    # to be unaffected.
    (momentCalculators["Sig"].momentResultsMeas - momentCalculators["Bkg"].momentResultsMeas).savePickle(f"{momentResultsFileBaseName}_meas.pkl")
    (momentCalculators["Sig"].momentResultsPhys - momentCalculators["Bkg"].momentResultsPhys).savePickle(f"{momentResultsFileBaseName}_phys.pkl")

  elif cfg.method == AnalysisConfig.MethodType.MAX_LIKELIHOOD_FIT:
    nmbFitAttempts          = 100
    nmbParallelFitProcesses = 100
    randomSeed              = 123456789
    for labelDataSample, momentCalculatorsDataSample in momentCalculators.items():
      with timer.timeThis(f"Time to fit moments to '{labelDataSample}' real data with {len(momentCalculatorsDataSample)} bins running {nmbFitAttempts} fit attempts in {nmbParallelFitProcesses} processes and subtracting background at moment level"):
        print(f"Fitting moments to '{labelDataSample}' real data with {len(momentCalculatorsDataSample)} bins running {nmbFitAttempts} fit attempts in {nmbParallelFitProcesses} processes and subtracting background at moment level")
        momentCalculatorsDataSample.fitMomentsMultipleAttempts(
          nmbFitAttempts          = nmbFitAttempts,
          nmbParallelFitProcesses = nmbParallelFitProcesses,
          randomSeed              = randomSeed,
        )
    # perform background subtraction at moment level
    (momentCalculators["Sig"].momentResultsPhys - momentCalculators["Bkg"].momentResultsPhys).savePickle(f"{momentResultsFileBaseName}_phys.pkl")

  else:
    raise ValueError(f"Unknown method {cfg.method}")


if __name__ == "__main__":
  cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarizedK- K_S Delta++ data
  # cfg = deepcopy(CFG_NIZAR)  # perform analysis of Nizar's polarized eta pi0 data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPP)  # perform analysis of unpolarized pi+ p data
  # cfg.method = AnalysisConfig.MethodType.LIN_ALG_BG_SUBTR_MOMENTS  # subtract background at moment level
  # cfg.method = AnalysisConfig.MethodType.MAX_LIKELIHOOD_FIT  # use FIT method for moment calculation

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
