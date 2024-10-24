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

from MomentCalculator import (
  binLabel,
  binTitle,
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
  MomentResult,
  QnMomentIndex,
)
from PlottingUtilities import (
  HistAxisBinning,
  MomentValueAndTruth,
  plotAngularDistr,
  plotComplexMatrix,
  plotMoments,
  plotMomentsInBin,
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
  normalizeMoments:         bool                     = False
  nmbBootstrapSamples:      int                      = 0
  # nmbBootstrapSamples:      int                      = 10000
  # plotAngularDistributions: bool                     = True
  plotAngularDistributions: bool                     = False
  # plotAccIntegralMatrices:  bool                     = True
  plotAccIntegralMatrices:  bool                     = False
  calcAccPsMoments:         bool                     = True
  # calcAccPsMoments:         bool                     = False
  limitNmbPsAccEvents:      int                      = 0
  # limitNmbPsAccEvents:      int                      = 100000
  binVarMass:               KinematicBinningVariable = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}", unit = "GeV/#it{c}^{2}", nmbDigits = 3)
  massBinning:              HistAxisBinning          = HistAxisBinning(nmbBins = 100, minVal = 0.4, maxVal = 1.4, _var = binVarMass)  # same binning as used by CLAS
  # massBinning:              HistAxisBinning          = HistAxisBinning(nmbBins = 1, minVal = 1.25, maxVal = 1.29, _var = binVarMass)  # f_2(1270) region

  def __post_init__(self) -> None:
    """Creates output directory and sets `outFileDirName`"""
    self.outFileDirName = Utilities.makeDirPath(f"./plotsPhotoProdPiPiUnpol.maxL_{self.maxL}")

CFG = AnalysisConfig()


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
  with threadController.limit(limits = 4):
    print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")
    timer.start("Total execution time")

    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()
    #TODO move into config object
    namePrefix = "norm" if CFG.normalizeMoments else "unnorm"

    # setup MomentCalculators for all mass bins
    momentIndices = MomentIndices(CFG.maxL)
    momentsInBins:  list[MomentCalculator] = []
    nmbPsGenEvents: list[int]              = []
    assert len(CFG.massBinning) > 0, f"Need at least one mass bin, but found {len(CFG.massBinning)}"
    with timer.timeThis(f"Time to load data and setup MomentCalculators for {len(CFG.massBinning)} bins"):
      for massBinIndex, massBinCenter in enumerate(CFG.massBinning):
        massBinRange = CFG.massBinning.binValueRange(massBinIndex)
        print(f"Preparing {CFG.binVarMass.name} bin [{massBinIndex} of {len(CFG.massBinning)}] at {massBinCenter} {CFG.binVarMass.unit} with range {massBinRange} {CFG.binVarMass.unit}")

        # load data for mass bin
        binMassRangeFilter = f"(({massBinRange[0]} < {CFG.binVarMass.name}) && ({CFG.binVarMass.name} < {massBinRange[1]}))"
        print(f"Loading real data from tree '{CFG.treeName}' in file '{CFG.dataFileName}' and applying filter {binMassRangeFilter}")
        dataInBin = ROOT.RDataFrame(CFG.treeName, CFG.dataFileName).Filter(binMassRangeFilter)
        print(f"Loaded {dataInBin.Count().GetValue()} data events; {dataInBin.Sum('eventWeight').GetValue()} background subtracted events")
        print(f"Loading accepted phase-space data from tree '{CFG.treeName}' in file '{CFG.psAccFileName}' and applying filter {binMassRangeFilter}")
        dataPsAccInBin = ROOT.RDataFrame(CFG.treeName, CFG.psAccFileName).Range(CFG.limitNmbPsAccEvents).Filter(binMassRangeFilter)
        print(f"Loading generated phase-space data from tree '{CFG.treeName}' in file '{CFG.psAccFileName}' and applying filter {binMassRangeFilter}")
        dataPsGenInBin = ROOT.RDataFrame(CFG.treeName, CFG.psGenFileName).Filter(binMassRangeFilter)
        nmbPsGenEvents.append(dataPsGenInBin.Count().GetValue())
        nmbPsAccEvents = dataPsAccInBin.Count().GetValue()
        print(f"Loaded phase-space events: number generated = {nmbPsGenEvents[-1]}; "
              f"number accepted = {nmbPsAccEvents}, "
              f" -> efficiency = {nmbPsAccEvents / nmbPsGenEvents[-1]:.3f}")

        # setup moment calculators for data
        dataSet = DataSet(
          data           = dataInBin,
          phaseSpaceData = dataPsAccInBin,
          nmbGenEvents   = nmbPsGenEvents[-1],
          polarization   = None,
        )
        momentsInBins.append(
          MomentCalculator(
            indices              = momentIndices,
            dataSet              = dataSet,
            binCenters           = {CFG.binVarMass : massBinCenter},
            integralFileBaseName = f"{CFG.outFileDirName}/integralMatrix",
          )
        )

        # plot angular distributions for mass bin
        if CFG.plotAngularDistributions:
          plotAngularDistr(
            dataPsAcc         = dataPsAccInBin,
            dataPsGen         = dataPsGenInBin,
            dataSignalAcc     = dataInBin,
            dataSignalGen     = None,
            pdfFileNamePrefix = f"{CFG.outFileDirName}/angDistr_{binLabel(momentsInBins[-1])}_"
          )
    moments = MomentCalculatorsKinematicBinning(momentsInBins)

    # calculate and plot integral matrix for all mass bins
    with timer.timeThis(f"Time to calculate integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating acceptance integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateIntegralMatrices(forceCalculation = True)
      print(f"Acceptance integral matrix for first bin at {CFG.massBinning[0]} {CFG.binVarMass.unit}:\n{moments[0].integralMatrix}")
      eigenVals, _ = moments[0].integralMatrix.eigenDecomp
      print(f"Sorted eigenvalues of acceptance integral matrix for first bin at {CFG.massBinning[0]} {CFG.binVarMass.unit}:\n{np.sort(eigenVals)}")
      # plot acceptance integral matrices for all kinematic bins
      if CFG.plotAccIntegralMatrices:
        for momentResultInBin in moments:
          label = binLabel(momentResultInBin)
          plotComplexMatrix(
            complexMatrix     = momentResultInBin.integralMatrix.matrixNormalized,
            pdfFileNamePrefix = f"{CFG.outFileDirName}/accMatrix_{label}_",
            axisTitles        = ("Physical Moment Index", "Measured Moment Index"),
            plotTitle         = f"{label}: "r"$\mathrm{\mathbf{I}}_\text{acc}$, ",
            zRangeAbs         = 1.2,
            zRangeImag        = 0.05,
          )
          plotComplexMatrix(
            complexMatrix     = momentResultInBin.integralMatrix.inverse,
            pdfFileNamePrefix = f"{CFG.outFileDirName}/accMatrixInv_{label}_",
            axisTitles        = ("Measured Moment Index", "Physical Moment Index"),
            plotTitle         = f"{label}: "r"$\mathrm{\mathbf{I}}_\text{acc}^{-1}$, ",
            zRangeAbs         = 5,
            zRangeImag        = 0.3,
          )

    if CFG.calcAccPsMoments:
      # calculate moments of accepted phase-space data
      with timer.timeThis(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads"):
        print(f"Calculating moments of phase-space MC data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
        moments.calculateMoments(dataSource = MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE, normalize = CFG.normalizeMoments)
        #TODO move into separate plotting script
        # plot kinematic dependences of all phase-space moments
        for qnIndex in momentIndices.qnIndices:
          HVals = tuple(MomentValueAndTruth(*momentsInBin.HMeas[qnIndex]) for momentsInBin in moments)
          plotMoments(
            HVals             = HVals,
            binning           = CFG.massBinning,
            normalizedMoments = CFG.normalizeMoments,
            momentLabel       = qnIndex.label,
            pdfFileNamePrefix = f"{CFG.outFileDirName}/{namePrefix}_{CFG.massBinning.var.name}_accPs_",
            histTitle         = qnIndex.title,
            plotLegend        = False,
          )
        # plot accepted phase-space moments in each kinematic bin
        for massBinIndex, momentResultInBin in enumerate(moments):
          label = binLabel(momentResultInBin)
          title = binTitle(momentResultInBin)
          print(f"Measured moments of accepted phase-space data for kinematic bin {title}:\n{momentResultInBin.HMeas}")
          print(f"Physical moments of accepted phase-space data for kinematic bin {title}:\n{momentResultInBin.HPhys}")
          # construct true moments for phase-space data
          HTruthPs = MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
          HTruthPs._valsFlatIndex[momentIndices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1 if CFG.normalizeMoments else nmbPsGenEvents[massBinIndex]  # ... except for H_0(0, 0)
          # set H_0^meas(0, 0) to 0 so that one can better see the other H_0^meas moments
          momentResultInBin.HMeas._valsFlatIndex[0] = 0
          # plot measured and physical moments; the latter should match the true moments exactly except for tiny numerical effects
          plotMomentsInBin(
            HData             = momentResultInBin.HMeas,
            normalizedMoments = CFG.normalizeMoments,
            HTruth            = None,
            pdfFileNamePrefix = f"{CFG.outFileDirName}/{namePrefix}_{label}_accPs_",
            plotLegend        = False,
          )
          plotMomentsInBin(
            HData             = momentResultInBin.HPhys,
            normalizedMoments = CFG.normalizeMoments,
            HTruth            = HTruthPs,
            pdfFileNamePrefix = f"{CFG.outFileDirName}/{namePrefix}_{label}_accPsCorr_"
          )

    # calculate moments of real data and write them to files
    #TODO calculate normalized and unnormalized moments
    momentResultsFileBaseName = f"{CFG.outFileDirName}/{namePrefix}_moments"
    with timer.timeThis(f"Time to calculate moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateMoments(normalize = CFG.normalizeMoments, nmbBootstrapSamples = CFG.nmbBootstrapSamples)
      moments.momentResultsMeas.save(f"{momentResultsFileBaseName}_meas.pkl")
      moments.momentResultsPhys.save(f"{momentResultsFileBaseName}_phys.pkl")

    timer.stop("Total execution time")
    print(timer.summary)
