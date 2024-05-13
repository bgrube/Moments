#!/usr/bin/env python3
# performs input-output study for eta pi0 Monte Carlo events

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=4

import functools
import numpy as np
import pandas as pd
import threadpoolctl
from typing import (
  Dict,
  List,
  Tuple,
)

import ROOT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  binLabel,
  binTitle,
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
  MomentResult,
  QnMomentIndex,
  QnWaveIndex,
)
from PlottingUtilities import (
  HistAxisBinning,
  MomentValueAndTruth,
  plotAngularDistr,
  plotComplexMatrix,
  plotMoments,
  plotMomentsBootstrapDiff1D,
  plotMomentsBootstrapDiffInBin,
  plotMomentsBootstrapDistributions1D,
  plotMomentsBootstrapDistributions2D,
  plotMomentsCovMatrices,
  plotMoments1D,
  plotMomentsInBin,
  plotPullsForMoment,
  plotPullParameters,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads basisFunctions.C
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def readPartialWaveAmplitudes(
  csvFileName:   str,
  massBinCenter: float,  # [GeV]
) -> List[AmplitudeValue]:
  """Reads partial-wave amplitudes values for given mass bin from CSV data"""
  df = pd.read_csv(csvFileName).astype({"mass": float})
  df = df.loc[np.isclose(df["mass"], massBinCenter)].drop(columns = ["mass"])
  print(f"Partial-wave amplitudes for mass bin at {massBinCenter} GeV:\n{df}")
  assert len(df) == 1, f"Expected exactly 1 row for mass-bin center {massBinCenter} GeV, but found {len(df)}"
  # Pandas cannot read-back complex values out of the box
  # there also seems to be no interest in fixing that; see <https://github.com/pandas-dev/pandas/issues/9379>
  # have to convert columns by hand
  ampSeries = df.astype('complex128').loc[df.index[0]] / 5.0  #TODO clarify why H_0(0, 0) is by a factor of 25 larger than number of generated events
  partialWaveAmplitudes = [
    # negative-reflectivity waves
    AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val = ampSeries["S0-" ]),  # S_0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val = ampSeries["D1--"]),  # D_-1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val = ampSeries["D0+-"]),  # D_0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val = ampSeries["D1+-"]),  # D_+1^-
    # positive-reflectivity waves
    AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val = ampSeries["S0+" ]),  # S_0^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val = ampSeries["D2-+"]),  # D_-2^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val = ampSeries["D2++"]),  # D_+2^+
  ]
  # print(f"!!! {partialWaveAmplitudes=}")
  return partialWaveAmplitudes


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
  with threadController.limit(limits = 3):
    print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of analysis
    outFileDirName       = Utilities.makeDirPath("./plotsMcPhotoProdEtaPi0")
    treeName             = "etaPi0"
    signalAccFileName    = "./dataMcPhotoProdEtaPi0/a0a2_signal_acc_flat.root"
    signalGenFileName    = "./dataMcPhotoProdEtaPi0/a0a2_signal_gen_flat.root"
    psAccFileName        = "./dataMcPhotoProdEtaPi0/a0a2_phaseSpace_acc_flat.root"
    psGenFileName        = "./dataMcPhotoProdEtaPi0/a0a2_phaseSpace_gen_flat.root"
    #!Note! partial-wave amplitudes for signal are defined in helicity frame
    # signalPWAmpsFileName = "./dataMcPhotoProdEtaPi0/a0a2_raw/a0a2_complex_amps.csv"
    signalPwAmpsFileName = "./dataMcPhotoProdEtaPi0/a0a2_bin_10_amps.csv"
    beamPolarization     = 1.0  #TODO read from tree
    # maxL                 = 1  # define maximum L quantum number of moments
    maxL                 = 5  # define maximum L quantum number of moments
    normalizeMoments     = False
    nmbBootstrapSamples  = 0
    # nmbBootstrapSamples  = 10000
    calcAccPsMoments     = True
    # calcAccPsMoments     = False
    # plotPulls            = True
    plotPulls            = False
    binVarMass           = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#eta#pi}^{0}}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    # massBinning          = HistAxisBinning(nmbBins = 28, minVal = 0.88, maxVal = 2.00, _var = binVarMass)
    # massBinning          = HistAxisBinning(nmbBins = 2, minVal = 1.28, maxVal = 1.36, _var = binVarMass)
    # massBinning          = HistAxisBinning(nmbBins = 1, minVal = 1.12, maxVal = 1.16, _var = binVarMass)
    # study with constant amplitude values taken form bin 10 at 1.30 GeV
    massBinning          = HistAxisBinning(nmbBins = 25, minVal = 0.92, maxVal = 1.92, _var = binVarMass)
    # massBinning          = HistAxisBinning(nmbBins = 1, minVal = 0.88 , maxVal = 0.92, _var = binVarMass)
    # massBinning          = HistAxisBinning(nmbBins = 1, minVal = 1.28 , maxVal = 1.32, _var = binVarMass)
    # massBinning          = HistAxisBinning(nmbBins = 1, minVal = 0.92 , maxVal = 1.92, _var = binVarMass)
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # load all signal and phase-space data
    print(f"Loading accepted signal data from tree '{treeName}' in file '{signalAccFileName}'")
    dataSignalAcc = ROOT.RDataFrame(treeName, signalAccFileName)
    print(f"Loading generated signal data from tree '{treeName}' in file '{signalGenFileName}'")
    dataSignalGen = ROOT.RDataFrame(treeName, signalGenFileName)
    print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}'")
    dataPsAcc = ROOT.RDataFrame(treeName, psAccFileName)
    print(f"Loading generated phase-space data from tree '{treeName}' in file '{psGenFileName}'")
    dataPsGen = ROOT.RDataFrame(treeName, psGenFileName)
    # plot total angular distributions
    plotAngularDistr(dataPsAcc, dataPsGen, dataSignalAcc, dataSignalGen, pdfFileNamePrefix = f"{outFileDirName}/angDistr_total_")

    # setup MomentCalculators for all mass bins
    momentIndices = MomentIndices(maxL)
    momentsInBins:      List[MomentCalculator] = []
    momentsInBinsTruth: List[MomentCalculator] = []
    nmbSignalGenEvents: List[int]              = []
    nmbPsGenEvents:     List[int]              = []
    assert len(massBinning) > 0, f"Need at least one mass bin, but found {len(massBinning)}"
    with timer.timeThis(f"Time to load data and setup MomentCalculators for {len(massBinning)} bins"):
      for massBinIndex, massBinCenter in enumerate(massBinning):
        massBinRange = massBinning.binValueRange(massBinIndex)
        print(f"Preparing {binVarMass.name} bin [{massBinIndex} of {len(massBinning)}] at {massBinCenter} {binVarMass.unit} with range {massBinRange} {binVarMass.unit}")

        # load data for mass bin
        binMassRangeFilter = f"(({massBinRange[0]} < {binVarMass.name}) && ({binVarMass.name} < {massBinRange[1]}))"
        print(f"Loading accepted signal data from tree '{treeName}' in file '{signalAccFileName}' and applying filter {binMassRangeFilter}")
        dataSignalAccInBin = ROOT.RDataFrame(treeName, signalAccFileName).Filter(binMassRangeFilter)
        nmbSignalAccEvents = dataSignalAccInBin.Count().GetValue()
        print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
        dataPsAccInBin = ROOT.RDataFrame(treeName, psAccFileName).Filter(binMassRangeFilter)
        nmbPsAccEvents = dataPsAccInBin.Count().GetValue()
        print(f"Loading generated signal data from tree '{treeName}' in file '{signalGenFileName}' and applying filter {binMassRangeFilter}")
        dataSignalGenInBin = ROOT.RDataFrame(treeName, signalGenFileName).Filter(binMassRangeFilter)
        nmbSignalGenEvents.append(dataSignalGenInBin.Count().GetValue())
        print(f"Loaded signal events: number generated = {nmbSignalGenEvents[-1]}, number accepted = {nmbSignalAccEvents}"
              f" -> efficiency = {nmbSignalAccEvents / nmbSignalGenEvents[-1]:.3f}")
        print(f"Loading generated phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
        dataPsGenInBin = ROOT.RDataFrame(treeName, psGenFileName).Filter(binMassRangeFilter)
        nmbPsGenEvents.append(dataPsGenInBin.Count().GetValue())
        print(f"Loaded phase-space events: number generated = {nmbPsGenEvents[-1]}, number accepted = {nmbPsAccEvents}"
              f" -> efficiency = {nmbPsAccEvents / nmbPsGenEvents[-1]:.3f}")

        # calculate true moments
        amplitudeSet = AmplitudeSet(amps = readPartialWaveAmplitudes(signalPwAmpsFileName, massBinCenter), tolerance = 1e-11)
        HTrue: MomentResult = amplitudeSet.photoProdMomentSet(maxL, printMoments = False, normalize = normalizeMoments)
        # scale true moments such that H_0(0, 0) is number of generated signal events
        scale = nmbSignalGenEvents[-1] / HTrue._valsFlatIndex[0]
        HTrue._valsFlatIndex *= scale
        print(f"True moment values:\n{HTrue}")

        # setup moment calculators for data
        dataSet = DataSet(beamPolarization, dataSignalAccInBin, phaseSpaceData = dataPsAccInBin, nmbGenEvents = nmbPsGenEvents[-1])
        momentsInBins.append(MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix", binCenters = {binVarMass : massBinCenter}))
        # setup moment calculator to hold true values
        momentsInBinsTruth.append(MomentCalculator(momentIndices, dataSet, binCenters = {binVarMass : massBinCenter}, _HPhys = HTrue))

        # plot angular distributions for mass bin
        plotAngularDistr(dataPsAccInBin, dataPsGenInBin, dataSignalAccInBin, dataSignalGenInBin, pdfFileNamePrefix = f"{outFileDirName}/angDistr_{binLabel(momentsInBins[-1])}_")
    moments      = MomentCalculatorsKinematicBinning(momentsInBins)
    momentsTruth = MomentCalculatorsKinematicBinning(momentsInBinsTruth)

    # calculate and plot integral matrix for all mass bins
    with timer.timeThis(f"Time to calculate acceptance integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating acceptance integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateIntegralMatrices(forceCalculation = True)
      print(f"Acceptance integral matrix for first bin at {massBinning[0]} {binVarMass.unit}:\n{moments[0].integralMatrix}")
      eigenVals, _ = moments[0].integralMatrix.eigenDecomp
      print(f"Sorted eigenvalues of acceptance integral matrix for first bin at {massBinning[0]} {binVarMass.unit}:\n{np.sort(eigenVals)}")
      # plot acceptance integral matrices for all kinematic bins
      for momentsInBin in moments:
        label = binLabel(momentsInBin)
        # title = binTitle(momentsInBin)
        plotComplexMatrix(momentsInBin.integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/accMatrix_{label}_",
                          axisTitles = ("Physical Moment Index", "Measured Moment Index"), plotTitle = f"{label}: "r"$\mathrm{\mathbf{I}}_\text{acc}$, ",
                          zRangeAbs = 1.1, zRangeImag = 0.075)
        plotComplexMatrix(momentsInBin.integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/accMatrixInv_{label}_",
                          axisTitles = ("Measured Moment Index", "Physical Moment Index"), plotTitle = f"{label}: "r"$\mathrm{\mathbf{I}}_\text{acc}^{-1}$, ",
                          zRangeAbs = 100, zRangeImag = 10)

    namePrefix = "norm" if normalizeMoments else "unnorm"
    if calcAccPsMoments:
      # calculate moments of accepted phase-space data
      with timer.timeThis(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads"):
        print(f"Calculating moments of phase-space MC data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
        moments.calculateMoments(dataSource = MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE, normalize = normalizeMoments)
        # plot accepted phase-space moments in each kinematic bin
        for massBinIndex, momentsInBin in enumerate(moments):
          label = binLabel(momentsInBin)
          title = binTitle(momentsInBin)
          print(f"Measured moments of accepted phase-space data for kinematic bin {title}:\n{momentsInBin.HMeas}")
          print(f"Physical moments of accepted phase-space data for kinematic bin {title}:\n{momentsInBin.HPhys}")
          # construct true moments for phase-space data
          HTruePs = MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
          HTruePs._valsFlatIndex[momentIndices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1 if normalizeMoments else nmbPsGenEvents[massBinIndex]  # ... except for H_0(0, 0)
          # set H_0^meas(0, 0) to 0 so that one can better see the other H_0^meas moments
          momentsInBin.HMeas._valsFlatIndex[0] = 0
          # plot measured and physical moments; the latter should match the true moments exactly except for tiny numerical effects
          plotMomentsInBin(momentsInBin.HMeas, normalizeMoments,                  pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_accPs_", plotLegend = False)
          # plotMomentsInBin(momentsInBin.HPhys, normalizeMoments, HTrue = HTruePs, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{binLabel}_accPsCorr_")
        # plot kinematic dependences of all phase-space moments
        for qnIndex in momentIndices.QnIndices():
          HVals = tuple(MomentValueAndTruth(*momentsInBin.HMeas[qnIndex]) for momentsInBin in moments)
          plotMoments(HVals, massBinning, normalizeMoments, momentLabel = qnIndex.label,
                      pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{massBinning.var.name}_accPs_", histTitle = qnIndex.title, plotLegend = False)

    # calculate and plot moments of signal data
    with timer.timeThis(f"Time to calculate moments of signal data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating moments of signal data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateMoments(normalize = normalizeMoments, nmbBootstrapSamples = nmbBootstrapSamples)
      # plot moments in each kinematic bin
      for massBinIndex, momentsInBin in enumerate(moments):
        label = binLabel(momentsInBin)
        title = binTitle(momentsInBin)
        print(f"Measured moments of signal data for kinematic bin {title}:\n{momentsInBin.HMeas}")
        print(f"Physical moments of signal data for kinematic bin {title}:\n{momentsInBin.HPhys}")
        plotMomentsInBin(momentsInBin.HPhys, normalizeMoments, momentsTruth[massBinIndex].HPhys,
                         pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_")
        #TODO also plot correlation matrices
        plotMomentsCovMatrices(momentsInBin.HPhys, pdfFileNamePrefix = f"{outFileDirName}/covMatrix_{label}_",
                               axisTitles = ("Physical Moment Index", "Physical Moment Index"), plotTitle = f"{label}: ")
        if nmbBootstrapSamples > 0:
          graphTitle = f"({label})"
          plotMomentsBootstrapDistributions1D(momentsInBin.HPhys, momentsTruth[massBinIndex].HPhys,
                                              pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_", histTitle = title)
          # plotMomentsBootstrapDistributions2D(momentsInBin.HPhys, momentsTruth[massBinIndex].HPhys,
          #                                     pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_", histTitle = title)
          plotMomentsBootstrapDiffInBin(momentsInBin.HPhys, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_", graphTitle = title)

      # plot kinematic dependences of all moments
      pullParameters: Dict[QnMomentIndex, Dict[bool, Tuple[Tuple[float, float], Tuple[float, float]]]] = {} # {index : {isReal : ((mean val, mean err), (sigma val, sigma err))}}
      for qnIndex in momentIndices.QnIndices():
        plotMoments1D(moments, qnIndex, massBinning, normalizeMoments, momentsTruth,
                      pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_", histTitle = qnIndex.title)
        if plotPulls:
          pullParameters[qnIndex] = plotPullsForMoment(moments, qnIndex, momentsTruth, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_", histTitle = qnIndex.title)
        if nmbBootstrapSamples > 0:
          plotMomentsBootstrapDiff1D(moments, qnIndex, massBinning, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_", graphTitle = qnIndex.title)
      if plotPulls:
        plotPullParameters(pullParameters, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_")

      print("Check H_0(0, 0) with true generated values:")  # instead of the values from the fit results
      H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
      for binIndex in range(len(moments)):
        H = moments[binIndex].HPhys[H000Index]
        HTruth = momentsTruth[binIndex].HPhys[H000Index]
        print(f"    {binIndex}: H_0(0, 0) = {H.val.real} +- {H.uncertRe}"
              f" vs. Truth = {HTruth.val.real} +- {HTruth.uncertRe}"
              f" vs. # gen = {nmbSignalGenEvents[binIndex]}")
        momentsTruth[binIndex].HPhys._valsFlatIndex[0] = complex(nmbSignalGenEvents[binIndex])
      plotMoments1D(moments, H000Index, massBinning, normalizeMoments, momentsTruth,
                    pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_gen_", histTitle = H000Index.title)

    timer.stop("Total execution time")
    print(timer.summary)
