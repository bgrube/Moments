#!/usr/bin/env python3
# performs moments analysis for eta pi0 real-data events

import functools
import numpy as np
import pandas as pd
import threadpoolctl
from typing import (
  List,
)

import ROOT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
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
  plotMoments1D,
  plotMomentsInBin,
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
  df = pd.read_csv(csvFileName, index_col = [0]).astype({"mass": float})
  df = df.loc[np.isclose(df["mass"], massBinCenter)].drop(columns = ["mass"])
  assert len(df) == 1, f"Expected exactly 1 row for mass-bin center {massBinCenter} GeV, but found {len(df)}"
  # Pandas cannot read-back complex values out of the box
  # there also seems to be no interest in fixing that; see <https://github.com/pandas-dev/pandas/issues/9379>
  # have to convert columns by hand
  ampSeries = df.astype('complex128').loc[df.index[0]]
  # add amplitudes of a_2(1320) and a_2(1700)
  a2AmpKeys = sorted([key for key in ampSeries.index if "::D" in key])  # keys of a_2(1320) amplitudes
  for a2AmpKey in a2AmpKeys:
    a2PrimeAmpKey = a2AmpKey.replace("::D", "::pD")  # key for a_2(1700) amplitude
    # print(f"!!! {a2AmpKey}: {ampSeries[a2AmpKey]} + {a2PrimeAmpKey}: {ampSeries[a2PrimeAmpKey]} = {ampSeries[a2AmpKey] + ampSeries[a2PrimeAmpKey]}")
    ampSeries[a2AmpKey] = ampSeries[a2AmpKey] + ampSeries[a2PrimeAmpKey]
    ampSeries.drop(a2PrimeAmpKey, inplace = True)
  # take amplitudes for polarization angle of 0 because their scale factor is 1
  print(f"Partial-wave amplitudes for mass bin at {massBinCenter} GeV:\n{ampSeries.filter(like = 'EtaPi0_000::')}")
  partialWaveAmplitudes = [
    # negative-reflectivity waves
    AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val = ampSeries['EtaPi0_000::NegativeRe::S0+-']),  # S_0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -2), val = ampSeries['EtaPi0_000::NegativeRe::D2--']),  # D_-2^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val = ampSeries['EtaPi0_000::NegativeRe::D1--']),  # D_-1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val = ampSeries['EtaPi0_000::NegativeRe::D0+-']),  # D_0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val = ampSeries['EtaPi0_000::NegativeRe::D1+-']),  # D_+1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +2), val = ampSeries['EtaPi0_000::NegativeRe::D2+-']),  # D_+2^-
    # positive-reflectivity waves
    AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val = ampSeries['EtaPi0_000::PositiveRe::S0++']),  # S_0^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val = ampSeries['EtaPi0_000::PositiveRe::D2-+']),  # D_-2^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -1), val = ampSeries['EtaPi0_000::PositiveRe::D1-+']),  # D_-1^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m =  0), val = ampSeries['EtaPi0_000::PositiveRe::D0++']),  # D_0^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +1), val = ampSeries['EtaPi0_000::PositiveRe::D1++']),  # D_+1^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val = ampSeries['EtaPi0_000::PositiveRe::D2++']),  # D_+2^+
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
    outFileDirName          = Utilities.makeDirPath("./plotsPhotoProdEtaPi0")
    treeName                = "etaPi0"
    dataFileName            = "./dataPhotoProdEtaPi0/data_flat.root"
    psAccFileName           = "./dataPhotoProdEtaPi0/phaseSpace_acc_flat.root"
    psGenFileName           = "./dataPhotoProdEtaPi0/phaseSpace_gen_flat.root"
    pwAmpsFileName          = "./dataPhotoProdEtaPi0/evaluate_amplitude/evaluate_amplitude.csv"
    beamPolarization        = 0.35062  # for polarization angle of 0
    # maxL                    = 1  # define maximum L quantum number of moments
    maxL                    = 5  # define maximum L quantum number of moments
    normalizeMoments        = False
    # plotAccIntegralMatrices = True
    plotAccIntegralMatrices = False
    # calcAccPsMoments        = True
    calcAccPsMoments        = False
    binVarMass              = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#eta#pi}^{0}}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    massBinning             = HistAxisBinning(nmbBins = 17, minVal = 1.04, maxVal = 1.72, _var = binVarMass)
    nmbOpenMpThreads        = ROOT.getNmbOpenMpThreads()

    # load all signal and phase-space data
    print(f"Loading real data from tree '{treeName}' in file '{dataFileName}'")
    data = ROOT.RDataFrame(treeName, dataFileName)
    print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}'")
    dataPsAcc = ROOT.RDataFrame(treeName, psAccFileName)
    print(f"Loading generated phase-space data from tree '{treeName}' in file '{psGenFileName}'")
    dataPsGen = ROOT.RDataFrame(treeName, psGenFileName)
    # plot total angular distributions
    plotAngularDistr(dataPsAcc, dataPsGen, data, dataSignalGen = None, pdfFileNamePrefix = f"{outFileDirName}/angDistr_total_")

    # setup MomentCalculators for all mass bins
    momentIndices = MomentIndices(maxL)
    momentsInBins:    List[MomentCalculator] = []
    momentsInBinsPwa: List[MomentCalculator] = []
    nmbPsGenEvents:   List[int]              = []
    assert len(massBinning) > 0, f"Need at least one mass bin, but found {len(massBinning)}"
    with timer.timeThis(f"Time to load data and setup MomentCalculators for {len(massBinning)} bins"):
      for massBinIndex, massBinCenter in enumerate(massBinning):
        massBinRange = massBinning.binValueRange(massBinIndex)
        print(f"Preparing {binVarMass.name} bin [{massBinIndex} of {len(massBinning)}] at {massBinCenter} {binVarMass.unit} with range {massBinRange} {binVarMass.unit}")

        # load data for mass bin
        binMassRangeFilter = f"(({massBinRange[0]} < {binVarMass.name}) && ({binVarMass.name} < {massBinRange[1]}))"
        print(f"Loading real data from tree '{treeName}' in file '{dataFileName}' and applying filter {binMassRangeFilter}")
        dataInBin = ROOT.RDataFrame(treeName, dataFileName).Filter(binMassRangeFilter)
        nmbDataEvents = dataInBin.Count().GetValue()
        print(f"Loaded {nmbDataEvents} data events")
        print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
        dataPsAccInBin = ROOT.RDataFrame(treeName, psAccFileName).Filter(binMassRangeFilter)
        nmbPsAccEvents = dataPsAccInBin.Count().GetValue()
        print(f"Loading generated phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
        dataPsGenInBin = ROOT.RDataFrame(treeName, psGenFileName).Filter(binMassRangeFilter)
        nmbPsGenEvents.append(dataPsGenInBin.Count().GetValue())
        print(f"Loaded phase-space events: number generated = {nmbPsGenEvents[-1]}, number accepted = {nmbPsAccEvents}"
              f" -> efficiency = {nmbPsAccEvents / nmbPsGenEvents[-1]:.3f}")

        # calculate moments from PWA fits result
        amplitudeSet = AmplitudeSet(amps = readPartialWaveAmplitudes(pwAmpsFileName, massBinCenter), tolerance = 1e-9)
        HPwa: MomentResult = amplitudeSet.photoProdMomentSet(maxL, printMoments = False, normalize = normalizeMoments)
        print(f"Moment values from partial-wave analysis:\n{HPwa}")

        # setup moment calculators for data
        dataSet = DataSet(beamPolarization, dataInBin, phaseSpaceData = dataPsAccInBin, nmbGenEvents = nmbPsGenEvents[-1])
        momentsInBins.append(MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix", binCenters = {binVarMass : massBinCenter}))
        # setup moment calculator to hold moment values from PWA result
        momentsInBinsPwa.append(MomentCalculator(momentIndices, dataSet, binCenters = {binVarMass : massBinCenter}, _HPhys = HPwa))

        # plot angular distributions for mass bin
        plotAngularDistr(dataPsAccInBin, dataPsGenInBin, dataInBin, dataSignalGen = None, pdfFileNamePrefix = f"{outFileDirName}/angDistr_{momentsInBins[-1].binLabel}_")
    moments = MomentCalculatorsKinematicBinning(momentsInBins)
    momentsPwa = MomentCalculatorsKinematicBinning(momentsInBinsPwa)

    # calculate and plot integral matrix for all mass bins
    with timer.timeThis(f"Time to calculate integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating acceptance integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateIntegralMatrices(forceCalculation = True)
      print(f"Acceptance integral matrix for first bin at {massBinning[0]} {binVarMass.unit}:\n{moments[0].integralMatrix}")
      eigenVals, _ = moments[0].integralMatrix.eigenDecomp
      print(f"Sorted eigenvalues of acceptance integral matrix for first bin at {massBinning[0]} {binVarMass.unit}:\n{np.sort(eigenVals)}")
      # plot acceptance integral matrices for all kinematic bins
      if plotAccIntegralMatrices:
        for momentsInBin in moments:
          binLabel = momentsInBin.binLabel
          plotComplexMatrix(momentsInBin.integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/accMatrix_{binLabel}_",
                            axisTitles = ("Physical Moment Index", "Measured Moment Index"), plotTitle = f"{binLabel}: "r"$\mathrm{\mathbf{I}}_\text{acc}$, ",
                            zRangeAbs = 1.5, zRangeImag = 0.25)
          plotComplexMatrix(momentsInBin.integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/accMatrixInv_{binLabel}_",
                            axisTitles = ("Measured Moment Index", "Physical Moment Index"), plotTitle = f"{binLabel}: "r"$\mathrm{\mathbf{I}}_\text{acc}^{-1}$, ",
                            zRangeAbs = 115, zRangeImag = 30)

    namePrefix = "norm" if normalizeMoments else "unnorm"
    if calcAccPsMoments:
      # calculate moments of accepted phase-space data
      with timer.timeThis(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads"):
        print(f"Calculating moments of phase-space MC data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
        moments.calculateMoments(dataSource = MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE, normalize = normalizeMoments)
        # plot accepted phase-space moments in each kinematic bin
        for massBinIndex, momentsInBin in enumerate(moments):
          binLabel = momentsInBin.binLabel
          binTitle = momentsInBin.binTitle
          print(f"Measured moments of accepted phase-space data for kinematic bin {binTitle}:\n{momentsInBin.HMeas}")
          print(f"Physical moments of accepted phase-space data for kinematic bin {binTitle}:\n{momentsInBin.HPhys}")
          # construct true moments for phase-space data
          HTruePs = MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
          HTruePs._valsFlatIndex[momentIndices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1 if normalizeMoments else nmbPsGenEvents[massBinIndex]  # ... except for H_0(0, 0)
          # set H_0^meas(0, 0) to 0 so that one can better see the other H_0^meas moments
          momentsInBin.HMeas._valsFlatIndex[0] = 0
          # plot measured and physical moments; the latter should match the true moments exactly except for tiny numerical effects
          plotMomentsInBin(momentsInBin.HMeas, normalizeMoments,                  pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{binLabel}_accPs_", plotLegend = False)
          # plotMomentsInBin(momentsInBin.HPhys, normalizeMoments, HTrue = HTruePs, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{binLabel}_accPsCorr_")
        # plot kinematic dependences of all measured moments
        for qnIndex in momentIndices.QnIndices():
          HVals = tuple(MomentValueAndTruth(*momentsInBin.HMeas[qnIndex], _binCenters = momentsInBin.binCenters) for momentsInBin in moments)
          plotMoments(HVals, massBinning, normalizeMoments, momentLabel = qnIndex.label,
                      pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{massBinning.var.name}_accPs_", histTitle = qnIndex.title, plotLegend = False)

    # calculate and plot moments of real data
    with timer.timeThis(f"Time to calculate moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateMoments(normalize = normalizeMoments)
      # plot moments in each kinematic bin
      for massBinIndex, momentsInBin in enumerate(moments):
        binLabel = momentsInBin.binLabel
        binTitle = momentsInBin.binTitle
        print(f"Measured moments of real data for kinematic bin {binTitle}:\n{momentsInBin.HMeas}")
        print(f"Physical moments of real data for kinematic bin {binTitle}:\n{momentsInBin.HPhys}")
        plotMomentsInBin(momentsInBin.HPhys, normalizeMoments, momentsPwa[massBinIndex].HPhys,
                         pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{binLabel}_", plotLegend = False)

      # plot kinematic dependences of all moments
      for qnIndex in momentIndices.QnIndices():
        plotMoments1D(moments, qnIndex, massBinning, normalizeMoments, momentsPwa,
                      pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_", histTitle = qnIndex.title, plotLegend = False)

    timer.stop("Total execution time")
    print(timer.summary)
