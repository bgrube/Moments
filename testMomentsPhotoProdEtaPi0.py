#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=4

import functools
import numpy as np
import pandas as pd
import threadpoolctl
from typing import (
  List,
  Tuple,
)

import ROOT

import MomentCalculator
from PlottingUtilities import (
  HistAxisBinning,
  plotComplexMatrix,
  plotMomentsBootstrapDiff1D,
  plotMomentsBootstrapDiffInBin,
  plotMomentsBootstrapDistributions1D,
  plotMomentsBootstrapDistributions2D,
  plotMomentsCovMatrices,
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
) -> List[MomentCalculator.AmplitudeValue]:
  """Reads partial-wave amplitudes values for given mass bin from CSV data"""
  df = pd.read_csv(csvFileName).astype({"mass": float})
  df = df.loc[np.isclose(df["mass"], massBinCenter)].drop(columns = ["mass"])
  print(f"Amplitudes for mass bin at {massBinCenter} GeV\n{df}")
  assert len(df) == 1, f"Expected exactly 1 row for mass-bin center {massBinCenter} GeV, but found {len(df)}"
  # Pandas cannot read-back complex values out of the box
  # there also seems to be no interest in fixing that; see <https://github.com/pandas-dev/pandas/issues/9379>
  # have to convert columns by hand
  s = df.astype('complex128').loc[df.index[0]] / 5.0  #TODO clarify why H_0(0, 0) is by a factor of 25 larger than number of generated events
  partialWaveAmplitudes = [
    # negative-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 0, m =  0), val = s['S0-' ]),  # S_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -1), val = s['D1--']),  # D_-1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m =  0), val = s['D0+-']),  # D_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +1), val = s['D1+-']),  # D_+1^-
    # positive-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 0, m =  0), val = s['S0+' ]),  # S_0^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -2), val = s['D2-+']),  # D_-2^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +2), val = s['D2++']),  # D_+2^+
  ]
  # print(f"!!! {partialWaveAmplitudes=}")
  return partialWaveAmplitudes


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 3):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of test case
    outFileDirName       = Utilities.makeDirPath("./plotsPhotoProdEtaPi0")
    treeName             = "etaPi0"
    signalAccFileName    = "./dataPhotoProdEtaPi0/a0a2_signal_acc_flat.root"
    signalGenFileName    = "./dataPhotoProdEtaPi0/a0a2_signal_gen_flat.root"
    signalPWAmpsFileName = "./dataPhotoProdEtaPi0/a0a2_raw/a0a2_complex_amps.csv"
    psAccFileName        = "./dataPhotoProdEtaPi0/a0a2_phaseSpace_acc_flat.root"
    psGenFileName        = "./dataPhotoProdEtaPi0/a0a2_phaseSpace_gen_flat.root"
    beamPolarization     = 1.0  #TODO read from tree
    maxL                 = 1  # define maximum L quantum number of moments
    # maxL                 = 5  # define maximum L quantum number of moments
    normalizeMoments     = False
    # nmbBootstrapSamples  = 0
    nmbBootstrapSamples  = 1000
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # plot all signal and phase-space data
    #TODO plot for each mass bin
    print(f"Loading accepted signal data from tree '{treeName}' in file '{signalAccFileName}'")
    dataSignalAcc = ROOT.RDataFrame(treeName, signalAccFileName)
    print(f"Loading generated signal data from tree '{treeName}' in file '{signalGenFileName}'")
    dataSignalGen = ROOT.RDataFrame(treeName, signalGenFileName)
    print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}'")
    dataPsAcc = ROOT.RDataFrame(treeName, psAccFileName)
    print(f"Loading generated phase-space data from tree '{treeName}' in file '{psGenFileName}'")
    dataPsGen = ROOT.RDataFrame(treeName, psGenFileName)
    nmbBinsSig = 15
    nmbBinsPs  = nmbBinsSig
    hists = (
      dataSignalAcc.Histo3D(
        ROOT.RDF.TH3DModel("hSignalAcc", ";cos#theta;#phi [deg];#Phi [deg]", nmbBinsSig, -1, +1, nmbBinsSig, -180, +180, nmbBinsSig, -180, +180),
        "cosTheta", "phiDeg", "PhiDeg"),
      dataSignalGen.Histo3D(
        ROOT.RDF.TH3DModel("hSignalGen", ";cos#theta;#phi [deg];#Phi [deg]", nmbBinsSig, -1, +1, nmbBinsSig, -180, +180, nmbBinsSig, -180, +180),
        "cosTheta", "phiDeg", "PhiDeg"),
      dataPsAcc.Histo3D(
        ROOT.RDF.TH3DModel("hPhaseSpaceAcc", ";cos#theta;#phi [deg];#Phi [deg]", nmbBinsPs, -1, +1, nmbBinsPs, -180, +180, nmbBinsPs, -180, +180),
        "cosTheta", "phiDeg", "PhiDeg"),
      dataPsGen.Histo3D(
        ROOT.RDF.TH3DModel("hPhaseSpaceGen", ";cos#theta;#phi [deg];#Phi [deg]", nmbBinsPs, -1, +1, nmbBinsPs, -180, +180, nmbBinsPs, -180, +180),
        "cosTheta", "phiDeg", "PhiDeg"),
    )
    for hist in hists:
      canv = ROOT.TCanvas()
      hist.SetMinimum(0)
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
      canv.SaveAs(f"{outFileDirName}/{hist.GetName()}.pdf")

    # loop over mass bins
    momentIndices = MomentCalculator.MomentIndices(maxL)
    binVarMass    = MomentCalculator.KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#eta#pi}^{0}}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    # massBinning   = HistAxisBinning(nmbBins = 28, minVal = 0.88, maxVal = 2.00, _var = binVarMass)
    # massBinning   = HistAxisBinning(nmbBins = 2, minVal = 1.28, maxVal = 1.36, _var = binVarMass)
    massBinning   = HistAxisBinning(nmbBins = 1, minVal = 1.12, maxVal = 1.16, _var = binVarMass)
    momentsInBins:      List[MomentCalculator.MomentCalculator] = []
    momentsInBinsTruth: List[MomentCalculator.MomentCalculator] = []
    nmbSignalGenEvents: List[float] = []
    assert len(massBinning) > 0, f"Need at least one mass bin, but found {len(massBinning)}"
    for massBinIndex, massBinCenter in enumerate(massBinning):
      massBinRange = massBinning.binValueRange(massBinIndex)
      print(f"Preparing {binVarMass.name} bin at {massBinCenter} {binVarMass.unit} with range {massBinRange} {binVarMass.unit}")

      # calculate true moments
      amplitudeSet = MomentCalculator.AmplitudeSet(amps = readPartialWaveAmplitudes(signalPWAmpsFileName, massBinCenter), tolerance = 1e-11)
      HTrue: MomentCalculator.MomentResult = amplitudeSet.photoProdMomentSet(maxL, printMoments = False, normalize = normalizeMoments)
      print(f"True moment values\n{HTrue}")

      # load data for mass bin
      binMassRangeFilter = f"(({massBinRange[0]} < {binVarMass.name}) && ({binVarMass.name} < {massBinRange[1]}))"
      print(f"Loading accepted signal data from tree '{treeName}' in file '{signalAccFileName}' and applying filter {binMassRangeFilter}")
      dataSignalAcc = ROOT.RDataFrame(treeName, signalAccFileName).Filter(binMassRangeFilter)
      print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
      dataPsAcc = ROOT.RDataFrame(treeName, psAccFileName).Filter(binMassRangeFilter)
      print(f"Loading generated signal data from tree '{treeName}' in file '{signalGenFileName}' and applying filter {binMassRangeFilter}")
      dataSignalGen = ROOT.RDataFrame(treeName, signalGenFileName).Filter(binMassRangeFilter)
      nmbSignalGenEvents.append(dataSignalGen.Count().GetValue())
      print(f"Loading generated phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
      dataPsGen = ROOT.RDataFrame(treeName, psGenFileName).Filter(binMassRangeFilter)
      nmbPsGenEvents = dataPsGen.Count().GetValue()

      # setup moment calculator for data
      dataSet = MomentCalculator.DataSet(beamPolarization, dataSignalAcc, phaseSpaceData = dataPsAcc, nmbGenEvents = nmbPsGenEvents)
      momentsInBins.append(MomentCalculator.MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix", _binCenters = {binVarMass : massBinCenter}))
      # setup moment calculator to hold true values
      momentsInBinsTruth.append(MomentCalculator.MomentCalculator(momentIndices, dataSet, _binCenters = {binVarMass : massBinCenter}, _HPhys = HTrue))
    moments      = MomentCalculator.MomentCalculatorsKinematicBinning(momentsInBins)
    momentsTruth = MomentCalculator.MomentCalculatorsKinematicBinning(momentsInBinsTruth)

    # calculate integral matrices
    matrixAxisTitles = ("Moment Index", ) * 2
    with timer.timeThis(f"Time to calculate integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      moments.calculateIntegralMatrices(forceCalculation = True)
      # print acceptance integral matrix for first kinematic bin
      print(f"Acceptance integral matrix for first bin\n{moments[0].integralMatrix}")
      eigenVals, _ = moments[0].integralMatrix.eigenDecomp
      print(f"Eigenvalues of acceptance integral matrix for first bin\n{np.sort(eigenVals)}")
      # plot acceptance integral matrices for all kinematic bins
      for momentsInBin in moments:
        binLabel = "_".join(momentsInBin.binLabels)
        # binTitle = ", ".join(momentsInBin.binTitles)
        plotComplexMatrix(momentsInBin.integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/accMatrix_{binLabel}_",
                          axisTitles = matrixAxisTitles, plotTitle = f"{binLabel}: "r"$\mathrm{\mathbf{I}}_\text{acc}$, ")
        plotComplexMatrix(momentsInBin.integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/accMatrixInv_{binLabel}",
                          axisTitles = matrixAxisTitles, plotTitle = f"{binLabel}: "r"$\mathrm{\mathbf{I}}_\text{acc}^{-1}$, ")

    #TODO add loop over mass bins
    # # calculate moments of accepted phase-space data
    # with timer.timeThis(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads"):
    #   momentCalculator.calculateMoments(dataSource = MomentCalculator.MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE)
    #   # print all moments
    #   print(f"Measured moments of accepted phase-space data\n{momentCalculator.HMeas}")
    #   print(f"Physical moments of accepted phase-space data\n{momentCalculator.HPhys}")
    #   # plot moments
    #   HTruePs = MomentCalculator.MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
    #   HTruePs._valsFlatIndex[momentIndices[MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1  # ... except H_0(0, 0), which is 1
    #   plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTruePs, pdfFileNamePrefix = f"{outFileDirName}/hPs_")

    # calculate moments of signal data
    with timer.timeThis(f"Time to calculate moments for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      moments.calculateMoments(normalize = normalizeMoments, nmbBootstrapSamples = nmbBootstrapSamples)
      # print all moments for first kinematic bin
      print(f"Measured moments of signal data for first kinematic bin\n{moments[0].HMeas}")
      print(f"Physical moments of signal data for first kinematic bin\n{moments[0].HPhys}")

      # plot moments in each kinematic bin
      namePrefix = "norm" if normalizeMoments else "unnorm"

      for massBinIndex, momentsInBin in enumerate(moments):
        binLabel = "_".join(momentsInBin.binLabels)
        binTitle = ", ".join(momentsInBin.binTitles)
        plotMomentsInBin(momentsInBin.HPhys, normalizeMoments, momentsTruth[massBinIndex].HPhys,
                         pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{binLabel}_")
        plotMomentsCovMatrices(momentsInBin.HPhys, pdfFileNamePrefix = f"{outFileDirName}/covMatrix_{binLabel}_",
                               axisTitles = matrixAxisTitles, plotTitle = f"{binLabel}: ")
        if nmbBootstrapSamples > 0:
          graphTitle = f"({binLabel})"
          plotMomentsBootstrapDistributions1D(momentsInBin.HPhys, momentsTruth[massBinIndex].HPhys,
                                              pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{binLabel}_", histTitle = binTitle)
          plotMomentsBootstrapDistributions2D(momentsInBin.HPhys, momentsTruth[massBinIndex].HPhys,
                                              pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{binLabel}_", histTitle = binTitle)
          plotMomentsBootstrapDiffInBin(momentsInBin.HPhys, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{binLabel}_", graphTitle = binTitle)

      # plot kinematic dependences of all moments
      for qnIndex in momentIndices.QnIndices():
        plotMoments1D(moments, qnIndex, massBinning, normalizeMoments, momentsTruth,
                      pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_", histTitle = qnIndex.title)
        if nmbBootstrapSamples > 0:
          plotMomentsBootstrapDiff1D(moments, qnIndex, massBinning, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_", graphTitle = qnIndex.title)

      print("Check H_0(0, 0) with true generated values:")  # instead of the values from the fit results
      H000Index = MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M =0)
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
