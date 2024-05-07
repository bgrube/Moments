#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3

import functools
import numpy as np
import threadpoolctl
from typing import List

import ROOT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  binLabel,
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
  MomentResult,
  QnWaveIndex,
)
from PlottingUtilities import (
  HistAxisBinning,
  plotComplexMatrix,
  plotMomentsBootstrapDiffInBin,
  plotMomentsBootstrapDistributions1D,
  plotMomentsInBin,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads basisFunctions.C
import testMomentsPhotoProd
import Utilities

# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  ROOT.gRandom.SetSeed(1234567890)
  # ROOT.EnableImplicitMT(10)
  setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 3):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of test case
    outFileDirName        = Utilities.makeDirPath("./plotsPhotoProdWeighted")
    nmbPwaMcEventsSig     = 1000
    nmbPwaMcEventsBkg     = 1000
    # nmbPwaMcEventsSig     = 10000000
    # nmbPwaMcEventsBkg     = 10000000
    nmbAcceptedPsMcEvents = 10000000
    beamPolarization      = 1.0
    # maxL                  = 1  # maximum L quantum number of moments to be calculated
    maxL                  = 5  # maximum L quantum number of moments to be calculated
    # normalizeMoments      = True
    normalizeMoments      = False
    # nmbBootstrapSamples   = 0
    nmbBootstrapSamples   = 10000
    # define angular distribution of signal
    partialWaveAmplitudesSig = [  # set of all possible waves up to ell = 2
      # negative-reflectivity waves
      AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.4 + 0.1j),  # P_-1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m =  0), val =  0.3 - 0.8j),  # P_0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.8 + 0.7j),  # P_+1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -2), val =  0.1 - 0.4j),  # D_-2^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val =  0.5 + 0.2j),  # D_-1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val = -0.1 - 0.2j),  # D_ 0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val =  0.2 - 0.1j),  # D_+1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.2 + 0.3j),  # D_+2^-
      # positive-reflectivity waves
      AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = -1), val =  0.5 - 0.1j),  # P_-1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.8 - 0.3j),  # P_0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.6 + 0.3j),  # P_+1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.2 + 0.1j),  # D_-2^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -1), val =  0.2 - 0.3j),  # D_-1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m =  0), val =  0.1 - 0.2j),  # D_ 0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +1), val =  0.2 + 0.5j),  # D_+1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val = -0.3 - 0.1j),  # D_+2^+
    ]
    amplitudeSetSig = AmplitudeSet(partialWaveAmplitudesSig)
    # define angular distribution of background
    partialWaveAmplitudesBkg = [  # set of all possible waves up to ell = 2
      # negative-reflectivity waves
      AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.9 + 0.7j),  # P_-1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m =  0), val = -0.6 + 0.4j),  # P_0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.9 - 0.8j),  # P_+1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -2), val = -1.0 - 0.7j),  # D_-2^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val = -0.8 - 0.7j),  # D_-1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val =  0.4 + 0.3j),  # D_ 0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val = -0.6 - 0.1j),  # D_+1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.1 - 0.9j),  # D_+2^-
      # positive-reflectivity waves
      AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = -1), val = -1.0 + 0.8j),  # P_-1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.2 + 0.2j),  # P_0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.0 - 0.3j),  # P_+1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.7 + 0.9j),  # D_-2^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -1), val = -0.4 - 0.5j),  # D_-1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m =  0), val = -0.3 + 0.2j),  # D_ 0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +1), val = -1.0 - 0.4j),  # D_+1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val =  0.5 - 0.2j),  # D_+2^+
    ]
    amplitudeSetBkg = AmplitudeSet(partialWaveAmplitudesBkg)
    # formulas for detection efficiency
    # x = cos(theta) in [-1, +1], y = phi in [-180, +180] deg, z = Phi in [-180, +180] deg
    # efficiencyFormula = "1"  # acc_perfect
    efficiencyFormula = "(1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / 1.5**3"  # acc_1; even in all variables
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # calculate true moment values and generate data from partial-wave amplitudes
    with timer.timeThis("Time to generate MC data from partial waves"):
      # generate signal distribution
      HTrueSig: MomentResult = amplitudeSetSig.photoProdMomentSet(maxL, normalize = (True if normalizeMoments else nmbPwaMcEventsSig))
      print(f"True moment values for signal:\n{HTrueSig}")
      dataPwaModelSig: ROOT.RDataFrame = testMomentsPhotoProd.genDataFromWaves(
        nmbPwaMcEventsSig, beamPolarization, amplitudeSetSig, efficiencyFormula, outFileNamePrefix = f"{outFileDirName}/", nameSuffix = "Sig", regenerateData = True)
      dataPwaModelSig = dataPwaModelSig.Define("discrVariable", "gRandom->Gaus(0, 0.1)")
      treeName = "data"
      fileNameSig = f"{outFileDirName}/intensitySig.photoProd.root"
      dataPwaModelSig.Snapshot(treeName, fileNameSig)
      dataPwaModelSig = ROOT.RDataFrame(treeName, fileNameSig)
      histDiscrSig = dataPwaModelSig.Histo1D(ROOT.RDF.TH1DModel("Signal", ";Discriminating variable;Count / 0.02", 100, -1, +1), "discrVariable").GetValue()
      # generate background distribution
      HTrueBkg: MomentResult = amplitudeSetBkg.photoProdMomentSet(maxL, normalize = (True if normalizeMoments else nmbPwaMcEventsBkg))
      print(f"True moment values for signal:\n{HTrueBkg}")
      dataPwaModelBkg: ROOT.RDataFrame = testMomentsPhotoProd.genDataFromWaves(
        nmbPwaMcEventsBkg, beamPolarization, amplitudeSetBkg, efficiencyFormula, outFileNamePrefix = f"{outFileDirName}/", nameSuffix = "Bkg", regenerateData = True)
      dataPwaModelBkg = dataPwaModelBkg.Define("discrVariable", "gRandom->Uniform(0, 2) - 1")
      fileNameBkg = f"{outFileDirName}/intensityBkg.photoProd.root"
      dataPwaModelBkg.Snapshot(treeName, fileNameBkg)
      dataPwaModelBkg = ROOT.RDataFrame(treeName, fileNameBkg)
      histDiscrBkg = dataPwaModelBkg.Histo1D(ROOT.RDF.TH1DModel("Background", ";Discriminating variable;Count / 0.02", 100, -1, +1), "discrVariable").GetValue()
      # concatenate signal and background data frames vertically
      dataPwaModel = ROOT.RDataFrame(treeName, (fileNameSig, fileNameBkg))
      # plot discriminating variable
      signalRange = (-0.3, +0.3)
      sideBands   = ((-1, -0.4), (+0.4, +1))
      histDiscr = dataPwaModel.Histo1D(ROOT.RDF.TH1DModel("Total", ";Discriminating variable;Count / 0.02", 100, -1, +1), "discrVariable").GetValue()
      histDiscr.SetLineWidth(2)
      histDiscrSig.SetLineColor(ROOT.kGreen + 2)
      histDiscrBkg.SetLineColor(ROOT.kRed   + 1)
      histDiscrSig.SetLineStyle(ROOT.kDashed)
      histDiscrBkg.SetLineStyle(ROOT.kDashed)
      histStack = ROOT.THStack("hDiscrVariableSim", ";Discriminating variable;Count / 0.02")
      histStack.Add(histDiscr)
      histStack.Add(histDiscrBkg)
      histStack.Add(histDiscrSig)
      canv = ROOT.TCanvas()
      histStack.Draw("NOSTACK")
      legend = canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
      # shade signal region and side bands
      canv.Update()
      box = ROOT.TBox()
      for bounds in (signalRange, *sideBands):
        box.SetFillColorAlpha(ROOT.kBlack, 0.15)
        box.DrawBox(bounds[0], canv.GetUymin(), bounds[1], canv.GetUymax())
      legend.Draw()
      canv.SaveAs(f"{outFileDirName}/{histStack.GetName()}.pdf")
      # define event weights
      dataPwaModel = dataPwaModel.Define("eventWeight", f"""
        if (({signalRange[0]} < discrVariable) and (discrVariable < {signalRange[1]}))
          return 1.0;
        else if (  (({sideBands[0][0]} < discrVariable) and (discrVariable < {sideBands[0][1]}))
                or (({sideBands[1][0]} < discrVariable) and (discrVariable < {sideBands[1][1]})))
          return -0.5;
        else
          return 0.0;
      """)
      hist = dataPwaModel.Histo1D(ROOT.RDF.TH1DModel("hDiscrVariableSimSbSubtr", ";Discriminating variable;Count / 0.02", 100, -1, +1), "discrVariable", "eventWeight")
      hist.Draw()
      canv.SaveAs(f"{outFileDirName}/{hist.GetName()}.pdf")

    # plot angular distributions of data generated from partial-wave amplitudes
    nmbBins = testMomentsPhotoProd.TH3_NMB_BINS
    histBinning = (nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180)
    hists = (
      (dataPwaModelSig.Filter(f"({signalRange[0]} < discrVariable) and (discrVariable < {signalRange[1]})")
                      .Histo3D(ROOT.RDF.TH3DModel("dataSig", testMomentsPhotoProd.TH3_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg")),
      (dataPwaModelBkg.Filter(f"(({sideBands[0][0]} < discrVariable) and (discrVariable < {sideBands[0][1]}))"
                           f"or (({sideBands[1][0]} < discrVariable) and (discrVariable < {sideBands[1][1]}))")
                      .Histo3D(ROOT.RDF.TH3DModel("dataBkg", testMomentsPhotoProd.TH3_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg")),
      dataPwaModel.Histo3D(ROOT.RDF.TH3DModel("data",        testMomentsPhotoProd.TH3_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg"),
      dataPwaModel.Histo3D(ROOT.RDF.TH3DModel("dataSbSubtr", testMomentsPhotoProd.TH3_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg", "eventWeight"),
    )
    for hist in hists:
      hist.SetMinimum(0)
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
      print(f"Integral of histogram '{hist.GetName()}' = {hist.Integral()}")
      canv.SaveAs(f"{outFileDirName}/{hist.GetName()}.pdf")
    print(f"Sum of weights = {dataPwaModel.Sum('eventWeight').GetValue()}")

    # generate accepted phase-space data
    with timer.timeThis("Time to generate phase-space MC data"):
      dataAcceptedPs = testMomentsPhotoProd.genAccepted2BodyPsPhotoProd(
        nmbAcceptedPsMcEvents, efficiencyFormula, outFileNamePrefix = f"{outFileDirName}/", regenerateData = True)

    # define input data
    data = dataPwaModel
    # data = dataPwaModelSig
    HTrue = HTrueSig
    # data = dataPwaModelBkg
    # HTrue = HTrueBkg

    # setup moment calculator
    momentIndices = MomentIndices(maxL)
    binVarMass    = KinematicBinningVariable(name = "mass", label = "#it{m}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    massBinning   = HistAxisBinning(nmbBins = 1, minVal = 1.0, maxVal = 2.0, _var = binVarMass)
    momentsInBins:      List[MomentCalculator] = []
    momentsInBinsTruth: List[MomentCalculator] = []
    for massBinCenter in massBinning:
      # dummy bins with identical data sets
      dataSet = DataSet(beamPolarization, data, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbAcceptedPsMcEvents)  #TODO nmbAcceptedPsMcEvents is not correct number to normalize integral matrix
      momentsInBins.append(MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix", binCenters = {binVarMass : massBinCenter}))
      # dummy truth values; identical for all bins
      momentsInBinsTruth.append(MomentCalculator(momentIndices, dataSet, binCenters = {binVarMass : massBinCenter}, _HPhys = HTrue))
    moments      = MomentCalculatorsKinematicBinning(momentsInBins)
    momentsTruth = MomentCalculatorsKinematicBinning(momentsInBinsTruth)

    # calculate integral matrix
    with timer.timeThis(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads"):
      moments.calculateIntegralMatrices(forceCalculation = True)
      # print acceptance integral matrix for first kinematic bin
      print(f"Acceptance integral matrix\n{moments[0].integralMatrix}")
      eigenVals, _ = moments[0].integralMatrix.eigenDecomp
      print(f"Eigenvalues of acceptance integral matrix\n{np.sort(eigenVals)}")
      # plot acceptance integral matrices for all kinematic bins
      for momentsInBin in moments:
        label = binLabel(momentsInBin)
        plotComplexMatrix(moments[0].integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/I_acc_{label}")
        plotComplexMatrix(moments[0].integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/I_inv_{label}")

    # calculate moments of data generated from partial-wave amplitudes
    with timer.timeThis(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads"):
      moments.calculateMoments(normalize = normalizeMoments, nmbBootstrapSamples = nmbBootstrapSamples)
      # print all moments for first kinematic bin
      print(f"Measured moments of data generated according to partial-wave amplitudes\n{moments[0].HMeas}")
      print(f"Physical moments of data generated according to partial-wave amplitudes\n{moments[0].HPhys}")
      # plot moments in each kinematic bin
      namePrefix = "norm" if normalizeMoments else "unnorm"
      label = binLabel(moments[0])
      plotMomentsInBin(moments[0].HPhys, normalizeMoments, HTrue, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_")
      if nmbBootstrapSamples > 0:
        plotMomentsBootstrapDistributions1D(moments[0].HPhys, HTrue, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_")
        plotMomentsBootstrapDiffInBin      (moments[0].HPhys,        pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_")

    timer.stop("Total execution time")
    print(timer.summary)
