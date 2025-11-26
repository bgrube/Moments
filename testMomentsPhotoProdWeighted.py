#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3


from __future__ import annotations

import functools
import numpy as np
import threadpoolctl

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
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
from testMomentsPhotoProd import (
  genAccepted2BodyPsPhotoProd,
  genData,
  TH3_ANG_NMB_BINS,
  TH3_ANG_TITLE,
)
import Utilities

# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def genSigAndBkgDataFromWaves(
  nmbEventsSig:      int,                 # number of signal events to generate
  nmbEventsBkg:      int,                 # number of background events to generate
  amplitudeSetSig:   AmplitudeSet,        # partial-wave amplitudes for signal distribution
  amplitudeSetBkg:   AmplitudeSet,        # partial-wave amplitudes for background distribution
  polarization:      float | None,        # photon-beam polarization
  outputDirName:     str,                 # name of output directory
  efficiencyFormula: str | None = None,   # detection efficiency used to generate data
  regenerateData:    bool       = False,  # if set data are regenerated although .root files exist
) -> tuple[ROOT.RDataFrame, ROOT.RDataFrame, ROOT.RDataFrame]:
  treeName = "data"

  print("Generating signal distribution")
  dataPwaModelSig: ROOT.RDataFrame = genData(
    nmbEvents         = nmbEventsSig,
    polarization      = polarization,
    inputData         = amplitudeSetSig,
    efficiencyFormula = efficiencyFormula,
    outFileNamePrefix = f"{outputDirName}/",
    nameSuffix        = "Sig",
    regenerateData    = regenerateData,
    additionalColDefs = (("discrVariable", "gRandom->Gaus(0, 0.1)"), ),
  )

  print("Generating background distribution")
  dataPwaModelBkg: ROOT.RDataFrame = genData(
    nmbEvents         = nmbEventsBkg,
    polarization      = polarization,
    inputData         = amplitudeSetBkg,
    efficiencyFormula = efficiencyFormula,
    outFileNamePrefix = f"{outputDirName}/",
    nameSuffix        = "Bkg",
    regenerateData    = regenerateData,
    additionalColDefs = (("discrVariable", "gRandom->Uniform(0, 2) - 1"), ),
  )

  # concatenate signal and background data frames vertically and define event weights
  signalRange = (-0.3, +0.3)
  sideBands   = ((-1, -0.4), (+0.4, +1))
  eventWeightFormula = \
  f"""
    if (({signalRange[0]} < discrVariable) and (discrVariable < {signalRange[1]}))
      return 1.0;
    else if (  (({sideBands[0][0]} < discrVariable) and (discrVariable < {sideBands[0][1]}))
        or (({sideBands[1][0]} < discrVariable) and (discrVariable < {sideBands[1][1]})))
      return -0.5;
    else
      return 0.0;
  """
  dataPwaModelTot = (
    ROOT.RDataFrame(treeName, (f"{outputDirName}/dataSig.root", f"{outputDirName}/dataBkg.root"))
        .Define("eventWeight", eventWeightFormula)
  )

  # plot discriminating variable
  histDef = (";Discriminating variable;Count / 0.02", 100, -1, +1)
  histDiscrSig = dataPwaModelSig.Histo1D(ROOT.RDF.TH1DModel("Signal",     *histDef), "discrVariable").GetValue()
  histDiscrBkg = dataPwaModelBkg.Histo1D(ROOT.RDF.TH1DModel("Background", *histDef), "discrVariable").GetValue()
  histDiscrTot = dataPwaModelTot.Histo1D(ROOT.RDF.TH1DModel("Total",      *histDef), "discrVariable").GetValue()
  histDiscrSig.SetLineColor(ROOT.kGreen + 2)
  histDiscrBkg.SetLineColor(ROOT.kRed   + 1)
  histDiscrSig.SetLineStyle(ROOT.kDashed)
  histDiscrBkg.SetLineStyle(ROOT.kDashed)
  histDiscrTot.SetLineWidth(2)
  histStack = ROOT.THStack("discrVariableSim", ";Discriminating variable;Count / 0.02")
  histStack.Add(histDiscrTot)
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
  canv.SaveAs(f"{outputDirName}/{histStack.GetName()}.pdf")
  hist = dataPwaModelTot.Histo1D(ROOT.RDF.TH1DModel("discrVariableSimSbSubtr", ";Discriminating variable;Count / 0.02", 100, -1, +1), "discrVariable", "eventWeight")
  hist.Draw()
  canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")

  # plot angular distributions of data generated from partial-wave amplitudes
  histBinning = (TH3_ANG_NMB_BINS, -1, +1, TH3_ANG_NMB_BINS, -180, +180, TH3_ANG_NMB_BINS, -180, +180)
  hists = (
    # total angular distribution
    dataPwaModelTot.Histo3D(ROOT.RDF.TH3DModel("data", TH3_ANG_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg"),
    (  # angular distribution in signal region
      dataPwaModelSig.Filter(f"({signalRange[0]} < discrVariable) and (discrVariable < {signalRange[1]})")
                     .Histo3D(ROOT.RDF.TH3DModel("dataSigRegion", TH3_ANG_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg")
    ),
    ( # angular distribution in side-band regions
      dataPwaModelBkg.Filter(f"(({sideBands[0][0]} < discrVariable) and (discrVariable < {sideBands[0][1]}))"
                          f"or (({sideBands[1][0]} < discrVariable) and (discrVariable < {sideBands[1][1]}))")
                     .Histo3D(ROOT.RDF.TH3DModel("dataSidebandRegion", TH3_ANG_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg")
    ),
    # angular distribution after side-band subtraction
    dataPwaModelTot.Histo3D(ROOT.RDF.TH3DModel("dataSbSubtr", TH3_ANG_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg", "eventWeight"),
  )
  for hist in hists:
    hist.SetMinimum(0)
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
    print(f"Integral of histogram '{hist.GetName()}' = {hist.Integral()}")
    canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")
  print(f"Sum of weights = {dataPwaModelTot.Sum('eventWeight').GetValue()}")

  return dataPwaModelTot, dataPwaModelSig, dataPwaModelBkg


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
    outputDirName         = Utilities.makeDirPath("./plotsTestPhotoProdWeighted")
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
    partialWaveAmplitudesSig: tuple[AmplitudeValue, ...] = (  # set of all possible partial waves up to ell = 2
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
    )
    amplitudeSetSig = AmplitudeSet(partialWaveAmplitudesSig)
    # define angular distribution of background
    partialWaveAmplitudesBkg: tuple[AmplitudeValue, ...] = (  # set of all possible partial waves up to ell = 2
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
    )
    amplitudeSetBkg = AmplitudeSet(partialWaveAmplitudesBkg)
    # formulas for detection efficiency
    # x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
    # efficiencyFormula = "1"  # acc_perfect
    efficiencyFormula = "(1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / 1.5**3"  # acc_1; even in all variables
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # calculate true moment values and generate data from partial-wave amplitudes
    with timer.timeThis("Time to generate MC data from partial waves"):
      HTruthSig: MomentResult = amplitudeSetSig.photoProdMomentResult(maxL, normalize = (True if normalizeMoments else nmbPwaMcEventsSig))
      HTruthBkg: MomentResult = amplitudeSetBkg.photoProdMomentResult(maxL, normalize = (True if normalizeMoments else nmbPwaMcEventsBkg))
      print(f"True moment values for signal:\n{HTruthSig}")
      print(f"True moment values for background:\n{HTruthBkg}")
      dataPwaModel, dataPwaModelSig, dataPwaModelBkg = genSigAndBkgDataFromWaves(
        nmbEventsSig      = nmbPwaMcEventsSig,
        nmbEventsBkg      = nmbPwaMcEventsBkg,
        amplitudeSetSig   = amplitudeSetSig,
        amplitudeSetBkg   = amplitudeSetBkg,
        polarization      = beamPolarization,
        outputDirName     = outputDirName,
        efficiencyFormula = efficiencyFormula,
        regenerateData    = True,
      )

    # generate accepted phase-space data
    with timer.timeThis("Time to generate phase-space MC data"):
      dataAcceptedPs = genAccepted2BodyPsPhotoProd(
        nmbEvents         = nmbAcceptedPsMcEvents,
        efficiencyFormula = efficiencyFormula,
        outFileNamePrefix = f"{outputDirName}/",
        regenerateData    = True,
      )

    # define input data
    data = dataPwaModel
    # data = dataPwaModelSig
    HTruth = HTruthSig
    # data = dataPwaModelBkg
    # HTruth = HTruthBkg

    # setup moment calculator
    momentIndices = MomentIndices(maxL)
    binVarMass    = KinematicBinningVariable(name = "mass", label = "#it{m}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    massBinning   = HistAxisBinning(nmbBins = 1, minVal = 1.0, maxVal = 2.0, _var = binVarMass)
    momentsInBins:      list[MomentCalculator] = []
    momentsInBinsTruth: list[MomentCalculator] = []
    for massBinCenter in massBinning:
      # dummy bins with identical data sets
      dataSet = DataSet(
        data           = data,
        phaseSpaceData = dataAcceptedPs,
        nmbGenEvents   = nmbAcceptedPsMcEvents,
        polarization   = beamPolarization,
      )  #TODO nmbAcceptedPsMcEvents is not correct number to normalize integral matrix
      momentsInBins.append(
        MomentCalculator(
          indicesMeas          = momentIndices,
          indicesPhys          = momentIndices,
          dataSet              = dataSet,
          integralFileBaseName = f"{outputDirName}/integralMatrix",
          binCenters           = {binVarMass : massBinCenter},
        )
      )
      # dummy truth values; identical for all bins
      momentsInBinsTruth.append(
        MomentCalculator(
          indicesMeas = momentIndices,
          indicesPhys = momentIndices,
          dataSet     = dataSet,
          binCenters  = {binVarMass : massBinCenter},
          _HPhys      = HTruth,
        )
      )
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
        plotComplexMatrix(moments[0].integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outputDirName}/I_acc_{label}")
        plotComplexMatrix(moments[0].integralMatrix.inverse,          pdfFileNamePrefix = f"{outputDirName}/I_inv_{label}")

    # calculate moments of data generated from partial-wave amplitudes
    with timer.timeThis(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads"):
      moments.calculateMoments(normalize = normalizeMoments, nmbBootstrapSamples = nmbBootstrapSamples)
      # print all moments for first kinematic bin
      print(f"Measured moments of data generated according to partial-wave amplitudes\n{moments[0].HMeas}")
      print(f"Physical moments of data generated according to partial-wave amplitudes\n{moments[0].HPhys}")
      # plot moments in each kinematic bin
      namePrefix = "norm" if normalizeMoments else "unnorm"
      label = binLabel(moments[0])
      plotMomentsInBin(moments[0].HPhys, normalizeMoments, HTruth, outFileNamePrefix = f"{outputDirName}/{namePrefix}_{label}_")
      if nmbBootstrapSamples > 0:
        plotMomentsBootstrapDistributions1D(moments[0].HPhys, HTruth, outFileNamePrefix = f"{outputDirName}/{namePrefix}_{label}_")
        plotMomentsBootstrapDiffInBin      (moments[0].HPhys,         outFileNamePrefix = f"{outputDirName}/{namePrefix}_{label}_")

    timer.stop("Total execution time")
    print(timer.summary)
