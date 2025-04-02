#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3


from __future__ import annotations

import functools
import numpy as np
import os
import threadpoolctl
from typing import TypedDict

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
  drawTF3,
  HistAxisBinning,
  plotComplexMatrix,
  plotMoments1D,
  plotMomentsInBin,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# default TH3 plotting options for angular distributions
TH3_ANG_NMB_BINS = 25
TH3_ANG_BINNINGS = (
  HistAxisBinning(TH3_ANG_NMB_BINS,   -1,   +1),
  HistAxisBinning(TH3_ANG_NMB_BINS, -180, +180),
  HistAxisBinning(TH3_ANG_NMB_BINS, -180, +180),
)
TH3_ANG_TITLE = ";cos#theta;#phi [deg];#Phi [deg]"
class Th3PlotKwargsType(TypedDict):
  binnings:  tuple[HistAxisBinning, HistAxisBinning, HistAxisBinning]
  histTitle: str
TH3_ANG_PLOT_KWARGS: Th3PlotKwargsType = {"histTitle" : TH3_ANG_TITLE, "binnings" : TH3_ANG_BINNINGS}


def genDataFromWaves(
  nmbEvents:         int,                 # number of events to generate
  polarization:      float,               # photon-beam polarization
  amplitudeSet:      AmplitudeSet,        # partial-wave amplitudes
  efficiencyFormula: str | None = None,   # detection efficiency used to generate data
  regenerateData:    bool       = False,  # if set data are regenerated although .root file exists
  outFileNamePrefix: str        = "./",   # name prefix for output files
  nameSuffix:        str        = "",     # suffix for functions and file names
) -> ROOT.RDataFrame:
  """Generates data according to set of partial-wave amplitudes (assuming rank 1) and given detection efficiency"""
  print(f"Generating {nmbEvents} events distributed according to PWA model {amplitudeSet} with photon-beam polarization {polarization} weighted by efficiency {efficiencyFormula}")

  # construct and draw efficiency function
  efficiencyFcn = ROOT.TF3(f"efficiencyGen{nameSuffix}", efficiencyFormula if efficiencyFormula else "1", -1, +1, -180, +180, -180, +180)
  drawTF3(efficiencyFcn, **TH3_ANG_PLOT_KWARGS, nmbPoints = 100, maxVal = 1.0,
    pdfFileName = f"{outFileNamePrefix}{efficiencyFcn.GetName()}.pdf")

  # construct TF3 for intensity distribution in Eq. (171)
  # x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
  intensityFormula = amplitudeSet.intensityFormula(
    polarization = polarization,
    thetaFormula = "std::acos(x)",
    phiFormula   = "TMath::DegToRad() * y",
    PhiFormula   = "TMath::DegToRad() * z",
    printFormula = True,
  )
  if efficiencyFormula:
    intensityFormula = f"{intensityFormula} * ({efficiencyFormula})"
  intensityFcn = ROOT.TF3(f"intensity{nameSuffix}", intensityFormula, -1, +1, -180, +180, -180, +180)
  # intensityFcn.SetTitle(";cos#theta;#phi [deg];#Phi [deg]")
  intensityFcn.SetNpx(100)  # used in numeric integration performed by GetRandom()
  intensityFcn.SetNpy(100)
  intensityFcn.SetNpz(100)
  intensityFcn.SetMinimum(0)
  drawTF3(intensityFcn, **TH3_ANG_PLOT_KWARGS, pdfFileName = f"{outFileNamePrefix}{intensityFcn.GetName()}.pdf")
  #TODO check for negative intensity values for wave set containing only P_+1^+ wave

  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = f"{outFileNamePrefix}data{nameSuffix}.root"
  if os.path.exists(fileName) and not regenerateData:
    print(f"Reading partial-wave MC data from '{fileName}'")
    df = ROOT.RDataFrame(treeName, fileName)
    nmbEvents = df.Count().GetValue()
    print(f"File '{fileName}' contains {nmbEvents} events")
    return df
  print(f"Generating MC events according to partial-wave model and writing them to '{fileName}'")
  RootUtilities.declareInCpp(**{intensityFcn.GetName() : intensityFcn})  # use Python object in C++
  dataPointFcn = f"""
    double cosTheta, phiDeg, PhiDeg;
    PyVars::{intensityFcn.GetName()}.GetRandom3(cosTheta, phiDeg, PhiDeg);
    std::vector<double> dataPoint = {{cosTheta, phiDeg, PhiDeg}};
    return dataPoint;
  """  # C++ code that throws random point in angular space
  df = (
    ROOT.RDataFrame(nmbEvents)
        .Define("dataPoint", dataPointFcn)
        .Define("cosTheta",  "dataPoint[0]")
        .Define("theta",     "std::acos(cosTheta)")
        .Define("phiDeg",    "dataPoint[1]")
        .Define("phi",       "TMath::DegToRad() * phiDeg")
        .Define("PhiDeg",    "dataPoint[2]")
        .Define("Phi",       "TMath::DegToRad() * PhiDeg")
        .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genDataFromWaves()" << endl; } return true;')  # the noop filter that logs when event loop is running
        .Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](["cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi"]))  # snapshot is needed or else the `dataPoint` column would be regenerated for every triggered loop
  )  #!NOTE! for some reason, this is very slow
  return df


def genAccepted2BodyPsPhotoProd(
  nmbEvents:         int,                 # number of events to generate
  efficiencyFormula: str | None = None,   # detection efficiency used for acceptance correction
  regenerateData:    bool       = False,  # if set data are regenerated although .root file exists
  outFileNamePrefix: str        = "./",   # name prefix for output files
) -> ROOT.RDataFrame:
  """Generates RDataFrame with two-body phase-space distribution weighted by given detection efficiency"""
  print(f"Generating {nmbEvents} events distributed according to two-body phase-space weighted by efficiency {efficiencyFormula}")
  # construct and draw efficiency function
  efficiencyFcn = ROOT.TF3("efficiencyReco", efficiencyFormula if efficiencyFormula else "1", -1, +1, -180, +180, -180, +180)
  drawTF3(efficiencyFcn, **TH3_ANG_PLOT_KWARGS, pdfFileName = f"{outFileNamePrefix}hEfficiencyReco.pdf", nmbPoints = 100, maxVal = 1.0)

  # generate isotropic distributions in cos theta, phi, and Phi and weight with efficiency function
  treeName = "data"
  fileName = f"{outFileNamePrefix}{efficiencyFcn.GetName()}.root"
  if os.path.exists(fileName) and not regenerateData:
    print(f"Reading accepted phase-space MC data from '{fileName}'")
    return ROOT.RDataFrame(treeName, fileName)
  print(f"Generating accepted phase-space MC data and writing them to '{fileName}'")
  #TODO avoid code doubling with genDataFromWaves() and corresponding function in testMomentsAlex
  RootUtilities.declareInCpp(efficiencyFcn = efficiencyFcn)  # use Python object in C++
  pointFcn = """
    double cosTheta, phiDeg, PhiDeg;
    PyVars::efficiencyFcn.GetRandom3(cosTheta, phiDeg, PhiDeg);
    std::vector<double> point = {cosTheta, phiDeg, PhiDeg};
    return point;
  """  # C++ code that throws random point in angular space
  df = (
    ROOT.RDataFrame(nmbEvents)
        .Define("point",    pointFcn)
        .Define("cosTheta", "point[0]")
        .Define("theta",    "std::acos(cosTheta)")
        .Define("phiDeg",   "point[1]")
        .Define("phi",      "TMath::DegToRad() * phiDeg")
        .Define("PhiDeg",   "point[2]")
        .Define("Phi",      "TMath::DegToRad() * PhiDeg")
        .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genData2BodyPsPhotoProd()" << endl; } return true;')  # noop filter that logs when event loop is running
        # .Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](["theta", "phi", "Phi"]))
        .Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](["cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi"]))  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
  )
  return df


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  ROOT.gRandom.SetSeed(1234567890)
  # ROOT.EnableImplicitMT(10)
  setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 5):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of test case
    outputDirName    = Utilities.makeDirPath("./plotsTestPhotoProd")
    nmbPwaMcEvents   = 1000
    nmbPsMcEvents    = 1000000
    beamPolarization = 1.0
    maxL             = 5  # define maximum L quantum number of moments
    partialWaveAmplitudes = [  # set of all possible waves up to ell = 2
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
    amplitudeSet = AmplitudeSet(partialWaveAmplitudes)
    # formulas for detection efficiency
    # x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
    # efficiencyFormulaGen = "1"  # acc_perfect
    efficiencyFormulaGen = "(1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / 1.5**3"  # acc_1; even in all variables
    # efficiencyFormulaGen = "(0.75 + 0.25 * x) * (0.75 + 0.25 * (y / 180)) * (0.75 + 0.25 * (z / 180))"  # acc_2; odd in all variables
    # efficiencyFormulaGen = "(0.6 + 0.4 * x) * (0.6 + 0.4 * (y / 180)) * (0.6 + 0.4 * (z / 180))"  # acc_3; odd in all variables
    # detune efficiency used to correct acceptance w.r.t. the one used to generate the data
    efficiencyFormulaDetune = ""
    # efficiencyFormulaDetune = "(0.35 + 0.15 * x) * (0.35 + 0.15 * (y / 180)) * (0.35 + 0.15 * (z / 180))"  # detune_odd; detune by odd terms
    # efficiencyFormulaDetune = "0.1 * (1.5 - y * y / (180 * 180)) / 1.5"  # detune_even; detune by even terms in phi only
    # efficiencyFormulaDetune = "0.1 * (1.5 - x * x) * (1.5 - z * z / (180 * 180)) / (1.5**2)"  # detune_even; detune by even terms in cos(theta) and Phi
    # efficiencyFormulaDetune = "0.1 * (1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / (1.5**3)"  # detune_even; detune by even terms in all variables
    if efficiencyFormulaDetune:
      efficiencyFcnDetune = ROOT.TF3("efficiencyDetune", efficiencyFormulaDetune, -1, +1, -180, +180, -180, +180)
      drawTF3(efficiencyFcnDetune, **TH3_ANG_PLOT_KWARGS, pdfFileName = f"{outputDirName}/hEfficiencyDetune.pdf", nmbPoints = 100, maxVal = 1.0)
      efficiencyFormulaReco = f"{efficiencyFormulaGen} + {efficiencyFormulaDetune}"
    else:
      efficiencyFormulaReco = efficiencyFormulaGen
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # calculate true moment values and generate data from partial-wave amplitudes
    t = timer.start("Time to generate MC data from partial waves")
    HTruth: MomentResult = amplitudeSet.photoProdMomentSet(maxL)
    print(f"True moment values\n{HTruth}")
    dataPwaModel = genDataFromWaves(nmbPwaMcEvents, beamPolarization, amplitudeSet, efficiencyFormulaGen, outFileNamePrefix = f"{outputDirName}/", regenerateData = True)
    t.stop()

    # plot data generated from partial-wave amplitudes
    canv = ROOT.TCanvas()
    nmbBins = 25
    hist = dataPwaModel.Histo3D(
      ROOT.RDF.TH3DModel("hData", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
      "cosTheta", "phiDeg", "PhiDeg")
    hist.SetMinimum(0)
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
    canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")

    # generate accepted phase-space data
    t = timer.start("Time to generate phase-space MC data")
    dataAcceptedPs = genAccepted2BodyPsPhotoProd(nmbPsMcEvents, efficiencyFormulaReco, outFileNamePrefix = f"{outputDirName}/", regenerateData = True)
    t.stop()

    # setup moment calculator
    momentIndices = MomentIndices(maxL)
    binVarMass    = KinematicBinningVariable(name = "mass", label = "#it{m}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    massBinning   = HistAxisBinning(nmbBins = 2, minVal = 1.0, maxVal = 2.0, _var = binVarMass)
    momentsInBins:      list[MomentCalculator] = []
    momentsInBinsTruth: list[MomentCalculator] = []
    for massBinCenter in massBinning:
      # dummy bins with identical data sets
      dataSet = DataSet(dataPwaModel, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbPsMcEvents, polarization = beamPolarization)  #TODO nmbPsMcEvents is not correct number to normalize integral matrix
      momentsInBins.append(MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outputDirName}/integralMatrix", binCenters = {binVarMass : massBinCenter}))
      # dummy truth values; identical for all bins
      momentsInBinsTruth.append(MomentCalculator(momentIndices, dataSet, binCenters = {binVarMass : massBinCenter}, _HPhys = HTruth))
    moments      = MomentCalculatorsKinematicBinning(momentsInBins)
    momentsTruth = MomentCalculatorsKinematicBinning(momentsInBinsTruth)

    # calculate integral matrices
    t = timer.start(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")
    moments.calculateIntegralMatrices(forceCalculation = True)
    # print acceptance integral matrix for first kinematic bin
    print(f"Acceptance integral matrix for first bin\n{moments[0].integralMatrix}")
    eigenVals, _ = moments[0].integralMatrix.eigenDecomp
    print(f"Eigenvalues of acceptance integral matrix for first bin\n{np.sort(eigenVals)}")
    # plot acceptance integral matrices for all kinematic bins
    for HData in moments:
      label = binLabel(HData)
      plotComplexMatrix(moments[0].integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outputDirName}/I_acc_{label}")
      plotComplexMatrix(moments[0].integralMatrix.inverse,          pdfFileNamePrefix = f"{outputDirName}/I_inv_{label}")
    t.stop()

    # calculate moments of data generated from partial-wave amplitudes
    t = timer.start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
    moments.calculateMoments()
    # print all moments for first kinematic bin
    print(f"Measured moments of data generated according to partial-wave amplitudes for first kinematic bin\n{moments[0].HMeas}")
    print(f"Physical moments of data generated according to partial-wave amplitudes for first kinematic bin\n{moments[0].HPhys}")
    # plot moments in each kinematic bin
    for HData in moments:
      label = binLabel(HData)
      plotMomentsInBin(HData = moments[0].HPhys, HTruth = HTruth, outFileNamePrefix = f"{outputDirName}/h{label}_")
    # plot kinematic dependences of all moments #TODO normalize H_0(0, 0) to total number of events
    for qnIndex in momentIndices.qnIndices:
      plotMoments1D(
        momentResults     = moments.momentResultsPhys,
        qnIndex           = qnIndex,
        binning           = massBinning,
        normalizedMoments = True,
        momentResultsTrue = momentsTruth.momentResultsPhys,
        outFileNamePrefix = f"{outputDirName}/h",
        histTitle         = qnIndex.title,
      )
    t.stop()

    timer.stop("Total execution time")
    print(timer.summary)
