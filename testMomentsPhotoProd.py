#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3

import ctypes
import functools
import numpy as np
import os
import subprocess
import threadpoolctl
from typing import (
  List,
  Optional,
  Tuple,
  TypedDict,
)

import ROOT

import MomentCalculator
import OpenMpUtilities
import PlottingUtilities
import RootUtilities
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# default TH3 plotting options
TH3_NMB_BINS = 25
TH3_BINNINGS = (
  PlottingUtilities.HistAxisBinning(TH3_NMB_BINS,   -1,   +1),
  PlottingUtilities.HistAxisBinning(TH3_NMB_BINS, -180, +180),
  PlottingUtilities.HistAxisBinning(TH3_NMB_BINS, -180, +180),
)
TH3_TITLE = ";cos#theta;#phi [deg];#Phi [deg]"
class Th3PlotKwargsType(TypedDict):
  binnings:  Tuple[PlottingUtilities.HistAxisBinning, PlottingUtilities.HistAxisBinning, PlottingUtilities.HistAxisBinning]
  histTitle: str
TH3_PLOT_KWARGS: Th3PlotKwargsType = {"histTitle" : TH3_TITLE, "binnings" : TH3_BINNINGS}


def genDataFromWaves(
  nmbEvents:         int,                            # number of events to generate
  polarization:      float,                          # photon-beam polarization
  amplitudeSet:      MomentCalculator.AmplitudeSet,  # partial-wave amplitudes
  efficiencyFormula: Optional[str] = None,           # detection efficiency used to generate data
  regenerateData:    bool = False,                   # if set data are regenerated although .root file exists
  outFileNamePrefix: str = "./",                     # name prefix for output files
  nameSuffix:        str = "",                       # suffix for functions and file names
) -> ROOT.RDataFrame:
  """Generates data according to set of partial-wave amplitudes (assuming rank 1) and given detection efficiency"""
  print(f"Generating {nmbEvents} events distributed according to PWA model {amplitudeSet} with photon-beam polarization {polarization} weighted by efficiency {efficiencyFormula}")
  # construct and draw efficiency function
  efficiencyFcn = ROOT.TF3(f"efficiencyGen{nameSuffix}", efficiencyFormula if efficiencyFormula else "1", -1, +1, -180, +180, -180, +180)
  PlottingUtilities.drawTF3(efficiencyFcn, **TH3_PLOT_KWARGS, nmbPoints = 100, maxVal = 1.0,
    pdfFileName = f"{outFileNamePrefix}{efficiencyFcn.GetName()}.pdf")

  # construct TF3 for intensity distribution in Eq. (153)
  # x = cos(theta) in [-1, +1], y = phi in [-180, +180] deg, z = Phi in [-180, +180] deg
  intensityComponentTerms: List[Tuple[str, str, str]] = []  # terms in sum of each intensity component
  for refl in (-1, +1):
    for amp1 in amplitudeSet.amplitudes(onlyRefl = refl):
      l1 = amp1.qn.l
      m1 = amp1.qn.m
      decayAmp1 = f"Ylm({l1}, {m1}, std::acos(x), TMath::DegToRad() * y)"
      for amp2 in amplitudeSet.amplitudes(onlyRefl = refl):
        l2 = amp2.qn.l
        m2 = amp2.qn.m
        decayAmp2 = f"Ylm({l2}, {m2}, std::acos(x), TMath::DegToRad() * y)"
        rhos: Tuple[complex, complex, complex] = amplitudeSet.photoProdSpinDensElements(refl, l1, l2, m1, m2)
        terms = tuple(
          f"{decayAmp1} * complexT({rho.real}, {rho.imag}) * std::conj({decayAmp2})"  # Eq. (153)
          if rho != 0 else "" for rho in rhos
        )
        intensityComponentTerms.append((terms[0], terms[1], terms[2]))
  # sum terms for each intensity component
  intensityComponentsFormula = []
  for iComponent in range(3):
    intensityComponentsFormula.append(f"({' + '.join(filter(None, (term[iComponent] for term in intensityComponentTerms)))})")
  # sum intensity components
  intensityFormula = (
    f"std::real({intensityComponentsFormula[0]} "
    f"- {intensityComponentsFormula[1]} * {polarization} * std::cos(2 * TMath::DegToRad() * z) "
    f"- {intensityComponentsFormula[2]} * {polarization} * std::sin(2 * TMath::DegToRad() * z))"
    + (f" * ({efficiencyFormula})" if efficiencyFormula else ""))  # Eq. (163)
  print(f"Intensity formula = {intensityFormula}")
  intensityFcn = ROOT.TF3(f"intensity{nameSuffix}", intensityFormula, -1, +1, -180, +180, -180, +180)
  intensityFcn.SetTitle(";cos#theta;#phi [deg];#Phi [deg]")
  intensityFcn.SetNpx(100)  # used in numeric integration performed by GetRandom()
  intensityFcn.SetNpy(100)
  intensityFcn.SetNpz(100)
  intensityFcn.SetMinimum(0)
  PlottingUtilities.drawTF3(intensityFcn, **TH3_PLOT_KWARGS, pdfFileName = f"{outFileNamePrefix}{intensityFcn.GetName()}.pdf")
  #TODO check for negative intensity values for wave set containing only P_+1^+ wave

  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = f"{outFileNamePrefix}{intensityFcn.GetName()}.photoProd.root"
  if os.path.exists(fileName) and not regenerateData:
    print(f"Reading partial-wave MC data from '{fileName}'")
    return ROOT.RDataFrame(treeName, fileName)
  print(f"Generating partial-wave MC data and writing them to '{fileName}'")
  RootUtilities.declareInCpp(**{intensityFcn.GetName() : intensityFcn})  # use Python object in C++
  pointFunc = f"""
    double cosTheta, phiDeg, PhiDeg;
    PyVars::{intensityFcn.GetName()}.GetRandom3(cosTheta, phiDeg, PhiDeg);
    std::vector<double> point = {{cosTheta, phiDeg, PhiDeg}};
    return point;
  """  # C++ code that throws random point in angular space
  df = ROOT.RDataFrame(nmbEvents) \
           .Define("point",    pointFunc) \
           .Define("cosTheta", "point[0]") \
           .Define("theta",    "std::acos(cosTheta)") \
           .Define("phiDeg",   "point[1]") \
           .Define("phi",      "TMath::DegToRad() * phiDeg") \
           .Define("PhiDeg",   "point[2]") \
           .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
           .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genDataFromWaves()" << endl; } return true;') \
           .Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](["cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi"]))
    # snapshot is needed or else the `point` column would be regenerated for every triggered loop
    # noop filter before snapshot logs when event loop is running
    # !Note! for some reason, this is very slow
  return df


def genAccepted2BodyPsPhotoProd(
  nmbEvents:         int,                   # number of events to generate
  efficiencyFormula: Optional[str] = None,  # detection efficiency used for acceptance correction
  regenerateData:    bool = False,          # if set data are regenerated although .root file exists
  outFileNamePrefix: str = "./",            # name prefix for output files
) -> ROOT.RDataFrame:
  """Generates RDataFrame with two-body phase-space distribution weighted by given detection efficiency"""
  print(f"Generating {nmbEvents} events distributed according to two-body phase-space weighted by efficiency {efficiencyFormula}")
  # construct and draw efficiency function
  efficiencyFcn = ROOT.TF3("efficiencyReco", efficiencyFormula if efficiencyFormula else "1", -1, +1, -180, +180, -180, +180)
  PlottingUtilities.drawTF3(efficiencyFcn, **TH3_PLOT_KWARGS, pdfFileName = f"{outFileNamePrefix}hEfficiencyReco.pdf", nmbPoints = 100, maxVal = 1.0)

  # generate isotropic distributions in cos theta, phi, and Phi and weight with efficiency function
  treeName = "data"
  fileName = f"{outFileNamePrefix}{efficiencyFcn.GetName()}.photoProd.root"
  if os.path.exists(fileName) and not regenerateData:
    print(f"Reading accepted phase-space MC data from '{fileName}'")
    return ROOT.RDataFrame(treeName, fileName)
  print(f"Generating accepted phase-space MC data and writing them to '{fileName}'")
  #TODO avoid code doubling with genDataFromWaves() and corresponding function in testMomentsAlex
  RootUtilities.declareInCpp(efficiencyFcn = efficiencyFcn)
  pointFunc = """
    double cosTheta, phiDeg, PhiDeg;
    PyVars::efficiencyFcn.GetRandom3(cosTheta, phiDeg, PhiDeg);
    std::vector<double> point = {cosTheta, phiDeg, PhiDeg};
    return point;
  """  # C++ code that throws random point in angular space
  df = ROOT.RDataFrame(nmbEvents) \
           .Define("point",    pointFunc) \
           .Define("cosTheta", "point[0]") \
           .Define("theta",    "std::acos(cosTheta)") \
           .Define("phiDeg",   "point[1]") \
           .Define("phi",      "TMath::DegToRad() * phiDeg") \
           .Define("PhiDeg",   "point[2]") \
           .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
           .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genData2BodyPSPhotoProd()" << endl; } return true;') \
           .Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](["theta", "phi", "Phi"]))
    # snapshot is needed or else the `point` column would be regenerated for every triggered loop
    # noop filter before snapshot logs when event loop is running
  return df


if __name__ == "__main__":
  Utilities.printGitInfo()
  ROOT.gROOT.SetBatch(True)
  ROOT.gRandom.SetSeed(1234567890)
  # ROOT.EnableImplicitMT(10)
  PlottingUtilities.setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 5):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    ROOT.gBenchmark.Start("Total execution time")

    # set parameters of test case
    outFileDirName   = "./plotsPhotoProd"
    nmbPwaMcEvents   = 1000
    nmbPsMcEvents    = 1000000
    beamPolarization = 1.0
    maxL             = 5  # define maximum L quantum number of moments
    partialWaveAmplitudes = [  # set of all possible waves up to ell = 2
      # negative-reflectivity waves
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.4 + 0.1j),  # P_-1^-
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m =  0), val =  0.3 - 0.8j),  # P_0^-
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.8 + 0.7j),  # P_+1^-
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -2), val =  0.1 - 0.4j),  # D_-2^-
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -1), val =  0.5 + 0.2j),  # D_-1^-
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m =  0), val = -0.1 - 0.2j),  # D_ 0^-
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +1), val =  0.2 - 0.1j),  # D_+1^-
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.2 + 0.3j),  # D_+2^-
      # positive-reflectivity waves
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = -1), val =  0.5 - 0.1j),  # P_-1^+
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.8 - 0.3j),  # P_0^+
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.6 + 0.3j),  # P_+1^+
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.2 + 0.1j),  # D_-2^+
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -1), val =  0.2 - 0.3j),  # D_-1^+
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m =  0), val =  0.1 - 0.2j),  # D_ 0^+
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +1), val =  0.2 + 0.5j),  # D_+1^+
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +2), val = -0.3 - 0.1j),  # D_+2^+
    ]
    amplitudeSet = MomentCalculator.AmplitudeSet(partialWaveAmplitudes)
    # formulas for detection efficiency
    # x = cos(theta) in [-1, +1], y = phi in [-180, +180] deg, z = Phi in [-180, +180] deg
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
      PlottingUtilities.drawTF3(efficiencyFcnDetune, **TH3_PLOT_KWARGS, pdfFileName = f"{outFileDirName}/hEfficiencyDetune.pdf", nmbPoints = 100, maxVal = 1.0)
      efficiencyFormulaReco = f"{efficiencyFormulaGen} + {efficiencyFormulaDetune}"
    else:
      efficiencyFormulaReco = efficiencyFormulaGen
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # calculate true moment values and generate data from partial-wave amplitudes
    ROOT.gBenchmark.Start("Time to generate MC data from partial waves")
    HTrue: MomentCalculator.MomentResult = amplitudeSet.photoProdMomentSet(maxL)
    print(f"True moment values\n{HTrue}")
    dataPwaModel = genDataFromWaves(nmbPwaMcEvents, beamPolarization, amplitudeSet, efficiencyFormulaGen, outFileNamePrefix = f"{outFileDirName}/", regenerateData = True)
    ROOT.gBenchmark.Stop("Time to generate MC data from partial waves")

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
    canv.SaveAs(f"{outFileDirName}/{hist.GetName()}.pdf")

    # generate accepted phase-space data
    ROOT.gBenchmark.Start("Time to generate phase-space MC data")
    dataAcceptedPs = genAccepted2BodyPsPhotoProd(nmbPsMcEvents, efficiencyFormulaReco, outFileNamePrefix = f"{outFileDirName}/", regenerateData = True)
    ROOT.gBenchmark.Stop("Time to generate phase-space MC data")

    # setup moment calculator
    momentIndices = MomentCalculator.MomentIndices(maxL)
    binVarMass = MomentCalculator.KinematicBinningVariable(name = "mass", label = "#it{m}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    massBinning = PlottingUtilities.HistAxisBinning(nmbBins = 2, minVal = 1.0, maxVal = 2.0, _var = binVarMass)
    momentsInBins:      List[MomentCalculator.MomentCalculator] = []
    momentsInBinsTruth: List[MomentCalculator.MomentCalculator] = []
    for massBinCenter in massBinning:
      # dummy bins with identical data sets
      dataSet = MomentCalculator.DataSet(beamPolarization, dataPwaModel, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbPsMcEvents)  #TODO nmbPsMcEvents is not correct number to normalize integral matrix
      momentsInBins.append(MomentCalculator.MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix", _binCenters = {binVarMass : massBinCenter}))
      # dummy truth values; identical for all bins
      momentsInBinsTruth.append(MomentCalculator.MomentCalculator(momentIndices, dataSet, _binCenters = {binVarMass : massBinCenter}, _HPhys = HTrue))
    moments      = MomentCalculator.MomentCalculatorsKinematicBinning(momentsInBins)
    momentsTruth = MomentCalculator.MomentCalculatorsKinematicBinning(momentsInBinsTruth)

    # calculate integral matrix
    ROOT.gBenchmark.Start(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")
    moments.calculateIntegralMatrices(forceCalculation = True)
    # print acceptance integral matrix for first kinematic bin
    print(f"Acceptance integral matrix\n{moments[0].integralMatrix}")
    eigenVals, _ = moments[0].integralMatrix.eigenDecomp
    print(f"Eigenvalues of acceptance integral matrix\n{np.sort(eigenVals)}")
    # plot acceptance integral matrices for all kinematic bins
    for HData in moments:
      binLabel = "_".join(HData.fileNameBinLabels)
      PlottingUtilities.plotComplexMatrix(moments[0].integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/I_acc_{binLabel}")
      PlottingUtilities.plotComplexMatrix(moments[0].integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/I_inv_{binLabel}")
    ROOT.gBenchmark.Stop(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")

    # calculate moments of data generated from partial-wave amplitudes
    ROOT.gBenchmark.Start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
    moments.calculateMoments()
    # print all moments for first kinematic bin
    print(f"Measured moments of data generated according to partial-wave amplitudes\n{moments[0].HMeas}")
    print(f"Physical moments of data generated according to partial-wave amplitudes\n{moments[0].HPhys}")
    # plot moments in each kinematic bin
    for HData in moments:
      binLabel = "_".join(HData.fileNameBinLabels)
      PlottingUtilities.plotMomentsInBin(HData = moments[0].HPhys, HTrue = HTrue, pdfFileNamePrefix = f"{outFileDirName}/h{binLabel}_")
    # plot kinematic dependences of all moments #TODO normalize H_0(0, 0) to total number of events
    for qnIndex in momentIndices.QnIndices():
      PlottingUtilities.plotMoments1D(moments, qnIndex, massBinning, momentsTruth, pdfFileNamePrefix = f"{outFileDirName}/h")
    ROOT.gBenchmark.Stop(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")

    ROOT.gBenchmark.Stop("Total execution time")
    _ = ctypes.c_float(0.0)  # dummy argument required by ROOT; sigh
    ROOT.gBenchmark.Summary(_, _)
    print("!Note! the 'TOTAL' time above is wrong; ignore")
