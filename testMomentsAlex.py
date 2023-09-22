#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3

import ctypes
import functools
import numpy as np
import os
import subprocess
from typing import (
  List,
  Optional,
  Tuple,
)

import ROOT

import MomentCalculator
import OpenMp
import PlottingUtilities
import RootUtilities
import testMomentsPhotoProd


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def printGitInfo() -> None:
  """Prints directory of this file and git hash in this directory"""
  repoDir = os.path.dirname(os.path.abspath(__file__))
  gitInfo = subprocess.check_output(["git", "describe", "--always"], cwd = repoDir).strip().decode()
  print(f"Running code in '{repoDir}', git version '{gitInfo}'")


def genDataFromWaves(
  nmbEvents:         int,                            # number of events to generate
  polarization:      float,                          # photon-beam polarization
  amplitudeSet:      MomentCalculator.AmplitudeSet,  # partial-wave amplitudes
  efficiencyHist:    Optional[ROOT.TH3D] = None,     # detection efficiency used to generate data
  regenerateData:    bool = False,                   # if set data are regenerated although .root file exists
  pdfFileNamePrefix: str = "./",                     # name prefix for output files
) -> ROOT.RDataFrame:
  """Generates data according to set of partial-wave amplitudes (assuming rank 1) and given detection efficiency"""
  print(f"Generating {nmbEvents} events distributed according to PWA model {amplitudeSet} with photon-beam polarization {polarization} weighted by efficiency histogram {efficiencyHist}")

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
        #TODO leave out terms when rho == 0
        terms = tuple(f"{decayAmp1} * complexT({rho.real}, {rho.imag}) * std::conj({decayAmp2})" for rho in rhos)  # Eq. (153)
        intensityComponentTerms.append((terms[0], terms[1], terms[2]))
  # sum terms for each intensity component
  intensityComponentsFormula = []
  for iComponent in range(3):
    intensityComponentsFormula.append(f"({' + '.join([term[iComponent] for term in intensityComponentTerms])})")
  # sum intensity components
  intensityFormula = (
    f"std::real({intensityComponentsFormula[0]} "
    f"- {intensityComponentsFormula[1]} * {polarization} * std::cos(2 * TMath::DegToRad() * z) "
    f"- {intensityComponentsFormula[2]} * {polarization} * std::sin(2 * TMath::DegToRad() * z))")  # Eq. (163)
  print(f"Intensity formula = {intensityFormula}")
  intensityFcn = ROOT.TF3("intensity", intensityFormula, -1, +1, -180, +180, -180, +180)
  intensityFcn.SetTitle(";cos#theta;#phi [deg];#Phi [deg]")
  intensityFcn.SetNpx(100)  # used in numeric integration performed by GetRandom()
  intensityFcn.SetNpy(100)
  intensityFcn.SetNpz(100)
  intensityFcn.SetMinimum(0)
  PlottingUtilities.drawTF3(intensityFcn, **testMomentsPhotoProd.TH3_PLOT_KWARGS, pdfFileName = f"{pdfFileNamePrefix}hIntensity.pdf")

  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = f"{intensityFcn.GetName()}.photoProd.root"
  if os.path.exists(fileName) and not regenerateData:
    print(f"Reading partial-wave MC data from '{fileName}'")
    return ROOT.RDataFrame(treeName, fileName)
  print(f"Generating partial-wave MC data and writing them to '{fileName}'")
  df = ROOT.RDataFrame(nmbEvents)
  RootUtilities.declareInCpp(intensityFcn = intensityFcn)  # use Python object in C++
  pointFunc = """
    double cosTheta, phiDeg, PhiDeg;
    PyVars::intensityFcn.GetRandom3(cosTheta, phiDeg, PhiDeg);
    std::vector<double> point = {cosTheta, phiDeg, PhiDeg};
    return point;
  """
  df.Define("point",    pointFunc) \
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
  return ROOT.RDataFrame(treeName, fileName)


if __name__ == "__main__":
  printGitInfo()
  OpenMp.setNmbOpenMpThreads(5)
  ROOT.gROOT.SetBatch(True)
  PlottingUtilities.setupPlotStyle()
  ROOT.gBenchmark.Start("Total execution time")

  # set parameters of test case
  plotDirName = "./plotsAlex"
  treeName = "ntFSGlueX_100_110_angles"
  signalFileName = "./Alex/tree_pippim__B4_gen_amp_030994.signal.root.angles"
  nmbSignalEvents = 218240
  acceptedPsFileName = "./Alex/tree_pippim__B4_gen_amp_030994.phaseSpace.root.angles"
  nmbAcceptedPsEvents = 210236  #TODO not correct number to normalize integral matrix
  # beamPolarization = 0.4  #TODO read from tree
  beamPolarization = 1.0  #TODO read from tree
  maxL = 5  # define maximum L quantum number of moments
  nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

  # calculate true moments
  partialWaveAmplitudes = [
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = +1), val = 1 + 0j),  # P_+1^+
  ]
  amplitudeSet = MomentCalculator.AmplitudeSet(partialWaveAmplitudes)
  HTrue: MomentCalculator.MomentResult = amplitudeSet.photoProdMomentSet(maxL)
  print(f"True moment values\n{HTrue}")

  # load data
  # print(f"Loading signal data from tree '{treeName}' in file '{signalFileName}'")
  # dataSignal = ROOT.RDataFrame(treeName, signalFileName)
  print(f"Loading accpepted phase-space data from tree '{treeName}' in file '{acceptedPsFileName}'")
  dataAcceptedPs = ROOT.RDataFrame(treeName, acceptedPsFileName)

  nmbBins = 25
  # plot signal and phase-space data
  hists = (
    # dataSignal.Histo3D(
    #   ROOT.RDF.TH3DModel("hSignal", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
    #   "cosTheta", "phiDeg", "PhiDeg"),
    dataAcceptedPs.Histo3D(
      ROOT.RDF.TH3DModel("hPhaseSpace", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
      "cosTheta", "phiDeg", "PhiDeg"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
    canv.SaveAs(f"{plotDirName}/{hist.GetName()}.pdf")

  print(f"Generating signal data")
  dataSignal = genDataFromWaves(nmbSignalEvents, beamPolarization, amplitudeSet, hists[0], pdfFileNamePrefix = f"{plotDirName}/", regenerateData = True)

  # setup moment calculator
  momentIndices = MomentCalculator.MomentIndices(maxL)
  # dataSet = MomentCalculator.DataSet(beamPolarization, dataSignal, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbAcceptedPsEvents)
  dataSet = MomentCalculator.DataSet(beamPolarization, dataSignal, phaseSpaceData = None, nmbGenEvents = nmbAcceptedPsEvents)
  momentCalculator = MomentCalculator.MomentCalculator(momentIndices, dataSet)

  # # calculate integral matrix
  # ROOT.gBenchmark.Start(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")
  # momentCalculator.calculateIntegralMatrix(forceCalculation = True)
  # # print acceptance integral matrix
  # print(f"Acceptance integral matrix\n{momentCalculator.integralMatrix}")
  # eigenVals, _ = momentCalculator.integralMatrix.eigenDecomp
  # print(f"Eigenvalues of acceptance integral matrix\n{np.sort(eigenVals)}")
  # # plot acceptance integral matrix
  # PlottingUtilities.plotComplexMatrix(momentCalculator.integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{plotDirName}/I_acc")
  # PlottingUtilities.plotComplexMatrix(momentCalculator.integralMatrix.inverse,          pdfFileNamePrefix = f"{plotDirName}/I_inv")
  # ROOT.gBenchmark.Stop(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")

  # # calculate moments of accepted phase-space data
  # ROOT.gBenchmark.Start(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")
  # momentCalculator.calculateMoments(dataSource = MomentCalculator.MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE)
  # # print all moments for first kinematic bin
  # print(f"Measured moments of accepted phase-space data\n{momentCalculator.HMeas}")
  # print(f"Physical moments of accepted phase-space data\n{momentCalculator.HPhys}")
  # # plot moments in each kinematic bin
  # HTruePs = MomentCalculator.MomentResult(momentIndices, label = "true")  # all true phase-space moment are 0 ...
  # HTruePs._valsFlatIndex[momentIndices.indexMap.flatIndex_for[MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1  # ... except H_0(0, 0), which is 1
  # PlottingUtilities.plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTruePs, pdfFileNamePrefix = f"{plotDirName}/hPs_")
  # ROOT.gBenchmark.Stop(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")

  # calculate moments of signal data
  ROOT.gBenchmark.Start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
  momentCalculator.calculateMoments()
  # print all moments for first kinematic bin
  print(f"Measured moments of signal data\n{momentCalculator.HMeas}")
  print(f"Physical moments of signal data\n{momentCalculator.HPhys}")
  # plot moments
  PlottingUtilities.plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTrue, pdfFileNamePrefix = f"{plotDirName}/h_")
  ROOT.gBenchmark.Stop(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")

  ROOT.gBenchmark.Stop("Total execution time")
  _ = ctypes.c_float(0.0)  # dummy argument required by ROOT; sigh
  ROOT.gBenchmark.Summary(_, _)
  print("!Note! the 'TOTAL' time above is wrong; ignore")

  OpenMp.restoreNmbOpenMpThreads()
