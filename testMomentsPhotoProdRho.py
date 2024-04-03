#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3

import functools
import numpy as np
import os
import threadpoolctl
from typing import (
  List,
  Tuple,
)

import ROOT

import MomentCalculator
import PlottingUtilities
import RootUtilities
import testMomentsPhotoProd
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def genDataFromWaves(
  nmbEvents:         int,                            # number of events to generate
  polarization:      float,                          # photon-beam polarization
  amplitudeSet:      MomentCalculator.AmplitudeSet,  # partial-wave amplitudes
  efficiencyHist:    ROOT.TH3D,                      # detection efficiency used to generate data
  regenerateData:    bool = False,                   # if set data are regenerated although .root file exists
  outFileNamePrefix: str = "./",                     # name prefix for output files
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
    f"- {intensityComponentsFormula[2]} * {polarization} * std::sin(2 * TMath::DegToRad() * z))")  # Eq. (163)
  print(f"Intensity formula = {intensityFormula}")
  # print(f"!!! I_0 = {intensityComponentsFormula[0]}")
  # I_1_formula = f"std::real({intensityComponentsFormula[1]} * {polarization})"
  # print(f"!!! I_1 = {I_1_formula}")
  # I_1 = ROOT.TF2("I_1", I_1_formula, -1, +1, -180, +180)
  # I_1.SetTitle(";cos#theta;#phi [deg]")
  # I_1.SetNpx(100)  # used in numeric integration performed by GetRandom()
  # I_1.SetNpy(100)
  # canv = ROOT.TCanvas()
  # I_1.Draw("COLZ")
  # canv.Print("I_1.pdf", "pdf")
  # I_2_formula = f"std::real({intensityComponentsFormula[2]} * {polarization})"
  # print(f"!!! I_2 = {I_2_formula}")
  # I_2 = ROOT.TF2("I_2", I_2_formula, -1, +1, -180, +180)
  # I_2.SetTitle(";cos#theta;#phi [deg]")
  # I_2.SetNpx(100)  # used in numeric integration performed by GetRandom()
  # I_2.SetNpy(100)
  # canv = ROOT.TCanvas()
  # I_2.Draw("COLZ")
  # canv.Print("I_2.pdf", "pdf")

  intensityFcn = ROOT.TF3("intensity", intensityFormula, -1, +1, -180, +180, -180, +180)
  intensityFcn.SetTitle(";cos#theta;#phi [deg];#Phi [deg]")
  intensityFcn.SetNpx(100)  # used in numeric integration performed by GetRandom()
  intensityFcn.SetNpy(100)
  intensityFcn.SetNpz(100)
  intensityFcn.SetMinimum(0)
  PlottingUtilities.drawTF3(intensityFcn, **testMomentsPhotoProd.TH3_PLOT_KWARGS, pdfFileName = f"{outFileNamePrefix}hIntensity.pdf")

  # intensity as calculated from Eqs. (9) to (12) in https://arxiv.org/abs/2305.09047 assuming P_\gamma = 1, and SCHC with NPE
  # i.e. rho^1_{1 -1} = +1/2 and rho^2_{1 -1} = -i/2 (and rho^0_{1 1} = +1/2)
  # parity conservation: rho^0_{-1 -1} = +1/2, rho^1_{-1 1} = +1/2, rho^2_{-1 1} = +i/2
  # identical formulas are obtained from the Z_l^m formulation and the
  # spin-density matrices in the reflectivity basis (see Eqs. (D8) and (D13) in PRD 100, 054017 (2019))
  intensityFormula2 = (
    "(3 / (8 * TMath::Pi())) * std::sin(std::acos(x)) * std::sin(std::acos(x)) "
    "* (1 + std::cos(2 * TMath::DegToRad() * y) * std::cos(2 * TMath::DegToRad() * z) + std::sin(2 * TMath::DegToRad() * y) * std::sin(2 * TMath::DegToRad() * z))")
  intensityFcn2 = ROOT.TF3("intensity2", intensityFormula2, -1, +1, -180, +180, -180, +180)
  intensityFcn2.SetTitle(";cos#theta;#phi [deg];#Phi [deg]")
  intensityFcn2.SetNpx(100)  # used in numeric integration performed by GetRandom()
  intensityFcn2.SetNpy(100)
  intensityFcn2.SetNpz(100)
  intensityFcn2.SetMinimum(0)
  PlottingUtilities.drawTF3(intensityFcn2, **testMomentsPhotoProd.TH3_PLOT_KWARGS, pdfFileName = f"{outFileNamePrefix}hIntensity2.pdf")

  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = f"{outFileNamePrefix}{intensityFcn.GetName()}.photoProd.root"
  if os.path.exists(fileName) and not regenerateData:
    print(f"Reading partial-wave MC data from '{fileName}'")
    return ROOT.RDataFrame(treeName, fileName)
  print(f"Generating partial-wave MC data and writing them to '{fileName}'")
  RootUtilities.declareInCpp(intensityFcn   = intensityFcn)    # use Python object in C++
  RootUtilities.declareInCpp(efficiencyHist = efficiencyHist)  # use Python object in C++
  # normalize efficiency histogram
  efficiencyHist.Scale(1 / efficiencyHist.GetMaximum())
  pointFunc = """
    double cosTheta, phiDeg, PhiDeg;
    // weight intensity function by efficiency histogram using rejection sampling
    do {
      // uniform distribution
      // cosTheta = gRandom->Uniform(2) - 1;
      // phiDeg   = 180 * (gRandom->Uniform(2) - 1);
      // PhiDeg   = 180 * (gRandom->Uniform(2) - 1);
      // distribution given by intensity function
      PyVars::intensityFcn.GetRandom3(cosTheta, phiDeg, PhiDeg);
    } while(gRandom->Uniform() > PyVars::efficiencyHist.GetBinContent(PyVars::efficiencyHist.FindBin(cosTheta, phiDeg, PhiDeg)));  // weight by efficiency histogram
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
           .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genDataFromWaves()" << endl; } return true;') \
           .Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](["cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi"]))
    # snapshot is needed or else the `point` column would be regenerated for every triggered loop
    # noop filter before snapshot logs when event loop is running
    # !Note! for some reason, this is very slow
  nmbBins = 25
  hist = df.Histo3D(ROOT.RDF.TH3DModel("hSignalSim", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180), "cosTheta", "phiDeg", "PhiDeg")
  canv = ROOT.TCanvas()
  hist.SetMinimum(0)
  hist.GetXaxis().SetTitleOffset(1.5)
  hist.GetYaxis().SetTitleOffset(2)
  hist.GetZaxis().SetTitleOffset(1.5)
  hist.Draw("BOX2Z")
  canv.SaveAs(f"{outFileNamePrefix}{hist.GetName()}.pdf")
  return df


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  PlottingUtilities.setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 5):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of test case
    # !Note! the SDMEs and hence the partial-wave amplitudes are defined in the rho helicity frame
    #        the rho decay angles need to be calculated in this frame
    outFileDirName      = Utilities.makeDirPath("./plotsPhotoProdRho")
    treeName            = "ntFSGlueX_100_110_angles"
    signalFileName      = "./dataPhotoProdRho/tree_pippim__B4_gen_amp_030994.signal.root.angles"
    nmbSignalEvents     = 218240
    acceptedPsFileName  = "./dataPhotoProdRho/tree_pippim__B4_gen_amp_030994.phaseSpace.root.angles"
    nmbAcceptedPsEvents = 210236  #TODO not correct number to normalize integral matrix
    beamPolarization    = 0.4  #TODO read from tree
    # beamPolarization    = 1.0  #TODO read from tree
    maxL                = 5  # define maximum L quantum number of moments
    # maxL                = 10  # define maximum L quantum number of moments
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # calculate true moments
    partialWaveAmplitudes = [
      MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = +1), val = 1 + 0j),  # P_+1^+
    ]
    amplitudeSet = MomentCalculator.AmplitudeSet(partialWaveAmplitudes)
    HTrue: MomentCalculator.MomentResult = amplitudeSet.photoProdMomentSet(maxL)
    print(f"True moment values\n{HTrue}")
    for refl in (-1, +1):
      for l in range(3):
        for m1 in range(-l, l + 1):
          for m2 in range(-l, l + 1):
            rhos = amplitudeSet.photoProdSpinDensElements(refl, l, l, m1, m2)
            if not all(rho == 0 for rho in rhos):
              print(f"!!! refl = {refl}, l = {l}, m = {m1}, m' = {m2}: {rhos}")

    # load data
    print(f"Loading signal data from tree '{treeName}' in file '{signalFileName}'")
    dataSignal = ROOT.RDataFrame(treeName, signalFileName).Range(10000)  # take only first 10k events
    # dataSignal = ROOT.RDataFrame(treeName, signalFileName)
    print(f"Loading accepted phase-space data from tree '{treeName}' in file '{acceptedPsFileName}'")
    # dataAcceptedPs = ROOT.RDataFrame(treeName, acceptedPsFileName).Range(10000)  # take only first 10k events
    dataAcceptedPs = ROOT.RDataFrame(treeName, acceptedPsFileName)

    nmbBins   = 25
    nmbBinsPs = nmbBins
    # nmbBinsPs = 100  # if histogram is used for weighting events
    # plot signal and phase-space data
    hists = (
      dataSignal.Histo3D(
        ROOT.RDF.TH3DModel("hSignal", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
        "cosTheta", "phiDeg", "PhiDeg"),
      dataAcceptedPs.Histo3D(
        ROOT.RDF.TH3DModel("hPhaseSpace", ";cos#theta;#phi [deg];#Phi [deg]", nmbBinsPs, -1, +1, nmbBinsPs, -180, +180, nmbBinsPs, -180, +180),
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

    # print(f"Generating signal data")
    # dataSignal = genDataFromWaves(10 * nmbSignalEvents, beamPolarization, amplitudeSet, hists[0].GetValue(), pdfFileNamePrefix = f"{outFileDirName}/", regenerateData = True)

    # setup moment calculator
    momentIndices = MomentCalculator.MomentIndices(maxL)
    dataSet = MomentCalculator.DataSet(beamPolarization, dataSignal, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbAcceptedPsEvents)
    # dataSet = MomentCalculator.DataSet(beamPolarization, dataSignal, phaseSpaceData = None, nmbGenEvents = nmbAcceptedPsEvents)
    momentCalculator = MomentCalculator.MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix")

    # calculate integral matrix
    t = timer.start(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")
    momentCalculator.calculateIntegralMatrix(forceCalculation = True)
    # print acceptance integral matrix
    print(f"Acceptance integral matrix\n{momentCalculator.integralMatrix}")
    eigenVals, _ = momentCalculator.integralMatrix.eigenDecomp
    print(f"Eigenvalues of acceptance integral matrix\n{np.sort(eigenVals)}")
    # plot acceptance integral matrix
    PlottingUtilities.plotComplexMatrix(momentCalculator.integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/I_acc")
    PlottingUtilities.plotComplexMatrix(momentCalculator.integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/I_inv")
    t.stop()

    # calculate moments of accepted phase-space data
    t = timer.start(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")
    momentCalculator.calculateMoments(dataSource = MomentCalculator.MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE)
    # print all moments
    print(f"Measured moments of accepted phase-space data\n{momentCalculator.HMeas}")
    print(f"Physical moments of accepted phase-space data\n{momentCalculator.HPhys}")
    # plot moments
    HTruePs = MomentCalculator.MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
    HTruePs._valsFlatIndex[momentIndices[MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1  # ... except H_0(0, 0), which is 1
    PlottingUtilities.plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTruePs, pdfFileNamePrefix = f"{outFileDirName}/hPs_")
    t.stop()

    # calculate moments of signal data
    t = timer.start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
    momentCalculator.calculateMoments()
    # print all moments for first kinematic bin
    print(f"Measured moments of signal data\n{momentCalculator.HMeas}")
    print(f"Physical moments of signal data\n{momentCalculator.HPhys}")
    # plot moments
    PlottingUtilities.plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTrue, pdfFileNamePrefix = f"{outFileDirName}/h_")
    t.stop()

    timer.stop("Total execution time")
    print(timer.summary)
