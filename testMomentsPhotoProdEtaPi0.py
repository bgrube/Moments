#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=4

import functools
import numpy as np
import pandas as pd
import threadpoolctl
from typing import (
  List,
)

import ROOT

import MomentCalculator
import PlottingUtilities
# import RootUtilities  # initializes OpenMP and loads wignerD.C
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def readPartialWaveAmplitudes(
  csvFileName:   str,
  massBinCenter: float,  # [GeV]
) -> List[MomentCalculator.AmplitudeValue]:
  """Reads partial-wave amplitudes values for given mass bin from CSV data"""
  df = pd.read_csv(csvFileName)
  df = df.loc[df["mass"] == massBinCenter].drop(columns = ["mass"])
  assert len(df) == 1, f"Expected exactly 1 row for mass-bin center {massBinCenter} GeV, but found {len(df)}"
  # Pandas cannot read-back complex values out of the box
  # there also seems to be no interest in fixing that <https://github.com/pandas-dev/pandas/issues/9379>
  # have to convert columns by hand
  s = df.astype('complex128').loc[df.index[0]]
  partialWaveAmplitudes = [
    # negative-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 0, m =  0), val = s['S0+']),   # S_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -1), val = s['D1--']),  # D_-1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m =  0), val = s['D0+-']),  # D_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +1), val = s['D1+-']),  # D_+1^-
    # positive-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 0, m =  0), val = s['S0+']),   # S_0^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -2), val = s['D2-+']),  # D_-2^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +2), val = s['D2++']),  # D_+2^+
  ]
  # print(f"!!! {partialWaveAmplitudes}")
  return partialWaveAmplitudes


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
    outFileDirName       = "./plotsPhotoProdEtaPi0"  #TODO create output dirs; make sure integrals are saved there as well
    treeName             = "etaPi0"
    # signalFileName       = "./dataPhotoProdEtaPi0/tree_pippim__B4_gen_amp_030994.signal.root.angles"
    # nmbSignalEvents      = 218240
    signalPWAmpsFileName = "./dataPhotoProdEtaPi0/a0a2_raw/a0a2_complex_amps.csv"
    # acceptedPsFileName   = "./dataPhotoProdEtaPi0/pol000_t010020_m104180_selectGenTandM_F2017_1_selected_acc_flat.phaseSpace.root"
    # nmbAcceptedPsEvents  = 56036  #TODO not the correct number to normalize integral matrix
    acceptedPsFileName   = "./dataPhotoProdEtaPi0/a0a2_phaseSpace_acc_flat.root"
    nmbAcceptedPsEvents  = 334283  #TODO not the correct number to normalize integral matrix -> events after BG subtraction
    beamPolarization     = 1.0  #TODO read from tree
    # maxL                 = 1  # define maximum L quantum number of moments
    maxL                 = 5  # define maximum L quantum number of moments
    # nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # calculate true moments
    amplitudeSet = MomentCalculator.AmplitudeSet(amps = readPartialWaveAmplitudes(signalPWAmpsFileName, 1.34), tolerance = 1e-11)
    HTrue: MomentCalculator.MomentResult = amplitudeSet.photoProdMomentSet(maxL)
    print(f"True moment values\n{HTrue}")
    for refl in (-1, +1):
      for l in range(3):
        for m1 in range(-l, l + 1):
          for m2 in range(-l, l + 1):
            rhos = amplitudeSet.photoProdSpinDensElements(refl, l, l, m1, m2)
            if not all(rho == 0 for rho in rhos):
              print(f"!!! refl = {refl}, l = {l}, l' = {l}, m = {m1}, m' = {m2}: {rhos}")
    # raise ValueError

    # load data
    # print(f"Loading signal data from tree '{treeName}' in file '{signalFileName}'")
    # dataSignal = ROOT.RDataFrame(treeName, signalFileName).Range(10000)  # take only first 10k events
    dataSignal = None
    print(f"Loading accepted phase-space data from tree '{treeName}' in file '{acceptedPsFileName}'")
    dataAcceptedPs = ROOT.RDataFrame(treeName, acceptedPsFileName)

    nmbBins   = 15
    nmbBinsPs = nmbBins
    # plot signal and phase-space data
    hists = (
      # dataSignal.Histo3D(
      #   ROOT.RDF.TH3DModel("hSignal", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
      #   "cosTheta", "phiDeg", "PhiDeg"),
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
    momentCalculator = MomentCalculator.MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix")

    # calculate integral matrix
    timer.start(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")
    momentCalculator.calculateIntegralMatrix(forceCalculation = True)
    # print acceptance integral matrix
    print(f"Acceptance integral matrix\n{momentCalculator.integralMatrix}")
    eigenVals, _ = momentCalculator.integralMatrix.eigenDecomp
    print(f"Eigenvalues of acceptance integral matrix\n{np.sort(eigenVals)}")
    # plot acceptance integral matrix
    PlottingUtilities.plotComplexMatrix(momentCalculator.integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/I_acc")
    PlottingUtilities.plotComplexMatrix(momentCalculator.integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/I_inv")
    timer.stop(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")

    # calculate moments of accepted phase-space data
    timer.start(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")
    momentCalculator.calculateMoments(dataSource = MomentCalculator.MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE)
    # print all moments
    print(f"Measured moments of accepted phase-space data\n{momentCalculator.HMeas}")
    print(f"Physical moments of accepted phase-space data\n{momentCalculator.HPhys}")
    # plot moments
    HTruePs = MomentCalculator.MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
    HTruePs._valsFlatIndex[momentIndices.indexMap.flatIndex_for[MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1  # ... except H_0(0, 0), which is 1
    PlottingUtilities.plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTruePs, pdfFileNamePrefix = f"{outFileDirName}/hPs_")
    timer.stop(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")

    # # calculate moments of signal data
    # timer.start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
    # momentCalculator.calculateMoments()
    # # print all moments for first kinematic bin
    # print(f"Measured moments of signal data\n{momentCalculator.HMeas}")
    # print(f"Physical moments of signal data\n{momentCalculator.HPhys}")
    # # plot moments
    # PlottingUtilities.plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTrue, pdfFileNamePrefix = f"{outFileDirName}/h_")
    # timer.stop(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")

    timer.stop("Total execution time")
    print(timer.summary())
