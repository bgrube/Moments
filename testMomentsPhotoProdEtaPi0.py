#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=4

import ctypes
import functools
import numpy as np
import os
import subprocess
import threadpoolctl

import ROOT

import MomentCalculator
import PlottingUtilities
import RootUtilities  # initializes OpenMP and loads wignerD.C


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def printGitInfo() -> None:
  """Prints directory of this file and git hash in this directory"""
  repoDir = os.path.dirname(os.path.abspath(__file__))
  gitInfo = subprocess.check_output(["git", "describe", "--always"], cwd = repoDir).strip().decode()
  print(f"Running code in '{repoDir}', git version '{gitInfo}'")


if __name__ == "__main__":
  printGitInfo()
  ROOT.gROOT.SetBatch(True)
  PlottingUtilities.setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 5):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    ROOT.gBenchmark.Start("Total execution time")  #TODO create Python timer object; put into Utilities module; move printGitInfo() there

    # set parameters of test case
    outFileDirName      = "./plotsPhotoProdEtaPi0"  #TODO create output dirs; make sure integrals are saved there as well
    treeName            = "etaPi0"
    # signalFileName      = "./dataPhotoProdEtaPi0/tree_pippim__B4_gen_amp_030994.signal.root.angles"
    # nmbSignalEvents     = 218240
    acceptedPsFileName  = "./dataPhotoProdEtaPi0/pol000_t010020_m104180_selectGenTandM_F2017_1_selected_acc_flat.phaseSpace.root"
    nmbAcceptedPsEvents = 56036  #TODO not the correct number to normalize integral matrix
    beamPolarization    = 0.4  #TODO read from tree
    # beamPolarization    = 1.0  #TODO read from tree
    # maxL                = 1  # define maximum L quantum number of moments
    maxL                = 5  # define maximum L quantum number of moments
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # # calculate true moments
    # partialWaveAmplitudes = [
    #   MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = +1), val = 1 + 0j),  # P_+1^+
    # ]
    # amplitudeSet = MomentCalculator.AmplitudeSet(partialWaveAmplitudes)
    # HTrue: MomentCalculator.MomentResult = amplitudeSet.photoProdMomentSet(maxL)
    # print(f"True moment values\n{HTrue}")
    # for refl in (-1, +1):
    #   for l in range(3):
    #     for m1 in range(-l, l + 1):
    #       for m2 in range(-l, l + 1):
    #         rhos = amplitudeSet.photoProdSpinDensElements(refl, l, l, m1, m2)
    #         if not all(rho == 0 for rho in rhos):
    #           print(f"!!! refl = {refl}, l = {l}, m = {m1}, m' = {m2}: {rhos}")

    # load data
    # print(f"Loading signal data from tree '{treeName}' in file '{signalFileName}'")
    # dataSignal = ROOT.RDataFrame(treeName, signalFileName).Range(10000)  # take only first 10k events
    dataSignal = None
    print(f"Loading accepted phase-space data from tree '{treeName}' in file '{acceptedPsFileName}'")
    dataAcceptedPs = ROOT.RDataFrame(treeName, acceptedPsFileName)

    nmbBins   = 10
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
    ROOT.gBenchmark.Start(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")
    momentCalculator.calculateIntegralMatrix(forceCalculation = True)
    # print acceptance integral matrix
    print(f"Acceptance integral matrix\n{momentCalculator.integralMatrix}")
    eigenVals, _ = momentCalculator.integralMatrix.eigenDecomp
    print(f"Eigenvalues of acceptance integral matrix\n{np.sort(eigenVals)}")
    # plot acceptance integral matrix
    PlottingUtilities.plotComplexMatrix(momentCalculator.integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/I_acc")
    PlottingUtilities.plotComplexMatrix(momentCalculator.integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/I_inv")
    ROOT.gBenchmark.Stop(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")

    # calculate moments of accepted phase-space data
    ROOT.gBenchmark.Start(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")
    momentCalculator.calculateMoments(dataSource = MomentCalculator.MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE)
    # print all moments
    print(f"Measured moments of accepted phase-space data\n{momentCalculator.HMeas}")
    print(f"Physical moments of accepted phase-space data\n{momentCalculator.HPhys}")
    # plot moments
    HTruePs = MomentCalculator.MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
    HTruePs._valsFlatIndex[momentIndices.indexMap.flatIndex_for[MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1  # ... except H_0(0, 0), which is 1
    PlottingUtilities.plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTruePs, pdfFileNamePrefix = f"{outFileDirName}/hPs_")
    ROOT.gBenchmark.Stop(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")

    # # calculate moments of signal data
    # ROOT.gBenchmark.Start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
    # momentCalculator.calculateMoments()
    # # print all moments for first kinematic bin
    # print(f"Measured moments of signal data\n{momentCalculator.HMeas}")
    # print(f"Physical moments of signal data\n{momentCalculator.HPhys}")
    # # plot moments
    # PlottingUtilities.plotMomentsInBin(HData = momentCalculator.HPhys, HTrue = HTrue, pdfFileNamePrefix = f"{outFileDirName}/h_")
    # ROOT.gBenchmark.Stop(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")

    ROOT.gBenchmark.Stop("Total execution time")
    _ = ctypes.c_float(0.0)  # dummy argument required by ROOT; sigh
    ROOT.gBenchmark.Summary(_, _)
    print("!Note! the 'TOTAL' time above is wrong; ignore")
