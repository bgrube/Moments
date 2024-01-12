#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3

import ctypes
import functools
import numpy as np
import os
import subprocess
from typing import List

import ROOT

import MomentCalculator
import OpenMp
from PlottingUtilities import (
  HistAxisBinning,
  plotComplexMatrix,
  plotMomentsInBin,
  setupPlotStyle,
)
import testMomentsPhotoProd


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def printGitInfo() -> None:
  """Prints directory of this file and git hash in this directory"""
  repoDir = os.path.dirname(os.path.abspath(__file__))
  gitInfo = subprocess.check_output(["git", "describe", "--always"], cwd = repoDir).strip().decode()
  print(f"Running code in '{repoDir}', git version '{gitInfo}'")


if __name__ == "__main__":
  printGitInfo()
  OpenMp.setNmbOpenMpThreads(5)
  ROOT.gROOT.SetBatch(True)
  ROOT.gRandom.SetSeed(1234567890)
  # ROOT.EnableImplicitMT(10)
  setupPlotStyle()
  ROOT.gBenchmark.Start("Total execution time")

  # set parameters of test case
  plotDirName           = "./plots"
  nmbPwaMcEventsSig     = 1000
  nmbPwaMcEventsBkg     = 1000
  # nmbPwaMcEventsSig     = 10000000
  # nmbPwaMcEventsBkg     = 10000000
  nmbAcceptedPsMcEvents = 10000000
  beamPolarization      = 1.0
  maxL                  = 5  # maximum L quantum number of moments to be calculated
  # define angular distribution of signal
  partialWaveAmplitudesSig = [  # set of all possible waves up to ell = 2
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
  amplitudeSetSig = MomentCalculator.AmplitudeSet(partialWaveAmplitudesSig)
  # define angular distribution of background
  partialWaveAmplitudesBkg = [  # set of all possible waves up to ell = 2
    # negative-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.9 + 0.7j),  # P_-1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m =  0), val = -0.6 + 0.4j),  # P_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.9 - 0.8j),  # P_+1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -2), val = -1.0 - 0.7j),  # D_-2^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -1), val = -0.8 - 0.7j),  # D_-1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m =  0), val =  0.4 + 0.3j),  # D_ 0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +1), val = -0.6 - 0.1j),  # D_+1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.1 - 0.9j),  # D_+2^-
    # positive-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = -1), val = -1.0 + 0.8j),  # P_-1^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.2 + 0.2j),  # P_0^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.0 - 0.3j),  # P_+1^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.7 + 0.9j),  # D_-2^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -1), val = -0.4 - 0.5j),  # D_-1^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m =  0), val = -0.3 + 0.2j),  # D_ 0^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +1), val = -1.0 - 0.4j),  # D_+1^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +2), val =  0.5 - 0.2j),  # D_+2^+
  ]
  amplitudeSetBkg = MomentCalculator.AmplitudeSet(partialWaveAmplitudesBkg)
  # formulas for detection efficiency
  # x = cos(theta) in [-1, +1], y = phi in [-180, +180] deg, z = Phi in [-180, +180] deg
  # efficiencyFormula = "1"  # acc_perfect
  efficiencyFormula = "(1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / 1.5**3"  # acc_1; even in all variables
  nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

  # calculate true moment values and generate data from partial-wave amplitudes
  ROOT.gBenchmark.Start("Time to generate MC data from partial waves")
  # generate signal distribution
  HTrueSig: MomentCalculator.MomentResult = amplitudeSetSig.photoProdMomentSet(maxL)
  print(f"True moment values for signal:\n{HTrueSig}")
  dataPwaModelSig: ROOT.RDataFrame = testMomentsPhotoProd.genDataFromWaves(
    nmbPwaMcEventsSig, beamPolarization, amplitudeSetSig, efficiencyFormula, pdfFileNamePrefix = f"{plotDirName}/", nameSuffix = "Sig", regenerateData = True)
  dataPwaModelSig = dataPwaModelSig.Define("discrVariable", "gRandom->Gaus(0, 0.1)")
  treeName = "data"
  fileNameSig = f"intensitySig.photoProd.root"
  dataPwaModelSig.Snapshot(treeName, fileNameSig)
  dataPwaModelSig = ROOT.RDataFrame(treeName, fileNameSig)
  histDiscrSig = dataPwaModelSig.Histo1D(ROOT.RDF.TH1DModel("Signal", ";Discriminatory variable;Count / 0.02", 100, -1, +1), "discrVariable").GetValue()
  # generate background distribution
  HTrueBkg: MomentCalculator.MomentResult = amplitudeSetBkg.photoProdMomentSet(maxL)
  print(f"True moment values for signal:\n{HTrueBkg}")
  dataPwaModelBkg: ROOT.RDataFrame = testMomentsPhotoProd.genDataFromWaves(
    nmbPwaMcEventsBkg, beamPolarization, amplitudeSetBkg, efficiencyFormula, pdfFileNamePrefix = f"{plotDirName}/", nameSuffix = "Bkg", regenerateData = True)
  dataPwaModelBkg = dataPwaModelBkg.Define("discrVariable", "gRandom->Uniform(0, 2) - 1")
  fileNameBkg = f"intensityBkg.photoProd.root"
  dataPwaModelBkg.Snapshot(treeName, fileNameBkg)
  dataPwaModelBkg = ROOT.RDataFrame(treeName, fileNameBkg)
  histDiscrBkg = dataPwaModelBkg.Histo1D(ROOT.RDF.TH1DModel("Background", ";Discriminatory variable;Count / 0.02", 100, -1, +1), "discrVariable").GetValue()
  # concatenate signal and background data frames vertically
  dataPwaModel = ROOT.RDataFrame(treeName, (fileNameSig, fileNameBkg))
  # plot discriminatory variable
  signalRange = (-0.3, +0.3)
  sideBands   = ((-1, -0.4), (+0.4, +1))
  histDiscr = dataPwaModel.Histo1D(ROOT.RDF.TH1DModel("Total", ";Discriminatory variable;Count / 0.02", 100, -1, +1), "discrVariable").GetValue()
  histDiscr.SetLineWidth(2)
  histDiscrSig.SetLineColor(ROOT.kGreen + 2)
  histDiscrBkg.SetLineColor(ROOT.kRed   + 1)
  histDiscrSig.SetLineStyle(ROOT.kDashed)
  histDiscrBkg.SetLineStyle(ROOT.kDashed)
  histStack = ROOT.THStack("hDiscrVariableSim", ";Discriminatory variable;Count / 0.02")
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
  canv.SaveAs(f"{plotDirName}/{histStack.GetName()}.pdf")
  # define event weights
  dataPwaModel = dataPwaModel.Define("eventWeight", f"""
    if (({signalRange[0]} < discrVariable) and (discrVariable < {signalRange[1]}))
      return 1.0;
    else if (   (({sideBands[0][0]} < discrVariable) and (discrVariable < {sideBands[0][1]}))
             or (({sideBands[1][0]} < discrVariable) and (discrVariable < {sideBands[1][1]})))
      return -0.5;
    else
      return 0.0;
  """)
  hist = dataPwaModel.Histo1D(ROOT.RDF.TH1DModel("hDiscrVariableSimSbSubtr", ";Discriminatory variable;Count / 0.02", 100, -1, +1), "discrVariable", "eventWeight")
  hist.Draw()
  canv.SaveAs(f"{plotDirName}/{hist.GetName()}.pdf")
  ROOT.gBenchmark.Stop("Time to generate MC data from partial waves")
  # raise ValueError

  # plot angular distributions of data generated from partial-wave amplitudes
  nmbBins = testMomentsPhotoProd.TH3_NMB_BINS
  histBinning = (nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180)
  hists = (
    dataPwaModelSig.Filter(f"({signalRange[0]} < discrVariable) and (discrVariable < {signalRange[1]})").Histo3D(
                            ROOT.RDF.TH3DModel("dataSig",     testMomentsPhotoProd.TH3_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg"),
    dataPwaModelBkg.Filter(f"(({sideBands[0][0]} < discrVariable) and (discrVariable < {sideBands[0][1]}))"
                        f"or (({sideBands[1][0]} < discrVariable) and (discrVariable < {sideBands[1][1]}))").Histo3D(
                            ROOT.RDF.TH3DModel("dataBkg",     testMomentsPhotoProd.TH3_TITLE, *histBinning), "cosTheta", "phiDeg", "PhiDeg"),
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
    canv.SaveAs(f"{plotDirName}/{hist.GetName()}.pdf")
  print(f"Sum of weights = {dataPwaModel.Sum('eventWeight').GetValue()}")
  # raise ValueError

  # generate accepted phase-space data
  ROOT.gBenchmark.Start("Time to generate phase-space MC data")
  dataAcceptedPs = testMomentsPhotoProd.genAccepted2BodyPsPhotoProd(nmbAcceptedPsMcEvents, efficiencyFormula, pdfFileNamePrefix = f"{plotDirName}/", regenerateData = True)
  ROOT.gBenchmark.Stop("Time to generate phase-space MC data")

  # define input data
  data = dataPwaModel
  # data = dataPwaModelSig
  HTrue = HTrueSig
  # data = dataPwaModelBkg
  # HTrue = HTrueBkg

  # setup moment calculator
  momentIndices = MomentCalculator.MomentIndices(maxL)
  binVarMass = MomentCalculator.KinematicBinningVariable(name = "mass", label = "#it{m}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
  massBinning = HistAxisBinning(nmbBins = 1, minVal = 1.0, maxVal = 2.0, _var = binVarMass)
  momentsInBins:      List[MomentCalculator.MomentCalculator] = []
  momentsInBinsTruth: List[MomentCalculator.MomentCalculator] = []
  for massBinCenter in massBinning:
    # dummy bins with identical data sets
    dataSet = MomentCalculator.DataSet(beamPolarization, data, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbAcceptedPsMcEvents)  #TODO nmbAcceptedPsMcEvents is not correct number to normalize integral matrix
    momentsInBins.append(MomentCalculator.MomentCalculator(momentIndices, dataSet, _binCenters = {binVarMass : massBinCenter}))
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
    plotComplexMatrix(moments[0].integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{plotDirName}/I_acc_{binLabel}")
    plotComplexMatrix(moments[0].integralMatrix.inverse,          pdfFileNamePrefix = f"{plotDirName}/I_inv_{binLabel}")
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
    plotMomentsInBin(HData = moments[0].HPhys, HTrue = HTrue, pdfFileNamePrefix = f"{plotDirName}/h{binLabel}_")
  ROOT.gBenchmark.Stop(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")

  ROOT.gBenchmark.Stop("Total execution time")
  _ = ctypes.c_float(0.0)  # dummy argument required by ROOT; sigh
  ROOT.gBenchmark.Summary(_, _)
  print("!Note! the 'TOTAL' time above is wrong; ignore")

  OpenMp.restoreNmbOpenMpThreads()
