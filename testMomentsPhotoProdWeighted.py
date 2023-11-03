#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3

import ctypes
import functools
import os
import subprocess
from typing import (
  Tuple,
  TypedDict,
)

import ROOT

import MomentCalculator
import OpenMp
import PlottingUtilities
import testMomentsPhotoProd


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def printGitInfo() -> None:
  """Prints directory of this file and git hash in this directory"""
  repoDir = os.path.dirname(os.path.abspath(__file__))
  gitInfo = subprocess.check_output(["git", "describe", "--always"], cwd = repoDir).strip().decode()
  print(f"Running code in '{repoDir}', git version '{gitInfo}'")


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


if __name__ == "__main__":
  printGitInfo()
  OpenMp.setNmbOpenMpThreads(5)
  ROOT.gROOT.SetBatch(True)
  ROOT.gRandom.SetSeed(1234567890)
  # ROOT.EnableImplicitMT(10)
  PlottingUtilities.setupPlotStyle()
  ROOT.gBenchmark.Start("Total execution time")

  # set parameters of test case
  plotDirName = "./plots"
  nmbPwaMcEventsSig = 10000000
  nmbPwaMcEventsBkg = 10000000
  nmbPsMcEvents = 1000000
  beamPolarization = 1.0
  # define angular distribution of signal
  partialWaveAmplitudesSig = [  # set of all possible waves up to ell = 2
    # negative-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.4 + 0.1j),  # P_-1^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m =  0), val =  0.3 - 0.8j),  # P_0^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.8 + 0.7j),  # P_+1^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -2), val =  0.1 - 0.4j),  # D_-2^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -1), val =  0.5 + 0.2j),  # D_-1^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m =  0), val = -0.1 - 0.2j),  # D_ 0^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +1), val =  0.2 - 0.1j),  # D_+1^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.2 + 0.3j),  # D_+2^-
    # # positive-reflectivity waves
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = -1), val =  0.5 - 0.1j),  # P_-1^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.8 - 0.3j),  # P_0^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.6 + 0.3j),  # P_+1^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.2 + 0.1j),  # D_-2^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -1), val =  0.2 - 0.3j),  # D_-1^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m =  0), val =  0.1 - 0.2j),  # D_ 0^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +1), val =  0.2 + 0.5j),  # D_+1^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +2), val = -0.3 - 0.1j),  # D_+2^+
  ]
  amplitudeSetSig = MomentCalculator.AmplitudeSet(partialWaveAmplitudesSig)
  # define angular distribution of background
  partialWaveAmplitudesBkg = [  # set of all possible waves up to ell = 2
    # negative-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.9 + 0.7j),  # P_-1^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m =  0), val = -0.6 + 0.4j),  # P_0^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.9 - 0.8j),  # P_+1^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -2), val = -1.0 - 0.7j),  # D_-2^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = -1), val = -0.8 - 0.7j),  # D_-1^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m =  0), val =  0.4 + 0.3j),  # D_ 0^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +1), val = -0.6 - 0.1j),  # D_+1^-
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.1 - 0.9j),  # D_+2^-
    # # positive-reflectivity waves
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = -1), val = -1.0 + 0.8j),  # P_-1^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.2 + 0.2j),  # P_0^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.0 - 0.3j),  # P_+1^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.7 + 0.9j),  # D_-2^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = -1), val = -0.4 - 0.5j),  # D_-1^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m =  0), val = -0.3 + 0.2j),  # D_ 0^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +1), val = -1.0 - 0.4j),  # D_+1^+
    # MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(refl = +1, l = 2, m = +2), val =  0.5 - 0.2j),  # D_+2^+
  ]
  amplitudeSetBkg = MomentCalculator.AmplitudeSet(partialWaveAmplitudesBkg)
  maxL = 3  # define maximum L quantum number of moments
  # formulas for detection efficiency
  # x = cos(theta) in [-1, +1], y = phi in [-180, +180] deg, z = Phi in [-180, +180] deg
  # efficiencyFormulaGen = "1"  # acc_perfect
  efficiencyFormulaGen = "(1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / 1.5**3"  # acc_1; even in all variables
  efficiencyFormulaReco = efficiencyFormulaGen
  nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()
  print(f"!!! {nmbOpenMpThreads =}")

  # calculate true moment values and generate data from partial-wave amplitudes
  ROOT.gBenchmark.Start("Time to generate MC data from partial waves")
  # generate signal distribution
  HTrueSig: MomentCalculator.MomentResult = amplitudeSetSig.photoProdMomentSet(maxL)
  print(f"True moment values for signal:\n{HTrueSig}")
  dataPwaModelSig: ROOT.RDataFrame = testMomentsPhotoProd.genDataFromWaves(
    nmbPwaMcEventsSig, beamPolarization, amplitudeSetSig, efficiencyFormulaGen, pdfFileNamePrefix = f"{plotDirName}/", nameSuffix = "Sig", regenerateData = True)
  dataPwaModelSig = dataPwaModelSig.Define("discrVariable", "gRandom->Gaus(0, 0.1)")
  treeName = "data"
  fileNameSig = f"intensitySig.photoProd.root"
  dataPwaModelSig.Snapshot(treeName, fileNameSig)
  # generate background distribution
  HTrueBkg: MomentCalculator.MomentResult = amplitudeSetBkg.photoProdMomentSet(maxL)
  print(f"True moment values for signal:\n{HTrueBkg}")
  dataPwaModelBkg: ROOT.RDataFrame = testMomentsPhotoProd.genDataFromWaves(
    nmbPwaMcEventsBkg, beamPolarization, amplitudeSetBkg, efficiencyFormulaGen, pdfFileNamePrefix = f"{plotDirName}/", nameSuffix = "Bkg", regenerateData = True)
  dataPwaModelBkg = dataPwaModelBkg.Define("discrVariable", "gRandom->Uniform(0, 2) - 1")
  fileNameBkg = f"intensityBkg.photoProd.root"
  dataPwaModelBkg.Snapshot(treeName, fileNameBkg)
  # concatenate signal and background data frames vertically
  dataPwaModel = ROOT.RDataFrame(treeName, (fileNameSig, fileNameBkg))
  # plot discriminatory variable
  hist = dataPwaModel.Histo1D(ROOT.RDF.TH1DModel("hDiscrVariableSim", ";Discriminatory variable", 100, -1, +1), "discrVariable")
  canv = ROOT.TCanvas()
  hist.Draw()
  canv.SaveAs(f"{plotDirName}/{hist.GetName()}.pdf")
  signalRange = (-0.3, +0.3)
  sideBands   = ((-1, -0.4), (+0.4, +1))
  dataPwaModel = dataPwaModel.Define("sbSubtractionWeight", f"""
    if (({signalRange[0]} < discrVariable) and (discrVariable < {signalRange[1]}))
      return 1.0;
    else if (   (({sideBands[0][0]} < discrVariable) and (discrVariable < {sideBands[0][1]}))
             or (({sideBands[1][0]} < discrVariable) and (discrVariable < {sideBands[1][1]})))
      return -0.5;
    else
      return 0.0;
  """)
  hist = dataPwaModel.Histo1D(ROOT.RDF.TH1DModel("hDiscrVariableSimSbSubtr", ";Discriminatory variable", 100, -1, +1), "discrVariable", "sbSubtractionWeight")
  hist.Draw()
  canv.SaveAs(f"{plotDirName}/{hist.GetName()}.pdf")
  ROOT.gBenchmark.Stop("Time to generate MC data from partial waves")

  # plot data generated from partial-wave amplitudes
  # canv = ROOT.TCanvas()
  nmbBins = 25
  histBinning = (nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180)
  hists = (
    dataPwaModelSig.Filter(f"({signalRange[0]} < discrVariable) and (discrVariable < {signalRange[1]})").Histo3D(
                         ROOT.RDF.TH3DModel("dataSig",     ";cos#theta;#phi [deg];#Phi [deg]", *histBinning), "cosTheta", "phiDeg", "PhiDeg"),
    dataPwaModelBkg.Filter(f"(({sideBands[0][0]} < discrVariable) and (discrVariable < {sideBands[0][1]}))"
                        f"or (({sideBands[1][0]} < discrVariable) and (discrVariable < {sideBands[1][1]}))").Histo3D(
                         ROOT.RDF.TH3DModel("dataBkg",     ";cos#theta;#phi [deg];#Phi [deg]", *histBinning), "cosTheta", "phiDeg", "PhiDeg"),
    dataPwaModel.Histo3D(ROOT.RDF.TH3DModel("data",        ";cos#theta;#phi [deg];#Phi [deg]", *histBinning), "cosTheta", "phiDeg", "PhiDeg"),
    dataPwaModel.Histo3D(ROOT.RDF.TH3DModel("dataSbSubtr", ";cos#theta;#phi [deg];#Phi [deg]", *histBinning), "cosTheta", "phiDeg", "PhiDeg", "sbSubtractionWeight"),
  )
  for hist in hists:
    hist.SetMinimum(0)
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
    canv.SaveAs(f"{plotDirName}/{hist.GetName()}.pdf")

  ROOT.gBenchmark.Stop("Total execution time")
  _ = ctypes.c_float(0.0)  # dummy argument required by ROOT; sigh
  ROOT.gBenchmark.Summary(_, _)
  print("!Note! the 'TOTAL' time above is wrong; ignore")

  OpenMp.restoreNmbOpenMpThreads()
