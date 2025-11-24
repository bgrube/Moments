#!/usr/bin/env python3
"""
This module plots the intensity distributions that correspond to the
moments estimated from data. The moment values are read from files
produced by the script `photoProdCalcMoments.py` that calculates the
moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

import functools
import os

import ROOT

from AnalysisConfig import HistAxisBinning
from dataPhotoProdPiPi.makeMomentsInputTree import (
  BeamPolInfo,
  BEAM_POL_INFOS,
)
from MomentCalculator import (
  MomentResult,
  MomentResultsKinematicBinning,
)
from PlottingUtilities import (
  drawTF3,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def plotIntensityFcn(
  momentResults:     MomentResult,
  massBinIndex:      int,
  beamPolInfo:       BeamPolInfo,
  outputDirName:     str,
  nmbBinsPerAxis:    int                             = 25,
  useIntensityTerms: MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.ALL,
) -> None:
  """Draw intensity function in given mass bin and save PDF to output directory"""
  print(f"Plotting intensity function for mass bin {massBinIndex}")
  if True:
    # draw intensity function as 3D plot
    # formula uses variables: x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
    intensityFormula = momentResults.intensityFormula(
      polarization      = beamPolInfo.pol,
      thetaFormula      = "std::acos(x)",
      phiFormula        = "TMath::DegToRad() * y",
      PhiFormula        = "TMath::DegToRad() * z",
      useIntensityTerms = useIntensityTerms,
    )
    intensityFcn = ROOT.TF3(f"intensityFcn_{useIntensityTerms.value}_bin_{massBinIndex}", intensityFormula, -1, +1, -180, +180, -180, +180)
    binnings = (
      HistAxisBinning(nmbBinsPerAxis,   -1,   +1),  # cos(theta)
      HistAxisBinning(nmbBinsPerAxis, -180, +180),  # phi
      HistAxisBinning(nmbBinsPerAxis, -180, +180),  # Phi
    )
    ROOT.gStyle.SetCanvasDefH(2400)  # increase resolution temporarily
    ROOT.gStyle.SetCanvasDefW(2400)
    histFcn, minVal, maxVal = drawTF3(
      fcn                = intensityFcn,
      binnings           = binnings,
      outFileName        = f"{outputDirName}/{intensityFcn.GetName()}.png",
      histTitle          = "Intensity Function;cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]",
      showNegativeValues = True,
    )
    ROOT.gStyle.SetCanvasDefH(600)  # revert back to default resolution
    ROOT.gStyle.SetCanvasDefW(600)
    if minVal < 0:
      print(f"WARNING: Intensity function for mass bin {massBinIndex} has negative values: minimum = {minVal}, maximum = {maxVal}")
    # draw negative part of intensity function (if any)
    intensityFormulaNeg = f"-({intensityFormula})"
    intensityFcnNeg = ROOT.TF3(f"{intensityFcn.GetName()}_neg", intensityFormulaNeg, -1, +1, -180, +180, -180, +180)
    histFcnNeg, _, _ = drawTF3(
      fcn                = intensityFcnNeg,
      binnings           = binnings,
      outFileName        = f"{outputDirName}/{intensityFcnNeg.GetName()}.pdf",
      histTitle          = "Intensity Function, Negative Part;cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]",
      showNegativeValues = False,
    )
    # draw projections of intensity function onto (cos(theta), phi) plane
    histProj = histFcn.Project3D("yx")  # NOTE: "yx" gives y = phi vs. x = cos(theta)
    canv = ROOT.TCanvas()
    ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # draw 2D plot with pos/neg color palette and symmetric z axis
    histProj.SetTitle(f"Intensity Function Projection;{histFcn.GetXaxis().GetTitle()};{histFcn.GetYaxis().GetTitle()}")
    zRange = abs(histProj.GetMinimum()) if histProj.GetMinimum() < 0 else 10.0  # choose z range to see negative values; but avoid zero range in case function positive
    histProj.SetMinimum(-zRange)
    histProj.SetMaximum(+zRange)
    histProj.Draw("COLZ")
    canv.SaveAs(f"{outputDirName}/{histProj.GetName()}.pdf")
    ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette
    histProjNeg = histFcnNeg.Project3D("yx")  # NOTE: "yx" gives y = phi vs. x = cos(theta)
    canv = ROOT.TCanvas()
    histProjNeg.SetTitle(f"Intensity Function Projection, Negative Part;{histFcnNeg.GetXaxis().GetTitle()};{histFcnNeg.GetYaxis().GetTitle()}")
    histProjNeg.Draw("COLZ")
    canv.SaveAs(f"{outputDirName}/{histProjNeg.GetName()}.pdf")
  if False:
    # draw intensity as function of phi_HF and Phi for fixed cos(theta)_HF value
    cosTheta = 0.0  # fixed value of cos(theta)_HF
    # formula uses variables: x = phi in [-180, +180] deg; y = Phi in [-180, +180] deg
    intensityFormulaFixedCosTheta = momentResults.intensityFormula(
      polarization      = beamPolInfo.pol,
      thetaFormula      = f"std::acos({cosTheta})",
      phiFormula        = "TMath::DegToRad() * x",
      PhiFormula        = "TMath::DegToRad() * y",
      useIntensityTerms = useIntensityTerms,
    )
    intensityFcnFixedCosTheta = ROOT.TF2(f"intensityFcn_fixedCosTheta_{useIntensityTerms.value}_bin_{massBinIndex}", intensityFormulaFixedCosTheta, -180, +180, -180, +180)
    intensityFcnFixedCosTheta.SetTitle(f"Intensity Function for cos#theta_{{HF}} = {cosTheta};#phi_{{HF}} [deg];#Phi [deg]")
    intensityFcnFixedCosTheta.SetNpx(100)
    intensityFcnFixedCosTheta.SetNpy(100)
    intensityFcnFixedCosTheta.SetMinimum(0)
    canv = ROOT.TCanvas()
    intensityFcnFixedCosTheta.Draw("COLZ")
    canv.SaveAs(f"{outputDirName}/{intensityFcnFixedCosTheta.GetName()}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("./rootlogon.C") == 0, "Error loading './rootlogon.C'"

  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  setupPlotStyle()

  dataPeriod        = "2018_08"
  tBinLabel         = "tbin_0.1_0.2"
  beamPolLabel      = "PARA_0"
  maxL              = 4

  timer.start("Total execution time")
  momentResultsFileName = f"./plotsPhotoProdPiPiPol/{dataPeriod}/{tBinLabel}/{beamPolLabel}.maxL_{maxL}/unnorm_moments_phys.pkl"
  print(f"Reading moments from file '{momentResultsFileName}'")
  momentResults = MomentResultsKinematicBinning.loadPickle(momentResultsFileName)
  for useIntensityTerms in (
    MomentResult.IntensityTermsType.ALL,
    MomentResult.IntensityTermsType.PARITY_CONSERVING,
    MomentResult.IntensityTermsType.PARITY_VIOLATING,
  ):
    for massBinIndex, momentResultsForBin in enumerate(momentResults):
    # for massBinIndex, momentResultsForBin in enumerate(momentResults[4:5]):
    # for massBinIndex, momentResultsForBin in enumerate(momentResults[11:12]):
    # for massBinIndex, momentResultsForBin in enumerate(momentResults[30:31]):
      print(f"Generating plot for {momentResultsForBin.binCenters=}")
      plotIntensityFcn(
        momentResults     = momentResultsForBin,
        massBinIndex      = massBinIndex,
        beamPolInfo       = BEAM_POL_INFOS[dataPeriod][beamPolLabel],
        outputDirName     = ".",
        nmbBinsPerAxis    = 50,
        useIntensityTerms = useIntensityTerms,
      )

  timer.stop("Total execution time")
  print(timer.summary)
