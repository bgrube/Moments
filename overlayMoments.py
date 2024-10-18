#!/usr/bin/env python3


from __future__ import annotations

import functools
import glob
import os

import ROOT

from MomentCalculator import (
  # binLabel,
  # binTitle,
  # constructMomentResultFrom,
  # DataSet,
  KinematicBinningVariable,
  # MomentCalculator,
  # MomentCalculatorsKinematicBinning,
  # MomentIndices,
  # MomentResult,
  MomentResultsKinematicBinning,
  # MomentValue,
  QnMomentIndex,
)
from PlottingUtilities import (
  HistAxisBinning,
  MomentValue,
  # plotAngularDistr,
  # plotComplexMatrix,
  # plotMoments,
  # plotMoments1D,
  # plotMomentsBootstrapDiffInBin,
  # plotMomentsBootstrapDistributions1D,
  # plotMomentsBootstrapDistributions2D,
  # plotMomentsCovMatrices,
  # plotMomentsInBin,
  setupPlotStyle,
)
# import RootUtilities  # importing initializes OpenMP and loads basisFunctions.C
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def overlayMoments1D(
  momentResultsToOverlay: dict[int, MomentResultsKinematicBinning],  # key: L_max moment results, value: moment results
  qnIndex:                QnMomentIndex,    # defines specific moment
  binning:                HistAxisBinning,  # binning to use for plot
  normalizedMoments:      bool = True,  # indicates whether moment values were normalized to H_0(0, 0)
  pdfFileNamePrefix:      str  = "",    # name prefix for output files
) -> None:
  """Overlays moments H_i(L, M) for different L_max as function of kinematical variable"""
  for momentPart, momentPartLabel in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
    histStack = ROOT.THStack(
      f"{pdfFileNamePrefix}overlay_{qnIndex.label}_{momentPart}",
      f"{qnIndex.title} {momentPartLabel};{binning.axisTitle};" + ("Normalized" if normalizedMoments else "Unnormalized") + " Moment Value",
    )
    for maxL, momentResults in momentResultsToOverlay.items():
      # filter out specific moment given by qnIndex
      HVals: tuple[MomentValue, ...] = tuple(HPhys[qnIndex] for HPhys in momentResults if qnIndex in HPhys)
      # create histogram with moments
      histData = ROOT.TH1D("#it{L}_{max} = "f"{maxL}", "", *binning.astuple)
      for HVal in HVals:
        if binning.var not in HVal.binCenters.keys():
          continue
        y, yErr = HVal.realPart(momentPart == "Re")
        binIndex = histData.GetXaxis().FindBin(HVal.binCenters[binning.var])
        histData.SetBinContent(binIndex, y)
        histData.SetBinError  (binIndex, 1e-100 if yErr < 1e-100 else yErr)  # ROOT does not draw points if uncertainty is zero; sigh
      histData.SetLineColor(ROOT.kRed + 1)
      histData.SetMarkerColor(ROOT.kRed + 1)
      histData.SetMarkerStyle(ROOT.kFullCircle)
      histData.SetMarkerSize(0.75)
      histStack.Add(histData, "PEX0")
    canv = ROOT.TCanvas()
    histStack.Draw("NOSTACK")
    # adjust y-range
    canv.Update()
    actualYRange = canv.GetUymax() - canv.GetUymin()
    yRangeFraction = 0.1 * actualYRange
    histStack.SetMinimum(canv.GetUymin() - yRangeFraction)
    histStack.SetMaximum(canv.GetUymax() + yRangeFraction)
    canv.BuildLegend(0.7, 0.85, 0.99, 0.99)
    canv.Update()
    if (canv.GetUymin() < 0) and (canv.GetUymax() > 0):
      zeroLine = ROOT.TLine()
      zeroLine.SetLineColor(ROOT.kBlack)
      zeroLine.SetLineStyle(ROOT.kDashed)
      xAxis = histStack.GetXaxis()
      zeroLine.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 0, xAxis.GetBinUpEdge(xAxis.GetLast()), 0)
    canv.SaveAs(f"{histStack.GetName()}.pdf")


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  setupPlotStyle()
  timer.start("Total execution time")

  # # set parameters of analysis
  momentsFileDirNamePattern = "./plotsPhotoProdPiPiUnpol.maxL_*"
  outFileDirName            = Utilities.makeDirPath("./plotsPhotoProdPiPiUnpolOverlay")
  normalizeMoments          = False
  binVarMass                = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}", unit = "GeV/#it{c}^{2}", nmbDigits = 3)
  massBinning               = HistAxisBinning(nmbBins = 100, minVal = 0.4, maxVal = 1.4, _var = binVarMass)  # same binning as used by CLAS

  namePrefix = "norm" if normalizeMoments else "unnorm"

  # load moment results
  momentResultsToOverlay: dict[int, MomentResultsKinematicBinning] = {}  # key: L_max moment results, value: moment results
  for fitResultDirName in sorted(tuple(found for found in glob.glob(momentsFileDirNamePattern) if os.path.isdir(found))):
    print(f"Loading moment results from directory {fitResultDirName}")
    maxL = int(fitResultDirName.split("_")[-1])
    momentResultsPhysFileName = f"{fitResultDirName}/{namePrefix}_moments_phys.pkl"
    try:
      momentResultsPhys = MomentResultsKinematicBinning.load(momentResultsPhysFileName)
    except FileNotFoundError as e:
      print(f"Cannot not find file '{momentResultsPhysFileName}'. Skipping directory '{fitResultDirName}'")
      continue
    momentResultsToOverlay[maxL] = momentResultsPhys
  # get largest maxL value
  maxMaxL = max(momentResultsToOverlay.keys())

  # ensure that all moment results have identical kinematic binning and identical order of kinematic bins
  momentResults: tuple[MomentResultsKinematicBinning, ...]         = tuple(momentResultsToOverlay.values())
  binCenters:    tuple[dict[KinematicBinningVariable, float], ...] = momentResults[0].binCenters  # bin centers of first moment result
  for momentResult in momentResults[1:]:
    assert momentResult.binCenters == binCenters

  # # plot moments in each kinematic bin
  # for massBinIndex, HPhys in enumerate(momentResultsPhys):
  #   label = binLabel(HPhys)
  #   title = binTitle(HPhys)
  #   # print(f"True moments for kinematic bin {title}:\n{HTrue}")
  #   print(f"Measured moments of real data for kinematic bin {title}:\n{HMeas}")
  #   print(f"Physical moments of real data for kinematic bin {title}:\n{HPhys}")
  #   plotMomentsInBin(
  #     HData             = HPhys,
  #     normalizedMoments = normalizeMoments,
  #     HTrue             = HClas,
  #     pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
  #     legendLabels      = ("Moment", "CLAS"),
  #     plotHTrueUncert   = True,
  #   )

  # plot kinematic dependences of all moments
  for qnIndex in momentResultsToOverlay[maxMaxL][0].indices.qnIndices:
    print(f"!!! {qnIndex=}")
    overlayMoments1D(
      momentResultsToOverlay = momentResultsToOverlay,
      qnIndex                = qnIndex,
      binning                = massBinning,
      normalizedMoments      = normalizeMoments,
      pdfFileNamePrefix      = f"{outFileDirName}/{namePrefix}_{massBinning.var.name}_",
    )

  timer.stop("Total execution time")
  print(timer.summary)
