#!/usr/bin/env python3


from __future__ import annotations

import functools
import glob
import os

import ROOT

from MomentCalculator import (
  KinematicBinningVariable,
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
from PlottingUtilities import (
  HistAxisBinning,
  MomentValue,
  setCbFriendlyStyle,
  setupPlotStyle,
)
# import RootUtilities  # importing initializes OpenMP and loads basisFunctions.C
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def overlayMoments1D(
  momentResultsToOverlay: dict[str, MomentResultsKinematicBinning],  # key: legend label, value: moment results
  qnIndex:                QnMomentIndex,    # defines specific moment
  binning:                HistAxisBinning,  # binning to use for plot
  normalizedMoments:      bool = True,  # indicates whether moment values were normalized to H_0(0, 0)
  pdfFileNamePrefix:      str  = "",    # name prefix for output files
) -> None:
  """Overlays moments H_i(L, M) from different analyses as function of kinematical variable"""
  print(f"Overlaying {qnIndex.label} moments as a function of the '{binning.var.name}' variable")
  for momentPart, momentPartLabel in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
    histStack = ROOT.THStack(
      f"{pdfFileNamePrefix}overlay_{qnIndex.label}_{momentPart}",
      f"{qnIndex.title} {momentPartLabel};{binning.axisTitle};" + ("Normalized" if normalizedMoments else "Unnormalized") + " Moment Value",
    )
    for index, (legendLabel, momentResults) in enumerate(momentResultsToOverlay.items()):
      # filter out specific moment given by qnIndex
      HVals: tuple[MomentValue, ...] = tuple(HPhys[qnIndex] for HPhys in momentResults if qnIndex in HPhys)
      # create histogram with moments
      histData = ROOT.TH1D(legendLabel, "", *binning.astuple)
      for HVal in HVals:
        if binning.var not in HVal.binCenters.keys():
          continue
        y, yErr = HVal.part(real = (momentPart == "Re"))
        binIndex = histData.GetXaxis().FindBin(HVal.binCenters[binning.var])
        histData.SetBinContent(binIndex, y)
        histData.SetBinError  (binIndex, 1e-100 if yErr < 1e-100 else yErr)  # ROOT does not draw points if uncertainty is zero; sigh
      setCbFriendlyStyle(histData, index)
      histStack.Add(histData, "PE1X0")
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

  # define what to overlay
  fitResults: tuple[tuple[str, str], ...] = (
    # (directory name, legend label)
    ("./plotsPhotoProdPiPiUnpol.maxL_2",  "#it{L}_{max} = 2"),
    # ("./plotsPhotoProdPiPiUnpol.maxL_4",  "#it{L}_{max} = 4"),
    ("./plotsPhotoProdPiPiUnpol.maxL_5",  "#it{L}_{max} = 5"),
    ("./plotsPhotoProdPiPiUnpol.maxL_8",  "#it{L}_{max} = 8"),
    ("./plotsPhotoProdPiPiUnpol.maxL_20", "#it{L}_{max} = 20"),
    # last entry defines which moments are plotted
  )
  # fitResults: tuple[tuple[str, str], ...] = (
  #   ("./plotsPhotoProdPiPiUnpol.rhoMc",  "Old MC"),
  #   ("./plotsPhotoProdPiPiUnpol.maxL_5", "New MC"),
  # )
  outFileDirName   = Utilities.makeDirPath("./plotsPhotoProdPiPiUnpolOverlay")
  normalizeMoments = False
  binVarMass       = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}", unit = "GeV/#it{c}^{2}", nmbDigits = 3)
  massBinning      = HistAxisBinning(nmbBins = 100, minVal = 0.4, maxVal = 1.4, _var = binVarMass)  # same binning as used by CLAS

  namePrefix = "norm" if normalizeMoments else "unnorm"

  # load moment results
  momentResultsToOverlay: dict[str, MomentResultsKinematicBinning] = {}  # key: legend label, value: moment results
  for fitResultDirName, fitResultLabel in fitResults:
    print(f"Loading moment results from directory {fitResultDirName}")
    momentResultsPhysFileName = f"{fitResultDirName}/{namePrefix}_moments_phys.pkl"
    try:
      momentResultsPhys = MomentResultsKinematicBinning.load(momentResultsPhysFileName)
    except FileNotFoundError as e:
      print(f"Cannot not find file '{momentResultsPhysFileName}'. Skipping directory '{fitResultDirName}'")
      continue
    momentResultsToOverlay[fitResultLabel] = momentResultsPhys

  # ensure that all moment results have identical kinematic binning and identical order of kinematic bins
  momentResults: tuple[MomentResultsKinematicBinning, ...]         = tuple(momentResultsToOverlay.values())
  binCenters:    tuple[dict[KinematicBinningVariable, float], ...] = momentResults[0].binCenters  # bin centers of first moment result
  for momentResult in momentResults[1:]:
    assert momentResult.binCenters == binCenters

  # plot kinematic dependences of all moments
  lastLabel = fitResults[-1][1]
  for qnIndex in momentResultsToOverlay[lastLabel][0].indices.qnIndices:
    overlayMoments1D(
      momentResultsToOverlay = momentResultsToOverlay,
      qnIndex                = qnIndex,
      binning                = massBinning,
      normalizedMoments      = normalizeMoments,
      pdfFileNamePrefix      = f"{outFileDirName}/{namePrefix}_{massBinning.var.name}_",
    )

  timer.stop("Total execution time")
  print(timer.summary)
