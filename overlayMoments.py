#!/usr/bin/env python3


from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import functools
import math

import ROOT

from MomentCalculator import (
  KinematicBinningVariable,
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
from photoProdCalcMoments import (
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_PWA,
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


def getHistFromMomentValues(
  HVals:      Sequence[MomentValue],
  binning:    HistAxisBinning,
  momentPart: str,  # "Re" or "Im"
  histName:   str = "",
  histTitle:  str = "",
) -> ROOT.TH1D:
  """Creates histogram with given binning from moment values"""
  histData = ROOT.TH1D(histName, histTitle, *binning.astuple)
  for HVal in HVals:
    if binning.var not in HVal.binCenters.keys():
      continue
    y, yErr = HVal.part(real = (momentPart == "Re"))
    binIndex = histData.GetXaxis().FindBin(HVal.binCenters[binning.var])
    histData.SetBinContent(binIndex, y)
    histData.SetBinError  (binIndex, 1e-100 if yErr < 1e-100 else yErr)  # ROOT does not draw points if uncertainty is zero; sigh
  return histData


def overlayMoments1D(
  momentResultsToOverlay: dict[str, tuple[MomentResultsKinematicBinning, float | None]],  # key: legend label, value: (moment results, optional scale factor)
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
    for index, (legendLabel, (momentResults, scaleFactor)) in enumerate(momentResultsToOverlay.items()):
      # filter out specific moment given by qnIndex
      HVals: tuple[MomentValue, ...] = tuple(HPhys[qnIndex] for HPhys in momentResults if qnIndex in HPhys)
      histData = getHistFromMomentValues(HVals, binning, momentPart, legendLabel)
      setCbFriendlyStyle(histData, styleIndex = index)
      if scaleFactor is not None:
        histData.Scale(scaleFactor)
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
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  normToFirstResult = True  # if set moments are normalized to H_0(0, 0) of first fit result
  fitResults: tuple[tuple[str, str, float | None], ...] = (  # tuple: (<directory name>, <legend label>, optional: <scale factor>); last entry defines which moments are plotted
    # (f"{cfg._outFileDirBaseName}.tbin_0.1_0.2.PARA_0.maxL_4",   "#Phi_{0} = 0#circ",   None),  # 1.419187e7 events, 4.345e6 combos
    # (f"{cfg._outFileDirBaseName}.tbin_0.1_0.2.PARA_135.maxL_4", "#Phi_{0} = 135#circ", None),  # 1.387172e7 events, 4.209e6 combos
    # (f"{cfg._outFileDirBaseName}.tbin_0.1_0.2.PERP_45.maxL_4",  "#Phi_{0} = 45#circ",  None),  # 1.335436e7 events, 4.100e6 combos
    # (f"{cfg._outFileDirBaseName}.tbin_0.1_0.2.PERP_90.maxL_4",  "#Phi_{0} = 90#circ",  None),  # 1.529724e7 events, 4.672e6 combos
    #
    (f"{cfg._outFileDirBaseName}.tbin_0.2_0.3.PARA_0.maxL_4",   "#Phi_{0} = 0#circ",   None),
    (f"{cfg._outFileDirBaseName}.tbin_0.2_0.3.PARA_135.maxL_4", "#Phi_{0} = 135#circ", None),
    (f"{cfg._outFileDirBaseName}.tbin_0.2_0.3.PERP_45.maxL_4",  "#Phi_{0} = 45#circ",  None),
    (f"{cfg._outFileDirBaseName}.tbin_0.2_0.3.PERP_90.maxL_4",  "#Phi_{0} = 90#circ",  None),
    #
    # (f"{cfg._outFileDirBaseName}.tbin_0.1_0.2.PARA_0.maxL_4", "0.1 < #minus t < 0.2 GeV^{2}", None),
    # (f"{cfg._outFileDirBaseName}.tbin_0.2_0.3.PARA_0.maxL_4", "0.2 < #minus t < 0.3 GeV^{2}", None),
  )
  outFileDirName = Utilities.makeDirPath(f"./plotsPhotoProdPiPi{'Unpol' if cfg.polarization is None else 'Pol'}.overlay")

  # load moment results
  momentResultsToOverlay: dict[str, tuple[MomentResultsKinematicBinning, float | None]] = {}  # key: legend label, value: (moment results, optional scale factor)
  for fitResultDirName, fitResultLabel, scaleFactor in fitResults:
    print(f"Loading moment results from directory {fitResultDirName}")
    momentResultsPhysFileName = f"{fitResultDirName}/{cfg.outFileNamePrefix}_moments_phys.pkl"
    try:
      momentResultsPhys = MomentResultsKinematicBinning.load(momentResultsPhysFileName)
    except FileNotFoundError as e:
      print(f"Cannot not find file '{momentResultsPhysFileName}'. Skipping directory '{fitResultDirName}'")
      continue
    momentResultsToOverlay[fitResultLabel] = (momentResultsPhys, scaleFactor)

  # ensure that all moment results have identical kinematic binning and identical order of kinematic bins
  momentResults: tuple[MomentResultsKinematicBinning, ...]         = tuple(value[0] for value in momentResultsToOverlay.values())
  binCenters:    tuple[dict[KinematicBinningVariable, float], ...] = momentResults[0].binCenters  # bin centers of first moment result
  for momentResult in momentResults[1:]:
    assert momentResult.binCenters == binCenters

  if normToFirstResult:
    # get integrals of H_0(0, 0) moments of all fit results
    H000Integrals: list[float] = []
    H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
    for moments, _ in momentResultsToOverlay.values():
      # get H_0(0, 0) moment values in all mass bins
      H000Vals: tuple[MomentValue, ...] = tuple(HPhys[H000Index] for HPhys in moments if H000Index in HPhys)
      # calculate integral of H_0(0, 0) moment
      H000Integrals.append(getHistFromMomentValues(H000Vals, cfg.massBinning, momentPart = "Re").Integral())
    # set scale factors such that all moments are normalized to H_0(0, 0) of the first fit result
    scaleFactors = [H000Integrals[0] / integral for integral in H000Integrals]
    for index, label in enumerate(momentResultsToOverlay.keys()):
      print(f"Applying scale factor {scaleFactors[index]} to fit result '{label}'")
      momentResultsToOverlay[label] = (momentResultsToOverlay[label][0], scaleFactors[index])

  # plot kinematic dependences of all moments
  lastLabel = fitResults[-1][1]
  for qnIndex in momentResultsToOverlay[lastLabel][0][0].indices.qnIndices:
    overlayMoments1D(
      momentResultsToOverlay = momentResultsToOverlay,
      qnIndex                = qnIndex,
      binning                = cfg.massBinning,
      normalizedMoments      = cfg.normalizeMoments,
      pdfFileNamePrefix      = f"{outFileDirName}/{cfg.outFileNamePrefix}_phys_{cfg.massBinning.var.name}_",
    )

  timer.stop("Total execution time")
  print(timer.summary)
