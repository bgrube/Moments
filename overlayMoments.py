#!/usr/bin/env python3


from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import functools

import ROOT

from MomentCalculator import (
  KinematicBinningVariable,
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
from AnalysisConfig import (
  CFG_KEVIN,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_JPAC,
  CFG_UNPOLARIZED_PIPI_PWA,
)
from PlottingUtilities import (
  HistAxisBinning,
  MomentValue,
  setCbFriendlyStyle,
  setupPlotStyle,
)
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
      setCbFriendlyStyle(histData, styleIndex = index, filledMarkers = True)
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
  cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarizedK- K_S Delta++ data
  # cfg.outFileDirBaseName += ".ideal"
  cfg.init()
  # normToFirstResult = True  # if set moments are normalized to H_0(0, 0) of first fit result
  normToFirstResult = False
  # tBinLabel         = "tbin_0.1_0.2"
  # tBinLabel         = "tbin_0.1_0.2.Gj.pi+"
  # tBinLabel         = "tbin_0.1_0.2.trackDistFdc"
  # tBinLabel         = "tbin_0.2_0.3"
  # tBinLabel         = "tbin_0.1_0.5"
  tBinLabel         = "tbin_0.4_0.5"
  fitResults: tuple[tuple[str, str, float | None], ...] = (  # tuple: (<directory name>, <legend label>, optional: <scale factor>); last entry defines which moments are plotted
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/PARA_0.maxL_4",   "Para #Phi_{0} = 0#circ",   None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/PARA_135.maxL_4", "Para #Phi_{0} = 135#circ", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/PERP_45.maxL_4",  "Perp #Phi_{0} = 45#circ",  None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/PERP_90.maxL_4",  "Perp #Phi_{0} = 90#circ",  None),
    #
    # (f"{cfg.outFileDirBaseName}.tbin_0.1_0.2/PARA_0.maxL_4", "0.1 < #minus t < 0.2 GeV^{2}", None),
    # (f"{cfg.outFileDirBaseName}.tbin_0.2_0.3/PARA_0.maxL_4", "0.2 < #minus t < 0.3 GeV^{2}", None),
    #
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/0_90.maxL_4",      "0#circ and 90#circ",               None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/-45_45.maxL_4",    "#minus 45#circ and #plus 45#circ", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/0_-45.maxL_4",     "0#circ and #minus 45#circ",        None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/45_90.maxL_4",     "45#circ and 90#circ",              None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/allOrient.maxL_4", "4 Orientations",                   0.5),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/allOrient.maxL_4", "4 Orientations",                   None),
    #
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/0_90.maxL_4", "0#circ and 90#circ, L_{max} = 4", None),
    # # (f"{cfg.outFileDirBaseName}.{tBinLabel}/0_90.maxL_5", "0#circ and 90#circ, L_{max} = 5", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/0_90.maxL_6", "0#circ and 90#circ, L_{max} = 6", None),
    # # (f"{cfg.outFileDirBaseName}.{tBinLabel}/0_90.maxL_7", "0#circ and 90#circ, L_{max} = 7", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/0_90.maxL_8", "0#circ and 90#circ, L_{max} = 8", None),
    #
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/allOrient.maxL_4", "4 Orientations, L_{max} = 4", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/allOrient.maxL_6", "4 Orientations, L_{max} = 6", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/allOrient.maxL_8", "4 Orientations, L_{max} = 8", None),
    #
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/allOrient.maxL_4",     "New", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/allOrient.maxL_4.bak", "Old", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/PARA_0.maxL_4",        "#pi^{-} Analyzer", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}.Hf.pi+/PARA_0.maxL_4", "#pi^{+} Analyzer", None),
    #
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/PARA_0.maxL_4",     "LinAlg", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}.fit/PARA_0.maxL_4", "Fit",    None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}/PARA_0.maxL_4",     "New",  None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}.bak/PARA_0.maxL_4", "Orig", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}.fit/PARA_0.maxL_4", "Orig", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}.linAlg/Unpol.maxL_4", "LinAlg", None),
    # (f"{cfg.outFileDirBaseName}.{tBinLabel}.fit/Unpol.maxL_4",    "Fit",    None),
    (f"{cfg.outFileDirBaseName}.{tBinLabel}.bak/Unpol.maxL_4", "JPAC", None),
    (f"{cfg.outFileDirBaseName}.{tBinLabel}/Unpol.maxL_4",     "HDDM", None),
  )
  outputDirName = Utilities.makeDirPath(f"{cfg.outFileDirBaseName}.{tBinLabel}.overlay")

  # load moment results
  momentResultsToOverlay: dict[str, tuple[MomentResultsKinematicBinning, float | None]] = {}  # key: legend label, value: (moment results, optional scale factor)
  for fitResultDirName, fitResultLabel, scaleFactor in fitResults:
    print(f"Loading moment results from directory {fitResultDirName}")
    momentResultsPhysFileName = f"{fitResultDirName}/{cfg.outFileNamePrefix}_moments_phys.pkl"
    try:
      momentResultsPhys = MomentResultsKinematicBinning.loadPickle(momentResultsPhysFileName)
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
    # set scale factors such that all moments are normalized to H_0(0, 0) of the first fit result
    firstMomentResults = list(momentResultsToOverlay.values())[0][0]
    for label, (moments, _) in momentResultsToOverlay.items():
      scaleFactor = moments.normalizeTo(
        firstMomentResults,
        normBinIndex = None,  # normalize to integral over mass bins
      )
      print(f"Applying scale factor {scaleFactor} to fit result '{label}'")
      momentResultsToOverlay[label] = (moments, scaleFactor)

  # plot kinematic dependences of all moments
  lastLabel = fitResults[-1][1]
  for qnIndex in momentResultsToOverlay[lastLabel][0][0].indices.qnIndices:
    overlayMoments1D(
      momentResultsToOverlay = momentResultsToOverlay,
      qnIndex                = qnIndex,
      binning                = cfg.massBinning,
      normalizedMoments      = cfg.normalizeMoments,
      pdfFileNamePrefix      = f"{outputDirName}/{cfg.outFileNamePrefix}_phys_{cfg.massBinning.var.name}_",
    )

  timer.stop("Total execution time")
  print(timer.summary)
