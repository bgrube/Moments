#!/usr/bin/env python3


from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import functools

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from moments.MomentCalculator import (
  KinematicBinningVariable,
  MomentResultsKinematicBinning,
  MomentValue,
  QnMomentIndex,
)
from workflow.AnalysisConfig import (
  CFG_KEVIN,
  CFG_POLARIZED_ETAPI0,
  CFG_POLARIZED_KSKL,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_ETAPETA,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_JPAC,
  CFG_UNPOLARIZED_PIPI_PWA,
)
from workflow.PlottingUtilities import (
  HistAxisBinning,
  setCbFriendlyStyle,
  setupPlotStyle,
)
from workflow import Utilities


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


@dataclass
class ResultToOverlay:
  """Stores input data that define a single overlay of moment results"""
  pklFilePath:    str  # path to pickled MomentResultsKinematicBinning file
  label:          str  # legend label for moment values
  scaleFactor:    float | None = None  # optional scale factor to apply to moment values
  _momentResults: MomentResultsKinematicBinning | None = None  # moment results loaded from pklFilePath

  @property
  def momentResults(self) -> MomentResultsKinematicBinning:
    """Returns moment results and ensures it exists"""
    assert self._momentResults is not None, f"Moment results for '{self.label}' have not been loaded yet. Call loadMomentResults() first."
    return self._momentResults

  def loadMomentResults(self) -> MomentResultsKinematicBinning:
    """Loads moment results from pklFilePath and caches them in _momentResults"""
    print(f"Loading moment results from file '{self.pklFilePath}'")
    try:
      self._momentResults = MomentResultsKinematicBinning.loadPickle(self.pklFilePath)
    except FileNotFoundError as e:
      print(f"Cannot not find file '{self.pklFilePath}'. Skipping.")
      raise
    return self._momentResults


def overlayMoments1D(
  resultsToOverlay:  Sequence[ResultToOverlay],
  qnIndex:           QnMomentIndex,    # defines specific moment
  binning:           HistAxisBinning,  # binning to use for plot
  normalizedMoments: bool = True,  # indicates whether moment values were normalized to H_0(0, 0)
  pdfFileNamePrefix: str  = "",    # name prefix for output files
  styleIndexOffset:  int  = 0,     # allows to offset style indices for overlaid plots
  styleIndexStride:  int  = 1,     # step size, by which style indices are incremented
  yAxisUnit:         str  = "",    # allows to override default y-axis title
) -> None:
  """Overlays moments from different analyses as function of kinematical variable"""
  print(f"Overlaying {qnIndex.label} moments as a function of the '{binning.var.name}' variable")
  for momentPart, momentPartLabel in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
    histStack = ROOT.THStack(
      f"{pdfFileNamePrefix}overlay_{qnIndex.label}_{momentPart}",
      f"{qnIndex.title} {momentPartLabel};{binning.axisTitle};" + (("Normalized" if normalizedMoments else "Unnormalized") + " Moment Value") + yAxisUnit,
    )
    for overlayIndex, resultToOverlay in enumerate(resultsToOverlay):
      # filter out specific moment given by qnIndex
      HVals: tuple[MomentValue, ...] = tuple(momentResult[qnIndex] for momentResult in resultToOverlay.momentResults if qnIndex in momentResult)
      histData = getHistFromMomentValues(
        HVals      = HVals,
        binning    = binning,
        momentPart = momentPart,
        histName   = resultToOverlay.label,
      )
      setCbFriendlyStyle(
        graphOrHist   = histData,
        styleIndex    = overlayIndex * styleIndexStride + styleIndexOffset,
        filledMarkers = True,
      )
      if resultToOverlay.scaleFactor is not None:
        print(f"Applying scale factor {resultToOverlay.scaleFactor} to moment result '{resultToOverlay.label}'")
        histData.Scale(resultToOverlay.scaleFactor)
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
  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarizedK- K_S Delta++ data
  # cfg = deepcopy(CFG_UNPOLARIZED_ETAPETA)  # perform analysis of Will's unpolarized eta' eta data
  # cfg = deepcopy(CFG_POLARIZED_ETAPI0)  # perform analysis of Nizar's polarized eta pi0 data
  cfg = deepcopy(CFG_POLARIZED_KSKL)  # perform analysis of Gabriel's polarized K_S K_L data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg.polarization = None  # treat data as unpolarized

  normToFirstResult = True  # if set moments are normalized to H_0(0, 0) of first moment result
  # normToFirstResult = False
  crossSectionScaleFactors = {
    # [ub / GeV^3] = 1 / ([40 MeV mass bin width] * [0.1 GeV^2 t bin width] * L)
    "2017_01" : 1.0 / (0.04 * 0.1 * 21.360196 * 1e6),  #  L(Spring 2017) = 21.360196 pb^{-1}
    "2018_08" : 1.0 / (0.04 * 0.1 * 39.260175 * 1e6),  #  L(Fall 2018)   = 39.260175 pb^{-1}
  }


  for dataPeriod in cfg.dataPeriods:
    for tBinLabel in cfg.tBinLabels:
      # scale factors to match Spring 2017 H_0(0, 0) integral for L_max = 4
      scaleFactor_2018_08_allOrient = 0.4916841615225002 if tBinLabel == "tbin_0.1_0.2" else \
                                      0.5159984154089572 if tBinLabel == "tbin_0.2_0.3" else \
                                      None
      scaleFactor_2018_08_PARA_0 = 1.8352200424810305  # scale factor to match Spring 2017 H_0(0, 0) integral for L_max = 4
      scaleFactor_2018_08_AMO = 7.241007048362434  # scale factor to match Fall 2018 PARA 0 H_0(0, 0) integral for L_max = 4
      resultsToOverlay: tuple[ResultToOverlay, ...] = (  # last moment result in this tuple defines, which moments are plotted
        # # eta pi0
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/All.maxL_4/{cfg.outFileNamePrefix}_moments_phys.pkl",     "GJ, L_{max} = 4", None),
        # ResultToOverlay(f"{cfg.outFileDirBasePath}.bak/{dataPeriod}/{tBinLabel}/All.maxL_4/{cfg.outFileNamePrefix}_moments_phys.pkl", "HF, L_{max} = 4", None),
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/All.maxL_4/{cfg.outFileNamePrefix}_moments_phys.pkl", "Physical", None),
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/All.maxL_4/{cfg.outFileNamePrefix}_moments_phys.pkl", "Measured", None),
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/All.maxL_5/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 5", None),
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/All.maxL_6/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 6", None),
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/All.maxL_7/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 7", None),
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/All.maxL_8/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 8", None),
        # #
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/Unpol.maxL_4/{cfg.outFileNamePrefix}_moments_phys.pkl", "LOWT",   None),
        # ResultToOverlay(f"{cfg.outFileDirBasePath}/{dataPeriod}/XSCUTS/Unpol.maxL_4/{cfg.outFileNamePrefix}_moments_phys.pkl",      "XSCUTS", None),
        # K_S K_L
        ResultToOverlay(f"{cfg.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel = 'PARA_0', maxL = 4)}/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 4"),
        ResultToOverlay(f"{cfg.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel = 'PARA_0', maxL = 6)}/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 6"),
        ResultToOverlay(f"{cfg.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel = 'PARA_0', maxL = 8)}/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 8"),
      )
      outputDirPath = Utilities.makeDirPath(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}.overlay")

      # load moment results
      for resultToOverlay in resultsToOverlay:
        resultToOverlay.loadMomentResults()

      # ensure that all moment results have identical kinematic binning and identical order of kinematic bins
      momentResults: tuple[MomentResultsKinematicBinning, ...]         = tuple(resultToOverlay.momentResults for resultToOverlay in resultsToOverlay)
      binCenters:    tuple[dict[KinematicBinningVariable, float], ...] = momentResults[0].binCenters  # bin centers of first moment result
      for momentResult in momentResults[1:]:
        assert momentResult.binCenters == binCenters

      if normToFirstResult:
        # set scale factors such that all moments are normalized to H_0(0, 0) of the first moment result
        firstMomentResults = resultsToOverlay[0].momentResults
        for resultToOverlay in resultsToOverlay:
          scaleFactor = resultToOverlay.momentResults.normalizeTo(
            firstMomentResults,
            normBinIndex = None,  # normalize to integral over mass bins
          )
          print(f"Applying scale factor {scaleFactor} to moment result '{resultToOverlay.label}'")

      # plot kinematic dependences of all moments
      for qnIndex in resultsToOverlay[-1].momentResults[0].indices.qnIndices:
        overlayMoments1D(
          resultsToOverlay  = resultsToOverlay,
          qnIndex           = qnIndex,
          binning           = cfg.massBinning,
          normalizedMoments = cfg.normalizeMoments,
          pdfFileNamePrefix = f"{outputDirPath}/{cfg.outFileNamePrefix}_phys_{cfg.massBinning.var.name}_",
          # styleIndexOffset  = 1,
          # styleIndexStride  = 2,
          # yAxisUnit         = " [#mub/GeV^{3}]",
        )

  timer.stop("Total execution time")
  print(timer.summary)
