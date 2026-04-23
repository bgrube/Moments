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
  CFG_POLARIZED_ETAPI0,
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
  styleIndexOffset:       int  = 0,     # allows to offset style indices for overlaid plots
  styleIndexStride:       int  = 1,     # step size, by which style indices are incremented
  yAxisUnit:              str  = "",    # allows to override default y-axis title
) -> None:
  """Overlays moments H_i(L, M) from different analyses as function of kinematical variable"""
  print(f"Overlaying {qnIndex.label} moments as a function of the '{binning.var.name}' variable")
  for momentPart, momentPartLabel in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
    histStack = ROOT.THStack(
      f"{pdfFileNamePrefix}overlay_{qnIndex.label}_{momentPart}",
      f"{qnIndex.title} {momentPartLabel};{binning.axisTitle};" + (("Normalized" if normalizedMoments else "Unnormalized") + " Moment Value") + yAxisUnit,
    )
    for index, (legendLabel, (momentResults, scaleFactor)) in enumerate(momentResultsToOverlay.items()):
      # filter out specific moment given by qnIndex
      HVals: tuple[MomentValue, ...] = tuple(HPhys[qnIndex] for HPhys in momentResults if qnIndex in HPhys)
      histData = getHistFromMomentValues(
        HVals      = HVals,
        binning    = binning,
        momentPart = momentPart,
        histName   = legendLabel,
      )
      setCbFriendlyStyle(
        graphOrHist   = histData,
        styleIndex    = index * styleIndexStride + styleIndexOffset,
        filledMarkers = True,
      )
      if scaleFactor is not None:
        print(f"Applying scale factor {scaleFactor} to fit result '{legendLabel}'")
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
  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarizedK- K_S Delta++ data
  cfg = deepcopy(CFG_POLARIZED_ETAPI0)  # perform analysis of Nizar's polarized eta pi0 data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg.outFileDirBaseName += ".ideal"
  cfg.init()
  # cfg.polarization = None  # treat data as unpolarized

  # normToFirstResult = True  # if set moments are normalized to H_0(0, 0) of first fit result
  normToFirstResult = False
  dataPeriods = (
    "merged",
    # "2017_01",
    # "2017_01_ver05",
    # "2018_08",
  )
  tBinLabels = (
    "t010020",
    "t020032",
    "t032050",
    "t050075",
    "t075100",
    # "tbin_0.100_0.114",  # lowest |t| bin of SDME analysis
    # "tbin_0.1_0.2",
    # "tbin_0.2_0.3",
    # "tbin_0.3_0.4",
    # "tbin_0.4_0.5",
  )
  crossSectionScaleFactors = {
    # [ub / GeV^3] = 1 / ([40 MeV mass bin width] * [0.1 GeV^2 t bin width] * L)
    "2017_01" : 1.0 / (0.04 * 0.1 * 21.360196 * 1e6),  #  L(Spring 2017) = 21.360196 pb^{-1}
    "2018_08" : 1.0 / (0.04 * 0.1 * 39.260175 * 1e6),  #  L(Fall 2018)   = 39.260175 pb^{-1}
  }


  for dataPeriod in dataPeriods:
    for tBinLabel in tBinLabels:
      # scale factors to match Spring 2017 H_0(0, 0) integral for L_max = 4
      scaleFactor_2018_08_allOrient = 0.4916841615225002 if tBinLabel == "tbin_0.1_0.2" else \
                                      0.5159984154089572 if tBinLabel == "tbin_0.2_0.3" else \
                                      None
      scaleFactor_2018_08_PARA_0 = 1.8352200424810305  # scale factor to match Spring 2017 H_0(0, 0) integral for L_max = 4
      scaleFactor_2018_08_AMO = 7.241007048362434  # scale factor to match Fall 2018 PARA 0 H_0(0, 0) integral for L_max = 4
      fitResults: tuple[tuple[str, str, float | None], ...] = (  # tuple: (<directory name>, <legend label>, optional: <scale factor>); last fit result defines which moments are plotted
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/PARA_0.maxL_4",          "Acc. Corr. Reco.", None),
        # (f"{cfg.outFileDirBaseName}.accTruth/{dataPeriod}/{tBinLabel}/PARA_0.maxL_4", "Acc. Corr. Truth", None),
        # (f"{cfg.outFileDirBaseName}.weightedMc.maxL_4/{dataPeriod}/{tBinLabel}/PARA_0.maxL_4",          "Acc. Corr. Reco.", None),
        # (f"{cfg.outFileDirBaseName}.weightedMc.accTruth.maxL_4/{dataPeriod}/{tBinLabel}/PARA_0.maxL_4", "Acc. Corr. Truth", None),
        #
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/PARA_0.maxL_4",       "Fall 2018, Old MC", None),
        # (f"{cfg.outFileDirBaseName}.oldMc/{dataPeriod}/{tBinLabel}/PARA_0.maxL_4", "Fall 2018, New MC", None),
        # (f"{cfg.outFileDirBaseName}/2017_01/{tBinLabel}/PARA_0.maxL_4",            "Spring 2017",       2.1646133913963),
        # (f"{cfg.outFileDirBaseName}/2018_08/{tBinLabel}/PARA_0.maxL_4",            "Fall 2018",         2.1646133913963),
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/allOrient.maxL_4",  "L_{max} = 4",  None),
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/allOrient.maxL_8",  "L_{max} = 8",  None),
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/allOrient.maxL_12", "L_{max} = 12", None),
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/allOrient.maxL_16", "L_{max} = 16", None),
        #
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/allOrient.maxL_4", "Spring 2017, L_{max} = 4", None),
        # (f"{cfg.outFileDirBaseName}/2018_08/{tBinLabel}/PARA_0.maxL_4",         "Fall 2018 PARA 0, L_{max} = 4",   scaleFactor_2018_08_PARA_0),  # scale factor to match Spring 2017 H_0(0, 0) integral
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/allOrient.maxL_8", "Spring 2017, L_{max} = 8", None),
        # (f"{cfg.outFileDirBaseName}/2018_08/{tBinLabel}/PARA_0.maxL_8",         "Fall 2018 PARA 0, L_{max} = 8",   scaleFactor_2018_08_PARA_0),  # use same scale factor as for L_max = 4
        #
        # (f"{cfg.outFileDirBaseName}/2018_08/{tBinLabel}/PARA_0.maxL_4", "PARA 0, L_{max} = 4", None),
        # (f"{cfg.outFileDirBaseName}/2018_08/{tBinLabel}/AMO.maxL_4",    "AMO, L_{max} = 4",    scaleFactor_2018_08_AMO),
        # (f"{cfg.outFileDirBaseName}/2018_08/{tBinLabel}/PARA_0.maxL_8", "PARA 0, L_{max} = 8", None),
        # (f"{cfg.outFileDirBaseName}/2018_08/{tBinLabel}/AMO.maxL_8",    "AMO, L_{max} = 8",    scaleFactor_2018_08_AMO),
        # #
        # (f"{cfg.outFileDirBaseName}.SDME/2017_01/tbin_0.1_0.2/PARA_0.maxL_4",     "ver04, L_{max} = 4", 0.1791785678624324),  # normalized to H_0(0, 0) for ver05, L_max = 4
        # (f"{cfg.outFileDirBaseName}.SDME/{dataPeriod}/{tBinLabel}/PARA_0.maxL_4", "ver05, L_{max} = 4", None),
        # (f"{cfg.outFileDirBaseName}.SDME/2017_01/tbin_0.1_0.2/PARA_0.maxL_8",     "ver04, L_{max} = 8", 0.1791785678624324),
        # (f"{cfg.outFileDirBaseName}.SDME/{dataPeriod}/{tBinLabel}/PARA_0.maxL_8", "ver05, L_{max} = 8", None),
        #
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/allOrient.maxL_4", "Spring 2017, L_{max} = 4", crossSectionScaleFactors[dataPeriod]),
        # (f"{cfg.outFileDirBaseName}/2018_08/{tBinLabel}/allOrient.maxL_4",      "Fall 2018, L_{max} = 4",   crossSectionScaleFactors["2018_08"]),
        #
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/All.maxL_4",     "GJ, L_{max} = 4", None),
        # (f"{cfg.outFileDirBaseName}.bak/{dataPeriod}/{tBinLabel}/All.maxL_4", "HF, L_{max} = 4", None),
        (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/All.maxL_4", "L_{max} = 4", None),
        (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/All.maxL_5", "L_{max} = 5", None),
        (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/All.maxL_6", "L_{max} = 6", None),
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/All.maxL_7", "L_{max} = 7", None),
        # (f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}/All.maxL_8", "L_{max} = 8", None),
      )
      outputDirName = Utilities.makeDirPath(f"{cfg.outFileDirBaseName}/{dataPeriod}/{tBinLabel}.overlay")
      # outputDirName = Utilities.makeDirPath(f"{cfg.outFileDirBaseName}/{tBinLabel}.overlay")

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

      # plot kinematic dependences of all moments
      lastLabel = fitResults[-1][1]
      for qnIndex in momentResultsToOverlay[lastLabel][0][0].indices.qnIndices:
        overlayMoments1D(
          momentResultsToOverlay = momentResultsToOverlay,
          qnIndex                = qnIndex,
          binning                = cfg.massBinning,
          normalizedMoments      = cfg.normalizeMoments,
          pdfFileNamePrefix      = f"{outputDirName}/{cfg.outFileNamePrefix}_phys_{cfg.massBinning.var.name}_",
          # styleIndexOffset       = 1,
          # styleIndexStride       = 2,
          # yAxisUnit              = " [#mub/GeV^{3}]",
        )

  timer.stop("Total execution time")
  print(timer.summary)
