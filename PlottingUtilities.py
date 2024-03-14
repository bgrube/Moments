"""Module that provides a collection of functions for plotting"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import matplotlib.pyplot as plt
import numpy as np
import nptyping as npt
import os
from scipy import stats
from typing import (
  Any,
  Dict,
  Iterator,
  List,
  Optional,
  Sequence,
  Tuple,
)

import ROOT

import MomentCalculator


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class MomentValueAndTruth(MomentCalculator.MomentValue):
  """Stores and provides access to single moment value and provides truth value"""
  truth:       Optional[complex] = None  # true moment value
  _binCenters: Optional[Dict[MomentCalculator.KinematicBinningVariable, float]] = None  # dictionary with bin centers

  # accessor that guarantees existence of optional field
  @property
  def binCenters(self) -> Dict[MomentCalculator.KinematicBinningVariable, float]:
    """Returns true moment value"""
    assert self._binCenters is not None, "self._binCenters must not be None"
    return self._binCenters

  def truthRealOrImag(
    self,
    realPart: bool,  # switched between real part (True) and imaginary part (False)
  ) -> float:
    """Returns real or imaginary part with corresponding uncertainty according to given flag"""
    assert self.truth is not None, "self.truth must not be None"
    if realPart:
      return self.truth.real
    else:
      return self.truth.imag


@dataclass
class HistAxisBinning:
  """Stores info that defines equidistant binning of an axis"""
  nmbBins: int    # number of bins
  minVal:  float  # lower limit
  maxVal:  float  # upper limit
  _var:    Optional[MomentCalculator.KinematicBinningVariable] = None  # optional info about bin variable

  def __len__(self) -> int:
    """Returns number of bins"""
    return self.nmbBins

  def __getitem__(
    self,
    subscript: int,
  ) -> float:
    """Returns bin center for given bin index"""
    if subscript < self.nmbBins:
      return self.minVal + (subscript + 0.5) * self.binWidth
    raise IndexError

  def __iter__(self) -> Iterator[float]:
    """Iterates over bin centers"""
    for i in range(len(self)):
      yield self[i]

  # accessor that guarantees existence of optional field
  @property
  def var(self) -> MomentCalculator.KinematicBinningVariable:
    """Returns info about binning variable"""
    assert self._var is not None, "self._var must not be None"
    return self._var

  @property
  def astuple(self) -> Tuple[int, float, float]:
    """Returns tuple with binning info that can be directly used in ROOT.THX() constructor"""
    return (self.nmbBins, self.minVal, self.maxVal)

  @property
  def axisTitle(self) -> str:
    """Returns axis title if info about binning variable is present"""
    if self._var is None:
      return ""
    return self._var.axisTitle

  @property
  def valueIntervalLength(self) -> float:
    """Returns length of value interval covered by binning, i.e. (maximum value - minimum value)"""
    if self.maxVal < self.minVal:
      self.maxVal, self.minVal = self.minVal, self.maxVal
    return self.maxVal - self.minVal

  @property
  def binWidth(self) -> float:
    """Returns bin width"""
    return self.valueIntervalLength / self.nmbBins

  def binValueRange(
    self,
    binIndex: int,
  ) -> Tuple[float, float]:
    """Returns value range for bin with given index"""
    binCenter = self[binIndex]
    return (binCenter - 0.5 * self.binWidth, binCenter + 0.5 * self.binWidth)

  @property
  def binValueRanges(self) -> Tuple[Tuple[float, float], ...]:
    """Returns value ranges for all bins"""
    return tuple((binCenter - 0.5 * self.binWidth, binCenter + 0.5 * self.binWidth) for binCenter in self)


def setupPlotStyle(rootlogonPath: str = "./rootlogon.C") -> None:
  """Defines ROOT plotting style"""
  ROOT.gROOT.LoadMacro(rootlogonPath)
  ROOT.gROOT.ForceStyle()
  ROOT.gStyle.SetCanvasDefW(600)
  ROOT.gStyle.SetCanvasDefH(600)
  ROOT.gStyle.SetPalette(ROOT.kBird)
  # ROOT.gStyle.SetPalette(ROOT.kViridis)
  ROOT.gStyle.SetLegendFillColor(ROOT.kWhite)
  ROOT.gStyle.SetLegendBorderSize(1)
  # ROOT.gStyle.SetOptStat("ni")  # show only name and integral
  # ROOT.gStyle.SetOptStat("i")  # show only integral
  ROOT.gStyle.SetOptStat("")
  ROOT.gStyle.SetStatFormat("8.8g")
  ROOT.gStyle.SetTitleColor(1, "X")  # fix that for some mysterious reason x-axis titles of 2D plots and graphs are white
  ROOT.gStyle.SetTitleOffset(1.35, "Y")


def plotComplexMatrix(
  complexMatrix:     npt.NDArray[npt.Shape["*, *"], npt.Complex128],  # matrix to plot
  pdfFileNamePrefix: str,  # name prefix for output files
) -> None:
  """Draws real and imaginary parts of given 2D array"""
  matricesToPlot = {
    "real" : np.real(complexMatrix),      # real part
    "imag" : np.imag(complexMatrix),      # imaginary part
    "abs"  : np.absolute(complexMatrix),  # absolute value
    "arg"  : np.angle(complexMatrix),     # phase
  }
  #TODO add plot titles
  #TODO add axis titles
  #TODO use same z-scale for all plots
  for plotLabel, matrix in matricesToPlot.items():
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    plt.savefig(f"{pdfFileNamePrefix}_{plotLabel}.pdf", transparent = True)
    plt.close(fig)


def drawTF3(
  fcn:         ROOT.TF3,                # function to plot
  binnings:    Tuple[HistAxisBinning, HistAxisBinning, HistAxisBinning],  # binnings of the 3 histogram axes: (x, y, z)
  pdfFileName: str,                     # name of PDF file to write
  histTitle:   str = "",                # histogram title
  nmbPoints:   Optional[int]   = None,  # number of function points; used in numeric integration performed by GetRandom()
  maxVal:      Optional[float] = None,  # maximum plot range
) -> None:
  """Draws given TF3 into histogram"""
  if nmbPoints:
    fcn.SetNpx(nmbPoints)  # used in numeric integration performed by GetRandom()
    fcn.SetNpy(nmbPoints)
    fcn.SetNpz(nmbPoints)
  canv = ROOT.TCanvas()
  # fcn.Draw("BOX2Z") does not work; sigh
  # draw function "by hand" instead
  histName = os.path.splitext(os.path.basename(pdfFileName))[0]
  fistFcn = ROOT.TH3F(histName, histTitle, *binnings[0].astuple, *binnings[1].astuple, *binnings[2].astuple)
  xAxis = fistFcn.GetXaxis()
  yAxis = fistFcn.GetYaxis()
  zAxis = fistFcn.GetZaxis()
  for xBin in range(1, binnings[0].nmbBins + 1):
    for yBin in range(1, binnings[1].nmbBins + 1):
      for zBin in range(1, binnings[2].nmbBins + 1):
        x = xAxis.GetBinCenter(xBin)
        y = yAxis.GetBinCenter(yBin)
        z = zAxis.GetBinCenter(zBin)
        fistFcn.SetBinContent(xBin, yBin, zBin, fcn.Eval(x, y, z))
  print(f"Drawing histogram '{histName}' for function '{fcn.GetName()}': minimum value = {fistFcn.GetMinimum()}, maximum value = {fistFcn.GetMaximum()}")
  fistFcn.SetMinimum(0)
  if maxVal:
    fistFcn.SetMaximum(maxVal)
  fistFcn.GetXaxis().SetTitleOffset(1.5)
  fistFcn.GetYaxis().SetTitleOffset(2)
  fistFcn.GetZaxis().SetTitleOffset(1.5)
  fistFcn.Draw("BOX2Z")
  canv.Print(pdfFileName, "pdf")


def plotMoments(
  HVals:             Sequence[MomentValueAndTruth],  # moment values extracted from data with (optional) true values
  binning:           Optional[HistAxisBinning] = None,  # if not None data are plotted as function of binning variable
  normalizedMoments: bool = True,  # indicates whether moment values were normalized to H_0(0, 0)
  momentLabel:       str = MomentCalculator.QnMomentIndex.momentSymbol,  # label used in output file name
  pdfFileNamePrefix: str = "h",  # name prefix for output files
  histTitle:         str = "",   # histogram title
) -> None:
  """Plots moments extracted from data along categorical axis and overlays the corresponding true values if given"""
  histBinning = HistAxisBinning(len(HVals), 0, len(HVals)) if binning is None else binning
  xAxisTitle = "" if binning is None else binning.axisTitle
  trueValues = any((H.truth is not None for H in HVals))

  # (i) plot moments from data and overlay with true values (if given)
  for momentPart, legendEntrySuffix in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
    histStack = ROOT.THStack(f"{pdfFileNamePrefix}Compare_{momentLabel}_{momentPart}",
                             f"{histTitle};{xAxisTitle};" + ("Normalized" if normalizedMoments else "Unnormalized") + " Moment Value")
    # create histogram with moments from data
    histData = ROOT.TH1D(f"Data {legendEntrySuffix}", "", *histBinning.astuple)
    for index, H in enumerate(HVals):
      if (binning is not None) and (binning._var not in H.binCenters.keys()):
        continue
      y, yErr = H.realOrImag(realPart = momentPart == "Re")
      binIndex = index + 1 if binning is None else histData.GetXaxis().FindBin(H.binCenters[binning.var])
      histData.SetBinContent(binIndex, y)
      histData.SetBinError  (binIndex, 1e-100 if yErr < 1e-100 else yErr)  # ROOT does not draw points if uncertainty is zero; sigh
      if binning is None:
        histData.GetXaxis().SetBinLabel(binIndex, H.qn.title)  # categorical x axis with moment labels
    histData.SetLineColor(ROOT.kRed)
    histData.SetMarkerColor(ROOT.kRed)
    histData.SetMarkerStyle(ROOT.kFullCircle)
    histData.SetMarkerSize(0.75)
    histStack.Add(histData, "PEX0")
    if trueValues:
      # create histogram with true values
      histTrue = ROOT.TH1D("True values", "", *histBinning.astuple)
      for index, H in enumerate(HVals):
        if H.truth is not None:
          y = H.truth.real if momentPart == "Re" else H.truth.imag
          binIndex = index + 1
          histTrue.SetBinContent(binIndex, y)
          histTrue.SetBinError  (binIndex, 1e-100)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
      histTrue.SetMarkerColor(ROOT.kBlue)
      histTrue.SetLineColor(ROOT.kBlue)
      histTrue.SetLineWidth(2)
      histStack.Add(histTrue, "PE")
    canv = ROOT.TCanvas()
    histStack.Draw("NOSTACK")
    # adjust y-range
    canv.Update()
    actualYRange = canv.GetUymax() - canv.GetUymin()
    yRangeFraction = 0.1 * actualYRange
    histStack.SetMaximum(canv.GetUymax() + yRangeFraction)
    histStack.SetMinimum(canv.GetUymin() - yRangeFraction)
    canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
    # adjust style of automatic zero line
    # does not work
    # histStack.GetHistogram().SetLineColor(ROOT.kBlack)
    # histStack.GetHistogram().SetLineStyle(ROOT.kDashed)
    # histStack.GetHistogram().SetLineWidth(1)  # add zero line; see https://root-forum.cern.ch/t/continuing-the-discussion-from-an-unwanted-horizontal-line-is-drawn-at-y-0/50877/1
    canv.Update()
    if (canv.GetUymin() < 0) and (canv.GetUymax() > 0):
      # print(f"???ZERO")
      zeroLine = ROOT.TLine()
      zeroLine.SetLineColor(ROOT.kBlack)
      zeroLine.SetLineStyle(ROOT.kDashed)
      xAxis = histStack.GetXaxis()
      zeroLine.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 0, xAxis.GetBinUpEdge(xAxis.GetLast()), 0)
    canv.SaveAs(f"{histStack.GetName()}.pdf")

    # (ii) plot residuals
    if trueValues:
      histResidualName = f"{pdfFileNamePrefix}Residuals_{momentLabel}_{momentPart}"
      histResidual = ROOT.TH1D(histResidualName,
        (f"{histTitle} " if histTitle else "") + f"Residuals {legendEntrySuffix};{xAxisTitle};(Data - Truth) / #it{{#sigma}}_{{Data}}",
        *histBinning.astuple)
      # calculate residuals; NaN flags histogram bins, for which truth info is missing
      residuals = np.full(len(HVals) if binning is None else len(binning), np.nan)
      indicesToMask: List[int] = []
      for index, H in enumerate(HVals):
        if (binning is not None) and (binning._var not in H.binCenters.keys()):
          continue
        if H.truth is not None:
          dataVal, dataValErr = H.realOrImag     (realPart = momentPart == "Re")
          trueVal             = H.truthRealOrImag(realPart = momentPart == "Re")
          binIndex = index if binning is None else histResidual.GetXaxis().FindBin(H.binCenters[binning.var]) - 1
          residuals[binIndex] = (dataVal - trueVal) / dataValErr if dataValErr > 0 else 0
          if normalizedMoments and (H.qn == MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M = 0)):
            indicesToMask.append(binIndex)  # exclude H_0(0, 0) from plotting and chi^2 calculation
      # calculate chi^2
      # if moments were normalized, exclude Re and Im of H_0(0, 0) because it is always 1 by definition
      # exclude values, for which truth value is missing (residual = NaN)
      residualsMasked = np.ma.fix_invalid(residuals)
      for i in indicesToMask:
        residualsMasked[i] = np.ma.masked
      if residualsMasked.count() == 0:
        print(f"All residuals masked; skipping '{histResidualName}.pdf'.")
      else:
        chi2     = np.sum(residualsMasked**2)
        ndf      = residualsMasked.count()
        chi2Prob = stats.distributions.chi2.sf(chi2, ndf)
        # fill histogram with residuals
        for (index,), residual in np.ma.ndenumerate(residualsMasked):  # set bin content only for unmasked residuals
          binIndex = index + 1
          histResidual.SetBinContent(binIndex, residual)
          histResidual.SetBinError  (binIndex, 1e-100)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
        if binning is None:
          for binIndex in range(1, histData.GetXaxis().GetNbins() + 1):  # copy all x-axis bin labels
            histResidual.GetXaxis().SetBinLabel(binIndex, histData.GetXaxis().GetBinLabel(binIndex))
        histResidual.SetMarkerColor(ROOT.kBlue)
        histResidual.SetLineColor(ROOT.kBlue)
        histResidual.SetLineWidth(2)
        histResidual.SetMinimum(-3)
        histResidual.SetMaximum(+3)
        canv = ROOT.TCanvas()
        histResidual.Draw("PE")
        # draw zero line
        xAxis = histResidual.GetXaxis()
        line = ROOT.TLine()
        line.SetLineStyle(ROOT.kDashed)
        line.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 0, xAxis.GetBinUpEdge(xAxis.GetLast()), 0)
        # shade 1 sigma region
        box = ROOT.TBox()
        box.SetFillColorAlpha(ROOT.kBlack, 0.15)
        box.DrawBox(xAxis.GetBinLowEdge(xAxis.GetFirst()), -1, xAxis.GetBinUpEdge(xAxis.GetLast()), +1)
        # draw chi^2 info
        label = ROOT.TLatex()
        label.SetNDC()
        label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignBottom)
        label.DrawLatex(0.12, 0.9075, f"#it{{#chi}}^{{2}}/n.d.f. = {chi2:.2f}/{ndf}, prob = {chi2Prob * 100:.0f}%")
        canv.SaveAs(f"{histResidualName}.pdf")


def plotMomentsInBin(
  HData:             MomentCalculator.MomentResult,  # moment values extracted from data
  normalizedMoments: bool = True,  # indicates whether moment values were normalized to H_0(0, 0)
  HTrue:             Optional[MomentCalculator.MomentResult] = None,  # true moment values
  pdfFileNamePrefix: str = "h",  # name prefix for output files
) -> None:
  """Plots H_0, H_1, and H_2 extracted from data along categorical axis and overlays the corresponding true values if given"""
  assert not HTrue or HData.indices == HTrue.indices, f"Moment sets don't match. Data moments: {HData.indices} vs. true moments: {HTrue.indices}."
  # generate separate plots for each moment index
  for momentIndex in range(3):
    HVals = tuple(MomentValueAndTruth(*HData[qnIndex], HTrue[qnIndex].val if HTrue else None) for qnIndex in HData.indices.QnIndices() if qnIndex.momentIndex == momentIndex)  # type: ignore
    plotMoments(HVals, normalizedMoments = normalizedMoments, momentLabel = f"{MomentCalculator.QnMomentIndex.momentSymbol}{momentIndex}", pdfFileNamePrefix = pdfFileNamePrefix)


def plotMoments1D(
  moments:           MomentCalculator.MomentCalculatorsKinematicBinning,  # moment values extracted from data
  qnIndex:           MomentCalculator.QnMomentIndex,  # defines specific moment
  binning:           HistAxisBinning,                 # binning to use for plot
  normalizedMoments: bool = True,  # indicates whether moment values were normalized to H_0(0, 0)
  momentsTruth:      Optional[MomentCalculator.MomentCalculatorsKinematicBinning] = None,  # true moment values
  pdfFileNamePrefix: str = "h",   # name prefix for output files
  histTitle:         str = "",    # histogram title
) -> None:
  """Plots moment H_i(L, M) extracted from data as function of kinematical variable and overlays the corresponding true values if given"""
  # filter out specific moment
  HVals = tuple(MomentValueAndTruth(
      *HData.HPhys[qnIndex],
      truth = None if momentsTruth is None else momentsTruth[binIndex].HPhys[qnIndex].val,
      _binCenters = HData.binCenters,
    ) for binIndex, HData in enumerate(moments))
  plotMoments(HVals, binning, normalizedMoments, momentLabel = qnIndex.label, pdfFileNamePrefix = f"{pdfFileNamePrefix}{binning.var.name}_", histTitle = histTitle)


def plotMomentsBootStrap(
  HData:             MomentCalculator.MomentResult,  # moment values extracted from data
  HTrue:             Optional[MomentCalculator.MomentResult] = None,  # true moment values
  pdfFileNamePrefix: str = "h",  # name prefix for output files
) -> None:
  """Plots bootstrap distributions for H_0, H_1, and H_2 and overlays the true value and the estimate from uncertainty propagation"""
  assert not HTrue or HData.indices == HTrue.indices, f"Moment sets don't match. Data moments: {HData.indices} vs. true moments: {HTrue.indices}."
  # generate separate plots for each moment index
  for qnIndex in HData.indices.QnIndices():
    HVal = MomentValueAndTruth(*HData[qnIndex], HTrue[qnIndex].val if HTrue else None)  # type: ignore
    assert HData._bsSamplesFlatIndex is not None, "Bootstrap samples must be present"
    HValsBs = HData._bsSamplesFlatIndex[HData.indices.indexMap.flatIndex_for[qnIndex]]
    print(f"!!! {HVal.qn.title}: {HVal=} vs. {HValsBs=}")
    for momentPart, legendEntrySuffix in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
      # create histogram with bootstrap samples
      vals = HValsBs.real if momentPart == "Re" else HValsBs.imag
      min = np.min(vals)
      max = np.max(vals)
      nmbBins = 100
      halfRange = (max - min) * 1.1 / 2.0
      center = (min + max) / 2.0
      histBs = ROOT.TH1D(f"{pdfFileNamePrefix}bootstrap_{HVal.qn.label}_{momentPart}", f";{HVal.qn.title} {legendEntrySuffix};Count",
                         nmbBins, center - halfRange, center + halfRange)
      np.vectorize(histBs.Fill)(vals)
      canv = ROOT.TCanvas()
      histBs.Draw("E")
      canv.SaveAs(f"{histBs.GetName()}.pdf")
