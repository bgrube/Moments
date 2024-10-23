"""Module that provides a collection of functions for plotting"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import functools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import nptyping as npt
import os
from scipy import stats
from typing import (
  Any,
  Iterator,
  Sequence,
)

import ROOT

from  MomentCalculator import (
  KinematicBinningVariable,
  MomentResult,
  MomentResultsKinematicBinning,
  MomentValue,
  QnMomentIndex,
)


# set default matplotlib font to STIX
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["mathtext.fontset"] = "stix"
# set default matplotlib font to Computer Modern Roman
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'cmr10'
# mpl.rcParams['mathtext.fontset'] = 'cm'


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# black + 7 colorblind-friendly colors rom M. Okabe and K. Ito, "How to make figures and presentations that are friendly to color blind people," University of Tokyo, 2002.
# see also Bang Wong, https://www.nature.com/articles/nmeth.1618.pdf
#     https://davidmathlogic.com/colorblind
#     https://yoshke.org/blog/colorblind-friendly-diagrams
COLORS_CB_FRIENDLY: tuple[str, ...] = (
  "#000000",  # black
  "#0072B2",  # blue
  "#D55E00",  # vermillion
  "#009E73",  # bluish green
  "#CC79A7",  # reddish purple
  "#56B4E9",  # sky blue
  "#E69F00",  # orange
  "#F0E442",  # yellow
)


#TODO move into separate module
def getRootColor(hexColor: str) -> int:
  """Returns ROOT color index for given hex string in form #RRGGBB; if color does not exist yet in ROOT it is created"""
  ROOT.TColor.SetColorThreshold(0)  # ensure GetColor() returns exact color
  return ROOT.TColor.GetColor(hexColor)

def getCbFriendlyRootColor(
  index:     int,
  skipBlack: bool = False,  # if set black color is not used
) -> int:
  """Returns ROOT color index for given index in colorblind-friendly palette"""
  return getRootColor(COLORS_CB_FRIENDLY[index + (1 if skipBlack else 0)])

# 11 filled marker styles
# the float is a relative scaling factor to obtain approximately equal apparent sizes
MARKERS_FILLED: tuple[tuple[int, float], ...] = (
  (ROOT.kFullCircle,            0.75),
  (ROOT.kFullSquare,            0.70),
  (ROOT.kFullDiamond,           1.00),
  (ROOT.kFullCross,             0.85),
  (ROOT.kFullCrossX,            0.85),
  (ROOT.kFullStar,              1.00),
  (ROOT.kFullFourTrianglesX,    0.90),
  (ROOT.kFullFourTrianglesPlus, 0.85),
  (ROOT.kFullTriangleUp,        0.85),
  (ROOT.kFullTriangleDown,      0.85),
  (ROOT.kFullDoubleDiamond,     1.10),
)
# 11 open marker styles
MARKERS_OPEN: tuple[tuple[int, float], ...] = (
  (ROOT.kOpenCircle,            0.75),
  (ROOT.kOpenSquare,            0.70),
  (ROOT.kOpenDiamond,           1.00),
  (ROOT.kOpenCross,             0.85),
  (ROOT.kOpenCrossX,            0.85),
  (ROOT.kOpenStar,              1.00),
  (ROOT.kOpenFourTrianglesX,    0.90),
  (ROOT.kOpenFourTrianglesPlus, 0.85),
  (ROOT.kOpenTriangleUp,        0.85),
  (ROOT.kOpenTriangleDown,      0.85),
  (ROOT.kOpenDoubleDiamond,     1.10),
)

#TODO take TObject and check wether it is TAttLine etc.
#TODO add cycle option; provide more styles by combining colors and markers
def setCbFriendlyStyle(
  graphOrHist:   ROOT.TGraph | ROOT.TH1,
  styleIndex:    int,  # index that switches between styles
  skipBlack:     bool  = True,  # if set black color is not used
  setMarker:     bool  = True,
  markerSize:    float = 1.5,
  filledMarkers: bool  = True,
) -> None:
  """Sets line color and marker style, color, and size of a TGraph or TH1 according to a style index"""
  nmbStyles = min(len(COLORS_CB_FRIENDLY) - (1 if skipBlack else 0), len(MARKERS_FILLED), len(MARKERS_OPEN))
  assert styleIndex < nmbStyles, f"The style index {styleIndex} goes beyond the maximum of {nmbStyles} styles that are implemented"
  color = getCbFriendlyRootColor(styleIndex, skipBlack)
  graphOrHist.SetLineColor(color)
  if setMarker:
    graphOrHist.SetMarkerColor(color)
    graphOrHist.SetMarkerStyle(MARKERS_FILLED[styleIndex][0] if filledMarkers else MARKERS_OPEN[styleIndex][0])
    graphOrHist.SetMarkerSize(markerSize * MARKERS_FILLED[styleIndex][1] if filledMarkers else MARKERS_OPEN[styleIndex][1])

@dataclass
class MomentValueAndTruth(MomentValue):
  """Stores and provides access to single moment value and provides truth value"""
  truth:         complex | None = None  # true moment value
  truthUncertRe: float | None   = None  # uncertainty of real part of true moment value
  truthUncertIm: float | None   = None  # uncertainty of imaginary part of true moment value

  @property
  def truthReal(self) -> tuple[float, float | None]:
    """Returns real part of true moment value with uncertainty"""
    assert self.truth is not None, "self.truth must not be None"
    return (self.truth.real, self.truthUncertRe)

  @property
  def truthImag(self) -> tuple[float, float | None]:
    """Returns imaginary part of true moment value with uncertainty"""
    assert self.truth is not None, "self.truth must not be None"
    return (self.truth.imag, self.truthUncertIm)

  def truthPart(
    self,
    real: bool,  # switches between real part (True) and imaginary part (False)
  ) -> tuple[float, float | None]:
    """Returns real or imaginary part of true moment value with corresponding uncertainty according to given flag"""
    if real:
      return self.truthReal
    else:
      return self.truthImag


@dataclass
class HistAxisBinning:
  """Stores info that defines equidistant binning of an axis"""
  nmbBins: int    # number of bins
  minVal:  float  # lower limit
  maxVal:  float  # upper limit
  _var:    KinematicBinningVariable | None = None  # optional info about bin variable

  # make HistAxisBinning behave like a list of bin centers
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
  def var(self) -> KinematicBinningVariable:
    """Returns info about binning variable"""
    assert self._var is not None, "self._var must not be None"
    return self._var

  @property
  def astuple(self) -> tuple[int, float, float]:
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
  ) -> tuple[float, float]:
    """Returns value range for bin with given index"""
    binCenter = self[binIndex]
    return (binCenter - 0.5 * self.binWidth, binCenter + 0.5 * self.binWidth)

  @property
  def binValueRanges(self) -> tuple[tuple[float, float], ...]:
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


def plotRealMatrix(
  matrix:      npt.NDArray[npt.Shape["*, *"], npt.Float64],  # matrix to plot
  pdfFileName: str,  # name of output file
  axisTitles:  tuple[str, str]                   = ("", ""),      # titles for x and y axes
  plotTitle:   str                               = "",            # title for plot
  zRange:      tuple[float | None, float | None] = (None, None),  # range for z-axis
  **kwargs:    Any,  # additional keyword arguments for plt.matshow()
) -> None:
  """Plots given matrix into PDF file with given name"""
  print(f"Plotting matrix '{plotTitle}' and writing plot to '{pdfFileName}'")
  fig, ax = plt.subplots()
  cax = ax.matshow(matrix, vmin = zRange[0], vmax = zRange[1], **kwargs)
  ax.xaxis.tick_bottom()
  fig.colorbar(cax)
  plt.title(plotTitle)
  plt.xlabel(axisTitles[0])
  plt.ylabel(axisTitles[1])
  plt.savefig(pdfFileName, transparent = True)
  plt.close(fig)


def plotComplexMatrix(
  complexMatrix:     npt.NDArray[npt.Shape["*, *"], npt.Complex128],  # matrix to plot
  pdfFileNamePrefix: str,  # name prefix for output files
  axisTitles:        tuple[str, str] = ("", ""),  # titles for x and y axes
  plotTitle:         str             = "",        # title for plot
  zRangeAbs:         float           = 1.1,       # range for absolute value and for abs(real part)
  zRangeImag:        float           = 0.075,     # range for abs(imag part)
  **kwargs:          Any,  # additional keyword arguments for plt.matshow()
) -> None:
  """Plots real and imaginary parts, absolute value and phase of given complex-valued matrix"""
  matricesToPlot = {
    ("real", "Real Part",      ( -zRangeAbs,  +zRangeAbs)) : np.real    (complexMatrix),
    ("imag", "Imag Part",      (-zRangeImag, +zRangeImag)) : np.imag    (complexMatrix),
    ("abs",  "Absolute Value", (          0,   zRangeAbs)) : np.absolute(complexMatrix),
    ("arg",  "Phase",          (       -180,        +180)) : np.angle   (complexMatrix, deg = True),  # [degree]
  }
  for (label, title, zRange), matrix in matricesToPlot.items():
    plotRealMatrix(matrix, f"{pdfFileNamePrefix}{label}.pdf",       axisTitles, plotTitle = plotTitle + title, zRange = zRange, **kwargs)
    plotRealMatrix(matrix, f"{pdfFileNamePrefix}{label}_color.pdf", axisTitles, plotTitle = plotTitle + title, zRange = zRange, cmap = "RdBu", **kwargs)


def drawTF3(
  fcn:         ROOT.TF3,  # function to plot
  binnings:    tuple[HistAxisBinning, HistAxisBinning, HistAxisBinning],  # binnings of the 3 histogram axes: (x, y, z)
  pdfFileName: str,                  # name of PDF file to write
  histTitle:   str          = "",    # histogram title
  nmbPoints:   int | None   = None,  # number of function points; used in numeric integration performed by GetRandom()
  maxVal:      float | None = None,  # maximum plot range
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


def makeMomentHistogram(
  HVals:      Sequence[MomentValueAndTruth],  # moment values with (optional) true values
  momentPart: str,  # "Re" or "Im"
  histName:   str,  # histogram name
  histTitle:  str                    = "",     # histogram title
  binning:    HistAxisBinning | None = None,   # if not None data are plotted as function of binning variable
  plotTruth:  bool                   = False,  # plot true moments
  plotUncert: bool                   = True,   # plot uncertainty of true moments
) -> ROOT.TH1D:
  """Returns histogram with moment values along categorical axis or along given binning"""
  assert momentPart in ("Re", "Im"), f"Invalid moment part '{momentPart}'"
  histBinning = HistAxisBinning(len(HVals), 0, len(HVals)) if binning is None else binning
  hist = ROOT.TH1D(histName, histTitle, *histBinning.astuple)
  for index, HVal in enumerate(HVals):
    if (binning is not None) and (binning.var not in HVal.binCenters.keys()):
      continue
    y, yErr = HVal.truthPart(real = (momentPart == "Re")) if plotTruth and HVal.truth is not None else HVal.part(real = (momentPart == "Re"))
    binIndex = index + 1 if binning is None else hist.GetXaxis().FindBin(HVal.binCenters[binning.var])
    hist.SetBinContent(binIndex, y)
    if plotUncert and (yErr is not None) and (yErr > 1e-100):  # yErr must not be zero, otherwise ROOT does not draw x error bars; sigh
      hist.SetBinError(binIndex, yErr)
    else:
      hist.SetBinError(binIndex, 1e-100)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
    if binning is None:
      hist.GetXaxis().SetBinLabel(binIndex, HVal.qn.title)  # categorical x axis with moment labels
  return hist


def plotMoments(
  HVals:             Sequence[MomentValueAndTruth],  # moment values extracted from data with (optional) true values
  binning:           HistAxisBinning | None                      = None,  # if not None data are plotted as function of binning variable
  normalizedMoments: bool                                        = True,  # indicates whether moment values were normalized to H_0(0, 0)
  momentLabel:       str                                         = QnMomentIndex.momentSymbol,  # label used in output file name #TODO does this default value actually work?
  pdfFileNamePrefix: str                                         = "",  # name prefix for output files
  histTitle:         str                                         = "",  # histogram title
  plotLegend:        bool                                        = True,
  legendLabels:      tuple[str | None, str | None]               = (None, None),  # labels for legend entries; None = use defaults
  plotTruthUncert:   bool                                        = False,  # plot uncertainty of true moments
  truthColor:        int                                         = ROOT.kBlue + 1,  # color used for true values
  histsToOverlay:    Sequence[tuple[ROOT.TH1D, str, str]] | None = None,  # histograms to overlay on top of data and (optional) true values; tuple: (histogram, draw option, legend entry label)
) -> None:
  """Plots moments extracted from data along categorical axis or along given binning and overlays the corresponding true values if given"""
  histBinning = HistAxisBinning(len(HVals), 0, len(HVals)) if binning is None else binning
  xAxisTitle = "" if binning is None else binning.axisTitle
  trueValues = any((HVal.truth is not None for HVal in HVals))

  # (i) plot moments from data and overlay with true values (if given)
  for momentPart, legendEntrySuffix in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
    histStack = ROOT.THStack(f"{pdfFileNamePrefix}compare_{momentLabel}_{momentPart}",
                             f"{histTitle};{xAxisTitle};" + ("Normalized" if normalizedMoments else "Unnormalized") + " Moment Value")
    # create histogram with moments from data
    histData = makeMomentHistogram(
      HVals      = HVals,
      momentPart = momentPart,
      histName   = f"{legendLabels[0] or 'Data'} {legendEntrySuffix}",
      histTitle  = "",
      binning    = binning,
      plotTruth  = False,
      plotUncert = True,
    )
    histData.SetLineColor(ROOT.kRed + 1)
    histData.SetMarkerColor(ROOT.kRed + 1)
    histData.SetMarkerStyle(ROOT.kFullCircle)
    histData.SetMarkerSize(0.75)
    histStack.Add(histData, "PE1X0")
    histTruth = None
    if trueValues:
      # create histogram with true values
      histTruth = makeMomentHistogram(
        HVals      = HVals,
        momentPart = momentPart,
        histName   = legendLabels[1] or "True Values",
        histTitle  = "",
        binning    = binning,
        plotTruth  = True,
        plotUncert = plotTruthUncert,
      )
      histTruth.SetMarkerColor(truthColor)
      histTruth.SetLineColor(truthColor)
      histTruth.SetLineWidth(2)
      histStack.Add(histTruth, "PE")
    if histsToOverlay is not None:
      for histOverlay, drawOption, _ in histsToOverlay:
        histStack.Add(histOverlay, drawOption)
    canv = ROOT.TCanvas()
    histStack.Draw("NOSTACK")
    histStack.GetXaxis().LabelsOption("V")
    # adjust y-range
    canv.Update()
    actualYRange = canv.GetUymax() - canv.GetUymin()
    yRangeFraction = 0.1 * actualYRange
    histStack.SetMinimum(canv.GetUymin() - yRangeFraction)
    histStack.SetMaximum(canv.GetUymax() + yRangeFraction)
    if plotLegend:
      legend = ROOT.TLegend(0.7, 0.85, 0.99, 0.99)
      legend.AddEntry(histData, histData.GetName(), "LP")
      if histTruth is not None:
        legend.AddEntry(histTruth, histTruth.GetName(), "LP")
      if histsToOverlay is not None:
        for histOverlay, _, legendEntryLabel in histsToOverlay:
          if legendEntryLabel:
            legend.AddEntry(histOverlay, legendEntryLabel, "LP")
      legend.Draw()
    canv.Update()
    if plotTruthUncert:
      # redraw data on top of "truth"
      histData.Draw("PE1X0 SAME")
    # adjust style of automatic zero line
    histStack.GetHistogram().SetLineColor(ROOT.kBlack)
    histStack.GetHistogram().SetLineStyle(ROOT.kDashed)
    # histStack.GetHistogram().SetLineWidth(1)  # add zero line; see https://root-forum.cern.ch/t/continuing-the-discussion-from-an-unwanted-horizontal-line-is-drawn-at-y-0/50877/1
    # if (canv.GetUymin() < 0) and (canv.GetUymax() > 0):
    #   zeroLine = ROOT.TLine()
    #   zeroLine.SetLineColor(ROOT.kBlack)
    #   zeroLine.SetLineStyle(ROOT.kDashed)
    #   xAxis = histStack.GetXaxis()
    #   zeroLine.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 0, xAxis.GetBinUpEdge(xAxis.GetLast()), 0)
    canv.SaveAs(f"{histStack.GetName()}.pdf")

    # (ii) plot residuals
    if trueValues:
      histResidualName = f"{pdfFileNamePrefix}residuals_{momentLabel}_{momentPart}"
      histResidual = ROOT.TH1D(histResidualName,
        (f"{histTitle} " if histTitle else "") + f"Residuals {legendEntrySuffix};{xAxisTitle};(Data - Truth) / #it{{#sigma}}_{{Data}}",
        *histBinning.astuple)
      # calculate residuals; NaN flags histogram bins, for which truth info is missing
      residuals = np.full(len(HVals) if binning is None else len(binning), np.nan)
      indicesToMask: list[int] = []
      for index, HVal in enumerate(HVals):
        if (binning is not None) and (binning.var not in HVal.binCenters.keys()):
          continue
        if HVal.truth is not None:
          dataVal, dataValErr = HVal.part     (real = (momentPart == "Re"))
          trueVal, _          = HVal.truthPart(real = (momentPart == "Re"))
          binIndex = index if binning is None else histResidual.GetXaxis().FindBin(HVal.binCenters[binning.var]) - 1
          residuals[binIndex] = (dataVal - trueVal) / dataValErr if dataValErr > 0 else 0
          if normalizedMoments and (HVal.qn == QnMomentIndex(momentIndex = 0, L = 0, M = 0)):
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
        histResidual.LabelsOption("V", "X")
        histResidual.SetMarkerColor(ROOT.kBlue + 1)
        histResidual.SetLineColor(ROOT.kBlue + 1)
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
  HData:             MomentResult,  # moments extracted from data
  normalizedMoments: bool                          = True,  # indicates whether moment values were normalized to H_0(0, 0)
  HTruth:            MomentResult | None           = None,  # true moments
  pdfFileNamePrefix: str                           = "",    # name prefix for output files
  plotLegend:        bool                          = True,
  legendLabels:      tuple[str | None, str | None] = (None, None),  # labels for legend entries; None = use defaults
  plotTruthUncert:   bool                          = False,  # plot uncertainty of true moments
  truthColor:        int                           = ROOT.kBlue + 1,  # color used for true values
) -> None:
  """Plots H_i extracted from data for each i separately; the H_i with the same i are plotted as a categorical axis and overlaid with the corresponding true values if given"""
  # ensure that indices of HData and HTruth are compatible
  # allow case where HTruth contains unpolarized as well as polarized moments but HData only unpolarized moments
  indicesTrueMoments = None
  if HTruth:
    if (HTruth.indices.polarized and not HData.indices.polarized):
      indicesTrueMoments = copy.deepcopy(HTruth.indices)
      indicesTrueMoments.polarized = False
      indicesTrueMoments.regenerateIndexMaps()
    else:
      indicesTrueMoments = HTruth.indices
  assert not HTruth or HData.indices == indicesTrueMoments, f"Moment sets don't match. Data moments: {HData.indices} vs. true moments: {indicesTrueMoments}."
  # generate separate plots for each moment index
  for momentIndex in range(HData.indices.momentIndexRange):
    HVals = tuple(
      MomentValueAndTruth(
        *HData[qnIndex],
        truth         = HTruth[qnIndex].val      if HTruth else None,
        truthUncertRe = HTruth[qnIndex].uncertRe if HTruth else None,
        truthUncertIm = HTruth[qnIndex].uncertIm if HTruth else None,
      ) for qnIndex in HData.indices.qnIndices if qnIndex.momentIndex == momentIndex
    )
    plotMoments(
      HVals             = HVals,
      binning           = None,
      normalizedMoments = normalizedMoments,
      momentLabel       = f"{QnMomentIndex.momentSymbol}{momentIndex}",
      pdfFileNamePrefix = pdfFileNamePrefix,
      plotLegend        = plotLegend,
      legendLabels      = legendLabels,
      plotTruthUncert   = plotTruthUncert,
      truthColor        = truthColor,
    )


def plotMoments1D(
  momentResults:     MomentResultsKinematicBinning,  # moments extracted from data
  qnIndex:           QnMomentIndex,    # defines specific moment
  binning:           HistAxisBinning,  # binning to use for plot
  normalizedMoments: bool                                        = True,  # indicates whether moment values were normalized to H_0(0, 0)
  momentResultsTrue: MomentResultsKinematicBinning | None        = None,  # true moments
  pdfFileNamePrefix: str                                         = "",    # name prefix for output files
  histTitle:         str                                         = "",    # histogram title
  plotLegend:        bool                                        = True,
  legendLabels:      tuple[str | None, str | None]               = (None, None),  # labels for legend entries; None = use defaults
  plotTruthUncert:   bool                                        = False,  # plot uncertainty of true moments
  truthColor:        int                                         = ROOT.kBlue + 1,  # color used for true values
  histsToOverlay:    Sequence[tuple[ROOT.TH1D, str, str]] | None = None,  # histograms to overlay on top of data and (optional) true values; tuple: (histogram, draw option, legend entry)
) -> None:
  """Plots moment H_i(L, M) extracted from data as function of kinematical variable and overlays the corresponding true values if given"""
  # filter out specific moment given by qnIndex
  HVals = tuple(
    MomentValueAndTruth(
      *HPhys[qnIndex],
      truth         = None if momentResultsTrue is None else momentResultsTrue[binIndex][qnIndex].val,
      truthUncertRe = None if momentResultsTrue is None else momentResultsTrue[binIndex][qnIndex].uncertRe,
      truthUncertIm = None if momentResultsTrue is None else momentResultsTrue[binIndex][qnIndex].uncertIm,
    ) for binIndex, HPhys in enumerate(momentResults)
  )
  plotMoments(
    HVals             = HVals,
    binning           = binning,
    normalizedMoments = normalizedMoments,
    momentLabel       = qnIndex.label,
    pdfFileNamePrefix = f"{pdfFileNamePrefix}{binning.var.name}_",
    histTitle         = histTitle,
    plotLegend        = plotLegend,
    legendLabels      = legendLabels,
    plotTruthUncert   = plotTruthUncert,
    truthColor        = truthColor,
    histsToOverlay    = histsToOverlay,
  )


def plotMomentsBootstrapDistributions1D(
  HData:             MomentResult,  # moments extracted from data
  HTruth:            MomentResult | None = None,  # true moments
  pdfFileNamePrefix: str                 = "",    # name prefix for output files
  histTitle:         str                 = "",    # histogram title
  nmbBins:           int                 = 100,   # number of bins for bootstrap histograms
  HTruthLabel:       str                 = "True value",  # label for true value in legend
) -> None:
  """Plots 1D bootstrap distributions for H_0, H_1, and H_2 and overlays the true value and the estimate from uncertainty propagation"""
  assert not HTruth or HData.indices == HTruth.indices, f"Moment sets don't match. Data moments: {HData.indices} vs. true moments: {HTruth.indices}."
  # generate separate plots for each moment index
  for qnIndex in HData.indices.qnIndices:
    HVal = MomentValueAndTruth(
      *HData[qnIndex],
      truth = HTruth[qnIndex].val if HTruth else None
    )
    assert HVal.hasBootstrapSamples, "Bootstrap samples must be present"
    for momentPart, legendEntrySuffix in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
      # create histogram with bootstrap samples
      momentSamplesBs = HVal.bootstrapSamplesPart(real = (momentPart == "Re"))
      min = np.min(momentSamplesBs)
      max = np.max(momentSamplesBs)
      halfRange = (max - min) * 1.1 / 2.0
      center = (min + max) / 2.0
      histBs = ROOT.TH1D(f"{pdfFileNamePrefix}bootstrap_{HVal.qn.label}_{momentPart}", f"{histTitle};{HVal.qn.title} {legendEntrySuffix};Count",
                         nmbBins, center - halfRange, center + halfRange)
      # fill histogram
      np.vectorize(histBs.Fill, otypes = [int])(momentSamplesBs)
      # draw histogram
      canv = ROOT.TCanvas()
      histBs.SetMinimum(0)
      histBs.SetLineColor(ROOT.kBlue + 1)
      histBs.Draw("E")
      # indicate bootstrap estimate
      meanBs   = np.mean(momentSamplesBs)
      stdDevBs = np.std(momentSamplesBs, ddof = 1)
      yCoord = histBs.GetMaximum() / 4
      markerBs = ROOT.TMarker(meanBs, yCoord, ROOT.kFullCircle)
      markerBs.SetMarkerColor(ROOT.kBlue + 1)
      markerBs.SetMarkerSize(0.75)
      markerBs.Draw()
      lineBs = ROOT.TLine(meanBs - stdDevBs, yCoord, meanBs + stdDevBs, yCoord)
      lineBs.SetLineColor(ROOT.kBlue + 1)
      lineBs.Draw()
      # indicate estimate from uncertainty propagation
      meanEst, stdDevEst = HVal.part(real = (momentPart == "Re"))
      markerEst = ROOT.TMarker(meanEst,  yCoord / 2, ROOT.kFullCircle)
      markerEst.SetMarkerColor(ROOT.kGreen + 2)
      markerEst.SetMarkerSize(0.75)
      markerEst.Draw()
      lineEst = ROOT.TLine(meanEst - stdDevEst, yCoord / 2, meanEst + stdDevEst, yCoord / 2)
      lineEst.SetLineColor(ROOT.kGreen + 2)
      lineEst.Draw()
      # plot Gaussian that corresponds to estimate from uncertainty propagation
      gaussian = ROOT.TF1("gaussian", "gausn(0)", center - halfRange, center + halfRange)
      gaussian.SetParameters(len(momentSamplesBs) * histBs.GetBinWidth(1), meanEst, stdDevEst)
      gaussian.SetLineColor(ROOT.kGreen + 2)
      gaussian.Draw("SAME")
      # print chi^2 of Gaussian and histogram
      chi2 = histBs.Chisquare(gaussian, "L")
      chi2Prob = stats.distributions.chi2.sf(chi2, nmbBins)
      label = ROOT.TLatex()
      label.SetNDC()
      label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignBottom)
      label.SetTextSize(0.04)
      label.SetTextColor(ROOT.kGreen + 2)
      label.DrawLatex(0.13, 0.85, f"#it{{#chi}}^{{2}}/n.d.f. = {chi2:.2f}/{nmbBins}, prob = {chi2Prob * 100:.0f}%")
      # indicate true value
      if HVal.truth is not None:
        truth = HVal.truthPart(real = (momentPart == "Re"))
        lineTruth = ROOT.TLine(truth, 0, truth, histBs.GetMaximum())
        lineTruth.SetLineColor(ROOT.kRed + 1)
        lineTruth.SetLineStyle(ROOT.kDashed)
        lineTruth.Draw()
      # add legend
      legend = ROOT.TLegend(0.7, 0.85, 0.99, 0.99)
      legend.AddEntry(histBs, "Bootstrap samples", "LE")
      entry = legend.AddEntry(markerBs, "Bootstrap estimate", "LP")
      entry.SetLineColor(ROOT.kBlue + 1)
      entry = legend.AddEntry(markerEst, "Nominal estimate", "LP")
      entry.SetLineColor(ROOT.kGreen + 2)
      legend.AddEntry(gaussian, "Nominal estimate Gaussian", "LP")
      if HVal.truth is not None:
        legend.AddEntry(lineTruth, HTruthLabel, "L")
      legend.Draw()
      canv.SaveAs(f"{histBs.GetName()}.pdf")


def plotMomentPairBootstrapDistributions2D(
  momentIndexPair:   tuple[QnMomentIndex, QnMomentIndex],  # indices of moments to plot
  HData:             MomentResult,  # moments extracted from data
  HTruth:            MomentResult | None = None,  # true moments
  pdfFileNamePrefix: str                 = "",    # name prefix for output files
  histTitle:         str                 = "",    # histogram title
  nmbBins:           int                 = 20,    # number of bins for bootstrap histograms
  HTruthLabel:       str                 = "True value",  # label for true value in legend
) -> None:
  """Plots 2D bootstrap distributions of two moment values and overlays the true values and the estimates from uncertainty propagation"""
  HVals = (MomentValueAndTruth(*HData[momentIndexPair[0]], truth = HTruth[momentIndexPair[0]].val if HTruth else None),
           MomentValueAndTruth(*HData[momentIndexPair[1]], truth = HTruth[momentIndexPair[1]].val if HTruth else None))
  assert all(HVal.hasBootstrapSamples for HVal in HVals), "Bootstrap samples must be present for both moments"
  assert len(HVals[0].bsSamples) == len(HVals[1].bsSamples), "Number of bootstrap samples must be the same for both moments"
  # loop over combinations of real and imag parts
  for realParts in ((True, True), (False, False), (True, False), (False, True)):  # True = real part, False = imaginary part
    momentParts         = tuple("Re"        if realPart else "Im"        for realPart in realParts)
    legendEntrySuffixes = tuple("Real Part" if realPart else "Imag Part" for realPart in realParts)
    # create histogram with bootstrap samples
    momentSamplesBs = np.vstack((HVals[0].bsSamples.real if realParts[0] else HVals[0].bsSamples.imag,
                                 HVals[1].bsSamples.real if realParts[1] else HVals[1].bsSamples.imag))
    mins = np.min(momentSamplesBs, axis = 1)
    maxs = np.max(momentSamplesBs, axis = 1)
    halfRanges = (maxs - mins) * 1.1 / 2.0
    centers    = (mins + maxs) / 2.0
    histBs = ROOT.TH2D(
      f"{pdfFileNamePrefix}bootstrap_{HVals[1].qn.label}_{momentParts[1]}_vs_{HVals[0].qn.label}_{momentParts[0]}",
      f"{histTitle};{HVals[0].qn.title} {legendEntrySuffixes[0]};{HVals[1].qn.title} {legendEntrySuffixes[1]}",
      nmbBins, centers[0] - halfRanges[0], centers[0] + halfRanges[0],
      nmbBins, centers[1] - halfRanges[1], centers[1] + halfRanges[1]
    )
    # fill histogram
    np.vectorize(histBs.Fill, otypes = [int])(momentSamplesBs[0, :], momentSamplesBs[1, :])
    # draw histogram
    canv = ROOT.TCanvas()
    histBs.Draw("COLZ")

    # indicate bootstrap estimate
    meansBs  = np.mean(momentSamplesBs, axis = 1)
    markerBs = ROOT.TMarker(meansBs[0], meansBs[1], ROOT.kFullCircle)
    markerBs.SetMarkerColor(ROOT.kBlue + 1)
    markerBs.SetMarkerSize(0.75)
    markerBs.Draw()
    # indicate bootstrap covariance matrix
    covBs = np.cov(momentSamplesBs, ddof = 1)
    covEigenValsBs, covEigenVectsBs = np.linalg.eig(covBs)
    # calculate lengths of major axes and rotation angle of ellipse
    r1    = np.sqrt(covEigenValsBs[0])
    r2    = np.sqrt(covEigenValsBs[1])
    theta = np.arctan2(covEigenVectsBs[1, 0], covEigenVectsBs[0, 0]) * 180.0 / np.pi
    # draw the ellipse
    #TODO improve by drawing major axes in kDotted
    ellipseBs = ROOT.TEllipse(*meansBs, r1, r2, 0, 360, theta)
    ellipseBs.SetLineColor(ROOT.kBlue + 1)
    ellipseBs.SetLineWidth(3)
    ellipseBs.SetFillStyle(0)
    ellipseBs.Draw()

    # indicate estimate from uncertainty propagation
    meansEst  = (HVals[0].part(realParts[0])[0], HVals[1].part(realParts[1])[0])
    markerEst = ROOT.TMarker(meansEst[0], meansEst[1], ROOT.kFullCircle)
    markerEst.SetMarkerColor(ROOT.kGreen + 2)
    markerEst.SetMarkerSize(0.75)
    markerEst.Draw()
    # indicate nominal covariance matrix
    covEst = HData.covariance(momentIndexPair, realParts)
    covEigenValsEst, covEigenVectsEst = np.linalg.eig(covEst)
    # calculate lengths of major axes and rotation angle of ellipse
    r1    = np.sqrt(covEigenValsEst[0])
    r2    = np.sqrt(covEigenValsEst[1])
    theta = np.arctan2(covEigenVectsEst[1, 0], covEigenVectsEst[0, 0]) * 180.0 / np.pi
    # draw the ellipse
    #TODO improve by drawing major axes in kDotted
    ellipseEst = ROOT.TEllipse(*meansEst, r1, r2, 0, 360, theta)
    ellipseEst.SetLineColor(ROOT.kGreen + 2)
    ellipseEst.SetLineWidth(2)
    ellipseEst.SetFillStyle(0)
    ellipseEst.Draw()

    # indicate true value
    plotTruth = all(HVal.truth is not None for HVal in HVals)
    if plotTruth:
      markerTruth = ROOT.TMarker(HVals[0].truthPart(realParts[0])[0], HVals[1].truthPart(realParts[1])[0], 31)
      markerTruth.SetMarkerColor(ROOT.kRed + 1)
      markerTruth.SetMarkerSize(1.25)
      markerTruth.Draw()

    # add legend
    legend = ROOT.TLegend(0.7, 0.75, 0.99, 0.99)
    entry = legend.AddEntry(markerBs, "Bootstrap estimate", "LP")
    entry.SetLineColor(ROOT.kBlue + 1)
    entry = legend.AddEntry(markerEst, "Nominal estimate", "LP")
    entry.SetLineColor(ROOT.kGreen + 2)
    if plotTruth:
      entry = legend.AddEntry(markerTruth, HTruthLabel, "P")
    legend.Draw()
    canv.SaveAs(f"{histBs.GetName()}.pdf")


def plotMomentsBootstrapDistributions2D(
  HData:             MomentResult,  # moments extracted from data
  HTruth:            MomentResult | None = None,  # true moments
  pdfFileNamePrefix: str                 = "",    # name prefix for output files
  histTitle:         str                 = "",    # histogram title
  nmbBins:           int                 = 20,    # number of bins for bootstrap histograms
  HTruthLabel:       str                 = "True value",  # label for true value in legend
) -> None:
  """Plots 2D bootstrap distributions of pairs of moment values that correspond to upper triangle of covariance matrix and overlays the true values and the estimates from uncertainty propagation"""
  momentIndexPairs = ((HData.indices[flatIndex0], HData.indices[flatIndex1])
                      for flatIndex0 in HData.indices.flatIndices
                      for flatIndex1 in HData.indices.flatIndices
                      if flatIndex0 < flatIndex1)
  for momentIndexPair in momentIndexPairs:
    plotMomentPairBootstrapDistributions2D(
      momentIndexPair   = momentIndexPair,
      HData             = HData,
      HTruth            = HTruth,
      pdfFileNamePrefix = pdfFileNamePrefix,
      histTitle         = histTitle,
      nmbBins           = nmbBins,
      HTruthLabel       = HTruthLabel,
    )


def plotMomentsCovMatrices(
  HData:             MomentResult,  # moments extracted from data
  pdfFileNamePrefix: str,  # name prefix for output files
  axisTitles:        tuple[str, str]                   = ("", ""),      # titles for x and y axes
  plotTitle:         str                               = "",            # title for plot
  zRange:            tuple[float | None, float | None] = (None, None),  # range for z-axis
):
  """Plots covariance matrices of moments and the difference w.r.t. the bootstrap estimates"""
  # get full composite covariance matrix from nominal estimate
  covMatrixCompEst  = HData.compositeCovarianceMatrix
  covMatrixCompDiff = None
  if HData.hasBootstrapSamples:
    # get full composite covariance matrix from bootstrap samples
    nmbMoments    = len(HData.indices)
    covMatricesBs = {
      (True,  True ) : np.zeros((nmbMoments, nmbMoments), dtype = npt.Float64),  # ReRe, symmetric
      (False, False) : np.zeros((nmbMoments, nmbMoments), dtype = npt.Float64),  # ImIm, symmetric
      (True,  False) : np.zeros((nmbMoments, nmbMoments), dtype = npt.Float64),  # ReIm, _not_ symmetric
    }
    for realParts, covMatrixBs in covMatricesBs.items():
      # !Note! the covariance matrices for ReRe and ImIm are
      # symmetric, so we need only the indices of the upper triangle
      # of the covariance matrix including the diagonal
      # !Note! the ReIm matrix is _not_ symmetric, so we need all indices
      momentIndexPairs = ((flatIndex0, flatIndex1)
                          for flatIndex0 in HData.indices.flatIndices
                          for flatIndex1 in HData.indices.flatIndices
                          if (realParts[0] != realParts[1]) or (flatIndex0 <= flatIndex1))
      for momentIndexPair in momentIndexPairs:
        # covariance = off-diagonal element of the 2 x 2 covariance matrix returned by np.cov():
        covMatrixBs[momentIndexPair[0], momentIndexPair[1]] = HData.covarianceBootstrap(momentIndexPair, realParts)[0, 1]
      if realParts[0] == realParts[1]:
        # symmetrize bootstrap covariance matrix for ReRe and ImIm
        covMatrixBs += covMatrixBs.T - np.diag(np.diag(covMatrixBs))
    covMatrixCompBs = np.block([
      [covMatricesBs[(True, True )],   covMatricesBs[(True,  False)]],
      [covMatricesBs[(True, False)].T, covMatricesBs[(False, False)]],
    ])
    # calculate relative difference of bootstrap and nominal covariance matrices
    # relative difference is defined as (covMatrixCompBs_ij - covMatrixCompEst_ij) / sqrt(covMatrixCompBs_ii * covMatrixCompBs_jj)
    norm = np.diag(np.reciprocal(np.sqrt(np.diag(covMatrixCompBs))))  # diagonal matrix with 1 / sqrt(covMatrixBs_ii)
    covMatrixCompDiff = norm @ (covMatrixCompBs - covMatrixCompEst) @ norm
  # plot covariance matrices for nominal estimates
  covMatricesEst = {
    (True,  True ) : ("ReRe", "Auto Covariance Real Parts"          , HData._covReReFlatIndex),  # ReRe, symmetric
    (False, False) : ("ImIm", "Auto Covariance Imag Parts"          , HData._covImImFlatIndex),  # ImIm, symmetric
    (True,  False) : ("ReIm", "Cross Covariance Real and Imag Parts", HData._covReImFlatIndex),  # ReIm, _not_ symmetric
  }
  for realParts, (label, title, covMatrixEst) in  covMatricesEst.items():
    plotRealMatrix(covMatrixEst, f"{pdfFileNamePrefix}{label}.pdf", axisTitles, plotTitle = plotTitle + title, zRange = zRange)
  # plot differences of bootstrap and nominal covariance matrices
  if covMatrixCompDiff is not None:
    covMatricesDiff = {
      (True,  True ) : covMatrixCompDiff[:nmbMoments, :nmbMoments],  # ReRe, symmetric
      (False, False) : covMatrixCompDiff[nmbMoments:, nmbMoments:],  # ImIm, symmetric
      (True,  False) : covMatrixCompDiff[:nmbMoments, nmbMoments:],  # ReIm, _not_ symmetric
    }
    for realParts, (label, title, _) in  covMatricesEst.items():
      covMatrixDiff = covMatricesDiff[realParts]
      range = max(abs(np.min(covMatrixDiff)), abs(np.max(covMatrixDiff)))
      plotRealMatrix(covMatrixDiff, f"{pdfFileNamePrefix}{label}_BSdiff.pdf", axisTitles, plotTitle = plotTitle + f"{title} BS Diff",
                     zRange = (-range, +range), cmap = "RdBu")


def plotMomentsBootstrapDiff(
  HVals:             Sequence[MomentValueAndTruth],  # moment values extracted from data
  momentLabel:       str,  # label used in graph names and output file name
  binning:           HistAxisBinning | None = None,  # binning to use for plot; if None moment index is used as x-axis
  pdfFileNamePrefix: str                    = "",    # name prefix for output files
  graphTitle:        str                    = "",    # graph title
) -> None:
  """Plots relative differences of estimates and their uncertainties for all given moments as function of moment index or binning variable"""
  assert all(HVal.hasBootstrapSamples for HVal in HVals), "Bootstrap samples must be present for all moments"
  # create graphs with relative differences
  graphMomentValDiff    = ROOT.TMultiGraph(f"{pdfFileNamePrefix}bootstrap_{momentLabel}_valDiff",
                                           f"{graphTitle};{('' if binning is None else binning.axisTitle)}" + ";(#it{H}_{BS} #minus #it{H}_{Nom}) / #it{#sigma}_{BS}")
  graphMomentUncertDiff = ROOT.TMultiGraph(f"{pdfFileNamePrefix}bootstrap_{momentLabel}_uncertDiff",
                                           f"{graphTitle};{('' if binning is None else binning.axisTitle)}" + ";(#it{#sigma}_{BS} #minus #it{#sigma}_{Nom}) / #it{#sigma}_{BS}")
  colors = {"Re": ROOT.kRed + 1, "Im": ROOT.kBlue + 1}
  xVals  = np.arange(len(HVals), dtype = npt.Float64) if binning is None else np.array([HVal.binCenters[binning.var] for HVal in HVals], dtype = npt.Float64)
  for momentPart, legendEntrySuffix in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
    # get nominal estimates and uncertainties
    momentValsEst   = np.array([HVal.part(real = (momentPart == "Re"))[0] for HVal in HVals], dtype = npt.Float64)
    momentUncertEst = np.array([HVal.part(real = (momentPart == "Re"))[1] for HVal in HVals], dtype = npt.Float64)
    # get bootstrap estimates and uncertainties
    momentsBs       = tuple(HVal.bootstrapEstimatePart(real = (momentPart == "Re")) for HVal in HVals)
    momentValsBs    = np.array([momentBs[0] for momentBs in momentsBs], dtype = npt.Float64)
    momentUncertBs  = np.array([momentBs[1] for momentBs in momentsBs], dtype = npt.Float64)
    # create graphs with relative differences
    graphMomentValDiffPart    = ROOT.TGraph(len(xVals), xVals, (momentValsBs   - momentValsEst)   / momentUncertBs)
    graphMomentUncertDiffPart = ROOT.TGraph(len(xVals), xVals, (momentUncertBs - momentUncertEst) / momentUncertBs)
    # improve style of graphs
    for graph in (graphMomentValDiffPart, graphMomentUncertDiffPart):
      graph.SetTitle(legendEntrySuffix)
      graph.SetMarkerColor(colors[momentPart])
      graph.SetMarkerStyle(ROOT.kOpenCircle)
      graph.SetMarkerSize(0.75)
    graphMomentValDiff.Add(graphMomentValDiffPart)
    graphMomentUncertDiff.Add(graphMomentUncertDiffPart)
  # draw graphs
  for graph in (graphMomentValDiff, graphMomentUncertDiff):
    canv = ROOT.TCanvas()
    histBinning = (len(HVals), -0.5, len(HVals) - 0.5) if binning is None else binning.astuple
    histDummy = ROOT.TH1D("dummy", "", *histBinning)  # dummy histogram needed for x-axis bin labels
    xAxis = histDummy.GetXaxis()
    if binning is None:
      for binIndex, HVal in enumerate(HVals):
        xAxis.SetBinLabel(binIndex + 1, HVal.qn.title)
    histDummy.LabelsOption("V", "X")
    histDummy.SetTitle(graph.GetTitle())
    histDummy.SetXTitle(graph.GetXaxis().GetTitle())
    histDummy.SetYTitle(graph.GetYaxis().GetTitle())
    histDummy.SetMinimum(-0.1)
    histDummy.SetMaximum(+0.1)
    histDummy.SetLineColor(ROOT.kBlack)
    histDummy.SetLineStyle(ROOT.kDashed)
    histDummy.Draw()
    graph.Draw("P")  # !NOTE! graphs don't have a SAME option; "SAME" will be interpreted as "A"
    # draw average difference
    for g in graph.GetListOfGraphs():
      avg = g.GetMean(2)
      avgLine = ROOT.TLine()
      avgLine.SetLineColor(g.GetMarkerColor())
      avgLine.SetLineStyle(ROOT.kDotted)
      avgLine.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), avg, xAxis.GetBinUpEdge(xAxis.GetLast()), avg)
    # draw legend
    legend = ROOT.TLegend(0.7, 0.75, 0.99, 0.99)
    for g in graph.GetListOfGraphs():
      legend.AddEntry(g, g.GetTitle(), "P")
    legend.Draw()
    canv.SaveAs(f"{graph.GetName()}.pdf")


def plotMomentsBootstrapDiff1D(
  momentResults:     MomentResultsKinematicBinning,  # moment values extracted from data
  qnIndex:           QnMomentIndex,    # defines specific moment
  binning:           HistAxisBinning,  # binning to use for plot
  pdfFileNamePrefix: str = "",  # name prefix for output files
  graphTitle:        str = "",  # graph title
) -> None:
  """Plots relative differences of estimates for moments and their uncertainties as a function of binning variable"""
  # get values of moment that corresponds to the given qnIndex in all kinematic bins
  HVals = tuple(
    MomentValueAndTruth(
      *HPhys[qnIndex],
      truth = None,
    ) for HPhys in momentResults
  )
  plotMomentsBootstrapDiff(
    HVals             = HVals,
    momentLabel       = qnIndex.label,
    binning           = binning,
    pdfFileNamePrefix = f"{pdfFileNamePrefix}{binning.var.name}_",
    graphTitle        = graphTitle,
  )


def plotMomentsBootstrapDiffInBin(
  HData:             MomentResult,  # moment values extracted from data
  pdfFileNamePrefix: str = "",      # name prefix for output files
  graphTitle:        str = "",      # graph title
) -> None:
  """Plots relative differences of estimates and their uncertainties for all moments"""
  for momentIndex in range(HData.indices.momentIndexRange):
    # get moments with index momentIndex
    HVals = tuple(
      MomentValueAndTruth(
        *HData[qnIndex],
        truth = None,
      ) for qnIndex in HData.indices.qnIndices if qnIndex.momentIndex == momentIndex
    )
    plotMomentsBootstrapDiff(
      HVals             = HVals,
      momentLabel       = f"{QnMomentIndex.momentSymbol}{momentIndex}",
      binning           = None,
      pdfFileNamePrefix = pdfFileNamePrefix,
      graphTitle        = graphTitle,
    )


def plotPullsForMoment(
  momentResults:     MomentResultsKinematicBinning,  # moments extracted from data
  qnIndex:           QnMomentIndex,  # defines specific moment
  momentResultsTrue: MomentResultsKinematicBinning | None = None,  # true moments
  pdfFileNamePrefix: str                                  = "",    # name prefix for output files
  histTitle:         str                                  = "",    # histogram title
) -> dict[bool, tuple[tuple[float, float], tuple[float, float]]]:  # Gaussian mean and sigma with uncertainties, both for real and imaginary parts
  """Plots pulls of moment with given qnIndex estimated from moment values in kinematic bins"""
  # filter out specific moment given by qnIndex
  HVals = tuple(
    MomentValueAndTruth(
      *HPhys[qnIndex],
      truth = None if momentResultsTrue is None else momentResultsTrue[binIndex][qnIndex].val,
    ) for binIndex, HPhys in enumerate(momentResults)
  )
  histStack = ROOT.THStack(f"{pdfFileNamePrefix}pulls_{qnIndex.label}", f"{histTitle};""(#it{#hat{H}} - #it{H}_{true}) / #it{#hat{#sigma}};Count")
  gaussPars: dict[bool, tuple[tuple[float, float], tuple[float, float]]] = {}  # Gaussian mean and sigma with uncertainties, both for real and imaginary parts
  for realPart in (False, True):
    # loop over kinematic bins and fill histogram with pulls
    histPull = ROOT.TH1D(f"{histStack.GetName()}_{'Re' if realPart else 'Im'}", "Real Part" if realPart else "Imag Part", 25, -5, +5)
    for HVal in HVals:
      if HVal.truth is not None:
        moment, momentErr = HVal.part(realPart)
        pull = (moment - HVal.truthPart(realPart)[0]) / momentErr
        histPull.Fill(pull)
    color = ROOT.kRed + 1 if realPart else ROOT.kBlue + 1
    histPull.SetLineColor(color)
    histPull.SetFillColor(color)
    histPull.SetFillStyle(3003)
    # fit Gaussian to pull histogram
    gauss = ROOT.TF1("gauss", "gausn", -4, +4)
    gauss.SetLineColor(color)
    histPull.Fit(gauss, "LEIMR")
    gaussPars[realPart] = ((gauss.GetParameter(1), gauss.GetParError(1)),  # mean
                           (gauss.GetParameter(2), gauss.GetParError(2)))  # sigma
    histStack.Add(histPull)
  # draw pulls
  canv = ROOT.TCanvas()
  histStack.Draw("NOSTACK")
  # draw legend
  canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
  # print Gaussian parameters for real and imaginary parts
  label = ROOT.TLatex()
  label.SetNDC()
  label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignBottom)
  label.SetTextSize(0.035)
  label.SetTextColor(ROOT.kRed + 1)
  label.DrawLatex(0.13, 0.85, f"Re: #it{{#mu}} = {gaussPars[True][0][0]:.2f} #pm {gaussPars[True][0][1]:.2f}, "
                               f"#it{{#sigma}} = {gaussPars[True][1][0]:.2f} #pm {gaussPars[True][1][1]:.2f}")
  label.SetTextColor(ROOT.kBlue + 1)
  label.DrawLatex(0.13, 0.80, f"Im: #it{{#mu}} = {gaussPars[False][0][0]:.2f} #pm {gaussPars[False][0][1]:.2f}, "
                               f"#it{{#sigma}} = {gaussPars[False][1][0]:.2f} #pm {gaussPars[False][1][1]:.2f}")
  canv.SaveAs(f"{histStack.GetName()}.pdf")
  return gaussPars


def plotPullParameters(
  pullParameters:    dict[QnMomentIndex, dict[bool, tuple[tuple[float, float], tuple[float, float]]]],  # {index : {isReal : ((mean val, mean err), (sigma val, sigma err))}}
  pdfFileNamePrefix: str = "",  # name prefix for output files
  histTitle:         str = "",  # histogram title
) -> None:
  """Plots Gaussian means and sigmas of pull distributions for each moment"""
  # generate separate plots for each moment index
  for momentIndex in range(3):  #TODO pass number of moment indices to function
    # get pull parameters for moment with given momentIndex
    pullPars = {qnIndex : pars for qnIndex, pars in pullParameters.items() if qnIndex.momentIndex == momentIndex}
    # create a histograms with the moments as the categorical axis and the Gaussian parameters as the y-axis
    histStack = ROOT.THStack(f"{pdfFileNamePrefix}pullParameters_H{momentIndex}", f"{histTitle}#it{{H}}_{{{momentIndex}}};;Parameter Value")
    for meanOrSigma in (0, 1):
      for realPart in (False, True):
        histPar = ROOT.TH1D(f"{'#it{#mu}' if meanOrSigma == 0 else '#it{#sigma}'} {'Real Part' if realPart else 'Imag Part'}", "", len(pullPars), 0, len(pullPars))
        for binIndex, (qnIndex, gaussPars) in enumerate(pullPars.items()):
          val,  valErr  = gaussPars[realPart][meanOrSigma]
          histPar.SetBinContent(binIndex + 1, val)
          histPar.SetBinError  (binIndex + 1, valErr)
          histPar.GetXaxis().SetBinLabel(binIndex + 1, qnIndex.title)  # categorical x axis with moment labels
        color = ROOT.kRed + 1 if realPart else ROOT.kBlue + 1
        histPar.SetLineColor(color)
        histPar.SetMarkerColor(color)
        histPar.SetMarkerStyle(ROOT.kFullCircle if meanOrSigma == 0 else ROOT.kOpenCircle)
        histPar.SetMarkerSize(0.75)
        histStack.Add(histPar)
    # draw histograms
    canv = ROOT.TCanvas()
    histStack.SetMinimum(-1)
    histStack.SetMaximum(+2)
    histStack.Draw("NOSTACK,E1P")
    histStack.GetXaxis().LabelsOption("V")
    # draw legend
    canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
    canv.Update()
    # draw horizontal lines at y = 0 and 1
    line = ROOT.TLine()
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(ROOT.kDashed)
    xAxis = histStack.GetXaxis()
    line.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 0, xAxis.GetBinUpEdge(xAxis.GetLast()), 0)
    line.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 1, xAxis.GetBinUpEdge(xAxis.GetLast()), 1)
    canv.SaveAs(f"{histStack.GetName()}.pdf")


def plotAngularDistr(
  dataPsAcc:         ROOT.RDataFrame,  # accepted phase-space data
  dataPsGen:         ROOT.RDataFrame,  # generated phase-space data
  dataSignalAcc:     ROOT.RDataFrame,  # accepted signal data
  dataSignalGen:     ROOT.RDataFrame | None = None,  # generated signal data
  pdfFileNamePrefix: str                    = "",    # name prefix for output files
  nmbBins3D:         int                    = 15,    # number of bins for 3D histograms
  nmbBins2D:         int                    = 40,    # number of bins for 2D histograms
):
  """Plot 2D and 3D angular distributions of signal and phase-space data"""
  @dataclass
  class DataInfo:
    dataFrame: ROOT.RDataFrame
    label:     str
    def useColumns(
      self,
      columns: Sequence[str],
      weightColName: str = "eventWeight",
    ) -> list[str]:
      cols = list(columns)
      if weightColName in self.dataFrame.GetColumnNames():
        cols += [weightColName]
      return cols
    def hasPhiColumn(self) -> bool:
      return "PhiDeg" in self.dataFrame.GetColumnNames()
  dataInfos = [
    DataInfo(dataSignalAcc, "signalAcc"),
    DataInfo(dataPsAcc,     "psAcc"),
    DataInfo(dataPsGen,     "psGen"),
  ]
  if dataSignalGen is not None:
    dataInfos += [DataInfo(dataSignalGen, "signalGen")]
  title2D   = ";cos#it{#theta};#it{#phi} [deg]"
  title3D   = title2D + ";#it{#Phi} [deg]"
  columns2D = ["cosTheta", "phiDeg"]
  columns3D = columns2D + ["PhiDeg"]
  hists = []
  for dataInfo in dataInfos:
    hists += [
      dataInfo.dataFrame.Histo2D(
        ROOT.RDF.TH2DModel(f"{dataInfo.label}_2D", title2D, nmbBins2D, -1, +1, nmbBins2D, -180, +180),
        *dataInfo.useColumns(columns2D)
      ),
    ]
    if dataInfo.hasPhiColumn():
      hists += [
        dataInfo.dataFrame.Histo3D(
          ROOT.RDF.TH3DModel(f"{dataInfo.label}_3D", title3D, nmbBins3D, -1, +1, nmbBins3D, -180, +180, nmbBins3D, -180, +180),
          *dataInfo.useColumns(columns3D)
        ),
      ]
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    histType = hist.IsA().GetName()
    if histType.startswith("TH3"):
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
    elif histType.startswith("TH2"):
      hist.Draw("COLZ")
    else:
      raise TypeError(f"Unexpected histogram type '{histType}'")
    canv.SaveAs(f"{pdfFileNamePrefix}{hist.GetName()}.pdf")
