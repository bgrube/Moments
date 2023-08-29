from dataclasses import dataclass, astuple
import matplotlib.pyplot as plt
import numpy as np
import nptyping as npt
import os
from scipy import stats
from typing import (
  Optional,
  Sequence,
  Tuple,
)

import ROOT

import MomentCalculator


@dataclass
class HistAxisBinning:
  '''Stores info that defines the binning of an axis'''
  nmbBins:   int
  minVal:    float
  maxVal:    float


def setupPlotStyle() -> None:
  '''Defines ROOT plotting style'''
  ROOT.gROOT.LoadMacro("./rootlogon.C")
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
  matrix:            npt.NDArray[npt.Shape["*, *"], npt.Complex128],  # matrix to plot
  pdfFileNamePrefix: str,  # name prefix for output files
) -> None:
  '''Draws real and imaginary parts of given 2D array'''
  dataToPlot = {
    "real" : np.real(matrix),      # real part
    "imag" : np.imag(matrix),      # imaginary part
    "abs"  : np.absolute(matrix),  # absolute value
    "arg"  : np.angle(matrix),     # phase
  }
  for plotLabel, data in dataToPlot.items():
    plt.figure().colorbar(plt.matshow(data))
    plt.savefig(f"{pdfFileNamePrefix}_{plotLabel}.pdf", transparent = True)
    plt.close()


def drawTF3(
  fcn:         ROOT.TF3,                # function to plot
  binnings:    Tuple[HistAxisBinning, HistAxisBinning, HistAxisBinning],  # binnings of the 3 histogram axes: (x, y, z)
  pdfFileName: str,                     # name of PDF file to write
  histTitle:   str = "",                # histogram title
  nmbPoints:   Optional[int]   = None,  # number of function points; used in numeric integration performed by GetRandom()
  maxVal:      Optional[float] = None,  # maximum plot range
) -> None:
  '''Draws given TF3 into histogram'''
  if nmbPoints:
    fcn.SetNpx(nmbPoints)  # used in numeric integration performed by GetRandom()
    fcn.SetNpy(nmbPoints)
    fcn.SetNpz(nmbPoints)
  canv = ROOT.TCanvas()
  # fcn.Draw("BOX2") does not work; sigh
  # draw function "by hand" instead
  histName = os.path.splitext(os.path.basename(pdfFileName))[0]
  fistFcn = ROOT.TH3F(histName, histTitle, *astuple(binnings[0]), *astuple(binnings[1]), *astuple(binnings[2]))
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
  HVals:             Sequence[MomentCalculator.MomentValueAndTruth],  # moment values extracted from data with (optional) true values
  momentLabel:       str = "H",  # label used in output file name
  pdfFileNamePrefix: str = "h",  # name prefix for output files
) -> None:
  '''Plots given moments extracted from data and overlays the corresponding true values if given'''
  nmbMoments = len(HVals)
  trueValues = any((H.truth is not None for H in HVals))

  # (i) plot moments from data and overlay with true values (if given)
  for momentPart, legendEntrySuffix in (("Re", "Real Part"), ("Im", "Imag Part")):  # plot real and imaginary parts separately
    histStack = ROOT.THStack(f"{pdfFileNamePrefix}Compare_{momentLabel}_{momentPart}", ";;Normalized Moment Value")
    # create histogram with moments from data
    histData = ROOT.TH1D(f"Data {legendEntrySuffix}", "", nmbMoments, 0, nmbMoments)
    for index, H in enumerate(HVals):
      y, yErr = H.realOrImag(realPart = momentPart == "Re")
      binIndex = index + 1
      histData.SetBinContent(binIndex, y)
      histData.SetBinError  (binIndex, 1e-100 if yErr < 1e-100 else yErr)  # ROOT does not draw points if uncertainty is zero; sigh
      histData.GetXaxis().SetBinLabel(binIndex, f"#it{{H}}_{{{H.qn.momentIndex}}}({H.qn.L}, {H.qn.M})")  # categorical x axis with moment labels
    histData.SetLineColor(ROOT.kRed)
    histData.SetMarkerColor(ROOT.kRed)
    histData.SetMarkerStyle(ROOT.kFullCircle)
    histData.SetMarkerSize(0.75)
    histStack.Add(histData, "PEX0")
    if trueValues:
      # create histogram with true values
      histTrue = ROOT.TH1D("True values", "", nmbMoments, 0, nmbMoments)
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
    ROOT.gPad.Update()
    actualYRange = ROOT.gPad.GetUymax() - ROOT.gPad.GetUymin()
    yRangeFraction = 0.1 * actualYRange
    histStack.SetMaximum(ROOT.gPad.GetUymax() + yRangeFraction)
    histStack.SetMinimum(ROOT.gPad.GetUymin() - yRangeFraction)
    # adjust style of automatic zero line
    histStack.GetHistogram().SetLineColor(ROOT.kBlack)
    histStack.GetHistogram().SetLineStyle(ROOT.kDashed)
    # hStack.GetHistogram().SetLineWidth(0)  # remove zero line; see https://root-forum.cern.ch/t/continuing-the-discussion-from-an-unwanted-horizontal-line-is-drawn-at-y-0/50877/1
    canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
    canv.SaveAs(f"{histStack.GetName()}.pdf")

    # (ii) plot residuals
    if trueValues:
      residuals = np.empty(len(HVals))
      for index, H in enumerate(HVals):
        if H.truth is not None:
          dataVal, dataValErr = H.realOrImag(realPart = momentPart == "Re")
          trueVal             = H.truth.real if momentPart == "Re" else H.truth.imag
          residuals[index] = (dataVal - trueVal) / dataValErr if dataValErr > 0 else 0
        else:
          residuals[index] = np.nan
      # calculate chi^2 excluding Re and Im of H_0(0, 0) because it is always 1 by definition
      indicesToMask = tuple(index for index, H in enumerate(HVals)
        if (H.qn == MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M = 0)) or np.isnan(residuals[index]))  # exclude H_0(0, 0) and NaN
      residualsMasked = residuals.view(np.ma.MaskedArray)
      for i in indicesToMask:
        residualsMasked[i] = np.ma.masked
      histResidualName = f"{pdfFileNamePrefix}Residuals_{momentLabel}_{momentPart}"
      if residualsMasked.count() == 0:
        print(f"All residuals masked; skipping '{histResidualName}.pdf'.")
      else:
        chi2     = np.sum(residualsMasked**2)
        ndf      = residualsMasked.count()
        chi2Prob = stats.distributions.chi2.sf(chi2, ndf)
        # create histogram with residuals
        histResidual = ROOT.TH1D(histResidualName, f"Residuals {legendEntrySuffix};;(Data - True) / #it{{#sigma}}_{{Data}}", nmbMoments, 0, nmbMoments)
        for (index,), residual in np.ma.ndenumerate(residualsMasked):  # set bin content only for unmasked residuals
          binIndex = index + 1
          histResidual.SetBinContent(binIndex, residual)
          histResidual.SetBinError  (binIndex, 1e-100)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
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
  HTrue:             Optional[MomentCalculator.MomentResult] = None,  # true moment values
  momentLabel:       str = "H",  # label used in output file name
  pdfFileNamePrefix: str = "h",  # name prefix for output files
) -> None:
  assert not HTrue or HData.indices == HTrue.indices, f"Moment sets don't match. Data moments: {HData.indices} vs. true moments: {HTrue.indices}."
  # generate separate plots for each moment index
  for momentIndex in range(3):
    HVals = tuple(MomentCalculator.MomentValueAndTruth(*HData[qnIndex], HTrue[qnIndex].val if HTrue else None) for qnIndex in HData.indices.QnIndices() if qnIndex.momentIndex == momentIndex)  # type: ignore
    plotMoments(HVals, momentLabel = f"{momentLabel}{momentIndex}", pdfFileNamePrefix = pdfFileNamePrefix)
