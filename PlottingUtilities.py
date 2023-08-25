from dataclasses import dataclass, astuple
import matplotlib.pyplot as plt
import numpy as np
import nptyping as npt
import os
from typing import Optional, Sequence, Tuple

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
  histName = os.path.splitext(os.path.basename(pdfFileName))[0]
  fcnHist = ROOT.TH3F(histName, histTitle, *astuple(binnings[0]), *astuple(binnings[1]), *astuple(binnings[2]))
  xAxis = fcnHist.GetXaxis()
  yAxis = fcnHist.GetYaxis()
  zAxis = fcnHist.GetZaxis()
  for xBin in range(1, binnings[0].nmbBins + 1):
    for yBin in range(1, binnings[1].nmbBins + 1):
      for zBin in range(1, binnings[2].nmbBins + 1):
        x = xAxis.GetBinCenter(xBin)
        y = yAxis.GetBinCenter(yBin)
        z = zAxis.GetBinCenter(zBin)
        fcnHist.SetBinContent(xBin, yBin, zBin, fcn.Eval(x, y, z))
  print(f"Drawing histogram '{histName}' for function '{fcn.GetName()}': minimum value = {fcnHist.GetMinimum()}, maximum value = {fcnHist.GetMaximum()}")
  fcnHist.SetMinimum(0)
  if maxVal:
    fcnHist.SetMaximum(maxVal)
  fcnHist.GetXaxis().SetTitleOffset(1.5)
  fcnHist.GetYaxis().SetTitleOffset(2)
  fcnHist.GetZaxis().SetTitleOffset(1.5)
  fcnHist.Draw("BOX2Z")
  canv.Print(pdfFileName, "pdf")


def plotMoments(
  HData:             Sequence[MomentCalculator.MomentValue],  # moment values extracted from data
  HTrue:             Optional[Sequence[MomentCalculator.MomentValue]] = None,  # true moment values
  momentLabel:       str = "H",  # label used in output file name
  pdfFileNamePrefix: str = "h",  # name prefix for output files
) -> None:
  '''Plots given moments extracted from data and overlays the corresponding true values if given'''
  for momentPart, legendEntrySuffix in (("Re", "Real Part"), ("Im", "Imag Part")):
    # (i) overlay moments from data and true values (if given)
    hStack = ROOT.THStack(f"{pdfFileNamePrefix}Compare_{momentLabel}_{momentPart}", ";;Normalized Moment Value")
    nmbMoments = len(HData)
    if HTrue:
      assert len(HTrue) == nmbMoments, f"Number of true moments ({len(HTrue)}) does not match match number of moments from data ({nmbMoments})"
    #TODO check that HTrue has same QN indices; or use extended dataclass for plotting
    # create histogram with moments from data
    hData = ROOT.TH1D(f"Data {legendEntrySuffix}", "", nmbMoments, 0, nmbMoments)
    for index, HDataVal in enumerate(HData):
      #TODO add member fcn
      y    = HDataVal.val.real if momentPart == "Re" else HDataVal.val.imag
      yErr = HDataVal.uncertRe if momentPart == "Im" else HDataVal.uncertIm
      hData.SetBinContent(index + 1, y)
      hData.SetBinError  (index + 1, 1e-100 if yErr < 1e-100 else yErr)  # ROOT does not draw points if uncertainty is zero; sigh
      hData.GetXaxis().SetBinLabel(index + 1, f"#it{{H}}_{{{HDataVal.qn.momentIndex}}}({HDataVal.qn.L}, {HDataVal.qn.M})")  # categorical x axis with moment labels
    hData.SetLineColor(ROOT.kRed)
    hData.SetMarkerColor(ROOT.kRed)
    hData.SetMarkerStyle(ROOT.kFullCircle)
    hData.SetMarkerSize(0.75)
    hStack.Add(hData, "PEX0")
    if HTrue:
      # create histogram with true values
      hTrue = ROOT.TH1D("True values", "", nmbMoments, 0, nmbMoments)
      for index, HTrueVal in enumerate(HTrue):  #TODO assumes that order is the same as in HData; better use extended data class
        y = HTrueVal.val.real if momentPart == "Re" else HTrueVal.val.imag
        hTrue.SetBinContent(index + 1, y)
        hTrue.SetBinError  (index + 1, 1e-100)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
      hTrue.SetMarkerColor(ROOT.kBlue)
      hTrue.SetLineColor(ROOT.kBlue)
      hTrue.SetLineWidth(2)
      hStack.Add(hTrue, "PE")
    canv = ROOT.TCanvas()
    hStack.Draw("NOSTACK")
    # adjust y-range
    ROOT.gPad.Update()
    actualYRange = ROOT.gPad.GetUymax() - ROOT.gPad.GetUymin()
    yRangeFraction = 0.1 * actualYRange
    hStack.SetMaximum(ROOT.gPad.GetUymax() + yRangeFraction)
    hStack.SetMinimum(ROOT.gPad.GetUymin() - yRangeFraction)
    # adjust style of automatic zero line
    hStack.GetHistogram().SetLineColor(ROOT.kBlack)
    hStack.GetHistogram().SetLineStyle(ROOT.kDashed)
    # hStack.GetHistogram().SetLineWidth(0)  # remove zero line; see https://root-forum.cern.ch/t/continuing-the-discussion-from-an-unwanted-horizontal-line-is-drawn-at-y-0/50877/1
    canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
    canv.SaveAs(f"{hStack.GetName()}.pdf")

    # # (ii) plot residuals
    # if HTrue:
    #   residuals = tuple((measVal[0] - HTrue[index]) / measVal[1] if measVal[1] > 0 else 0 for index, measVal in enumerate(HData))
    #   hResidual = ROOT.TH1D(f"{pdfFileNamePrefix}Residuals_H{momentIndex if useMomentSubscript else ''}_{momentPart}",
    #     f"Residuals #it{{H}}{labelSubscript} {legendEntrySuffix};;(measured - input) / #sigma_{{measured}}", nmbMoments, 0, nmbMoments)
    #   chi2 = sum(tuple(residual**2 for residual in residuals[1 if momentIndex == 0 else 0:]))  # exclude Re and Im of H_0(0, 0) from chi^2
    #   ndf  = len(residuals[1 if momentIndex == 0 else 0:])
    #   for index, residual in enumerate(residuals):
    #     hResidual.SetBinContent(index + 1, residual)
    #     hResidual.SetBinError  (index + 1, 1e-100)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
    #     hResidual.GetXaxis().SetBinLabel(index + 1, hData.GetXaxis().GetBinLabel(index + 1))
    #   hResidual.SetMarkerColor(ROOT.kBlue)
    #   hResidual.SetLineColor(ROOT.kBlue)
    #   hResidual.SetLineWidth(2)
    #   hResidual.SetMinimum(-3)
    #   hResidual.SetMaximum(+3)
    #   canv = ROOT.TCanvas()
    #   hResidual.Draw("PE")
    #   # draw zero line
    #   xAxis = hResidual.GetXaxis()
    #   line = ROOT.TLine()
    #   line.SetLineStyle(ROOT.kDashed)
    #   line.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 0, xAxis.GetBinUpEdge(xAxis.GetLast()), 0)
    #   # shade 1 sigma region
    #   box = ROOT.TBox()
    #   box.SetFillColorAlpha(ROOT.kBlack, 0.15)
    #   box.DrawBox(xAxis.GetBinLowEdge(xAxis.GetFirst()), -1, xAxis.GetBinUpEdge(xAxis.GetLast()), +1)
    #   # draw chi^2 info
    #   label = ROOT.TLatex()
    #   label.SetNDC()
    #   label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignBottom)
    #   label.DrawLatex(0.12, 0.9075, f"#it{{#chi}}^{{2}}/n.d.f. = {chi2:.2f}/{ndf}, prob = {stats.distributions.chi2.sf(chi2, ndf) * 100:.0f}%")
    #   canv.SaveAs(f"{hResidual.GetName()}.pdf")


def plotMomentsInBin(
  HData:             MomentCalculator.MomentResult,  # moment values extracted from data
  HTrue:             Optional[MomentCalculator.MomentResult] = None,  # true moment values
  momentLabel:       str = "H",  # label used in output file name
  pdfFileNamePrefix: str = "h",  # name prefix for output files
) -> None:
  assert not HTrue or HData.index == HTrue.index, f"Moment sets don't match. Data moments: {HData.index} vs. true moments: {HTrue.index}."
  # generate separate plots for each moment index
  for momentIndex in range(3):
    HDataVals = tuple(HData[qnIndex] for qnIndex in HData.index.QnIndices() if qnIndex.momentIndex == momentIndex)
    if HTrue:
      HTrueVals = tuple(HTrue[qnIndex] for qnIndex in HData.index.QnIndices() if qnIndex.momentIndex == momentIndex)
    else:
      HTrueVals = None
    plotMoments(HDataVals, HTrueVals, momentLabel = f"{momentLabel}{momentIndex}", pdfFileNamePrefix = pdfFileNamePrefix)
