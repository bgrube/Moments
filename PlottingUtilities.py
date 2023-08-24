from dataclasses import dataclass, astuple
import matplotlib.pyplot as plt
import numpy as np
import nptyping as npt
import os
from typing import Optional, Tuple

import ROOT


binning1DType = Tuple[str, int, float, float]

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
  matrix:        npt.NDArray[npt.Shape["*, *"], npt.Complex128],  # matrix to plot
  pdfNamePrefix: str,  # name prefix for output files
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
    plt.savefig(f"{pdfNamePrefix}_{plotLabel}.pdf", transparent = True)
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
