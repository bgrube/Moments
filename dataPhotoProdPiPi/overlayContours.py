#!/usr/bin/env python3


from __future__ import annotations

import array
from dataclasses import dataclass
import os
from typing import Sequence

import ROOT


@dataclass
class ContourPlotDef:
  histName:    str  # name of histogram to get contours from
  levels:      int | Sequence[float]  # number of contours or explicit list of contour levels
  title:       str | None = None  # title of overlay plot
  labelFormat: str = ".0f"  # format string to use for contour level labels


def makeContourOverlayPlot(
  histBase:             ROOT.TH2,  # 2D histogram to draw as base
  histContours:         ROOT.TH2,  # 2D histogram to generate contours from
  contourLevels:        int | Sequence[float],  # number of contours or explicit list of contour levels
  outputDirName:        str,  # directory to save output plot in
  # contourTextColor:     int | None = ROOT.kRed - 10,  # color to use for contour level labels
  contourTextColor:     int | None = ROOT.kPink + 10,  # color to use for contour level labels
  contourLabelFormat:   str        = ".0f",  # format string to use for contour level labels
  minNmbPointsForLabel: int        = 10,     # minimum number of points on contour required to draw label
  title:                str | None = None,   # title of overlay plot
) -> None:
  """Overlay contour lines derived from `histContours` over histogram given by `histBase`."""
  # see https://root.cern/doc/v636/hist102__TH2__contour__list_8C.html
  # create contours
  print(f"Creating contours for histogram '{histContours.GetName()}'")
  if isinstance(contourLevels, int):
    histContours.SetContour(contourLevels)
  else:
    histContours.SetContour(len(contourLevels), array.array('d', contourLevels))
  canv = ROOT.TCanvas()
  histContours.Draw("CONT0 Z LIST")
  canv.Update()
  # loop over contours and copy graphs
  contours = ROOT.gROOT.GetListOfSpecials().FindObject("contours")
  # ROOT.gROOT.GetListOfSpecials().ls()
  graphs: dict[float, list[ROOT.TGraph]] = {}
  for contourLevelIndex in range(contours.GetSize()):
    contoursAtLevel = contours.At(contourLevelIndex)
    contourLevel = histContours.GetContourLevel(contourLevelIndex)
    print(f"Contour with index {contourLevelIndex} and level {contourLevel} has {contoursAtLevel.GetSize()} graphs")
    if contoursAtLevel.GetSize() > 0:
      graphs[contourLevel] = [graph.Clone() for graph in contoursAtLevel]
  # draw base histogram and overlay contours
  print(f"Overlaying histogram '{histBase.GetName()}' with contours")
  if title is not None:
    histBase.SetTitle(title)
  histBase.SetContour(100)
  histBase.SetStats(False)
  # using TExec trick to switch palettes on the fly
  # see https://root-forum.cern.ch/t/multiple-color-palettes-with-python-freezes-program/39124
  # and https://root.cern.ch/doc/master/multipalette_8C.html
  exPalletteHist = ROOT.TExec("exPalletteHist", "gStyle->SetPalette(kBird);")
  exPalletteHist.Draw()
  histBase.Draw("COLZ")
  # draw all contour graphs
  exPalletteContours = ROOT.TExec("exPalletteContours", "gStyle->SetPalette(kSolar);")
  # exPalletteContours = ROOT.TExec("exPalletteContours", "gStyle->SetPalette(kFuchsia);")
  # exPalletteContours = ROOT.TExec("exPalletteContours", "gStyle->SetPalette(kCherry);")
  # exPalletteContours = ROOT.TExec("exPalletteContours", "gStyle->SetPalette(kRust);")
  exPalletteContours.Draw()
  for graphsAtLevel in graphs.values():
    for graph in graphsAtLevel:
      # print(f"    Drawing contour graph with {graph.GetN()} points")
      graph.SetLineWidth(2)
      graph.Draw("C PLC")
  # add contour level labels
  latex = ROOT.TLatex()
  latex.SetTextSize(0.03)
  if contourTextColor is not None:
    latex.SetTextColor(contourTextColor)
  for contourLevel, graphsAtLevel in graphs.items():
    # latex.DrawLatex(graphsAtLevel[0].GetPointX(0), graphsAtLevel[0].GetPointY(0), f"{contourLevel:{contourLabelFormat}}")
    for graph in graphsAtLevel:
      nmbPoints = graph.GetN()
      if nmbPoints < minNmbPointsForLabel:  # ignore small contours
        continue
      pointIndex = nmbPoints // 2  # place label approximately in middle of contour
      latex.DrawLatex(graph.GetPointX(pointIndex), graph.GetPointY(pointIndex), f"{contourLevel:{contourLabelFormat}}" if contourLevel >= 0 else f"#minus {abs(contourLevel):{contourLabelFormat}}")
  canv.SaveAs(f"{outputDirName}/{histContours.GetName()}_contourOverlay.pdf")
  ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  # plotsBaseFile = ROOT.TFile.Open("./polarized/2018_08/tbin_0.1_0.2/PiPi/plots_REAL_DATA/PARA_0/plots.root", "READ")
  plotsBaseFile = ROOT.TFile.Open("./polarized/2018_08/tbin_0.1_0.2/PiPi/plots_ACCEPTED_PHASE_SPACE/PARA_0/plots.root", "READ")
  histBase = plotsBaseFile.Get("anglesHFPiPi_0.72_0.76")
  plotsContoursDir = "./polarized/2018_08/tbin_0.1_0.2/PiPi/plots_REAL_DATA/PARA_0/anglesHFcorrelations"
  contourPlotDefs = [
    ContourPlotDef(
      histName = "anglesHFPiPiCorrEbeam",
      levels = [8.30, 8.35, 8.40, 8.45, 8.50, 8.55],
      title = "Correlation with E_{beam}",
      labelFormat = ".2f",
    ),
    ContourPlotDef(
      histName = "anglesHFPiPiCorrmomLabP",
      levels = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44],
      title = "Correlation with p_{p}",
      labelFormat = ".2f",
    ),
    ContourPlotDef(
      histName = "anglesHFPiPiCorrmomLabPip",
      levels = [1, 2, 3, 4, 5, 6, 7],
      title = "Correlation with p_{#pi^{+}}",
    ),
  ]

  for contourPlotDef in contourPlotDefs:
    plotsContoursFile = ROOT.TFile.Open(f"{plotsContoursDir}/{contourPlotDef.histName}.root", "READ")
    makeContourOverlayPlot(
      histBase           = histBase,
      histContours       = plotsContoursFile.Get(contourPlotDef.histName),
      contourLevels      = contourPlotDef.levels,
      outputDirName      = plotsContoursDir,
      contourLabelFormat = contourPlotDef.labelFormat,
      title              = contourPlotDef.title,
    )
