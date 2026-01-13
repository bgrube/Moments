#!/usr/bin/env python3


from __future__ import annotations

import array
import os
from turtle import title
from typing import Sequence

import ROOT


def makeContourOverlayPlot(
  histBase:           ROOT.TH2,  # 2D histogram to draw as base
  histContours:       ROOT.TH2,  # 2D histogram to generate contours from
  contourLevels:      int | Sequence[float],  # give number of contours or explicit list of contour levels
  contourLineColor:   int | None = ROOT.kWhite,  # color to use for contour lines
  contourTextColor:   int | None = ROOT.kWhite,  # color to use for contour level labels
  contourLabelFormat: str = ".0f",  # format string to use for contour level labels
  title:              str | None = None,  # title of overlay plot
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
  histBase.Draw("COLZ")
  # draw all contour graphs
  for graphsAtLevel in graphs.values():
    for graph in graphsAtLevel:
      if contourLineColor is not None:
        graph.SetLineColor(contourLineColor)
      graph.Draw("C")
  # add contour level labels
  latex = ROOT.TLatex()
  latex.SetTextSize(0.03)
  if contourTextColor is not None:
    latex.SetTextColor(contourTextColor)
  for contourLevel, graphsAtLevel in graphs.items():
    latex.DrawLatex(graphsAtLevel[0].GetPointX(0), graphsAtLevel[0].GetPointY(0), f"{contourLevel:{contourLabelFormat}}")
  canv.SaveAs(f"./{histContours.GetName()}_contourOverlay.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  # func = ROOT.TF2("Func", "x", -1, 1, -1, 1)
  # histContours = func.GetHistogram()
  # histBase = histContours
  # contourLevels = 5
  # plotsBaseFile = ROOT.TFile.Open("./polarized/2018_08/tbin_0.1_0.2/PiPi/plots_REAL_DATA/PARA_0/plots.root", "READ")
  plotsBaseFile = ROOT.TFile.Open("./polarized/2018_08/tbin_0.1_0.2/PiPi/plots_ACCEPTED_PHASE_SPACE/PARA_0/plots.root", "READ")
  histBase = plotsBaseFile.Get("anglesHFPiPi_0.72_0.76")
  # plotsContoursFile = ROOT.TFile.Open("./anglesHFPiPiCorrEbeam.root", "READ")
  # histContours = plotsContoursFile.Get("anglesHFPiPiCorrEbeam")
  # contourLevels = [8.30, 8.35, 8.40, 8.45, 8.50, 8.55]
  # title = "Correlation with E_{beam}"
  plotsContoursFile = ROOT.TFile.Open("./anglesHFPiPiCorrmomLabP.root", "READ")
  histContours = plotsContoursFile.Get("anglesHFPiPiCorrmomLabP")
  contourLevels = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44]
  title = "Correlation with p_{p}"
  makeContourOverlayPlot(
    histBase           = histBase,
    histContours       = histContours,
    contourLevels      = contourLevels,
    contourLineColor   = ROOT.kWhite,
    contourTextColor   = ROOT.kWhite,
    contourLabelFormat = ".2f",
    title              = title,
  )
