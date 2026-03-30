#!/usr/bin/env python3


from __future__ import annotations

import array
from dataclasses import dataclass
import os
from typing import Sequence

import ROOT


@dataclass
class ContourPlotDef:
  """Definition of a contour overlay plot"""
  histName:             str  # name of histogram to get contours from
  levels:               int | Sequence[float]  # number of contours or explicit list of contour levels
  title:                str | None = None  # title of overlay plot
  labelFormat:          str = ".0f"  # format string to use for contour level labels
  labelUnit:            str = ""  # TLaTex string appended to contour values
  labelTextColor:       int | None = ROOT.kPink + 10  # color to use for contour level labels
  minNmbPointsForLabel: int = 10  # minimum number of points on contour required to draw label


def makeContourOverlayPlot(
  histBase:       ROOT.TH2,  # 2D histogram to draw as base
  histContours:   ROOT.TH2,  # 2D histogram to generate contours from
  contourPlotDef: ContourPlotDef,  # definition of contour plot
  outputDirName:  str,  # directory to save output plot in
) -> None:
  """Overlay contour lines derived from `histContours` over histogram given by `histBase`"""
  if histContours == ROOT.nullptr:
    print(f"Error: contour histogram '{contourPlotDef.histName}' does not exist; skipping.")
    return
  # see https://root.cern/doc/v636/hist102__TH2__contour__list_8C.html
  # create contours
  print(f"Creating contours for histogram '{histContours.GetName()}'")
  if isinstance(contourPlotDef.levels, int):
    histContours.SetContour(contourPlotDef.levels)
  else:
    histContours.SetContour(len(contourPlotDef.levels), array.array('d', contourPlotDef.levels))
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
  if contourPlotDef.title is not None:
    histBase.SetTitle(contourPlotDef.title)
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
  if contourPlotDef.labelTextColor is not None:
    latex.SetTextColor(contourPlotDef.labelTextColor)
  for contourLevel, graphsAtLevel in graphs.items():
    # latex.DrawLatex(graphsAtLevel[0].GetPointX(0), graphsAtLevel[0].GetPointY(0), f"{contourLevel:{contourPlotDef.labelFormat}}")
    for graph in graphsAtLevel:
      nmbPoints = graph.GetN()
      if nmbPoints < contourPlotDef.minNmbPointsForLabel:  # ignore small contours
        continue
      pointIndex = nmbPoints // 2  # place label approximately in middle of contour
      levelLabel = f"{contourLevel:{contourPlotDef.labelFormat}}" if contourLevel >= 0 else f"#minus {abs(contourLevel):{contourPlotDef.labelFormat}}"
      levelLabel += contourPlotDef.labelUnit
      latex.DrawLatex(graph.GetPointX(pointIndex), graph.GetPointY(pointIndex), levelLabel)
  canv.SaveAs(f"{outputDirName}/{histContours.GetName()}_contourOverlay.pdf")
  ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  # dataPeriod = "2017_01"
  dataPeriod = "2017_01_ver05"
  # dataPeriod = "2018_08"

  tBinLabel = "tbin_0.100_0.114"  # lowest |t| bin of SDME analysis
  # tBinLabel = "tbin_0.1_0.2"
  # tBinLabel = "tbin_0.2_0.3"
  # tBinLabel = "tbin_0.3_0.4"
  # tBinLabel = "tbin_0.4_0.5"

  beamPolLabel = "PARA_0"
  # beamPolLabel = "PARA_135"
  # beamPolLabel = "PERP_45"
  # beamPolLabel = "PERP_90"
  # beamPolLabel = "AMO"
  # beamPolLabel = "Unpol"

  # dataType = "REAL_DATA"
  dataType = "ACCEPTED_PHASE_SPACE"

  plotsDirName = f"./polarized/{dataPeriod}/{tBinLabel}/PiPi/plots_{dataType}/{beamPolLabel}"
  with ROOT.TFile.Open(f"{plotsDirName}/plots.root", "READ") as plotsBaseFile:
    histBase = plotsBaseFile.Get("anglesHFPiPi_0.72_0.76")
    plotsContoursDir = f"{plotsDirName}/anglesHFCorrelations"
    contourPlotDefs = [
      ContourPlotDef(
        histName    = "anglesHFPiPiCorr_Ebeam",
        levels      = [8.30, 8.35, 8.40, 8.45, 8.50, 8.55],
        title       = "Correlation with E_{beam}",
        labelFormat = ".2f",
        labelUnit   = " GeV",
      ),
      ContourPlotDef(
        histName    = "anglesHFPiPiCorr_momLabP",
        levels      = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44],
        title       = "Correlation with p_{p}^{lab}",
        labelFormat = ".2f",
        labelUnit   = " GeV",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_momLabPip",
        levels    = [1, 2, 3, 4, 5, 6, 7],
        title     = "Correlation with p_{#pi^{#plus}}^{lab}",
        labelUnit = " GeV",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_momLabPim",
        levels    = [1, 2, 3, 4, 5, 6, 7],
        title     = "Correlation with p_{#pi^{#minus}}^{lab}",
        labelUnit = " GeV",
      ),
      ContourPlotDef(
        histName    = "anglesHFPiPiCorr_thetaLabPDeg",
        levels      = [71.0, 71.5, 72.0, 72.5, 73.0],
        title       = "Correlation with #theta_{p}^{lab}",
        labelFormat = ".1f",
        labelUnit   = "#circ",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_thetaLabPipDeg",
        levels    = [2, 4, 6, 8, 10, 12, 14],
        title     = "Correlation with #theta_{#pi^{#plus}}^{lab}",
        labelUnit = "#circ",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_thetaLabPimDeg",
        levels    = [2, 4, 6, 8, 10, 12, 14],
        title     = "Correlation with #theta_{#pi^{#minus}}^{lab}",
        labelUnit = "#circ",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_phiLabPDeg",
        levels    = [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        title     = "Correlation with #phi_{p}^{lab}",
        labelUnit = "#circ",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_phiLabPipDeg",
        levels    = [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        title     = "Correlation with #phi_{#pi^{#plus}}^{lab}",
        labelUnit = "#circ",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_phiLabPimDeg",
        levels    = [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        title     = "Correlation with #phi_{#pi^{#minus}}^{lab}",
        labelUnit = "#circ",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_PhiPiPiDeg",
        levels    = [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        title     = "Correlation with #Phi",
        labelUnit = "#circ",
      ),
      ContourPlotDef(
        histName  = "anglesHFPiPiCorr_PsiHFPiPiDeg",
        levels    = [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        title     = "Correlation with #Psi",
        labelUnit = "#circ",
      ),
      ContourPlotDef(
        histName    = "anglesHFPiPiCorr_massPipP",
        levels      = [1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75],
        title       = "Correlation with m_{#pi^{#plus} p}",
        labelFormat = ".2f",
        # labelUnit   = " GeV",
      ),
      ContourPlotDef(
        histName    = "anglesHFPiPiCorr_massPimP",
        levels      = [1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75],
        title       = "Correlation with m_{#pi^{#minus} p}",
        labelFormat = ".2f",
        # labelUnit   = " GeV",
      ),
    ]

    for contourPlotDef in contourPlotDefs:
      contourPlotFilePath = f"{plotsContoursDir}/{contourPlotDef.histName}.root"
      try:
        with ROOT.TFile.Open(contourPlotFilePath, "READ") as plotsContoursFile:
          makeContourOverlayPlot(
            histBase       = histBase,
            histContours   = plotsContoursFile.Get(contourPlotDef.histName),
            contourPlotDef = contourPlotDef,
            outputDirName  = plotsContoursDir,
          )
      except OSError as error:
        print(f"Error opening contour file for histogram '{contourPlotDef.histName}': {error}; skipping.")
        continue
