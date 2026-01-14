#!/usr/bin/env python3


from __future__ import annotations

import os

import ROOT
from matplotlib.pyplot import hist


def makeComparisonPlotPosNegPhiHF(
  plotsFile:     ROOT.TFile,  # file containing histograms to plot
  histBaseName:  str,  # name of histogram to plot
  outputDirName: str,  # directory to save output plot in
) -> None:
  """Generates comparison plot for positive and negative phi_HF"""
  print(f"Generating phi_HF < 0, phi_HF > 0 comparison plot for histogram '{histBaseName}'")
  histPos = plotsFile.Get(f"{histBaseName}_phiDegHFPiPiPos")
  histNeg = plotsFile.Get(f"{histBaseName}_phiDegHFPiPiNeg")
  histCompName = f"{histBaseName}_phiDegHFPiPiPosNeg"
  canv = ROOT.TCanvas()
  if histPos.GetDimension() == 1:
    # overlay 1D histograms
    histStack = ROOT.THStack(histCompName, f";{histPos.GetXaxis().GetTitle()};{histPos.GetYaxis().GetTitle()}")
    histStack.Add(histPos, "PE1X0")
    histStack.Add(histNeg, "PE1X0")
    histPos.SetLineColor  (ROOT.kRed + 1)
    histPos.SetMarkerColor(ROOT.kRed + 1)
    histPos.SetTitle("#phi_{HF} > 0")
    histPos.SetStats(False)
    histNeg.SetLineColor  (ROOT.kBlue + 1)
    histNeg.SetMarkerColor(ROOT.kBlue + 1)
    histNeg.SetTitle("#phi_{HF} < 0")
    histNeg.SetStats(False)
    histStack.Draw("NOSTACK")
    canv.BuildLegend(0.75, 0.91, 0.99, 0.99)
  elif histPos.GetDimension() == 2:
    # plot difference of 2D histograms for phi > 0 and phi < 0
    ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # draw 2D plot with pos/neg color palette and symmetric z axis
    histDiff = histPos.Clone(histCompName)
    histDiff.Add(histNeg, -1)
    histDiff.SetTitle("(#phi_{HF} > 0)#minus (#phi_{HF} < 0)")
    zRange = abs(histDiff.GetMinimum()) if histDiff.GetMinimum() < 0 else 10.0  # choose z range to see negative values; but avoid zero range in case function positive
    histDiff.SetMinimum(-zRange)
    histDiff.SetMaximum(+zRange)
    histDiff.Draw("COLZ")
    # for some reason, the background color of the stats box is overwritten; reset it to white
    canv.Update()
    stats = canv.GetPrimitive("stats")
    if stats is not ROOT.nullptr:
      stats.SetFillColor(ROOT.kWhite)
      stats.SetX1NDC(0.75)
      stats.SetX2NDC(0.99)
      stats.SetY1NDC(0.95)
      stats.SetY2NDC(0.99)
  canv.SaveAs(f"{outputDirName}/{histCompName}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetLegendBorderSize(1)

  plotsDirName = "./polarized/2018_08/tbin_0.1_0.2/PiPi/plots_REAL_DATA/PARA_0"
  # plotsDirName = "./polarized/2018_08/tbin_0.1_0.2/PiPi/plots_ACCEPTED_PHASE_SPACE/PARA_0"
  plotsFile = ROOT.TFile.Open(f"{plotsDirName}/plots.root", "READ")

  for histBaseName in [
    "Ebeam",
    "momLabP",
    "momLabXP",
    "momLabYP",
    "momLabPip",
    "momLabXPip",
    "momLabYPip",
    "momLabPim",
    "momLabXPim",
    "momLabYPim",
    "thetaDegLabP",
    "thetaDegLabPip",
    "thetaDegLabPim",
    "phiDegLabP",
    "phiDegLabPip",
    "phiDegLabPim",
    "momLabYPVsMomLabXP",
    "momLabYPipVsMomLabXPip",
    "momLabYPimVsMomLabXPim",
    "thetaDegLabPVsMomLabP",
    "thetaDegLabPipVsMomLabPip",
    "thetaDegLabPimVsMomLabPim",
    "phiDegLabPVsThetaDegLabP",
    "phiDegLabPipVsThetaDegLabPip",
    "phiDegLabPimVsThetaDegLabPim",
  ]:
    makeComparisonPlotPosNegPhiHF(
      plotsFile     = plotsFile,
      histBaseName  = histBaseName,
      outputDirName = plotsDirName,
    )
