#!/usr/bin/env python3


from __future__ import annotations

import os

import ROOT


def makeComparisonPlotPosNegPhiHF(
  plotsFile:     ROOT.TFile,  # file containing histograms to plot
  histBaseName:  str,  # name of histogram to plot
  outputDirName: str,  # directory to save output plot in
) -> None:
  """Generates comparison plot for positive and negative phi_HF"""
  print(f"Generating phi_HF < 0, phi_HF > 0 comparison plot for histogram '{histBaseName}'")
  histPos = plotsFile.Get(f"{histBaseName}_phiHFPiPiDegPos")
  histNeg = plotsFile.Get(f"{histBaseName}_phiHFPiPiDegNeg")
  histCompName = f"{histBaseName}_phiHFPiPiDegPosNeg"
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
    zRange = max(abs(histDiff.GetMinimum()), abs(histDiff.GetMaximum()))
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
    # # plot 2D asymmetry for phi > 0 and phi < 0
    # histDiff = histPos.Clone(f"{histBaseName}_phiHFPiPiDegPosNegDiff")
    # histDiff.Add(histNeg, -1)
    # histSum = histPos.Clone(f"{histBaseName}_phiHFPiPiDegPosNegSum")
    # histSum.Add(histNeg, +1)
    # histAsym = histDiff.Clone(histCompName)
    # histAsym.Divide(histSum)
    # histAsym.SetTitle("#phi_{HF} Asymmetry")
    # histAsym.SetMinimum(-0.1)
    # histAsym.SetMaximum(+0.1)
    # histAsym.SetStats(False)
    # histAsym.Draw("COLZ")
  canv.SaveAs(f"{outputDirName}/{histCompName}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetLegendBorderSize(1)

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

  dataType = "REAL_DATA"
  # dataType = "ACCEPTED_PHASE_SPACE"

  plotsDirName = f"./polarized/{dataPeriod}/{tBinLabel}/PiPi/plots_{dataType}/{beamPolLabel}"
  with ROOT.TFile.Open(f"{plotsDirName}/plots.root", "READ") as plotsFile:
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
      "thetaLabPDeg",
      "thetaLabPipDeg",
      "thetaLabPimDeg",
      "phiLabPDeg",
      "phiLabPipDeg",
      "phiLabPimDeg",
      "momLabYPVsMomLabXP",
      "momLabYPipVsMomLabXPip",
      "momLabYPimVsMomLabXPim",
      "thetaLabPDegVsMomLabP",
      "thetaLabPipDegVsMomLabPip",
      "thetaLabPimDegVsMomLabPim",
      "phiLabPDegVsThetaLabPDeg",
      "phiLabPipDegVsThetaLabPipDeg",
      "phiLabPimDegVsThetaLabPimDeg",
    ]:
      makeComparisonPlotPosNegPhiHF(
        plotsFile     = plotsFile,
        histBaseName  = histBaseName,
        outputDirName = plotsDirName,
      )
