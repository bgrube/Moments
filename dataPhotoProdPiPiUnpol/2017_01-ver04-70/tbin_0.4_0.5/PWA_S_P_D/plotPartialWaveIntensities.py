#!/usr/bin/env python3


from __future__ import annotations

import ROOT

import PlottingUtilities


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  PlottingUtilities.setupPlotStyle(rootlogonPath = "../../rootlogon.C")

  plotFileName = "./pwa_plots_weight1.root"
  plotFile = ROOT.TFile.Open(plotFileName, "READ")

  graphNamesToLabels = {
    "S0"    : "#it{S}_{0}",
    "Pm1"   : "#it{P}_{#kern[-0.3]{#minus 1}}",
    "P0"    : "#it{P}_{#kern[-0.3]{0}}",
    "P1"    : "#it{P}_{#kern[-0.3]{#plus 1}}",
    "Dm2"   : "#it{D}_{#minus 2}",
    "Dm1"   : "#it{D}_{#minus 1}",
    "D0"    : "#it{D}_{0}",
    "D1"    : "#it{D}_{#plus 1}",
    "D2"    : "#it{D}_{#plus 2}",
    "Total" : "Total",
  }

  for graphName, graphLabel in graphNamesToLabels.items():
    graph = plotFile.Get(graphName)
    canv = ROOT.TCanvas()
    graph.Draw("AP")
    graph.SetMarkerStyle(ROOT.kFullCircle)
    graph.SetMarkerSize(0.75)
    graph.SetMarkerColor(ROOT.kBlue + 1)
    graph.SetLineColor(ROOT.kBlue + 1)
    graph.SetMinimum(0)
    if graphName == "Total":
      graph.SetMaximum(55e3)
    else:
      graph.SetMaximum(15e3)
    graph.SetTitle(graphLabel + ";#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}} [GeV/#it{c}^{2}];Intensity / 20 MeV/#it{c}^{2}")
    canv.SaveAs(f"intensity_{graphName}.pdf")
