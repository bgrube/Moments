#!/usr/bin/env python3


from __future__ import annotations

import ROOT

import PlottingUtilities


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  PlottingUtilities.setupPlotStyle(rootlogonPath = "../../rootlogon.C")

  plotFileName = "./pwa_plots_SPD.root"
  plotFile = ROOT.TFile.Open(plotFileName, "READ")

  graphNamesToLabels = {
    "S0"    : "#it{S}_{0}",
    "Pm1"   : "#it{P}_{#kern[-0.3]{#minus 1}}",
    "P0"    : "#it{P}_{#kern[-0.3]{0}}",
    "P1"    : "#it{P}_{#kern[-0.3]{#plus 1}}",
    "Dm2"   : "#it{D}_{#minus 2}",
    "Dm2"   : "#it{D}_{#minus 2}",
    "D0"    : "#it{D}_{0}",
    "D1"    : "#it{D}_{#plus 1}",
    "D2"    : "#it{D}_{#plus 2}",
    "Total" : "Total",
  }

  axisTitles = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}} [GeV/#it{c}^{2}];Intensity / 20 MeV/#it{c}^{2}"
  for graphName, graphLabel in graphNamesToLabels.items():
    canv = ROOT.TCanvas()
    graphs = {"p" : None, "m" : None}
    if graphName == "Total":
      graph = plotFile.Get(graphName)
      graph.SetTitle(graphLabel)
      graphs["p"] = graph
    else:
      for refl, reflLabel in {"p" : "^{#plus}", "m" : "^{#minus}"}.items():
        graph = plotFile.Get(graphName + refl)
        graph.SetTitle(graphLabel + reflLabel)
        graphs[refl] = graph
    multiGraph = ROOT.TMultiGraph(graphName, f";{axisTitles}")
    for refl, graph in graphs.items():
      if graph is not None:
        graph.SetMarkerStyle(ROOT.kFullCircle)
        graph.SetMarkerSize(0.75)
        graph.SetMarkerColor(ROOT.kRed + 1 if refl == "p" else ROOT.kBlue + 1)
        graph.SetLineColor  (ROOT.kRed + 1 if refl == "p" else ROOT.kBlue + 1)
        multiGraph.Add(graph)
    multiGraph.SetMinimum(0)
    multiGraph.Draw("AP")
    multiGraph.GetXaxis().SetRangeUser(0.28, 2.28)  # [GeV]
    canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
    canv.SaveAs(f"intensity_{graphName}.pdf")
