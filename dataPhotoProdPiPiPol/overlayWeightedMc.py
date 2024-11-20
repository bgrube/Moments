#!/usr/bin/env python3


from __future__ import annotations

import ROOT


def bookHistogram(
  df:         ROOT.RDataFrame,
  name:       str,
  title:      str,
  binning:    tuple[int, float, float],
  columnName: str,
) -> ROOT.RResultPtr[ROOT.TH1D]:
  """Books histograms with or without event weight depending on the presence of the corresponding column in the input data"""
  if "eventWeight" in df.GetColumnNames():
    return df.Histo1D(ROOT.RDF.TH1DModel(name, title, *binning), columnName, "eventWeight")
  else:
    return df.Histo1D(ROOT.RDF.TH1DModel(name, title, *binning), columnName)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.gStyle.SetLegendFillColor(ROOT.kWhite)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  dataFileName       = "./data_flat.root"
  weightedMcFileName = "./psAccData_weighted_flat.maxL_4.root"
  treeName           = "PiPi"
  massRange          = (0.7, 0.8)

  massRangeFilter = f"(({massRange[0]} < mass) && (mass < {massRange[1]}))"
  dataToOverlay: dict[str, ROOT.RDataFrame] = {
    "RealData"   : ROOT.RDataFrame(treeName, dataFileName      ).Filter(massRangeFilter),
    "WeightedMc" : ROOT.RDataFrame(treeName, weightedMcFileName).Filter(massRangeFilter),
  }

  hists: dict[str, tuple[ROOT.RResultPtr[ROOT.TH1D], ...]]  = {}
  yAxisLabel = "RF-Sideband Subtracted Combos"
  for label, df in dataToOverlay.items():
    hists[label] = (
      bookHistogram(df, f"h{label}MassPiPi",   ";m_{#pi#pi} [GeV];" + yAxisLabel, (100,    0.28,    2.28), "mass"    ),
      bookHistogram(df, f"h{label}HfCosTheta", ";cos#theta_{HF};"   + yAxisLabel, ( 50,   -1,      +1   ), "cosTheta"),
      bookHistogram(df, f"h{label}HfPhiDeg",   ";#phi_{HF} [deg];"  + yAxisLabel, ( 50, -180,    +180   ), "phiDeg"  ),
      bookHistogram(df, f"h{label}PhiDeg",     ";#Phi [deg];"       + yAxisLabel, ( 50, -180,    +180   ), "PhiDeg"  ),
    )
  for histIndex, histData in enumerate(hists["RealData"]):
    print(f"Overlaying histograms '{histData.GetName()}'")
    histWeightedMc = hists["WeightedMc"][histIndex]
    canv = ROOT.TCanvas()
    histStack = ROOT.THStack(histWeightedMc.GetName(), histWeightedMc.GetTitle())
    histWeightedMc.SetName("Weighted MC")
    histData.SetName("Real Data")
    histStack.Add(histWeightedMc.GetValue(), "HIST E")
    histStack.Add(histData.GetValue(),       "E")
    histData.SetLineColor      (ROOT.kRed + 1)
    histWeightedMc.SetLineColor(ROOT.kBlue + 1)
    histData.SetMarkerColor      (ROOT.kRed + 1)
    histWeightedMc.SetMarkerColor(ROOT.kBlue + 1)
    histWeightedMc.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
    scaleFactor = histData.Integral() / histWeightedMc.Integral()
    histWeightedMc.Scale(scaleFactor)
    histStack.Draw("NOSTACK")
    histStack.GetXaxis().SetTitle(histWeightedMc.GetXaxis().GetTitle())
    histStack.GetYaxis().SetTitle(histWeightedMc.GetYaxis().GetTitle())
    canv.BuildLegend(0.75, 0.85, 0.99, 0.99)
    chi2PerBin = histWeightedMc.Chi2Test(histData.GetValue(), "WW P CHI2") / histWeightedMc.GetNbinsX()
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignTop)
    label.DrawLatex(0.15, 0.99, f"#it{{#chi}}^{{2}}/bin = {chi2PerBin:.2f}")
    canv.SaveAs(f"{histStack.GetName()}.pdf")
