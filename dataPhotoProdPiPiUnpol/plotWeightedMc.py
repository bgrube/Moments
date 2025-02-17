#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
  fields,
)

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


@dataclass
class DataToOverlay:
  """Stores data frames for real data and weighted MC"""
  realData:   ROOT.RDataFrame
  weightedMc: ROOT.RDataFrame

@dataclass
class HistsToOverlay:
  """Stores histograms for real data and weighted MC"""
  # tuples are assumed to contain histograms in identical order
  realData:   tuple[ROOT.RResultPtr[ROOT.TH1D], ...] = field(default_factory = tuple)
  weightedMc: tuple[ROOT.RResultPtr[ROOT.TH1D], ...] = field(default_factory = tuple)


def plotDistributions1D(
  dataFileName:       str,
  weightedMcFileName: str,
  treeName:           str,
  filter:             str,
  histTitle:          str = "",
  pdfOutputDir:       str = ".",
  pdfFileMameSuffix:  str = "",
  yAxisLabel:         str = "Events"
) -> None:
  dataToOverlay = DataToOverlay(
    realData   = ROOT.RDataFrame(treeName, dataFileName      ).Filter(filter),
    weightedMc = ROOT.RDataFrame(treeName, weightedMcFileName).Filter(filter),
  )
  histsToOverlay = HistsToOverlay()
  for member in fields(dataToOverlay):  # loop over members of dataclass
    df    = getattr(dataToOverlay, member.name)
    label = member.name
    label = label[0].upper() + label[1:]  # make sure first character is upper case
    hists = (
      bookHistogram(df, f"h{label}MassPiPi",   histTitle + ";m_{#pi#pi} [GeV];" + yAxisLabel, (56,    0.28,    1.40), "mass"    ),
      bookHistogram(df, f"h{label}HfCosTheta", histTitle + ";cos#theta_{HF};"   + yAxisLabel, (25,   -1,      +1   ), "cosTheta"),
      bookHistogram(df, f"h{label}HfPhiDeg",   histTitle + ";#phi_{HF} [deg];"  + yAxisLabel, (25, -180,    +180   ), "phiDeg"  ),
    )
    setattr(histsToOverlay, member.name, hists)
  for histIndex, histData in enumerate(histsToOverlay.realData):
    histWeightedMc = histsToOverlay.weightedMc[histIndex]
    print(f"Overlaying histograms '{histData.GetName()}' and '{histWeightedMc.GetName()}'")
    canv = ROOT.TCanvas()
    histStack = ROOT.THStack(histWeightedMc.GetName(), histWeightedMc.GetTitle())
    histWeightedMc.SetTitle("Weighted MC")
    histData.SetTitle      ("Real Data")
    histStack.Add(histWeightedMc.GetValue(), "HIST E")
    histStack.Add(histData.GetValue(),       "EP")
    histData.SetLineColor      (ROOT.kRed + 1)
    histWeightedMc.SetLineColor(ROOT.kBlue + 1)
    histData.SetMarkerStyle      (ROOT.kFullCircle)
    histData.SetMarkerSize       (0.75)
    histData.SetMarkerColor      (ROOT.kRed + 1)
    histWeightedMc.SetMarkerColor(ROOT.kBlue + 1)
    histWeightedMc.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
    scaleFactor = histData.Integral() / histWeightedMc.Integral()
    histWeightedMc.Scale(scaleFactor)
    histStack.Draw("NOSTACK")
    # histStack.SetMaximum(1.1 * histStack.GetMaximum())
    histStack.GetXaxis().SetTitle(histWeightedMc.GetXaxis().GetTitle())
    histStack.GetYaxis().SetTitle(histWeightedMc.GetYaxis().GetTitle())
    canv.BuildLegend(0.75, 0.85, 0.99, 0.99)
    chi2PerBin = histWeightedMc.Chi2Test(histData.GetValue(), "WW P CHI2") / histWeightedMc.GetNbinsX()
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignTop)
    label.DrawLatex(0.15, 0.89, f"#it{{#chi}}^{{2}}/bin = {chi2PerBin:.2f}")
    canv.SaveAs(f"{pdfOutputDir}/{histStack.GetName()}{pdfFileMameSuffix}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gROOT.LoadMacro("../rootlogon.C")
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  dataFileName = "./data_flat.PiPi.root"  # real data
  for maxL in (2, 4, 5, 6, 8, 10, 12, 14):
    weightedMcDirName  = f"../plotsPhotoProdPiPiUnpol.maxL_{maxL}"
    weightedMcFileName = f"{weightedMcDirName}/psAccData_weighted_flat.maxL_{maxL}.root"
    # weightedMcFileName = f"{weightedMcDirName}/psAccData_weighted_pwa_SPD_flat.maxL_{maxL}.root"
    treeName           = "PiPi"
    # CLAS binning
    massMin            = 0.4  # [GeV]
    massBinWidth       = 0.1  # [GeV]
    nmbBins            = 10
    # # PWA binning
    # massMin            = 0.28  # [GeV]
    # massBinWidth       = 0.08  # [GeV]
    # nmbBins            = 14

    print(f"Overlaying histograms for full mass range")
    plotDistributions1D(
      dataFileName       = dataFileName,
      weightedMcFileName = weightedMcFileName,
      treeName           = treeName,
      filter             = "(true)",
      histTitle          = f"{massMin:.2f} < m_{{#pi#pi}} < {massMin + nmbBins * massBinWidth:.2f} GeV",
      pdfOutputDir       = weightedMcDirName,
    )
    for massBinIndex in range(nmbBins):
      massBinMin = massMin + massBinIndex * massBinWidth
      massBinMax = massBinMin + massBinWidth
      print(f"Overlaying histograms for mass range [{massBinMin:.2f}, {massBinMax:.2f}] GeV")
      massRangeFilter = f"(({massBinMin} < mass) && (mass < {massBinMax}))"
      plotDistributions1D(
        dataFileName       = dataFileName,
        weightedMcFileName = weightedMcFileName,
        treeName           = treeName,
        filter             = massRangeFilter,
        histTitle          = f"{massBinMin:.2f} < m_{{#pi#pi}} < {massBinMax:.2f} GeV",
        pdfOutputDir       = weightedMcDirName,
        pdfFileMameSuffix  = f"_{massBinMin:.2f}_{massBinMax:.2f}",
      )

    if False:
      # overlaying weight distributions for 2 mass bins
      weightedMc = ROOT.RDataFrame(treeName, weightedMcFileName)
      histsWeight = (
        weightedMc.Filter("((0.64 < mass) && (mass < 0.66))").Histo1D(ROOT.RDF.TH1DModel("hWeightedMcWeights_0.65", "m_{#pi#pi} = 0.65 GeV", 100, 0, 4e3), "intensityWeight"),
        weightedMc.Filter("((0.66 < mass) && (mass < 0.68))").Histo1D(ROOT.RDF.TH1DModel("hWeightedMcWeights_0.67", "m_{#pi#pi} = 0.67 GeV", 100, 0, 4e3), "intensityWeight"),
      )
      canv = ROOT.TCanvas()
      # canv.SetLogy(1)
      histStack = ROOT.THStack("hWeightedMcWeights", ";Weight;Events")
      histStack.Add(histsWeight[0].GetValue())
      histStack.Add(histsWeight[1].GetValue())
      histsWeight[0].SetLineColor(ROOT.kRed + 1)
      histsWeight[1].SetLineColor(ROOT.kBlue + 1)
      histsWeight[0].SetMarkerColor(ROOT.kRed + 1)
      histsWeight[1].SetMarkerColor(ROOT.kBlue + 1)
      histStack.Draw("NOSTACK")
      canv.BuildLegend(0.75, 0.85, 0.99, 0.99)
      canv.SaveAs(f"{weightedMcDirName}/{histStack.GetName()}.pdf")
