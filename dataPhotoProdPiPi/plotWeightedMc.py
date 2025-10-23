#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
  fields,
)

import ROOT


def bookHistogram1D(
  df:         ROOT.RDataFrame,
  name:       str,
  title:      str,
  binning:    tuple[int, float, float],
  columnName: str,
) -> ROOT.RResultPtr[ROOT.TH1D]:
  """Books 1D histograms with or without event weight depending on the presence of the corresponding column in the input data"""
  if "eventWeight" in df.GetColumnNames():
    return df.Histo1D(ROOT.RDF.TH1DModel(name, title, *binning), columnName, "eventWeight")
  else:
    return df.Histo1D(ROOT.RDF.TH1DModel(name, title, *binning), columnName)

def bookHistogram2D(
  df:          ROOT.RDataFrame,
  name:        str,
  title:       str,
  binning:     tuple[int, float, float, int, float, float],
  columnNames: tuple[str, str],
) -> ROOT.RResultPtr[ROOT.TH2D]:
  """Books 2D histograms with or without event weight depending on the presence of the corresponding column in the input data"""
  if "eventWeight" in df.GetColumnNames():
    return df.Histo2D(ROOT.RDF.TH2DModel(name, title, *binning), *columnNames, "eventWeight")
  else:
    return df.Histo2D(ROOT.RDF.TH2DModel(name, title, *binning), *columnNames)


@dataclass
class DataToOverlay:
  """Stores data frames for data and weighted MC"""
  data:       ROOT.RDataFrame
  weightedMc: ROOT.RDataFrame

@dataclass
class HistsToOverlay:
  """Stores histograms for data and weighted MC"""
  # tuples are assumed to contain histograms in identical order
  data:       tuple[ROOT.RResultPtr[ROOT.TH1D], ...] = field(default_factory = tuple)
  weightedMc: tuple[ROOT.RResultPtr[ROOT.TH1D], ...] = field(default_factory = tuple)


def plotDistributions1D(
  dataFileName:       str,
  weightedMcFileName: str,
  treeName:           str,
  filter:             str,
  histTitle:          str = "",
  pdfFileMameSuffix:  str = "",
  yAxisLabel:         str = "RF-Sideband Subtracted Combos"
) -> None:
  """Overlays 1D distributions from data and weighted Monte Carlo"""
  dataToOverlay = DataToOverlay(
    data   = ROOT.RDataFrame(treeName, dataFileName      ).Filter(filter),
    weightedMc = ROOT.RDataFrame(treeName, weightedMcFileName).Filter(filter),
  )
  hists1DToOverlay = HistsToOverlay()
  for member in fields(dataToOverlay):  # loop over members of dataclass
    df    = getattr(dataToOverlay, member.name)
    label = member.name
    label = label[0].upper() + label[1:]  # make sure first character is upper case
    hists1D = (
      bookHistogram1D(df, f"h{label}MassPiPi",   histTitle + ";m_{#pi#pi} [GeV];" + yAxisLabel, (100,    0.28,    2.28), "mass"    ),
      bookHistogram1D(df, f"h{label}HfCosTheta", histTitle + ";cos#theta_{HF};"   + yAxisLabel, ( 50,   -1,      +1   ), "cosTheta"),
      bookHistogram1D(df, f"h{label}HfPhiDeg",   histTitle + ";#phi_{HF} [deg];"  + yAxisLabel, ( 50, -180,    +180   ), "phiDeg"  ),
      bookHistogram1D(df, f"h{label}PhiDeg",     histTitle + ";#Phi [deg];"       + yAxisLabel, ( 50, -180,    +180   ), "PhiDeg"  ),
    )
    setattr(hists1DToOverlay, member.name, hists1D)
  for histIndex, histData in enumerate(hists1DToOverlay.data):
    histWeightedMc = hists1DToOverlay.weightedMc[histIndex]
    print(f"Overlaying histograms '{histData.GetName()}' and '{histWeightedMc.GetName()}'")
    canv = ROOT.TCanvas()
    histStack = ROOT.THStack(histWeightedMc.GetName(), histWeightedMc.GetTitle())
    histWeightedMc.SetTitle("Weighted MC")
    histData.SetTitle      ("Data")
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
    # histStack.SetMaximum(1.1 * histStack.GetMaximum())
    histStack.GetXaxis().SetTitle(histWeightedMc.GetXaxis().GetTitle())
    histStack.GetYaxis().SetTitle(histWeightedMc.GetYaxis().GetTitle())
    canv.BuildLegend(0.75, 0.85, 0.99, 0.99)
    chi2PerBin = histWeightedMc.Chi2Test(histData.GetValue(), "WW P CHI2") / histWeightedMc.GetNbinsX()
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignTop)
    label.DrawLatex(0.15, 0.89, f"#it{{#chi}}^{{2}}/bin = {chi2PerBin:.2f}")
    canv.SaveAs(f"{histStack.GetName()}{pdfFileMameSuffix}.pdf")


def plotDistributions2D(
  dataFileName:       str,
  weightedMcFileName: str,
  treeName:           str,
  filter:             str,
  pdfFileMameSuffix:  str = "",
) -> None:
  """Plots 2D distributions from data and weighted Monte Carlo and their differences"""
  dataToOverlay = DataToOverlay(
    data   = ROOT.RDataFrame(treeName, dataFileName      ).Filter(filter),
    weightedMc = ROOT.RDataFrame(treeName, weightedMcFileName).Filter(filter),
  )
  hists2DToCompare = HistsToOverlay()
  for member in fields(dataToOverlay):  # loop over members of dataclass
    df    = getattr(dataToOverlay, member.name)
    label = member.name
    label = label[0].upper() + label[1:]  # make sure first character is upper case
    hists2D = (
      # bookHistogram2D(df, f"h{label}MassPiPiVsHfCosTheta", ";m_{#pi#pi} [GeV];cos#theta_{HF}",  (20, 0.28, 2.28, 25, -1,   +1  ), ("mass",     "cosTheta")),
      # bookHistogram2D(df, f"h{label}MassPiPiVsHfPhiDeg",   ";m_{#pi#pi} [GeV];#phi_{HF} [deg]", (20, 0.28, 2.28, 25, -180, +180), ("mass",     "phiDeg"  )),
      # bookHistogram2D(df, f"h{label}MassPiPiVsPhiDeg",     ";m_{#pi#pi} [GeV];#Phi [deg]",      (20, 0.28, 2.28, 25, -180, +180), ("mass",     "PhiDeg"  )),
      bookHistogram2D(df, f"h{label}AnglesHf",             ";cos#theta_{HF};#phi_{HF} [deg]",   (25, -1,   +1,   25, -180, +180), ("cosTheta", "phiDeg"  )),
      bookHistogram2D(df, f"h{label}PhiDegVsHfCosTheta",   ";cos#theta_{HF};#Phi [deg]",        (25, -1,   +1,   25, -180, +180), ("cosTheta", "PhiDeg"  )),
      bookHistogram2D(df, f"h{label}PhiDegVsHfPhiDeg",     ";#phi_{HF} [deg];#Phi [deg]",       (25, -180, +180, 25, -180, +180), ("phiDeg",   "PhiDeg"  )),
    )
    setattr(hists2DToCompare, member.name, hists2D)
  for histIndex, histData in enumerate(hists2DToCompare.data):
    print(f"Plotting histogram '{histData.GetName()}'")
    histData.SetTitle("Data")
    canv = ROOT.TCanvas()
    histData.Draw("COLZ")
    canv.SaveAs(f"{histData.GetName()}{pdfFileMameSuffix}.pdf")

    histWeightedMc = hists2DToCompare.weightedMc[histIndex]
    print(f"Plotting histograms '{histWeightedMc.GetName()}'")
    histWeightedMc.SetTitle("Weighted MC")
    canv = ROOT.TCanvas()
    histWeightedMc.Draw("COLZ")
    canv.SaveAs(f"{histWeightedMc.GetName()}{pdfFileMameSuffix}.pdf")

    print(f"Plotting difference of histograms '{histData.GetName()}' - '{histWeightedMc.GetName()}'")
    scaleFactor = histData.Integral() / histWeightedMc.Integral()
    histWeightedMc.Scale(scaleFactor)
    histPulls = histData.Clone(f"{histData.GetName()}_pulls")
    histPulls.Add(histWeightedMc.GetValue(), -1)  # data - weighted MC
    # divide each bin by its uncertainty to get pull
    for xBin in range(1, histPulls.GetNbinsX() + 1):
      for yBin in range(1, histPulls.GetNbinsY() + 1):
        binError = histPulls.GetBinError(xBin, yBin)
        if binError > 0:
          histPulls.SetBinContent(xBin, yBin, histPulls.GetBinContent(xBin, yBin) / binError)
        else:
          histPulls.SetBinContent(xBin, yBin, 0)
    histPulls.SetTitle("(Data #minus Weighted MC) / #sigma_{Data}")
    canv = ROOT.TCanvas()
    histPulls.Draw("COLZ")
    canv.SaveAs(f"{histPulls.GetName()}{pdfFileMameSuffix}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  dataFileName       = "./data_flat.root"
  weightedMcFileName = "./psAccData_weighted_flat.maxL_4.root"
  # weightedMcFileName = "./psAccData_weighted_flat.maxL_5.root"
  # weightedMcFileName = "./psAccData_weighted_flat.maxL_6.root"
  # weightedMcFileName = "./psAccData_weighted_flat.maxL_7.root"
  # weightedMcFileName = "./psAccData_weighted_flat.maxL_8.root"
  # weightedMcFileName = "./psAccData_weighted_pwa_SPD_flat.maxL_4.root"
  treeName           = "PiPi"
  massMin            = 0.28  # [GeV]
  massBinWidth       = 0.1   # [GeV]
  nmbBins            = 20

  print(f"Overlaying histograms for full mass range")
  plotDistributions1D(
    dataFileName       = dataFileName,
    weightedMcFileName = weightedMcFileName,
    treeName           = treeName,
    filter             = "(true)",
    histTitle          = f"{massMin:.2f} < m_{{#pi#pi}} < {massMin + nmbBins * massBinWidth:.2f} GeV",
  )
  plotDistributions2D(
    dataFileName       = dataFileName,
    weightedMcFileName = weightedMcFileName,
    treeName           = treeName,
    filter             = "(true)",
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
      pdfFileMameSuffix  = f"_{massBinMin:.2f}_{massBinMax:.2f}",
    )
    plotDistributions2D(
      dataFileName       = dataFileName,
      weightedMcFileName = weightedMcFileName,
      treeName           = treeName,
      filter             = massRangeFilter,
      pdfFileMameSuffix  = f"_{massBinMin:.2f}_{massBinMax:.2f}",
    )
