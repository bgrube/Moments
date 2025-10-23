#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
  fields,
)
import functools

import ROOT

from makeKinematicPlots import (
  bookHistogram,
  HistListType,
  HistogramDefinition,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class DataToOverlay:
  """Stores data frames for data and weighted MC"""
  realData:   ROOT.RDataFrame
  weightedMc: ROOT.RDataFrame

@dataclass
class HistsToOverlay:
  """Stores histograms for data and weighted MC"""
  # tuples are assumed to contain histograms in identical order
  realData:   HistListType = field(default_factory = list)
  weightedMc: HistListType = field(default_factory = list)


def plotDistributions1D(
  realDataFileName:   str,
  weightedMcFileName: str,
  treeName:           str,
  outputDirName:      str = ".",
  filter:             str = "(true)",
  pairLabel:          str = "PiPi",
  histTitle:          str = "",
  pdfFileMameSuffix:  str = "",
  yAxisLabel:         str = "RF-Sideband Subtracted Combos",
) -> None:
  """Overlays 1D distributions from real data and weighted Monte Carlo"""
  dataToOverlay = DataToOverlay(
    realData   = ROOT.RDataFrame(treeName, realDataFileName  ).Filter(filter),
    weightedMc = ROOT.RDataFrame(treeName, weightedMcFileName).Filter(filter),
  )
  hists1DToOverlay = HistsToOverlay()
  for member in fields(dataToOverlay):  # loop over members of `DataToOverlay`
    df    = getattr(dataToOverlay, member.name)
    label = member.name
    applyWeights = (label == "realData" and df.HasColumn("eventWeight"))
    label = label[0].upper() + label[1:]  # make sure first character is upper case
    label = pairLabel + label  # e.g. "PiPiRealData" or "PiPiWeightedMc"
    hists1D = [
      bookHistogram(df, HistogramDefinition(f"mass{label}",       histTitle + ";m_{#pi#pi} [GeV];" + yAxisLabel, ((50,    0.28,    2.28), ), ("mass",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"cosThetaHf{label}", histTitle + ";cos#theta_{HF};"   + yAxisLabel, ((50,   -1,      +1   ), ), ("cosTheta", )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"phiDegHf{label}",   histTitle + ";#phi_{HF} [deg];"  + yAxisLabel, ((50, -180,    +180   ), ), ("phiDeg",   )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PhiDeg{label}",     histTitle + ";#Phi [deg];"       + yAxisLabel, ((50, -180,    +180   ), ), ("PhiDeg",   )), applyWeights),
    ]
    setattr(hists1DToOverlay, member.name, hists1D)
  for histIndex, histRealData in enumerate(hists1DToOverlay.realData):
    histWeightedMc = hists1DToOverlay.weightedMc[histIndex]
    print(f"Overlaying histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
    canv = ROOT.TCanvas()
    histStack = ROOT.THStack(histWeightedMc.GetName(), histWeightedMc.GetTitle())
    histWeightedMc.SetTitle("Weighted MC")
    histRealData.SetTitle  ("Real data")
    histStack.Add(histWeightedMc.GetValue(), "HIST E")
    histStack.Add(histRealData.GetValue(),   "E")
    histRealData.SetLineColor  (ROOT.kRed + 1)
    histWeightedMc.SetLineColor(ROOT.kBlue + 1)
    histRealData.SetMarkerColor  (ROOT.kRed + 1)
    histWeightedMc.SetMarkerColor(ROOT.kBlue + 1)
    histWeightedMc.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
    scaleFactor = histRealData.Integral() / histWeightedMc.Integral()
    histWeightedMc.Scale(scaleFactor)
    histStack.Draw("NOSTACK")
    # histStack.SetMaximum(1.1 * histStack.GetMaximum())
    histStack.GetXaxis().SetTitle(histWeightedMc.GetXaxis().GetTitle())
    histStack.GetYaxis().SetTitle(histWeightedMc.GetYaxis().GetTitle())
    canv.BuildLegend(0.75, 0.85, 0.99, 0.99)
    chi2PerBin = histWeightedMc.Chi2Test(histRealData.GetValue(), "WW P CHI2") / histWeightedMc.GetNbinsX()
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignTop)
    label.DrawLatex(0.15, 0.89, f"#it{{#chi}}^{{2}}/bin = {chi2PerBin:.2f}")
    canv.SaveAs(f"{outputDirName}/{histStack.GetName()}{pdfFileMameSuffix}.pdf")


# def plotDistributions2D(
#   dataFileName:       str,
#   weightedMcFileName: str,
#   treeName:           str,
#   filter:             str,
#   pdfFileMameSuffix:  str = "",
# ) -> None:
#   """Plots 2D distributions from data and weighted Monte Carlo and their differences"""
#   dataToOverlay = DataToOverlay(
#     realData   = ROOT.RDataFrame(treeName, dataFileName      ).Filter(filter),
#     weightedMc = ROOT.RDataFrame(treeName, weightedMcFileName).Filter(filter),
#   )
#   hists2DToCompare = HistsToOverlay()
#   for member in fields(dataToOverlay):  # loop over members of dataclass
#     df    = getattr(dataToOverlay, member.name)
#     label = member.name
#     label = label[0].upper() + label[1:]  # make sure first character is upper case
#     hists2D = (
#       # bookHistogram2D(df, f"h{label}MassPiPiVsHfCosTheta", ";m_{#pi#pi} [GeV];cos#theta_{HF}",  (20, 0.28, 2.28, 25, -1,   +1  ), ("mass",     "cosTheta")),
#       # bookHistogram2D(df, f"h{label}MassPiPiVsHfPhiDeg",   ";m_{#pi#pi} [GeV];#phi_{HF} [deg]", (20, 0.28, 2.28, 25, -180, +180), ("mass",     "phiDeg"  )),
#       # bookHistogram2D(df, f"h{label}MassPiPiVsPhiDeg",     ";m_{#pi#pi} [GeV];#Phi [deg]",      (20, 0.28, 2.28, 25, -180, +180), ("mass",     "PhiDeg"  )),
#       bookHistogram2D(df, f"h{label}AnglesHf",             ";cos#theta_{HF};#phi_{HF} [deg]",   (25, -1,   +1,   25, -180, +180), ("cosTheta", "phiDeg"  )),
#       bookHistogram2D(df, f"h{label}PhiDegVsHfCosTheta",   ";cos#theta_{HF};#Phi [deg]",        (25, -1,   +1,   25, -180, +180), ("cosTheta", "PhiDeg"  )),
#       bookHistogram2D(df, f"h{label}PhiDegVsHfPhiDeg",     ";#phi_{HF} [deg];#Phi [deg]",       (25, -180, +180, 25, -180, +180), ("phiDeg",   "PhiDeg"  )),
#     )
#     setattr(hists2DToCompare, member.name, hists2D)
#   for histIndex, histData in enumerate(hists2DToCompare.realData):
#     print(f"Plotting histogram '{histData.GetName()}'")
#     histData.SetTitle("Data")
#     canv = ROOT.TCanvas()
#     histData.Draw("COLZ")
#     canv.SaveAs(f"{histData.GetName()}{pdfFileMameSuffix}.pdf")

#     histWeightedMc = hists2DToCompare.weightedMc[histIndex]
#     print(f"Plotting histograms '{histWeightedMc.GetName()}'")
#     histWeightedMc.SetTitle("Weighted MC")
#     canv = ROOT.TCanvas()
#     histWeightedMc.Draw("COLZ")
#     canv.SaveAs(f"{histWeightedMc.GetName()}{pdfFileMameSuffix}.pdf")

#     print(f"Plotting difference of histograms '{histData.GetName()}' - '{histWeightedMc.GetName()}'")
#     scaleFactor = histData.Integral() / histWeightedMc.Integral()
#     histWeightedMc.Scale(scaleFactor)
#     histPulls = histData.Clone(f"{histData.GetName()}_pulls")
#     histPulls.Add(histWeightedMc.GetValue(), -1)  # data - weighted MC
#     # divide each bin by its uncertainty to get pull
#     for xBin in range(1, histPulls.GetNbinsX() + 1):
#       for yBin in range(1, histPulls.GetNbinsY() + 1):
#         binError = histPulls.GetBinError(xBin, yBin)
#         if binError > 0:
#           histPulls.SetBinContent(xBin, yBin, histPulls.GetBinContent(xBin, yBin) / binError)
#         else:
#           histPulls.SetBinContent(xBin, yBin, 0)
#     histPulls.SetTitle("(Data #minus Weighted MC) / #sigma_{Data}")
#     canv = ROOT.TCanvas()
#     histPulls.Draw("COLZ")
#     canv.SaveAs(f"{histPulls.GetName()}{pdfFileMameSuffix}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  dataFileName       = "./polarized/2018_08/tbin_0.1_0.2/PiPi/data_flat_PARA_0.root"
  weightedMcFileName = "./polarized/2018_08/tbin_0.1_0.2/PiPi/weighted.maxL_4/PARA_0/data_weighted_flat.root"
  outputDirName      = "./polarized/2018_08/tbin_0.1_0.2/PiPi/weighted.maxL_4/PARA_0/"
  treeName           = "PiPi"
  massMin            = 0.28  # [GeV]
  massBinWidth       = 0.04  # [GeV]
  nmbBins            = 50

  print(f"Overlaying histograms for full mass range and writing plots into '{outputDirName}'")
  plotDistributions1D(
    realDataFileName   = dataFileName,
    weightedMcFileName = weightedMcFileName,
    treeName           = treeName,
    outputDirName      = outputDirName,
    histTitle          = f"{massMin:.2f} < m_{{#pi#pi}} < {massMin + nmbBins * massBinWidth:.2f} GeV",
  )
  # plotDistributions2D(
  #   dataFileName       = dataFileName,
  #   weightedMcFileName = weightedMcFileName,
  #   treeName           = treeName,
  # )
  for massBinIndex in range(nmbBins):
    massBinMin = massMin + massBinIndex * massBinWidth
    massBinMax = massBinMin + massBinWidth
    print(f"Overlaying histograms for mass range [{massBinMin:.2f}, {massBinMax:.2f}] GeV")
    massRangeFilter = f"(({massBinMin} < mass) && (mass < {massBinMax}))"
    plotDistributions1D(
      realDataFileName   = dataFileName,
      weightedMcFileName = weightedMcFileName,
      treeName           = treeName,
      outputDirName      = outputDirName,
      filter             = massRangeFilter,
      histTitle          = f"{massBinMin:.2f} < m_{{#pi#pi}} < {massBinMax:.2f} GeV",
      pdfFileMameSuffix  = f"_{massBinMin:.2f}_{massBinMax:.2f}",
    )
    # plotDistributions2D(
    #   dataFileName       = dataFileName,
    #   weightedMcFileName = weightedMcFileName,
    #   treeName           = treeName,
    #   filter             = massRangeFilter,
    #   pdfFileMameSuffix  = f"_{massBinMin:.2f}_{massBinMax:.2f}",
    # )
