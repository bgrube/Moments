#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
  fields,
)
import functools
import os

import ROOT

from makeKinematicPlots import (
  bookHistogram,
  defineColumnsForPlots,
  HistListType,
  HistogramDefinition,
)
from makeMomentsInputTree import (
  BEAM_POL_INFOS,
  CPP_CODE_BEAM_POL_PHI,
  CPP_CODE_FLIPYAXIS,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  getDataFrameWithCorrectEventWeights,
  InputDataFormat,
  SubSystemInfo,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class DataToOverlay:
  """Stores data frames for data and weighted MC"""
  realData:   ROOT.RDataFrame
  weightedMc: ROOT.RDataFrame

  def Filter(
    self,
    filterExpr: str
  ) -> DataToOverlay:
    """Returns a new `DataToOverlay` instance with data frames filtered by `filterExpr`"""
    return DataToOverlay(
      realData   = self.realData.Filter  (filterExpr),
      weightedMc = self.weightedMc.Filter(filterExpr),
    )


@dataclass
class HistsToOverlay:
  """Stores histograms for data and weighted MC"""
  # tuples are assumed to contain histograms in identical order
  realData:   HistListType = field(default_factory = list)
  weightedMc: HistListType = field(default_factory = list)


def makePlots(
  dataToOverlay:     DataToOverlay,
  outputDirName:     str = ".",
  pairLabel:         str = "PiPi",
  histTitle:         str = "",
  pdfFileNameSuffix: str = "",
  yAxisLabel:        str = "RF-Sideband Subtracted Combos",
  colNameSuffix:     str = "",  # suffix appended to column names
) -> None:
  """Overlays 1D distributions and compares 2D distributions from real data and weighted Monte Carlo"""
  histsToOverlay = HistsToOverlay()
  for member in fields(dataToOverlay):  # loop over members of `DataToOverlay`
    df = getattr(dataToOverlay, member.name)
    label = member.name
    applyWeights = (label == "realData" and df.HasColumn("eventWeight"))
    label = label[0].upper() + label[1:]  # make sure first character is upper case
    histNameSuffix = pairLabel + label  # e.g. "PiPiRealData" or "PiPiWeightedMc"
    hists = [
      # distributions in X rest frame
      # 1D histograms
      bookHistogram(df, HistogramDefinition(f"mass{histNameSuffix}",       histTitle + ";m_{#pi#pi} [GeV];" + yAxisLabel, (( 50,    0.28,    2.28), ), (f"mass{colNameSuffix}",       )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"cosThetaHF{histNameSuffix}", histTitle + ";cos#theta_{HF};"   + yAxisLabel, ((100,   -1,      +1   ), ), (f"cosThetaHF{colNameSuffix}", )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"phiDegHF{histNameSuffix}",   histTitle + ";#phi_{HF} [deg];"  + yAxisLabel, (( 72, -180,    +180   ), ), (f"phiDegHF{colNameSuffix}",   )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PhiDeg{histNameSuffix}",     histTitle + ";#Phi [deg];"       + yAxisLabel, (( 72, -180,    +180   ), ), (f"PhiDeg{colNameSuffix}",     )), applyWeights),
      # 2D histograms
      # bookHistogram(df, HistogramDefinition(f"mass{pairLabel}VsCosThetaHF{histNameSuffix}", ";m_{#pi#pi} [GeV];cos#theta_{HF}",  ((50, 0.28, 2.28), (100,   -1,   +1)), (f"mass{colNameSuffix}", "cosThetaHF{histNameSuffix}")), applyWeights),
      # bookHistogram(df, HistogramDefinition(f"mass{pairLabel}VsPhiDegHF{histNameSuffix}",   ";m_{#pi#pi} [GeV];#phi_{HF} [deg]", ((50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{colNameSuffix}", "phiDegHF{histNameSuffix}"  )), applyWeights),
      # bookHistogram(df, HistogramDefinition(f"mass{pairLabel}VsPhiDeg{histNameSuffix}",     ";m_{#pi#pi} [GeV];#Phi [deg]",      ((50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{colNameSuffix}", "PhiDeg"                    )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"anglesHF{histNameSuffix}",           ";cos#theta_{HF};#phi_{HF} [deg]", ((100, -1,   +1  ), (72, -180, +180)), (f"cosThetaHF{colNameSuffix}", f"phiDegHF{colNameSuffix}")), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PhiDegVsCosThetaHF{histNameSuffix}", ";cos#theta_{HF};#Phi [deg]",      ((100, -1,   +1  ), (72, -180, +180)), (f"cosThetaHF{colNameSuffix}", f"PhiDeg{colNameSuffix}"  )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PhiDegVsPhiDegHF{histNameSuffix}",   ";#phi_{HF} [deg];#Phi [deg]",     (( 72, -180, +180), (72, -180, +180)), (f"phiDegHF{colNameSuffix}",   f"PhiDeg{colNameSuffix}"  )), applyWeights),
    ]
    setattr(histsToOverlay, member.name, hists)
  for histIndex, histRealData in enumerate(histsToOverlay.realData):
    histWeightedMc = histsToOverlay.weightedMc[histIndex]
    # normalize weighted MC to integral of real data
    normFactor = histRealData.Integral() / histWeightedMc.Integral()
    histWeightedMc.Scale(normFactor)
    histWeightedMc.SetMinimum(0)
    histRealData.SetMinimum  (0)
    histWeightedMc.SetTitle("Weighted MC")
    histRealData.SetTitle  ("Real data")
    # generate plots
    ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
    if "TH1" in histRealData.ClassName():
      # generate 1D overlay plots
      print(f"Overlaying 1D histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
      ROOT.gStyle.SetOptStat(False)
      canv = ROOT.TCanvas()
      histStack = ROOT.THStack(histWeightedMc.GetName(), histWeightedMc.GetTitle())
      histRealData.SetLineColor  (ROOT.kRed + 1)
      histWeightedMc.SetLineColor(ROOT.kBlue + 1)
      histRealData.SetMarkerColor  (ROOT.kRed + 1)
      histWeightedMc.SetMarkerColor(ROOT.kBlue + 1)
      histWeightedMc.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
      histStack.Add(histWeightedMc.GetValue(), "HIST E")
      histStack.Add(histRealData.GetValue(),   "E")
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
      canv.SaveAs(f"{outputDirName}/{histStack.GetName()}{pdfFileNameSuffix}.pdf")
    elif "TH2" in histRealData.ClassName():
      # generate 2D comparison plots
      print(f"Plotting real-data 2D histogram '{histRealData.GetName()}'")
      ROOT.gStyle.SetOptStat("i")
      canv = ROOT.TCanvas()
      maxZ = histRealData.GetMaximum() * 1.1
      histRealData.SetMaximum(maxZ)
      histRealData.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{histRealData.GetName()}{pdfFileNameSuffix}.pdf")
      print(f"Plotting weighted-MC 2D histogram '{histWeightedMc.GetName()}'")
      canv = ROOT.TCanvas()
      histWeightedMc.SetMaximum(maxZ)
      histWeightedMc.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{histWeightedMc.GetName()}{pdfFileNameSuffix}.pdf")
      print(f"Plotting pulls of 2D histograms '{histRealData.GetName()}' - '{histWeightedMc.GetName()}'")
      ROOT.gStyle.SetOptStat(False)
      histPulls = histRealData.Clone(f"{histWeightedMc.GetName()}_pulls")
      histPulls.Add(histWeightedMc.GetValue(), -1)  # real data - weighted MC
      # divide each bin by its uncertainty to get pull
      for xBin in range(1, histPulls.GetNbinsX() + 1):
        for yBin in range(1, histPulls.GetNbinsY() + 1):
          binError = histPulls.GetBinError(xBin, yBin)
          if binError > 0:
            histPulls.SetBinContent(xBin, yBin, histPulls.GetBinContent(xBin, yBin) / binError)
          else:
            histPulls.SetBinContent(xBin, yBin, 0)
      histPulls.SetTitle("(Real data#minus Weighted MC)/#sigma_{Real data}")
      canv = ROOT.TCanvas()
      #TODO improve color scale and z-range for pulls
      histPulls.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{histPulls.GetName()}{pdfFileNameSuffix}.pdf")
    else:
      raise RuntimeError(f"Unsupported histogram type '{histRealData.ClassName()}'")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)

  dataPeriod   = "2018_08"
  tBinLabel    = "tbin_0.1_0.2"
  beamPolLabel = "PARA_0"
  massMin      = 0.28  # [GeV]
  massBinWidth = 0.04  # [GeV]
  nmbBins      = 50

  # realDataDirName     = f"./polarized/{dataPeriod}/{tBinLabel}/PiPi"
  # weightedDataDirName = f"{realDataDirName}/weighted.maxL_4/{beamPolLabel}"
  # dataToOverlay = DataToOverlay(
  #   realData   = ROOT.RDataFrame("PiPi", f"{realDataDirName}/data_flat_{beamPolLabel}.root"),
  #   weightedMc = ROOT.RDataFrame("PiPi", f"{weightedDataDirName}/data_weighted_flat.root"  ),
  # )

  dataDirName         = f"./polarized/{dataPeriod}/{tBinLabel}"
  # weightedDataDirName = f"{dataDirName}/PiPi/weighted.maxL_4/{beamPolLabel}"
  weightedDataDirName = f".."
  dataToOverlay = DataToOverlay(
    realData   = getDataFrameWithCorrectEventWeights(
      dataSigRegionFileNames  = (f"{dataDirName}/Alex/amptools_tree_signal_{beamPolLabel}.root", ),
      dataBkgRegionFileNames  = (f"{dataDirName}/Alex/amptools_tree_bkgnd_{beamPolLabel}.root",  ),
      treeName                = "kin",
      friendSigRegionFileName = f"./polarized/{dataPeriod}/{tBinLabel}/data_sig_{beamPolLabel}.root.weights",
      friendBkgRegionFileName = f"./polarized/{dataPeriod}/{tBinLabel}/data_bkg_{beamPolLabel}.root.weights",
    ),
    weightedMc = ROOT.RDataFrame("PiPi", f"{weightedDataDirName}/data_weighted_flat.root"),
  )
  for member in fields(dataToOverlay):  # loop over members of `DataToOverlay`
    df = getattr(dataToOverlay, member.name)
    df = defineColumnsForPlots(
      df              = df,
      inputDataFormat = InputDataFormat.AMPTOOLS,
      subSystem       = SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim", lvRecoilLabel = "recoil", pairTLatexLabel = "#pi#pi"),
      beamPolInfo     = BEAM_POL_INFOS[dataPeriod][beamPolLabel],
    )
    setattr(dataToOverlay, member.name, df)

  print(f"Overlaying histograms for full mass range and writing plots into '{weightedDataDirName}'")
  colNameSuffix = "PiPi"
  makePlots(
    dataToOverlay = dataToOverlay,
    outputDirName = weightedDataDirName,
    histTitle     = f"{massMin:.2f} < m_{{#pi#pi}} < {massMin + nmbBins * massBinWidth:.2f} GeV",
    colNameSuffix = colNameSuffix,
  )
  for massBinIndex in range(nmbBins):
    massBinMin = massMin + massBinIndex * massBinWidth
    massBinMax = massBinMin + massBinWidth
    print(f"Overlaying histograms for mass range [{massBinMin:.2f}, {massBinMax:.2f}] GeV")
    massRangeFilter = f"(({massBinMin} < mass{colNameSuffix}) && (mass{colNameSuffix} < {massBinMax}))"
    makePlots(
      dataToOverlay     = dataToOverlay.Filter(massRangeFilter),
      outputDirName     = weightedDataDirName,
      histTitle         = f"{massBinMin:.2f} < m_{{#pi#pi}} < {massBinMax:.2f} GeV",
      pdfFileNameSuffix = f"_{massBinMin:.2f}_{massBinMax:.2f}",
      colNameSuffix     = colNameSuffix,
    )
