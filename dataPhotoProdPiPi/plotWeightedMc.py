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
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
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
  pdfFileNameSuffix: str = "",
  yAxisLabel:        str = "RF-Sideband Subtracted Combos",
  colNameSuffix:     str = "",  # suffix appended to column names
) -> None:
  """Overlays 1D distributions and compares 2D distributions from real data and weighted Monte Carlo"""
  histsToOverlay = HistsToOverlay()
  # loop over members of `DataToOverlay` and book histograms for `realData` and `weightedMc`
  for member in fields(dataToOverlay):
    df = getattr(dataToOverlay, member.name)
    label = member.name
    applyWeights = (label == "realData" and df.HasColumn("eventWeight"))
    label = label[0].upper() + label[1:]  # make sure first character is upper case
    histNameSuffix = pairLabel + "_" + label  # e.g. "PiPi_RealData" or "PiPi_WeightedMc"
    hists = [
      # distributions in X rest frame
      # 1D histograms
      bookHistogram(df, HistogramDefinition(f"mass{histNameSuffix}",       ";m_{#pi#pi} [GeV];"                   + yAxisLabel, (( 50,    0.28,    2.28), ), (f"mass{colNameSuffix}",       )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"minusT{histNameSuffix}",     ";#minus t_{#pi#pi} [GeV^{2}];"        + yAxisLabel, ((100,    0,       1   ), ), (f"minusT{colNameSuffix}",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"cosThetaHF{histNameSuffix}", ";cos#theta_{HF};"                     + yAxisLabel, ((100,   -1,      +1   ), ), (f"cosThetaHF{colNameSuffix}", )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"phiDegHF{histNameSuffix}",   ";#phi_{HF} [deg];"                    + yAxisLabel, (( 72, -180,    +180   ), ), (f"phiDegHF{colNameSuffix}",   )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PhiDeg{histNameSuffix}",     ";#Phi [deg];"                         + yAxisLabel, (( 72, -180,    +180   ), ), (f"PhiDeg{colNameSuffix}",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PsiDegHF{histNameSuffix}",   ";#Psi = (#Phi#minus#phi_{HF}) [deg];" + yAxisLabel, (( 72, -180,    +180   ), ), (f"PsiDegHF{colNameSuffix}",   )), applyWeights),
      # 2D histograms
      # bookHistogram(df, HistogramDefinition(f"CosThetaHF{pairLabel}VsMass{histNameSuffix}", ";m_{#pi#pi} [GeV];cos#theta_{HF}",  ((50, 0.28, 2.28), (50,   -1,   +1)), (f"mass{colNameSuffix}", "cosThetaHF{colNameSuffix}")), applyWeights),
      # bookHistogram(df, HistogramDefinition(f"PhiDegHF{pairLabel}VsMass{histNameSuffix}",   ";m_{#pi#pi} [GeV];#phi_{HF} [deg]", ((50, 0.28, 2.28), (36, -180, +180)), (f"mass{colNameSuffix}", "phiDegHF{colNameSuffix}"  )), applyWeights),
      # bookHistogram(df, HistogramDefinition(f"PhiDeg{pairLabel}VsMass{histNameSuffix}",     ";m_{#pi#pi} [GeV];#Phi [deg]",      ((50, 0.28, 2.28), (36, -180, +180)), (f"mass{colNameSuffix}", "PhiDeg"                   )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"anglesHF{histNameSuffix}",                 ";cos#theta_{HF};#phi_{HF} [deg]", ((50,   -1,   +1), (36, -180, +180)), (f"cosThetaHF{colNameSuffix}", f"phiDegHF{colNameSuffix}")), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PhiDegVsCosThetaHF{histNameSuffix}",       ";cos#theta_{HF};#Phi [deg]",      ((50,   -1,   +1), (36, -180, +180)), (f"cosThetaHF{colNameSuffix}", f"PhiDeg{colNameSuffix}"  )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PhiDegVsPhiDegHF{histNameSuffix}",         ";#phi_{HF} [deg];#Phi [deg]",     ((36, -180, +180), (36, -180, +180)), (f"phiDegHF{colNameSuffix}",   f"PhiDeg{colNameSuffix}"  )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PsiDegHFPiPiVsCosThetaHF{histNameSuffix}", ";cos#theta_{HF};#Psi [deg]",      ((50,   -1,   +1), (36, -180, +180)), (f"cosThetaHF{colNameSuffix}", f"PsiDegHF{colNameSuffix}")), applyWeights),
      bookHistogram(df, HistogramDefinition(f"PsiDegHFPiPiVsPhiDegHF{histNameSuffix}",   ";#phi_{HF} [deg];#Psi [deg]",     ((36, -180, +180), (36, -180, +180)), (f"phiDegHF{colNameSuffix}",   f"PsiDegHF{colNameSuffix}")), applyWeights),
      # distributions in lab frame
      # 1D histograms
      bookHistogram(df, HistogramDefinition(f"Ebeam_{histNameSuffix}",          ";E_{beam} [GeV];"                    + yAxisLabel, ((100,    8,    9  ), ), ("Ebeam",          )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabP_{histNameSuffix}",        ";p_{p} [GeV];"                       + yAxisLabel, ((100,    0,    1  ), ), ("momLabP",        )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabXP_{histNameSuffix}",       ";p_{x}^{p} [GeV];"                   + yAxisLabel, ((100,   -0.5, +0.5), ), ("momLabXP",       )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabYP_{histNameSuffix}",       ";p_{y}^{p} [GeV];"                   + yAxisLabel, ((100,   -0.5, +0.5), ), ("momLabYP",       )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabZP_{histNameSuffix}",       ";p_{z}^{p} [GeV];"                   + yAxisLabel, ((100,    0,    0.5), ), ("momLabZP",       )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabPip_{histNameSuffix}",      ";p_{#pi^{#plus}} [GeV];"             + yAxisLabel, ((100,    0,   10  ), ), ("momLabPip",      )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabXPip_{histNameSuffix}",     ";p_{x}^{#pi^{#plus}} [GeV];"         + yAxisLabel, ((100,   -1.2, +1.2), ), ("momLabXPip",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabYPip_{histNameSuffix}",     ";p_{y}^{#pi^{#plus}} [GeV];"         + yAxisLabel, ((100,   -1.2, +1.2), ), ("momLabYPip",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabZPip_{histNameSuffix}",     ";p_{z}^{#pi^{#plus}} [GeV];"         + yAxisLabel, ((100,   -1,   +9  ), ), ("momLabZPip",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabPim_{histNameSuffix}",      ";p_{#pi^{#minus}} [GeV];"            + yAxisLabel, ((100,    0,   10  ), ), ("momLabPim",      )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabXPim_{histNameSuffix}",     ";p_{x}^{#pi^{#minus}} [GeV];"        + yAxisLabel, ((100,   -1.2, +1.2), ), ("momLabXPim",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabYPim_{histNameSuffix}",     ";p_{y}^{#pi^{#minus}} [GeV];"        + yAxisLabel, ((100,   -1.2, +1.2), ), ("momLabYPim",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabZPim_{histNameSuffix}",     ";p_{z}^{#pi^{#minus}} [GeV];"        + yAxisLabel, ((100,   -1,   +9  ), ), ("momLabZPim",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"thetaDegLabP_{histNameSuffix}",   ";#theta_{p}^{lab} [deg];"            + yAxisLabel, ((100,    0,   80  ), ), ("thetaDegLabP",   )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"thetaDegLabPip_{histNameSuffix}", ";#theta_{#pi^{#plus}}^{lab} [deg];"  + yAxisLabel, ((100,    0,   80  ), ), ("thetaDegLabPip", )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"thetaDegLabPim_{histNameSuffix}", ";#theta_{#pi^{#minus}}^{lab} [deg];" + yAxisLabel, ((100,    0,   80  ), ), ("thetaDegLabPim", )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"phiDegLabP_{histNameSuffix}",     ";#phi_{p}^{lab} [deg];"              + yAxisLabel, ((100, -180, +180  ), ), ("phiDegLabP",     )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"phiDegLabPip_{histNameSuffix}",   ";#phi_{#pi^{#plus}}^{lab} [deg];"    + yAxisLabel, ((100, -180, +180  ), ), ("phiDegLabPip",   )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"phiDegLabPim_{histNameSuffix}",   ";#phi_{#pi^{#minus}}^{lab} [deg];"   + yAxisLabel, ((100, -180, +180  ), ), ("phiDegLabPim",   )), applyWeights),
      # 2D histograms
      bookHistogram(df, HistogramDefinition(f"momLabYPVsMomLabXP_{histNameSuffix}",        ";p_{x}^{p} [GeV];p_{y}^{p} [GeV];",                         (( 50, -0.5, +0.5), ( 50, -0.5, +0.5)), ("momLabXP",   "momLabYP"      )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabYPipVsMomLabXPip_{histNameSuffix}",    ";p_{x}^{#pi^{#plus}} [GeV];p_{y}^{#pi^{#plus}} [GeV];",     (( 50, -0.8, +0.8), ( 50, -0.8, +0.8)), ("momLabXPip", "momLabYPip"    )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"momLabYPimVsMomLabXPim_{histNameSuffix}",    ";p_{x}^{#pi^{#minus}} [GeV];p_{y}^{#pi^{#minus}} [GeV];",   (( 50, -0.8, +0.8), ( 50, -0.8, +0.8)), ("momLabXPim", "momLabYPim"    )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"thetaDegLabPVsMomLabP_{histNameSuffix}",     ";p_{p} [GeV];#theta_{p}^{lab} [deg]",                       ((100,  0,    1),   (100, 60,   80  )), ("momLabP",    "thetaDegLabP"  )), applyWeights),
      bookHistogram(df, HistogramDefinition(f"thetaDegLabPipVsMomLabPip_{histNameSuffix}", ";p_{#pi^{#plus}} [GeV];#theta_{#pi^{#plus}}^{lab} [deg]",   ((100,  0,   10),   (100,  0,   30  )), ("momLabPip",  "thetaDegLabPip")), applyWeights),
      bookHistogram(df, HistogramDefinition(f"thetaDegLabPimVsMomLabPim_{histNameSuffix}", ";p_{#pi^{#minus}} [GeV];#theta_{#pi^{#minus}}^{lab} [deg]", ((100,  0,   10),   (100,  0,   30  )), ("momLabPim",  "thetaDegLabPim")), applyWeights),
    ]
    setattr(histsToOverlay, member.name, hists)
  for histIndex, histRealData in enumerate(histsToOverlay.realData):
    histWeightedMc = histsToOverlay.weightedMc[histIndex]
    print(f"Comparing histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
    histWeightedMc.SetMinimum(0)
    histRealData.SetMinimum  (0)
    histWeightedMc.SetTitle("Weighted MC")
    histRealData.SetTitle  ("Real data")
    # normalize weighted MC to integral of real data
    weightedMcIntegral = histWeightedMc.Integral()
    if weightedMcIntegral != 0:
      histWeightedMc.Scale(histRealData.Integral() / weightedMcIntegral)
    else:
      print(f"??? Warning: weighted-MC histogram '{histWeightedMc.GetName()}' has zero integral, cannot normalize to real data!")
    # generate plots
    ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
    if "TH1" in histRealData.ClassName():
      # generate 1D overlay plots
      print(f"Overlaying 1D histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
      print("(under-, overflow) fractions: "
            f"'{histRealData.GetName()  }' = ({histRealData.GetBinContent(0)                            / histRealData.Integral()}, "
                                            f"{histRealData.GetBinContent(histRealData.GetNbinsX() + 1) / histRealData.Integral()})"
            f" and '{histWeightedMc.GetName()}' = ({histWeightedMc.GetBinContent(0)                              / histWeightedMc.Integral()}, "
                                                 f"{histWeightedMc.GetBinContent(histWeightedMc.GetNbinsX() + 1) / histWeightedMc.Integral()})")
      ROOT.gStyle.SetOptStat(False)
      canv = ROOT.TCanvas()
      histStack = ROOT.THStack(histWeightedMc.GetName(), "")
      histRealData.SetLineColor  (ROOT.kRed + 1)
      histWeightedMc.SetLineColor(ROOT.kBlue + 1)
      histRealData.SetMarkerColor  (ROOT.kRed + 1)
      histWeightedMc.SetMarkerColor(ROOT.kBlue + 1)
      histWeightedMc.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
      histStack.Add(histWeightedMc.GetValue(), "HIST E")
      histStack.Add(histRealData.GetValue(),   "E")
      histStack.Draw("NOSTACK")
      histStack.SetMaximum(1.1 * max(histRealData.GetMaximum(), histWeightedMc.GetMaximum()))
      histStack.GetXaxis().SetTitle(histWeightedMc.GetXaxis().GetTitle())
      histStack.GetYaxis().SetTitle(histWeightedMc.GetYaxis().GetTitle())
      canv.BuildLegend(0.75, 0.85, 0.99, 0.99)
      nmbNonZeroBins = 0
      for binIndex in range(1, histRealData.GetNbinsX() + 1):
        # both histograms have same binning by construction
        if histRealData.GetBinContent(binIndex) > 0 or histWeightedMc.GetBinContent(binIndex) > 0:
          nmbNonZeroBins += 1
      chi2PerBin = histWeightedMc.Chi2Test(histRealData.GetValue(), "WW P CHI2") / nmbNonZeroBins
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
      maxZ = histRealData.GetMaximum()
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
      histPulls.SetTitle("(Real data#minusWeighted MC)/#sigma_{Real data}")
      canv = ROOT.TCanvas()
      # draw pull plot with different color palette and symmetric z axis
      ROOT.gStyle.SetPalette(ROOT.kLightTemperature)
      zRange = max(abs(histPulls.GetMinimum()), abs(histPulls.GetMaximum()))
      histPulls.SetMinimum(-zRange)
      histPulls.SetMaximum(zRange)
      histPulls.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{histPulls.GetName()}{pdfFileNameSuffix}.pdf")
      ROOT.gStyle.SetPalette(ROOT.kBird)  # restore previous color palette
    else:
      raise RuntimeError(f"Unsupported histogram type '{histRealData.ClassName()}'")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)

  dataPeriods   = (
    # "2017_01",
    "2018_08",
  )
  tBinLabels    = (
    "tbin_0.1_0.2",
    # "tbin_0.2_0.3",
    # "tbin_0.3_0.4",
    # "tbin_0.4_0.5",
  )
  beamPolLabels = (
    "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
  )
  massMin      = 0.28  # [GeV]
  massBinWidth = 0.04  # [GeV]
  nmbBins      = 50
  subSystem    = SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim", lvRecoilLabel = "recoil", pairTLatexLabel = "#pi#pi")

  for dataPeriod in dataPeriods:
    print(f"Generating plots for data period '{dataPeriod}':")
    for tBinLabel in tBinLabels:
      print(f"Generating plots for t bin '{tBinLabel}':")
      for beamPolLabel in beamPolLabels:
        beamPolInfo = BEAM_POL_INFOS[dataPeriod][beamPolLabel]
        print(f"Generating plots for beam-polarization orientation '{beamPolLabel}'"
              + (f": pol = {beamPolInfo.pol:.4f}, PhiLab = {beamPolInfo.PhiLab:.1f} deg" if beamPolInfo is not None else ""))
        # load data in AMPTOOLS format
        dataDirName          = f"./polarized/{dataPeriod}/{tBinLabel}"
        weightedDataDirName  = f"{dataDirName}/{subSystem.pairLabel}/weighted.maxL_4/{beamPolLabel}"
        weightedDataFileName = f"{weightedDataDirName}/phaseSpace_acc_weighted_raw.root"
        dataToOverlay = DataToOverlay(
          realData   = getDataFrameWithCorrectEventWeights(
            dataSigRegionFileNames  = (f"{dataDirName}/Alex/amptools_tree_signal_{beamPolLabel}.root", ),
            dataBkgRegionFileNames  = (f"{dataDirName}/Alex/amptools_tree_bkgnd_{beamPolLabel}.root",  ),
            treeName                = "kin",
            friendSigRegionFileName = f"{dataDirName}/data_sig_{beamPolLabel}.root.weights",
            friendBkgRegionFileName = f"{dataDirName}/data_bkg_{beamPolLabel}.root.weights",
          ),
          weightedMc = ROOT.RDataFrame(subSystem.pairLabel, weightedDataFileName),
        )
        print(f"Loaded weighted-MC data from '{weightedDataFileName}'")
        # loop over members of `DataToOverlay` and define columns needed for plotting for `realData` and `weightedMc`
        for member in fields(dataToOverlay):
          df = getattr(dataToOverlay, member.name)
          df = defineColumnsForPlots(
            df              = df,
            inputDataFormat = InputDataFormat.AMPTOOLS,
            subSystem       = subSystem,
            beamPolInfo     = BEAM_POL_INFOS[dataPeriod][beamPolLabel],
          )
          setattr(dataToOverlay, member.name, df)
        # plot overlays for full mass range and for individual mass bins
        print(f"Overlaying histograms for full mass range and writing plots into '{weightedDataDirName}'")
        makePlots(
          dataToOverlay = dataToOverlay,
          outputDirName = weightedDataDirName,
          colNameSuffix = subSystem.pairLabel,
        )
        for massBinIndex in range(nmbBins):
          massBinMin = massMin + massBinIndex * massBinWidth
          massBinMax = massBinMin + massBinWidth
          print(f"Overlaying histograms for mass bin {massBinIndex} with range [{massBinMin:.2f}, {massBinMax:.2f}] GeV")
          massRangeFilter = f"(({massBinMin} < mass{subSystem.pairLabel}) && (mass{subSystem.pairLabel} < {massBinMax}))"
          makePlots(
            dataToOverlay     = dataToOverlay.Filter(massRangeFilter),
            outputDirName     = weightedDataDirName,
            pdfFileNameSuffix = f"_{massBinMin:.2f}_{massBinMax:.2f}",
            colNameSuffix     = subSystem.pairLabel,
          )
