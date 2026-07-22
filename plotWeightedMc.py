#!/usr/bin/env python3


from __future__ import annotations

from copy import deepcopy
from dataclasses import (
  dataclass,
  field,
  fields,
)
import functools
import os

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from AnalysisConfig import (
  AnalysisConfig,
  BEAM_POL_INFOS,
  CFG_POLARIZED_PIPI,
  CFG_POLARIZED_ETAPI0,
  SubsystemInfo,
)
from makeKinematicPlots import (
  bookHistogram,
  defineColumnsForPlots,
  HistListType,
  HistogramDefinition,
)
from makeMomentsInputTree import (
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TWO_BODY_ANGLES,
)
from MomentCalculator import MomentResult
from PlottingUtilities import (
  HistAxisBinning,
  setupPlotStyle,
)
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class DataToOverlay:
  """Stores data frames for data and weighted MC"""
  realData:   ROOT.RDataFrame
  weightedMc: ROOT.RDataFrame

  def filter(
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


def adjustStatsBox(canv: ROOT.TCanvas) -> None:
  """Adjust stats box, if present"""
  canv.Update()
  stats = canv.GetPrimitive("stats")
  if stats is not ROOT.nullptr:
    stats.SetFillColor(ROOT.kWhite)
    stats.SetX1NDC(0.75)
    stats.SetX2NDC(0.99)
    stats.SetY1NDC(0.95)
    stats.SetY2NDC(0.99)


def makePlots(
  dataToOverlay:         DataToOverlay,
  subsystem:             SubsystemInfo,
  outputDirPath:         str = ".",
  pdfFileNameSuffix:     str = "",
  yAxisLabel:            str = "RF-Sideband Subtracted Combos",
  weightedMcScaleFactor: float | None = None,  # if float, all weighted-MC histograms are scaled by this factor; if None, each weighted-MC histogram is scaled to integral of corresponding real-data histogram
  nmbBinsAzim:           int = 72,   # number of bins for azimuthal variables
  nmbBinsOther:          int = 100,  # number of bins for other variables
  massBinning:           tuple[int, float, float] = (50, 0.28, 2.28),  # binning for mass variable in histograms of distributions in X rest frame
) -> None:
  """Overlays 1D distributions and compares 2D distributions from real data and weighted Monte Carlo"""
  histsToOverlay = HistsToOverlay()
  # loop over members of `DataToOverlay` and book histograms for `realData` and `weightedMc`
  pairLabel    = subsystem.pairLabel
  pairTLatex   = subsystem.pairTLatexLabel
  ATLatex      = subsystem.ATLatexLabel
  BTLatex      = subsystem.BTLatexLabel
  recoilTLatex = subsystem.recoilTLatexLabel
  for dataToOverlayField in fields(dataToOverlay):
    # define histograms
    label = dataToOverlayField.name
    label = label[0].upper() + label[1:]  # make sure first character is upper case
    histDefs: list[HistogramDefinition] = []
    if True:
    # if False:
      # distributions in X rest frame
      histNameSuffix = f"{pairLabel}_{label}"  # i.e. "_RealData" or "_WeightedMc"
      histDefs += [
        # 1D histograms
        HistogramDefinition(f"mass{histNameSuffix}",        f";m_{{{pairTLatex}}} [GeV];"            + yAxisLabel, (massBinning,                     ), (f"mass{pairLabel}",       )),
        HistogramDefinition(f"minusT{histNameSuffix}",      f";#minus t_{{{pairTLatex}}} [GeV^{2}];" + yAxisLabel, ((nmbBinsOther,   0,       1   ), ), (f"minusT{pairLabel}",     )),
        HistogramDefinition(f"cosThetaHF{histNameSuffix}",  ";cos#theta_{HF};"                       + yAxisLabel, ((nmbBinsOther,  -1,      +1   ), ), (f"cosThetaHF{pairLabel}", )),
        HistogramDefinition(f"phiHF{pairLabel}Deg_{label}", ";#phi_{HF} [deg];"                      + yAxisLabel, ((nmbBinsAzim, -180,    +180   ), ), (f"phiHF{pairLabel}Deg",   )),
        HistogramDefinition(f"cosThetaGJ{histNameSuffix}",  ";cos#theta_{GJ};"                       + yAxisLabel, ((nmbBinsOther,  -1,      +1   ), ), (f"cosThetaGJ{pairLabel}", )),
        HistogramDefinition(f"phiGJ{pairLabel}Deg_{label}", ";#phi_{GJ} [deg];"                      + yAxisLabel, ((nmbBinsAzim, -180,    +180   ), ), (f"phiGJ{pairLabel}Deg",   )),
        HistogramDefinition(f"Phi{pairLabel}Deg_{label}",   ";#Phi [deg];"                           + yAxisLabel, ((nmbBinsAzim, -180,    +180   ), ), (f"Phi{pairLabel}Deg",     )),
        HistogramDefinition(f"PsiHF{pairLabel}Deg_{label}", ";#Psi = (#Phi#minus#phi_{HF}) [deg];"   + yAxisLabel, ((nmbBinsAzim, -180,    +180   ), ), (f"PsiHF{pairLabel}Deg",   )),
        HistogramDefinition(f"PsiGJ{pairLabel}Deg_{label}", ";#Psi = (#Phi#minus#phi_{GJ}) [deg];"   + yAxisLabel, ((nmbBinsAzim, -180,    +180   ), ), (f"PsiGJ{pairLabel}Deg",   )),
        # 2D histograms
        HistogramDefinition(f"cosThetaHF{pairLabel}VsMass{histNameSuffix}",      f";m_{{{pairTLatex}}} [GeV];cos#theta_{{HF}}",          (massBinning,                     (nmbBinsOther // 2,   -1,   +1)), (f"mass{pairLabel}",       f"cosThetaHF{pairLabel}")),
        HistogramDefinition(f"phiHF{pairLabel}DegVsMass{histNameSuffix}",        f";m_{{{pairTLatex}}} [GeV];#phi_{{HF}} [deg]",         (massBinning,                     (nmbBinsAzim  // 2, -180, +180)), (f"mass{pairLabel}",       f"phiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"cosThetaGJ{pairLabel}VsMass{histNameSuffix}",      f";m_{{{pairTLatex}}} [GeV];cos#theta_{{GJ}}",          (massBinning,                     (nmbBinsOther // 2,   -1,   +1)), (f"mass{pairLabel}",       f"cosThetaGJ{pairLabel}")),
        HistogramDefinition(f"phiGJ{pairLabel}DegVsMass{histNameSuffix}",        f";m_{{{pairTLatex}}} [GeV];#phi_{{GJ}} [deg]",         (massBinning,                     (nmbBinsAzim  // 2, -180, +180)), (f"mass{pairLabel}",       f"phiGJ{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsMass{histNameSuffix}",          f";m_{{{pairTLatex}}} [GeV];#Phi [deg]",                (massBinning,                     (nmbBinsAzim  // 2, -180, +180)), (f"mass{pairLabel}",       f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"anglesHF{histNameSuffix}",                         ";cos#theta_{HF};#phi_{HF} [deg]",                      ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaHF{histNameSuffix}",    ";cos#theta_{HF};#Phi [deg]",                           ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaHF{pairLabel}", f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiHF{pairLabel}Deg_{label}",   ";#phi_{HF} [deg];#Phi [deg]",                          ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiHF{pairLabel}Deg",   f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"PsiHF{pairLabel}DegVsCosThetaHF{histNameSuffix}",  ";cos#theta_{HF};#Psi [deg]",                           ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaHF{pairLabel}", f"PsiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"PsiHF{pairLabel}DegVsPhiHF{pairLabel}Deg_{label}", ";#phi_{HF} [deg];#Psi [deg]",                          ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiHF{pairLabel}Deg",   f"PsiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"anglesGJ{histNameSuffix}",                         ";cos#theta_{GJ};#phi_{GJ} [deg]",                      ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiGJ{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaGJ{histNameSuffix}",    ";cos#theta_{GJ};#Phi [deg]",                           ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaGJ{pairLabel}", f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiGJ{pairLabel}Deg_{label}",   ";#phi_{GJ} [deg];#Phi [deg]",                          ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiGJ{pairLabel}Deg",   f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"PsiGJ{pairLabel}DegVsCosThetaGJ{histNameSuffix}",  ";cos#theta_{GJ};#Psi [deg]",                           ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaGJ{pairLabel}", f"PsiGJ{pairLabel}Deg"  )),
        HistogramDefinition(f"PsiGJ{pairLabel}DegVsPhiGJ{pairLabel}Deg_{label}", ";#phi_{GJ} [deg];#Psi [deg]",                          ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiGJ{pairLabel}Deg",   f"PsiGJ{pairLabel}Deg"  )),
        HistogramDefinition(f"phiHF{pairLabel}DegVsPhiLabADeg_{label}",          f";#phi_{{{ATLatex}}}^{{lab}} [deg];#phi_{{HF}} [deg]", ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiLabADeg",            f"phiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"phiHF{pairLabel}DegVsPhiLabBDeg_{label}",          f";#phi_{{{BTLatex}}}^{{lab}} [deg];#phi_{{HF}} [deg]", ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiLabBDeg",            f"phiHF{pairLabel}Deg"  )),
      ]
    if True:
    # if False:
      # distributions in lab frame
      for filter, title, histNameSuffix in [
        ("",                           "",              f"_{label}"                       ),  # all data
        (f"(phiHF{pairLabel}Deg > 0)", "#phi_{HF} > 0", f"_phiHF{pairLabel}DegPos_{label}"),
        (f"(phiHF{pairLabel}Deg < 0)", "#phi_{HF} < 0", f"_phiHF{pairLabel}DegNeg_{label}"),
      ]:
        histDefs += [
          # 1D histograms
          HistogramDefinition(f"Ebeam{histNameSuffix}",             title + ";E_{beam} [GeV];"                      + yAxisLabel, ((nmbBinsOther,     8,    9  ), ), ("Ebeam",             ), filter),
          HistogramDefinition(f"momLabRecoil{histNameSuffix}",      title + ";p_{p} [GeV];"                         + yAxisLabel, ((nmbBinsOther,     0,    1  ), ), ("momLabRecoil",      ), filter),
          # HistogramDefinition(f"momLabXRecoil{histNameSuffix}",     title + ";p_{x}^{p} [GeV];"                     + yAxisLabel, ((nmbBinsOther,    -0.5, +0.5), ), ("momLabXRecoil",     ), filter),
          # HistogramDefinition(f"momLabYRecoil{histNameSuffix}",     title + ";p_{y}^{p} [GeV];"                     + yAxisLabel, ((nmbBinsOther,    -0.5, +0.5), ), ("momLabYRecoil",     ), filter),
          # HistogramDefinition(f"momLabZRecoil{histNameSuffix}",     title + ";p_{z}^{p} [GeV];"                     + yAxisLabel, ((nmbBinsOther,     0,    0.5), ), ("momLabZRecoil",     ), filter),
          HistogramDefinition(f"momLabXRecoil{histNameSuffix}",     title + ";p_{x}^{p} [GeV];"                     + yAxisLabel, ((nmbBinsOther,    -1,   +1  ), ), ("momLabXRecoil",     ), filter),
          HistogramDefinition(f"momLabYRecoil{histNameSuffix}",     title + ";p_{y}^{p} [GeV];"                     + yAxisLabel, ((nmbBinsOther,    -1,   +1  ), ), ("momLabYRecoil",     ), filter),
          HistogramDefinition(f"momLabZRecoil{histNameSuffix}",     title + ";p_{z}^{p} [GeV];"                     + yAxisLabel, ((nmbBinsOther,     0,    1  ), ), ("momLabZRecoil",     ), filter),
          HistogramDefinition(f"momLabA{histNameSuffix}",           title + f";p_{{{ATLatex}}} [GeV];"              + yAxisLabel, ((nmbBinsOther,     0,   10  ), ), ("momLabA",           ), filter),
          HistogramDefinition(f"momLabXA{histNameSuffix}",          title + f";p_{{x}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1  , +1  ), ), ("momLabXA",          ), filter),
          HistogramDefinition(f"momLabYA{histNameSuffix}",          title + f";p_{{y}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1  , +1  ), ), ("momLabYA",          ), filter),
          HistogramDefinition(f"momLabZA{histNameSuffix}",          title + f";p_{{z}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1,   +9  ), ), ("momLabZA",          ), filter),
          HistogramDefinition(f"momLabB{histNameSuffix}",           title + f";p_{{{BTLatex}}} [GeV];"              + yAxisLabel, ((nmbBinsOther,     0,   10  ), ), ("momLabB",           ), filter),
          HistogramDefinition(f"momLabXB{histNameSuffix}",          title + f";p_{{x}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1  , +1  ), ), ("momLabXB",          ), filter),
          HistogramDefinition(f"momLabYB{histNameSuffix}",          title + f";p_{{y}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1  , +1  ), ), ("momLabYB",          ), filter),
          HistogramDefinition(f"momLabZB{histNameSuffix}",          title + f";p_{{z}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1,   +9  ), ), ("momLabZB",          ), filter),
          HistogramDefinition(f"thetaLabRecoilDeg{histNameSuffix}", title + ";#theta_{p}^{lab} [deg];"              + yAxisLabel, ((nmbBinsOther,    45,   85  ), ), ("thetaLabRecoilDeg", ), filter),
          HistogramDefinition(f"thetaLabADeg{histNameSuffix}",      title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];" + yAxisLabel, ((nmbBinsOther,     0,   40  ), ), ("thetaLabADeg",      ), filter),
          HistogramDefinition(f"thetaLabBDeg{histNameSuffix}",      title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];" + yAxisLabel, ((nmbBinsOther,     0,   40  ), ), ("thetaLabBDeg",      ), filter),
          HistogramDefinition(f"phiLabRecoilDeg{histNameSuffix}",   title + ";#phi_{p}^{lab} [deg];"                + yAxisLabel, ((nmbBinsAzim,   -180, +180  ), ), ("phiLabRecoilDeg",   ), filter),
          HistogramDefinition(f"phiLabADeg{histNameSuffix}",        title + f";#phi_{{{ATLatex}}}^{{lab}} [deg];"   + yAxisLabel, ((nmbBinsAzim,   -180, +180  ), ), ("phiLabADeg",        ), filter),
          HistogramDefinition(f"phiLabBDeg{histNameSuffix}",        title + f";#phi_{{{BTLatex}}}^{{lab}} [deg];"   + yAxisLabel, ((nmbBinsAzim,   -180, +180  ), ), ("phiLabBDeg",        ), filter),
          # 2D histograms
          HistogramDefinition(f"momLabYRecoilVsMomLabXRecoil{histNameSuffix}",       title + ";p_{x}^{p} [GeV];p_{y}^{p} [GeV];",                                      ((nmbBinsOther // 2, -1,   +1  ), (nmbBinsOther // 2,    -1,   +1  )), ("momLabXRecoil",     "momLabYRecoil"     ), filter),
          HistogramDefinition(f"momLabYAVsMomLabXA{histNameSuffix}",                 title + f";p_{{x}}^{{{ATLatex}}} [GeV];p_{{y}}^{{{ATLatex}}} [GeV];",             ((nmbBinsOther // 2, -1  , +1  ), (nmbBinsOther // 2,    -1  , +1  )), ("momLabXA",          "momLabYA"          ), filter),
          HistogramDefinition(f"momLabYBVsMomLabXB{histNameSuffix}",                 title + f";p_{{x}}^{{{BTLatex}}} [GeV];p_{{y}}^{{{BTLatex}}} [GeV];",             ((nmbBinsOther // 2, -1  , +1  ), (nmbBinsOther // 2,    -1  , +1  )), ("momLabXB",          "momLabYB"          ), filter),
          HistogramDefinition(f"thetaLabRecoilDegVsMomLabRecoil{histNameSuffix}",    title + ";p_{p} [GeV];#theta_{p}^{lab} [deg]",                                    ((nmbBinsOther // 2,  0,    1  ), (nmbBinsOther // 2,    45,   85  )), ("momLabRecoil",      "thetaLabRecoilDeg" ), filter),
          HistogramDefinition(f"thetaLabADegVsMomLabA{histNameSuffix}",              title + f";p_{{{ATLatex}}} [GeV];#theta_{{{ATLatex}}}^{{lab}} [deg]",             ((nmbBinsOther // 2,  0,   10  ), (nmbBinsOther // 2,     0,   40  )), ("momLabA",           "thetaLabADeg"      ), filter),
          HistogramDefinition(f"thetaLabBDegVsMomLabB{histNameSuffix}",              title + f";p_{{{BTLatex}}} [GeV];#theta_{{{BTLatex}}}^{{lab}} [deg]",             ((nmbBinsOther // 2,  0,   10  ), (nmbBinsOther // 2,     0,   40  )), ("momLabB",           "thetaLabBDeg"      ), filter),
          # HistogramDefinition(f"phiLabRecoilDegVsThetaLabRecoilDeg{histNameSuffix}", title + ";#theta_{p}^{lab} [deg];#phi_{p}^{lab} [deg];",                          ((nmbBinsOther // 2, 60,   80  ), (nmbBinsAzim // 2, -180, +180  )), ("thetaLabRecoilDeg", "phiLabRecoilDeg"   ), filter),
          # HistogramDefinition(f"phiLabADegVsThetaLabADeg{histNameSuffix}",           title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];#phi_{{{ATLatex}}}^{{lab}} [deg];", ((nmbBinsOther // 2,  0,   30  ), (nmbBinsAzim // 2, -180, +180  )), ("thetaLabADeg",      "phiLabADeg"        ), filter),
          # HistogramDefinition(f"phiLabBDegVsThetaLabBDeg{histNameSuffix}",           title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];#phi_{{{BTLatex}}}^{{lab}} [deg];", ((nmbBinsOther // 2,  0,   30  ), (nmbBinsAzim // 2, -180, +180  )), ("thetaLabBDeg",      "phiLabBDeg"        ), filter),
        ]
    # book histograms
    df = getattr(dataToOverlay, dataToOverlayField.name)
    hists = []
    for histDef in histDefs:
      # hists.append(bookHistogram(df, histDef, applyWeights = (dataToOverlayField.name == "realData" and df.HasColumn("eventWeight"))))
      hists.append(bookHistogram(df, histDef, applyWeights = df.HasColumn("eventWeight")))
    setattr(histsToOverlay, dataToOverlayField.name, hists)
  outRootFilePath = f"{outputDirPath}/plots.root"
  with ROOT.TFile.Open(outRootFilePath, "RECREATE"):
    print(f"Writing histograms to '{outRootFilePath}'")
    for histRealData, histWeightedMc in zip(histsToOverlay.realData, histsToOverlay.weightedMc):
      print(f"Comparing histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
      weightedMcIntegral = histWeightedMc.Integral()
      if weightedMcScaleFactor is None:
        # compute scale factor from integrals
        if weightedMcIntegral != 0:
          weightedMcScaleFactor = histRealData.Integral() / weightedMcIntegral
        else:
          print(f"??? Warning: weighted-MC histogram '{histWeightedMc.GetName()}' has zero integral, cannot normalize to real data!")
      if weightedMcScaleFactor is not None:
        print(f"Scaling weighted-MC histogram '{histWeightedMc.GetName()}' by factor {weightedMcScaleFactor:.6f}'")
        histWeightedMc.Scale(weightedMcScaleFactor)  # scale weighted MC either by computed or given factor
      histRealData.SetMinimum  (0)
      histWeightedMc.SetMinimum(0)  # needs to be set after Scale()
      # generate plots
      ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
      if histRealData.GetDimension() == 1:
        # generate 1D overlay plots
        print(f"Overlaying 1D histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
        realDataIntegral = histRealData.Integral()
        if realDataIntegral != 0 and weightedMcIntegral != 0:
          print("(under-, overflow) fractions: "
                f"'{histRealData.GetName()  }' = ({histRealData.GetBinContent(0)                            / realDataIntegral}, "
                                                f"{histRealData.GetBinContent(histRealData.GetNbinsX() + 1) / realDataIntegral})"
                f" and '{histWeightedMc.GetName()}' = ({histWeightedMc.GetBinContent(0)                              / weightedMcIntegral}, "
                                                    f"{histWeightedMc.GetBinContent(histWeightedMc.GetNbinsX() + 1) / weightedMcIntegral})")
        ROOT.gStyle.SetOptStat(False)
        canv = ROOT.TCanvas()
        histStack = ROOT.THStack(histWeightedMc.GetName(), histWeightedMc.GetTitle())
        histRealData.SetTitle  ("Real data")
        histWeightedMc.SetTitle("Weighted MC")
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
        histStack.SetTitle(f"#it{{#chi}}^{{2}}/bin = {histWeightedMc.Chi2Test(histRealData.GetValue(), 'WW P CHI2/NDF'):.2g}")
        canv.SaveAs(f"{outputDirPath}/{histStack.GetName()}{pdfFileNameSuffix}.pdf")
        histStack.Write()
      elif histRealData.GetDimension() == 2:
        # generate 2D comparison plots
        histRealData.SetTitle  ("Real data")
        histWeightedMc.SetTitle("Weighted MC")
        print(f"Plotting real-data 2D histogram '{histRealData.GetName()}'")
        ROOT.gStyle.SetOptStat("i")
        canv = ROOT.TCanvas()
        maxZ = histRealData.GetMaximum()
        histRealData.SetMaximum(maxZ)
        histRealData.Draw("COLZ")
        adjustStatsBox(canv)
        canv.SaveAs(f"{outputDirPath}/{histRealData.GetName()}{pdfFileNameSuffix}.pdf")
        histRealData.Write()
        print(f"Plotting weighted-MC 2D histogram '{histWeightedMc.GetName()}'")
        canv = ROOT.TCanvas()
        histWeightedMc.SetMaximum(maxZ)
        histWeightedMc.Draw("COLZ")
        adjustStatsBox(canv)
        canv.SaveAs(f"{outputDirPath}/{histWeightedMc.GetName()}{pdfFileNameSuffix}.pdf")
        histWeightedMc.Write()
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
        # draw pull plot with pos/neg color palette and symmetric z axis
        ROOT.gStyle.SetPalette(ROOT.kLightTemperature)
        zRange = max(abs(histPulls.GetMinimum()), abs(histPulls.GetMaximum()))
        histPulls.SetMinimum(-zRange)
        histPulls.SetMaximum(+zRange)
        histPulls.Draw("COLZ")
        canv.SaveAs(f"{outputDirPath}/{histPulls.GetName()}{pdfFileNameSuffix}.pdf")
        histPulls.Write()
        ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette
      else:
        raise RuntimeError(f"Unsupported histogram type '{histRealData.ClassName()}'")


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  timer.start("Total execution time")
  ROOT.gROOT.SetBatch(True)
  ROOT.EnableImplicitMT()
  setupPlotStyle()

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)


  # cfg = deepcopy(CFG_POLARIZED_PIPI)
  # massBinning  = HistAxisBinning(nmbBins = 50, minVal = 0.28, maxVal = 2.28)

  cfg = deepcopy(CFG_POLARIZED_ETAPI0)
  BEAM_POL_INFOS["merged"]["All"].pol    = "Pol"
  BEAM_POL_INFOS["merged"]["All"].PhiLab = "BeamAngle"
  nmbBinsAzim  = 36  # number of bins for angular variables
  nmbBinsOther = 50  # number of bins for other variables
  massBinning  = HistAxisBinning(nmbBins = 17, minVal = 1.04, maxVal = 1.72)
  additionalColumnDefs = {
    "realData"   : {"eventWeight" : "weightASBS"},  # use this column as event weights for real data
    "weightedMc" : {},                              # no additional columns to define for weighted MC
  }
  additionalFilterDefs = [f"(({massBinning.minVal} < mass{cfg.subsystem.pairLabel}) && (mass{cfg.subsystem.pairLabel} < {massBinning.maxVal}))"]

  useIntensityTerms: MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.ALL
  # useIntensityTerms: MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.PARITY_CONSERVING
  # useIntensityTerms: MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.PARITY_VIOLATING

  print(f"Using analysis configuration:\n{cfg}")
  print(f"Generating weighted MC plots for subsystem '{cfg.subsystem}':")
  for dataPeriod in cfg.dataPeriods:
    print(f"Generating plots for data period '{dataPeriod}':")
    for tBinLabel in cfg.tBinLabels:
      print(f"Generating plots for t bin '{tBinLabel}':")
      for beamPolLabel in cfg.beamPolLabels:
        beamPolInfo = BEAM_POL_INFOS[dataPeriod][beamPolLabel]
        print(f"Generating plots for beam-polarization orientation '{beamPolLabel}': {beamPolInfo}")
        inputFilePath = cfg.inputFilePath(AnalysisConfig.DataType.REAL_DATA, dataPeriod, tBinLabel, beamPolLabel)
        for maxL in cfg.maxLs:
          print(f"Generating plots for L_max = {maxL}:")
          #TODO move these paths to `AnalysisConfig`?
          weightedDataDirPath  = f"{cfg.convertedDataDirBasePath(dataPeriod, tBinLabel)}/weightedMc.maxL_{maxL}/{beamPolLabel}"
          weightedDataFilePath = f"{weightedDataDirPath}/phaseSpace_acc_weighted_input_{useIntensityTerms.value}_reweighted.root"
          print(f"Loading input data of type '{AnalysisConfig.DataType.REAL_DATA}' from '{inputFilePath}'")
          print(f"Loading weighted-MC data from '{weightedDataFilePath}'")
          dataToOverlay = DataToOverlay(
            realData   = ROOT.RDataFrame(cfg.inputTreeName, inputFilePath),
            weightedMc = ROOT.RDataFrame(cfg.subsystem.pairLabel, weightedDataFilePath),
          )
          # loop over members of `DataToOverlay` and define columns needed for plotting for `realData` and `weightedMc`
          for dataToOverlayField in fields(dataToOverlay):
            df = getattr(dataToOverlay, dataToOverlayField.name)  # get value of class member with name `dataToOverlayField.name`
            df = defineColumnsForPlots(
              df                   = df,
              inputDataFormat      = AnalysisConfig.DataFormat.AMPTOOLS,
              subsystem            = cfg.subsystem,
              beamPolInfo          = BEAM_POL_INFOS[dataPeriod][beamPolLabel],
              additionalColumnDefs = additionalColumnDefs[dataToOverlayField.name],
              additionalFilterDefs = additionalFilterDefs,  # apply additional filters to both real data and weighted MC
            )
            setattr(dataToOverlay, dataToOverlayField.name, df)  # set value of class member with name `dataToOverlayField.name`
          # plot overlays for full mass range and for individual mass bins
          plotDirPath = f"{weightedDataDirPath}/plots_{useIntensityTerms.value}"
          print(f"Overlaying histograms for full mass range and writing plots into '{plotDirPath}'")
          os.makedirs(plotDirPath, exist_ok = True)
          makePlots(
            dataToOverlay = dataToOverlay,
            subsystem     = cfg.subsystem,
            outputDirPath = plotDirPath,
            nmbBinsAzim   = nmbBinsAzim,
            nmbBinsOther  = nmbBinsOther,
            massBinning   = massBinning.astuple,
          )
          # if True:
          if False:
            for massBinIndex in range(massBinning.nmbBins):
              massBinMin = massBinning.minVal + massBinIndex * massBinning.binWidth
              massBinMax = massBinMin + massBinning.binWidth
              print(f"Overlaying histograms for mass bin {massBinIndex} with range [{massBinMin:.2f}, {massBinMax:.2f}] GeV")
              massRangeFilter = f"(({massBinMin} < mass{cfg.subsystem.pairLabel}) && (mass{cfg.subsystem.pairLabel} < {massBinMax}))"
              makePlots(
                dataToOverlay     = dataToOverlay.filter(massRangeFilter),
                subsystem         = cfg.subsystem,
                outputDirPath     = plotDirPath,
                pdfFileNameSuffix = f"_{massBinMin:.2f}_{massBinMax:.2f}",
                nmbBinsAzim       = nmbBinsAzim,
                nmbBinsOther      = nmbBinsOther,
                massBinning       = massBinning.astuple,
              )
