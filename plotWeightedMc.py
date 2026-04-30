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
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TWO_BODY_ANGLES,
  getDataFrameWithCorrectEventWeights,
  InputDataFormat,
  SubSystemInfo,
)
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
  subSystem:             SubSystemInfo,
  outputDirName:         str = ".",
  pdfFileNameSuffix:     str = "",
  yAxisLabel:            str = "RF-Sideband Subtracted Combos",
  weightedMcScaleFactor: float | None = None,  # if float, all weighted-MC histograms are scaled by this factor; if None, each weighted-MC histogram is scaled to integral of corresponding real-data histogram
  nmbBinsAzim:           int = 72,   # number of bins for angular variables
  nmbBinsOther:          int = 100,  # number of bins for other variables
  massBinning:           tuple[int, float, float] = (50, 0.28, 2.28),  # binning for mass variable in histograms of distributions in X rest frame
) -> None:
  """Overlays 1D distributions and compares 2D distributions from real data and weighted Monte Carlo"""
  histsToOverlay = HistsToOverlay()
  # loop over members of `DataToOverlay` and book histograms for `realData` and `weightedMc`
  pairLabel    = subSystem.pairLabel
  pairTLatex   = subSystem.pairTLatexLabel
  ATLatex      = subSystem.ATLatexLabel
  BTLatex      = subSystem.BTLatexLabel
  recoilTLatex = subSystem.recoilTLatexLabel
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
        HistogramDefinition(f"cosThetaHF{pairLabel}VsMass{histNameSuffix}",      f";m_{{{pairTLatex}}} [GeV];cos#theta_{{HF}}",  (massBinning,                     (nmbBinsOther // 2,   -1,   +1)), (f"mass{pairLabel}",       f"cosThetaHF{pairLabel}")),
        HistogramDefinition(f"phiHF{pairLabel}DegVsMass{histNameSuffix}",        f";m_{{{pairTLatex}}} [GeV];#phi_{{HF}} [deg]", (massBinning,                     (nmbBinsAzim  // 2, -180, +180)), (f"mass{pairLabel}",       f"phiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"cosThetaGJ{pairLabel}VsMass{histNameSuffix}",      f";m_{{{pairTLatex}}} [GeV];cos#theta_{{GJ}}",  (massBinning,                     (nmbBinsOther // 2,   -1,   +1)), (f"mass{pairLabel}",       f"cosThetaGJ{pairLabel}")),
        HistogramDefinition(f"phiGJ{pairLabel}DegVsMass{histNameSuffix}",        f";m_{{{pairTLatex}}} [GeV];#phi_{{GJ}} [deg]", (massBinning,                     (nmbBinsAzim  // 2, -180, +180)), (f"mass{pairLabel}",       f"phiGJ{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsMass{histNameSuffix}",          f";m_{{{pairTLatex}}} [GeV];#Phi [deg]",        (massBinning,                     (nmbBinsAzim  // 2, -180, +180)), (f"mass{pairLabel}",       f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"anglesHF{histNameSuffix}",                         ";cos#theta_{HF};#phi_{HF} [deg]",              ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaHF{histNameSuffix}",    ";cos#theta_{HF};#Phi [deg]",                   ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaHF{pairLabel}", f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiHF{pairLabel}Deg_{label}",   ";#phi_{HF} [deg];#Phi [deg]",                  ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiHF{pairLabel}Deg",   f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"PsiHF{pairLabel}DegVsCosThetaHF{histNameSuffix}",  ";cos#theta_{HF};#Psi [deg]",                   ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaHF{pairLabel}", f"PsiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"PsiHF{pairLabel}DegVsPhiHF{pairLabel}Deg_{label}", ";#phi_{HF} [deg];#Psi [deg]",                  ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiHF{pairLabel}Deg",   f"PsiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"anglesGJ{histNameSuffix}",                         ";cos#theta_{GJ};#phi_{GJ} [deg]",              ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiGJ{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaGJ{histNameSuffix}",    ";cos#theta_{GJ};#Phi [deg]",                   ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaGJ{pairLabel}", f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiGJ{pairLabel}Deg_{label}",   ";#phi_{GJ} [deg];#Phi [deg]",                  ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiGJ{pairLabel}Deg",   f"Phi{pairLabel}Deg"    )),
        HistogramDefinition(f"PsiGJ{pairLabel}DegVsCosThetaGJ{histNameSuffix}",  ";cos#theta_{GJ};#Psi [deg]",                   ((nmbBinsOther // 2,   -1,   +1), (nmbBinsAzim  // 2, -180, +180)), (f"cosThetaGJ{pairLabel}", f"PsiGJ{pairLabel}Deg"  )),
        HistogramDefinition(f"PsiGJ{pairLabel}DegVsPhiGJ{pairLabel}Deg_{label}", ";#phi_{GJ} [deg];#Psi [deg]",                  ((nmbBinsAzim  // 2, -180, +180), (nmbBinsAzim  // 2, -180, +180)), (f"phiGJ{pairLabel}Deg",   f"PsiGJ{pairLabel}Deg"  )),
      ]
    if True:
    # if False:
      # distributions in lab frame
      for filter, title, histNameSuffix in [
        ("",                           "",              f"_{label}"                       ),  # all data
        # (f"(phiHF{pairLabel}Deg > 0)", "#phi_{HF} > 0", f"_phiHF{pairLabel}DegPos_{label}"),
        # (f"(phiHF{pairLabel}Deg < 0)", "#phi_{HF} < 0", f"_phiHF{pairLabel}DegNeg_{label}"),
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
          # HistogramDefinition(f"momLabXA{histNameSuffix}",          title + f";p_{{x}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -0.8, +0.8), ), ("momLabXA",          ), filter),
          # HistogramDefinition(f"momLabYA{histNameSuffix}",          title + f";p_{{y}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -0.8, +0.8), ), ("momLabYA",          ), filter),
          HistogramDefinition(f"momLabXA{histNameSuffix}",          title + f";p_{{x}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1.5, +1.5), ), ("momLabXA",          ), filter),
          HistogramDefinition(f"momLabYA{histNameSuffix}",          title + f";p_{{y}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1.5, +1.5), ), ("momLabYA",          ), filter),
          HistogramDefinition(f"momLabZA{histNameSuffix}",          title + f";p_{{z}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1,   +9  ), ), ("momLabZA",          ), filter),
          HistogramDefinition(f"momLabB{histNameSuffix}",           title + f";p_{{{BTLatex}}} [GeV];"              + yAxisLabel, ((nmbBinsOther,     0,   10  ), ), ("momLabB",           ), filter),
          # HistogramDefinition(f"momLabXB{histNameSuffix}",          title + f";p_{{x}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -0.8, +0.8), ), ("momLabXB",          ), filter),
          # HistogramDefinition(f"momLabYB{histNameSuffix}",          title + f";p_{{y}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -0.8, +0.8), ), ("momLabYB",          ), filter),
          HistogramDefinition(f"momLabXB{histNameSuffix}",          title + f";p_{{x}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1.5, +1.5), ), ("momLabXB",          ), filter),
          HistogramDefinition(f"momLabYB{histNameSuffix}",          title + f";p_{{y}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((nmbBinsOther,    -1.5, +1.5), ), ("momLabYB",          ), filter),
          # HistogramDefinition(f"thetaLabRecoilDeg{histNameSuffix}", title + ";#theta_{p}^{lab} [deg];"              + yAxisLabel, ((nmbBinsOther,     0,   80  ), ), ("thetaLabRecoilDeg", ), filter),
          # HistogramDefinition(f"thetaLabADeg{histNameSuffix}",      title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];" + yAxisLabel, ((nmbBinsOther,     0,   80  ), ), ("thetaLabADeg",      ), filter),
          # HistogramDefinition(f"thetaLabBDeg{histNameSuffix}",      title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];" + yAxisLabel, ((nmbBinsOther,     0,   80  ), ), ("thetaLabBDeg",      ), filter),
          HistogramDefinition(f"thetaLabRecoilDeg{histNameSuffix}", title + ";#theta_{p}^{lab} [deg];"              + yAxisLabel, ((nmbBinsOther,    45,   65  ), ), ("thetaLabRecoilDeg", ), filter),
          HistogramDefinition(f"thetaLabADeg{histNameSuffix}",      title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];" + yAxisLabel, ((nmbBinsOther,     0,   40  ), ), ("thetaLabADeg",      ), filter),
          HistogramDefinition(f"thetaLabBDeg{histNameSuffix}",      title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];" + yAxisLabel, ((nmbBinsOther,     0,   40  ), ), ("thetaLabBDeg",      ), filter),
          # HistogramDefinition(f"phiLabRecoilDeg{histNameSuffix}",   title + ";#phi_{p}^{lab} [deg];"                + yAxisLabel, ((nmbBinsAzim, -180, +180  ), ), ("phiLabRecoilDeg",   ), filter),
          # HistogramDefinition(f"phiLabADeg{histNameSuffix}",        title + f";#phi_{{{ATLatex}}}^{{lab}} [deg];"   + yAxisLabel, ((nmbBinsAzim, -180, +180  ), ), ("phiLabADeg",        ), filter),
          # HistogramDefinition(f"phiLabBDeg{histNameSuffix}",        title + f";#phi_{{{BTLatex}}}^{{lab}} [deg];"   + yAxisLabel, ((nmbBinsAzim, -180, +180  ), ), ("phiLabBDeg",        ), filter),
          # 2D histograms
          # HistogramDefinition(f"momLabYRecoilVsMomLabXRecoil{histNameSuffix}",       title + ";p_{x}^{p} [GeV];p_{y}^{p} [GeV];",                                      ((nmbBinsOther // 2, -0.5, +0.5), (nmbBinsOther // 2,    -0.5, +0.5)), ("momLabXRecoil",     "momLabYRecoil"     ), filter),
          # HistogramDefinition(f"momLabYAVsMomLabXA{histNameSuffix}",                 title + f";p_{{x}}^{{{ATLatex}}} [GeV];p_{{y}}^{{{ATLatex}}} [GeV];",             ((nmbBinsOther // 2, -0.8, +0.8), (nmbBinsOther // 2,    -0.8, +0.8)), ("momLabXA",          "momLabYA"          ), filter),
          # HistogramDefinition(f"momLabYBVsMomLabXB{histNameSuffix}",                 title + f";p_{{x}}^{{{BTLatex}}} [GeV];p_{{y}}^{{{BTLatex}}} [GeV];",             ((nmbBinsOther // 2, -0.8, +0.8), (nmbBinsOther // 2,    -0.8, +0.8)), ("momLabXB",          "momLabYB"          ), filter),
          HistogramDefinition(f"momLabYRecoilVsMomLabXRecoil{histNameSuffix}",       title + ";p_{x}^{p} [GeV];p_{y}^{p} [GeV];",                                      ((nmbBinsOther // 2, -1,   +1  ), (nmbBinsOther // 2,    -1,   +1  )), ("momLabXRecoil",     "momLabYRecoil"     ), filter),
          HistogramDefinition(f"momLabYAVsMomLabXA{histNameSuffix}",                 title + f";p_{{x}}^{{{ATLatex}}} [GeV];p_{{y}}^{{{ATLatex}}} [GeV];",             ((nmbBinsOther // 2, -1.5, +1.5), (nmbBinsOther // 2,    -1.5, +1.5)), ("momLabXA",          "momLabYA"          ), filter),
          HistogramDefinition(f"momLabYBVsMomLabXB{histNameSuffix}",                 title + f";p_{{x}}^{{{BTLatex}}} [GeV];p_{{y}}^{{{BTLatex}}} [GeV];",             ((nmbBinsOther // 2, -1.5, +1.5), (nmbBinsOther // 2,    -1.5, +1.5)), ("momLabXB",          "momLabYB"          ), filter),
          # HistogramDefinition(f"thetaLabRecoilDegVsMomLabRecoil{histNameSuffix}",    title + ";p_{p} [GeV];#theta_{p}^{lab} [deg]",                                    ((nmbBinsOther // 2,  0,    1  ), (nmbBinsOther // 2,    60,   80  )), ("momLabRecoil",      "thetaLabRecoilDeg" ), filter),
          # HistogramDefinition(f"thetaLabADegVsMomLabA{histNameSuffix}",              title + f";p_{{{ATLatex}}} [GeV];#theta_{{{ATLatex}}}^{{lab}} [deg]",             ((nmbBinsOther // 2,  0,   10  ), (nmbBinsOther // 2,     0,   30  )), ("momLabA",           "thetaLabADeg"      ), filter),
          # HistogramDefinition(f"thetaLabBDegVsMomLabB{histNameSuffix}",              title + f";p_{{{BTLatex}}} [GeV];#theta_{{{BTLatex}}}^{{lab}} [deg]",             ((nmbBinsOther // 2,  0,   10  ), (nmbBinsOther // 2,     0,   30  )), ("momLabB",           "thetaLabBDeg"      ), filter),
          HistogramDefinition(f"thetaLabRecoilDegVsMomLabRecoil{histNameSuffix}",    title + ";p_{p} [GeV];#theta_{p}^{lab} [deg]",                                    ((nmbBinsOther // 2,  0.7,  1  ), (nmbBinsOther // 2,    45,   65  )), ("momLabRecoil",      "thetaLabRecoilDeg" ), filter),
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
  for histRealData, histWeightedMc in zip(histsToOverlay.realData, histsToOverlay.weightedMc):
    print(f"Comparing histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
    if weightedMcScaleFactor is None:
      # compute scale factor from integrals
      weightedMcIntegral = histWeightedMc.Integral()
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
      print("(under-, overflow) fractions: "
            f"'{histRealData.GetName()  }' = ({histRealData.GetBinContent(0)                            / histRealData.Integral()}, "
                                            f"{histRealData.GetBinContent(histRealData.GetNbinsX() + 1) / histRealData.Integral()})"
            f" and '{histWeightedMc.GetName()}' = ({histWeightedMc.GetBinContent(0)                              / histWeightedMc.Integral()}, "
                                                 f"{histWeightedMc.GetBinContent(histWeightedMc.GetNbinsX() + 1) / histWeightedMc.Integral()})")
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
      canv.SaveAs(f"{outputDirName}/{histStack.GetName()}{pdfFileNameSuffix}.pdf")
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
      canv.SaveAs(f"{outputDirName}/{histRealData.GetName()}{pdfFileNameSuffix}.pdf")
      print(f"Plotting weighted-MC 2D histogram '{histWeightedMc.GetName()}'")
      canv = ROOT.TCanvas()
      histWeightedMc.SetMaximum(maxZ)
      histWeightedMc.Draw("COLZ")
      adjustStatsBox(canv)
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
      # draw pull plot with pos/neg color palette and symmetric z axis
      ROOT.gStyle.SetPalette(ROOT.kLightTemperature)
      zRange = max(abs(histPulls.GetMinimum()), abs(histPulls.GetMaximum()))
      histPulls.SetMinimum(-zRange)
      histPulls.SetMaximum(+zRange)
      histPulls.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{histPulls.GetName()}{pdfFileNameSuffix}.pdf")
      ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette
    else:
      raise RuntimeError(f"Unsupported histogram type '{histRealData.ClassName()}'")


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  timer.start("Total execution time")
  ROOT.gROOT.SetBatch(True)
  ROOT.EnableImplicitMT()
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("./rootlogon.C") == 0, "Error loading './rootlogon.C'"
  setupPlotStyle()

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)

  # dataPeriods = (
  #   # "2017_01",
  #   "2018_08",
  # )
  # tBinLabels = (
  #   "tbin_0.1_0.2",
  #   # "tbin_0.2_0.3",
  #   # "tbin_0.3_0.4",
  #   # "tbin_0.4_0.5",
  # )
  # beamPolLabels = (
  #   "PARA_0",
  #   # "PARA_135",
  #   # "PERP_45",
  #   # "PERP_90",
  # )
  # maxL              = 4
  # # maxL              = 8
  # massMin           = 0.28  # [GeV]
  # massBinWidth      = 0.04  # [GeV]
  # nmbBins           = 50
  # subSystem         = SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim", lvRecoilLabel = "recoil", pairTLatexLabel = "#pi#pi")
  # useIntensityTerms = "allTerms"
  # # useIntensityTerms = "parityConserving"
  # # useIntensityTerms = "parityViolating"

  subSystem = SubSystemInfo(
    pairLabel     = "EtaPi0", pairTLatexLabel   = "#eta#pi^{0}",
    lvALabel      = "eta",    ATLatexLabel      = "#eta",
    lvBLabel      = "pi0",    BTLatexLabel      = "#pi^{0}",
    lvRecoilLabel = "recoil", recoilTLatexLabel = "p",
  )
  dataDirBasePath = "./dataPhotoProdEtaPi0/polarized"
  inputDataDirName = "Nizar"  # subdirectory in where data files are stored
  dataPeriods = ("merged", )
  tBinLabels = (
    # "t010020",
    # "t020032",
    # "t032050",
    "t050075",
    # "t075100",
  )
  beamPolLabels = ("All", )
  maxL = 4
  # maxL = 8
  # useIntensityTerms = "allTerms"  #TODO use MomentResult.IntensityTermsType
  useIntensityTerms = "parityConserving"
  # useIntensityTerms = "parityViolating"
  BEAM_POL_INFOS["merged"]["All"].pol    = "Pol"
  BEAM_POL_INFOS["merged"]["All"].PhiLab = "BeamAngle"
  treeName = "kin"
  useSeparateBackgroundFiles = False
  nmbBinsAzim  = 36  # number of bins for angular variables
  nmbBinsOther = 50  # number of bins for other variables
  massBinning  = HistAxisBinning(nmbBins = 17, minVal = 1.04, maxVal = 1.72)
  additionalColumnDefs = {
    "realData"   : {"eventWeight" : "weightASBS"},  # use this column as event weights for real data
    "weightedMc" : {},                              # no additional columns to define for weighted MC
  }
  additionalFilterDefs = [f"(({massBinning.minVal} < mass{subSystem.pairLabel}) && (mass{subSystem.pairLabel} < {massBinning.maxVal}))"]

  for dataPeriod in dataPeriods:
    print(f"Generating plots for data period '{dataPeriod}':")
    for tBinLabel in tBinLabels:
      print(f"Generating plots for t bin '{tBinLabel}':")
      for beamPolLabel in beamPolLabels:
        beamPolInfo = BEAM_POL_INFOS[dataPeriod][beamPolLabel]
        print(f"Generating plots for beam-polarization orientation '{beamPolLabel}': {beamPolInfo}")
        # load data in AMPTOOLS format
        dataDirPath          = f"{dataDirBasePath}/{dataPeriod}/{tBinLabel}"
        weightedDataDirPath  = f"{dataDirPath}/{subSystem.pairLabel}/weightedMc.maxL_{maxL}/{beamPolLabel}"
        weightedDataFilePath = f"{weightedDataDirPath}/phaseSpace_acc_weighted_raw_{useIntensityTerms}_reweighted.root"
        dataToOverlay = DataToOverlay(
          realData   = (
            getDataFrameWithCorrectEventWeights(
              dataSigRegionFileNames  = (f"{dataDirPath}/{inputDataDirName}/amptools_tree_signal_{beamPolLabel}.root", ),
              dataBkgRegionFileNames  = (f"{dataDirPath}/{inputDataDirName}/amptools_tree_bkgnd_{beamPolLabel}.root",  ),
              treeName                = "kin",
              friendSigRegionFileName = f"{dataDirPath}/data_sig_{beamPolLabel}.root.weights",
              friendBkgRegionFileName = f"{dataDirPath}/data_bkg_{beamPolLabel}.root.weights",
            ) if useSeparateBackgroundFiles else
            ROOT.RDataFrame(treeName, f"{dataDirPath}/{inputDataDirName}/amptools_tree_data_{beamPolLabel}.root")
          ),
          weightedMc = ROOT.RDataFrame(subSystem.pairLabel, weightedDataFilePath),
        )
        print(f"Loaded weighted-MC data from '{weightedDataFilePath}'")
        # loop over members of `DataToOverlay` and define columns needed for plotting for `realData` and `weightedMc`
        for dataToOverlayField in fields(dataToOverlay):
          df = getattr(dataToOverlay, dataToOverlayField.name)  # get value of class member with name `dataToOverlayField.name`
          df = defineColumnsForPlots(
            df                   = df,
            inputDataFormat      = InputDataFormat.AMPTOOLS,
            subSystem            = subSystem,
            beamPolInfo          = BEAM_POL_INFOS[dataPeriod][beamPolLabel],
            additionalColumnDefs = additionalColumnDefs[dataToOverlayField.name],
            additionalFilterDefs = additionalFilterDefs,  # apply additional filters to both real data and weighted MC
          )
          setattr(dataToOverlay, dataToOverlayField.name, df)  # set value of class member with name `dataToOverlayField.name`
        # plot overlays for full mass range and for individual mass bins
        plotDirName = f"{weightedDataDirPath}/plots_{useIntensityTerms}"
        print(f"Overlaying histograms for full mass range and writing plots into '{plotDirName}'")
        os.makedirs(plotDirName, exist_ok = True)
        makePlots(
          dataToOverlay = dataToOverlay,
          subSystem     = subSystem,
          outputDirName = plotDirName,
          nmbBinsAzim   = nmbBinsAzim,
          nmbBinsOther  = nmbBinsOther,
          massBinning   = massBinning.astuple,
        )
        if True:
        # if False:
          for massBinIndex in range(massBinning.nmbBins):
            massBinMin = massBinning.minVal + massBinIndex * massBinning.binWidth
            massBinMax = massBinMin + massBinning.binWidth
            print(f"Overlaying histograms for mass bin {massBinIndex} with range [{massBinMin:.2f}, {massBinMax:.2f}] GeV")
            massRangeFilter = f"(({massBinMin} < mass{subSystem.pairLabel}) && (mass{subSystem.pairLabel} < {massBinMax}))"
            makePlots(
              dataToOverlay     = dataToOverlay.filter(massRangeFilter),
              subSystem         = subSystem,
              outputDirName     = plotDirName,
              pdfFileNameSuffix = f"_{massBinMin:.2f}_{massBinMax:.2f}",
              nmbBinsAzim       = nmbBinsAzim,
              nmbBinsOther      = nmbBinsOther,
              massBinning       = massBinning.astuple,
            )
