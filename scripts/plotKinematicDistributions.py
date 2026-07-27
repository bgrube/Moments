#!/usr/bin/env python3
"""
This module plots kinematic distributions from input data.

Usage: Run this module as a script to generate kinematic plots.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import os

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from workflow.AnalysisConfig import (
  AnalysisConfig,
  BeamPolInfo,
  BEAM_POL_INFOS,
  CFG_POLARIZED_KSKL,
  CFG_POLARIZED_PIPI,
  SubsystemInfo,
)
from workflow.DataConversionUtilities import (
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  CPP_CODE_TWO_BODY_ANGLES,
  defineColumnsForPlots,
  lorentzVectors,
)
from workflow.PlottingUtilities import (
  bookHistogram,
  decomposeHistEvenOdd,
  drawHorizontalZeroLine,
  HistAxisBinning,
  HistogramDefinition,
  HistListType,
  HistRResultPtrType,
  HistType,
  setupPlotStyle,
)
from workflow import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def bookHistograms(
  df:                   ROOT.RDataFrame,
  inputDataType:        AnalysisConfig.DataType,
  subsystem:            SubsystemInfo,
  beamPolInfo:          BeamPolInfo | None,
  subsystemMassBinning: HistAxisBinning | None = None,  # if not None, histograms will be booked in bins of the subsystem mass
) -> tuple[HistListType, list[str]]:
  """Books histograms for kinematic plots and returns the list of histograms and the names of histograms to decompose into even/odd parts"""
  print(f"Booking histograms for input data type '{inputDataType}' and subsystem '{subsystem}'")
  # applyWeights = (inputDataType == AnalysisConfig.DataType.REAL_DATA and df.HasColumn("eventWeight"))
  applyWeights = df.HasColumn("eventWeight")
  if applyWeights:
    print(f"Applying event weights from column 'eventWeight'")
  else:
    print(f"Not applying event weights; 'eventWeight' column does not exist")
  yAxisLabel = "RF-Sideband Subtracted Combos" if applyWeights else "Combos"
  histNamesEvenOdd: list[str] = []
  histDefs: list[HistogramDefinition] = []
  pairLabel    = subsystem.pairLabel
  ATLatex      = subsystem.ATLatexLabel
  BTLatex      = subsystem.BTLatexLabel
  recoilTLatex = subsystem.recoilTLatexLabel

  # define histograms for lab quantities
  if True:
  # if False:
    for filter, title, histNameSuffix in [
      ("",                           "",              ""                        ),  # all data
      (f"(phiHF{pairLabel}Deg > 0)", "#phi_{HF} > 0", f"_phiHF{pairLabel}DegPos"),
      (f"(phiHF{pairLabel}Deg < 0)", "#phi_{HF} < 0", f"_phiHF{pairLabel}DegNeg"),
    ]:
      histDefs += [
        # 1D histograms
        HistogramDefinition(f"Ebeam{histNameSuffix}",             title + ";E_{beam} [GeV];"                           + yAxisLabel, ((100,   8,    9  ), ), ("Ebeam",             ), filter),
        HistogramDefinition(f"momLabRecoil{histNameSuffix}",      title + f";p_{{{recoilTLatex}}} [GeV];"              + yAxisLabel, ((100,   0,    1  ), ), ("momLabRecoil",      ), filter),
        HistogramDefinition(f"momLabXRecoil{histNameSuffix}",     title + f";p_{{x}}^{{{recoilTLatex}}} [GeV];"        + yAxisLabel, ((100,  -0.5, +0.5), ), ("momLabXRecoil",     ), filter),
        HistogramDefinition(f"momLabYRecoil{histNameSuffix}",     title + f";p_{{y}}^{{{recoilTLatex}}} [GeV];"        + yAxisLabel, ((100,  -0.5, +0.5), ), ("momLabYRecoil",     ), filter),
        HistogramDefinition(f"momLabZRecoil{histNameSuffix}",     title + f";p_{{z}}^{{{recoilTLatex}}} [GeV];"        + yAxisLabel, ((100,   0,    0.5), ), ("momLabZRecoil",     ), filter),
        HistogramDefinition(f"momLabA{histNameSuffix}",           title + f";p_{{{ATLatex}}} [GeV];"                   + yAxisLabel, ((100,   0,   10  ), ), ("momLabA",           ), filter),
        HistogramDefinition(f"momLabXA{histNameSuffix}",          title + f";p_{{x}}^{{{ATLatex}}} [GeV];"             + yAxisLabel, ((100,  -0.8, +0.8), ), ("momLabXA",          ), filter),
        HistogramDefinition(f"momLabYA{histNameSuffix}",          title + f";p_{{y}}^{{{ATLatex}}} [GeV];"             + yAxisLabel, ((100,  -0.8, +0.8), ), ("momLabYA",          ), filter),
        HistogramDefinition(f"momLabZA{histNameSuffix}",          title + f";p_{{z}}^{{{ATLatex}}} [GeV];"             + yAxisLabel, ((100,  -1,   +9  ), ), ("momLabZA",          ), filter),
        HistogramDefinition(f"momLabB{histNameSuffix}",           title + f";p_{{{BTLatex}}} [GeV];"                   + yAxisLabel, ((100,   0,   10  ), ), ("momLabB",           ), filter),
        HistogramDefinition(f"momLabXB{histNameSuffix}",          title + f";p_{{x}}^{{{BTLatex}}} [GeV];"             + yAxisLabel, ((100,  -0.8, +0.8), ), ("momLabXB",          ), filter),
        HistogramDefinition(f"momLabYB{histNameSuffix}",          title + f";p_{{y}}^{{{BTLatex}}} [GeV];"             + yAxisLabel, ((100,  -0.8, +0.8), ), ("momLabYB",          ), filter),
        HistogramDefinition(f"momLabZB{histNameSuffix}",          title + f";p_{{z}}^{{{BTLatex}}} [GeV];"             + yAxisLabel, ((100,  -1,   +9  ), ), ("momLabZB",          ), filter),
        HistogramDefinition(f"thetaLabRecoilDeg{histNameSuffix}", title + f";#theta_{{{recoilTLatex}}}^{{lab}} [deg];" + yAxisLabel, ((100,   0,   80  ), ), ("thetaLabRecoilDeg", ), filter),
        HistogramDefinition(f"thetaLabADeg{histNameSuffix}",      title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];"      + yAxisLabel, ((100,   0,   80  ), ), ("thetaLabADeg",      ), filter),
        HistogramDefinition(f"thetaLabBDeg{histNameSuffix}",      title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];"      + yAxisLabel, ((100,   0,   80  ), ), ("thetaLabBDeg",      ), filter),
        HistogramDefinition(f"phiLabRecoilDeg{histNameSuffix}",   title + f";#phi_{{{recoilTLatex}}}^{{lab}} [deg];"   + yAxisLabel, ((72, -180, +180  ), ), ("phiLabRecoilDeg",   ), filter),
        HistogramDefinition(f"phiLabADeg{histNameSuffix}",        title + f";#phi_{{{ATLatex}}}^{{lab}} [deg];"        + yAxisLabel, ((72, -180, +180  ), ), ("phiLabADeg",        ), filter),
        HistogramDefinition(f"phiLabBDeg{histNameSuffix}",        title + f";#phi_{{{BTLatex}}}^{{lab}} [deg];"        + yAxisLabel, ((72, -180, +180  ), ), ("phiLabBDeg",        ), filter),
        HistogramDefinition(f"massRecoil{histNameSuffix}",        title + f";m_{{{recoilTLatex}}} [GeV];"              + yAxisLabel, ((100,   0.8,  1.8), ), ("massRecoil",        ), filter),
        HistogramDefinition(f"massA{histNameSuffix}",             title + f";m_{{{ATLatex}}} [GeV];"                   + yAxisLabel, ((100,   0,    1  ), ), ("massA",             ), filter),
        HistogramDefinition(f"massB{histNameSuffix}",             title + f";m_{{{BTLatex}}} [GeV];"                   + yAxisLabel, ((100,   0,    1  ), ), ("massB",             ), filter),
        # 2D histograms
        HistogramDefinition(f"momLabYRecoilVsMomLabXRecoil{histNameSuffix}",       title + f";p_{{x}}^{{{recoilTLatex}}} [GeV];p_{{y}}^{{{recoilTLatex}}} [GeV];",             ((100,  -0.5, +0.5), (100,  -0.5, +0.5)), ("momLabXRecoil",     "momLabYRecoil"     ), filter),
        HistogramDefinition(f"momLabYAVsMomLabXA{histNameSuffix}",                 title + f";p_{{x}}^{{{ATLatex}}} [GeV];p_{{y}}^{{{ATLatex}}} [GeV];",                       ((100,  -0.8, +0.8), (100,  -0.8, +0.8)), ("momLabXA",          "momLabYA"          ), filter),
        HistogramDefinition(f"momLabYBVsMomLabXB{histNameSuffix}",                 title + f";p_{{x}}^{{{BTLatex}}} [GeV];p_{{y}}^{{{BTLatex}}} [GeV];",                       ((100,  -0.8, +0.8), (100,  -0.8, +0.8)), ("momLabXB",          "momLabYB"          ), filter),
        HistogramDefinition(f"thetaLabRecoilDegVsMomLabRecoil{histNameSuffix}",    title + f";p_{{{recoilTLatex}}} [GeV];#theta_{{{recoilTLatex}}}^{{lab}} [deg]",             ((100,   0,    1  ), (100,  60,   80  )), ("momLabRecoil",      "thetaLabRecoilDeg" ), filter),
        HistogramDefinition(f"thetaLabADegVsMomLabA{histNameSuffix}",              title + f";p_{{{ATLatex}}} [GeV];#theta_{{{ATLatex}}}^{{lab}} [deg]",                       ((100,   0,   10  ), (100,   0,   30  )), ("momLabA",           "thetaLabADeg"      ), filter),
        HistogramDefinition(f"thetaLabBDegVsMomLabB{histNameSuffix}",              title + f";p_{{{BTLatex}}} [GeV];#theta_{{{BTLatex}}}^{{lab}} [deg]",                       ((100,   0,   10  ), (100,   0,   30  )), ("momLabB",           "thetaLabBDeg"      ), filter),
        HistogramDefinition(f"phiLabRecoilDegVsThetaLabRecoilDeg{histNameSuffix}", title + f";#theta_{{{recoilTLatex}}}^{{lab}} [deg];#phi_{{{recoilTLatex}}}^{{lab}} [deg];", ((100,  60,   80  ), (72, -180, +180  )), ("thetaLabRecoilDeg", "phiLabRecoilDeg"   ), filter),
        HistogramDefinition(f"phiLabBDegVsphiLabADeg{histNameSuffix}",             title + f";#phi_{{{ATLatex}}}^{{lab}} [deg];#phi_{{{BTLatex}}}^{{lab}} [deg];",             ((72, -180, +180  ), (72, -180, +180  )), ("phiLabADeg",        "phiLabBDeg"        ), filter),
        HistogramDefinition(f"phiLabADegVsThetaLabADeg{histNameSuffix}",           title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];#phi_{{{ATLatex}}}^{{lab}} [deg];",           ((100,   0,   30  ), (72, -180, +180  )), ("thetaLabADeg",      "phiLabADeg"        ), filter),
        HistogramDefinition(f"phiLabBDegVsThetaLabBDeg{histNameSuffix}",           title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];#phi_{{{BTLatex}}}^{{lab}} [deg];",           ((100,   0,   30  ), (72, -180, +180  )), ("thetaLabBDeg",      "phiLabBDeg"        ), filter),
      ]

  # define histograms for angular distributions of the subsystem
  pairTLatex = subsystem.pairTLatexLabel
  # title = pairTLatex
  title = ""
  if True:
  # if False:
    histDefs += [
      # 1D histograms
      HistogramDefinition(f"cosThetaHF{pairLabel}", f"{title};cos#theta_{{HF}};"  + yAxisLabel, ((100,   -1,   +1), ), (f"cosThetaHF{pairLabel}", )),
      HistogramDefinition(f"cosThetaGJ{pairLabel}", f"{title};cos#theta_{{GJ}};"  + yAxisLabel, ((100,   -1,   +1), ), (f"cosThetaGJ{pairLabel}", )),
      HistogramDefinition(f"phiHF{pairLabel}Deg",   f"{title};#phi_{{HF}} [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"phiHF{pairLabel}Deg",   )),
      HistogramDefinition(f"phiGJ{pairLabel}Deg",   f"{title};#phi_{{GJ}} [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"phiGJ{pairLabel}Deg",   )),
      # 2D histograms
      HistogramDefinition(f"anglesHF{pairLabel}", f"{title};cos#theta_{{HF}};#phi_{{HF}} [deg]", ((50, -1, +1), (36, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiHF{pairLabel}Deg")),
      HistogramDefinition(f"anglesGJ{pairLabel}", f"{title};cos#theta_{{GJ}};#phi_{{GJ}} [deg]", ((50, -1, +1), (36, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiGJ{pairLabel}Deg")),
      HistogramDefinition(f"phiHF{pairLabel}DegVsPhiLabADeg", f"{title};#phi_{{{ATLatex}}}^{{lab}} [deg];#phi_{{HF}} [deg]", (( 72, -180, +180), (72, -180, +180)), (f"phiLabADeg", f"phiHF{pairLabel}Deg")),
      HistogramDefinition(f"phiHF{pairLabel}DegVsPhiLabBDeg", f"{title};#phi_{{{BTLatex}}}^{{lab}} [deg];#phi_{{HF}} [deg]", (( 72, -180, +180), (72, -180, +180)), (f"phiLabBDeg", f"phiHF{pairLabel}Deg")),
    ]
    if beamPolInfo is not None:
      histDefs += [
        # 1D histograms
        HistogramDefinition(f"Phi{pairLabel}Deg", f"{title};#Phi [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"Phi{pairLabel}Deg", )),
        # 2D histograms
        HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaHF{pairLabel}", f"{title};cos#theta_{{HF}};#Phi [deg]",                      ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaHF{pairLabel}", f"Phi{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaGJ{pairLabel}", f"{title};cos#theta_{{GJ}};#Phi [deg]",                      ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaGJ{pairLabel}", f"Phi{pairLabel}Deg"  )),
        HistogramDefinition(f"phiHF{pairLabel}DegVsPhi{pairLabel}Deg",   f"{title};#Phi [deg];#phi_{{HF}} [deg]",                     (( 72, -180, +180), (72, -180, +180)), (f"Phi{pairLabel}Deg",     f"phiHF{pairLabel}Deg")),
        HistogramDefinition(f"phiGJ{pairLabel}DegVsPhi{pairLabel}Deg",   f"{title};#Phi [deg];#phi_{{GJ}} [deg]",                     (( 72, -180, +180), (72, -180, +180)), (f"Phi{pairLabel}Deg",     f"phiGJ{pairLabel}Deg")),
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiLabRecoilDeg",       f"{title};#phi_{{{recoilTLatex}}}^{{lab}} [deg];#Phi [deg]", (( 72, -180, +180), (72, -180, +180)), ("phiLabRecoilDeg",        f"Phi{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiLabADeg",            f"{title};#phi_{{{ATLatex}}}^{{lab}} [deg];#Phi [deg]",      (( 72, -180, +180), (72, -180, +180)), ("phiLabADeg",             f"Phi{pairLabel}Deg"  )),
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiLabBDeg",            f"{title};#phi_{{{BTLatex}}}^{{lab}} [deg];#Phi [deg]",      (( 72, -180, +180), (72, -180, +180)), ("phiLabBDeg",             f"Phi{pairLabel}Deg"  )),
        # 3D histograms
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiHF{pairLabel}DegVsCosThetaHF{pairLabel}", f"{title};cos#theta_{{HF}};#phi_{{HF}} [deg];#Phi [deg]", ((25, -1, +1), (24, -180, +180), (24, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiHF{pairLabel}Deg", f"Phi{pairLabel}Deg")),
        HistogramDefinition(f"Phi{pairLabel}DegVsPhiGJ{pairLabel}DegVsCosThetaGJ{pairLabel}", f"{title};cos#theta_{{GJ}};#phi_{{GJ}} [deg];#Phi [deg]", ((25, -1, +1), (24, -180, +180), (24, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiGJ{pairLabel}Deg", f"Phi{pairLabel}Deg")),
      ]
      histNamesEvenOdd += [
        f"Phi{pairLabel}DegVsPhiHF{pairLabel}DegVsCosThetaHF{pairLabel}",
        f"Phi{pairLabel}DegVsPhiGJ{pairLabel}DegVsCosThetaGJ{pairLabel}",
      ]

  # define histograms for mass and angular distributions of the subsystem
  if True:
  # if False:
    histDefs += [
      # 1D histograms
      HistogramDefinition(f"mass{pairLabel}",   f";m_{{{pairTLatex}}} [GeV];"              + yAxisLabel, ((400, 0.28, 2.28), ), (f"mass{pairLabel}",   )),
      HistogramDefinition(f"minusT{pairLabel}", f";#minus t_{{{pairTLatex}}} [GeV^{{2}}];" + yAxisLabel, ((400, 0,    1),    ), (f"minusT{pairLabel}", )),
      # 2D histograms
      HistogramDefinition(f"cosThetaHF{pairLabel}VsMass{pairLabel}", f";m_{{{pairTLatex}}} [GeV];cos#theta_{{HF}}",                      ((50, 0.28, 2.28), (100,   -1,   +1)), (f"mass{pairLabel}", f"cosThetaHF{pairLabel}")),
      HistogramDefinition(f"phiHF{pairLabel}DegVsMass{pairLabel}",   f";m_{{{pairTLatex}}} [GeV];#phi_{{HF}} [deg]",                     ((50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}", f"phiHF{pairLabel}Deg"  )),
      HistogramDefinition(f"cosThetaGJ{pairLabel}VsMass{pairLabel}", f";m_{{{pairTLatex}}} [GeV];cos#theta_{{GJ}}",                      ((50, 0.28, 2.28), (100,   -1,   +1)), (f"mass{pairLabel}", f"cosThetaGJ{pairLabel}")),
      HistogramDefinition(f"phiGJ{pairLabel}DegVsMass{pairLabel}",   f";m_{{{pairTLatex}}} [GeV];#phi_{{GJ}} [deg]",                     ((50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}", f"phiGJ{pairLabel}Deg"  )),
      HistogramDefinition(f"MinusT{pairLabel}VsMass{pairLabel}",     f";m_{{{pairTLatex}}} [GeV];#minus t_{{{pairTLatex}}} [GeV^{{2}}]", ((50, 0.28, 2.28), ( 50,    0,    1)), (f"mass{pairLabel}", f"minusT{pairLabel}"    )),
    ]
    if beamPolInfo is not None:
      histDefs += [
        HistogramDefinition(f"Phi{pairLabel}DegVsMass{pairLabel}", f";m_{{{pairTLatex}}} [GeV];#Phi [deg]", ((50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}", f"Phi{pairLabel}Deg")),
      ]
  # create histograms in mass bins
  if subsystemMassBinning is not None:
    massBinWidth = (subsystemMassBinning.maxVal - subsystemMassBinning.minVal) / subsystemMassBinning.nmbBins
    for binIndex in range(0, subsystemMassBinning.nmbBins):
      massBinMin     = subsystemMassBinning.minVal + binIndex * massBinWidth
      massBinMax     = massBinMin + massBinWidth
      massBinFilter  = f"({massBinMin} < mass{pairLabel}) and (mass{pairLabel} < {massBinMax})"
      histNameSuffix = f"_{massBinMin:.2f}_{massBinMax:.2f}"
      histDefs += [
        # 1D histograms
        HistogramDefinition(f"cosThetaHF{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{HF}};"  + yAxisLabel, ((100,   -1,   +1), ), (f"cosThetaHF{pairLabel}", ), massBinFilter),
        HistogramDefinition(f"cosThetaGJ{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{GJ}};"  + yAxisLabel, ((100,   -1,   +1), ), (f"cosThetaGJ{pairLabel}", ), massBinFilter),
        HistogramDefinition(f"phiHF{pairLabel}Deg{histNameSuffix}",   f"{title};#phi_{{HF}} [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"phiHF{pairLabel}Deg",   ), massBinFilter),
        HistogramDefinition(f"phiGJ{pairLabel}Deg{histNameSuffix}",   f"{title};#phi_{{GJ}} [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"phiGJ{pairLabel}Deg",   ), massBinFilter),
        # 2D histograms
        HistogramDefinition(f"anglesHF{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{HF}};#phi_{{HF}} [deg]", ((50, -1, +1), (36, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiHF{pairLabel}Deg"), massBinFilter),
        HistogramDefinition(f"anglesGJ{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{GJ}};#phi_{{GJ}} [deg]", ((50, -1, +1), (36, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiGJ{pairLabel}Deg"), massBinFilter),
      ]
      histNamesEvenOdd += [
        f"phiHF{pairLabel}Deg{histNameSuffix}",
        f"phiGJ{pairLabel}Deg{histNameSuffix}",
        f"anglesHF{pairLabel}{histNameSuffix}",
        f"anglesGJ{pairLabel}{histNameSuffix}",
      ]
      if beamPolInfo is not None:
        histDefs += [
          # 1D histograms
          HistogramDefinition(f"Phi{pairLabel}Deg{histNameSuffix}", f"{title};#Phi [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"Phi{pairLabel}Deg", ), massBinFilter),
          # 2D histograms
          HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaHF{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{HF}};#Phi [deg]",  ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaHF{pairLabel}", f"Phi{pairLabel}Deg"  ), massBinFilter),
          HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaGJ{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{GJ}};#Phi [deg]",  ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaGJ{pairLabel}", f"Phi{pairLabel}Deg"  ), massBinFilter),
          HistogramDefinition(f"phiHF{pairLabel}DegVsPhi{pairLabel}Deg{histNameSuffix}",   f"{title};#Phi [deg];#phi_{{HF}} [deg]", (( 72, -180, +180), (72, -180, +180)), (f"Phi{pairLabel}Deg",     f"phiHF{pairLabel}Deg"), massBinFilter),
          HistogramDefinition(f"phiGJ{pairLabel}DegVsPhi{pairLabel}Deg{histNameSuffix}",   f"{title};#Phi [deg];#phi_{{GJ}} [deg]", (( 72, -180, +180), (72, -180, +180)), (f"Phi{pairLabel}Deg",     f"phiGJ{pairLabel}Deg"), massBinFilter),
        ]
        histNamesEvenOdd += [
          f"phiHF{pairLabel}DegVsPhi{pairLabel}Deg{histNameSuffix}",
          f"phiGJ{pairLabel}DegVsPhi{pairLabel}Deg{histNameSuffix}",
        ]

# book histograms
  hists = []
  for histDef in histDefs:
    hists.append(bookHistogram(df, histDef, applyWeights))
  if applyWeights:
    hists.append(bookHistogram(
      df,
      HistogramDefinition("eventWeight", ";Event weight;Combos", ((100, -1, +2), ), ("eventWeight", )),
      applyWeights = False,
    ))
  print(f"Booked {len(hists)} histograms")
  return hists, histNamesEvenOdd


def makePlot(
  hist:          HistType | HistRResultPtrType,
  outputDirPath: str,
) -> None:
  """Plots given histogram into PDF file in the given output directory"""
  print(f"Plotting histogram '{hist.GetName()}'")
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
  canv = ROOT.TCanvas()
  if hist.GetDimension() == 2 and str(hist.GetName()).startswith("mass"):
    canv.SetLogz(1)
  if hist.GetDimension() == 3:
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
  else:
    hist.Draw("COLZ")
  # adjust stats box, if present
  canv.Update()
  stats = canv.GetPrimitive("stats")
  if stats is not ROOT.nullptr:
    stats.SetFillColor(ROOT.kWhite)
    stats.SetX1NDC(0.75)
    stats.SetX2NDC(0.99)
    stats.SetY1NDC(0.95)
    stats.SetY2NDC(0.99)
  # draw zero line
  if hist.GetDimension() == 1:
    drawHorizontalZeroLine(canv)
  canv.SaveAs(f"{outputDirPath}/{hist.GetName()}.pdf")


def makePlots(
  hists:            HistListType,
  histNamesEvenOdd: list[str],
  outputDirPath:    str,
) -> None:
  """Writes histograms to ROOT file and generates PDF plots"""
  for hist in hists:
    if hist.GetMinimum() >= 0:
      hist.SetMinimum(0)
  # add phi-even and phi-odd histograms to list of histograms to plot
  histsEvenOdd = []
  histsOdd     = []  # need to keep track of odd histograms to set color palette
  for histName in histNamesEvenOdd:
    histEvenOdd = [hist for hist in hists if hist.GetName() == histName]
    assert len(histEvenOdd) == 1, f"Expected exactly one histogram with name '{histName}', but found {len(histEvenOdd)}"
    histOdd, histEven, histSum = decomposeHistEvenOdd(histEvenOdd[0])
    # if histOdd.GetDimension() == 2:
    #   for hist in (histOdd, histEven, histSum):
    #     hist.Rebin2D(4, 3)  # reduce number of bins for better visibility
    histOddValRange = max(abs(histOdd.GetMaximum()), abs(histOdd.GetMinimum()))
    histOdd.SetMaximum(+histOddValRange)
    histOdd.SetMinimum(-histOddValRange)
    histEven.SetMinimum(0)
    histSum.SetMinimum(0)
    histsEvenOdd += [histOdd, histEven, histSum]
    histsOdd.append(histOdd)
  hists += histsEvenOdd
  # plot all histograms
  os.makedirs(outputDirPath, exist_ok = True)
  outRootFilePath = f"{outputDirPath}/plots.root"
  with ROOT.TFile.Open(outRootFilePath, "RECREATE"):
    print(f"Writing histograms to '{outRootFilePath}'")
    for hist in hists:
      if   (isinstance(hist, (ROOT.TH1D, ROOT.TH2D, ROOT.TH3D))                                                                and hist            in histsOdd) \
        or (isinstance(hist, (ROOT.RDF.RResultPtr[ROOT.TH1D], ROOT.RDF.RResultPtr[ROOT.TH2D], ROOT.RDF.RResultPtr[ROOT.TH3D])) and hist.GetValue() in histsOdd):
        ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # use pos/neg color palette and symmetric z axis
      makePlot(hist, outputDirPath)
      hist.Write()
      ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette


def makeAnglesHFCorrelationPlot(
  df:                   ROOT.RDataFrame,
  subsystem:            SubsystemInfo,
  kinVarNameCorr:       str,  # column name to correlate with helicity-frame angles
  outputDirPath:        str,  # directory to save output plot in
  histNameSuffix:       str = "",
  additionalFilterDefs: list[str] = [],  # additional filter conditions to apply
) -> None:
  """Produces 2D correlation plot of helicity-frame angles with given RDataFrame column"""
  print(f"Generating correlation plot of helicity-frame angles with '{kinVarNameCorr}' for {subsystem.pairLabel} subsystem")
  if not df.HasColumn(kinVarNameCorr):
    print(f"Warning: input RDataFrame does not have column '{kinVarNameCorr}'. Cannot generate correlation plot.")
    return
  if additionalFilterDefs:
    for filterDef in additionalFilterDefs:
      print(f"Applying additional filter '{filterDef}'")
      df = df.Filter(filterDef)
  applyWeights = df.HasColumn("eventWeight")
  if applyWeights:
    print("Applying event weights")
  pairLabel = subsystem.pairLabel
  pairTLatexLabel = subsystem.pairTLatexLabel
  xColName = f"cosThetaHF{pairLabel}"
  yColName = f"phiHF{pairLabel}Deg"
  # fill 2D histogram in helicity-frame angles with average values of kinVarNameCorr
  #TODO replace this code by RDataFrame's Profile2D function
  histCorr = ROOT.TH2D(
    f"anglesHF{pairLabel}Corr_{kinVarNameCorr}{histNameSuffix}",
    f"{pairTLatexLabel};cos#theta_{{HF}};#phi_{{HF}} [deg]",
    20,   -1,   +1,
    18, -180, +180,
  )
  for xBinIndex in range(1, histCorr.GetNbinsX() + 1):
    for yBinIndex in range(1, histCorr.GetNbinsY() + 1):
      xBinRange = (histCorr.GetXaxis().GetBinLowEdge(xBinIndex), histCorr.GetXaxis().GetBinUpEdge(xBinIndex))
      yBinRange = (histCorr.GetYaxis().GetBinLowEdge(yBinIndex), histCorr.GetYaxis().GetBinUpEdge(yBinIndex))
      cellFilter = f"(({xBinRange[0]} < {xColName} and {xColName} < {xBinRange[1]}) and ({yBinRange[0]} < {yColName} and {yColName} < {yBinRange[1]}))"
      dfCell = df.Filter(cellFilter)  # select events in current 2D cell
      average = 0.0
      if dfCell.Count().GetValue() > 0:
        if applyWeights:
          # calculate weighted average
          assert not dfCell.HasColumn("weightedKinVar"), "RDataFrame already has 'weightedKinVar' column. This should not happen."
          dfCell = dfCell.Define("weightedKinVar", f"(Double32_t)({kinVarNameCorr} * eventWeight)")
          average = dfCell.Sum("weightedKinVar").GetValue() / dfCell.Sum("eventWeight").GetValue()
        else:
          average = dfCell.Sum(kinVarNameCorr).GetValue() / dfCell.Count().GetValue()
      print(f"Average value for column '{kinVarNameCorr}' in cell ({xBinIndex} = {xBinRange}, {yBinIndex} = {yBinRange}): {average}")
      histCorr.SetBinContent(xBinIndex, yBinIndex, average)
  # write plot PDF and ROOT file
  os.makedirs(outputDirPath, exist_ok = True)
  makePlot(histCorr, outputDirPath)
  with ROOT.TFile.Open(f"{outputDirPath}/{histCorr.GetName()}.root", "RECREATE"):
    histCorr.Write()


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
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)

  additionalColumnDefs = {  # additional columns for each data type
    AnalysisConfig.DataType.REAL_DATA             : {},
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : {},
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : {},
  }
  additionalFilterDefs = {  # additional filters for each data type
    AnalysisConfig.DataType.REAL_DATA             : [],
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : [],
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : [],
  }

  cfg = deepcopy(CFG_POLARIZED_PIPI)
  subsystemMassBinning = None  # do not generate plots in mass bins
  additionalFilterDefs = {  # kinematic range used in SDME analysis; for 2017_01_ver05 data
    AnalysisConfig.DataType.REAL_DATA             : ["(0.60 < massPiPi and massPiPi < 0.88)"],
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : ["(0.60 < massPiPi and massPiPi < 0.88)"],
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : ["(0.60 < massPiPi and massPiPi < 0.88)"],
  }
  # cfg = deepcopy(CFG_POLARIZED_KSKL)
  # subsystemMassBinning      = HistAxisBinning(nmbBins = 14, minVal = 1.2, maxVal = 2.6)  # 100 MeV wide bins; generate plots for these mass bins
  # additionalColumnDefs[AnalysisConfig.DataType.REAL_DATA]["eventWeight"] = "Weight"  # use this column as event weight
  # additionalColumnDefs = {"eventWeight" : "weightASBS"}  # use this column as event weights
  # BEAM_POL_INFOS["merged"]["All"].pol    = "Pol"
  # BEAM_POL_INFOS["merged"]["All"].PhiLab = "BeamAngle"

  print(f"Using analysis configuration:\n{cfg}")
  print(f"Generating plots for subsystem '{cfg.subsystem}':")
  for dataPeriod in cfg.dataPeriods:
    print(f"Generating plots for data period '{dataPeriod}':")
    for tBinLabel in cfg.tBinLabels:
      print(f"Generating plots for t bin '{tBinLabel}':")
      for beamPolLabel in cfg.beamPolLabels:  #TODO process only 1 orientation for MC data
        beamPolInfo = BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel]
        print(f"Generating plots for beam-polarization orientation '{beamPolLabel}': {beamPolInfo}")
        for inputDataType, inputDataFormat in cfg.inputDataFormats.items():
          print(f"Generating plots for input data type '{inputDataType}' in format '{inputDataFormat}'")
          inputFilePath = cfg.inputFilePath(inputDataType, dataPeriod, tBinLabel, beamPolLabel)
          print(f"Loading input data of type '{inputDataType}' from '{inputFilePath}'")
          df = ROOT.RDataFrame(cfg.inputTreeName, inputFilePath)  # real data must contains combined signal and background data with correct event weights
          dfSubsystem = defineColumnsForPlots(
            df                   = df,
            inputDataFormat      = inputDataFormat,
            subsystem            = cfg.subsystem,
            beamPolInfo          = beamPolInfo,
            additionalColumnDefs = additionalColumnDefs[inputDataType],
            additionalFilterDefs = additionalFilterDefs[inputDataType],
          ).Filter((f'if (rdfentry_ == 0) {{ std::cout << "Running event loop for subsystem {cfg.subsystem.pairLabel}" << std::endl; }} return true;'))  # no-op filter that logs when event loop is running
          outputDirPath = f"{cfg.convertedDataDirBasePath(dataPeriod, tBinLabel)}/plots_{inputDataType.name}/{beamPolLabel}"
          if True:
          # if False:
            makePlots(
              *bookHistograms(
                df                   = dfSubsystem,
                inputDataType        = inputDataType,
                subsystem            = cfg.subsystem,
                beamPolInfo          = beamPolInfo,
                subsystemMassBinning = subsystemMassBinning,
              ),
              outputDirPath = outputDirPath,
            )
          # if True:
          if False:
            # make correlation plots; currently only for rho(770) -> pi+ pi- subsystem
            additionalFilterDefs = ["(0.72 < massPiPi and massPiPi < 0.76)", ]  # select mass bin at rho(770) peak
            outputDirPath = f"{outputDirPath}/anglesHFCorrelations"
            print(f"Writing helicity-frame angles correlation plots to '{outputDirPath}'")
            lvs = lorentzVectors(dataFormat = inputDataFormat)
            dfSubsystem = dfSubsystem.Define(f"massPipP", f"(Double32_t)massPair({lvs['pip']}, {lvs['recoil']})")
            dfSubsystem = dfSubsystem.Define(f"massPimP", f"(Double32_t)massPair({lvs['pim']}, {lvs['recoil']})")
            for kinVarNameCorr in [
              # "Ebeam",
              # "momLabRecoil",
              "momLabA",
              "momLabB",
              # "thetaLabRecoilDeg",
              "thetaLabADeg",
              "thetaLabBDeg",
              "phiLabRecoilDeg",
              "phiLabADeg",
              "phiLabBDeg",
              # f"Phi{cfg.subsystem.pairLabel}Deg",
              # f"PsiHF{cfg.subsystem.pairLabel}Deg",
              # "massPipP",
              # "massPimP",
            ]:
              makeAnglesHFCorrelationPlot(
                df                   = dfSubsystem,
                subsystem            = cfg.subsystem,
                kinVarNameCorr       = kinVarNameCorr,
                outputDirPath        = outputDirPath,
                additionalFilterDefs = additionalFilterDefs,
              )

  timer.stop("Total execution time")
  print(timer.summary)
