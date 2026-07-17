#!/usr/bin/env python3
"""
This module plots kinematic distributions from input data.

Usage: Run this module as a script to generate kinematic plots.
"""


from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import functools
import os
from typing import Union
from typing_extensions import TypeAlias
# from typing import TypeAlias  # for Python 3.10+

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from AnalysisConfig import (
  AnalysisConfig,
  BeamPolInfo,
  BEAM_POL_INFOS,
  CFG_POLARIZED_KSKL,
  defineOverwriteRDataFrame,
  SubsystemInfo,
)
from makeMomentsInputTree import (
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  CPP_CODE_TWO_BODY_ANGLES,
  CPP_CODE_TWO_BODY_ANGLES_NIZAR,
  defineDataFrameColumns,
  lorentzVectors,
)
from PlottingUtilities import (
  drawHorizontalZeroLine,
  HistAxisBinning,
  setupPlotStyle,
)
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def defineColumnsForPlots(
  df:                   ROOT.RDataFrame,
  inputDataFormat:      AnalysisConfig.DataFormat,
  subsystem:            SubsystemInfo,
  beamPolInfo:          BeamPolInfo | None,
  additionalColumnDefs: dict[str, str] = {},  # additional columns to define
  additionalFilterDefs: list[str]      = [],  # additional filter conditions to apply
) -> ROOT.RDataFrame:
  """Defines RDataFrame columns for kinematic plots"""
  lvs = lorentzVectors(dataFormat = inputDataFormat)
  lvTarget = lvs["target"]
  lvBeam   = lvs["beam"]
  lvRecoil = lvs[subsystem.lvRecoilLabel]
  lvA      = lvs[subsystem.lvALabel]
  lvB      = lvs[subsystem.lvBLabel]
  for frame in (AnalysisConfig.CoordSysType.HF, AnalysisConfig.CoordSysType.GJ):
    #TODO move to better place; maybe to AnalysisConfig or SubsystemInfo
    #!NOTE! coordinate system definitions for beam + target -> pi+ + pi- + recoil (all momenta in XRF):
    #    HF for pi+ pi- meson system:  use pi+  as analyzer and z_HF = -p_recoil and y_HF = p_recoil x p_beam
    #    HF for pi+- p  baryon system: use pi+- as analyzer and z_HF = -p_pi-+   and y_HF = p_beam   x p_pi-+
    #    GJ for pi+ pi- meson system:  use pi+  as analyzer and z_GJ = p_beam    and y_HF = p_recoil x p_beam
    #    GJ for pi+- p  baryon system: use pi+- as analyzer and z_GJ = p_target  and y_HF = p_beam   x p_pi-+
    #  particle A is the analyzer, particle B is the other particle in the pair, and the recoil is the third particle in the final state
    df = defineDataFrameColumns(
      df                   = df,
      lvTarget             = lvTarget,
      lvBeam               = lvBeam,
      lvRecoil             = lvRecoil,
      lvA                  = lvA,
      lvB                  = lvB,
      beamPolInfo          = beamPolInfo,
      frame                = frame,
      additionalColumnDefs = additionalColumnDefs,
      additionalFilterDefs = additionalFilterDefs,
      colNameSuffix        = subsystem.pairLabel,
    )
    # define additional columns for subsystem
    if beamPolInfo is not None:
      df = (
        df.Define(f"Psi{frame.name}{subsystem.pairLabel}Deg", f"(Double32_t)(fixAzimuthalAngleRange(Phi{subsystem.pairLabel} - phi{frame.name}{subsystem.pairLabel}) * TMath::RadToDeg())")
      )
  # define additional columns that are independent of subsystem
  df = defineOverwriteRDataFrame(df, f"mass{subsystem.pairLabel}Sq", f"(Double32_t)std::pow(mass{subsystem.pairLabel}, 2)")
  df = defineOverwriteRDataFrame(df, "Ebeam",                        f"(Double32_t)ROOT::Math::PxPyPzEVector({lvBeam}).E()")
  # track kinematics
  df = defineOverwriteRDataFrame(df, "momLabRecoil",      f"(Double32_t)ROOT::Math::PxPyPzEVector({lvRecoil}).P()")
  df = defineOverwriteRDataFrame(df, "momLabXRecoil",     f"(Double32_t)ROOT::Math::PxPyPzEVector({lvRecoil}).X()")
  df = defineOverwriteRDataFrame(df, "momLabYRecoil",     f"(Double32_t)ROOT::Math::PxPyPzEVector({lvRecoil}).Y()")
  df = defineOverwriteRDataFrame(df, "momLabZRecoil",     f"(Double32_t)ROOT::Math::PxPyPzEVector({lvRecoil}).Z()")
  df = defineOverwriteRDataFrame(df, "momLabA",           f"(Double32_t)ROOT::Math::PxPyPzEVector({lvA}).P()")
  df = defineOverwriteRDataFrame(df, "momLabXA",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvA}).X()")
  df = defineOverwriteRDataFrame(df, "momLabYA",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvA}).Y()")
  df = defineOverwriteRDataFrame(df, "momLabZA",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvA}).Z()")
  df = defineOverwriteRDataFrame(df, "momLabB",           f"(Double32_t)ROOT::Math::PxPyPzEVector({lvB}).P()")
  df = defineOverwriteRDataFrame(df, "momLabXB",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvB}).X()")
  df = defineOverwriteRDataFrame(df, "momLabYB",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvB}).Y()")
  df = defineOverwriteRDataFrame(df, "momLabZB",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvB}).Z()")
  df = defineOverwriteRDataFrame(df, "thetaLabRecoilDeg", f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvRecoil}).Theta() * TMath::RadToDeg())")
  df = defineOverwriteRDataFrame(df, "thetaLabADeg",      f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvA}).Theta()      * TMath::RadToDeg())")
  df = defineOverwriteRDataFrame(df, "thetaLabBDeg",      f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvB}).Theta()      * TMath::RadToDeg())")
  df = defineOverwriteRDataFrame(df, "phiLabRecoilDeg",   f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvRecoil}).phi()   * TMath::RadToDeg())")
  df = defineOverwriteRDataFrame(df, "phiLabADeg",        f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvA}).phi()        * TMath::RadToDeg())")
  df = defineOverwriteRDataFrame(df, "phiLabBDeg",        f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvB}).phi()        * TMath::RadToDeg())")
  df = defineOverwriteRDataFrame(df, "massRecoil",        f"(Double32_t)ROOT::Math::PxPyPzEVector({lvRecoil}).M()")
  df = defineOverwriteRDataFrame(df, "massA",             f"(Double32_t)ROOT::Math::PxPyPzEVector({lvA}).M()")
  df = defineOverwriteRDataFrame(df, "massB",             f"(Double32_t)ROOT::Math::PxPyPzEVector({lvB}).M()")
  # print(f"!!! {df.GetDefinedColumnNames()=}")
  return df


@dataclass
class HistogramDefinition:
  """Stores information needed to define a histogram"""
  name:        str  # name of the histogram
  title:       str  # title of the histogram
  binning:     tuple[tuple[int, float, float], ] | tuple[tuple[int, float, float], tuple[int, float, float]] | tuple[tuple[int, float, float], tuple[int, float, float], tuple[int, float, float]]
               # 1D binning: ((number of x bins, minimum x value, maximum x value), )
               # 2D binning: ((number of x bins, minimum x value, maximum x value), (number of y bins, minimum y value, maximum y value))
               # 3D binning: ((number of x bins, minimum x value, maximum x value), (number of y bins, minimum y value, maximum y value), (number of z bins, minimum z value, maximum z value))
  columnNames: tuple[str, ] | tuple[str, str] | tuple[str, str, str]  # name(s) of RDataFrame column(s) to plot
  filter:      str = ""  # optional filter condition to apply to the histogram


HistType:           TypeAlias = Union[ROOT.TH1D, ROOT.TH2D, ROOT.TH3D]
HistRResultPtrType: TypeAlias = Union[ROOT.RDF.RResultPtr[ROOT.TH1D], ROOT.RDF.RResultPtr[ROOT.TH2D], ROOT.RDF.RResultPtr[ROOT.TH3D]]
HistListType:       TypeAlias = list[Union[HistType, HistRResultPtrType]]

def bookHistogram(
  df:           ROOT.RDataFrame,
  histDef:      HistogramDefinition,
  applyWeights: bool,
) -> HistType | HistRResultPtrType:
  """Books a single histogram according to the given definition and returns it"""
  # apply optional filter
  dfHist = df.Filter(histDef.filter) if histDef.filter else df
  # get functions to book histogram of correct dimension
  histDimension = len(histDef.binning)
  if histDimension not in (1, 2, 3):
    raise NotImplementedError(f"Booking of {histDimension}D histograms is not implemented")
  dfHistoNDFunc = (
    dfHist.Histo1D if histDimension == 1 else
    dfHist.Histo2D if histDimension == 2 else
    dfHist.Histo3D
  )
  # flatten binning into single tuple
  binning = tuple(entry for binningTuple in histDef.binning for entry in binningTuple)
  # book histogram with or without event weights
  if applyWeights:
    return dfHistoNDFunc((histDef.name, histDef.title, *binning), *histDef.columnNames, "eventWeight")
  else:
    return dfHistoNDFunc((histDef.name, histDef.title, *binning), *histDef.columnNames)


def calcEvenOddValue(
  hist:          HistType,
  histEven:      HistType,
  histOdd:       HistType,
  posBinIndices: tuple[int, ...],
  negBinIndices: tuple[int, ...],
) -> None:
  """Calculates even or odd value for given positive and negative bin indices"""
  assert hist.GetDimension() == histEven.GetDimension() == histOdd.GetDimension(), "Histogram dimensions must be identical!"
  assert len(posBinIndices) == len(negBinIndices), "Dimensions of positive and negative bin indices must be the same!"
  assert len(posBinIndices) == hist.GetDimension(), "Dimensions of bin indices must match histogram dimensions!"
  phiPosVal = hist.GetBinContent(*posBinIndices)
  phiNegVal = hist.GetBinContent(*negBinIndices)
  #TODO uncertainties and correlations are not calculated and set correctly
  phiOddVal  = (phiPosVal - phiNegVal) / 2
  phiEvenVal = (phiPosVal + phiNegVal) / 2
  histOdd.SetBinContent (*posBinIndices, +phiOddVal)
  histOdd.SetBinContent (*negBinIndices, -phiOddVal)
  histEven.SetBinContent(*posBinIndices, phiEvenVal)
  histEven.SetBinContent(*negBinIndices, phiEvenVal)


def decomposeHistEvenOdd(hist: HistType) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1] | tuple[ROOT.TH2, ROOT.TH2, ROOT.TH2] | tuple[ROOT.TH3, ROOT.TH3, ROOT.TH3]:
  """Decomposes a histogram into even and odd parts based on symmetry along the phi axis, which must be the y axis for 2D or 3D histograms; returns (odd, even, odd + even)"""
  histOdd  = hist.Clone(f"{hist.GetName()}_odd")
  histEven = hist.Clone(f"{hist.GetName()}_even")
  if hist.GetDimension() == 1:
    assert hist.GetNbinsX() % 2 == 0, "Number of phi bins must be even!"
    for phiBinNegIndex in range(1, hist.GetNbinsX() // 2 + 1):  # only need to loop over negative half of phi bins
      phiBinPosIndex = hist.GetXaxis().FindBin(-hist.GetXaxis().GetBinCenter(phiBinNegIndex))
      calcEvenOddValue(hist, histEven, histOdd, (phiBinPosIndex, ), (phiBinNegIndex, ))
  elif hist.GetDimension() == 2:
    assert hist.GetNbinsY() % 2 == 0, "Number of phi bins must be even!"
    for thetaBinIndex in range(1, hist.GetNbinsX() + 1):
      for phiBinNegIndex in range(1, hist.GetNbinsY() // 2 + 1):  # only need to loop over negative half of phi bins
        phiBinPosIndex = hist.GetYaxis().FindBin(-hist.GetYaxis().GetBinCenter(phiBinNegIndex))
        calcEvenOddValue(hist, histEven, histOdd, (thetaBinIndex, phiBinPosIndex), (thetaBinIndex, phiBinNegIndex))
  elif hist.GetDimension() == 3:
    assert hist.GetNbinsY() % 2 == 0, "Number of phi bins must be even!"
    for thetaBinIndex in range(1, hist.GetNbinsX() + 1):
      for PhiBinIndex in range(1, hist.GetNbinsZ() + 1):
        for phiBinNegIndex in range(1, hist.GetNbinsY() // 2 + 1):  # only need to loop over negative half of phi bins
          phiBinPosIndex = hist.GetYaxis().FindBin(-hist.GetYaxis().GetBinCenter(phiBinNegIndex))
          calcEvenOddValue(hist, histEven, histOdd, (thetaBinIndex, phiBinPosIndex, PhiBinIndex), (thetaBinIndex, phiBinNegIndex, PhiBinIndex))
  else:
    raise NotImplementedError(f"Decomposition of {hist.GetDimension()}D histograms is not implemented")
  histSum = hist.Clone(f"{hist.GetName()}_sum")
  histSum.Add(histOdd, histEven)
  return histOdd, histEven, histSum


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
        HistogramDefinition(f"momLabYRecoilVsMomLabXRecoil{histNameSuffix}",       title + f";p_{{x}}^{{{recoilTLatex}}} [GeV];p_{{y}}^{{{recoilTLatex}}} [GeV];",             ((100, -0.5, +0.5), (100,  -0.5, +0.5)), ("momLabXRecoil",     "momLabYRecoil"     ), filter),
        HistogramDefinition(f"momLabYAVsMomLabXA{histNameSuffix}",                 title + f";p_{{x}}^{{{ATLatex}}} [GeV];p_{{y}}^{{{ATLatex}}} [GeV];",                       ((100, -0.8, +0.8), (100,  -0.8, +0.8)), ("momLabXA",          "momLabYA"          ), filter),
        HistogramDefinition(f"momLabYBVsMomLabXB{histNameSuffix}",                 title + f";p_{{x}}^{{{BTLatex}}} [GeV];p_{{y}}^{{{BTLatex}}} [GeV];",                       ((100, -0.8, +0.8), (100,  -0.8, +0.8)), ("momLabXB",          "momLabYB"          ), filter),
        HistogramDefinition(f"thetaLabRecoilDegVsMomLabRecoil{histNameSuffix}",    title + f";p_{{{recoilTLatex}}} [GeV];#theta_{{{recoilTLatex}}}^{{lab}} [deg]",             ((100,  0,    1  ), (100,  60,   80  )), ("momLabRecoil",      "thetaLabRecoilDeg" ), filter),
        HistogramDefinition(f"thetaLabADegVsMomLabA{histNameSuffix}",              title + f";p_{{{ATLatex}}} [GeV];#theta_{{{ATLatex}}}^{{lab}} [deg]",                       ((100,  0,   10  ), (100,   0,   30  )), ("momLabA",           "thetaLabADeg"      ), filter),
        HistogramDefinition(f"thetaLabBDegVsMomLabB{histNameSuffix}",              title + f";p_{{{BTLatex}}} [GeV];#theta_{{{BTLatex}}}^{{lab}} [deg]",                       ((100,  0,   10  ), (100,   0,   30  )), ("momLabB",           "thetaLabBDeg"      ), filter),
        HistogramDefinition(f"phiLabRecoilDegVsThetaLabRecoilDeg{histNameSuffix}", title + f";#theta_{{{recoilTLatex}}}^{{lab}} [deg];#phi_{{{recoilTLatex}}}^{{lab}} [deg];", ((100, 60,   80  ), (72, -180, +180  )), ("thetaLabRecoilDeg", "phiLabRecoilDeg"   ), filter),
        HistogramDefinition(f"phiLabADegVsThetaLabADeg{histNameSuffix}",           title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];#phi_{{{ATLatex}}}^{{lab}} [deg];",           ((100,  0,   30  ), (72, -180, +180  )), ("thetaLabADeg",      "phiLabADeg"        ), filter),
        HistogramDefinition(f"phiLabBDegVsThetaLabBDeg{histNameSuffix}",           title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];#phi_{{{BTLatex}}}^{{lab}} [deg];",           ((100,  0,   30  ), (72, -180, +180  )), ("thetaLabBDeg",      "phiLabBDeg"        ), filter),
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
  outputDirName: str,
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
  canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")


def makePlots(
  hists:            HistListType,
  histNamesEvenOdd: list[str],
  outputDirName:    str,
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
  os.makedirs(outputDirName, exist_ok = True)
  outRootFileName = f"{outputDirName}/plots.root"
  with ROOT.TFile.Open(outRootFileName, "RECREATE"):
    print(f"Writing histograms to '{outRootFileName}'")
    for hist in hists:
      if   (isinstance(hist, (ROOT.TH1D, ROOT.TH2D, ROOT.TH3D))                                                                and hist            in histsOdd) \
        or (isinstance(hist, (ROOT.RDF.RResultPtr[ROOT.TH1D], ROOT.RDF.RResultPtr[ROOT.TH2D], ROOT.RDF.RResultPtr[ROOT.TH3D])) and hist.GetValue() in histsOdd):
        ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # use pos/neg color palette and symmetric z axis
      makePlot(hist, outputDirName)
      hist.Write()
      ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette


def makeAnglesHFCorrelationPlot(
  df:                   ROOT.RDataFrame,
  subsystem:            SubsystemInfo,
  kinVarNameCorr:       str,  # column name to correlate with helicity-frame angles
  outputDirName:        str,  # directory to save output plot in
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
  os.makedirs(outputDirName, exist_ok = True)
  makePlot(histCorr, outputDirName)
  with ROOT.TFile.Open(f"{outputDirName}/{histCorr.GetName()}.root", "RECREATE"):
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
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES_NIZAR)

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

  # parameters for polarized K_S K_L data
  cfg = deepcopy(CFG_POLARIZED_KSKL)
  # massBinning = HistAxisBinning(nmbBins = 17, minVal  = 1.04, maxVal  = 1.72)  # generate plots in these bins
  subsystemMassBinning      = HistAxisBinning(nmbBins = 14, minVal = 1.2, maxVal = 2.6)  # 100 MeV wide bins
  additionalColumnDefs[AnalysisConfig.DataType.REAL_DATA]["eventWeight"] = "Weight"  # use this column as event weight
  # additionalColumnDefs = {"eventWeight" : "weightASBS"}  # use this column as event weights
  # additionalFilterDefs = ["(0.60 < massPiPi and massPiPi < 0.88)", "(0.100 < minusTPiPi and minusTPiPi < 0.114)"]  #kinematic range used in SDME analysis
  # useSeparateBackgroundFiles = True  # if True, signal and background regions are stored in separate input files
  useSeparateBackgroundFiles = False
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
          inputFilePaths = cfg.inputFilePaths(inputDataType, dataPeriod, tBinLabel, beamPolLabel)
          df = ROOT.RDataFrame(cfg.inputTreeName, inputFilePaths)  # real data must contains combined signal and background data with correct event weights
          dfSubsystem = defineColumnsForPlots(
            df                   = df,
            inputDataFormat      = inputDataFormat,
            subsystem            = cfg.subsystem,
            beamPolInfo          = beamPolInfo,
            additionalColumnDefs = additionalColumnDefs[inputDataType],
            additionalFilterDefs = additionalFilterDefs[inputDataType],
          ).Filter((f'if (rdfentry_ == 0) {{ std::cout << "Running event loop for subsystem {cfg.subsystem.pairLabel}" << std::endl; }} return true;'))  # no-op filter that logs when event loop is running
          outputDirName = f"{cfg.outputDataDirBasePath(dataPeriod, tBinLabel)}/plots_{inputDataType.name}/{beamPolLabel}"
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
              outputDirName = outputDirName,
            )
          # if True:
          if False:
            # make correlation plots; currently only for rho(770) -> pi+ pi- subsystem
            additionalFilterDefs = ["(0.72 < massPiPi and massPiPi < 0.76)", ]  # select mass bin at rho(770) peak
            outputDirName = f"{outputDirName}/anglesHFCorrelations"
            print(f"Writing helicity-frame angles correlation plots to '{outputDirName}'")
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
                outputDirName        = outputDirName,
                additionalFilterDefs = additionalFilterDefs,
              )

  timer.stop("Total execution time")
  print(timer.summary)
