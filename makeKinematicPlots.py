#!/usr/bin/env python3
"""
This module plots kinematic distributions from input data.

Usage: Run this module as a script to generate kinematic plots.
"""


from __future__ import annotations

from dataclasses import dataclass
import functools
import os
from typing import Union
from typing_extensions import TypeAlias
# from typing import TypeAlias  # for Python 3.10+

import ROOT

from makeMomentsInputTree import (
  BeamPolInfo,
  BEAM_POL_INFOS,
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_TWO_BODY_ANGLES,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  CoordSysType,
  defineDataFrameColumns,
  defineOverwrite,
  getDataFrameWithCorrectEventWeights,
  InputDataType,
  InputDataFormat,
  SubSystemInfo,
  lorentzVectors,
)
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def defineColumnsForPlots(
  df:                   ROOT.RDataFrame,
  inputDataFormat:      InputDataFormat,
  subSystem:            SubSystemInfo,
  beamPolInfo:          BeamPolInfo | None,
  additionalColumnDefs: dict[str, str] = {},  # additional columns to define
  additionalFilterDefs: list[str]      = [],  # additional filter conditions to apply
) -> ROOT.RDataFrame:
  """Defines RDataFrame columns for kinematic plots"""
  lvs = lorentzVectors(dataFormat = inputDataFormat)
  ALabel      = subSystem.lvALabel
  BLabel      = subSystem.lvBLabel
  recoilLabel = subSystem.lvRecoilLabel
  for frame in (CoordSysType.HF, CoordSysType.GJ):
    #!NOTE! coordinate system definitions for beam + target -> pi+ + pi- + recoil (all momenta in XRF):
    #    HF for pi+ pi- meson system:  use pi+  as analyzer and z_HF = -p_recoil and y_HF = p_recoil x p_beam
    #    HF for pi+- p  baryon system: use pi+- as analyzer and z_HF = -p_pi-+   and y_HF = p_beam   x p_pi-+
    #    GJ for pi+ pi- meson system:  use pi+  as analyzer and z_GJ = p_beam    and y_HF = p_recoil x p_beam
    #    GJ for pi+- p  baryon system: use pi+- as analyzer and z_GJ = p_target  and y_HF = p_beam   x p_pi-+
    #  particle A is the analyzer, particle B is the other particle in the pair, and the recoil is the third particle in the final state
    df = defineDataFrameColumns(
      df                   = df,
      lvTarget             = lvs["target"],
      lvBeam               = lvs["beam"],  #TODO "beam" for GJ pi+- p baryon system is p_target
      lvRecoil             = lvs[recoilLabel],
      lvA                  = lvs[ALabel],
      lvB                  = lvs[BLabel],
      beamPolInfo          = beamPolInfo,
      frame                = frame,
      additionalColumnDefs = additionalColumnDefs,
      additionalFilterDefs = additionalFilterDefs,
      colNameSuffix        = subSystem.pairLabel,
    )
    # define additional columns for subsystem
    if beamPolInfo is not None:
      df = (
        df.Define(f"Psi{frame.name}{subSystem.pairLabel}Deg", f"(Double32_t)(fixAzimuthalAngleRange(Phi{subSystem.pairLabel} - phi{frame.name}{subSystem.pairLabel}) * TMath::RadToDeg())")
      )
  # define additional columns that are independent of subsystem
  df = defineOverwrite(df, f"mass{subSystem.pairLabel}Sq", f"(Double32_t)std::pow(mass{subSystem.pairLabel}, 2)")
  df = defineOverwrite(df, "Ebeam",                        f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs['beam']}).E()")
  # track kinematics
  df = defineOverwrite(df, "momLabRecoil",      f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[recoilLabel]}).P()")
  df = defineOverwrite(df, "momLabXRecoil",     f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[recoilLabel]}).X()")
  df = defineOverwrite(df, "momLabYRecoil",     f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[recoilLabel]}).Y()")
  df = defineOverwrite(df, "momLabZRecoil",     f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[recoilLabel]}).Z()")
  df = defineOverwrite(df, "momLabA",           f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[ALabel]}).P()")
  df = defineOverwrite(df, "momLabXA",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[ALabel]}).X()")
  df = defineOverwrite(df, "momLabYA",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[ALabel]}).Y()")
  df = defineOverwrite(df, "momLabZA",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[ALabel]}).Z()")
  df = defineOverwrite(df, "momLabB",           f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[BLabel]}).P()")
  df = defineOverwrite(df, "momLabXB",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[BLabel]}).X()")
  df = defineOverwrite(df, "momLabYB",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[BLabel]}).Y()")
  df = defineOverwrite(df, "momLabZB",          f"(Double32_t)ROOT::Math::PxPyPzEVector({lvs[BLabel]}).Z()")
  df = defineOverwrite(df, "thetaLabRecoilDeg", f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvs[recoilLabel]}).Theta() * TMath::RadToDeg())")
  df = defineOverwrite(df, "thetaLabADeg",      f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvs[ALabel]}).Theta()      * TMath::RadToDeg())")
  df = defineOverwrite(df, "thetaLabBDeg",      f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvs[BLabel]}).Theta()      * TMath::RadToDeg())")
  df = defineOverwrite(df, "phiLabRecoilDeg",   f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvs[recoilLabel]}).phi()   * TMath::RadToDeg())")
  df = defineOverwrite(df, "phiLabADeg",        f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvs[ALabel]}).phi()        * TMath::RadToDeg())")
  df = defineOverwrite(df, "phiLabBDeg",        f"(Double32_t)(ROOT::Math::PxPyPzEVector({lvs[BLabel]}).phi()        * TMath::RadToDeg())")
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
  hist:          ROOT.TH1 | ROOT.TH2 | ROOT.TH3,
  histEven:      ROOT.TH1 | ROOT.TH2 | ROOT.TH3,
  histOdd:       ROOT.TH1 | ROOT.TH2 | ROOT.TH3,
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


def decomposeHistEvenOdd(
  hist: ROOT.TH1 | ROOT.TH2 | ROOT.TH3,
) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1] | tuple[ROOT.TH2, ROOT.TH2, ROOT.TH2] | tuple[ROOT.TH3, ROOT.TH3, ROOT.TH3]:
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
  df:            ROOT.RDataFrame,
  inputDataType: InputDataType,
  subSystem:     SubSystemInfo,
  beamPolInfo:   BeamPolInfo | None,
) -> tuple[HistListType, list[str]]:
  """Books histograms for kinematic plots and returns the list of histograms and the names of histograms to decompose into even/odd parts"""
  print(f"Booking histograms for input data type '{inputDataType}' and subsystem '{subSystem}'")
  applyWeights = (inputDataType == InputDataType.REAL_DATA and df.HasColumn("eventWeight"))
  yAxisLabel = "RF-Sideband Subtracted Combos" if applyWeights else "Combos"
  histNamesEvenOdd: list[str] = []
  histDefs: list[HistogramDefinition] = []
  pairLabel = subSystem.pairLabel
  # define histograms that are independent of subsystem
  ATLatex      = subSystem.ATLatexLabel
  BTLatex      = subSystem.BTLatexLabel
  recoilTLatex = subSystem.recoilTLatexLabel
  if True:
  # if False:
    if pairLabel == "PiPi" or pairLabel == "EtaPi0":  # although histograms are independent of subsystem; use mesonic subsystem to define them
      for filter, title, histNameSuffix in [
        ("",                           "",              ""                        ),  # all data
        (f"(phiHF{pairLabel}Deg > 0)", "#phi_{HF} > 0", f"_phiHF{pairLabel}DegPos"),
        (f"(phiHF{pairLabel}Deg < 0)", "#phi_{HF} < 0", f"_phiHF{pairLabel}DegNeg"),
      ]:
        histDefs += [
          # 1D histograms
          HistogramDefinition(f"Ebeam{histNameSuffix}",             title + ";E_{beam} [GeV];"                      + yAxisLabel, ((100,   8,    9  ), ), ("Ebeam",             ), filter),
          HistogramDefinition(f"momLabRecoil{histNameSuffix}",      title + ";p_{p} [GeV];"                         + yAxisLabel, ((100,   0,    1  ), ), ("momLabRecoil",      ), filter),
          HistogramDefinition(f"momLabXRecoil{histNameSuffix}",     title + ";p_{x}^{p} [GeV];"                     + yAxisLabel, ((100,  -0.5, +0.5), ), ("momLabXRecoil",     ), filter),
          HistogramDefinition(f"momLabYRecoil{histNameSuffix}",     title + ";p_{y}^{p} [GeV];"                     + yAxisLabel, ((100,  -0.5, +0.5), ), ("momLabYRecoil",     ), filter),
          HistogramDefinition(f"momLabZRecoil{histNameSuffix}",     title + ";p_{z}^{p} [GeV];"                     + yAxisLabel, ((100,   0,    0.5), ), ("momLabZRecoil",     ), filter),
          HistogramDefinition(f"momLabA{histNameSuffix}",           title + f";p_{{{ATLatex}}} [GeV];"              + yAxisLabel, ((100,   0,   10  ), ), ("momLabA",           ), filter),
          HistogramDefinition(f"momLabXA{histNameSuffix}",          title + f";p_{{x}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((100,  -0.8, +0.8), ), ("momLabXA",          ), filter),
          HistogramDefinition(f"momLabYA{histNameSuffix}",          title + f";p_{{y}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((100,  -0.8, +0.8), ), ("momLabYA",          ), filter),
          HistogramDefinition(f"momLabZA{histNameSuffix}",          title + f";p_{{z}}^{{{ATLatex}}} [GeV];"        + yAxisLabel, ((100,  -1,   +9  ), ), ("momLabZA",          ), filter),
          HistogramDefinition(f"momLabB{histNameSuffix}",           title + f";p_{{{BTLatex}}} [GeV];"              + yAxisLabel, ((100,   0,   10  ), ), ("momLabB",           ), filter),
          HistogramDefinition(f"momLabXB{histNameSuffix}",          title + f";p_{{x}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((100,  -0.8, +0.8), ), ("momLabXB",          ), filter),
          HistogramDefinition(f"momLabYB{histNameSuffix}",          title + f";p_{{y}}^{{{BTLatex}}} [GeV];"        + yAxisLabel, ((100,  -0.8, +0.8), ), ("momLabYB",          ), filter),
          HistogramDefinition(f"thetaLabRecoilDeg{histNameSuffix}", title + ";#theta_{p}^{lab} [deg];"              + yAxisLabel, ((100,   0,   80  ), ), ("thetaLabRecoilDeg", ), filter),
          HistogramDefinition(f"thetaLabADeg{histNameSuffix}",      title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];" + yAxisLabel, ((100,   0,   80  ), ), ("thetaLabADeg",      ), filter),
          HistogramDefinition(f"thetaLabBDeg{histNameSuffix}",      title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];" + yAxisLabel, ((100,   0,   80  ), ), ("thetaLabBDeg",      ), filter),
          HistogramDefinition(f"phiLabRecoilDeg{histNameSuffix}",   title + ";#phi_{p}^{lab} [deg];"                + yAxisLabel, ((72, -180, +180  ), ), ("phiLabRecoilDeg",   ), filter),
          HistogramDefinition(f"phiLabADeg{histNameSuffix}",        title + f";#phi_{{{ATLatex}}}^{{lab}} [deg];"   + yAxisLabel, ((72, -180, +180  ), ), ("phiLabADeg",        ), filter),
          HistogramDefinition(f"phiLabBDeg{histNameSuffix}",        title + f";#phi_{{{BTLatex}}}^{{lab}} [deg];"   + yAxisLabel, ((72, -180, +180  ), ), ("phiLabBDeg",        ), filter),
          # 2D histograms
          HistogramDefinition(f"momLabYRecoilVsMomLabXRecoil{histNameSuffix}",       title + ";p_{x}^{p} [GeV];p_{y}^{p} [GeV];",                                      ((100, -0.5, +0.5), (100,  -0.5, +0.5)), ("momLabXRecoil",     "momLabYRecoil"     ), filter),
          HistogramDefinition(f"momLabYAVsMomLabXA{histNameSuffix}",                 title + f";p_{{x}}^{{{ATLatex}}} [GeV];p_{{y}}^{{{ATLatex}}} [GeV];",             ((100, -0.8, +0.8), (100,  -0.8, +0.8)), ("momLabXA",          "momLabYA"          ), filter),
          HistogramDefinition(f"momLabYBVsMomLabXB{histNameSuffix}",                 title + f";p_{{x}}^{{{BTLatex}}} [GeV];p_{{y}}^{{{BTLatex}}} [GeV];",             ((100, -0.8, +0.8), (100,  -0.8, +0.8)), ("momLabXB",          "momLabYB"          ), filter),
          HistogramDefinition(f"thetaLabRecoilDegVsMomLabRecoil{histNameSuffix}",    title + ";p_{p} [GeV];#theta_{p}^{lab} [deg]",                                    ((100,  0,    1  ), (100,  60,   80  )), ("momLabRecoil",      "thetaLabRecoilDeg" ), filter),
          HistogramDefinition(f"thetaLabADegVsMomLabA{histNameSuffix}",              title + f";p_{{{ATLatex}}} [GeV];#theta_{{{ATLatex}}}^{{lab}} [deg]",             ((100,  0,   10  ), (100,   0,   30  )), ("momLabA",           "thetaLabADeg"      ), filter),
          HistogramDefinition(f"thetaLabBDegVsMomLabB{histNameSuffix}",              title + f";p_{{{BTLatex}}} [GeV];#theta_{{{BTLatex}}}^{{lab}} [deg]",             ((100,  0,   10  ), (100,   0,   30  )), ("momLabB",           "thetaLabBDeg"      ), filter),
          HistogramDefinition(f"phiLabRecoilDegVsThetaLabRecoilDeg{histNameSuffix}", title + ";#theta_{p}^{lab} [deg];#phi_{p}^{lab} [deg];",                          ((100, 60,   80  ), (72, -180, +180  )), ("thetaLabRecoilDeg", "phiLabRecoilDeg"   ), filter),
          HistogramDefinition(f"phiLabADegVsThetaLabADeg{histNameSuffix}",           title + f";#theta_{{{ATLatex}}}^{{lab}} [deg];#phi_{{{ATLatex}}}^{{lab}} [deg];", ((100,  0,   30  ), (72, -180, +180  )), ("thetaLabADeg",      "phiLabADeg"        ), filter),
          HistogramDefinition(f"phiLabBDegVsThetaLabBDeg{histNameSuffix}",           title + f";#theta_{{{BTLatex}}}^{{lab}} [deg];#phi_{{{BTLatex}}}^{{lab}} [deg];", ((100,  0,   30  ), (72, -180, +180  )), ("thetaLabBDeg",      "phiLabBDeg"        ), filter),
        ]
  # define subsystem-dependent histograms
  pairTLatex = subSystem.pairTLatexLabel
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
      HistogramDefinition(f"anglesHF{pairLabel}", f"{title};cos#theta_{{HF}};#phi_{{HF}} [deg]", ((100, -1, +1), (72, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiHF{pairLabel}Deg")),
      HistogramDefinition(f"anglesGJ{pairLabel}", f"{title};cos#theta_{{GJ}};#phi_{{GJ}} [deg]", ((100, -1, +1), (72, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiGJ{pairLabel}Deg")),
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
  if pairLabel == "PiPi" or pairLabel == "EtaPi0":  # plots for mesonic subsystem
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
    # create histograms in m_pipi bins
    # if True:
    if False:
      #TODO generalize to other mesonic subsystems
      massPiPiRange = (0.28, 2.28)  # [GeV]
      massPiPiNmbBins = 50
      massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
      for binIndex in range(0, massPiPiNmbBins):
        massPiPiBinMin    = massPiPiRange[0] + binIndex * massPiPiBinWidth
        massPiPiBinMax    = massPiPiBinMin + massPiPiBinWidth
        massPiPiBinFilter = f"({massPiPiBinMin} < massPiPi) and (massPiPi < {massPiPiBinMax})"
        histNameSuffix    = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
        histDefs += [
          # 1D histograms
          HistogramDefinition(f"cosThetaHF{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{HF}};"  + yAxisLabel, ((100,   -1,   +1), ), (f"cosThetaHF{pairLabel}", ), massPiPiBinFilter),
          HistogramDefinition(f"cosThetaGJ{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{GJ}};"  + yAxisLabel, ((100,   -1,   +1), ), (f"cosThetaGJ{pairLabel}", ), massPiPiBinFilter),
          HistogramDefinition(f"phiHF{pairLabel}Deg{histNameSuffix}",   f"{title};#phi_{{HF}} [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"phiHF{pairLabel}Deg",   ), massPiPiBinFilter),
          HistogramDefinition(f"phiGJ{pairLabel}Deg{histNameSuffix}",   f"{title};#phi_{{GJ}} [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"phiGJ{pairLabel}Deg",   ), massPiPiBinFilter),
          # 2D histograms
          HistogramDefinition(f"anglesHF{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{HF}};#phi_{{HF}} [deg]", ((50, -1, +1), (36, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiHF{pairLabel}Deg"), massPiPiBinFilter),
          HistogramDefinition(f"anglesGJ{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{GJ}};#phi_{{GJ}} [deg]", ((50, -1, +1), (36, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiGJ{pairLabel}Deg"), massPiPiBinFilter),
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
            HistogramDefinition(f"Phi{pairLabel}Deg{histNameSuffix}", f"{title};#Phi [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"Phi{pairLabel}Deg", ), massPiPiBinFilter),
            # 2D histograms
            HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaHF{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{HF}};#Phi [deg]",  ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaHF{pairLabel}", f"Phi{pairLabel}Deg"  ), massPiPiBinFilter),
            HistogramDefinition(f"Phi{pairLabel}DegVsCosThetaGJ{pairLabel}{histNameSuffix}", f"{title};cos#theta_{{GJ}};#Phi [deg]",  ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaGJ{pairLabel}", f"Phi{pairLabel}Deg"  ), massPiPiBinFilter),
            HistogramDefinition(f"phiHF{pairLabel}DegVsPhi{pairLabel}Deg{histNameSuffix}",   f"{title};#Phi [deg];#phi_{{HF}} [deg]", (( 72, -180, +180), (72, -180, +180)), (f"Phi{pairLabel}Deg",     f"phiHF{pairLabel}Deg"), massPiPiBinFilter),
            HistogramDefinition(f"phiGJ{pairLabel}DegVsPhi{pairLabel}Deg{histNameSuffix}",   f"{title};#Phi [deg];#phi_{{GJ}} [deg]", (( 72, -180, +180), (72, -180, +180)), (f"Phi{pairLabel}Deg",     f"phiGJ{pairLabel}Deg"), massPiPiBinFilter),
          ]
          histNamesEvenOdd += [
            f"phiHF{pairLabel}DegVsPhi{pairLabel}Deg{histNameSuffix}",
            f"phiGJ{pairLabel}DegVsPhi{pairLabel}Deg{histNameSuffix}",
          ]
  else:  # baryonic subsystems
    if True:
    # if False:
      histDefs += [
        # 1D histograms
        HistogramDefinition(f"mass{pairLabel}",   f";m_{{{pairTLatex}}} [GeV];"              + yAxisLabel, ((400, 1, 5 ), ), (f"mass{pairLabel}",   )),
        HistogramDefinition(f"minusT{pairLabel}", f";#minus t_{{{pairTLatex}}} [GeV^{{2}}];" + yAxisLabel, ((100, 0, 15), ), (f"minusT{pairLabel}", )),
        # 2D histograms
        HistogramDefinition(f"cosThetaHF{pairLabel}VsMass{pairLabel}", f";m_{{{pairTLatex}}} [GeV];cos#theta_{{HF}}",                      ((50, 1, 5), (100,   -1,   +1)), (f"mass{pairLabel}", f"cosThetaHF{pairLabel}")),
        HistogramDefinition(f"phiHF{pairLabel}DegVsMass{pairLabel}",   f";m_{{{pairTLatex}}} [GeV];#phi_{{HF}} [deg]",                     ((50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}", f"phiHF{pairLabel}Deg"  )),
        HistogramDefinition(f"cosThetaGJ{pairLabel}VsMass{pairLabel}", f";m_{{{pairTLatex}}} [GeV];cos#theta_{{GJ}}",                      ((50, 1, 5), (100,   -1,   +1)), (f"mass{pairLabel}", f"cosThetaGJ{pairLabel}")),
        HistogramDefinition(f"phiGJ{pairLabel}DegVsMass{pairLabel}",   f";m_{{{pairTLatex}}} [GeV];#phi_{{GJ}} [deg]",                     ((50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}", f"phiGJ{pairLabel}Deg"  )),
        HistogramDefinition(f"MinusT{pairLabel}VsMass{pairLabel}",     f";m_{{{pairTLatex}}} [GeV];#minus t_{{{pairTLatex}}} [GeV^{{2}}]", ((50, 1, 5), ( 50,    0,    1)), (f"mass{pairLabel}", f"minusT{pairLabel}"    )),
      ]
      if beamPolInfo is not None:
        histDefs += [
          HistogramDefinition(f"Phi{pairLabel}DegVsMass{pairLabel}", f";m_{{{pairTLatex}}} [GeV];#Phi [deg]", ((50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}", f"Phi{pairLabel}Deg")),
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
  canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")


def makePlots(
  hists:            HistListType,
  histNamesEvenOdd: list[str],
  outputDirName:    str,
) -> None:
  """Writes histograms to ROOT file and generates PDF plots"""
  for hist in hists:
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
  subSystem:            SubSystemInfo,
  kinVarNameCorr:       str,  # column name to correlate with helicity-frame angles
  outputDirName:        str,  # directory to save output plot in
  histNameSuffix:       str = "",
  additionalFilterDefs: list[str] = [],  # additional filter conditions to apply
) -> None:
  """Produces 2D correlation plot of helicity-frame angles with given RDataFrame column"""
  print(f"Generating correlation plot of helicity-frame angles with '{kinVarNameCorr}' for {subSystem.pairLabel} subsystem")
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
  pairLabel = subSystem.pairLabel
  pairTLatexLabel = subSystem.pairTLatexLabel
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
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("./rootlogon.C") == 0, "Error loading './rootlogon.C'"

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  # # parameters for pi+ pi- data
  # subSystems: tuple[SubSystemInfo, ...] = (  # particle pairs to analyze; particle A is the analyzer
  #   SubSystemInfo(pairLabel = "PiPi",   lvALabel = "pip", lvBLabel = "pim",    lvRecoilLabel = "recoil", pairTLatexLabel = "#pi#pi"       ),
  #   # SubSystemInfo(pairLabel = "PipP",   lvALabel = "pip", lvBLabel = "recoil", lvRecoilLabel = "pim",    pairTLatexLabel = "p#pi^{#plus}" ),
  #   # SubSystemInfo(pairLabel = "PimP",   lvALabel = "pim", lvBLabel = "recoil", lvRecoilLabel = "pip",    pairTLatexLabel = "p{{BTLatexLabel}}"),
  # )
  # dataDirName   = "./dataPhotoProdPiPi/polarized"
  # dataPeriods   = (
  #   # "2017_01",
  #   "2017_01_ver05",
  #   # "2018_08",
  # )
  # tBinLabels    = (
  #   "tbin_0.100_0.114",  # lowest |t| bin of SDME analysis
  #   # "tbin_0.1_0.2",
  #   # "tbin_0.2_0.3",
  #   # "tbin_0.3_0.4",
  #   # "tbin_0.4_0.5",
  # )
  # beamPolLabels = (
  #   "PARA_0",
  #   # "PARA_135",
  #   # "PERP_45",
  #   # "PERP_90",
  #   # "AMO",
  # )
  # useSeparateBackgroundFiles = True  # if True, signal and background regions are stored in separate input files
  # additionalColumnDefs = {}
  # additionalFilterDefs = []
  # additionalFilterDefs = ["(0.60 < massPiPi and massPiPi < 0.88)", "(0.100 < minusTPiPi and minusTPiPi < 0.114)"]  #kinematic range used in SDME analysis

  # parameters for eta pi0 data
  subSystems: tuple[SubSystemInfo, ...] = (  # particle pairs to analyze; particle A is the analyzer
    SubSystemInfo(pairLabel = "EtaPi0", lvALabel = "eta", ATLatexLabel = "#eta", lvBLabel = "pi0", BTLatexLabel = "#pi^{0}", lvRecoilLabel = "recoil", recoilTLatexLabel ="p", pairTLatexLabel = "#eta#pi^{0}"  ),
    # SubSystemInfo(pairLabel = "EtaPi0", lvALabel = "pi0", ATLatexLabel = "#pi^{0}", lvBLabel = "eta", BTLatexLabel = "#eta", lvRecoilLabel = "recoil", recoilTLatexLabel ="p", pairTLatexLabel = "#eta#pi^{0}"  ),
  )
  dataDirName   = "./dataPhotoProdEtaPi0/polarized"
  dataPeriods   = (
    "merged",
  )
  tBinLabels    = (
    "t010020",
    "t020032",
    "t032050",
    "t050075",
    "t075100",
  )
  beamPolLabels = (
    "All",
  )
  BEAM_POL_INFOS["merged"]["All"].pol    = "Pol"
  BEAM_POL_INFOS["merged"]["All"].PhiLab = "BeamAngle"
  useSeparateBackgroundFiles = False
  additionalColumnDefs = {"eventWeight" : "weightASBS"}  # use this column as event weights
  # additionalColumnDefs = {}  #TODO no weights for generated MC
  additionalFilterDefs = []

  treeName = "kin"
  inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in ampTools format
    InputDataType.REAL_DATA             : InputDataFormat.AMPTOOLS,
    InputDataType.ACCEPTED_PHASE_SPACE  : InputDataFormat.AMPTOOLS,
    InputDataType.GENERATED_PHASE_SPACE : InputDataFormat.AMPTOOLS,
  }

  for dataPeriod in dataPeriods:
    print(f"Generating plots for data period '{dataPeriod}':")
    for tBinLabel in tBinLabels:
      print(f"Generating plots for t bin '{tBinLabel}':")
      # inputDataDirName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/Alex"
      inputDataDirName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/Nizar"
      for inputDataType, inputDataFormat in inputDataFormats.items():
        print(f"Generating plots for input data type '{inputDataType}' in format '{inputDataFormat}'")
        for beamPolLabel in beamPolLabels:  #TODO process only 1 orientation for MC data
          beamPolInfo = BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel]
          beamPolInfoPrint: list[str] = []
          if beamPolInfo is not None:
            if isinstance(beamPolInfo.pol, float):
              beamPolInfoPrint.append(f"pol = {beamPolInfo.pol:.4f}")
            if isinstance(beamPolInfo.PhiLab, float):
              beamPolInfoPrint.append(f"PhiLab = {beamPolInfo.PhiLab:.1f} deg")
            if len(beamPolInfoPrint) > 0:
              beamPolInfoPrint[0] = f": {beamPolInfoPrint[0]}"
          print(f"Generating plots for beam-polarization orientation '{beamPolLabel}'{', '.join(beamPolInfoPrint)}")
          df = None
          if inputDataType == InputDataType.REAL_DATA:
            # combine signal and background region data with correct event weights into one RDataFrame
            df = (
              getDataFrameWithCorrectEventWeights(
                dataSigRegionFileNames  = (f"{inputDataDirName}/amptools_tree_signal_{beamPolLabel}.root", ),
                dataBkgRegionFileNames  = (f"{inputDataDirName}/amptools_tree_bkgnd_{beamPolLabel}.root",  ),
                treeName                = treeName,
                friendSigRegionFileName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/data_sig_{beamPolLabel}.root.weights",
                friendBkgRegionFileName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/data_bkg_{beamPolLabel}.root.weights",
              ) if useSeparateBackgroundFiles else
              ROOT.RDataFrame(treeName, f"{inputDataDirName}/amptools_tree_data_{beamPolLabel}.root")
            )
          elif inputDataType == InputDataType.ACCEPTED_PHASE_SPACE:
            print(f"Loading accepted phase space data from '{inputDataDirName}/amptools_tree_accepted*.root'")
            df = ROOT.RDataFrame(treeName, f"{inputDataDirName}/amptools_tree_accepted*.root")
          elif inputDataType == InputDataType.GENERATED_PHASE_SPACE:
            print(f"Loading generated phase space data from '{inputDataDirName}/amptools_tree_thrown*.root'")
            df = ROOT.RDataFrame(treeName, f"{inputDataDirName}/amptools_tree_thrown*.root")
          else:
            raise RuntimeError(f"Unsupported input data type '{inputDataType}'")
          for subSystem in subSystems:
            print(f"Generating plots for subsystem '{subSystem}':")
            dfSubSystem = defineColumnsForPlots(
              df                   = df,
              inputDataFormat      = inputDataFormat,
              subSystem            = subSystem,
              beamPolInfo          = beamPolInfo,
              additionalColumnDefs = additionalColumnDefs,
              additionalFilterDefs = additionalFilterDefs,
            ).Filter((f'if (rdfentry_ == 0) {{ std::cout << "Running event loop for subsystem \'{subSystem}\'" << std::endl; }} return true;'))  # no-op filter that logs when event loop is running
            outputDirName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/{subSystem.pairLabel}/plots_{inputDataType.name}/{beamPolLabel}"
            if True:
            # if False:
              makePlots(
                *bookHistograms(
                  df            = dfSubSystem,
                  inputDataType = inputDataType,
                  subSystem     = subSystem,
                  beamPolInfo   = beamPolInfo,
                ),
                outputDirName = outputDirName,
              )
            # if True:
            if False:
              additionalFilterDefs = ["(0.72 < massPiPi and massPiPi < 0.76)", ]  # select mass bin at rho(770) peak
              outputDirName = f"{outputDirName}/anglesHFCorrelations"
              print(f"Writing helicity-frame angles correlation plots to '{outputDirName}'")
              lvs = lorentzVectors(dataFormat = inputDataFormat)
              dfSubSystem = dfSubSystem.Define(f"massPipP", f"(Double32_t)massPair({lvs['pip']}, {lvs['recoil']})")
              dfSubSystem = dfSubSystem.Define(f"massPimP", f"(Double32_t)massPair({lvs['pim']}, {lvs['recoil']})")
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
                # f"Phi{subSystem.pairLabel}Deg",
                # f"PsiHF{subSystem.pairLabel}Deg",
                # "massPipP",
                # "massPimP",
              ]:
                makeAnglesHFCorrelationPlot(
                  df                   = dfSubSystem,
                  subSystem            = subSystem,
                  kinVarNameCorr       = kinVarNameCorr,
                  outputDirName        = outputDirName,
                  additionalFilterDefs = additionalFilterDefs,
                )

  timer.stop("Total execution time")
  print(timer.summary)
