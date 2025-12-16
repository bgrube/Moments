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
  CPP_CODE_ANGLES_GLUEX_AMPTOOLS,
  CPP_CODE_BEAM_POL_PHI,
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_FLIPYAXIS,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  CoordSysType,
  defineDataFrameColumns,
  getDataFrameWithCorrectEventWeights,
  InputDataType,
  InputDataFormat,
  SubSystemInfo,
  lorentzVectors,
)


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
  dfResult = df
  for frame in (CoordSysType.HF, CoordSysType.GJ):
    #!NOTE! coordinate system definitions for beam + target -> pi+ + pi- + recoil (all momenta in XRF):
    #    HF for pi+ pi- meson system:  use pi+  as analyzer and z_HF = -p_recoil and y_HF = p_recoil x p_beam
    #    HF for pi+- p  baryon system: use pi+- as analyzer and z_HF = -p_pi-+   and y_HF = p_beam   x p_pi-+
    #    GJ for pi+ pi- meson system:  use pi+  as analyzer and z_GJ = p_beam    and y_HF = p_recoil x p_beam
    #    GJ for pi+- p  baryon system: use pi+- as analyzer and z_GJ = p_target  and y_HF = p_beam   x p_pi-+
    #  particle A is the analyzer, particle B is the other particle in the pair, and the recoil is the third particle in the final state
    dfResult = defineDataFrameColumns(
      df                   = dfResult,
      lvTarget             = lvs["target"],
      lvBeam               = lvs["beam"],  #TODO "beam" for GJ pi+- p baryon system is p_target
      lvRecoil             = lvs[subSystem.lvRecoilLabel],
      lvA                  = lvs[subSystem.lvALabel],
      lvB                  = lvs[subSystem.lvBLabel],
      beamPolInfo          = beamPolInfo,
      frame                = frame,
      flipYAxis            = (frame == CoordSysType.HF) and subSystem.pairLabel == "PiPi",  # only flip y axis for pi+ pi- system in HF frame
      additionalColumnDefs = additionalColumnDefs,
      additionalFilterDefs = additionalFilterDefs,
      colNameSuffix        = subSystem.pairLabel,
    )
    # define additional columns for subsystem
    dfResult = (
      dfResult.Define(f"PsiDeg{frame.name}{subSystem.pairLabel}", f"(Double32_t)(fixAzimuthalAngleRange(Phi{subSystem.pairLabel} - phi{frame.name}{subSystem.pairLabel}) * TMath::RadToDeg())")
    )
  # define additional columns that are independent of subsystem
  dfResult = (
    dfResult.Define(f"mass{subSystem.pairLabel}Sq", f"(Double32_t)std::pow(mass{subSystem.pairLabel}, 2)")
            .Define("Ebeam",                        f"(Double32_t)TLorentzVector({lvs['beam']}).E()")
            # track kinematics
            .Define("momLabP",        f"(Double32_t)TLorentzVector({lvs['recoil']}).P()")
            .Define("momLabXP",       f"(Double32_t)TLorentzVector({lvs['recoil']}).X()")
            .Define("momLabYP",       f"(Double32_t)TLorentzVector({lvs['recoil']}).Y()")
            .Define("momLabZP",       f"(Double32_t)TLorentzVector({lvs['recoil']}).Z()")
            .Define("momLabPip",      f"(Double32_t)TLorentzVector({lvs['pip'   ]}).P()")
            .Define("momLabXPip",     f"(Double32_t)TLorentzVector({lvs['pip'   ]}).X()")
            .Define("momLabYPip",     f"(Double32_t)TLorentzVector({lvs['pip'   ]}).Y()")
            .Define("momLabZPip",     f"(Double32_t)TLorentzVector({lvs['pip'   ]}).Z()")
            .Define("momLabPim",      f"(Double32_t)TLorentzVector({lvs['pim'   ]}).P()")
            .Define("momLabXPim",     f"(Double32_t)TLorentzVector({lvs['pim'   ]}).X()")
            .Define("momLabYPim",     f"(Double32_t)TLorentzVector({lvs['pim'   ]}).Y()")
            .Define("momLabZPim",     f"(Double32_t)TLorentzVector({lvs['pim'   ]}).Z()")
            .Define("thetaDegLabP",   f"(Double32_t)(TLorentzVector({lvs['recoil']}).Theta() * TMath::RadToDeg())")
            .Define("thetaDegLabPip", f"(Double32_t)(TLorentzVector({lvs['pip'   ]}).Theta() * TMath::RadToDeg())")
            .Define("thetaDegLabPim", f"(Double32_t)(TLorentzVector({lvs['pim'   ]}).Theta() * TMath::RadToDeg())")
            .Define("phiDegLabP",     f"(Double32_t)(TLorentzVector({lvs['recoil']}).Phi()   * TMath::RadToDeg())")
            .Define("phiDegLabPip",   f"(Double32_t)(TLorentzVector({lvs['pip'   ]}).Phi()   * TMath::RadToDeg())")
            .Define("phiDegLabPim",   f"(Double32_t)(TLorentzVector({lvs['pim'   ]}).Phi()   * TMath::RadToDeg())")
  )
  # print(f"!!! {df.GetDefinedColumnNames()=}")
  # print(f"!!! {dfResult.GetDefinedColumnNames()=}")
  return dfResult


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
HistListType:       TypeAlias = Union[list[HistType], list[HistRResultPtrType]]

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


def bookHistograms(
  df:            ROOT.RDataFrame,
  inputDataType: InputDataType,
  subSystem:     SubSystemInfo,
) -> HistListType:
  """Books histograms for kinematic plots and returns the list of histograms"""
  applyWeights = (inputDataType == InputDataType.REAL_DATA and df.HasColumn("eventWeight"))
  yAxisLabel = "RF-Sideband Subtracted Combos" if applyWeights else "Combos"
  # define histograms that are independent of subsystem
  histDefs: list[HistogramDefinition] = [
    HistogramDefinition("Ebeam",          ";E_{beam} [GeV];"                    + yAxisLabel, ((100, 8,  9), ), ("Ebeam",          )),
    HistogramDefinition("momLabP",        ";p_{p} [GeV];"                       + yAxisLabel, ((100, 0,  1), ), ("momLabP",        )),
    HistogramDefinition("momLabPip",      ";p_{#pi^{#plus}} [GeV];"             + yAxisLabel, ((100, 0, 10), ), ("momLabPip",      )),
    HistogramDefinition("momLabPim",      ";p_{#pi^{#minus}} [GeV];"            + yAxisLabel, ((100, 0, 10), ), ("momLabPim",      )),
    HistogramDefinition("thetaDegLabP",   ";#theta_{p}^{lab} [deg];"            + yAxisLabel, ((100, 0, 80), ), ("thetaDegLabP",   )),
    HistogramDefinition("thetaDegLabPip", ";#theta_{#pi^{#plus}}^{lab} [deg];"  + yAxisLabel, ((100, 0, 80), ), ("thetaDegLabPip", )),
    HistogramDefinition("thetaDegLabPim", ";#theta_{#pi^{#minus}}^{lab} [deg];" + yAxisLabel, ((100, 0, 80), ), ("thetaDegLabPim", )),
    HistogramDefinition("thetaDegLabPVsMomLabP",     ";p_{p} [GeV];#theta_{p}^{lab} [deg]",                       ((100, 0,  1), (100, 60, 80)), ("momLabP",   "thetaDegLabP"  )),
    HistogramDefinition("thetaDegLabPipVsMomLabPip", ";p_{#pi^{#plus}} [GeV];#theta_{#pi^{#plus}}^{lab} [deg]",   ((100, 0, 10), (100,  0, 30)), ("momLabPip", "thetaDegLabPip")),
    HistogramDefinition("thetaDegLabPimVsMomLabPim", ";p_{#pi^{#minus}} [GeV];#theta_{#pi^{#minus}}^{lab} [deg]", ((100, 0, 10), (100,  0, 30)), ("momLabPim", "thetaDegLabPim")),
  ]
  # define subsystem-dependent histograms
  pairLabel = subSystem.pairLabel
  pairTLatexLabel = subSystem.pairTLatexLabel
  histDefs += [
    HistogramDefinition(f"cosThetaHF{pairLabel}", f"{pairTLatexLabel};cos#theta_{{HF}};"  + yAxisLabel, ((100,   -1,   +1), ), (f"cosThetaHF{pairLabel}", )),
    HistogramDefinition(f"cosThetaGJ{pairLabel}", f"{pairTLatexLabel};cos#theta_{{GJ}};"  + yAxisLabel, ((100,   -1,   +1), ), (f"cosThetaGJ{pairLabel}", )),
    HistogramDefinition(f"phiDegHF{pairLabel}",   f"{pairTLatexLabel};#phi_{{HF}} [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"phiDegHF{pairLabel}",   )),
    HistogramDefinition(f"phiDegGJ{pairLabel}",   f"{pairTLatexLabel};#phi_{{GJ}} [deg];" + yAxisLabel, (( 72, -180, +180), ), (f"phiDegGJ{pairLabel}",   )),
    HistogramDefinition(f"PhiDeg{pairLabel}",     f"{pairTLatexLabel};#Phi [deg];"        + yAxisLabel, (( 72, -180, +180), ), (f"PhiDeg{pairLabel}",     )),
    HistogramDefinition(f"anglesHF{pairLabel}",           f"{pairTLatexLabel};cos#theta_{{HF}};#phi_{{HF}} [deg]", ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiDegHF{pairLabel}")),
    HistogramDefinition(f"anglesGJ{pairLabel}",           f"{pairTLatexLabel};cos#theta_{{GJ}};#phi_{{GJ}} [deg]", ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiDegGJ{pairLabel}")),
    HistogramDefinition(f"PhiDegVsCosThetaHF{pairLabel}", f"{pairTLatexLabel};cos#theta_{{HF}};#Phi [deg]",        ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaHF{pairLabel}", f"PhiDeg{pairLabel}"  )),
    HistogramDefinition(f"PhiDegVsCosThetaGJ{pairLabel}", f"{pairTLatexLabel};cos#theta_{{GJ}};#Phi [deg]",        ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaGJ{pairLabel}", f"PhiDeg{pairLabel}"  )),
    HistogramDefinition(f"PhiDegVsPhiDegHF{pairLabel}",   f"{pairTLatexLabel};#phi_{{HF}} [deg];#Phi [deg]",       (( 72, -180, +180), (72, -180, +180)), (f"phiDegHF{pairLabel}",   f"PhiDeg{pairLabel}"  )),
    HistogramDefinition(f"PhiDegVsPhiDegGJ{pairLabel}",   f"{pairTLatexLabel};#phi_{{GJ}} [deg];#Phi [deg]",       (( 72, -180, +180), (72, -180, +180)), (f"phiDegGJ{pairLabel}",   f"PhiDeg{pairLabel}"  )),
    HistogramDefinition(f"PhiDeg{pairLabel}VsPhiDegGJ{pairLabel}VsCosThetaGJ{pairLabel}", f"{pairTLatexLabel};cos#theta_{{GJ}};#phi_{{GJ}} [deg];#Phi [deg]", ((25, -1, +1), (25, -180, +180), (25, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiDegGJ{pairLabel}", f"PhiDeg{pairLabel}")),
    HistogramDefinition(f"PhiDeg{pairLabel}VsPhiDegHF{pairLabel}VsCosThetaHF{pairLabel}", f"{pairTLatexLabel};cos#theta_{{HF}};#phi_{{HF}} [deg];#Phi [deg]", ((25, -1, +1), (25, -180, +180), (25, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiDegHF{pairLabel}", f"PhiDeg{pairLabel}")),
  ]
  if pairLabel == "PiPi":
    histDefs += [
      HistogramDefinition(f"mass{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];"              + yAxisLabel, ((400, 0.28, 2.28), ), (f"mass{pairLabel}",   )),
      HistogramDefinition(f"minusT{pairLabel}", f";#minus t_{{{pairTLatexLabel}}} [GeV^{{2}}];" + yAxisLabel, ((100, 0,    1),    ), (f"minusT{pairLabel}", )),
      HistogramDefinition(f"CosThetaGJ{pairLabel}VsMass{pairLabel}", f";m_{{{pairTLatexLabel}}} [GeV];cos#theta_{{GJ}}",                           ((50, 0.28, 2.28), (100,   -1,   +1)), (f"mass{pairLabel}", f"cosThetaGJ{pairLabel}")),
      HistogramDefinition(f"PhiDegGJ{pairLabel}VsMass{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];#phi_{{GJ}}",                                ((50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}", f"phiDegGJ{pairLabel}"  )),
      HistogramDefinition(f"CosThetaHF{pairLabel}VsMass{pairLabel}", f";m_{{{pairTLatexLabel}}} [GeV];cos#theta_{{HF}}",                           ((50, 0.28, 2.28), (100,   -1,   +1)), (f"mass{pairLabel}", f"cosThetaHF{pairLabel}")),
      HistogramDefinition(f"PhiDegHF{pairLabel}VsMass{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];#phi_{{HF}}",                                ((50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}", f"phiDegHF{pairLabel}"  )),
      HistogramDefinition(f"PhiDegVsMass{pairLabel}",                f";m_{{{pairTLatexLabel}}} [GeV];#Phi",                                       ((50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}", f"PhiDeg{pairLabel}"    )),
      HistogramDefinition(f"MinusT{pairLabel}VsMass{pairLabel}",     f";m_{{{pairTLatexLabel}}} [GeV];#minus t_{{{pairTLatexLabel}}} [GeV^{{2}}]", ((50, 0.28, 2.28), ( 50,    0,    1)), (f"mass{pairLabel}", f"minusT{pairLabel}"    )),
    ]
    # create histograms in m_pipi bins
    if True:
      massPiPiRange = (0.28, 2.28)  # [GeV]
      massPiPiNmbBins = 50
      massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
      for binIndex in range(0, massPiPiNmbBins):
        massPiPiBinMin    = massPiPiRange[0] + binIndex * massPiPiBinWidth
        massPiPiBinMax    = massPiPiBinMin + massPiPiBinWidth
        massPiPiBinFilter = f"({massPiPiBinMin} < massPiPi) and (massPiPi < {massPiPiBinMax})"
        histNameSuffix    = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
        histDefs += [
          HistogramDefinition(f"anglesGJ{pairLabel}{histNameSuffix}",           f"{pairTLatexLabel};cos#theta_{{GJ}};#phi_{{GJ}} [deg]", ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaGJ{pairLabel}", f"phiDegGJ{pairLabel}"), massPiPiBinFilter),
          HistogramDefinition(f"anglesHF{pairLabel}{histNameSuffix}",           f"{pairTLatexLabel};cos#theta_{{HF}};#phi_{{HF}} [deg]", ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaHF{pairLabel}", f"phiDegHF{pairLabel}"), massPiPiBinFilter),
          HistogramDefinition(f"PhiDegVsCosThetaHF{pairLabel}{histNameSuffix}", f"{pairTLatexLabel};cos#theta_{{HF}};#Phi [deg]",        ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaHF{pairLabel}", f"PhiDeg{pairLabel}"  ), massPiPiBinFilter),
          HistogramDefinition(f"PhiDegVsCosThetaGJ{pairLabel}{histNameSuffix}", f"{pairTLatexLabel};cos#theta_{{GJ}};#Phi [deg]",        ((100,   -1,   +1), (72, -180, +180)), (f"cosThetaGJ{pairLabel}", f"PhiDeg{pairLabel}"  ), massPiPiBinFilter),
          HistogramDefinition(f"PhiDegVsPhiDegHF{pairLabel}{histNameSuffix}",   f"{pairTLatexLabel};#phi_{{HF}} [deg];#Phi [deg]",       (( 72, -180, +180), (72, -180, +180)), (f"phiDegHF{pairLabel}",   f"PhiDeg{pairLabel}"  ), massPiPiBinFilter),
          HistogramDefinition(f"PhiDegVsPhiDegGJ{pairLabel}{histNameSuffix}",   f"{pairTLatexLabel};#phi_{{GJ}} [deg];#Phi [deg]",       (( 72, -180, +180), (72, -180, +180)), (f"phiDegGJ{pairLabel}",   f"PhiDeg{pairLabel}"  ), massPiPiBinFilter),
        ]
  else:
    histDefs += [
      HistogramDefinition(f"mass{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];"              + yAxisLabel, ((400, 1, 5 ), ), (f"mass{pairLabel}",   )),
      HistogramDefinition(f"minusT{pairLabel}", f";#minus t_{{{pairTLatexLabel}}} [GeV^{{2}}];" + yAxisLabel, ((100, 0, 15), ), (f"minusT{pairLabel}", )),
      HistogramDefinition(f"CosThetaGJ{pairLabel}VsMass{pairLabel}", f";m_{{{pairTLatexLabel}}} [GeV];cos#theta_{{GJ}}",                           ((50, 1, 5), (100,   -1,   +1)), (f"mass{pairLabel}", f"cosThetaGJ{pairLabel}")),
      HistogramDefinition(f"PhiDegGJ{pairLabel}VsMass{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];#phi_{{GJ}}",                                ((50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}", f"phiDegGJ{pairLabel}"  )),
      HistogramDefinition(f"CosThetaHF{pairLabel}VsMass{pairLabel}", f";m_{{{pairTLatexLabel}}} [GeV];cos#theta_{{HF}}",                           ((50, 1, 5), (100,   -1,   +1)), (f"mass{pairLabel}", f"cosThetaHF{pairLabel}")),
      HistogramDefinition(f"PhiDegHF{pairLabel}VsMass{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];#phi_{{HF}}",                                ((50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}", f"phiDegHF{pairLabel}"  )),
      HistogramDefinition(f"PhiDegVsMass{pairLabel}",                f";m_{{{pairTLatexLabel}}} [GeV];#Phi",                                       ((50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}", f"PhiDeg{pairLabel}"    )),
      HistogramDefinition(f"MinusT{pairLabel}VsMass{pairLabel}",     f";m_{{{pairTLatexLabel}}} [GeV];#minus t_{{{pairTLatexLabel}}} [GeV^{{2}}]", ((50, 1, 5), ( 50,    0,    1)), (f"mass{pairLabel}", f"minusT{pairLabel}"    )),
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
  return hists


def makePlots(
  hists:         HistListType,
  outputDirName: str,
) -> None:
  """Writes histograms to ROOT file and generates PDF plots"""
  os.makedirs(outputDirName, exist_ok = True)
  outRootFileName = f"{outputDirName}/plots.root"
  outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  outRootFile.cd()
  print(f"Writing histograms to '{outRootFileName}'")
  for hist in hists:
    print(f"Generating histogram '{hist.GetName()}'")
    ROOT.gStyle.SetOptStat("i")
    # ROOT.gStyle.SetOptStat(1111111)
    ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
    canv = ROOT.TCanvas()
    if "TH2" in hist.ClassName() and str(hist.GetName()).startswith("mass"):
      canv.SetLogz(1)
    hist.SetMinimum(0)
    if "TH3" in hist.ClassName():
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
    else:
      hist.Draw("COLZ")
    hist.Write()
    canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")
  outRootFile.Close()


def decomposeHistEvenOdd(hist: ROOT.TH3) -> tuple[ROOT.TH3, ROOT.TH3, ROOT.TH3]:
  """Decomposes a 3D histogram into even and odd parts based on symmetry along the phi axis, which must be the y axis; returns (odd, even, odd + even)"""
  histOdd  = hist.Clone(f"{hist.GetName()}_odd")
  histEven = hist.Clone(f"{hist.GetName()}_even")
  assert hist.GetNbinsY() % 2 == 0, "Number of phi bins must be even!"
  for thetaBin in range(1, hist.GetNbinsX() + 1):
    for PhiBin in range(1, hist.GetNbinsZ() + 1):
      for phiBinNeg in range(1, hist.GetNbinsY() // 2 + 1):  # only need to loop over half of phi bins
        phiBinPos = hist.GetYaxis().FindBin(-hist.GetYaxis().GetBinCenter(phiBinNeg))
        phiPosVal = hist.GetBinContent(thetaBin, phiBinPos, PhiBin)
        phiNegVal = hist.GetBinContent(thetaBin, phiBinNeg, PhiBin)
        phiOddVal  = (phiPosVal - phiNegVal) / 2
        phiEvenVal = (phiPosVal + phiNegVal) / 2
        histOdd.SetBinContent (thetaBin, phiBinPos, PhiBin, +phiOddVal)
        histOdd.SetBinContent (thetaBin, phiBinNeg, PhiBin, -phiOddVal)
        histEven.SetBinContent(thetaBin, phiBinPos, PhiBin, phiEvenVal)
        histEven.SetBinContent(thetaBin, phiBinNeg, PhiBin, phiEvenVal)
  histSum = hist.Clone(f"{hist.GetName()}_sum")
  histSum.Add(histOdd, histEven)
  return histOdd, histEven, histSum


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_ANGLES_GLUEX_AMPTOOLS)
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  dataDirName   = "./polarized"
  dataPeriods   = (
    # "2017_01",
    "2018_08",
  )
  tBinLabels    = (
    "tbin_0.1_0.2",
    "tbin_0.2_0.3",
    "tbin_0.3_0.4",
    "tbin_0.4_0.5",
  )
  beamPolLabels = (
    "PARA_0",
    "PARA_135",
    "PERP_45",
    "PERP_90",
  )
  inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in ampTools format
    InputDataType.REAL_DATA             : InputDataFormat.AMPTOOLS,
    InputDataType.ACCEPTED_PHASE_SPACE  : InputDataFormat.AMPTOOLS,
    InputDataType.GENERATED_PHASE_SPACE : InputDataFormat.AMPTOOLS,
  }
  subSystems: tuple[SubSystemInfo, ...] = (  # particle pairs to analyze; particle A is the analyzer
    SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim",    lvRecoilLabel = "recoil", pairTLatexLabel = "#pi#pi"       ),
    SubSystemInfo(pairLabel = "PipP", lvALabel = "pip", lvBLabel = "recoil", lvRecoilLabel = "pim",    pairTLatexLabel = "p#pi^{#plus}" ),
    SubSystemInfo(pairLabel = "PimP", lvALabel = "pim", lvBLabel = "recoil", lvRecoilLabel = "pip",    pairTLatexLabel = "p#pi^{#minus}"),
  )
  additionalColumnDefs = {}
  additionalFilterDefs = []
  treeName             = "kin"

  for dataPeriod in dataPeriods:
    print(f"Generating plots for data period '{dataPeriod}':")
    for tBinLabel in tBinLabels:
      print(f"Generating plots for t bin '{tBinLabel}':")
      inputDataDirName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/Alex"
      for inputDataType, inputDataFormat in inputDataFormats.items():
        print(f"Generating plots for input data type '{inputDataType}' in format '{inputDataFormat}'")
        for beamPolLabel in beamPolLabels:  #TODO process only 1 orientation for MC data
          beamPolInfo = BEAM_POL_INFOS[dataPeriod][beamPolLabel]
          print(f"Generating plots for beam-polarization orientation '{beamPolLabel}'"
                + (f": pol = {beamPolInfo.pol:.4f}, PhiLab = {beamPolInfo.PhiLab:.1f} deg" if beamPolInfo is not None else ""))
          df = None
          if inputDataType == InputDataType.REAL_DATA:
            # combine signal and background region data with correct event weights into one RDataFrame
            df = getDataFrameWithCorrectEventWeights(
              dataSigRegionFileNames  = (f"{inputDataDirName}/amptools_tree_signal_{beamPolLabel}.root", ),
              dataBkgRegionFileNames  = (f"{inputDataDirName}/amptools_tree_bkgnd_{beamPolLabel}.root",  ),
              treeName                = treeName,
              friendSigRegionFileName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/data_sig_{beamPolLabel}.root.weights",
              friendBkgRegionFileName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/data_bkg_{beamPolLabel}.root.weights",
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
            )
            makePlots(
              hists = bookHistograms(
                df            = dfSubSystem,
                inputDataType = inputDataType,
                subSystem     = subSystem,
              ),
              outputDirName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/{subSystem.pairLabel}/plots_{inputDataType.name}/{beamPolLabel}",
            )
