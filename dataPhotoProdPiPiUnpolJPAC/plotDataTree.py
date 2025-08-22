#!/usr/bin/env python3


from __future__ import annotations

import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  defineDataFrameColumns,
  lorentzVectorsJpac,
  readDataJpac,
)


def plotHistograms(
  df:            ROOT.RDataFrame,
  outputDirName: str,
  frame:         str,
  plotJpac:      bool = True,
) -> None:
  """Plots histograms"""
  dfPlot = df
  hists = [
    dfPlot.Histo1D(ROOT.RDF.TH1DModel("hMcMinusT",                ";#minus t [GeV^{2}];Count",      100,  0,     1),   "minusT"),
    dfPlot.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPi",              ";m_{#pi#pi} [GeV];Count",        100,  0.4,   1.4), "mass"),
    dfPlot.Histo1D(ROOT.RDF.TH1DModel(f"hMcCosTheta{frame}PiPi", f";cos#theta_{{{frame}}};Count",   100, -1,    +1),   "cosTheta"),
    dfPlot.Histo1D(ROOT.RDF.TH1DModel(f"hMcPhi{frame}PiPi",      f";#phi_{{{frame}}} [deg]; Count", 120, -180,  +180), "phiDeg"),
    dfPlot.Histo2D(ROOT.RDF.TH2DModel(f"hMcAngles{frame}PiPi",             f";cos#theta_{{{frame}}};#phi_{{{frame}}} [deg]", 100, -1,  +1,    72, -180, +180), "cosTheta", "phiDeg"),
    dfPlot.Histo2D(ROOT.RDF.TH2DModel(f"hMcMassPiPiVsCosTheta{frame}PiPi", f";m_{{#pi#pi}} [GeV];cos#theta_{{{frame}}}",      50,  0.4, 1.4, 100,   -1,   +1), "mass",     "cosTheta"),
    dfPlot.Histo2D(ROOT.RDF.TH2DModel(f"hMcMassPiPiVsPhiDeg{frame}PiPi",   f";m_{{#pi#pi}} [GeV];#phi_{{{frame}}}",           50,  0.4, 1.4,  72, -180, +180), "mass",     "phiDeg"),
  ]
  if plotJpac:
    hists += [
      dfPlot.Histo1D(ROOT.RDF.TH1DModel("hMcMinusTJpac",         ";#minus t [GeV^{2}];Count", 100,  0,     1),   "minusTJpac"),
      dfPlot.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPiJPac",       ";m_{#pi#pi} [GeV];Count",   100,  0.4,   1.4), "massJpac"),
      dfPlot.Histo1D(ROOT.RDF.TH1DModel("hMcCosThetaHfPiPiJpac", ";cos#theta_{Hf};Count",     100, -1,    +1),   "cosThetaJpac"),
      dfPlot.Histo1D(ROOT.RDF.TH1DModel(f"hMcPhiHfPiPiJpac",     ";#phi_{Hf} [deg]; Count",   120, -180,  +180), "phiDegJpac"),
    ]

  print(f"Writing histograms to '{outputDirName}'")
  for hist in hists:
      print(f"Generating histogram '{hist.GetName()}'")
      canv = ROOT.TCanvas()
      hist.SetMinimum(0)
      hist.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)

  inputData: dict[str, str] = {  # mapping of t-bin labels to input file names
    "tbin_0.4_0.5" : "./mc/mc_full_model/mc0.4-0.5_ful.dat",
    # "tbin_0.5_0.6" : "./mc/mc_full_model/mc0.5-0.6_ful.dat",
    # "tbin_0.6_0.7" : "./mc/mc_full_model/mc0.6-0.7_ful.dat",
    # "tbin_0.7_0.8" : "./mc/mc_full_model/mc0.7-0.8_ful.dat",
    # "tbin_0.8_0.9" : "./mc/mc_full_model/mc0.8-0.9_ful.dat",
    # "tbin_0.9_1.0" : "./mc/mc_full_model/mc0.9-1.0_ful.dat",
  }
  # outputDirName  = "mc_full"
  outputDirName  = "foo"
  outputTreeName = "PiPi"
  outputColumns  = ("cosTheta", "theta", "phi", "phiDeg", "mass", "minusT")
  frame          = "Hf"

  for tBinLabel, inputFileName in inputData.items():
    os.makedirs(f"{outputDirName}/{tBinLabel}", exist_ok = True)
    df = defineDataFrameColumns(
      df    = readDataJpac(inputFileName),
      frame = frame,
      **lorentzVectorsJpac(),
    )
    print(f"ROOT DataFrame columns: {list(df.GetColumnNames())}")
    print(f"ROOT DataFrame entries: {df.Count().GetValue()}")
    plotHistograms(
      df            = df,
      outputDirName = f"{outputDirName}/{tBinLabel}",
      frame         = frame,
      plotJpac      = True,
    )
