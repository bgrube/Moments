#!/usr/bin/env python3


import os

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C")
  ROOT.gROOT.LoadMacro("../rootlogon.C")
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  tBinLabel     = "tbin_0.1_0.2"
  # tBinLabel     = "tbin_0.2_0.3"
  beamPolLabels = ("PARA_0", "PARA_135", "PERP_45", "PERP_90")

  histsMassPiPi   = []
  histsMassPiPiMc = []
  for beamPolLabel in beamPolLabels:
    inputFileName = f"./{tBinLabel}/{beamPolLabel}/dataPlots.root"
    print(f"Reading data from '{inputFileName}'")
    inputFile = ROOT.TFile.Open(inputFileName, "READ")
    histStack = inputFile.Get("hMassPiPiDataAndMc")
    histsMassPiPiMc.append(histStack.GetHists().At(0))
    histsMassPiPi.append  (histStack.GetHists().At(1))
  histMassPiPiAllBeamPol = histsMassPiPi[0]
  for hist in histsMassPiPi[1:]:
    histMassPiPiAllBeamPol.Add(hist)
  histMassPiPiAllBeamPol.SetName("Data, 4 Orientations")
  print(f"Number if combos: {histMassPiPiAllBeamPol.Integral()}")
  canv = ROOT.TCanvas()
  histStack = ROOT.THStack("hMassPiPiDataAndMcAllBeamPol", ";m_{#pi#pi} [GeV];Events / 40 MeV")
  histStack.Add(histsMassPiPiMc[0])
  histStack.Add(histMassPiPiAllBeamPol)
  histsMassPiPiMc[0].SetLineColor(ROOT.kBlue + 1)
  histsMassPiPiMc[0].SetMarkerColor(ROOT.kBlue + 1)
  histMassPiPiAllBeamPol.SetLineColor(ROOT.kRed + 1)
  histMassPiPiAllBeamPol.SetMarkerColor(ROOT.kRed + 1)
  histStack.Draw("NOSTACK")
  canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
  canv.SaveAs(f"./{tBinLabel}/{histStack.GetName()}.pdf")
