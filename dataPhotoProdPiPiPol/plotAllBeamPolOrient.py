#!/usr/bin/env python3


import os

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  tBinLabel     = "tbin_0.1_0.2"
  # tBinLabel     = "tbin_0.2_0.3"
  beamPolLabels = ("PARA_0", "PARA_135", "PERP_45", "PERP_90")

  histsMassPiPi   = []
  histsMassPiPiMc = []
  for beamPolLabel in beamPolLabels:
    inputFileName = f"./{tBinLabel}/data_{beamPolLabel}/dataPlots.root"
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

  # plot histograms of `hadd`ed ROOT file
  inputFileName = f"./{tBinLabel}/dataPlots.root"
  histNamesToPlot = [
    "hDataMassPiPi",
  ]
  massPiPiRange = (0.28, 2.28)  # [GeV]
  massPiPiNmbBins = 50
  massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
  for binIndex in range(0, massPiPiNmbBins):
    massPiPiBinMin = massPiPiRange[0] + binIndex * massPiPiBinWidth
    massPiPiBinMax = massPiPiBinMin + massPiPiBinWidth
    histNameSuffix = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
    histNamesToPlot += [f"hDataAnglesGjPiPi{histNameSuffix}", f"hDataAnglesHfPiPi{histNameSuffix}"]
  print(f"Reading data from '{inputFileName}'")
  inputFile = ROOT.TFile.Open(inputFileName, "READ")
  for histName in histNamesToPlot:
    print(f"Plotting histogram '{histName}'")
    hist = inputFile.Get(histName)
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.Draw("COLZ")
    canv.SaveAs(f"./{tBinLabel}/{histName}.pdf")
