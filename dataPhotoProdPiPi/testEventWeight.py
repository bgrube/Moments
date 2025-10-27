#!/usr/bin/env python3


from __future__ import annotations

import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_MASSPAIR,
  InputDataFormat,
  lorentzVectors,
)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)

  # inputDataFileNamePatternSig = "./polarized/2018_08/tbin_0.4_0.5/Alex/amptools_tree_signal_PARA_0.root"
  # inputDataFileNamePatternBkg = "./polarized/2018_08/tbin_0.4_0.5/Alex/amptools_tree_bkgnd_PARA_0.root"
  inputDataFileNamePatternSig = "./polarized/2017_01/tbin_0.2_0.3/Alex/amptools_tree_signal_????_*.root"
  inputDataFileNamePatternBkg = "./polarized/2017_01/tbin_0.2_0.3/Alex/amptools_tree_bkgnd_????_*.root"
  treeName = "kin"

  dfs = {
    "Sig" : ROOT.RDataFrame(treeName, inputDataFileNamePatternSig),
    "Bkg" : ROOT.RDataFrame(treeName, inputDataFileNamePatternBkg),
  }
  lvs = lorentzVectors(dataFormat = InputDataFormat.AMPTOOLS)
  dfs = {label : df.Define("massPiPi", f"(Double32_t)massPair({lvs['pip']}, {lvs['pim']})") for label, df in dfs.items()}
  hists = {
    label : df.Histo1D(
      ROOT.RDF.TH1DModel(f"massPiPi_{label}", f";m_{{#pi#pi}} [GeV];Combos {label}", 400, 0.28, 2.28),
      "massPiPi",
      "Weight",
    ).GetValue()
    for label, df in dfs.items()
  }

  for label, hist in hists.items():
    ROOT.gStyle.SetOptStat("i")
    ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.Draw("E")
    canv.SaveAs(f"{hist.GetName()}.pdf")
  canv = ROOT.TCanvas()
  histStack = ROOT.THStack("massPiPi", f";{hists['Sig'].GetXaxis().GetTitle()};{hists['Sig'].GetYaxis().GetTitle()}")
  hist = hists["Sig"]
  hist.SetLineColor  (ROOT.kGreen + 2)
  hist.SetMarkerColor(ROOT.kGreen + 2)
  hist.SetTitle("Sig")
  histStack.Add(hist)
  hist = hists["Bkg"]
  hist.SetLineColor  (ROOT.kBlue + 1)
  hist.SetMarkerColor(ROOT.kBlue + 1)
  hist.SetTitle("Bkg")
  histStack.Add(hist)
  hist = hists["Sig"].Clone("massPiPi_subtracted")
  hist.Add(hists["Bkg"], -1)
  hist.SetLineColor  (ROOT.kRed + 1)
  hist.SetMarkerColor(ROOT.kRed + 1)
  hist.SetTitle("Sig#minus Bkg")
  histStack.Add(hist)
  histStack.Draw("E NOSTACK")
  canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
  canv.SaveAs(f"{histStack.GetName()}.pdf")
  canv = ROOT.TCanvas()
  hist.SetMinimum(0)
  hist.SetLineColor  (ROOT.kBlack)
  hist.SetMarkerColor(ROOT.kBlack)
  hist.SetTitle("")
  hist.Draw("E")
  canv.SaveAs(f"{hist.GetName()}.pdf")
