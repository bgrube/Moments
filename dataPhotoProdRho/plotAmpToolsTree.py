#!/usr/bin/env python3


import os
import sys

import ROOT

sys.path.append('../')  # quick hack; should make a module
import PlottingUtilities


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  print("Loading ./FSRootMacros.C...")
  assert ROOT.gROOT.LoadMacro("./FSRootMacros.C+") == 0, "Error loading './FSRootMacros.C'."
  PlottingUtilities.setupPlotStyle(rootlogonPath = "../rootlogon.C")

  ampToolsFileName = "./amptools_tree_030994.root"
  ampToolsTreeName = "kin"

  lvBeam   = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"
  lvRecoil = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"
  lvPip    = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"
  lvPim    = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"
  polarizationAngle = 45.0
  df = (
    ROOT.RDataFrame(ampToolsTreeName, ampToolsFileName)
        .Define("GjCosTheta", f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})")
        .Define("GjTheta",    "std::acos(GjCosTheta)")
        .Define("GjPhi",      f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})")
        .Define("GjPhiDeg",   "GjPhi * TMath::RadToDeg()")
        .Define("HfCosTheta", f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})")
        .Define("HfTheta",    "std::acos(HfCosTheta)")
        .Define("HfPhi",      f"FSMath::helphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})")
        .Define("HfPhiDeg",   "HfPhi * TMath::RadToDeg()")
        .Define("Phi",        f"MyFSMath::bigPhi({lvRecoil}, {lvBeam}, {polarizationAngle})")
        .Define("PhiDeg",     "Phi * TMath::RadToDeg()")
  )
  nmbBins = 25
  hists = (
    df.Histo3D(ROOT.RDF.TH3DModel("hAmpToolsGj", ";cos#theta_{GJ};#phi_{GJ} [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180), "GjCosTheta", "GjPhiDeg", "PhiDeg"),
    df.Histo3D(ROOT.RDF.TH3DModel("hAmpToolsHf", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180), "HfCosTheta", "HfPhiDeg", "PhiDeg"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
    canv.SaveAs(f"{hist.GetName()}.pdf")
