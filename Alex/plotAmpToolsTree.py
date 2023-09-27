#!/usr/bin/env python3


import os
import sys

import ROOT

sys.path.append('../')  # quick hack; should make a module
import PlottingUtilities


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  print("Loading ./FSRootMacros.C...")
  assert ROOT.gROOT.LoadMacro("./FSRootMacros.C+") == 0; "Could not load macro './FSRootMacros.C'."
  PlottingUtilities.setupPlotStyle(rootlogonPath = "../rootlogon.C")

  ampToolsFileName = "./amptools_tree_030994.root"
  ampToolsTreeName = "kin"

  lvBeam   = "beam_Px,   beam_Py,   beam_Pz,   beam_En"
  lvRecoil = "recoil_Px, recoil_Py, recoil_Pz, recoil_En"
  lvPip    = "pip_Px,    pip_Py,    pip_Pz,    pip_En"
  lvPim    = "pim_Px,    pim_Py,    pim_Pz,    pim_En"
  polarizationAngle = 45.0
  df = ROOT.RDataFrame(ampToolsTreeName, ampToolsFileName) \
           .Define("beam_En",    "beam_p4_kin.Energy()") \
           .Define("beam_Px",    "beam_p4_kin.Px()") \
           .Define("beam_Py",    "beam_p4_kin.Py()") \
           .Define("beam_Pz",    "beam_p4_kin.Pz()") \
           .Define("recoil_En",  "p_p4_kin.Energy()") \
           .Define("recoil_Px",  "p_p4_kin.Px()") \
           .Define("recoil_Py",  "p_p4_kin.Py()") \
           .Define("recoil_Pz",  "p_p4_kin.Pz()") \
           .Define("pip_En",     "pip_p4_kin.Energy()") \
           .Define("pip_Px",     "pip_p4_kin.Px()") \
           .Define("pip_Py",     "pip_p4_kin.Py()") \
           .Define("pip_Pz",     "pip_p4_kin.Pz()") \
           .Define("pim_En",     "pim_p4_kin.Energy()") \
           .Define("pim_Px",     "pim_p4_kin.Px()") \
           .Define("pim_Py",     "pim_p4_kin.Py()") \
           .Define("pim_Pz",     "pim_p4_kin.Pz()") \
           .Define("GjCosTheta", f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})") \
           .Define("GjTheta",    "std::acos(GjCosTheta)") \
           .Define("GjPhi",      f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})") \
           .Define("GjPhiDeg",   "GjPhi * TMath::RadToDeg()") \
           .Define("HfCosTheta", f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})") \
           .Define("HfTheta",    "std::acos(HfCosTheta)") \
           .Define("HfPhi",      f"FSMath::helphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})") \
           .Define("HfPhiDeg",   "HfPhi * TMath::RadToDeg()") \
           .Define("Phi",        f"MyFSMath::bigPhi({lvRecoil}, {lvBeam}, {polarizationAngle})") \
           .Define("PhiDeg",     "Phi * TMath::RadToDeg()")
  # hist = df.Histo1D(ROOT.RDF.TH1DModel("hTest", ";Event Number", 100, 0, 20000), "event")
  # hist = df.Histo1D(ROOT.RDF.TH1DModel("hTest", ";p_{x}^{#pi^{+}} [GeV/c]", 100, 0, 10), "pip_px")
  # hist = df.Histo1D(ROOT.RDF.TH1DModel("hTest", ";cos#theta_{GJ}", 100, -1, +1), "GjCosTheta")
  # hist = df.Histo1D(ROOT.RDF.TH1DModel("hTest", ";#phi_{GJ} [deg]", 100, -180, +180), "GjPhiDeg")
  # canv = ROOT.TCanvas()
  # hist.SetMinimum(0)
  # hist.Draw()
  # canv.SaveAs(f"{hist.GetName()}.pdf")
  nmbBins = 25
  hists = (
    df.Histo3D(ROOT.RDF.TH3DModel("hSignalGj", ";cos#theta_{GJ};#phi_{GJ} [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180), "GjCosTheta", "GjPhiDeg", "PhiDeg"),
    df.Histo3D(ROOT.RDF.TH3DModel("hSignalHf", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180), "HfCosTheta", "HfPhiDeg", "PhiDeg"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
    canv.SaveAs(f"{hist.GetName()}.pdf")
