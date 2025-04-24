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
  assert ROOT.gROOT.LoadMacro("./FSRootMacros.C+") == 0, "Error loading './FSRootMacros.C'."
  PlottingUtilities.setupPlotStyle(rootlogonPath = "../rootlogon.C")

  ampToolsFileName = "./gen_amp_030994.root"
  ampToolsTreeName = "kin"

  lvBeam   = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"
  lvRecoil = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"
  lvPip    = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # not clear whether correct index is 1 or 2
  lvPim    = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # not clear whether correct index is 1 or 2
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
    df.Histo3D(ROOT.RDF.TH3DModel("hGenAmpGj", ";cos#theta_{GJ};#phi_{GJ} [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180), "GjCosTheta", "GjPhiDeg", "PhiDeg"),
    df.Histo3D(ROOT.RDF.TH3DModel("hGenAmpHf", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180), "HfCosTheta", "HfPhiDeg", "PhiDeg"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
    canv.SaveAs(f"{hist.GetName()}.pdf")
