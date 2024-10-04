#!/usr/bin/env python3


import os

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  # ROOT.gStyle.SetOptStat("")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")

  #TODO perform sideband subtraction
  ampToolsFileName = "./amptools_tree_data_tbin1_ebin4.root"
  # ampToolsFileName = "./amptools_tree_bkgnd_tbin1_ebin4.root"
  ampToolsTreeName = "kin"

  #TODO plot pipi mass
  lvBeam   = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"
  lvRecoil = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"
  lvPip    = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"
  lvPim    = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"
  df = ROOT.RDataFrame(ampToolsTreeName, ampToolsFileName) \
           .Define("GjCosTheta", f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})") \
           .Define("GjTheta",    "std::acos(GjCosTheta)") \
           .Define("GjPhi",      f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})") \
           .Define("GjPhiDeg",   "GjPhi * TMath::RadToDeg()") \
           .Define("HfCosTheta", f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})") \
           .Define("HfTheta",    "std::acos(HfCosTheta)") \
           .Define("HfPhi",      f"FSMath::helphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})") \
           .Define("HfPhiDeg",   "HfPhi * TMath::RadToDeg()")
  hists = (
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesGj", ";cos#theta_{GJ};#phi_{GJ} [deg]", 50, -1, +1, 50, -180, +180), "GjCosTheta", "GjPhiDeg"),
    df.Histo2D(ROOT.RDF.TH2DModel("hdataAnglesHf", ";cos#theta_{HF};#phi_{HF} [deg]", 50, -1, +1, 50, -180, +180), "HfCosTheta", "HfPhiDeg"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.Draw("COLZ")
    canv.SaveAs(f"{hist.GetName()}.pdf")
