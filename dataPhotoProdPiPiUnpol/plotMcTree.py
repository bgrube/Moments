#!/usr/bin/env python3


import os

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  # declare C++ function to calculate invariant mass of a particle
  CPP_CODE = """
  double
	mass(const double Px, const double Py, const double Pz, const double E)
	{
		const TLorentzVector p(Px, Py, Pz, E);
		return p.M();
	}
  """
  ROOT.gInterpreter.Declare(CPP_CODE)

  # read MC data in AmpTools format
  # mcDataFileName = "./amptools_tree_thrown_tbin1_ebin4.root"
  mcDataFileName = "./amptools_tree_accepted_tbin1_ebin4.root"
  treeName = "kin"

  lvBeam   = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"
  lvRecoil = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"
  lvPip    = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # not clear whether correct index is 1 or 2
  lvPim    = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # not clear whether correct index is 1 or 2
  df = ROOT.RDataFrame(treeName, mcDataFileName) \
           .Define("FsMassRecoil", f"mass({lvRecoil})") \
           .Define("FsMassPip",    f"mass({lvPip})") \
           .Define("FsMassPim",    f"mass({lvPim})") \
           .Define("GjCosTheta",   f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})") \
           .Define("GjTheta",      "std::acos(GjCosTheta)") \
           .Define("GjPhi",        f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})") \
           .Define("GjPhiDeg",     "GjPhi * TMath::RadToDeg()") \
           .Define("HfCosTheta",   f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})") \
           .Define("HfTheta",      "std::acos(HfCosTheta)") \
           .Define("HfPhi",        f"FSMath::helphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})") \
           .Define("HfPhiDeg",     "HfPhi * TMath::RadToDeg()")
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassRecoil", ";m_{Recoil} [GeV]",       100, 0, 2), "FsMassRecoil"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPip",    ";m_{#pi^{#plus}} [GeV]",  100, 0, 2), "FsMassPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPim",    ";m_{#pi^{#minus}} [GeV]", 100, 0, 2), "FsMassPim"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesGj",     ";cos#theta_{GJ};#phi_{GJ} [deg]", 50, -1, +1, 50, -180, +180), "GjCosTheta", "GjPhiDeg"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesHf",     ";cos#theta_{HF};#phi_{HF} [deg]", 50, -1, +1, 50, -180, +180), "HfCosTheta", "HfPhiDeg"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.Draw("COLZ")
    canv.SaveAs(f"{hist.GetName()}.pdf")
