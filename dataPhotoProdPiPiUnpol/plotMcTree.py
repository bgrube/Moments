#!/usr/bin/env python3


import os

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

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
  # declare C++ function to calculate invariant mass of a pair of particles
  CPP_CODE = """
	double
	massPair(
		const double Px1, const double Py1, const double Pz1, const double E1,
		const double Px2, const double Py2, const double Pz2, const double E2
	)	{
		const TLorentzVector p1(Px1, Py1, Pz1, E1);
		const TLorentzVector p2(Px2, Py2, Pz2, E2);
		return (p1 + p2).M();
	}
  """
  ROOT.gInterpreter.Declare(CPP_CODE)

  # read MC data in AmpTools format
  # mcDataFileName = "./amptools_tree_thrown_tbin1_ebin4_rho.root"
  # mcDataFileName = "./amptools_tree_accepted_tbin1_ebin4_rho.root"
  # mcDataFileName = "./amptools_tree_thrown_tbin1_ebin4*.root"
  mcDataFileName = "./amptools_tree_accepted_tbin1_ebin4*.root"
  treeName = "kin"

  lvBeam   = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"
  lvRecoil = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"
  lvPip    = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # not clear whether correct index is 1 or 2
  lvPim    = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # not clear whether correct index is 1 or 2
  df = (
    ROOT.RDataFrame(treeName, mcDataFileName)
        .Define("FsMassRecoil", f"mass({lvRecoil})")
        .Define("FsMassPip",    f"mass({lvPip})")
        .Define("FsMassPim",    f"mass({lvPim})")
        .Define("MassPiPi",     f"massPair({lvPip}, {lvPim})")
        .Define("MassPipP",     f"massPair({lvPip}, {lvRecoil})")
        .Define("MassPimP",     f"massPair({lvPim}, {lvRecoil})")
        .Define("GjCosTheta",   f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})")
        .Define("GjTheta",      "std::acos(GjCosTheta)")
        .Define("GjPhi",        f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})")
        .Define("GjPhiDeg",     "GjPhi * TMath::RadToDeg()")
        .Define("HfCosTheta",   f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})")
        .Define("HfTheta",      "std::acos(HfCosTheta)")
        .Define("HfPhi",        f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam})")
        .Define("HfPhiDeg",     "HfPhi * TMath::RadToDeg()")
  )
  yAxisLabel = "Events"
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassRecoil", ";m_{Recoil} [GeV];"        + yAxisLabel, 100, 0,      2),      "FsMassRecoil"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPip",    ";m_{#pi^{#plus}} [GeV];"   + yAxisLabel, 100, 0,      2),      "FsMassPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPim",    ";m_{#pi^{#minus}} [GeV];"  + yAxisLabel, 100, 0,      2),      "FsMassPim"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPi",     ";m_{#pi#pi} [GeV];"        + yAxisLabel, 400, 0.28,   2.28),   "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPiClas", ";m_{#pi#pi} [GeV];"        + yAxisLabel, 200, 0,      2),      "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPipP",     ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel, 400, 1,      5),      "MassPipP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPipPClas", ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel,  72, 1,      2.8),    "MassPipP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPimP",     ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel, 400, 1,      5),      "MassPimP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPimPClas", ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel,  72, 1,      2.8),    "MassPimP"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesGj",         ";cos#theta_{GJ};#phi_{GJ} [deg]",   50, -1,   +1,   72, -180, +180), "GjCosTheta", "GjPhiDeg"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesHf",         ";cos#theta_{HF};#phi_{HF} [deg]",   50, -1,   +1,   72, -180, +180), "HfCosTheta", "HfPhiDeg"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassVsHfCosTheta", ";m_{#pi#pi} [GeV];cos#theta_{HF}", 100, 0.28, 2.28, 72,   -1,   +1), "MassPiPi",   "HfCosTheta"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassVsHfPhiDeg",   ";m_{#pi#pi} [GeV];#phi_{HF}",      100, 0.28, 2.28, 72, -180, +180), "MassPiPi",   "HfPhiDeg"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.Draw("COLZ")
    canv.SaveAs(f"{hist.GetName()}.pdf")
