#!/usr/bin/env python3


import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_MASSPAIR,
  CPP_CODE_BIGPHI,
)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_BIGPHI)
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

  # data for lowest t bin [0.1, 0.2] GeV^2
  beamPolAngle = 0.0
  # mcDataFileName = "./pipi_gluex_coh/amptools_tree_thrown_30274_31057.root"
  mcDataFileName = "./pipi_gluex_coh/amptools_tree_accepted_30274_31057.root"
  treeName = "kin"

  # read MC data in AmpTools formatand plot distributions
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
        .Define("GjTheta",       "std::acos(GjCosTheta)")
        .Define("GjPhi",        f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})")
        .Define("GjPhiDeg",      "GjPhi * TMath::RadToDeg()")
        .Define("HfCosTheta",   f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})")
        .Define("HfTheta",       "std::acos(HfCosTheta)")
        .Define("HfPhi",        f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam})")
        .Define("HfPhiDeg",      "HfPhi * TMath::RadToDeg()")
        .Define("Phi",          f"bigPhi({lvRecoil}, {lvBeam}, {beamPolAngle})")
        .Define("PhiDeg",        "Phi * TMath::RadToDeg()")
  )
  yAxisLabel = "Events"
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassRecoil", ";m_{Recoil} [GeV];"        + yAxisLabel, 100, 0,      2),      "FsMassRecoil"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPip",    ";m_{#pi^{#plus}} [GeV];"   + yAxisLabel, 100, 0,      2),      "FsMassPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPim",    ";m_{#pi^{#minus}} [GeV];"  + yAxisLabel, 100, 0,      2),      "FsMassPim"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPi",     ";m_{#pi#pi} [GeV];"        + yAxisLabel, 400, 0.28,   2.28),   "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPiPwa",  ";m_{#pi#pi} [GeV];"        + yAxisLabel,  50, 0.28,   2.28),   "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPipP",     ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel, 400, 1,      5),      "MassPipP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPimP",     ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel, 400, 1,      5),      "MassPimP"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesGj",         ";cos#theta_{GJ};#phi_{GJ} [deg]",  100, -1,   +1,    72, -180, +180), "GjCosTheta", "GjPhiDeg"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesHf",         ";cos#theta_{HF};#phi_{HF} [deg]",  100, -1,   +1,    72, -180, +180), "HfCosTheta", "HfPhiDeg"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassVsHfCosTheta", ";m_{#pi#pi} [GeV];cos#theta_{HF}",  50, 0.28, 2.28, 100,   -1,   +1), "MassPiPi",   "HfCosTheta"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassVsHfPhiDeg",   ";m_{#pi#pi} [GeV];#phi_{HF}",       50, 0.28, 2.28,  72, -180, +180), "MassPiPi",   "HfPhiDeg"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassVsPhiDeg",     ";m_{#pi#pi} [GeV];#Phi",            50, 0.28, 2.28,  72, -180, +180), "MassPiPi",   "PhiDeg"),
    df.Histo3D(ROOT.RDF.TH3DModel("hMcPhiDegVsHfPhiDegVsHfCosTheta", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 25, -1, +1, 25, -180, +180, 25, -180, +180), "HfCosTheta", "HfPhiDeg", "PhiDeg"),
  )
  for hist in hists:
    print(f"Generating histogram '{hist.GetName()}'")
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    if "TH3" in hist.ClassName():
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
    else:
      hist.Draw("COLZ")
    canv.SaveAs(f"{hist.GetName()}.pdf")