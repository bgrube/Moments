#!/usr/bin/env python3


import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_MASSPAIR,
  CPP_CODE_BIGPHI,
)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C")
  ROOT.gROOT.LoadMacro("../rootlogon.C")
  ROOT.gStyle.SetOptStat("i")
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

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
  CPP_CODE = """
	double
	mandelstamT(
		const double Px1, const double Py1, const double Pz1, const double E1,
		const double Px2, const double Py2, const double Pz2, const double E2
	) {
		const TLorentzVector p1(Px1, Py1, Pz1, E1);
		const TLorentzVector p2(Px2, Py2, Pz2, E2);
		return (p1 - p2).M2();
	}
  """
  ROOT.gInterpreter.Declare(CPP_CODE)

  # data for lowest t bin [0.1, 0.2] GeV^2
  beamPolAngle = 0.0
  # mcDataFileName = "./pipi_gluex_coh/amptools_tree_thrown_30274_31057.root"
  # mcDataFileName = "./pipi_gluex_coh/amptools_tree_accepted_30274_31057.root"
  # mcDataFileName = "./MC_100M/amptools_tree_thrown_30274_31057.root"
  # mcDataFileName = "./MC_100M/amptools_tree_acc_thrown_30274_31057.root"
  mcDataFileName = "./MC_100M/amptools_tree_accepted_30274_31057.root"
  treeName = "kin"

  # read MC data in AmpTools format and plot distributions
  lvBeam   = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"
  lvTarget = "0,                0,                0,                0.93827208816"    # proton at rest in lab frame
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
        .Define("minusT",       f"-mandelstamT({lvTarget}, {lvRecoil})")
        .Define("PhiDeg",       f"bigPhi({lvRecoil}, {lvBeam}, {beamPolAngle}) * TMath::RadToDeg()")
        # pi+pi- system
        .Define("GjCosThetaPiPi", f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})")
        .Define("GjPhiDegPiPi",   f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam}) * TMath::RadToDeg()")
        .Define("HfCosThetaPiPi", f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})")
        .Define("HfPhiDegPiPi",   f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam}) * TMath::RadToDeg()")
  )
  yAxisLabel = "Events"
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassRecoil", ";m_{Recoil} [GeV];"        + yAxisLabel, 100, 0,      2),    "FsMassRecoil"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPip",    ";m_{#pi^{#plus}} [GeV];"   + yAxisLabel, 100, 0,      2),    "FsMassPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPim",    ";m_{#pi^{#minus}} [GeV];"  + yAxisLabel, 100, 0,      2),    "FsMassPim"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPi",     ";m_{#pi#pi} [GeV];"        + yAxisLabel, 400, 0.28,   2.28), "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPiPwa",  ";m_{#pi#pi} [GeV];"        + yAxisLabel,  50, 0.28,   2.28), "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPipP",     ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel, 400, 1,      5),    "MassPipP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPimP",     ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel, 400, 1,      5),    "MassPimP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMinusT",       ";#minus t [GeV^{2}];"      + yAxisLabel, 100, 0,      1),    "minusT"),
    # pi+pi- system
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesGjPiPi",             ";cos#theta_{GJ};#phi_{GJ} [deg]",  100, -1,   +1,    72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesHfPiPi",             ";cos#theta_{HF};#phi_{HF} [deg]",  100, -1,   +1,    72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsHfCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{HF}",  50, 0.28, 2.28, 100,   -1,   +1), "MassPiPi",   "HfCosThetaPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsHfPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{HF}",       50, 0.28, 2.28,  72, -180, +180), "MassPiPi",   "HfPhiDegPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsPhiDeg",         ";m_{#pi#pi} [GeV];#Phi",            50, 0.28, 2.28,  72, -180, +180), "MassPiPi",   "PhiDeg"),
    df.Histo3D(ROOT.RDF.TH3DModel("hMcPhiDegVsHfPhiDegPiPiVsHfCosThetaPiPi", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 25, -1, +1, 25, -180, +180, 25, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi", "PhiDeg"),
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
