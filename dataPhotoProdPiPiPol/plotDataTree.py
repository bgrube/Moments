#!/usr/bin/env python3


import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_BIGPHI,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  getDataFrameWithFixedEventWeights,
)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C")
  ROOT.gROOT.LoadMacro("../rootlogon.C")
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_BIGPHI)
  # Alex' code to calculate helicity angles
  CPP_CODE = """
	TVector3
	helPiPlusVector(
		const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
		const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
		const double PxProton, const double PyProton, const double PzProton, const double EProton,
		const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
	) {
		// boost all 4-vectors into the resonance rest frame
		TLorentzVector locPiPlusP4_Resonance (PxPip,    PyPip,    PzPip,    EPip);
		TLorentzVector locPiMinusP4_Resonance(PxPim,    PyPim,    PzPim,    EPim);
		TLorentzVector locProtonP4_Resonance (PxProton, PyProton, PzProton, EProton);
		TLorentzVector locBeamP4_Resonance   (PxBeam,   PyBeam,   PzBeam,   EBeam);
		const TLorentzVector resonanceP4 = locPiPlusP4_Resonance + locPiMinusP4_Resonance;
		const TVector3 boostP3 = -resonanceP4.BoostVector();
		locPiPlusP4_Resonance.Boost (boostP3);
		locPiMinusP4_Resonance.Boost(boostP3);
		locProtonP4_Resonance.Boost (boostP3);
		locBeamP4_Resonance.Boost   (boostP3);

		// COORDINATE SYSTEM:
		// Normal to the production plane
		const TVector3 y = (locBeamP4_Resonance.Vect().Unit().Cross(-locProtonP4_Resonance.Vect().Unit())).Unit();
		// Helicity: z-axis opposite proton in rho rest frame
		const TVector3 z = -locProtonP4_Resonance.Vect().Unit();
		const TVector3 x = y.Cross(z).Unit();
		const TVector3 v(locPiPlusP4_Resonance.Vect() * x, locPiPlusP4_Resonance.Vect() * y, locPiPlusP4_Resonance.Vect() * z);
		return v;
	}

	double
	helcostheta_Alex(
		const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
		const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
		const double PxProton, const double PyProton, const double PzProton, const double EProton,
		const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
	) {
		const TVector3 v = helPiPlusVector(
			PxPip,    PyPip,    PzPip,    EPip,
			PxPim,    PyPim,    PzPim,    EPim,
			PxProton, PyProton, PzProton, EProton,
			PxBeam,   PyBeam,   PzBeam,   EBeam
		);
		return v.CosTheta();
	}

	double
	helphideg_Alex(
		const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
		const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
		const double PxProton, const double PyProton, const double PzProton, const double EProton,
		const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
	) {
		const TVector3 v = helPiPlusVector(
			PxPip,    PyPip,    PzPip,    EPip,
			PxPim,    PyPim,    PzPim,    EPim,
			PxProton, PyProton, PzProton, EProton,
			PxBeam,   PyBeam,   PzBeam,   EBeam
		);
		return v.Phi() * TMath::RadToDeg();
	}
  """
  ROOT.gInterpreter.Declare(CPP_CODE)

  # beamPolAngle           = 0.0
  # dataSigRegionFileNames = ("./pipi_gluex_coh/amptools_tree_data_PARA_0_30274_31057.root", )
  # dataBkgRegionFileNames = ("./pipi_gluex_coh/amptools_tree_bkgnd_PARA_0_30274_31057.root", )
  # dataSigRegionFileNames = ("./pipi_gluex_coh/ver70/amptools_tree_data_PARA_0_30274_31057.root", )
  # dataBkgRegionFileNames = ("./pipi_gluex_coh/ver70/amptools_tree_bkgnd_PARA_0_30274_31057.root", )
  # beamPolAngle           = 135.0
  # dataSigRegionFileNames = ("./pipi_gluex_coh/ver70/amptools_tree_data_PARA_135_30274_31057.root", )
  # dataBkgRegionFileNames = ("./pipi_gluex_coh/ver70/amptools_tree_bkgnd_PARA_135_30274_31057.root", )
  # beamPolAngle           = 45.0
  # dataSigRegionFileNames = ("./pipi_gluex_coh/ver70/amptools_tree_data_PERP_45_30274_31057.root", )
  # dataBkgRegionFileNames = ("./pipi_gluex_coh/ver70/amptools_tree_bkgnd_PERP_45_30274_31057.root", )
  beamPolAngle           = 90.0
  dataSigRegionFileNames = ("./pipi_gluex_coh/ver70/amptools_tree_data_PERP_90_30274_31057.root", )
  dataBkgRegionFileNames = ("./pipi_gluex_coh/ver70/amptools_tree_bkgnd_PERP_90_30274_31057.root", )
  # mcDataFileNames        = ("./pipi_gluex_coh/amptools_tree_accepted_30274_31057.root", )
  # mcDataFileNames        = ("./pipi_gluex_coh/MC_100M/amptools_tree_accepted_30274_31057.root", )
  # mcDataFileNames        = ("./pipi_gluex_coh/MC_10M_rho_t/amptools_tree_accepted_30274_31057.root", )
  mcDataFileNames        = ("./pipi_gluex_coh/MC_100M/amptools_tree_accepted_30274_31057.root", "./pipi_gluex_coh/MC_10M_rho_t/amptools_tree_accepted_30274_31057.root")
  treeName               = "kin"

  # read in real data in AmpTools format and plot RF-sideband subtracted distributions
  lvBeam   = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"
  lvTarget = "0,                0,                0,                0.93827208816"    # proton at rest in lab frame
  lvRecoil = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"
  lvPip    = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"
  lvPim    = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"
  print(f"Reading data from tree '{treeName}' in signal file(s) '{dataSigRegionFileNames}' and background file(s) '{dataBkgRegionFileNames}'")
  df = (
    getDataFrameWithFixedEventWeights(
      dataSigRegionFileNames  = dataSigRegionFileNames,
      dataBkgRegionFileNames  = dataBkgRegionFileNames,
      treeName                = treeName,
      friendSigRegionFileName = "data_sig.plot.root.weights",
      friendBkgRegionFileName = "data_bkg.plot.root.weights",
    ).Define("MassPiPi",           f"massPair({lvPip}, {lvPim})")
     .Define("MassPipP",           f"massPair({lvPip}, {lvRecoil})")
     .Define("MassPimP",           f"massPair({lvPim}, {lvRecoil})")
     .Define("MassPiPiSq",         "std::pow(MassPiPi, 2)")
     .Define("MassPipPSq",         "std::pow(MassPipP, 2)")
     .Define("MassPimPSq",         "std::pow(MassPimP, 2)")
     .Define("minusT",             f"-mandelstamT({lvTarget}, {lvRecoil})")
     .Define("PhiDeg",             f"bigPhi({lvRecoil}, {lvBeam}, {beamPolAngle}) * TMath::RadToDeg()")
     # pi+pi- system
     .Define("GjCosThetaPiPi",     f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})")
     .Define("GjPhiDegPiPi",       f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam}) * TMath::RadToDeg()")
     .Define("HfCosThetaPiPi",     f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})")
     .Define("HfCosThetaPiPiDiff", f"HfCosThetaPiPi - helcostheta_Alex({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})")
     .Define("HfPhiDegPiPi",       f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam}) * TMath::RadToDeg()")
     .Define("HfPhiDegPiPiDiff",   f"HfPhiDegPiPi - helphideg_Alex({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})")
  )
  yAxisLabel = "RF-Sideband Subtracted Combos"
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hDataEbeam",              ";E_{beam} [GeV];"          + yAxisLabel, 100, 8,        9),    "E_Beam",             "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPi",           ";m_{#pi#pi} [GeV];"        + yAxisLabel, 400, 0.28,   2.28),   "MassPiPi",           "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPiPwa" ,       ";m_{#pi#pi} [GeV];"        + yAxisLabel,  50, 0.28,   2.28),   "MassPiPi",           "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPipP",           ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel, 400, 1,      5),      "MassPipP",           "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPimP",           ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel, 400, 1,      5),      "MassPimP",           "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMinusT",             ";#minus t [GeV^{2}];"      + yAxisLabel, 100, 0,      1),      "minusT",             "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataHfCosThetaPiPiDiff", ";#Delta cos#theta_{HF}",                1000, -3e-13, +3e-13), "HfCosThetaPiPiDiff", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataHfPhiDegPiPiDiff",   ";#Delta #phi_{HF} [deg]",               1000, -1e-11, +1e-11), "HfPhiDegPiPiDiff",   "eventWeight"),
    # pi+pi- system
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesGjPiPi",             ";cos#theta_{GJ};#phi_{GJ} [deg]",     100, -1,   +1,     72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesHfPiPi",             ";cos#theta_{HF};#phi_{HF} [deg]",     100, -1,   +1,     72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsGjCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{GJ}",     50,  0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "GjCosThetaPiPi", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsGjPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{GJ}",          50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "GjPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsHfCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{HF}",     50,  0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "HfCosThetaPiPi", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsHfPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{HF}",          50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "HfPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsPhiDeg",         ";m_{#pi#pi} [GeV];#Phi",               50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "PhiDeg",         "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsMinusT",         ";m_{#pi#pi} [GeV];#minus t [GeV^{2}]", 50,  0.28, 2.28,  50,    0,    1), "MassPiPi",       "minusT",         "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz1",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#plus}}^{2} [GeV^{2}]",  100, 0, 6, 100, 0.5, 16.5), "MassPiPiSq", "MassPipPSq", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz2",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#minus}}^{2} [GeV^{2}]", 100, 0, 6, 100, 0.5, 16.5), "MassPiPiSq", "MassPimPSq", "eventWeight"),
    df.Histo3D(ROOT.RDF.TH3DModel("hDataPhiDegVsHfPhiDegPiPiVsHfCosThetaPiPi", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 25, -1, +1, 25, -180, +180, 25, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi", "PhiDeg", "eventWeight"),
  )
  for hist in hists:
    print(f"Generating histogram '{hist.GetName()}'")
    canv = ROOT.TCanvas()
    if "TH2" in hist.ClassName() and str(hist.GetName()).startswith("hDataMass"):
      canv.SetLogz(1)
    hist.SetMinimum(0)
    if "TH3" in hist.ClassName():
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
    else:
      hist.Draw("COLZ")
    canv.SaveAs(f"{hist.GetName()}.pdf")

  # overlay pipi mass distributions from data and accepted phase-space MC
  lvPip = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # not clear whether correct index is 1 or 2
  lvPim = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # not clear whether correct index is 1 or 2
  dfMc = (
    ROOT.RDataFrame(treeName, mcDataFileNames)
        .Define("MassPiPi", f"massPair({lvPip}, {lvPim})")
  )
  histMassPiPiMc   = dfMc.Histo1D(ROOT.RDF.TH1DModel("Accepted Phase-Space MC", "", 50, 0.28, 2.28), "MassPiPi")
  histMassPiPiData = df.Histo1D  (ROOT.RDF.TH1DModel("RF-subtracted Data",      "", 50, 0.28, 2.28), "MassPiPi", "eventWeight")
  canv = ROOT.TCanvas()
  histStack = ROOT.THStack("hMassPiPiDataAndMc", ";m_{#pi#pi} [GeV];Events / 40 MeV")
  histStack.Add(histMassPiPiMc.GetValue())
  histStack.Add(histMassPiPiData.GetValue())
  histMassPiPiMc.SetLineColor  (ROOT.kBlue + 1)
  histMassPiPiData.SetLineColor(ROOT.kRed + 1)
  histMassPiPiMc.SetMarkerColor  (ROOT.kBlue + 1)
  histMassPiPiData.SetMarkerColor(ROOT.kRed + 1)
  histStack.Draw("NOSTACK")
  canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
  canv.SaveAs(f"{histStack.GetName()}.pdf")
