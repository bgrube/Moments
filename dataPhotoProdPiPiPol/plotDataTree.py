#!/usr/bin/env python3


import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_BIGPHI,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  getDataFrameWithFixedEventWeights,
  BEAM_POL_INFOS,
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
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)
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

  beamPolLabel           = "PARA_0"
  # beamPolLabel           = "PARA_135"
  # beamPolLabel           = "PERP_45"
  # beamPolLabel           = "PERP_90"
  beamPolAngle           = BEAM_POL_INFOS[beamPolLabel].beamPolPhi
  # tBinLabel              = "tbin_0.1_0.2"
  tBinLabel              = "tbin_0.2_0.3"
  dataBaseDirName        = f"./pipi_gluex_coh/{tBinLabel}"
  dataSigRegionFileNames = (f"{dataBaseDirName}/amptools_tree_data_{beamPolLabel}_30274_31057.root", )
  dataBkgRegionFileNames = (f"{dataBaseDirName}/amptools_tree_bkgnd_{beamPolLabel}_30274_31057.root", )
  # mcDataFileNames        = (f"{dataBaseDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root", )
  # mcDataFileNames        = (f"{dataBaseDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root", )
  # mcDataFileNames        = (f"{dataBaseDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root", f"{dataBaseDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root")
  # mcDataFileNames        = (f"{dataBaseDirName}/MC_ps/amptools_tree_accepted_30274_31057.root", )
  # mcDataFileNames        = (f"{dataBaseDirName}/MC_rho/amptools_tree_accepted_30274_31057.root", )
  mcDataFileNames        = (f"{dataBaseDirName}/MC_ps/amptools_tree_accepted_30274_31057.root", f"{dataBaseDirName}/MC_rho/amptools_tree_accepted_30274_31057.root")
  treeName               = "kin"
  outputDirName          = f"{tBinLabel}/{beamPolLabel}"

  # read in real data in AmpTools format and plot RF-sideband subtracted distributions
  lvBeam   = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"
  lvTarget = "0,                0,                0,                0.93827208816"    # proton at rest in lab frame
  lvRecoil = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"
  lvPip    = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"
  lvPim    = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"
  print(f"Reading data from tree '{treeName}' in signal file(s) {dataSigRegionFileNames} and background file(s) '{dataBkgRegionFileNames}'")
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
     # track momenta
     .Define("MomLabPip",          f"TLorentzVector({lvPip}).P()")
     .Define("MomLabPim",          f"TLorentzVector({lvPim}).P()")
     .Define("ThetaLabPip",        f"TLorentzVector({lvPip}).Theta() * TMath::RadToDeg()")
     .Define("ThetaLabPim",        f"TLorentzVector({lvPim}).Theta() * TMath::RadToDeg()")
     .Define("DistFdcPip",         f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvPip})")
     .Define("DistFdcPim",         f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvPim})")
  )
  yAxisLabel = "RF-Sideband Subtracted Combos"
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hDataEbeam",              ";E_{beam} [GeV];"                     + yAxisLabel, 100, 8,      9),    "E_Beam",      "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPi",           ";m_{#pi#pi} [GeV];"                   + yAxisLabel, 400, 0.28,   2.28), "MassPiPi",    "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPiPwa" ,       ";m_{#pi#pi} [GeV];"                   + yAxisLabel,  50, 0.28,   2.28), "MassPiPi",    "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPipP",           ";m_{p#pi^{#plus}} [GeV];"             + yAxisLabel, 400, 1,      5),    "MassPipP",    "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPimP",           ";m_{p#pi^{#minus}} [GeV];"            + yAxisLabel, 400, 1,      5),    "MassPimP",    "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMinusT",             ";#minus t [GeV^{2}];"                 + yAxisLabel, 100, 0,      1),    "minusT",      "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMomLabPip",          ";p_{#pi^{#plus}} [GeV];"              + yAxisLabel, 100, 0,     10),    "MomLabPip",   "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMomLabPim",          ";p_{#pi^{#minus}} [GeV];"             + yAxisLabel, 100, 0,     10),    "MomLabPim",   "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataThetaLabPip",        ";#theta_{#pi^{#plus}}^{lab} [deg];"   + yAxisLabel, 100, 0,     30),    "ThetaLabPip", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataThetaLabPim",        ";#theta_{#pi^{#minus}}^{lab} [deg];"  + yAxisLabel, 100, 0,     30),    "ThetaLabPim", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataDistFdcPip",         ";#Delta r_{#pi^{#plus}}^{FDC} [cm];"  + yAxisLabel, 100, 0,     40),    "DistFdcPip",  "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataDistFdcPim",         ";#Delta r_{#pi^{#minus}}^{FDC} [cm];" + yAxisLabel, 100, 0,     40),    "DistFdcPim",  "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataHfCosThetaPiPiDiff", ";#Delta cos#theta_{HF}",  1000, -3e-13, +3e-13), "HfCosThetaPiPiDiff", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataHfPhiDegPiPiDiff",   ";#Delta #phi_{HF} [deg]", 1000, -1e-11, +1e-11), "HfPhiDegPiPiDiff",   "eventWeight"),
    #
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesGjPiPi",             ";cos#theta_{GJ};#phi_{GJ} [deg]",     100, -1,   +1,     72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesHfPiPi",             ";cos#theta_{HF};#phi_{HF} [deg]",     100, -1,   +1,     72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsGjCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{GJ}",     50,  0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "GjCosThetaPiPi", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsGjPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{GJ}",          50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "GjPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsHfCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{HF}",     50,  0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "HfCosThetaPiPi", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsHfPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{HF}",          50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "HfPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsPhiDeg",         ";m_{#pi#pi} [GeV];#Phi",               50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "PhiDeg",         "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsMinusT",         ";m_{#pi#pi} [GeV];#minus t [GeV^{2}]", 50,  0.28, 2.28,  50,    0,    1), "MassPiPi",       "minusT",         "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz1",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#plus}}^{2} [GeV^{2}]",   100, 0,  6, 100, 0.5, 16.5), "MassPiPiSq", "MassPipPSq",  "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz2",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#minus}}^{2} [GeV^{2}]",  100, 0,  6, 100, 0.5, 16.5), "MassPiPiSq", "MassPimPSq",  "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataThetaLabVsMomLabPip",      ";p_{#pi^{#plus}} [GeV];#theta_{#pi^{#plus}}^{lab} [deg]",    100, 0, 10, 100, 0,   15),   "MomLabPip",  "ThetaLabPip", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataThetaLabVsMomLabPim",      ";p_{#pi^{#minus}} [GeV];#theta_{#pi^{#minus}}^{lab} [deg]",  100, 0, 10, 100, 0,   15),   "MomLabPim",  "ThetaLabPim", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDistFdcVsMomLabPip",       ";p_{#pi^{#plus}} [GeV];#Delta r_{#pi^{#plus}}^{FDC} [cm]",   100, 0, 10, 100, 0,   20),   "MomLabPip",  "DistFdcPip",  "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDistFdcVsMomLabPim",       ";p_{#pi^{#minus}} [GeV];#Delta r_{#pi^{#minus}}^{FDC} [cm]", 100, 0, 10, 100, 0,   20),   "MomLabPim",  "DistFdcPim",  "eventWeight"),
    df.Histo3D(ROOT.RDF.TH3DModel("hDataPhiDegVsHfPhiDegPiPiVsHfCosThetaPiPi", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 25, -1, +1, 25, -180, +180, 25, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi", "PhiDeg", "eventWeight"),
  )
  os.makedirs(outputDirName, exist_ok = True)
  outRootFileName = f"{outputDirName}/dataPlots.root"
  outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  print(f"Writing histograms to '{outRootFileName}'")
  outRootFile.cd()
  # write real-data histograms to ROOT file and generate PDF plots
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
    hist.Write()
    canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")

  if True:
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
    histStack.Write()
    canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
    canv.SaveAs(f"{outputDirName}/{histStack.GetName()}.pdf")

  outRootFile.Close()
