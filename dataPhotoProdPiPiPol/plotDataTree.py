#!/usr/bin/env python3


import os

import ROOT

from makeMomentsInputTree import (
  BEAM_POL_INFOS,
  CPP_CODE_BIGPHI,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  getDataFrameWithCorrectEventWeights,
  lorentzVectors,
)


# Alex' code to calculate helicity angles
CPP_CODE_ALEX = """
TVector3
helPiPlusVector(
	const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
	const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
	const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
	const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
	// boost all 4-vectors into the resonance rest frame
	TLorentzVector locPiPlusP4_Resonance (PxPip,    PyPip,    PzPip,    EPip);
	TLorentzVector locPiMinusP4_Resonance(PxPim,    PyPim,    PzPim,    EPim);
	TLorentzVector locRecoilP4_Resonance (PxRecoil, PyRecoil, PzRecoil, ERecoil);
	TLorentzVector locBeamP4_Resonance   (PxBeam,   PyBeam,   PzBeam,   EBeam);
	const TLorentzVector resonanceP4 = locPiPlusP4_Resonance + locPiMinusP4_Resonance;
	const TVector3 boostP3 = -resonanceP4.BoostVector();
	locPiPlusP4_Resonance.Boost (boostP3);
	locPiMinusP4_Resonance.Boost(boostP3);
	locRecoilP4_Resonance.Boost (boostP3);
	locBeamP4_Resonance.Boost   (boostP3);

	// COORDINATE SYSTEM:
	// Normal to the production plane
	const TVector3 y = (locBeamP4_Resonance.Vect().Unit().Cross(-locRecoilP4_Resonance.Vect().Unit())).Unit();
	// Helicity: z-axis opposite recoil proton in rho rest frame
	const TVector3 z = -locRecoilP4_Resonance.Vect().Unit();
	const TVector3 x = y.Cross(z).Unit();
	const TVector3 v(locPiPlusP4_Resonance.Vect() * x, locPiPlusP4_Resonance.Vect() * y, locPiPlusP4_Resonance.Vect() * z);
	return v;
}

double
helcostheta_Alex(
	const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
	const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
	const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
	const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
	const TVector3 v = helPiPlusVector(
		PxPip,    PyPip,    PzPip,    EPip,
		PxPim,    PyPim,    PzPim,    EPim,
		PxRecoil, PyRecoil, PzRecoil, ERecoil,
		PxBeam,   PyBeam,   PzBeam,   EBeam
	);
	return v.CosTheta();
}

double
helphideg_Alex(
	const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
	const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
	const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
	const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
	const TVector3 v = helPiPlusVector(
		PxPip,    PyPip,    PzPip,    EPip,
		PxPim,    PyPim,    PzPim,    EPim,
		PxRecoil, PyRecoil, PzRecoil, ERecoil,
		PxBeam,   PyBeam,   PzBeam,   EBeam
	);
	return v.Phi() * TMath::RadToDeg();
}
"""


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_BIGPHI)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)
  ROOT.gInterpreter.Declare(CPP_CODE_ALEX)

  beamPolLabel           = "PARA_0"
  # beamPolLabel           = "PARA_135"
  # beamPolLabel           = "PERP_45"
  # beamPolLabel           = "PERP_90"
  beamPolAngle           = BEAM_POL_INFOS[beamPolLabel].beamPolPhi
  tBinLabel              = "tbin_0.1_0.2"
  # tBinLabel              = "tbin_0.2_0.3"
  dataInputDirName       = f"./pipi_gluex_coh/{tBinLabel}"
  dataSigRegionFileNames = (f"{dataInputDirName}/amptools_tree_data_{beamPolLabel}_30274_31057.root", )
  dataBkgRegionFileNames = (f"{dataInputDirName}/amptools_tree_bkgnd_{beamPolLabel}_30274_31057.root", )
  # mcDataFileNames        = (f"{dataInputDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root", )
  # mcDataFileNames        = (f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root", )
  mcDataFileNames        = (f"{dataInputDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root", f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root")
  # mcDataFileNames        = (f"{dataInputDirName}/MC_ps/amptools_tree_accepted_30274_31057.root", )
  # mcDataFileNames        = (f"{dataInputDirName}/MC_rho/amptools_tree_accepted_30274_31057.root", )
  # mcDataFileNames        = (f"{dataInputDirName}/MC_ps/amptools_tree_accepted_30274_31057.root", f"{dataInputDirName}/MC_rho/amptools_tree_accepted_30274_31057.root")
  treeName               = "kin"
  outputDirName          = f"{tBinLabel}/dataPlots_{beamPolLabel}"

  # create RDataFrame from real data in AmpTools format and define columns
  lvs = lorentzVectors(realData = True)
  print(f"Reading data from tree '{treeName}' in signal file(s) {dataSigRegionFileNames} and background file(s) '{dataBkgRegionFileNames}'")
  df = (
    getDataFrameWithCorrectEventWeights(
      dataSigRegionFileNames  = dataSigRegionFileNames,
      dataBkgRegionFileNames  = dataBkgRegionFileNames,
      treeName                = treeName,
      friendSigRegionFileName = "data_sig.plot.root.weights",
      friendBkgRegionFileName = "data_bkg.plot.root.weights",
    ).Define("MassPiPi",           f"massPair({lvs['lvPip']}, {lvs['lvPim']})")
     .Define("MassPipP",           f"massPair({lvs['lvPip']}, {lvs['lvRecoil']})")
     .Define("MassPimP",           f"massPair({lvs['lvPim']}, {lvs['lvRecoil']})")
     .Define("MassPiPiSq",         "std::pow(MassPiPi, 2)")
     .Define("MassPipPSq",         "std::pow(MassPipP, 2)")
     .Define("MassPimPSq",         "std::pow(MassPimP, 2)")
     .Define("minusT",             f"-mandelstamT({lvs['lvTarget']}, {lvs['lvRecoil']})")
     .Define("PhiDeg",             f"bigPhi({lvs['lvRecoil']}, {lvs['lvBeam']}, {beamPolAngle}) * TMath::RadToDeg()")
     # pi+pi- system
     .Define("GjCosThetaPiPi",     f"FSMath::gjcostheta({lvs['lvPip']}, {lvs['lvPim']}, {lvs['lvBeam']})")
     .Define("GjPhiDegPiPi",       f"FSMath::gjphi({lvs['lvPip']}, {lvs['lvPim']}, {lvs['lvRecoil']}, {lvs['lvBeam']}) * TMath::RadToDeg()")
     .Define("HfCosThetaPiPi",     f"FSMath::helcostheta({lvs['lvPip']}, {lvs['lvPim']}, {lvs['lvRecoil']})")
     .Define("HfCosThetaPiPiDiff", f"HfCosThetaPiPi - helcostheta_Alex({lvs['lvPip']}, {lvs['lvPim']}, {lvs['lvRecoil']}, {lvs['lvBeam']})")
     .Define("HfPhiDegPiPi",       f"FSMath::helphi({lvs['lvPim']}, {lvs['lvPip']}, {lvs['lvRecoil']}, {lvs['lvBeam']}) * TMath::RadToDeg()")
     .Define("HfPhiDegPiPiDiff",   f"HfPhiDegPiPi - helphideg_Alex({lvs['lvPip']}, {lvs['lvPim']}, {lvs['lvRecoil']}, {lvs['lvBeam']})")
     # track momenta
     .Define("MomLabPip",          f"TLorentzVector({lvs['lvPip']}).P()")
     .Define("MomLabPim",          f"TLorentzVector({lvs['lvPim']}).P()")
     .Define("ThetaLabPip",        f"TLorentzVector({lvs['lvPip']}).Theta() * TMath::RadToDeg()")
     .Define("ThetaLabPim",        f"TLorentzVector({lvs['lvPim']}).Theta() * TMath::RadToDeg()")
     .Define("DistFdcPip",         f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvs['lvPip']})")
     .Define("DistFdcPim",         f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvs['lvPim']})")
    #  .Filter("(DistFdcPip > 4) and (DistFdcPim > 4)")  # require minimum distance of tracks at FDC position [cm]
  )

  # define real-data histograms applying RF-sideband subtraction
  yAxisLabel = "RF-Sideband Subtracted Combos"
  hists = [
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
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsGjPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{GJ} [deg]",    50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "GjPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsHfCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{HF}",     50,  0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "HfCosThetaPiPi", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsHfPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{HF} [deg]",    50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "HfPhiDegPiPi",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsPhiDeg",         ";m_{#pi#pi} [GeV];#Phi [deg]",         50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "PhiDeg",         "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsMinusT",         ";m_{#pi#pi} [GeV];#minus t [GeV^{2}]", 50,  0.28, 2.28,  50,    0,    1), "MassPiPi",       "minusT",         "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz1",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#plus}}^{2} [GeV^{2}]",   100, 0,  6, 100, 0.5, 16.5), "MassPiPiSq", "MassPipPSq",  "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz2",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#minus}}^{2} [GeV^{2}]",  100, 0,  6, 100, 0.5, 16.5), "MassPiPiSq", "MassPimPSq",  "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataThetaLabVsMomLabPip",      ";p_{#pi^{#plus}} [GeV];#theta_{#pi^{#plus}}^{lab} [deg]",    100, 0, 10, 100, 0,   15),   "MomLabPip",  "ThetaLabPip", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataThetaLabVsMomLabPim",      ";p_{#pi^{#minus}} [GeV];#theta_{#pi^{#minus}}^{lab} [deg]",  100, 0, 10, 100, 0,   15),   "MomLabPim",  "ThetaLabPim", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDistFdcVsMomLabPip",       ";p_{#pi^{#plus}} [GeV];#Delta r_{#pi^{#plus}}^{FDC} [cm]",   100, 0, 10, 100, 0,   20),   "MomLabPip",  "DistFdcPip",  "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDistFdcVsMomLabPim",       ";p_{#pi^{#minus}} [GeV];#Delta r_{#pi^{#minus}}^{FDC} [cm]", 100, 0, 10, 100, 0,   20),   "MomLabPim",  "DistFdcPim",  "eventWeight"),
    df.Histo3D(ROOT.RDF.TH3DModel("hDataPhiDegVsHfPhiDegPiPiVsHfCosThetaPiPi", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 25, -1, +1, 25, -180, +180, 25, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi", "PhiDeg", "eventWeight"),
  ]
  # create histograms for GJ and HF angles in m_pipi bins
  massPiPiRange = (0.28, 2.28)  # [GeV]
  massPiPiNmbBins = 50
  massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
  for binIndex in range(0, massPiPiNmbBins):
    massPiPiBinMin    = massPiPiRange[0] + binIndex * massPiPiBinWidth
    massPiPiBinMax    = massPiPiBinMin + massPiPiBinWidth
    massPiPiBinFilter = f"({massPiPiBinMin} < MassPiPi) and (MassPiPi < {massPiPiBinMax})"
    histNameSuffix    = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
    hists += [
      df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hDataAnglesGjPiPi{histNameSuffix}", ";cos#theta_{GJ};#phi_{GJ} [deg]", 100, -1, +1, 72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi"),
      df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hDataAnglesHfPiPi{histNameSuffix}", ";cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi"),
    ]

  # write real-data histograms to ROOT file and generate PDF plots
  os.makedirs(outputDirName, exist_ok = True)
  outRootFileName = f"{outputDirName}/dataPlots.root"
  outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  outRootFile.cd()
  print(f"Writing histograms to '{outRootFileName}'")
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
    histMassPiPiMc.SetLineColor    (ROOT.kBlue + 1)
    histMassPiPiMc.SetMarkerColor  (ROOT.kBlue + 1)
    histMassPiPiData.SetLineColor  (ROOT.kRed  + 1)
    histMassPiPiData.SetMarkerColor(ROOT.kRed  + 1)
    histStack.Draw("NOSTACK")
    canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
    histStack.Write()
    canv.SaveAs(f"{outputDirName}/{histStack.GetName()}.pdf")

  outRootFile.Close()
