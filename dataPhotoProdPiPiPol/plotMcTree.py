#!/usr/bin/env python3


import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_BIGPHI,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C")
  ROOT.gROOT.LoadMacro("../rootlogon.C")
  ROOT.gStyle.SetOptStat("i")
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_BIGPHI)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)
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

  beamPolAngle     = 0.0
  tBinLabel        = "tbin_0.1_0.2"
  # tBinLabel        = "tbin_0.2_0.3"
  dataInputDirName = f"./pipi_gluex_coh/{tBinLabel}"
  # mcDataFileNames  = (f"{dataInputDirName}/MC_100M/amptools_tree_thrown_30274_31057.root", )
  # mcDataFileNames  = (f"{dataInputDirName}/MC_100M/amptools_tree_acc_thrown_30274_31057_noMcut.root", )
  mcDataFileNames  = (f"{dataInputDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root", )
  # mcDataFileNames  = (f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_thrown_30274_31057.root", )
  # mcDataFileNames  = (f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_acc_thrown_30274_31057_notcut.root", )
  # mcDataFileNames  = (f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root", )
  # mcDataFileNames  = (f"{dataInputDirName}/MC_100M/amptools_tree_thrown_30274_31057.root",            f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_thrown_30274_31057.root")
  # mcDataFileNames  = (f"{dataInputDirName}/MC_100M/amptools_tree_acc_thrown_30274_31057_noMcut.root", f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_acc_thrown_30274_31057_notcut.root")
  # mcDataFileNames  = (f"{dataInputDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root",   f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root")
  # mcDataFileNames  = (f"{dataInputDirName}/MC_ps/amptools_tree_thrown_30274_31057.root",     f"{dataInputDirName}/MC_rho/amptools_tree_thrown_30274_31057.root")
  # mcDataFileNames  = (f"{dataInputDirName}/MC_ps/amptools_tree_acc_thrown_30274_31057.root", f"{dataInputDirName}/MC_rho/amptools_tree_acc_thrown_30274_31057.root")
  # mcDataFileNames  = (f"{dataInputDirName}/MC_ps/amptools_tree_accepted_30274_31057.root",   f"{dataInputDirName}/MC_rho/amptools_tree_accepted_30274_31057.root")
  treeName         = "kin"
  outputDirName    = f"{tBinLabel}/Mc_beamPolAngle_{beamPolAngle:.0f}"

  # read MC data in AmpTools format and plot distributions
  lvBeam   = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"
  lvTarget = "0,                0,                0,                0.93827208816"    # proton at rest in lab frame
  lvRecoil = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"
  lvPip    = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # not clear whether correct index is 1 or 2
  lvPim    = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # not clear whether correct index is 1 or 2
  print(f"Reading MC data from tree '{treeName}' in file(s) {mcDataFileNames}")
  df = (
    ROOT.RDataFrame(treeName, mcDataFileNames)
        .Define("FsMassRecoil",   f"mass({lvRecoil})")
        .Define("FsMassPip",      f"mass({lvPip})")
        .Define("FsMassPim",      f"mass({lvPim})")
        .Define("MassPiPi",       f"massPair({lvPip}, {lvPim})")
        .Define("MassPipP",       f"massPair({lvPip}, {lvRecoil})")
        .Define("MassPimP",       f"massPair({lvPim}, {lvRecoil})")
        .Define("MassPiPiSq",     "std::pow(MassPiPi, 2)")
        .Define("MassPipPSq",     "std::pow(MassPipP, 2)")
        .Define("MassPimPSq",     "std::pow(MassPimP, 2)")
        .Define("minusT",         f"-mandelstamT({lvTarget}, {lvRecoil})")
        .Define("PhiDeg",         f"bigPhi({lvRecoil}, {lvBeam}, {beamPolAngle}) * TMath::RadToDeg()")
        # pi+pi- system
        .Define("GjCosThetaPiPi", f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})")
        .Define("GjPhiDegPiPi",   f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam}) * TMath::RadToDeg()")
        .Define("HfCosThetaPiPi", f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})")
        .Define("HfPhiDegPiPi",   f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam}) * TMath::RadToDeg()")
        # track momenta
        .Define("MomLabPip",      f"TLorentzVector({lvPip}).P()")
        .Define("MomLabPim",      f"TLorentzVector({lvPim}).P()")
        .Define("ThetaLabPip",    f"TLorentzVector({lvPip}).Theta() * TMath::RadToDeg()")
        .Define("ThetaLabPim",    f"TLorentzVector({lvPim}).Theta() * TMath::RadToDeg()")
        .Define("DistFdcPip",     f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvPip})")
        .Define("DistFdcPim",     f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvPim})")
        # .Filter("(DistFdcPip > 4) and (DistFdcPim > 4)")  # require minimum distance of tracks at FDC position [cm]
  )
  yAxisLabel = "Events"
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassRecoil", ";m_{Recoil} [GeV];"                   + yAxisLabel, 100, 0,      2),    "FsMassRecoil"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPip",    ";m_{#pi^{#plus}} [GeV];"              + yAxisLabel, 100, 0,      2),    "FsMassPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPim",    ";m_{#pi^{#minus}} [GeV];"             + yAxisLabel, 100, 0,      2),    "FsMassPim"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPi",     ";m_{#pi#pi} [GeV];"                   + yAxisLabel, 400, 0.28,   2.28), "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPiPiPwa",  ";m_{#pi#pi} [GeV];"                   + yAxisLabel,  50, 0.28,   2.28), "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPipP",     ";m_{p#pi^{#plus}} [GeV];"             + yAxisLabel, 400, 1,      5),    "MassPipP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMassPimP",     ";m_{p#pi^{#minus}} [GeV];"            + yAxisLabel, 400, 1,      5),    "MassPimP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMinusT",       ";#minus t [GeV^{2}];"                 + yAxisLabel, 100, 0,      1),    "minusT"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMomLabPip",    ";p_{#pi^{#plus}} [GeV];"              + yAxisLabel, 100, 0,     10),    "MomLabPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMomLabPim",    ";p_{#pi^{#minus}} [GeV];"             + yAxisLabel, 100, 0,     10),    "MomLabPim"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcThetaLabPip",  ";#theta_{#pi^{#plus}}^{lab} [deg];"   + yAxisLabel, 100, 0,     30),    "ThetaLabPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcThetaLabPim",  ";#theta_{#pi^{#minus}}^{lab} [deg];"  + yAxisLabel, 100, 0,     30),    "ThetaLabPim"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcDistFdcPip",   ";#Delta r_{#pi^{#plus}}^{FDC} [cm];"  + yAxisLabel, 100, 0,     40),    "DistFdcPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcDistFdcPim",   ";#Delta r_{#pi^{#minus}}^{FDC} [cm];" + yAxisLabel, 100, 0,     40),    "DistFdcPim"),
    df.Filter("(0.70 < MassPiPi) and (MassPiPi < 0.85)").Histo1D(ROOT.RDF.TH1DModel("hMcMinusTRho", ";#minus t [GeV^{2}];" + yAxisLabel, 100, 0, 1), "minusT"),
    # pi+pi- system
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesGjPiPi",             ";cos#theta_{GJ};#phi_{GJ} [deg]",     100, -1,   +1,    72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcAnglesHfPiPi",             ";cos#theta_{HF};#phi_{HF} [deg]",     100, -1,   +1,    72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsGjCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{GJ}",     50, 0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "GjCosThetaPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsGjPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{GJ}",          50, 0.28, 2.28,  72, -180, +180), "MassPiPi",       "GjPhiDegPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsHfCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{HF}",     50, 0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "HfCosThetaPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsHfPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{HF}",          50, 0.28, 2.28,  72, -180, +180), "MassPiPi",       "HfPhiDegPiPi"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsPhiDeg",         ";m_{#pi#pi} [GeV];#Phi",               50, 0.28, 2.28,  72, -180, +180), "MassPiPi",       "PhiDeg"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcMassPiPiVsMinusT",         ";m_{#pi#pi} [GeV];#minus t [GeV^{2}]", 50,  0.28, 2.28, 50,    0,    1), "MassPiPi",       "minusT"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcDalitz1",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#plus}}^{2} [GeV^{2}]",   100, 0, 6,  100, 0.5, 16.5), "MassPiPiSq", "MassPipPSq"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcDalitz2",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#minus}}^{2} [GeV^{2}]",  100, 0, 6,  100, 0.5, 16.5), "MassPiPiSq", "MassPimPSq"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcThetaLabVsMomLabPip",      ";p_{#pi^{#plus}} [GeV];#theta_{#pi^{#plus}}^{lab} [deg]",    100, 0, 10, 100, 0,   15),   "MomLabPip",  "ThetaLabPip"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcThetaLabVsMomLabPim",      ";p_{#pi^{#minus}} [GeV];#theta_{#pi^{#minus}}^{lab} [deg]",  100, 0, 10, 100, 0,   15),   "MomLabPim",  "ThetaLabPim"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcDistFdcVsMomLabPip",       ";p_{#pi^{#plus}} [GeV];#Delta r_{#pi^{#plus}}^{FDC} [cm]",   100, 0, 10, 100, 0,   20),   "MomLabPip",  "DistFdcPip"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcDistFdcVsMomLabPim",       ";p_{#pi^{#minus}} [GeV];#Delta r_{#pi^{#minus}}^{FDC} [cm]", 100, 0, 10, 100, 0,   20),   "MomLabPim",  "DistFdcPim"),
    df.Histo3D(ROOT.RDF.TH3DModel("hMcPhiDegVsHfPhiDegPiPiVsHfCosThetaPiPi", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 25, -1, +1, 25, -180, +180, 25, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi", "PhiDeg"),
    df.Filter("(0.70 < MassPiPi) and (MassPiPi < 0.85)").Histo2D(ROOT.RDF.TH2DModel("hMcAnglesGjPiPiRho", ";cos#theta_{GJ};#phi_{GJ} [deg]", 100, -1, +1, 72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi"),
    df.Filter("(0.70 < MassPiPi) and (MassPiPi < 0.85)").Histo2D(ROOT.RDF.TH2DModel("hMcAnglesHfPiPiRho", ";cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi"),
  )
  # write MC histograms to ROOT file and generate PDF plots
  os.makedirs(outputDirName, exist_ok = True)
  outRootFileName = f"{outputDirName}/mcPlots.root"
  outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  print(f"Writing histograms to '{outRootFileName}'")
  outRootFile.cd()
  for hist in hists:
    print(f"Generating histogram '{hist.GetName()}'")
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    if "TH2" in hist.ClassName() and str(hist.GetName()) =="hMcMassPiPiVsMinusT":
      canv.SetLogz(1)
    if "TH3" in hist.ClassName():
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
    else:
      hist.Draw("COLZ")
    hist.Write()
    canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")
