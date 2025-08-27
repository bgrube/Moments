#!/usr/bin/env python3


import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_BIGPHI,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  lorentzVectors,
)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
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
  # mcDataFileNames  = (f"{dataInputDirName}/MC_ps/amptools_tree_accepted_30274_31057.root")
  # mcDataFileNames  = (f"{dataInputDirName}/MC_ps/amptools_tree_thrown_30274_31057.root",     f"{dataInputDirName}/MC_rho/amptools_tree_thrown_30274_31057.root")
  # mcDataFileNames  = (f"{dataInputDirName}/MC_ps/amptools_tree_acc_thrown_30274_31057.root", f"{dataInputDirName}/MC_rho/amptools_tree_acc_thrown_30274_31057.root")
  # mcDataFileNames  = (f"{dataInputDirName}/MC_ps/amptools_tree_accepted_30274_31057.root",   f"{dataInputDirName}/MC_rho/amptools_tree_accepted_30274_31057.root")
  treeName         = "kin"
  outputDirName    = f"{tBinLabel}/McPlots_beamPolAngle_{beamPolAngle:.0f}"

  # create RDataFrame from MC data in AmpTools format and define columns
  lvs = lorentzVectors(realData = False)
  print(f"Reading MC data from tree '{treeName}' in file(s) {mcDataFileNames}")
  df = (
    ROOT.RDataFrame(treeName, mcDataFileNames)
        .Define("FsMassRecoil",   f"mass({lvs['lvRecoil']})")
        .Define("FsMassPip",      f"mass({lvs['lvPip']})")
        .Define("FsMassPim",      f"mass({lvs['lvPim']})")
        .Define("MassPiPi",       f"massPair({lvs['lvPip']}, {lvs['lvPim']})")
        .Define("MassPipP",       f"massPair({lvs['lvPip']}, {lvs['lvRecoil']})")
        .Define("MassPimP",       f"massPair({lvs['lvPim']}, {lvs['lvRecoil']})")
        .Define("MassPiPiSq",     "std::pow(MassPiPi, 2)")
        .Define("MassPipPSq",     "std::pow(MassPipP, 2)")
        .Define("MassPimPSq",     "std::pow(MassPimP, 2)")
        .Define("minusT",         f"-mandelstamT({lvs['lvTarget']}, {lvs['lvRecoil']})")
        .Define("PhiDeg",         f"bigPhi({lvs['lvRecoil']}, {lvs['lvBeam']}, {beamPolAngle}) * TMath::RadToDeg()")
        # pi+pi- system
        .Define("GjCosThetaPiPi", f"FSMath::gjcostheta({lvs['lvPip']}, {lvs['lvPim']}, {lvs['lvBeam']})")
        .Define("GjPhiDegPiPi",   f"FSMath::gjphi({lvs['lvPip']}, {lvs['lvPim']}, {lvs['lvRecoil']}, {lvs['lvBeam']}) * TMath::RadToDeg()")
        .Define("HfCosThetaPiPi", f"FSMath::helcostheta({lvs['lvPip']}, {lvs['lvPim']}, {lvs['lvRecoil']})")
        .Define("HfPhiDegPiPi",   f"FSMath::helphi({lvs['lvPim']}, {lvs['lvPip']}, {lvs['lvRecoil']}, {lvs['lvBeam']}) * TMath::RadToDeg()")
        # track momenta
        .Define("MomLabPip",      f"TLorentzVector({lvs['lvPip']}).P()")
        .Define("MomLabPim",      f"TLorentzVector({lvs['lvPim']}).P()")
        .Define("ThetaLabPip",    f"TLorentzVector({lvs['lvPip']}).Theta() * TMath::RadToDeg()")
        .Define("ThetaLabPim",    f"TLorentzVector({lvs['lvPim']}).Theta() * TMath::RadToDeg()")
        .Define("DistFdcPip",     f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvs['lvPip']})")
        .Define("DistFdcPim",     f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvs['lvPim']})")
        # .Filter("(DistFdcPip > 4) and (DistFdcPim > 4)")  # require minimum distance of tracks at FDC position [cm]
  )

  # define MC histograms
  yAxisLabel = "Events"
  hists = [
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
    df.Filter("(0.70 < MassPiPi) and (MassPiPi < 0.85)").Histo2D(ROOT.RDF.TH2DModel("hMcAnglesGjPiPiRho",  ";cos#theta_{GJ};#phi_{GJ} [deg]", 100, -1, +1, 72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi"),
    df.Filter("(0.70 < MassPiPi) and (MassPiPi < 0.85)").Histo2D(ROOT.RDF.TH2DModel("hMcAnglesHfPiPiRho",  ";cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi"),
  ]
  # create acceptance histograms for GJ and HF angles in m_pipi bins
  massPiPiRange    = (0.28, 2.28)  # [GeV]
  massPiPiNmbBins  = 50
  massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
  for binIndex in range(0, massPiPiNmbBins):
    massPiPiBinMin    = massPiPiRange[0] + binIndex * massPiPiBinWidth
    massPiPiBinMax    = massPiPiBinMin + massPiPiBinWidth
    massPiPiBinFilter = f"({massPiPiBinMin} < MassPiPi) and (MassPiPi < {massPiPiBinMax})"
    histNameSuffix    = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
    hists += [
      df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hMcAnglesGjPiPi{histNameSuffix}", ";cos#theta_{GJ};#phi_{GJ} [deg]", 100, -1, +1, 72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi"),
      df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hMcAnglesHfPiPi{histNameSuffix}", ";cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi"),
    ]

  # write MC histograms to ROOT file and generate PDF plots
  os.makedirs(outputDirName, exist_ok = True)
  outRootFileName = f"{outputDirName}/mcPlots.root"
  outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  outRootFile.cd()
  print(f"Writing histograms to '{outRootFileName}'")
  for hist in hists:
    print(f"Plotting histogram '{hist.GetName()}'")
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

outRootFile.Close()
