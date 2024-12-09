#!/usr/bin/env python3


from __future__ import annotations

import ctypes
import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_MAKEPAIR,
  defineAngleFormulas,
  lorentzVectors,
)

def convertGraphToHist(
  graph:     ROOT.TGraphErrors,
  binning:   tuple[int, float, float],
  histName:  str,
  histTitle: str = "",
) -> ROOT.TH1D:
  """Converts `TGraphErrors` to `TH1D` assuming equidistant binning"""
  hist = ROOT.TH1D(histName, histTitle, *binning)
  for pointIndex in range(graph.GetN()):
    x = ctypes.c_double(0.0)
    y = ctypes.c_double(0.0)
    graph.GetPoint(pointIndex, x, y)
    yErr = graph.GetErrorY(pointIndex)
    histBin = hist.FindFixBin(x)
    hist.SetBinContent(histBin, y)
    hist.SetBinError  (histBin, yErr)
  return hist


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.gStyle.SetLegendFillColor(ROOT.kWhite)
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
  ROOT.gInterpreter.Declare(CPP_CODE_MAKEPAIR)
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

  dataSigRegionFileName = "./amptools_tree_data_tbin1_ebin4.root"
  dataBkgRegionFileName = "./amptools_tree_bkgnd_tbin1_ebin4.root"
  mcDataFileName        = "./amptools_tree_accepted_tbin1_ebin4*.root"
  treeName = "kin"

  # create friend trees with correct weights
  for dataFileName, weightFormula in [(dataSigRegionFileName, "Weight"), (dataBkgRegionFileName, "-Weight")]:
    friendFileName = f"{dataFileName}.weights"
    if os.path.exists(friendFileName):
      print(f"File '{friendFileName}' already exists, skipping creation of friend tree")
      continue
    print(f"Creating file '{friendFileName}' that contains friend tree with weights for file '{dataFileName}'")
    ROOT.RDataFrame(treeName, dataFileName) \
        .Define("eventWeight", weightFormula) \
        .Snapshot(treeName, friendFileName, ["eventWeight"])
  # attach friend trees to data tree
  dataTChain = ROOT.TChain(treeName)
  weightTChain = ROOT.TChain(treeName)
  for dataFileName in [dataSigRegionFileName, dataBkgRegionFileName]:
    dataTChain.Add(dataFileName)
    weightTChain.Add(f"{dataFileName}.weights")
  dataTChain.AddFriend(weightTChain)

  # read in real data in AmpTools format and plot RF-sideband subtracted distributions
  lvBeam, lvRecoilP, lvPip, lvPim = lorentzVectors(realData = True)
  df = ROOT.RDataFrame(dataTChain)
  # define columns
  for pairLabel, pairLvs, lvRecoil in (
    ("PiPi", (lvPip, lvPim    ), lvRecoilP),
    ("PipP", (lvPip, lvRecoilP), lvPim    ),
    ("PimP", (lvPim, lvRecoilP), lvPip    ),
  ):  # loop over two-body subsystems of pi+ pi- p final state
    for frame in ("Hf", "Gj"):  # loop over rest frame definitions
      df = defineAngleFormulas(
        df,
        lvBeam, lvRecoil, pairLvs[0], pairLvs[1],
        frame,
        columnNames = {  # names of columns to define: key: column, value: name
          "cosThetaCol" : f"{frame}{pairLabel}CosTheta",
          "thetaCol"    : f"{frame}{pairLabel}Theta",
          "phiCol"      : f"{frame}{pairLabel}Phi",
        },
      )
    df = (
      df.Define(f"Mass{pairLabel}",           f"massPair({pairLvs[0]}, {pairLvs[1]})")
        .Define(f"Hf{pairLabel}CosThetaDiff", f"Hf{pairLabel}CosTheta - helcostheta_Alex({pairLvs[0]}, {pairLvs[1]}, {lvRecoil}, {lvBeam})")
        .Define(f"Hf{pairLabel}PhiDegDiff",   f"Hf{pairLabel}PhiDeg   - helphideg_Alex  ({pairLvs[0]}, {pairLvs[1]}, {lvRecoil}, {lvBeam})")
    )

  yAxisLabel = "RF-Sideband Subtracted Combos"
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hDataEbeam",           ";E_{beam} [GeV];"          + yAxisLabel,   50, 3.55,   3.80),   "E_Beam",         "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPi",        ";m_{#pi#pi} [GeV];"        + yAxisLabel,  400, 0.28,   2.28),   "MassPiPi",       "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPiClas",    ";m_{#pi#pi} [GeV];"        + yAxisLabel,  200, 0,      2),      "MassPiPi",       "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPipP",        ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel,  400, 1,      5),      "MassPipP",       "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPipPClas",    ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel,   72, 1,      2.8),    "MassPipP",       "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPimP",        ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel,  400, 1,      5),      "MassPimP",       "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPimPClas",    ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel,   72, 1,      2.8),    "MassPimP",       "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataHfCosTheta_diff", ";#Delta cos#theta_{HF}",                 1000, -3e-13, +3e-13), "HfPiPiCosThetaDiff", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataHfPhiDeg_diff",   ";#Delta #phi_{HF} [deg]",                1000, -1e-11, +1e-11), "HfPiPiPhiDegDiff",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesGj",         ";cos#theta_{GJ};#phi_{GJ} [deg]",  50, -1,   +1,    50, -180, +180), "GjPiPiCosTheta", "GjPiPiPhiDeg",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesHf",         ";cos#theta_{HF};#phi_{HF} [deg]",  50, -1,   +1,    50, -180, +180), "HfPiPiCosTheta", "HfPiPiPhiDeg",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassVsGjCosThetaPwa", ";m_{#pi#pi} [GeV];cos#theta_{GJ}",  56,  0.28, 1.40, 72,   -1,   +1), "MassPiPi",   "GjPiPiCosTheta", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassVsGjPhiDegPwa",   ";m_{#pi#pi} [GeV];#phi_{GJ}",       56,  0.28, 1.40, 72, -180, +180), "MassPiPi",   "GjPiPiPhiDeg",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassVsHfCosThetaPwa", ";m_{#pi#pi} [GeV];cos#theta_{HF}",  56,  0.28, 1.40, 72,   -1,   +1), "MassPiPi",   "HfPiPiCosTheta", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassVsHfPhiDegPwa",   ";m_{#pi#pi} [GeV];#phi_{HF}",       56,  0.28, 1.40, 72, -180, +180), "MassPiPi",   "HfPiPiPhiDeg",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassVsHfCosTheta",    ";m_{#pi#pi} [GeV];cos#theta_{HF}", 100,  0.28, 2.28, 72,   -1,   +1), "MassPiPi",   "HfPiPiCosTheta", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataMassVsHfPhiDeg",      ";m_{#pi#pi} [GeV];#phi_{HF}",      100,  0.28, 2.28, 72, -180, +180), "MassPiPi",   "HfPiPiPhiDeg",   "eventWeight"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.Draw("COLZ")
    canv.SaveAs(f"{hist.GetName()}.pdf")

  # check against Alex' RF-sideband subtracted histograms
  histFileNameAlex = "./plots_tbin1_ebin4.root"
  histNamesAlex = {  # map histogram names
    "hDataMassPiPi"         : "M",
    "hDataMassPipP"         : "Deltapp",
    "hDataMassPimP"         : "Deltaz",
    "hDataMassVsHfCosTheta" : "MassVsCosth",
    "hDataMassVsHfPhiDeg"   : "MassVsphi",
  }
  histFileAlex = ROOT.TFile.Open(histFileNameAlex, "READ")
  for hist in hists:
    if not hist.GetName() in histNamesAlex:
      continue
    histAlex = histFileAlex.Get(histNamesAlex[hist.GetName()])
    histDiff = hist.Clone(f"{hist.GetName()}_diff")
    histDiff.Add(histAlex, -1)
    canv = ROOT.TCanvas()
    histDiff.Draw("COLZ")
    canv.SaveAs(f"{hist.GetName()}_diff.pdf")
  histFileAlex.Close()

  # overlay pipi mass distributions from data, accepted phase-space MC, and total acceptance-weighted intensity from PWA
  lvPip = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # not clear whether correct index is 1 or 2
  lvPim = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # not clear whether correct index is 1 or 2
  dfMc = ROOT.RDataFrame(treeName, mcDataFileName) \
             .Define("MassPiPi", f"massPair({lvPip}, {lvPim})")
  histMassPiPiMc   = dfMc.Histo1D(ROOT.RDF.TH1DModel("Accepted Phase-Space MC", "", 200, 0, 2), "MassPiPi")
  histMassPiPiData = df.Histo1D  (ROOT.RDF.TH1DModel("RF-subtracted Data",      "", 200, 0, 2), "MassPiPi", "eventWeight")
  pwaPlotFile = ROOT.TFile.Open("./pwa_plots3.root", "READ")
  histMassPiPiPwa = convertGraphToHist(
    graph    = pwaPlotFile.Get("Total"),
    binning  = (56, 0.28, 1.40),
    histName = "PWA Total Intensity * 0.5",
  )
  histMassPiPiPwa.Scale(0.5)
  canv = ROOT.TCanvas()
  histStack = ROOT.THStack("hMassPiPiDataAndMc", ";m_{#pi#pi} [GeV];Events / 10 MeV")
  # histStack.Add(histMassPiPiMc.GetValue())
  histStack.Add(histMassPiPiData.GetValue())
  histStack.Add(histMassPiPiPwa)
  histMassPiPiMc.SetLineColor    (ROOT.kBlue  + 1)
  histMassPiPiMc.SetMarkerColor  (ROOT.kBlue  + 1)
  histMassPiPiData.SetLineColor  (ROOT.kRed   + 1)
  histMassPiPiData.SetMarkerColor(ROOT.kRed   + 1)
  histMassPiPiPwa.SetLineColor   (ROOT.kGreen + 2)
  histMassPiPiPwa.SetMarkerColor (ROOT.kGreen + 2)
  histStack.Draw("NOSTACK")
  canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
  canv.SaveAs(f"{histStack.GetName()}.pdf")
