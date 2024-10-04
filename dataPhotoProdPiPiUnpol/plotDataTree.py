#!/usr/bin/env python3


import os

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
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

  dataSigRegionFileName = "./amptools_tree_data_tbin1_ebin4.root"
  dataBkgRegionFileName = "./amptools_tree_bkgnd_tbin1_ebin4.root"
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
  lvBeam   = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"
  lvRecoil = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"
  lvPip    = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"
  lvPim    = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"
  df = ROOT.RDataFrame(dataTChain) \
           .Define("MassPiPi",   f"massPair({lvPip}, {lvPim})") \
           .Define("MassPipP",   f"massPair({lvPip}, {lvRecoil})") \
           .Define("MassPimP",   f"massPair({lvPim}, {lvRecoil})") \
           .Define("GjCosTheta", f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})") \
           .Define("GjTheta",    "std::acos(GjCosTheta)") \
           .Define("GjPhi",      f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})") \
           .Define("GjPhiDeg",   "GjPhi * TMath::RadToDeg()") \
           .Define("HfCosTheta", f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})") \
           .Define("HfTheta",    "std::acos(HfCosTheta)") \
           .Define("HfPhi",      f"FSMath::helphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})") \
           .Define("HfPhiDeg",   "HfPhi * TMath::RadToDeg()")
  hists = (
    df.Histo1D(ROOT.RDF.TH1DModel("hDataEbeam",    ";E_{beam} [GeV]",    50, 3.55, 3.80), "E_Beam", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPi", ";m_{#pi#pi} [GeV]", 100, 0, 2), "MassPiPi", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPipP", ";m_{#pi#pi} [GeV]", 100, 1, 3), "MassPipP", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPimP", ";m_{#pi#pi} [GeV]", 100, 1, 3), "MassPimP", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesGj", ";cos#theta_{GJ};#phi_{GJ} [deg]", 50, -1, +1, 50, -180, +180), "GjCosTheta", "GjPhiDeg", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesHf", ";cos#theta_{HF};#phi_{HF} [deg]", 50, -1, +1, 50, -180, +180), "HfCosTheta", "HfPhiDeg", "eventWeight"),
  )
  for hist in hists:
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.Draw("COLZ")
    canv.SaveAs(f"{hist.GetName()}.pdf")
