#!/usr/bin/env python3


import os

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
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

  dataSigRegionFileName = "./amptools_tree_data_tbin1_ebin4.root"
  dataBkgRegionFileName = "./amptools_tree_bkgnd_tbin1_ebin4.root"
  mcDataFileName        = "./amptools_tree_accepted_tbin1_ebin4.root"
  treeName              = "kin"
  outputTreeName        = "PiPi"

  # convert real data
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
  lvBeam   = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"
  lvRecoil = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"
  lvPip    = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"
  lvPim    = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"
  ROOT.RDataFrame(dataTChain) \
      .Define("mass",     f"massPair({lvPip}, {lvPim})") \
      .Define("cosTheta", f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})") \
      .Define("theta",    "std::acos(cosTheta)") \
      .Define("phi",      f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam})") \
      .Define("phiDeg",   "phi * TMath::RadToDeg()") \
      .Snapshot(outputTreeName, "data_flat.root", ("mass", "cosTheta", "theta", "phi", "phiDeg", "eventWeight"))
  #TODO investigate why FSMath::helphi(lvA, lvB, lvRecoil, lvBeam) yields value that differs by 180 deg from helphideg_Alex(lvA, lvB, lvRecoil, lvBeam)

  # convert MC data
  lvBeam   = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"
  lvRecoil = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"
  lvPip    = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  #TODO not clear whether correct index is 1 or 2
  lvPim    = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  #TODO not clear whether correct index is 1 or 2
  ROOT.RDataFrame(treeName, mcDataFileName) \
      .Define("mass",        f"massPair({lvPip}, {lvPim})") \
      .Define("cosTheta",    f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})") \
      .Define("theta",       "std::acos(cosTheta)") \
      .Define("phi",         f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam})") \
      .Define("phiDeg",      "phi * TMath::RadToDeg()") \
      .Define("eventWeight", "Weight") \
      .Snapshot(outputTreeName, "acc_phase_space_flat.root", ("mass", "cosTheta", "theta", "phi", "phiDeg", "eventWeight"))
