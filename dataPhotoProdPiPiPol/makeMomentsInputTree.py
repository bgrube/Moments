#!/usr/bin/env python3


import os

import ROOT


# C++ function to calculate invariant mass of a pair of particles
CPP_CODE_MASSPAIR = """
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


# C++ function to calculate azimuthal angle of photon polarization vector
CPP_CODE_BIGPHI = """
// returns azimuthal angle of photon polarization vector in lab frame [rad]
// for beam + target -> X + recoil and X -> a + b
//     D                    C
// code taken from https://github.com/JeffersonLab/halld_sim/blob/538677ee1347891ccefa5780e01b158e035b49b1/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
double
bigPhi(
	const double PxPC, const double PyPC, const double PzPC, const double EnPC,  // recoil
	const double PxPD, const double PyPD, const double PzPD, const double EnPD,  // beam
	const double polAngle = 0  // polarization angle [deg]
) {
	const TLorentzVector recoil(PxPC, PyPC, PzPC, EnPC);
	const TLorentzVector beam  (PxPD, PyPD, PzPD, EnPD);
	const TVector3 yAxis = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();  // normal of production plane in lab frame
	const TVector3 eps(1, 0, 0);  // reference beam polarization vector at 0 degrees in lab frame
	double Phi = polAngle * TMath::DegToRad() + atan2(yAxis.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(yAxis)));  // angle in lab frame [rad]
	// ensure [-pi, +pi] range
	while (Phi > TMath::Pi()) {
		Phi -= TMath::TwoPi();
	}
	while (Phi < -TMath::Pi()) {
		Phi += TMath::TwoPi();
	}
	return Phi;
}
"""


def defineDataFrameColumns(
  df:           ROOT.RDataFrame,
  beamPol:      float,  # photon beam polarization
  beamPolAngle: float,  # photon beam polarization angle in lab [deg]
  lvBeam:       str,    # function-argument list with Lorentz-vector components of beam photon
  lvRecoil:     str,    # function-argument list with Lorentz-vector components of recoil proton
  lvPip:        str,    # function-argument list with Lorentz-vector components of pi^+
  lvPim:        str,    # function-argument list with Lorentz-vector components of pi^-
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns for moments analysis"""
  return (
    df.Define("beamPol",    f"(Double32_t){beamPol}")
      .Define("beamPolPhi", f"(Double32_t){beamPolAngle}")
      .Define("mass",       f"(Double32_t)massPair({lvPip}, {lvPim})")
      .Define("cosTheta",   f"(Double32_t)FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})")
      .Define("theta",       "(Double32_t)std::acos(cosTheta)")
      .Define("phi",        f"(Double32_t)FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam})")
      .Define("phiDeg",      "(Double32_t)phi * TMath::RadToDeg()")
      .Define("Phi",        f"(Double32_t)bigPhi({lvRecoil}, {lvBeam}, beamPolPhi)")
      .Define("PhiDeg",      "(Double32_t)Phi * TMath::RadToDeg()")
  )


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_BIGPHI)

  # data for lowest t bin [0.1, 0.2] GeV^2
  beamPol               = 0.3519
  beamPolAngle          = 0.0
  dataSigRegionFileName = "./pipi_gluex_coh/amptools_tree_data_PARA_0_30274_31057.root"
  dataBkgRegionFileName = "./pipi_gluex_coh/amptools_tree_bkgnd_PARA_0_30274_31057.root"
  phaseSpaceAccFileName = "./pipi_gluex_coh/amptools_tree_accepted_30274_31057.root"
  phaseSpaceGenFileName = "./pipi_gluex_coh/amptools_tree_thrown_30274_31057.root"
  treeName              = "kin"
  outputTreeName        = "PiPi"
  outputColumns         = ("beamPol", "beamPolPhi", "cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi", "mass")

  # convert real data
  # create friend trees with correct weights
  for dataFileName, weightFormula in [(dataSigRegionFileName, "Weight"), (dataBkgRegionFileName, "-Weight")]:
    friendFileName = f"{os.path.basename(dataFileName)}.weights"
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
    weightTChain.Add(f"{os.path.basename(dataFileName)}.weights")
  dataTChain.AddFriend(weightTChain)
  outFileName = "data_flat.root"
  print(f"Writing file '{outFileName}' with real data")
  defineDataFrameColumns(
    df           = ROOT.RDataFrame(dataTChain),
    beamPol      = beamPol,
    beamPolAngle = beamPolAngle,
    lvBeam       = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()",
    lvRecoil     = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()",
    lvPip        = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()",
    lvPim        = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()",
  ).Snapshot(outputTreeName, outFileName, outputColumns + ("eventWeight", ))
  #TODO investigate why FSMath::helphi(lvA, lvB, lvRecoil, lvBeam) yields value that differs by 180 deg from helphideg_Alex(lvA, lvB, lvRecoil, lvBeam)

  # convert MC data
  for mcFileName, outFileName in [(phaseSpaceAccFileName, "phaseSpace_acc_flat.root"), (phaseSpaceGenFileName, "phaseSpace_gen_flat.root")]:
    print(f"Writing file '{outFileName}' with MC data")
    defineDataFrameColumns(
      df           = ROOT.RDataFrame(treeName, mcFileName),
      beamPol      = beamPol,
      beamPolAngle = beamPolAngle,
      lvBeam       = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam",
      lvRecoil     = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]",
      lvPip        = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]",  #TODO not clear whether correct index is 1 or 2
      lvPim        = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]",  #TODO not clear whether correct index is 1 or 2
    ).Snapshot(outputTreeName, outFileName, outputColumns)
