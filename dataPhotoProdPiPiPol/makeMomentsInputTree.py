#!/usr/bin/env python3


from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

import os

import ROOT


# C++ function to calculate invariant mass of a pair of particles
CPP_CODE_MASSPAIR = """
double
massPair(
	const double Px1, const double Py1, const double Pz1, const double E1,  // 4-momentum of particle 1 [GeV]
	const double Px2, const double Py2, const double Pz2, const double E2   // 4-momentum of particle 2 [GeV]
)	{
	const TLorentzVector p1(Px1, Py1, Pz1, E1);
	const TLorentzVector p2(Px2, Py2, Pz2, E2);
	return (p1 + p2).M();
}
"""

# C++ function to calculate mandelstam t = (p1 - p2)^2
CPP_CODE_MANDELSTAM_T = """
double
mandelstamT(
  const double Px1, const double Py1, const double Pz1, const double E1,  // 4-momentum of particle 1 [GeV]
  const double Px2, const double Py2, const double Pz2, const double E2   // 4-momentum of particle 2 [GeV]
) {
  const TLorentzVector p1(Px1, Py1, Pz1, E1);
  const TLorentzVector p2(Px2, Py2, Pz2, E2);
  return (p1 - p2).M2();
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
	const double PxPC, const double PyPC, const double PzPC, const double EnPC,  // 4-momentum of recoil [GeV]
	const double PxPD, const double PyPD, const double PzPD, const double EnPD,  // 4-momentum of beam [GeV]
	const double beamPolPhi = 0  // azimuthal angle of photon beam polarization in lab [deg]
) {
	const TLorentzVector recoil(PxPC, PyPC, PzPC, EnPC);
	const TLorentzVector beam  (PxPD, PyPD, PzPD, EnPD);
	const TVector3 yAxis = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();  // normal of production plane in lab frame
	const TVector3 eps(1, 0, 0);  // reference beam polarization vector at 0 degrees in lab frame
	double Phi = beamPolPhi * TMath::DegToRad() + atan2(yAxis.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(yAxis)));  // angle between photon polarization and production plane in lab frame [rad]
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

# C++ function to calculate radial distance of particle track at FDC position assuming straight tracks
CPP_CODE_TRACKDISTFDC = """
// Code used by Naomi to cut out events with very forward-going track
// see https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6180
//     https://halldweb.jlab.org/wiki-private/index.php/Tracking-9-7-2023
//     https://halldweb.jlab.org/wiki-private/index.php/Tracking-12-12-2024
double
trackDistFdc(
	const double primVertZ,  // z position of primary vertex [cm]
  const double Px, const double Py, const double Pz, const double E  // 4-momentum of particle [GeV]
) {
  const TLorentzVector p(Px, Py, Pz, E);
	const double deltaZ = 176.939 - primVertZ;  // distance along z of primary vertex and FDC [cm]
	const double deltaR = deltaZ * tan(p.Theta());
	return deltaR;
}
"""


def defineDataFrameColumns(
  df:                  ROOT.RDataFrame,
  beamPol:             float,  # photon beam polarization
  beamPolPhi:          float,  # azimuthal angle of photon beam polarization in lab [deg]
  lvTarget:            str,    # function-argument list with Lorentz-vector components of target proton
  lvBeam:              str,    # function-argument list with Lorentz-vector components of beam photon
  lvRecoil:            str,    # function-argument list with Lorentz-vector components of recoil proton
  lvPip:               str,    # function-argument list with Lorentz-vector components of pi^+
  lvPim:               str,    # function-argument list with Lorentz-vector components of pi^-
  applyAdditionalCuts: bool = False,  # apply additional cuts to remove forward-going tracks
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns for moments analysis"""
  df = (
    df.Define("beamPol",    f"(Double32_t){beamPol}")
      .Define("beamPolPhi", f"(Double32_t){beamPolPhi}")
      .Define("cosTheta",   f"(Double32_t)FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})")
      .Define("theta",       "(Double32_t)std::acos(cosTheta)")
      .Define("phi",        f"(Double32_t)FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam})")
      .Define("phiDeg",      "(Double32_t)phi * TMath::RadToDeg()")
      .Define("Phi",        f"(Double32_t)bigPhi({lvRecoil}, {lvBeam}, beamPolPhi)")
      .Define("PhiDeg",      "(Double32_t)Phi * TMath::RadToDeg()")
      .Define("mass",       f"(Double32_t)massPair({lvPip}, {lvPim})")
      .Define("minusT",     f"(Double32_t)-mandelstamT({lvTarget}, {lvRecoil})")
      # .Range(100)  # limit number of entries for testing
  )
  if applyAdditionalCuts:
    df = (
      df.Define("DistFdcPip", f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvPip})")
        .Define("DistFdcPim", f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvPim})")
        .Filter("(DistFdcPip > 4) and (DistFdcPim > 4)")  # require minimum distance of tracks at FDC position [cm]
    )
  return df


DATA_TCHAIN = ROOT.TChain()  # use global variables to avoid garbage collection
def getDataFrameWithCorrectEventWeights(
  dataSigRegionFileNames:    Sequence[str],
  dataBkgRegionFileNames:    Sequence[str],
  treeName:                  str,
  friendSigRegionFileName:   str  = "data_sig.root.weights",
  friendBkgRegionFileName:   str  = "data_bkg.root.weights",
  forceOverwriteFriendFiles: bool = True,
) -> ROOT.RDataFrame:
  """Create friend trees with correct event weights and attach them to data tree"""
  for dataFileNames, weightFormula, friendFileName in (
    (dataSigRegionFileNames,  "Weight", friendSigRegionFileName),
    (dataBkgRegionFileNames, "-Weight", friendBkgRegionFileName),
  ):
    if not forceOverwriteFriendFiles and os.path.exists(friendFileName):
      print(f"File '{friendFileName}' already exists, skipping creation of event-weight friend tree")
      continue
    print(f"Creating file '{friendFileName}' that contains friend tree with event weights for file '{dataFileNames}'")
    ROOT.RDataFrame(treeName, dataFileNames) \
        .Define("eventWeight", weightFormula) \
        .Snapshot(treeName, friendFileName, ["eventWeight"])
  DATA_TCHAIN.Reset()
  DATA_TCHAIN.SetName(treeName)
  weightTChain = ROOT.TChain(treeName)
  for dataFileNames, friendFileName in (
    (dataSigRegionFileNames, friendSigRegionFileName),
    (dataBkgRegionFileNames, friendBkgRegionFileName),
  ):
    for dataFileName in dataFileNames:
      DATA_TCHAIN.Add(dataFileName)
    weightTChain.Add(friendFileName)
  DATA_TCHAIN.AddFriend(weightTChain)
  return ROOT.RDataFrame(DATA_TCHAIN)


def lorentzVectors(realData: bool = True) -> dict[str, str]:
  """Returns Lorentz-vectors for beam photon ("lvBeam"), target proton ("lvTarget"), recoil proton ("lvRecoil"), pi+ ("lvPip"), and pi- ("lvPim")"""
  lvs = {}
  lvs["lvTarget"] = "0, 0, 0, 0.938271999359130859375"  # proton mass value from phase-space generator
  if realData:
    lvs["lvBeam"]   = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"  # beam photon
    lvs["lvRecoil"] = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"     # recoil proton
    lvs["lvPip"]    = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"   # pi+
    lvs["lvPim"]    = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"   # pi-
  else:
    lvs["lvBeam"]   = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"           # beam photon
    lvs["lvRecoil"] = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"  # recoil proton
    lvs["lvPip"]    = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # pi+  #TODO not clear whether correct index is 1 or 2
    lvs["lvPim"]    = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # pi-  #TODO not clear whether correct index is 1 or 2
  return lvs


@dataclass
class BeamPolInfo:
  """Stores information about input and output data"""
  beamPol:        float  # photon-beam polarization
  beamPolPhi:     float  # azimuthal angle of photon beam polarization in lab [deg]
  inputTreeName:  str = "kin"
  outputTreeName: str = "PiPi"

BEAM_POL_INFOS = {
  "PARA_0" : BeamPolInfo(
    beamPol    = 0.3537,
    beamPolPhi = 1.77,
  ),
  "PARA_135" : BeamPolInfo(
    beamPol    = 0.3512,
    beamPolPhi = -41.57,
  ),
  "PERP_45" : BeamPolInfo(
    beamPol    = 0.3484,
    beamPolPhi = 47.85,
  ),
  "PERP_90" : BeamPolInfo(
    beamPol    = 0.3472,
    beamPolPhi = 94.50,
  ),
}


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_BIGPHI)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  # Spring 2017 data
  # use azimuthal angles of photon beam polarization listed in Tab. 2 of https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=3977&version=6
  tBinLabel              = "tbin_0.1_0.2"
  # tBinLabel              = "tbin_0.2_0.3"
  dataInputDirName       = f"./pipi_gluex_coh/{tBinLabel}"
  # phaseSpaceGenFileNames = (f"{dataInputDirName}/MC_100M/amptools_tree_thrown_30274_31057.root", )
  # phaseSpaceAccFileNames = (f"{dataInputDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root", )
  # phaseSpaceGenFileNames = (f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_thrown_30274_31057.root", )
  # phaseSpaceAccFileNames = (f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root", )
  phaseSpaceGenFileNames = (f"{dataInputDirName}/MC_100M/amptools_tree_thrown_30274_31057.root",          f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_thrown_30274_31057.root")
  phaseSpaceAccFileNames = (f"{dataInputDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root", f"{dataInputDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root")
  # phaseSpaceGenFileNames = (f"{dataInputDirName}/MC_ps/amptools_tree_thrown_30274_31057.root",   f"{dataInputDirName}/MC_rho/amptools_tree_thrown_30274_31057.root")
  # phaseSpaceAccFileNames = (f"{dataInputDirName}/MC_ps/amptools_tree_accepted_30274_31057.root", f"{dataInputDirName}/MC_rho/amptools_tree_accepted_30274_31057.root")
  outputColumns          = ("beamPol", "beamPolPhi", "cosTheta", "theta", "phi", "phiDeg", "Phi", "PhiDeg", "mass", "minusT")
  # applyAdditionalCuts    = True  # apply additional cuts to remove forward-going tracks
  applyAdditionalCuts    = False  # apply additional cuts to remove forward-going tracks

  outputDirName = tBinLabel
  os.makedirs(outputDirName, exist_ok = True)
  for dataLabel, dataInfo in BEAM_POL_INFOS.items():
    # convert real data
    inputSigRegionFileNames = (f"{dataInputDirName}/amptools_tree_data_{dataLabel}_30274_31057.root", )
    inputBkgRegionFileNames = (f"{dataInputDirName}/amptools_tree_bkgnd_{dataLabel}_30274_31057.root", )
    realDataOutputFileName  = f"{outputDirName}/data_flat_{dataLabel}.root"
    print(f"Writing '{dataLabel}' real data from {inputSigRegionFileNames} and {inputBkgRegionFileNames} to file '{realDataOutputFileName}'")
    defineDataFrameColumns(
      df = getDataFrameWithCorrectEventWeights(
        dataSigRegionFileNames  = inputSigRegionFileNames,
        dataBkgRegionFileNames  = inputBkgRegionFileNames,
        treeName                = dataInfo.inputTreeName,
        friendSigRegionFileName = f"{outputDirName}/data_sig_{dataLabel}.root.weights",
        friendBkgRegionFileName = f"{outputDirName}/data_bkg_{dataLabel}.root.weights",
      ),
      beamPol             = dataInfo.beamPol,
      beamPolPhi          = dataInfo.beamPolPhi,
      applyAdditionalCuts = applyAdditionalCuts,
      **lorentzVectors(realData = True),
    ).Snapshot(dataInfo.outputTreeName, realDataOutputFileName, outputColumns + ("eventWeight", ))
    #TODO investigate why FSMath::helphi(lvA, lvB, lvRecoil, lvBeam) yields value that differs by 180 deg from helphideg_Alex(lvA, lvB, lvRecoil, lvBeam)

    # convert accepted phase-space MC data
    phaseSpaceAccOutputFileName = f"{outputDirName}/phaseSpace_acc_flat_{dataLabel}.root"
    print(f"Writing '{dataLabel}' accepted phase-space MC data from file(s) {phaseSpaceAccFileNames} to file '{phaseSpaceAccOutputFileName}'")
    defineDataFrameColumns(
      df                  = ROOT.RDataFrame(dataInfo.inputTreeName, phaseSpaceAccFileNames),
      beamPol             = dataInfo.beamPol,
      beamPolPhi          = dataInfo.beamPolPhi,
      applyAdditionalCuts = applyAdditionalCuts,
      **lorentzVectors(realData = True),
    ).Snapshot(dataInfo.outputTreeName, phaseSpaceAccOutputFileName, outputColumns)

    # convert thrown phase-space MC data
    phaseSpaceGenOutputFileName = f"{outputDirName}/phaseSpace_gen_flat_{dataLabel}.root"
    print(f"Writing '{dataLabel}' thrown phase-space MC data from file(s) {phaseSpaceGenFileNames} to file '{phaseSpaceGenOutputFileName}'")
    defineDataFrameColumns(
      df                  = ROOT.RDataFrame(dataInfo.inputTreeName, phaseSpaceGenFileNames),
      beamPol             = dataInfo.beamPol,
      beamPolPhi          = dataInfo.beamPolPhi,
      applyAdditionalCuts = False,
      **lorentzVectors(realData = False),
    ).Snapshot(dataInfo.outputTreeName, phaseSpaceGenOutputFileName, outputColumns)
