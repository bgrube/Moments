"""Module that provides utility functions converting input data into format expected by `MomentCalculator`"""

from __future__ import annotations

from collections.abc import Sequence
import functools
import numpy as np
import os
import pandas as pd

import ROOT

from workflow.AnalysisConfig import (
  AnalysisConfig,
  BeamPolInfo,
  defineOverwriteRDataFrame,
)
from workflow.RootUtilities import loadFSROOTLibraries


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


DATA_TCHAINS: list[ROOT.TChain] = []  # use global variable to avoid garbage collection
def getDataFrameWithCorrectEventWeights(
  dataSigRegionFilePaths:    Sequence[str],  # file paths of input data files for signal region
  dataBkgRegionFilePaths:    Sequence[str],  # file paths of input data files for background region
  treeName:                  str,            # name of tree in input files
  sigRegionWeightFormula:    str  = "Weight",   # formula for calculating event weight for signal events
  bkgRegionWeightFormula:    str  = "-Weight",  # formula for calculating event weight for background events
  friendSigRegionFilePath:   str  = "./data_sig.root.weights",  # file path for friend tree that contains event weights for signal region
  friendBkgRegionFilePath:   str  = "./data_bkg.root.weights",  # file path for friend tree that contains event weights for background region
  forceOverwriteFriendFiles: bool = True,  # if False existing friend files will be used and assumed to contain the correct event weights
  weightColNameOutput:       str  = "eventWeight",  # name of column in friend trees that contains event weights
) -> ROOT.RDataFrame:
  """Creates friend trees with correct event weights and attaches them to data tree; must not be used in multi-threaded mode"""
  if ROOT.IsImplicitMTEnabled():
    raise RuntimeError("getDataFrameWithCorrectEventWeights() must not be used in multi-threaded mode")
  # write corrected weights into friend trees
  for dataFilePath, weightFormula, friendFilePath in (
    (dataSigRegionFilePaths, sigRegionWeightFormula, friendSigRegionFilePath),
    (dataBkgRegionFilePaths, bkgRegionWeightFormula, friendBkgRegionFilePath),
  ):
    print(f"Processing file(s) {dataFilePath}")
    if not forceOverwriteFriendFiles and os.path.exists(friendFilePath):
      print(f"File '{friendFilePath}' already exists, skipping creation of event-weight friend tree")
      continue
    print(f"Writing friend tree '{treeName}' with '{weightColNameOutput}' = '{weightFormula}' column to file '{friendFilePath}'")
    ROOT.RDataFrame(treeName, dataFilePath) \
        .Define(weightColNameOutput, weightFormula) \
        .Snapshot(treeName, friendFilePath, [weightColNameOutput])  #!NOTE! when multi-threading is enabled, the order of entries is not guaranteed to be preserved
  # chain trees for signal and background regions and add friend trees with weights
  dataTChain   = ROOT.TChain(treeName)
  weightTChain = ROOT.TChain(treeName)
  for dataFilePath, friendFilePath in (
    (dataSigRegionFilePaths, friendSigRegionFilePath),
    (dataBkgRegionFilePaths, friendBkgRegionFilePath),
  ):
    for dataFilePath in dataFilePath:
      dataTChain.Add(dataFilePath)
    weightTChain.Add(friendFilePath)
  dataTChain.AddFriend(weightTChain)
  #TODO have a look at <https://root.cern/doc/v632/classROOT_1_1RDataFrame.html#rdf-from-spec> to build data frame.
  DATA_TCHAINS.append(dataTChain)  # avoid garbage collection of TChain
  return ROOT.RDataFrame(dataTChain)


# C++ function to calculate invariant mass of a pair of particles
CPP_CODE_MASSPAIR = """
double
massPair(
	const double Px1, const double Py1, const double Pz1, const double E1,  // 4-momentum of particle 1 [GeV]
	const double Px2, const double Py2, const double Pz2, const double E2   // 4-momentum of particle 2 [GeV]
) {
	const TLorentzVector p1(Px1, Py1, Pz1, E1);
	const TLorentzVector p2(Px2, Py2, Pz2, E2);
	return (p1 + p2).M();
}
"""

# C++ function to calculate Mandelstam t = (p1 - p2)^2
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

# C++ function to limit range of azimuthal angle to [-pi, +pi]
CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE = """
double
fixAzimuthalAngleRange(double angle)  // [rad]
{
	// ensure [-pi, +pi] range
	while (angle > TMath::Pi()) {
		angle -= TMath::TwoPi();
	}
	while (angle < -TMath::Pi()) {
		angle += TMath::TwoPi();
	}
	return angle;
}
"""

# C++ function to calculate angles and beam polarization angle for moment analysis
CPP_CODE_TWO_BODY_ANGLES = """
// calculates helicity or Gottfried-Jackson angles (theta, phi)
// and azimuthal angle Phi between photon polarization and production plane in lab frame
// for reaction beam + target -> resonance + recoil with resonance -> A + B
// angles are returned as vector (cos(theta), phi [rad], Phi [rad])
// code taken from GlueX AmpTools: https://github.com/JeffersonLab/halld_sim/blob/39b18bdbab88192275fed57fda161f9a52d04422/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
enum CoordSysType {
  HF = 0,  // helicity frame
  GJ = 1,  // Gottfried-Jackson frame
};

std::vector<Double32_t>
twoBodyAngles(
	const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam,    // 4-momentum of beam [GeV]
	const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,  // 4-momentum of recoil [GeV]
	const double PxPA,     const double PyPA,     const double PzPA,     const double EPA,      // 4-momentum of particle A (analyzer) [GeV]
	const double PxPB,     const double PyPB,     const double PzPB,     const double EPB,      // 4-momentum of particle B [GeV]
	const CoordSysType coordSysType,   // coordinate system type for angle definitions
	const double beamPolPhiLabDeg = 0  // azimuthal angle of photon beam polarization in lab [deg]
) {
	// 4-vectors in lab frame
	const TLorentzVector beam  (PxBeam,   PyBeam,   PzBeam,   EBeam);
	const TLorentzVector recoil(PxRecoil, PyRecoil, PzRecoil, ERecoil);
	const TLorentzVector pA    (PxPA,     PyPA,     PzPA,     EPA);
	const TLorentzVector pB    (PxPB,     PyPB,     PzPB,     EPB);
	// boost 4-vectors to resonance rest frame
	const TLorentzVector resonance = pA + pB;
	const TLorentzRotation resonanceBoost(-resonance.BoostVector());
	const TLorentzVector beamRF   = resonanceBoost * beam;
	const TLorentzVector recoilRF = resonanceBoost * recoil;
	const TLorentzVector pARF     = resonanceBoost * pA;
	// define axes of coordinate system
	const TVector3 yAxis = beam.Vect().Cross(-recoil.Vect()).Unit();  // normal of production plane in lab frame
	const TVector3 zAxis = [&]() -> TVector3 {  // z axis depends on coordinate system type
		if (coordSysType == HF) {
			return -recoilRF.Vect().Unit();  // helicity frame: opposite to recoil proton in resonance rest frame
		} else if (coordSysType == GJ) {
			return beamRF.Vect().Unit();     // Gottfried-Jackson frame: along beam direction in resonance rest frame
		} else {
			throw std::runtime_error(std::string("Unsupported coordinate system type '") + std::to_string(coordSysType) + "'");
		}
	}();
	const TVector3 xAxis = yAxis.Cross(zAxis).Unit();  // right-handed coordinate system
	// calculate angles of particle A (analyzer) in selected frame
	const TVector3 pA_frame(pARF.Vect() * xAxis, pARF.Vect() * yAxis, pARF.Vect() * zAxis);  // vector of particle A in selected frame
	const double cosTheta = pA_frame.CosTheta();  // polar angle of particle A
	const double phi      = pA_frame.Phi();       // azimuthal angle of particle A [rad]
	// calculate azimuthal angle between beam polarization and production plane
	const TVector3 eps(1, 0, 0);  // reference beam polarization vector at 0 degrees in lab frame
	const double Phi = beamPolPhiLabDeg * TMath::DegToRad() + atan2(yAxis.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(yAxis)));  // angle between photon polarization and production plane in lab frame [rad]
	return std::vector<Double32_t>{cosTheta, phi, fixAzimuthalAngleRange(Phi)};
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


def lorentzVectors(dataFormat: AnalysisConfig.DataFormat) -> dict[str, str]:
  """Returns Lorentz-vectors for beam photon ("beam"), target proton ("target"), recoil proton ("recoil"), pi+ ("pip"), and pi- ("pim")"""
  lvs = {}
  lvs["target"] = "0, 0, 0, 0.938271999359130859375"  # proton mass value from phase-space generator
  if dataFormat == AnalysisConfig.DataFormat.ALEX:
    lvs["beam"  ] = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"  # beam photon
    lvs["recoil"] = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"     # recoil proton
    lvs["pip"   ] = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"   # pi+
    lvs["pim"   ] = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"   # pi-
  elif dataFormat == AnalysisConfig.DataFormat.AMPTOOLS:
    lvs["beam"  ] = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"           # beam photon
    lvs["recoil"] = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"  # recoil proton
    # pi+ pi- channel
    lvs["pip"   ] = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # pi+
    lvs["pim"   ] = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # pi-
    # eta pi0 channel
    lvs["pi0"   ] = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # eta
    lvs["eta"   ] = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # pi0
  elif dataFormat == AnalysisConfig.DataFormat.JPAC_MC:
    # kinematic variables according to Eq. (1) in Bibrzycki et al., PRD 111, 014002 (2025)
    # gamma (q) + p (p1) -> pi+ (k1) + pi- (k2) + p (p2)
    # four-momenta are defined as
    #               (p_x, p_y, p_z, E)
    lvs["beam"  ] = "q1,  q2,  q3,  q0"   # beam photon
    lvs["target"] = "p11, p12, p13, p10"  # target proton
    lvs["recoil"] = "p21, p22, p23, p20"  # recoil proton
    lvs["pip"   ] = "k11, k12, k13, k10"  # pi+
    lvs["pim"   ] = "k21, k22, k23, k20"  # pi-
  elif dataFormat == AnalysisConfig.DataFormat.TLORENTZVECTORS:
    lvs["beam"  ] = "lvBeamLab.X(),   lvBeamLab.Y(),   lvBeamLab.Z(),   lvBeamLab.E()"    # beam photon
    lvs["target"] = "lvTargetLab.X(), lvTargetLab.Y(), lvTargetLab.Z(), lvTargetLab.E()"  # target proton
    lvs["recoil"] = "lvRecoilLab.X(), lvRecoilLab.Y(), lvRecoilLab.Z(), lvRecoilLab.E()"  # recoil proton
    lvs["pip"   ] = "lvPipLab.X(),    lvPipLab.Y(),    lvPipLab.Z(),    lvPipLab.E()"     # pi+
    lvs["pim"   ] = "lvPimLab.X(),    lvPimLab.Y(),    lvPimLab.Z(),    lvPimLab.E()"     # pi-
  elif dataFormat == AnalysisConfig.DataFormat.FSROOT:
    lvs["beam"  ] = "PxPB, PyPB, PzPB, EnPB"  # beam photon
    lvs["recoil"] = "PxP1, PyP1, PzP1, EnP1"  # recoil proton
    lvs["etap"  ] = "PxP2, PyP2, PzP2, EnP2"  # eta'
    lvs["eta"   ] = "PxP3, PyP3, PzP3, EnP3"  # eta
    lvs["K_S"   ] = "PxP2, PyP2, PzP2, EnP2"  # not mass-constrained K_S
    lvs["K_L"   ] = "PxP3, PyP3, PzP3, EnP3"  # missing mass-constrained K_L
  else:
    raise RuntimeError(f"Unsupported data format type '{dataFormat}'")
  return lvs


def defineDataFrameColumns(
  df:                   ROOT.RDataFrame,
  lvTarget:             str,  # function-argument list with Lorentz-vector components of target proton
  lvBeam:               str,  # function-argument list with Lorentz-vector components of beam photon
  lvRecoil:             str,  # function-argument list with Lorentz-vector components of recoil proton
  lvA:                  str,  # function-argument list with Lorentz-vector components of daughter A (analyzer)
  lvB:                  str,  # function-argument list with Lorentz-vector components of daughter B
  beamPolInfo:          BeamPolInfo | None          = None,  # photon beam polarization
  frame:                AnalysisConfig.CoordSysType = AnalysisConfig.CoordSysType.HF,  # reference frame for angle definitions
  additionalColumnDefs: dict[str, str]              = {},  # additional columns to define
  additionalFilterDefs: list[str]                   = [],  # additional filter conditions to apply
  colNameSuffix:        str                         = "",  # suffix appended to column names
  defineFSROOTAngles:   bool                        = False,  # if True, define additional columns with angles calculated using FSROOT
) -> ROOT.RDataFrame:
  """Defines columns for (A, B) pair mass, squared four-momentum transferred from beam to recoil, and angles (cos(theta), phi) of particle A in X rest frame for reaction beam + target -> X + recoil with X -> A + B using the given Lorentz-vector components"""
  print(f"Defining angles in '{frame}' frame using '{lvA}' as analyzer and '{lvRecoil}' as recoil")
  angColNameSuffix = frame.name + colNameSuffix if colNameSuffix else ""  # column name suffixes are only used for plotting
  coordSysTypeStr = None
  if frame == AnalysisConfig.CoordSysType.HF:
    coordSysTypeStr = 'CoordSysType::HF'
  elif frame == AnalysisConfig.CoordSysType.GJ:
    coordSysTypeStr = 'CoordSysType::GJ'
  else:
    raise ValueError(f"Unsupported coordinate system type '{frame}'")
  if additionalColumnDefs:  # define additional columns before all other variables to allow for use in angle definitions
    for columnName, columnFormula in additionalColumnDefs.items():
      print(f"Defining additional column '{columnName}' = '{columnFormula}'")
      df = defineOverwriteRDataFrame(df, columnName, columnFormula)
  df = (
    df.Define(f"angles{angColNameSuffix}",   f"twoBodyAngles({lvBeam}, {lvRecoil}, {lvA}, {lvB}, {coordSysTypeStr}, {'0' if beamPolInfo is None else beamPolInfo.PhiLab})")  # cos(theta), phi [rad], Phi [rad]
      .Define(f"cosTheta{angColNameSuffix}", f"angles{angColNameSuffix}[0]")
      .Define(f"phi{angColNameSuffix}",      f"angles{angColNameSuffix}[1]")
  )
  if defineFSROOTAngles:
    loadFSROOTLibraries()
    #TODO there seems to be a bug in the FSROOT functions that calculate phi
    #     when using the same analyzer as for cosTheta, phi is flipped by 180 deg
    #     this difference is seen when comparing to GlueX AmpTools function and also when comparing to PWA results
    #     switching the analyzer for the phi calculation cures this problem
    #     in general, switching the analyzer flips sign of moments with odd M
    if frame == AnalysisConfig.CoordSysType.HF:
      df = (
        # use z_HF = -p_recoil, A as analyzer, and y_HF = -(p_beam x p_recoil)
        df.Define(f"cosTheta{angColNameSuffix}_FSROOT", f"(Double32_t)FSMath::helcostheta({lvA}, {lvB}, {lvRecoil})")
          .Define(f"phi{angColNameSuffix}_FSROOT",      f"(Double32_t)FSMath::helphi({lvB}, {lvA}, {lvRecoil}, {lvBeam})")  # need to switch analyzer to make it agree
      )
    elif frame == AnalysisConfig.CoordSysType.GJ:
      df = (
        # use z_GJ = p_beam, A as analyzer, and y_GJ = -(p_beam x p_recoil)
        df.Define(f"cosTheta{angColNameSuffix}_FSROOT", f"(Double32_t)FSMath::gjcostheta({lvA}, {lvB}, {lvBeam})")  #!NOTE! signature is different from FSMath::helcostheta (see FSBasic/FSMath.h)
          .Define(f"phi{angColNameSuffix}_FSROOT",      f"(Double32_t)FSMath::gjphi({lvB}, {lvA}, {lvRecoil}, {lvBeam})")  # need to switch analyzer to make it agree
      )
  df = (
    df.Define(f"theta{angColNameSuffix}",  f"(Double32_t)std::acos(cosTheta{angColNameSuffix})")
      .Define(f"phi{angColNameSuffix}Deg", f"(Double32_t)(phi{angColNameSuffix} * TMath::RadToDeg())")
  )
  # allow for redefinition of already existing frame-independent columns if function is called for several frames
  df = defineOverwriteRDataFrame(df, f"mass{colNameSuffix}",   f"(Double32_t)massPair({lvA}, {lvB})")
  df = defineOverwriteRDataFrame(df, f"minusT{colNameSuffix}", f"(Double32_t)-mandelstamT({lvTarget}, {lvRecoil})")
  if beamPolInfo is not None:
    df = defineOverwriteRDataFrame(df, f"beamPol{colNameSuffix}",          f"(Double32_t){beamPolInfo.pol}")
    df = defineOverwriteRDataFrame(df, f"beamPolPhiLab{colNameSuffix}Deg", f"(Double32_t){beamPolInfo.PhiLab}")
    df = defineOverwriteRDataFrame(df, f"Phi{colNameSuffix}",              f"angles{angColNameSuffix}[2]")
    df = defineOverwriteRDataFrame(df, f"Phi{colNameSuffix}Deg",           f"(Double32_t)(Phi{colNameSuffix} * TMath::RadToDeg())")
  if additionalFilterDefs:
    for filterDef in additionalFilterDefs:
      print(f"Applying additional filter '{filterDef}'")
      df = df.Filter(filterDef)
  return df


def readDataJpac(inputFilePath: str) -> ROOT.RDataFrame:
  """Reads JPAC data from an ASCII file into a ROOT RDataFrame"""
  print(f"Reading file '{inputFilePath}'")
  pandasDf = pd.read_csv(inputFilePath, sep = r"\s+")
  pandasDf["t"]  *= -1.0  # flip sign of t to make it positive
  pandasDf["phi"] = np.degrees(pandasDf["phi"])  # convert to angle to degrees
  pandasDf.loc[pandasDf["phi"] > 180, "phi"] -= 360  # shift phi angle into [-180, +180] deg range by applying (if phi > 180 then phi -= 360)
  # rename columns to avoid name clashes and match naming convention
  pandasDf.rename(columns = {"t"     : "minusTJpac"  }, inplace = True)
  pandasDf.rename(columns = {"mpipi" : "massJpac"    }, inplace = True)
  pandasDf.rename(columns = {"phi"   : "phiDegJpac"  }, inplace = True)
  pandasDf.rename(columns = {"costh" : "cosThetaJpac"}, inplace = True)
  # print(f"DataFrame shape: {pandasDf.shape}")
  # print(f"Columns: {list(pandasDf.columns)}")
  # convert Pandas DataFrame into ROOT RDataFrame
  print("Converting data to ROOT.RDataFrame")
  arrayDict = {column : np.array(pandasDf[column]) for column in pandasDf}
  rootDf: ROOT.RDataFrame = ROOT.RDF.MakeNumpyDataFrame(arrayDict)
  return rootDf
