#!/usr/bin/env python3
"""
This module converts input data into the format expected by `MomentCalculator`.

Usage: Run this module as a script to convert input data files.
"""


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import (
  dataclass,
  field,
)
from enum import Enum
import functools
import numpy as np
import os
import pandas as pd
import subprocess
import tempfile

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`  #TODO add this to all scripts that generate plots

from MomentCalculator import (
  KinematicBinningVariable,
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
from PlottingUtilities import (
  HistAxisBinning,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class BeamPolInfo:
  """Stores information about beam polarization for a specific orientation"""
  pol:    float | str  # photon-beam polarization magnitude value or column name
  PhiLab: float | str  # azimuthal angle of photon beam polarization in lab frame [deg] value or column name

  def __str__(self) -> str:
    result = "pol = "
    if isinstance(self.pol, float):
      result += f"{self.pol:.4f}"
    else:
      result += f"'{self.pol}'"
    result += ", PhiLab = "
    if isinstance(self.PhiLab, float):
      result += f"{self.PhiLab:.1f} deg"
    else:
      result += f"'{self.PhiLab}'"
    return result


# polarization values from Version 9 of `makePolVals` tool from https://halldweb.jlab.org/wiki-private/index.php/TPOL_Polarization
# beam polarization angles in lab frame taken from `Lab Phi` column of tables 2 to 5 in GlueX-doc-3977
BEAM_POL_INFOS: dict[str, dict[str, BeamPolInfo | None]] = {  # data period : {beam-polarization orientation : BeamPolInfo(...)}
  "merged" : {  # several merged data periods with different polarization values
    "All" : BeamPolInfo(  # read polarization values from the given column names
      pol    = "beamPol",
      PhiLab = "beamPolPhiLabDeg",
    ),
  },
  "2017_01" : {  # polarization magnitudes obtained by running `.x makePolVals.C(17, 1, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      pol    = 0.3537,
      PhiLab = 1.8,
    ),
    "PERP_45" : BeamPolInfo(
      pol    = 0.3484,
      PhiLab = 47.9,
    ),
    "PERP_90" : BeamPolInfo(
      pol    = 0.3472,
      PhiLab = 94.5,
    ),
    "PARA_135" : BeamPolInfo(
      pol    = 0.3512,
      PhiLab = -41.6,
    ),
    "AMO" : None,
    "Unpol" : None,
  },
  "2018_01" : {  # polarization magnitudes obtained by running `.x makePolVals.C(18, 1, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      pol    = 0.3420,
      PhiLab = 4.1,
    ),
    "PERP_45" : BeamPolInfo(
      pol    = 0.3474,
      PhiLab = 48.5,
    ),
    "PERP_90" : BeamPolInfo(
      pol    = 0.3478,
      PhiLab = 94.2,
    ),
    "PARA_135" : BeamPolInfo(
      pol    = 0.3517,
      PhiLab = -42.4,
    ),
    "AMO" : None,
    "Unpol" : None,
  },
  "2018_08" : {  # polarization magnitudes obtained by running `.x makePolVals.C(18, 2, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      pol    = 0.3563,
      PhiLab = 3.3,
    ),
    "PERP_45" : BeamPolInfo(
      pol    = 0.3403,
      PhiLab = 48.3,
    ),
    "PERP_90" : BeamPolInfo(
      pol    = 0.3430,
      PhiLab = 92.9,
    ),
    "PARA_135" : BeamPolInfo(
      pol    = 0.3523,
      PhiLab = -42.1,
    ),
    "AMO" : None,
    "Unpol" : None,
  },
}


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


class InputDataFormat(Enum):
  ALEX            = 0  # Alex' data format  #TODO improve naming
  AMPTOOLS        = 1  # AmpTools format
  JPAC_MC         = 2  # MC truth data in JPAC text format
  TLORENTZVECTORS = 3  # TLorentzVector for each particle
  FSROOT_RECO     = 4  # FSROOT format for reconstructed
  FSROOT_TRUTH    = 5  # FSROOT format for MC truth

def lorentzVectors(dataFormat: InputDataFormat) -> dict[str, str]:
  """Returns Lorentz-vectors for beam photon ("beam"), target proton ("target"), recoil proton ("recoil"), pi+ ("pip"), and pi- ("pim")"""
  lvs = {}
  lvs["target"] = "0, 0, 0, 0.938271999359130859375"  # proton mass value from phase-space generator
  if dataFormat == InputDataFormat.ALEX:
    lvs["beam"  ] = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"  # beam photon
    lvs["recoil"] = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"     # recoil proton
    lvs["pip"   ] = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"   # pi+
    lvs["pim"   ] = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"   # pi-
  elif dataFormat == InputDataFormat.AMPTOOLS:
    lvs["beam"  ] = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"           # beam photon
    lvs["recoil"] = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"  # recoil proton
    # pi+ pi- channel
    lvs["pip"   ] = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # pi+
    lvs["pim"   ] = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # pi-
    # eta pi0 channel
    lvs["pi0"   ] = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # eta
    lvs["eta"   ] = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # pi0
  elif dataFormat == InputDataFormat.JPAC_MC:
    # kinematic variables according to Eq. (1) in Bibrzycki et al., PRD 111, 014002 (2025)
    # gamma (q) + p (p1) -> pi+ (k1) + pi- (k2) + p (p2)
    # four-momenta are defined as
    #                 (p_x, p_y, p_z, E)
    lvs["beam"  ] = "q1,  q2,  q3,  q0"   # beam photon
    lvs["target"] = "p11, p12, p13, p10"  # target proton
    lvs["recoil"] = "p21, p22, p23, p20"  # recoil proton
    lvs["pip"   ] = "k11, k12, k13, k10"  # pi+
    lvs["pim"   ] = "k21, k22, k23, k20"  # pi-
  elif dataFormat == InputDataFormat.TLORENTZVECTORS:
    lvs["beam"  ] = "lvBeamLab.X(),   lvBeamLab.Y(),   lvBeamLab.Z(),   lvBeamLab.E()"    # beam photon
    lvs["target"] = "lvTargetLab.X(), lvTargetLab.Y(), lvTargetLab.Z(), lvTargetLab.E()"  # target proton
    lvs["recoil"] = "lvRecoilLab.X(), lvRecoilLab.Y(), lvRecoilLab.Z(), lvRecoilLab.E()"  # recoil proton
    lvs["pip"   ] = "lvPipLab.X(),    lvPipLab.Y(),    lvPipLab.Z(),    lvPipLab.E()"     # pi+
    lvs["pim"   ] = "lvPimLab.X(),    lvPimLab.Y(),    lvPimLab.Z(),    lvPimLab.E()"     # pi-
  elif dataFormat == InputDataFormat.FSROOT_RECO or dataFormat == InputDataFormat.FSROOT_TRUTH:
    prefix = "MC" if dataFormat == InputDataFormat.FSROOT_TRUTH else ""
    lvs["beam"  ] = f"{prefix}PxPB, {prefix}PyPB, {prefix}PzPB, {prefix}EnPB"  # beam photon
    lvs["recoil"] = f"{prefix}PxP1, {prefix}PyP1, {prefix}PzP1, {prefix}EnP1"  # recoil proton
    lvs["etap"  ] = f"{prefix}PxP2, {prefix}PyP2, {prefix}PzP2, {prefix}EnP2"  # eta'
    lvs["eta"   ] = f"{prefix}PxP3, {prefix}PyP3, {prefix}PzP3, {prefix}EnP3"  # eta
  else:
    raise RuntimeError(f"Unsupported data format type '{dataFormat}'")
  return lvs


def defineOverwrite(
  df:         ROOT.RDataFrame,
  colName:    str,
  newFormula: str,
) -> ROOT.RDataFrame:
  """If column `colName` exists, redefines it using formula `newFormula`"""
  if not df.HasColumn(colName):
    # print(f"Defining column '{colName}' = '{newFormula}'")
    df = df.Define(colName, newFormula)
  else:
    print(f"Redefining column '{colName}' to '{newFormula}'")
    df = df.Redefine(colName, newFormula)
  return df


class CoordSysType(Enum):
  HF = 0  # helicity frame
  GJ = 1  # Gottfried-Jackson frame


def defineDataFrameColumns(
  df:                   ROOT.RDataFrame,
  lvTarget:             str,  # function-argument list with Lorentz-vector components of target proton
  lvBeam:               str,  # function-argument list with Lorentz-vector components of beam photon
  lvRecoil:             str,  # function-argument list with Lorentz-vector components of recoil proton
  lvA:                  str,  # function-argument list with Lorentz-vector components of daughter A (analyzer)
  lvB:                  str,  # function-argument list with Lorentz-vector components of daughter B
  beamPolInfo:          BeamPolInfo | None = None,  # photon beam polarization
  frame:                CoordSysType       = CoordSysType.HF,  # reference frame for angle definitions
  additionalColumnDefs: dict[str, str]     = {},  # additional columns to define
  additionalFilterDefs: list[str]          = [],  # additional filter conditions to apply
  colNameSuffix:        str                = "",  # suffix appended to column names
) -> ROOT.RDataFrame:
  """Defines columns for (A, B) pair mass, squared four-momentum transferred from beam to recoil, and angles (cos(theta), phi) of particle A in X rest frame for reaction beam + target -> X + recoil with X -> A + B using the given Lorentz-vector components"""
  print(f"Defining angles in '{frame}' frame using '{lvA}' as analyzer and '{lvRecoil}' as recoil")
  angColNameSuffix = frame.name + colNameSuffix if colNameSuffix else ""  # column name suffixes are only used for plotting
  coordSysTypeStr = None
  if frame == CoordSysType.HF:
    coordSysTypeStr = 'CoordSysType::HF'
  elif frame == CoordSysType.GJ:
    coordSysTypeStr = 'CoordSysType::GJ'
  else:
    raise ValueError(f"Unsupported coordinate system type '{frame}'")
  df = (
    df.Define(f"angles{angColNameSuffix}",   f"twoBodyAngles({lvBeam}, {lvRecoil}, {lvA}, {lvB}, {coordSysTypeStr}, {'0' if beamPolInfo is None else beamPolInfo.PhiLab})")
      .Define(f"cosTheta{angColNameSuffix}", f"angles{angColNameSuffix}[0]")
      .Define(f"phi{angColNameSuffix}",      f"angles{angColNameSuffix}[1]")
  )
  if True:
  # if False:
    #TODO there seems to be a bug in the FSROOT functions that calculate phi
    #     when using the same analyzer as for cosTheta, phi is flipped by 180 deg
    #     this difference is seen when comparing to GlueX AmpTools function and also when comparing to PWA results
    #     switching the analyzer for the phi calculation cures this problem
    #     in general, switching the analyzer flips sign of moments with odd M
    if frame == CoordSysType.HF:
      df = (
        # use z_HF = -p_recoil, A as analyzer, and y_HF = -(p_beam x p_recoil)
        df.Define(f"cosTheta{angColNameSuffix}_FSROOT", f"(Double32_t)FSMath::helcostheta({lvA}, {lvB}, {lvRecoil})")
          .Define(f"phi{angColNameSuffix}_FSROOT",      f"(Double32_t)FSMath::helphi({lvB}, {lvA}, {lvRecoil}, {lvBeam})")  # need to switch analyzer to make it agree
      )
    elif frame == CoordSysType.GJ:
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
  df = defineOverwrite(df, f"mass{colNameSuffix}",   f"(Double32_t)massPair({lvA}, {lvB})")
  df = defineOverwrite(df, f"minusT{colNameSuffix}", f"(Double32_t)-mandelstamT({lvTarget}, {lvRecoil})")
  if beamPolInfo is not None:
    df = defineOverwrite(df, f"beamPol{colNameSuffix}",          f"(Double32_t){beamPolInfo.pol}")
    df = defineOverwrite(df, f"beamPolPhiLab{colNameSuffix}Deg", f"(Double32_t){beamPolInfo.PhiLab}")
    df = defineOverwrite(df, f"Phi{colNameSuffix}",              f"angles{angColNameSuffix}[2]")
    df = defineOverwrite(df, f"Phi{colNameSuffix}Deg",           f"(Double32_t)(Phi{colNameSuffix} * TMath::RadToDeg())")
  if additionalColumnDefs:
    for columnName, columnFormula in additionalColumnDefs.items():
      print(f"Defining additional column '{columnName}' = '{columnFormula}'")
      df = defineOverwrite(df, columnName, columnFormula)
  if additionalFilterDefs:
    for filterDef in additionalFilterDefs:
      print(f"Applying additional filter '{filterDef}'")
      df = df.Filter(filterDef)
  return df


DATA_TCHAINS: list[ROOT.TChain] = []  # use global variable to avoid garbage collection
def getDataFrameWithCorrectEventWeights(
  dataSigRegionFileNames:    Sequence[str],  # file names of input data files for signal region
  dataBkgRegionFileNames:    Sequence[str],  # file names of input data files for background region
  treeName:                  str,            # name of tree in input files
  sigRegionWeightFormula:    str  = "Weight",   # formula for calculating event weight for signal events
  bkgRegionWeightFormula:    str  = "-Weight",  # formula for calculating event weight for background events
  friendSigRegionFileName:   str  = "data_sig.root.weights",  # file name for friend tree that contains event weights for signal region
  friendBkgRegionFileName:   str  = "data_bkg.root.weights",  # file name for friend tree that contains event weights for background region
  forceOverwriteFriendFiles: bool = True,  # if False existing friend files will be used and assumed to contain the correct event weights
) -> ROOT.RDataFrame:
  """Create friend trees with correct event weights and attach them to data tree"""
  # write corrected weights into friend trees
  for dataFileNames, weightFormula, friendFileName in (
    (dataSigRegionFileNames, sigRegionWeightFormula, friendSigRegionFileName),
    (dataBkgRegionFileNames, bkgRegionWeightFormula, friendBkgRegionFileName),
  ):
    print(f"Processing file(s) {dataFileNames}")
    if not forceOverwriteFriendFiles and os.path.exists(friendFileName):
      print(f"File '{friendFileName}' already exists, skipping creation of event-weight friend tree")
      continue
    print(f"Writing event-weight friend tree to file '{friendFileName}'")
    ROOT.RDataFrame(treeName, dataFileNames) \
        .Define("eventWeight", weightFormula) \
        .Snapshot(treeName, friendFileName, ["eventWeight"])
  # chain trees for signal and background regions and add friend trees with weights
  dataTChain   = ROOT.TChain(treeName)
  weightTChain = ROOT.TChain(treeName)
  for dataFileNames, friendFileName in (
    (dataSigRegionFileNames, friendSigRegionFileName),
    (dataBkgRegionFileNames, friendBkgRegionFileName),
  ):
    for dataFileName in dataFileNames:
      dataTChain.Add(dataFileName)
    weightTChain.Add(friendFileName)
  dataTChain.AddFriend(weightTChain)
  #TODO have a look at <https://root.cern/doc/v632/classROOT_1_1RDataFrame.html#rdf-from-spec> to build data frame.
  DATA_TCHAINS.append(dataTChain)  # avoid garbage collection of TChain
  return ROOT.RDataFrame(dataTChain)


def readDataJpac(inputFileName: str) -> ROOT.RDataFrame:
  """Reads JPAC data from an ASCII file into a ROOT RDataFrame"""
  print(f"Reading file '{inputFileName}'")
  pandasDf = pd.read_csv(inputFileName, sep = r"\s+")
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

def reweightData(
  dataToWeight: ROOT.RDataFrame,  # data to reweight
  treeName:     str,              # name of TTree holding the data
  variableName: str,              # column name corresponding to kinematic variable whose distribution is to be reweighted
  targetDistr:  ROOT.TH1D,        # histogram with target distribution
) -> ROOT.RDataFrame:
  """Generic function that reweights data in given RDataFrame such that the distribution of the given variable matches the target distribution in the given histogram"""
  # get histogram of current distribution using same binning as targetDistribution
  currentDistr = dataToWeight.Histo1D(
    ROOT.RDF.TH1DModel(
      f"{variableName}Distr", f"Current Distribution;{variableName}",
      targetDistr.GetNbinsX(), targetDistr.GetXaxis().GetXmin(), targetDistr.GetXaxis().GetXmax()
    ),
    variableName,
  ).GetValue()
  # normalize target and current histograms such that they represent the corresponding PDFs
  targetDistr.Scale (1.0 / targetDistr.Integral() )
  currentDistr.Scale(1.0 / currentDistr.Integral())
  # calculate the weight as the ratio of target and current PDF
  weightsHist = targetDistr.Clone("weightsHist")
  weightsHist.SetTitle("Weights")
  weightsHist.Divide(currentDistr)
  if True:
  # if False:
    # save plots of distributions
    for hist in (currentDistr, targetDistr, weightsHist):
      canv = ROOT.TCanvas()
      hist.Draw()
      canv.SaveAs(f"{hist.GetName()}.root")
  # add columns for rejection sampling to input data
  RootUtilities.declareInCpp(weightsHist = weightsHist)  # use Python TH1D object in C++  #TODO this can only be called once; otherwise this call crashes in ROOT
  dataToWeight = (
    dataToWeight.Define("reweightingWeight", f"(Double32_t)PyVars::weightsHist.GetBinContent(PyVars::weightsHist.FindBin({variableName}))")
                .Define("reweightingRndNmb",  "(Double32_t)gRandom->Rndm()")  # random number uniformly distributed in [0, 1]
  )
  tmpFileName = tempfile.mktemp(dir = "./", prefix = "unweighted.", suffix = ".root")
  dataToWeight.Snapshot(treeName, tmpFileName)  # write unweighted data to temporary file to ensure that random column is filled only once
  dataToWeight = ROOT.RDataFrame(treeName, tmpFileName)  # read data back from temporary file
  nmbEvents = dataToWeight.Count().GetValue()  # number of events before reweighting
  # determine maximum weight
  maxWeight = dataToWeight.Max("reweightingWeight").GetValue()
  print(f"Maximum weight is {maxWeight}")
  # apply weights by accepting each event with probability reweightingWeight / maxWeight
  reweightedData = (
    dataToWeight.Define("acceptEventReweight", f"(bool)(reweightingRndNmb < (reweightingWeight / {maxWeight}))")
                .Filter("acceptEventReweight == true")
  )
  nmbWeightedEvents = reweightedData.Count().GetValue()
  print(f"After reweighting, the sample contains {nmbWeightedEvents} accepted events; reweighting efficiency is {nmbWeightedEvents / nmbEvents}")
  # subprocess.run(f"rm --force --verbose {tmpFileName}", shell = True)  #TODO this does not work as the RDataFrame based on this file is passed to the calling code
  return reweightedData


def reweightKinDistribution(
  dataToWeight:    ROOT.RDataFrame,  # data to reweight
  treeName:        str,              # name of TTree holding the data
  binning:         HistAxisBinning,  # binning of kinematic variable whose distribution is to be reweighted
  targetDistrFrom: str | MomentResultsKinematicBinning,  # construct target distribution from given data file name or from H_0(0, 0) in given moment results
  outFileName:     str,  # name of file to write data into
  outputColumns:   Sequence[str] = (),  # columns to write into output file; if empty, all columns are written
) -> None:
  """Reweights distribution of given kinematic variable of given data according to the kinematic distribution of data in given file name or according to kinematic dependence of H_0(0, 0) in given moment results"""
  print(f"Reweighting {binning.var.name} dependence")
  targetDistr = None
  if isinstance(targetDistrFrom, str):
    # construct target distribution from real data
    print(f"Constructing target distribution from column '{binning.var.name}' in tree '{treeName}' in file '{targetDistrFrom}'")
    dataTarget = ROOT.RDataFrame(treeName, targetDistrFrom)
    targetDistr = dataTarget.Histo1D(
      ROOT.RDF.TH1DModel(f"{binning.var.name}DistrTarget", f"Target Distribution;{binning.axisTitle}", *binning.astuple),
      binning.var.name,
      "eventWeight",
    ).GetValue()
    # set under- and overflow bins to zero
    targetDistr.SetBinContent(0, 0.0)  # underflow bin
    targetDistr.SetBinContent(targetDistr.GetNbinsX() + 1, 0.0)  # overflow bin
  elif isinstance(targetDistrFrom, MomentResultsKinematicBinning):
    # construct target distribution from H_0(0, 0) values in kinematic bins
    targetDistr = ROOT.TH1D(f"{binning.var.name}DistrTarget", f"#it{{H}}_{{0}}(0, 0);{binning.axisTitle}", *binning.astuple)
    H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
    for momentResultsForBin in targetDistrFrom:
      binCenter = momentResultsForBin.binCenters[binning.var]
      targetDistr.SetBinContent(targetDistr.FindBin(binCenter), momentResultsForBin[H000Index].real[0])
  else:
    raise TypeError(f"Invalid {type(targetDistrFrom)=}. Must be str or MomentResultsKinematicBinning.")
  # reweight data
  originalColumns = list(dataToWeight.GetColumnNames())
  reweightedData = reweightData(
    dataToWeight = dataToWeight,
    treeName     = treeName,
    variableName = binning.var.name,
    targetDistr  = targetDistr,
  )
  print(f"Writing reweighted data to file '{outFileName}'")
  reweightedData.Snapshot(treeName, outFileName, originalColumns if not outputColumns else outputColumns)
  if True:
  # if False:
    # overlay target distribution and distribution after reweighting
    reweightedDistr = reweightedData.Histo1D(
      ROOT.RDF.TH1DModel(f"{binning.var.name}DistrReweighted", "Weighted MC", *binning.astuple),
      binning.var.name,
    ).GetValue()
    targetDistr.Scale(reweightedDistr.Integral() / targetDistr.Integral())
    histStack = ROOT.THStack(f"{binning.var.name}DataAndMc", f";{binning.axisTitle};Count")
    histStack.Add(targetDistr)
    histStack.Add(reweightedDistr)
    targetDistr.SetLineColor  (ROOT.kRed + 1)
    targetDistr.SetMarkerColor(ROOT.kRed + 1)
    reweightedDistr.SetLineColor  (ROOT.kBlue + 1)
    reweightedDistr.SetMarkerColor(ROOT.kBlue + 1)
    canv = ROOT.TCanvas()
    histStack.Draw("NOSTACK")
    canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
    canv.SaveAs(f"{outFileName}.{binning.var.name}.pdf")


@dataclass
class SubSystemInfo:
  """Stores information about a two-body subsystem (particle pair)"""
  pairLabel:         str  # label for particle pair (e.g. "PiPi" for pi+ pi- pair)
  lvALabel:          str  # label of Lorentz-vector of daughter A (analyzer)
  lvBLabel:          str  # label of Lorentz-vector of daughter B
  lvRecoilLabel:     str  # label of Lorentz-vector of recoil particle
  ATLatexLabel:      str = ""  # optional LaTeX label for particle A (analyzer)
  BTLatexLabel:      str = ""  # optional LaTeX label for particle B
  recoilTLatexLabel: str = ""  # optional LaTeX label for recoil particle
  pairTLatexLabel:   str = ""  # optional LaTeX label for particle pair (e.g. "#pi#pi" for pi+ pi- pair)

class InputDataType(Enum):  #TODO use AnalysisConfig.DataType instead
  REAL_DATA             = 0
  GENERATED_PHASE_SPACE = 3
  ACCEPTED_PHASE_SPACE  = 4

@dataclass
class DataSetInfo:
  """Stores information about a data set"""
  subsystem:            SubSystemInfo
  inputType:            InputDataType
  inputFormat:          InputDataFormat
  dataPeriod:           str
  tBinLabel:            str
  inputFileNames:       tuple[str, ...] | tuple[tuple[str, ...], tuple[str, ...]]  # either a tuple of input file names for MC or a tuple with 2 tuples of input file names for real data (signal region, background region)
  inputTreeName:        str
  outputFileName:       str
  outputTreeName:       str
  outputColumns:        tuple[str, ...]
  beamPolLabel:         str                = ""
  beamPolInfo:          BeamPolInfo | None = None  # photon beam polarization
  additionalColumnDefs: dict[str, str]     = field(default_factory=dict)
  additionalFilterDefs: list[str]          = field(default_factory=list)


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  timer.start("Total execution time")
  ROOT.gROOT.SetBatch(True)
  ROOT.EnableImplicitMT()
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("./rootlogon.C") == 0, "Error loading './rootlogon.C'"
  setupPlotStyle()

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  frame = CoordSysType.HF  # helicity frame, i.e. z_HF = -p_recoil
  subsystems: list[SubSystemInfo] = []
  dataSets:   list[DataSetInfo]   = []

  # set up polarized pi+pi- data
  if False:
    subsystems      = [  # particle pairs to analyze; particle A is the analyzer
      SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim",    lvRecoilLabel = "recoil"),
      # SubSystemInfo(pairLabel = "PipP", lvALabel = "pip", lvBLabel = "recoil", lvRecoilLabel = "pim"   ),
      # SubSystemInfo(pairLabel = "PimP", lvALabel = "pim", lvBLabel = "recoil", lvRecoilLabel = "pip"   ),
    ]
    dataDirBaseName = "./dataPhotoProdPiPi/polarized"
    dataPeriods     = (
      # "2017_01",
      "2017_01_ver05",  #!NOTE! SDME analysis: 0.60 < m_pipi < 0.88 GeV
      # "2018_08",
    )
    tBinLabels      = (
      "tbin_0.100_0.114",  # lowest |t| bin of SDME analysis  #TODO actual upper limit seems to 0.11364635 GeV^2
      # "tbin_0.1_0.2",
      # "tbin_0.2_0.3",
      # "tbin_0.3_0.4",
      # "tbin_0.4_0.5",
    )
    beamPolLabels   = (
      "PARA_0",
      # "PARA_135",
      # "PERP_45",
      # "PERP_90",
      # "AMO",
    )
    inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in AmpTools format
      InputDataType.REAL_DATA             : InputDataFormat.AMPTOOLS,
      InputDataType.ACCEPTED_PHASE_SPACE  : InputDataFormat.AMPTOOLS,
      InputDataType.GENERATED_PHASE_SPACE : InputDataFormat.AMPTOOLS,
    }
    # outputColumnsUnpolarized = ("cosTheta", "theta", "phi", "phiDeg", "mass", "minusT")
    # outputColumnsPolarized   = (("beamPol", "beamPolPhiLabDeg", "Phi", "PhiDeg")
    outputColumnsUnpolarized = ("theta", "phi", "mass", "minusT")
    outputColumnsPolarized   = ("beamPol", "beamPolPhiLabDeg", "Phi")
    additionalColumnDefs     = {}
    additionalFilterDefs     = []
    if False:  # cut away forward tracks in reconstructed data
      lvs = lorentzVectors(dataFormat = InputDataFormat.ALEX)
      additionalColumnDefs = {
        "DistFdcPip": f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvs['pip']})",
        "DistFdcPim": f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvs['pim']})",
      }
      additionalFilterDefs = ["(DistFdcPip > 4) and (DistFdcPim > 4)"]  # require minimum distance of tracks at FDC position [cm]
    # reweightMinusTDistribution = True
    reweightMinusTDistribution = False

    for dataPeriod in dataPeriods:
      print(f"Setting up data period '{dataPeriod}':")
      for tBinLabel in tBinLabels:
        print(f"Setting up t bin '{tBinLabel}':")
        for subsystem in subsystems:
          print(f"Setting up subsystem '{subsystem}':")
          inputDataDirBaseName  = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/Alex"
          outputDataDirBaseName = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}"
          os.makedirs(outputDataDirBaseName, exist_ok = True)
          for beamPolLabel in beamPolLabels:
            beamPolInfo = BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel]
            print(f"Setting up beam-polarization orientation '{beamPolLabel}'"
                  + (f": pol = {beamPolInfo.pol:.4f}, PhiLab = {beamPolInfo.PhiLab:.1f} deg" if beamPolInfo is not None else ""))
            for inputDataType, inputDataFormat in inputDataFormats.items():
              print(f"Setting up input data type '{inputDataType}' with format '{inputDataFormat}':")
              outputColumns = outputColumnsUnpolarized + (() if beamPolInfo is None else outputColumnsPolarized)
              dataSet = DataSetInfo(
                subsystem            = subsystem,
                inputType            = inputDataType,
                inputFormat          = inputDataFormat,
                dataPeriod           = dataPeriod,
                tBinLabel            = tBinLabel,
                beamPolLabel         = beamPolLabel,
                beamPolInfo          = beamPolInfo,
                inputFileNames       = (
                  ((f"{inputDataDirBaseName}/amptools_tree_signal_{beamPolLabel}.root", ),  # real data: signal and background
                   (f"{inputDataDirBaseName}/amptools_tree_bkgnd_{beamPolLabel}.root",  )) if inputDataType == InputDataType.REAL_DATA else
                  (f"{inputDataDirBaseName}/amptools_tree_accepted*.root",               ) if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  # (f"{inputDataDirBaseName}/amptools_tree_truthAccepted*.root",          ) if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  (f"{inputDataDirBaseName}/amptools_tree_thrown*.root",                 )  # inputDataType == InputDataType.GENERATED_PHASE_SPACE
                ),
                inputTreeName        = "kin",
                outputFileName       = (
                  f"{outputDataDirBaseName}/data_flat_{beamPolLabel}.root"           if inputDataType == InputDataType.REAL_DATA else
                  f"{outputDataDirBaseName}/phaseSpace_acc_flat_{beamPolLabel}.root" if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  # f"{outputDataDirBaseName}/phaseSpace_accTruth_flat_{beamPolLabel}.root" if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  f"{outputDataDirBaseName}/phaseSpace_gen_flat_{beamPolLabel}.root"  # inputDataType == InputDataType.GENERATED_PHASE_SPACE
                ),
                outputTreeName       = subsystem.pairLabel,
                outputColumns        = (
                  outputColumns + ("eventWeight", ) if inputDataType == InputDataType.REAL_DATA else
                  outputColumns  # no event weights for MC data
                ),
                additionalColumnDefs = (
                  additionalColumnDefs if inputDataType == InputDataType.REAL_DATA or inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  {}  # no additional variables for MC truth
                ),
                additionalFilterDefs = (
                  additionalFilterDefs if inputDataType == InputDataType.REAL_DATA or inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  []  # no additional selection cuts for MC truth
                ),
              )
              dataSets.append(dataSet)

  # setup unpolarized pi+pi- data
  if False:
    dataDirBaseName           = "./unpolarized"
    dataPeriods           = (
      "2017_01",
      "2018_08",
    )
    tBinLabels            = ("tbin_0.4_0.5", )
    outputColumns         = ("cosTheta", "theta", "phi", "phiDeg", "mass", "minusT")
    additionalColumnDefs  = {}
    additionalFilterDefs  = []
    inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in AmpTools format
      InputDataType.REAL_DATA             : InputDataFormat.ALEX,
      InputDataType.ACCEPTED_PHASE_SPACE  : InputDataFormat.ALEX,
      InputDataType.GENERATED_PHASE_SPACE : InputDataFormat.AMPTOOLS,
    }
    #TODO merge with loop for polarized data sets and move into function
    for dataPeriod in dataPeriods:
      print(f"Setting up data period '{dataPeriod}':")
      for tBinLabel in tBinLabels:
        print(f"Setting up t bin '{tBinLabel}':")
        for subsystem in subsystems:
          print(f"Setting up subsystem '{subsystem}':")
          inputDataDirBaseName  = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/Alex"
          outputDataDirBaseName = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}"
          os.makedirs(outputDataDirBaseName, exist_ok = True)
          for inputDataType, inputDataFormat in inputDataFormats.items():
            print(f"Setting up input data type '{inputDataType}' with format '{inputDataFormat}':")
            dataSet = DataSetInfo(  # real data (signal + background)
              subsystem            = subsystem,
              inputType            = inputDataType,
              inputFormat          = inputDataFormat,
              dataPeriod           = dataPeriod,
              tBinLabel            = tBinLabel,
              inputFileNames       = (
                ((f"{inputDataDirBaseName}/amptools_tree_signal.root", ),  # real data: signal and background
                 (f"{inputDataDirBaseName}/amptools_tree_bkgnd.root",  ) ) if inputDataType == InputDataType.REAL_DATA else
                (f"{inputDataDirBaseName}/amptools_tree_accepted*.root", ) if inputDataType == InputDataType.mcReco else
                (f"{inputDataDirBaseName}/amptools_tree_thrown*.root",   )  # inputDataType == InputDataType.GENERATED_PHASE_SPACE
              ),
              inputTreeName        = "kin",
              outputFileName       = (
                f"{outputDataDirBaseName}/data_flat.root"           if inputDataType == InputDataType.REAL_DATA else
                f"{outputDataDirBaseName}/phaseSpace_acc_flat.root" if inputDataType == InputDataType.mcReco else
                f"{outputDataDirBaseName}/phaseSpace_gen_flat.root"  # inputDataType == InputDataType.GENERATED_PHASE_SPACE
              ),
              outputTreeName       = subsystem.pairLabel,
                outputColumns        = (
                  outputColumns + ("eventWeight", ) if inputDataType == InputDataType.REAL_DATA else
                  outputColumns  # no event weights for MC data
                ),
                additionalColumnDefs = (
                  additionalColumnDefs if inputDataType == InputDataType.REAL_DATA or inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  {}  # no additional variables for MC truth
                ),
                additionalFilterDefs = (
                  additionalFilterDefs if inputDataType == InputDataType.REAL_DATA or inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  []  # no additional selection cuts for MC truth
                ),
            )
            dataSets.append(dataSet)

  # set up polarized eta pi0 -> 4 gamma data from Nizar's analysis
  if False:
    frame           = CoordSysType.GJ  # Gottfried-Jackson frame, i.e. z_GJ = p_beam
    subsystem       = SubSystemInfo(pairLabel = "EtaPi0", lvALabel = "eta", lvBLabel = "pi0", lvRecoilLabel = "recoil")
    dataDirBaseName = f"./dataPhotoProd{subsystem.pairLabel}/polarized"
    dataPeriods     = (
      "merged",
    )
    tBinLabels      = (
      "t010020",
      "t020032",
      "t032050",
      "t050075",
      "t075100",
    )
    beamPolLabel    = "All"  # input files contain all beam polarization orientations merged together
    beamPolInfo     = BeamPolInfo(  # read beam polarization info from input tree
      pol    = "Pol",
      PhiLab = "BeamAngle",
    )
    inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in AmpTools format
      InputDataType.REAL_DATA             : InputDataFormat.AMPTOOLS,
      InputDataType.ACCEPTED_PHASE_SPACE  : InputDataFormat.AMPTOOLS,
      InputDataType.GENERATED_PHASE_SPACE : InputDataFormat.AMPTOOLS,
    }
    outputColumnsUnpolarized = ("theta", "phi", "mass", "minusT")
    outputColumnsPolarized   = ("beamPol", "beamPolPhiLabDeg", "Phi")
    additionalColumnDefs     = {"eventWeight" : "weightASBS"}  # use this column as event weights
    additionalFilterDefs     = []
    # reweightMinusTDistribution = True
    reweightMinusTDistribution = False

    print(f"Setting up subsystem '{subsystem}':")
    for dataPeriod in dataPeriods:
      print(f"Setting up data period '{dataPeriod}':")
      for tBinLabel in tBinLabels:
        print(f"Setting up t bin '{tBinLabel}':")
        inputDataDirBaseName  = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/Nizar"
        outputDataDirBaseName = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}"
        os.makedirs(outputDataDirBaseName, exist_ok = True)
        for inputDataType, inputDataFormat in inputDataFormats.items():
          print(f"Setting up input data type '{inputDataType}' with format '{inputDataFormat}':")
          outputColumns = outputColumnsUnpolarized + (() if beamPolInfo is None else outputColumnsPolarized)
          dataSet = DataSetInfo(
            subsystem            = subsystem,
            inputType            = inputDataType,
            inputFormat          = inputDataFormat,
            dataPeriod           = dataPeriod,
            tBinLabel            = tBinLabel,
            beamPolLabel         = beamPolLabel,
            beamPolInfo          = beamPolInfo,
            inputFileNames       = (
              (f"{inputDataDirBaseName}/amptools_tree_data_{beamPolLabel}.root",     ) if inputDataType == InputDataType.REAL_DATA else
              (f"{inputDataDirBaseName}/amptools_tree_accepted_{beamPolLabel}.root", ) if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
              (f"{inputDataDirBaseName}/amptools_tree_thrown_{beamPolLabel}.root",   )  # inputDataType == InputDataType.GENERATED_PHASE_SPACE
            ),
            inputTreeName        = "kin",
            outputFileName       = (
              f"{outputDataDirBaseName}/data_flat_{beamPolLabel}.root"           if inputDataType == InputDataType.REAL_DATA else
              f"{outputDataDirBaseName}/phaseSpace_acc_flat_{beamPolLabel}.root" if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
              f"{outputDataDirBaseName}/phaseSpace_gen_flat_{beamPolLabel}.root"  # inputDataType == InputDataType.GENERATED_PHASE_SPACE
            ),
            outputTreeName       = subsystem.pairLabel,
            outputColumns        = (
              outputColumns + ("eventWeight", ) if inputDataType == InputDataType.REAL_DATA else
              outputColumns  # no event weights for MC data
            ),
            additionalColumnDefs = (
              additionalColumnDefs if inputDataType == InputDataType.REAL_DATA or inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
              {}  # no additional variables for MC truth
            ),
            additionalFilterDefs = (
              additionalFilterDefs if inputDataType == InputDataType.REAL_DATA or inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
              []  # no additional selection cuts for MC truth
            ),
          )
          dataSets.append(dataSet)

  # set up unpolarized eta' eta data from Will's analysis
  if True:
    subsystem       = SubSystemInfo(pairLabel = "EtapEta", lvALabel = "etap", lvBLabel = "eta", lvRecoilLabel = "recoil")
    dataDirBaseName = f"./dataPhotoProd{subsystem.pairLabel}/unpolarized"
    dataPeriods     = ("2018_08",)
    tBinLabels      = ("ALLT", )
    beamPolLabel    = "Unpol"
    inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in AmpTools format
      InputDataType.REAL_DATA             : InputDataFormat.FSROOT_RECO,
      InputDataType.ACCEPTED_PHASE_SPACE  : InputDataFormat.FSROOT_RECO,
      InputDataType.GENERATED_PHASE_SPACE : InputDataFormat.FSROOT_TRUTH,
    }
    outputColumnsUnpolarized = ("theta", "phi", "mass", "minusT")
    # reweightMinusTDistribution = True
    reweightMinusTDistribution = False

    print(f"Setting up subsystem '{subsystem}':")
    for dataPeriod in dataPeriods:
      print(f"Setting up data period '{dataPeriod}':")
      for tBinLabel in tBinLabels:
        print(f"Setting up t bin '{tBinLabel}':")
        inputDataDirBaseName  = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/Will"
        outputDataDirBaseName = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}"
        os.makedirs(outputDataDirBaseName, exist_ok = True)
        for inputDataType, inputDataFormat in inputDataFormats.items():
          print(f"Setting up input data type '{inputDataType}' with format '{inputDataFormat}':")
          dataSet = DataSetInfo(
            subsystem      = subsystem,
            inputType      = inputDataType,
            inputFormat    = inputDataFormat,
            dataPeriod     = dataPeriod,
            tBinLabel      = tBinLabel,
            inputFileNames = (
              (f"{inputDataDirBaseName}/tree_data_{beamPolLabel}.root",     ) if inputDataType == InputDataType.REAL_DATA else
              (f"{inputDataDirBaseName}/tree_accepted_{beamPolLabel}.root", ) if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
              (f"{inputDataDirBaseName}/tree_thrown_{beamPolLabel}.root",   )  # inputDataType == InputDataType.GENERATED_PHASE_SPACE  #TODO fix file format
            ),
            inputTreeName  = "ntFSGlueX_100_4000110" if inputDataType == InputDataType.GENERATED_PHASE_SPACE else "kin",
            outputFileName = (
              f"{outputDataDirBaseName}/data_flat_{beamPolLabel}.root"           if inputDataType == InputDataType.REAL_DATA else
              f"{outputDataDirBaseName}/phaseSpace_acc_flat_{beamPolLabel}.root" if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
              f"{outputDataDirBaseName}/phaseSpace_gen_flat_{beamPolLabel}.root"  # inputDataType == InputDataType.GENERATED_PHASE_SPACE
            ),
            outputTreeName = subsystem.pairLabel,
            outputColumns  = (
              outputColumnsUnpolarized if inputDataType == InputDataType.GENERATED_PHASE_SPACE else  # no event weights for MC data
              outputColumnsUnpolarized + ("eventWeight", )
            ),
          )
          dataSets.append(dataSet)

  # process data sets
  for dataSet in dataSets:
    df = None
    if dataSet.inputType == InputDataType.REAL_DATA:
      # combine signal and background region data with correct event weights into one RDataFrame
      outputDataDirBaseName = os.path.dirname(dataSet.outputFileName)
      df = (
        getDataFrameWithCorrectEventWeights(
          dataSigRegionFileNames  = dataSet.inputFileNames[0],
          dataBkgRegionFileNames  = dataSet.inputFileNames[1],
          treeName                = dataSet.inputTreeName,
          friendSigRegionFileName = f"{outputDataDirBaseName}/data_sig_{dataSet.beamPolLabel}.root.weights",
          friendBkgRegionFileName = f"{outputDataDirBaseName}/data_bkg_{dataSet.beamPolLabel}.root.weights",
        ) if len(dataSet.inputFileNames) == 2 else
        ROOT.RDataFrame(dataSet.inputTreeName, dataSet.inputFileNames[0])  # if only one tuple of input file names is given, assume that it already contains combined signal and background data with correct event weights
      )
    elif dataSet.inputType == InputDataType.ACCEPTED_PHASE_SPACE or dataSet.inputType == InputDataType.GENERATED_PHASE_SPACE:
      # read all MC files into one RDataFrame
      df = ROOT.RDataFrame(dataSet.inputTreeName, dataSet.inputFileNames)
    else:
      raise RuntimeError(f"Unsupported input data type '{dataSet.inputType}'")
    print(f"Converting {dataSet.inputType} data with {dataSet.inputFormat} format for '{dataSet.subsystem.pairLabel}' subsystem, "
          f"'{dataSet.dataPeriod}' period, '{dataSet.tBinLabel}' t bin, and {dataSet.beamPolLabel or 'no'} beam polarization from file(s) {dataSet.inputFileNames}")
    lvs = lorentzVectors(dataFormat = dataSet.inputFormat)
    df = defineDataFrameColumns(
      df                   = df,
      lvTarget             = lvs["target"],
      lvBeam               = lvs["beam"],  #TODO "beam" for GJ pi+- p baryon system is p_target
      lvRecoil             = lvs[dataSet.subsystem.lvRecoilLabel],
      lvA                  = lvs[dataSet.subsystem.lvALabel],
      lvB                  = lvs[dataSet.subsystem.lvBLabel],
      beamPolInfo          = dataSet.beamPolInfo,
      frame                = frame,
      additionalColumnDefs = dataSet.additionalColumnDefs,
      additionalFilterDefs = dataSet.additionalFilterDefs,
    ).Filter(('if (rdfentry_ == 0) { std::cout << "Running event loop" << std::endl; } return true;'))  # no-op filter that logs when event loop is running
    if reweightMinusTDistribution and dataSet.inputType == InputDataType.ACCEPTED_PHASE_SPACE:
      #TODO this is currently only implemented for the bin 0.1 < |t| < 0.2 GeV^2/c^2
      # reweight -t distribution to match that of real data
      outputFileNameReweighted = dataSet.outputFileName.replace(".root", ".reweighted_minusT.root")
      reweightKinDistribution(
        dataToWeight    = df,
        treeName        = dataSet.outputTreeName,
        binning         = HistAxisBinning(
          nmbBins = 50, minVal = 0.1, maxVal = 0.2,
          _var = KinematicBinningVariable(name= "minusT", label = "#minus#it{t}", unit = "GeV^{2}/#it{c}^{2}", nmbDigits = 3),
        ),
        targetDistrFrom = f"{outputDataDirBaseName}/data_flat_{dataSet.beamPolLabel}.root",
        outFileName     = outputFileNameReweighted,
        outputColumns   = dataSet.outputColumns,
      )
    else:
      print(f"Writing converted data to file '{dataSet.outputFileName}'")
      df.Snapshot(dataSet.outputTreeName, dataSet.outputFileName, dataSet.outputColumns)

  timer.stop("Total execution time")
  print(timer.summary)
