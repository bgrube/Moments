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
import pandas as pd

import os

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class BeamPolInfo:
  """Stores information about beam polarization for a specific orientation"""
  pol:    float  # photon-beam polarization magnitude
  PhiLab: float  # azimuthal angle of photon beam polarization in lab frame [deg]

# polarization values from Version 9 of `makePolVals` tool from https://halldweb.jlab.org/wiki-private/index.php/TPOL_Polarization
# beam polarization angles in lab frame taken from `Lab Phi` column of tables 2 to 5 in GlueX-doc-3977
BEAM_POL_INFOS: dict[str, dict[str, BeamPolInfo | None]] = {  # data period : {beam-polarization orientation : BeamPolInfo(...)}
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

# C++ function to calculate angles for moment analysis
CPP_CODE_ANGLES_GLUEX_AMPTOOLS = """
// calculates helicity angles and azimuthal angle between photon polarization and production plane in lab frame
// for reaction beam + target -> resonance + recoil with resonance -> A + B
// angles are returned as vector (cos(theta_HF), phi_HF [rad], Phi [rad])
// code taken from GlueX AmpTools: https://github.com/JeffersonLab/halld_sim/blob/39b18bdbab88192275fed57fda161f9a52d04422/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
std::vector<Double32_t>
twoBodyHelicityFrameAngles(
	const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam,    // 4-momentum of beam [GeV]
	const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,  // 4-momentum of recoil [GeV]
	const double PxPA,     const double PyPA,     const double PzPA,     const double EPA,      // 4-momentum of particle A (analyzer) [GeV]
	const double PxPB,     const double PyPB,     const double PzPB,     const double EPB,      // 4-momentum of particle B [GeV]
	const double beamPolPhiLab = 0  // azimuthal angle of photon beam polarization in lab [deg]
) {
	// 4-vectors in lab frame
	const TLorentzVector beam  (PxBeam,   PyBeam,   PzBeam,   EBeam);
	const TLorentzVector recoil(PxRecoil, PyRecoil, PzRecoil, ERecoil);
	const TLorentzVector pA    (PxPA,     PyPA,     PzPA,     EPA);
	const TLorentzVector pB    (PxPB,     PyPB,     PzPB,     EPB);
	// boost 4-vectors to resonance rest frame
	const TLorentzVector resonance = pA + pB;
	// const TVector3 boost = -resonance.BoostVector();
	// beam  (boost);
	// recoil(boost);
	// pA    (boost);
	const TLorentzRotation resonanceBoost(-resonance.BoostVector());
	const TLorentzVector beamRF   = resonanceBoost * beam;
	const TLorentzVector recoilRF = resonanceBoost * recoil;
	const TLorentzVector pARF     = resonanceBoost * pA;
	// define axes of coordinate system
	// const TVector3 y = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();  // normal to the production plane from lab momenta
	const TVector3 y = beam.Vect().Cross(-recoil.Vect()).Unit();  // normal to the production plane from lab momenta
	const TVector3 z = -recoilRF.Vect().Unit();  // helicity frame: z axis opposite to recoil proton in resonance rest frame
	const TVector3 x = y.Cross(z).Unit();  // right-handed coordinate system
	// calculate helicity-frame angles of particle A and angle between polarization and production plane
	const TVector3 pAHF(pARF.Vect() * x, pARF.Vect() * y, pARF.Vect() * z);  // vector of particle A (analyzer) in helicity frame
	const Double32_t cosThetaHF = pAHF.CosTheta();  // polar angle of particle A
	const Double32_t phiHF      = pAHF.Phi();  // azimuthal angle of particle A [rad]
	const TVector3 eps(1, 0, 0);  // reference beam polarization vector at 0 degrees in lab frame
	const Double32_t Phi = fixAzimuthalAngleRange(  // angle between photon polarization and production plane in lab frame [rad]
		beamPolPhiLab * TMath::DegToRad() + atan2(y * eps, beam.Vect().Unit() * (eps.Cross(y))));
	return std::vector<Double32_t>{cosThetaHF, phiHF, Phi};
}
"""


# C++ function to calculate azimuthal angle of photon polarization vector
CPP_CODE_BEAM_POL_PHI = """
// returns azimuthal angle of photon polarization vector in lab frame [rad]
// for beam + target -> X + recoil and X -> a + b
//     D                    C
// code taken from https://github.com/JeffersonLab/halld_sim/blob/538677ee1347891ccefa5780e01b158e035b49b1/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
double
beamPolPhi(
	const double PxPC, const double PyPC, const double PzPC, const double EnPC,  // 4-momentum of recoil [GeV]
	const double PxPD, const double PyPD, const double PzPD, const double EnPD,  // 4-momentum of beam [GeV]
	const double beamPolPhiLab = 0  // azimuthal angle of photon beam polarization in lab [deg]
) {
	const TLorentzVector recoil(PxPC, PyPC, PzPC, EnPC);
	const TLorentzVector beam  (PxPD, PyPD, PzPD, EnPD);
	const TVector3 yAxis = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();  // normal of production plane in lab frame
	const TVector3 eps(1, 0, 0);  // reference beam polarization vector at 0 degrees in lab frame
	const double Phi = beamPolPhiLab * TMath::DegToRad() + atan2(yAxis.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(yAxis)));  // angle between photon polarization and production plane in lab frame [rad]
	return fixAzimuthalAngleRange(Phi);
}
"""

# C++ function to flip y axis if needed
CPP_CODE_FLIPYAXIS = """
// flips y axis `flip` is true; corresponds to shifting azimuthal angle by 180 degrees
double
flipYAxis(
	double     phi,
	const bool flip = false
) {
	if (not flip) {
		return phi;
	}
  //TODO is this correct? or should it be phi -> -phi?
	phi += TMath::Pi();
	return fixAzimuthalAngleRange(phi);
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
    lvs["pip"   ] = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # pi+
    lvs["pim"   ] = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # pi-
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
  else:
    raise RuntimeError(f"Unsupported data format type '{dataFormat}'")
  return lvs


class CoordSysType(Enum):
  HF = 0  # helicity frame
  GJ = 1  # Gottfried-Jackson frame

def DefineOverwrite(
  df:         ROOT.RDataFrame,
  colName:    str,
  newFormula: str,
) -> ROOT.RDataFrame:
  """If column `colName` exists, redefines it using formula `newExpr`"""
  if not df.HasColumn(colName):
    # print(f"Defining column '{colName}' = '{newExpr}'")
    df = df.Define(colName, newFormula)
  else:
    print(f"Redefining column '{colName}' to '{newFormula}'")
    df = df.Redefine(colName, newFormula)
  return df

def defineDataFrameColumns(
  df:                   ROOT.RDataFrame,
  lvTarget:             str,  # function-argument list with Lorentz-vector components of target proton
  lvBeam:               str,  # function-argument list with Lorentz-vector components of beam photon
  lvRecoil:             str,  # function-argument list with Lorentz-vector components of recoil proton
  lvA:                  str,  # function-argument list with Lorentz-vector components of daughter A (analyzer)
  lvB:                  str,  # function-argument list with Lorentz-vector components of daughter B
  beamPolInfo:          BeamPolInfo | None = None,  # photon beam polarization
  frame:                CoordSysType       = CoordSysType.HF,  # reference frame for angle definitions
  flipYAxis:            bool               = False,  # if set y-axis of reference frame is inverted
  additionalColumnDefs: dict[str, str]     = {},  # additional columns to define
  additionalFilterDefs: list[str]          = [],  # additional filter conditions to apply
  colNameSuffix:        str                = "",  # suffix appended to column names
) -> ROOT.RDataFrame:
  """Defines columns for (A, B) pair mass, squared four-momentum transferred from beam to recoil, and angles (cos(theta), phi) of particle A in X rest frame for reaction beam + target -> X + recoil with X -> A + B using the given Lorentz-vector components"""
  print(f"Defining angles in '{frame}' frame using '{lvA}' as analyzer and '{lvRecoil}' as recoil")
  angColNameSuffix = frame.name + colNameSuffix if colNameSuffix else ""  # column name suffixes are only used for plotting
  if frame == CoordSysType.HF:
    df = (
      # use z_HF = -p_recoil, A as analyzer, and y_HF = (p_beam x p_recoil)
      df.Define(f"angles{angColNameSuffix}",   f"twoBodyHelicityFrameAngles({lvBeam}, {lvRecoil}, {lvA}, {lvB}, {'0' if beamPolInfo is None else beamPolInfo.PhiLab})")
        .Define(f"cosTheta{angColNameSuffix}", f"angles{angColNameSuffix}[0]")
        .Define(f"phi{angColNameSuffix}",      f"angles{angColNameSuffix}[1]")
    )
  elif frame == CoordSysType.GJ:
    #TODO use local C++ code instead FSRoot functions
    df = (
      # use z_GJ = p_beam, A as analyzer, and y_GJ = (p_beam x p_recoil), if flipYAxis is False else -y_GJ
      df.Define(f"cosTheta{angColNameSuffix}", f"(Double32_t)FSMath::gjcostheta({lvA}, {lvB}, {lvBeam})")  #!NOTE! signature is different from FSMath::helcostheta (see FSBasic/FSMath.h)
        .Define(f"phi{angColNameSuffix}",      f"(Double32_t)flipYAxis(FSMath::gjphi({lvA}, {lvB}, {lvRecoil}, {lvBeam}), {'true' if flipYAxis else 'false'})")
    )
  else:
    raise ValueError(f"Unsupported coordinate system type '{frame}'")
  df = (
    df.Define(f"theta{angColNameSuffix}",  f"(Double32_t)std::acos(cosTheta{angColNameSuffix})")
      .Define(f"phiDeg{angColNameSuffix}", f"(Double32_t)(phi{angColNameSuffix} * TMath::RadToDeg())")  #TODO only write out columns actually needed for moment calculation; move everything else to plotting module
  )
  # allow for redefinition of already existing columns with identical formula if function is called for several frames
  df = DefineOverwrite(df, f"mass{colNameSuffix}",   f"(Double32_t)massPair({lvA}, {lvB})")
  df = DefineOverwrite(df, f"minusT{colNameSuffix}", f"(Double32_t)-mandelstamT({lvTarget}, {lvRecoil})")
  if beamPolInfo is not None:
    df = DefineOverwrite(df, f"beamPol{colNameSuffix}",       f"(Double32_t){beamPolInfo.pol}")
    df = DefineOverwrite(df, f"beamPolPhiLab{colNameSuffix}", f"(Double32_t){beamPolInfo.PhiLab}")
    #TODO Use Phi from `angles` column
    df = DefineOverwrite(df, f"Phi{colNameSuffix}",           f"(Double32_t)beamPolPhi({lvRecoil}, {lvBeam}, beamPolPhiLab{colNameSuffix})")
    df = DefineOverwrite(df, f"PhiDeg{colNameSuffix}",        f"(Double32_t)(Phi{colNameSuffix} * TMath::RadToDeg())")
  if additionalColumnDefs:
    for columnName, columnFormula in additionalColumnDefs.items():
      print(f"Defining additional column '{columnName}' = '{columnFormula}'")
      df = df.Define(columnName, columnFormula)
  if additionalFilterDefs:
    for filterDef in additionalFilterDefs:
      print(f"Applying additional filter '{filterDef}'")
      df = df.Filter(filterDef)
  return df


DATA_TCHAINS: list[ROOT.TChain] = []  # use global variable to avoid garbage collection
def getDataFrameWithCorrectEventWeights(
  dataSigRegionFileNames:    Sequence[str],
  dataBkgRegionFileNames:    Sequence[str],
  treeName:                  str,
  friendSigRegionFileName:   str  = "data_sig.root.weights",
  friendBkgRegionFileName:   str  = "data_bkg.root.weights",
  forceOverwriteFriendFiles: bool = True,
) -> ROOT.RDataFrame:
  """Create friend trees with correct event weights and attach them to data tree"""
  # write corrected weights into friend trees
  for dataFileNames, weightFormula, friendFileName in (
    (dataSigRegionFileNames,  "Weight", friendSigRegionFileName),
    (dataBkgRegionFileNames, "-Weight", friendBkgRegionFileName),
  ):
    print(f"Loading real-data file(s) {dataFileNames}")
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


@dataclass
class SubSystemInfo:
  """Stores information about a two-body subsystem (particle pair)"""
  pairLabel:       str  # label for particle pair (e.g. "PiPi" for pi+ pi- pair)
  lvALabel:        str  # label of Lorentz-vector of daughter A (analyzer)
  lvBLabel:        str  # label of Lorentz-vector of daughter B
  lvRecoilLabel:   str  # label of Lorentz-vector of recoil particle
  pairTLatexLabel: str = ""  # optional LaTeX label for particle pair (e.g. "#pi#pi" for pi+ pi- pair)

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
  ROOT.gROOT.SetBatch(True)
  ROOT.EnableImplicitMT()
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_ANGLES_GLUEX_AMPTOOLS)
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  # ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  frame = CoordSysType.HF  # helicity frame, i.e. z_HF = -p_recoil
  subsystems: tuple[SubSystemInfo, ...] = (  # particle pairs to analyze; particle A is the analyzer
    SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim",    lvRecoilLabel = "recoil"),
    # SubSystemInfo(pairLabel = "PipP", lvALabel = "pip", lvBLabel = "recoil", lvRecoilLabel = "pim"   ),
    # SubSystemInfo(pairLabel = "PimP", lvALabel = "pim", lvBLabel = "recoil", lvRecoilLabel = "pip"   ),
  )
  dataSets: list[DataSetInfo] = []

  # set up polarized pi+pi- real data
  if True:
    dataDirName   = "./polarized"
    dataPeriods   = (
      "2017_01",
      # "2018_08",
    )
    tBinLabels    = (
      "tbin_0.1_0.2",
      "tbin_0.2_0.3",
      # "tbin_0.3_0.4",
      # "tbin_0.4_0.5",
    )
    beamPolLabels = (
      "PARA_0",
      "PARA_135",
      "PERP_45",
      "PERP_90",
      "AMO",
    )
    inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in ampTools format
      InputDataType.REAL_DATA             : InputDataFormat.AMPTOOLS,
      InputDataType.ACCEPTED_PHASE_SPACE  : InputDataFormat.AMPTOOLS,
      InputDataType.GENERATED_PHASE_SPACE : InputDataFormat.AMPTOOLS,
    }
    # outputColumnsUnpolarized = ("cosTheta", "theta", "phi", "phiDeg", "mass", "minusT")
    # outputColumnsPolarized   = (("beamPol", "beamPolPhiLab", "Phi", "PhiDeg")
    outputColumnsUnpolarized = ("theta", "phi", "mass")
    outputColumnsPolarized   = ("beamPol", "beamPolPhiLab", "Phi")
    additionalColumnDefs     = {}
    additionalFilterDefs     = []
    if False:  # cut away forward tracks in reconstructed data
      lvs = lorentzVectors(dataFormat = InputDataFormat.ALEX)
      additionalColumnDefs = {
        "DistFdcPip": f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvs['pip']})",
        "DistFdcPim": f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvs['pim']})",
      }
      additionalFilterDefs = ["(DistFdcPip > 4) and (DistFdcPim > 4)"]  # require minimum distance of tracks at FDC position [cm]

    for dataPeriod in dataPeriods:
      print(f"Setting up data period '{dataPeriod}':")
      for tBinLabel in tBinLabels:
        print(f"Setting up t bin '{tBinLabel}':")
        for subsystem in subsystems:
          print(f"Setting up subsystem '{subsystem}':")
          inputDataDirName  = f"{dataDirName}/{dataPeriod}/{tBinLabel}/Alex"
          outputDataDirName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}"
          os.makedirs(outputDataDirName, exist_ok = True)
          for beamPolLabel in beamPolLabels:
            beamPolInfo = BEAM_POL_INFOS[dataPeriod][beamPolLabel]
            print(f"Setting up beam-polarization orientation '{beamPolLabel}'"
                  + (f": pol = {beamPolInfo.pol:.4f}, PhiLab = {beamPolInfo.PhiLab:.1f} deg" if beamPolInfo is not None else ""))
            for inputDataType, inputDataFormat in inputDataFormats.items():
              print(f"Setting up input data type '{inputDataType}' with format '{inputDataFormat}':")
              outputColumns = outputColumnsUnpolarized + () if beamPolInfo is None else outputColumnsPolarized
              dataSet = DataSetInfo(
                subsystem            = subsystem,
                inputType            = inputDataType,
                inputFormat          = inputDataFormat,
                dataPeriod           = dataPeriod,
                tBinLabel            = tBinLabel,
                beamPolLabel         = beamPolLabel,
                beamPolInfo          = beamPolInfo,
                inputFileNames       = (
                  ((f"{inputDataDirName}/amptools_tree_signal_{beamPolLabel}.root", ),  # real data: signal and background
                   (f"{inputDataDirName}/amptools_tree_bkgnd_{beamPolLabel}.root",  )) if inputDataType == InputDataType.REAL_DATA else
                   (f"{inputDataDirName}/amptools_tree_accepted*.root", )              if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  #  (f"{inputDataDirName}/amptools_tree_truthAccepted*.root", )         if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                   (f"{inputDataDirName}/amptools_tree_thrown*.root", )                 # inputDataType == InputDataType.mcTruth
                ),
                inputTreeName        = "kin",
                outputFileName       = (
                  f"{outputDataDirName}/data_flat_{beamPolLabel}.root"           if inputDataType == InputDataType.REAL_DATA else
                  f"{outputDataDirName}/phaseSpace_acc_flat_{beamPolLabel}.root" if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  # f"{outputDataDirName}/phaseSpace_accTruth_flat_{beamPolLabel}.root" if inputDataType == InputDataType.ACCEPTED_PHASE_SPACE else
                  f"{outputDataDirName}/phaseSpace_gen_flat_{beamPolLabel}.root"  # inputDataType == InputDataType.mcTruth
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

  # setup unpolarized pi+pi- real data
  if False:
    dataDirName           = "./unpolarized"
    dataPeriods           = (
      "2017_01",
      "2018_08",
    )
    tBinLabels            = ("tbin_0.4_0.5", )
    outputColumns         = ("cosTheta", "theta", "phi", "phiDeg", "mass", "minusT")
    additionalColumnDefs  = {}
    additionalFilterDefs  = []
    inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in ampTools format
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
          inputDataDirName  = f"{dataDirName}/{dataPeriod}/{tBinLabel}/Alex"
          outputDataDirName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}"
          os.makedirs(outputDataDirName, exist_ok = True)
          for inputDataType, inputDataFormat in inputDataFormats.items():
            print(f"Setting up input data type '{inputDataType}' with format '{inputDataFormat}':")
            dataSet = DataSetInfo(  # real data (signal + background)
              subsystem            = subsystem,
              inputType            = inputDataType,
              inputFormat          = inputDataFormat,
              dataPeriod           = dataPeriod,
              tBinLabel            = tBinLabel,
              inputFileNames       = (
                ((f"{inputDataDirName}/amptools_tree_signal.root", ),  # real data: signal and background
                 (f"{inputDataDirName}/amptools_tree_bkgnd.root",  ))   if inputDataType == InputDataType.REAL_DATA else
                 (f"{inputDataDirName}/amptools_tree_accepted*.root", ) if inputDataType == InputDataType.mcReco else
                 (f"{inputDataDirName}/amptools_tree_thrown*.root", )    # inputDataType == InputDataType.mcTruth
              ),
              inputTreeName        = "kin",
              outputFileName       = (
                f"{outputDataDirName}/data_flat.root"           if inputDataType == InputDataType.REAL_DATA else
                f"{outputDataDirName}/phaseSpace_acc_flat.root" if inputDataType == InputDataType.mcReco else
                f"{outputDataDirName}/phaseSpace_gen_flat.root"  # inputDataType == InputDataType.mcTruth
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

  # process data sets
  for dataSet in dataSets:
    df = None
    if dataSet.inputType == InputDataType.REAL_DATA:
      # combine signal and background region data with correct event weights into one RDataFrame
      outputDataDirName = os.path.dirname(dataSet.outputFileName)
      df = getDataFrameWithCorrectEventWeights(
        dataSigRegionFileNames  = dataSet.inputFileNames[0],
        dataBkgRegionFileNames  = dataSet.inputFileNames[1],
        treeName                = dataSet.inputTreeName,
        friendSigRegionFileName = f"{outputDataDirName}/data_sig_{dataSet.beamPolLabel}.root.weights",
        friendBkgRegionFileName = f"{outputDataDirName}/data_bkg_{dataSet.beamPolLabel}.root.weights",
      )
    elif dataSet.inputType == InputDataType.ACCEPTED_PHASE_SPACE or dataSet.inputType == InputDataType.GENERATED_PHASE_SPACE:
      # read all MC files into one RDataFrame
      df = ROOT.RDataFrame(dataSet.inputTreeName, dataSet.inputFileNames)
    else:
      raise RuntimeError(f"Unsupported input data type '{dataSet.inputType}'")
    print(f"Converting {dataSet.inputType} data with {dataSet.inputFormat} format for {dataSet.subsystem.pairLabel} subsystem, {dataSet.dataPeriod}, {dataSet.tBinLabel}, and {dataSet.beamPolLabel} from file(s) {dataSet.inputFileNames} to file '{dataSet.outputFileName}'")
    lvs = lorentzVectors(dataFormat = dataSet.inputFormat)
    defineDataFrameColumns(
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
    ).Snapshot(dataSet.outputTreeName, dataSet.outputFileName, dataSet.outputColumns)
