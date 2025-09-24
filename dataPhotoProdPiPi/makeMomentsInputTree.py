#!/usr/bin/env python3


from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
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
BEAM_POL_INFOS: dict[str, dict[str, BeamPolInfo]] = {  # data period : {beam orientation : BeamPolInfo(...)}
  "2017_01" : {  # polarization magnitudes obtained by running `.x makePolVals.C(17, 1, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      pol    = 0.3537,
      PhiLab = 1.8,
    ),
    # "PERP_45" : BeamPolInfo(
    #   pol    = 0.3484,
    #   PhiLab = 47.9,
    # ),
    # "PERP_90" : BeamPolInfo(
    #   pol    = 0.3472,
    #   PhiLab = 94.5,
    # ),
    # "PARA_135" : BeamPolInfo(
    #   pol    = 0.3512,
    #   PhiLab = -41.6,
    # ),
  },
  # "2018_01" : {  # polarization magnitudes obtained by running `.x makePolVals.C(18, 1, 0, 75)` in ROOT shell
  #   "PARA_0" : BeamPolInfo(
  #     pol    = 0.3420,
  #     PhiLab = 4.1,
  #   ),
  #   "PERP_45" : BeamPolInfo(
  #     pol    = 0.3474,
  #     PhiLab = 48.5,
  #   ),
  #   "PERP_90" : BeamPolInfo(
  #     pol    = 0.3478,
  #     PhiLab = 94.2,
  #   ),
  #   "PARA_135" : BeamPolInfo(
  #     pol    = 0.3517,
  #     PhiLab = -42.4,
  #   ),
  # },
  # "2018_08" : {  # polarization magnitudes obtained by running `.x makePolVals.C(18, 2, 0, 75)` in ROOT shell
  #   "PARA_0" : BeamPolInfo(
  #     pol    = 0.3563,
  #     PhiLab = 3.3,
  #   ),
  #   "PERP_45" : BeamPolInfo(
  #     pol    = 0.3403,
  #     PhiLab = 48.3,
  #   ),
  #   "PERP_90" : BeamPolInfo(
  #     pol    = 0.3430,
  #     PhiLab = 92.9,
  #   ),
  #   "PARA_135" : BeamPolInfo(
  #     pol    = 0.3523,
  #     PhiLab = -42.1,
  #   ),
  # },
}


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
	double Phi = beamPolPhiLab * TMath::DegToRad() + atan2(yAxis.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(yAxis)));  // angle between photon polarization and production plane in lab frame [rad]
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
	phi += TMath::Pi();
	// ensure [-pi, +pi] range
	while (phi > TMath::Pi()) {
		phi -= TMath::TwoPi();
	}
	while (phi < -TMath::Pi()) {
		phi += TMath::TwoPi();
	}
	return phi;
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


class InputDataFormatType(Enum):
    ampToolsRdReco  = 1  # reconstructed real data in ampTools format
    ampToolsMcReco  = 2  # reconstructed MC data in ampTools format
    ampToolsMcTruth = 3  # MC truth data in ampTools format
    jpacMc          = 4  # MC truth data in JPAC text format
    TLorentzVectors = 5  # data with TLorentzVectors for particle 4-momenta

def lorentzVectors(dataFormat: InputDataFormatType) -> dict[str, str]:
  """Returns Lorentz-vectors for beam photon ("lvBeam"), target proton ("lvTarget"), recoil proton ("lvRecoil"), pi+ ("lvPip"), and pi- ("lvPim")"""
  lvs = {}
  lvs["lvTarget"] = "0, 0, 0, 0.938271999359130859375"  # proton mass value from phase-space generator
  if dataFormat == InputDataFormatType.ampToolsRdReco or dataFormat == InputDataFormatType.ampToolsMcReco:
    lvs["lvBeam"  ] = "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()"  # beam photon
    lvs["lvRecoil"] = "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()"     # recoil proton
    lvs["lvPip"   ] = "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()"   # pi+
    lvs["lvPim"   ] = "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()"   # pi-
  elif dataFormat == InputDataFormatType.ampToolsMcTruth:
    lvs["lvBeam"  ] = "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam"           # beam photon
    lvs["lvRecoil"] = "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]"  # recoil proton
    lvs["lvPip"   ] = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # pi+  #TODO not clear whether correct index is 1 or 2
    lvs["lvPim"   ] = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # pi-  #TODO not clear whether correct index is 1 or 2
  elif dataFormat == InputDataFormatType.jpacMc:
    # kinematic variables according to Eq. (1) in BIBRZYCKI et al., PD 111, 014002 (2025)
    # gamma (q) + p (p1) -> pi+ (k1) + pi- (k2) + p (p2)
    # four-momenta are defined as
    #                 (p_x, p_y, p_z, E)
    lvs["lvBeam"  ] = "q1,  q2,  q3,  q0"   # beam photon
    lvs["lvTarget"] = "p11, p12, p13, p10"  # target proton
    lvs["lvRecoil"] = "p21, p22, p23, p20"  # recoil proton
    lvs["lvPip"   ] = "k11, k12, k13, k10"  # pi+
    lvs["lvPim"   ] = "k21, k22, k23, k20"  # pi-
  elif dataFormat == InputDataFormatType.TLorentzVectors:
    lvs["lvBeam"  ] = "lvBeamLab.X(),   lvBeamLab.Y(),   lvBeamLab.Z(),   lvBeamLab.E()"    # beam photon
    lvs["lvTarget"] = "lvTargetLab.X(), lvTargetLab.Y(), lvTargetLab.Z(), lvTargetLab.E()"  # target proton
    lvs["lvRecoil"] = "lvRecoilLab.X(), lvRecoilLab.Y(), lvRecoilLab.Z(), lvRecoilLab.E()"  # recoil proton
    lvs["lvPip"   ] = "lvPipLab.X(),    lvPipLab.Y(),    lvPipLab.Z(),    lvPipLab.E()"     # pi+
    lvs["lvPim"   ] = "lvPimLab.X(),    lvPimLab.Y(),    lvPimLab.Z(),    lvPimLab.E()"     # pi-
  else:
    raise RuntimeError(f"Unsupported data format type '{dataFormat}'")
  return lvs


class CoordSysType(Enum):
    Hf = 1  # helicity frame
    Gj = 2  # Gottfried-Jackson frame

def defineDataFrameColumns(
  df:                       ROOT.RDataFrame,
  lvTarget:                 str,  # function-argument list with Lorentz-vector components of target proton
  lvBeam:                   str,  # function-argument list with Lorentz-vector components of beam photon
  lvRecoil:                 str,  # function-argument list with Lorentz-vector components of recoil proton
  lvA:                      str,  # function-argument list with Lorentz-vector components of daughter A (analyzer)
  lvB:                      str,  # function-argument list with Lorentz-vector components of daughter B
  beamPolInfo:              BeamPolInfo | None = None,  # photon beam polarization
  frame:                    CoordSysType       = CoordSysType.Hf,  # reference frame for angle definitions
  flipYAxis:                bool               = True,  # if set y-axis of reference frame is inverted
  additionalColumnDefs:     dict[str, str]     = {},  # additional columns to define
  additionalFilterDefs:     list[str]          = [],  # additional filter conditions to apply
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns for moments analysis"""
  """Defines formulas for (A, B) pair mass, and angles (cos(theta), phi) of particle A in X rest frame for reaction beam + target -> X + recoil with X -> A + B using the given Lorentz-vector components"""
  print(f"Defining angles in '{frame}' frame and using '{lvA}' as analyzer and '{lvRecoil}' as recoil")
  df = (
    df.Define("cosTheta", "(Double32_t)" + (f"FSMath::helcostheta({lvA}, {lvB}, {lvRecoil})" if frame == CoordSysType.Hf else
                                            f"FSMath::gjcostheta ({lvA}, {lvB}, {lvBeam})"))  #!NOTE! frames have different signatures (see FSBasic/FSMath.h)
      .Define("theta",    "(Double32_t)std::acos(cosTheta)")
      #TODO there seems to be a bug in the way FSRoot calculates phi (at least for the HF frame)
      #     when using the same analyzer as for cosTheta, i.e. pi+, phi is flipped by 180 deg
      #     this difference is seen when comparing to Alex' function and also when comparing to the PWA result
      #     flipping the y-axis, i.e. switching the analyzer to pi- cures this problem
      # switching between pi+ and pi- analyzer flips sign of moments with odd M
      # # use pi+ as analyzer and y_HF/GJ = p_beam x p_recoil
      # .Define("phi",      "(Double32_t)" + (f"FSMath::helphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})" if frame == CoordSysType.Hf else
      #                                       f"FSMath::gjphi ({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})"))
      # #WORKAROUND use pi- as analyzer and y_HF/GJ = p_beam x p_recoil
      # .Define("phi",      "(Double32_t)" + (f"FSMath::helphi({lvB}, {lvA}, {lvRecoil}, {lvBeam})" if frame == CoordSysType.Hf else
      #                                       f"FSMath::gjphi ({lvB}, {lvA}, {lvRecoil}, {lvBeam})"))
      .Define("phi",
        # use A as analyzer and y_HF/GJ = (p_beam x p_recoil), if flipYAxis is False else -yHF
        "(Double32_t)" +
        (      f"flipYAxis(FSMath::helphi({lvA}, {lvB}, {lvRecoil}, {lvBeam}), {'true' if flipYAxis else 'false'})" if CoordSysType.Hf  # use z_HF = -p_recoil
          else f"flipYAxis(FSMath::gjphi ({lvA}, {lvB}, {lvRecoil}, {lvBeam}), {'true' if flipYAxis else 'false'})")                    # use z_GJ = p_beam
      )
      .Define("phiDeg",   "(Double32_t)(phi * TMath::RadToDeg())")
      .Define("mass",    f"(Double32_t)massPair({lvA}, {lvB})")
      .Define("minusT",  f"(Double32_t)-mandelstamT({lvTarget}, {lvRecoil})")
  )
  if beamPolInfo is not None:
    df = (
      df.Define("beamPol",       f"(Double32_t){beamPolInfo.pol}")
        .Define("beamPolPhiLab", f"(Double32_t){beamPolInfo.PhiLab}")
        .Define("Phi",           f"(Double32_t)beamPolPhi({lvRecoil}, {lvBeam}, beamPolPhiLab)")
        .Define("PhiDeg",         "(Double32_t)(Phi * TMath::RadToDeg())")
    )
  if additionalColumnDefs:
    for columnName, columnFormula in additionalColumnDefs.items():
      print(f"Defining additional column '{columnName}' = '{columnFormula}'")
      df = df.Define(columnName, columnFormula)
  if additionalFilterDefs:
    for filterDef in additionalFilterDefs:
      print(f"Applying additional filter '{filterDef}'")
      df = df.Filter(filterDef)
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
  # write corrected weights into friend trees
  for dataFileNames, weightFormula, friendFileName in (
    (dataSigRegionFileNames,  "Weight", friendSigRegionFileName),
    (dataBkgRegionFileNames, "-Weight", friendBkgRegionFileName),
  ):
    if not forceOverwriteFriendFiles and os.path.exists(friendFileName):
      print(f"File '{friendFileName}' already exists, skipping creation of event-weight friend tree")
      continue
    print(f"Creating file '{friendFileName}' that contains friend tree with event weights for file {dataFileNames}")
    ROOT.RDataFrame(treeName, dataFileNames) \
        .Define("eventWeight", weightFormula) \
        .Snapshot(treeName, friendFileName, ["eventWeight"])
  # chain trees for signal and background regions and add friend trees with weights
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
  pairLabel:     str  # label for particle pair (e.g. "PiPi" for pi+ pi- pair)
  lvALabel:      str  # label of Lorentz-vector of daughter A (analyzer)
  lvBLabel:      str  # label of Lorentz-vector of daughter B
  lvRecoilLabel: str  # label of Lorentz-vector of recoil particle

@dataclass
class DataSetInfo:
  """Stores information about a data set"""
  subsystem:            SubSystemInfo
  inputFormat:          InputDataFormatType
  dataPeriod:           str
  tBinLabel:            str
  beamOrientation:      str
  inputFileNames:       tuple[str, ...] | tuple[tuple[str, ...], tuple[str, ...]]  # either a tuple of input file names for MC or a tuple with 2 tuples of input file names for real data (signal region, background region)
  inputTreeName:        str
  outputFileName:       str
  outputTreeName:       str
  outputColumns:        tuple[str, ...]
  additionalColumnDefs: dict[str, str] = field(default_factory=dict)
  additionalFilterDefs: list[str]      = field(default_factory=list)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  frame = CoordSysType.Hf  # helicity frame: use z_HF = -p_recoil

  # polarized data
  inputDataBaseDirName  = "../dataPhotoProdPiPiPol/pipi_gluex_coh"
  tBinLabels            = ("tbin_0.1_0.2", )
  outputDataBaseDirName = "./polarized"
  outputColumns         = ("beamPol", "beamPolPhiLab", "cosTheta", "theta", "phi", "phiDeg", "Phi", "PhiDeg", "mass", "minusT")
  additionalColumnDefs  = {}
  additionalFilterDefs  = []
  if False:  # cut away forward tracks in reconstructed data
    lvs = lorentzVectors(dataFormat = InputDataFormatType.ampToolsRdReco)
    additionalColumnDefs = {
      "DistFdcPip": f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvs['lvPip']})",
      "DistFdcPim": f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvs['lvPim']})",
    }
    additionalFilterDefs = ["(DistFdcPip > 4) and (DistFdcPim > 4)"]  # require minimum distance of tracks at FDC position [cm]

  subsystems: tuple[SubSystemInfo, ...] = (
      SubSystemInfo(pairLabel = "PiPi", lvALabel = "lvPip", lvBLabel = "lvPim",      lvRecoilLabel = "lvRecoil"),
      # SubSystemInfo(pairLabel = "PipP", lvALabel = "lvPip", lvBLabel = "lvRecoil", lvRecoilLabel = "lvPim"   ),
      # SubSystemInfo(pairLabel = "PimP", lvALabel = "lvPim", lvBLabel = "lvRecoil", lvRecoilLabel = "lvPip"   ),
    )
  dataSetsPol: list[DataSetInfo] = []
  for dataPeriod in BEAM_POL_INFOS:
    print(f"Setting up data period '{dataPeriod}':")
    for tBinLabel in tBinLabels:
      print(f"Setting up t bin '{tBinLabel}':")
      for subsystem in subsystems:
        print(f"Setting up subsystem '{subsystem}':")
        inputDataDirName  = f"{inputDataBaseDirName}/{tBinLabel}"
        outputDataDirName = f"{outputDataBaseDirName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}"
        os.makedirs(outputDataDirName, exist_ok = True)
        for beamOrientation, beamPolInfo in BEAM_POL_INFOS[dataPeriod].items():
          print(f"Setting up beam orientation '{beamOrientation}': pol = {beamPolInfo.pol:.4f}, PhiLab = {beamPolInfo.PhiLab:.1f} deg")
          dataSetRd = DataSetInfo(  # real data (signal + background)
            subsystem            = subsystem,
            inputFormat          = InputDataFormatType.ampToolsRdReco,
            dataPeriod           = dataPeriod,
            tBinLabel            = tBinLabel,
            beamOrientation      = beamOrientation,
            inputFileNames       = ((f"{inputDataDirName}/amptools_tree_data_{beamOrientation}_30274_31057.root", ),
                                    (f"{inputDataDirName}/amptools_tree_bkgnd_{beamOrientation}_30274_31057.root",)),
            inputTreeName        = "kin",
            outputFileName       = f"{outputDataDirName}/data_flat_{beamOrientation}.root",
            outputTreeName       = subsystem.pairLabel,
            outputColumns        = outputColumns + ("eventWeight", ),
            additionalColumnDefs = additionalColumnDefs,
            additionalFilterDefs = additionalFilterDefs,
          )
          dataSetsPol.append(dataSetRd)
          dataSetPsAcc = deepcopy(dataSetRd)  # accepted phase-space MC
          dataSetPsAcc.inputFormat    = InputDataFormatType.ampToolsMcReco
          dataSetPsAcc.inputFileNames = (f"{inputDataDirName}/MC_100M/amptools_tree_accepted_30274_31057_noMcut.root",
                                         f"{inputDataDirName}/MC_10M_rho_t/amptools_tree_accepted_30274_31057_notcut.root")
          dataSetPsAcc.outputFileName = f"{outputDataDirName}/phaseSpace_acc_flat_{beamOrientation}.root"
          dataSetPsAcc.outputColumns  = outputColumns
          dataSetsPol.append(dataSetPsAcc)
          dataSetPsGen = deepcopy(dataSetPsAcc)  # generated phase-space MC
          dataSetPsGen.inputFormat          = InputDataFormatType.ampToolsMcTruth
          dataSetPsGen.inputFileNames       = (f"{inputDataDirName}/MC_100M/amptools_tree_thrown_30274_31057.root",
                                               f"{inputDataDirName}/MC_10M_rho_t/amptools_tree_thrown_30274_31057.root")
          dataSetPsGen.outputFileName       = f"{outputDataDirName}/phaseSpace_gen_flat_{beamOrientation}.root"
          dataSetPsGen.additionalColumnDefs = {}  # no selection cuts for generated MC
          dataSetPsGen.additionalFilterDefs = []
          dataSetsPol.append(dataSetPsGen)

  dataSets = dataSetsPol
  for dataSet in dataSets:
    df = None
    if dataSet.inputFormat == InputDataFormatType.ampToolsRdReco:
      # combine signal and background region data with correct event weights into one RDataFrame
      outputDataDirName = os.path.dirname(dataSet.outputFileName)
      df = getDataFrameWithCorrectEventWeights(
        dataSigRegionFileNames  = dataSet.inputFileNames[0],
        dataBkgRegionFileNames  = dataSet.inputFileNames[1],
        treeName                = dataSet.inputTreeName,
        friendSigRegionFileName = f"{outputDataDirName}/data_sig_{dataSet.beamOrientation}.root.weights",
        friendBkgRegionFileName = f"{outputDataDirName}/data_bkg_{dataSet.beamOrientation}.root.weights",
      )
    elif dataSet.inputFormat == InputDataFormatType.ampToolsMcReco or dataSet.inputFormat == InputDataFormatType.ampToolsMcTruth:
      # read all MC files into one RDataFrame
      df = ROOT.RDataFrame(dataSet.inputTreeName, dataSet.inputFileNames)
    else:
      raise RuntimeError(f"Unsupported input data format '{dataSet.inputFormat}'")
    print(f"Converting {dataSet.inputFormat} data for {dataSet.dataPeriod}, {dataSet.tBinLabel}, and {dataSet.beamOrientation} from file(s) {dataSet.inputFileNames} to file '{dataSet.outputFileName}'")
    lvs = lorentzVectors(dataFormat = dataSet.inputFormat)
    defineDataFrameColumns(
      df                   = df,
      lvTarget             = lvs["lvTarget"],
      lvBeam               = lvs["lvBeam"],
      lvRecoil             = lvs[dataSet.subsystem.lvRecoilLabel],
      lvA                  = lvs[dataSet.subsystem.lvALabel],
      lvB                  = lvs[dataSet.subsystem.lvBLabel],
      beamPolInfo          = BEAM_POL_INFOS[dataSet.dataPeriod][dataSet.beamOrientation],
      frame                = frame,
      additionalColumnDefs = dataSet.additionalColumnDefs,
      additionalFilterDefs = dataSet.additionalFilterDefs
    ).Snapshot(dataSet.outputTreeName, dataSet.outputFileName, dataSet.outputColumns)
