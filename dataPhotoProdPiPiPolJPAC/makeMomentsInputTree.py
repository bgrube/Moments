#!/usr/bin/env python3


from __future__ import annotations

import functools
import numpy as np
import pandas as pd
import os

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


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


def readData(inputFileName: str) -> ROOT.RDataFrame:
  """Reads data from an ASCII file into a ROOT RDataFrame"""
  print(f"Reading file '{inputFileName}'")
  pandasDf = pd.read_csv(inputFileName, sep=r"\s+")
  pandasDf.rename(columns = {"phi" : "phiOrig"}, inplace = True)
  # print(f"DataFrame shape: {pandasDf.shape}")
  # print(f"Columns: {list(pandasDf.columns)}")
  # convert Pandas DataFrame into ROOT RDataFrame
  print("Converting data to ROOT.RDataFrame")
  arrayDict = {column : np.array(pandasDf[column]) for column in pandasDf}
  rootDf: ROOT.RDataFrame = ROOT.RDF.MakeNumpyDataFrame(arrayDict)
  return rootDf


def lorentzVectors() -> dict[str, str]:
  """Returns Lorentz-vectors for beam photon ("lvBeam"), target proton ("lvTarget"), recoil proton ("lvRecoil"), pi+ ("lvPip"), and pi- ("lvPim")"""
  lvs = {}
  # kinematic variables according to Eq. (1) in BIBRZYCKI et al., PD 111, 014002 (2025)
  # gamma (q) + p (p1) -> pi+ (k1) + pi- (k2) + p (p2)
  # four-momenta are defined as (p_x, p_y, p_z, E)
  lvs["lvBeam"  ] = "q1,  q2,  q3,  q0"   # beam photon
  lvs["lvTarget"] = "p11, p12, p13, p10"  # target proton
  lvs["lvRecoil"] = "p21, p22, p23, p20"  # recoil proton
  lvs["lvPip"   ] = "k11, k12, k13, k10"  # pi+
  lvs["lvPim"   ] = "k21, k22, k23, k20"  # pi-
  return lvs


def defineDataFrameColumns(
  df:         ROOT.RDataFrame,
  beamPol:    float,  # photon beam polarization
  beamPolPhi: float,  # azimuthal angle of photon beam polarization in lab [deg]
  lvTarget:   str,    # function-argument list with Lorentz-vector components of target proton
  lvBeam:     str,    # function-argument list with Lorentz-vector components of beam photon
  lvRecoil:   str,    # function-argument list with Lorentz-vector components of recoil proton
  lvPip:      str,    # function-argument list with Lorentz-vector components of pi^+
  lvPim:      str,    # function-argument list with Lorentz-vector components of pi^-
  frame:      str  = "Hf",  # can be either "Hf" for helicity or "Gj" for Gottfried-Jackson frame
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns for moments analysis"""
  assert frame == "Hf" or frame == "Gj", f"Unknown frame '{frame}'"
  print(f"Defining angles in '{frame}' frame and using pi^+ as analyzer")
  df = (
    df.Define("beamPol",    f"(Double32_t){beamPol}")
      .Define("beamPolPhi", f"(Double32_t){beamPolPhi}")
      .Define("cosTheta",    "(Double32_t)" + (f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})" if frame == "Hf" else
                                               f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})"))  #!NOTE! frames have different signatures (see FSBasic/FSMath.h)
      .Define("theta",       "(Double32_t)std::acos(cosTheta)")
      # switching between pi+ and pi- analyzer flips sign of moments with odd M
      # # use pi+ as analyzer and y_HF/GJ = p_beam x p_recoil
      # .Define("phi",         "(Double32_t)" + (f"FSMath::helphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})" if frame == "Hf" else
      #                                          f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})"))
      # use pi- as analyzer and y_HF/GJ = p_beam x p_recoil
      #TODO there seems to be a bug in the calculation of phi (at least for the HF frame)
      #     when using the same analyzer as for cosTheta, i.e. pi+, phi is flipped by 180 deg
      #     this difference is seen when comparing to Alex' function and also when comparing to the PWA result
      #     switching the analyzer to pi- cures this problem
      .Define("phi",         "(Double32_t)" + (f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam})" if frame == "Hf" else
                                               f"FSMath::gjphi({lvPim}, {lvPip}, {lvBeam})"))
      .Define("phiDeg",      "(Double32_t)phi * TMath::RadToDeg()")
      .Define("Phi",        f"(Double32_t)bigPhi({lvRecoil}, {lvBeam}, beamPolPhi)")
      .Define("PhiDeg",      "(Double32_t)Phi * TMath::RadToDeg()")
      .Define("mass",       f"(Double32_t)massPair({lvPip}, {lvPim})")
      .Define("minusT",     f"(Double32_t)-mandelstamT({lvTarget}, {lvRecoil})")
  )
  return df


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_BIGPHI)

  inputData: dict[str, str] = {  # mapping of t-bin labels to input file names
    "tbin_0.4_0.5" : "./mc/mc_full_model/mc0.4-0.5_ful.dat",
    "tbin_0.5_0.6" : "./mc/mc_full_model/mc0.5-0.6_ful.dat",
    "tbin_0.6_0.7" : "./mc/mc_full_model/mc0.6-0.7_ful.dat",
    "tbin_0.7_0.8" : "./mc/mc_full_model/mc0.7-0.8_ful.dat",
    "tbin_0.8_0.9" : "./mc/mc_full_model/mc0.8-0.9_ful.dat",
    "tbin_0.9_1.0" : "./mc/mc_full_model/mc0.9-1.0_ful.dat",
  }
  dataLabel      = "PARA_0"
  beamPol        = 1.0
  beamPolPhi     = 0.0
  outputTreeName = "data"
  outputColumns  = ("beamPol", "beamPolPhi", "cosTheta", "theta", "phi", "phiDeg", "Phi", "PhiDeg", "mass", "minusT")
  frame          = "Hf"

  for tBinLabel, inputFileName in inputData.items():
    outputDirName  = tBinLabel
    os.makedirs(outputDirName, exist_ok = True)
    outputFileName = f"{outputDirName}/data_flat_{dataLabel}.root"

    df = defineDataFrameColumns(
      df         = readData(inputFileName),
      beamPol    = beamPol,
      beamPolPhi = beamPolPhi,
      frame      = frame,
      **lorentzVectors(),
    # ).Snapshot(outputTreeName, outputFileName, outputColumns)
    ).Snapshot(outputTreeName, outputFileName)  # write all columns
    print(f"ROOT DataFrame columns: {list(df.GetColumnNames())}")
    print(f"ROOT DataFrame entries: {df.Count().GetValue()}")
