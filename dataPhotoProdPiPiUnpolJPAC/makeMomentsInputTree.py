#!/usr/bin/env python3


from __future__ import annotations

import numpy as np
import pandas as pd
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


def readDataJpac(inputFileName: str) -> ROOT.RDataFrame:
  """Reads JPAC data from an ASCII file into a ROOT RDataFrame"""
  print(f"Reading file '{inputFileName}'")
  pandasDf = pd.read_csv(inputFileName, sep = r"\s+")
  pandasDf["t"]  *= -1.0  # flip sign
  pandasDf["phi"] = np.degrees(pandasDf["phi"])  # convert to degrees
  pandasDf.loc[pandasDf["phi"] > 180, "phi"] -= 360  # apply: if phi > 180 then phi -= 360
  # rename columns to match naming convention
  pandasDf.rename(columns = {"t"     : "minusTJpac"},   inplace = True)
  pandasDf.rename(columns = {"mpipi" : "massJpac"},     inplace = True)
  pandasDf.rename(columns = {"phi"   : "phiDegJpac"},   inplace = True)
  pandasDf.rename(columns = {"costh" : "cosThetaJpac"}, inplace = True)
  # print(f"DataFrame shape: {pandasDf.shape}")
  # print(f"Columns: {list(pandasDf.columns)}")
  # convert Pandas DataFrame into ROOT RDataFrame
  print("Converting data to ROOT.RDataFrame")
  arrayDict = {column : np.array(pandasDf[column]) for column in pandasDf}
  rootDf: ROOT.RDataFrame = ROOT.RDF.MakeNumpyDataFrame(arrayDict)
  return rootDf


def lorentzVectorsJpac() -> dict[str, str]:
  """Returns Lorentz-vectors for beam photon ("lvBeam"), target proton ("lvTarget"), recoil proton ("lvRecoil"), pi+ ("lvPip"), and pi- ("lvPim") for JPAC data"""
  lvs = {}
  # kinematic variables according to Eq. (1) in BIBRZYCKI et al., PD 111, 014002 (2025)
  # gamma (q) + p (p1) -> pi+ (k1) + pi- (k2) + p (p2)
  # four-momenta are defined as
  #                 (p_x, p_y, p_z, E)
  lvs["lvBeam"  ] = "q1,  q2,  q3,  q0"   # beam photon
  lvs["lvTarget"] = "p11, p12, p13, p10"  # target proton
  lvs["lvRecoil"] = "p21, p22, p23, p20"  # recoil proton
  lvs["lvPip"   ] = "k11, k12, k13, k10"  # pi+
  lvs["lvPim"   ] = "k21, k22, k23, k20"  # pi-
  return lvs


def lorentzVectorsTlv() -> dict[str, str]:
  """Returns Lorentz-vectors for beam photon ("lvBeam"), target proton ("lvTarget"), recoil proton ("lvRecoil"), pi+ ("lvPip"), and pi- ("lvPim") for data with TLorentzVectors"""
  lvs = {}
  lvs["lvBeam"  ] = "lvBeamLab.X(),   lvBeamLab.Y(),   lvBeamLab.Z(),   lvBeamLab.E()"    # beam photon
  lvs["lvTarget"] = "lvTargetLab.X(), lvTargetLab.Y(), lvTargetLab.Z(), lvTargetLab.E()"  # target proton
  lvs["lvRecoil"] = "lvRecoilLab.X(), lvRecoilLab.Y(), lvRecoilLab.Z(), lvRecoilLab.E()"  # recoil proton
  lvs["lvPip"   ] = "lvPipLab.X(),    lvPipLab.Y(),    lvPipLab.Z(),    lvPipLab.E()"     # pi+
  lvs["lvPim"   ] = "lvPimLab.X(),    lvPimLab.Y(),    lvPimLab.Z(),    lvPimLab.E()"     # pi-
  return lvs


def defineDataFrameColumns(
  df:       ROOT.RDataFrame,
  lvTarget: str,  # function-argument list with Lorentz-vector components of target proton
  lvBeam:   str,  # function-argument list with Lorentz-vector components of beam photon
  lvRecoil: str,  # function-argument list with Lorentz-vector components of recoil proton
  lvPip:    str,  # function-argument list with Lorentz-vector components of pi^+
  lvPim:    str,  # function-argument list with Lorentz-vector components of pi^-
  frame:    str  = "Hf",  # can be either "Hf" for helicity or "Gj" for Gottfried-Jackson frame
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns for moment analysis"""
  assert frame == "Hf" or frame == "Gj", f"Unknown frame '{frame}'"
  print(f"Defining angles in '{frame}' frame and using pi^+ as analyzer")
  df = (
    df.Define("cosTheta",     "(Double32_t)" + (f"FSMath::helcostheta({lvPip}, {lvPim}, {lvRecoil})" if frame == "Hf" else
                                                f"FSMath::gjcostheta({lvPip}, {lvPim}, {lvBeam})"))  #!NOTE! frames have different signatures (see FSBasic/FSMath.h)
      .Define("theta",        "(Double32_t)std::acos(cosTheta)")
      # switching between pi+ and pi- analyzer flips sign of moments with odd M
      # # use pi+ as analyzer and y_HF/GJ = p_beam x p_recoil
      # .Define("phi",         "(Double32_t)" + (f"FSMath::helphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})" if frame == "Hf" else
      #                                          f"FSMath::gjphi({lvPip}, {lvPim}, {lvRecoil}, {lvBeam})"))
      # use pi- as analyzer and y_HF/GJ = p_beam x p_recoil
      #TODO there seems to be a bug in the calculation of phi (at least for the HF frame)
      #     when using the same analyzer as for cosTheta, i.e. pi+, phi is flipped by 180 deg
      #     this difference is seen when comparing to Alex' function and also when comparing to the PWA result
      #     switching the analyzer to pi- cures this problem
      .Define("phi",          "(Double32_t)" + (f"FSMath::helphi({lvPim}, {lvPip}, {lvRecoil}, {lvBeam})" if frame == "Hf" else
                                                f"FSMath::gjphi({lvPim}, {lvPip}, {lvBeam})"))
      .Define("phiDeg",       "(Double32_t)phi * TMath::RadToDeg()")
      .Define("mass",        f"(Double32_t)massPair({lvPip}, {lvPim})")
      .Define("minusT",      f"(Double32_t)-mandelstamT({lvTarget}, {lvRecoil})")
      .Define("eventWeight", f"(Double32_t)1.0")
  )
  return df


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.C"
  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)

  inputData: dict[str, str] = {  # mapping of t-bin labels to input file names
    "tbin_0.4_0.5" : "./mc/mc_full_model/mc0.4-0.5_ful.dat",
    "tbin_0.5_0.6" : "./mc/mc_full_model/mc0.5-0.6_ful.dat",
    "tbin_0.6_0.7" : "./mc/mc_full_model/mc0.6-0.7_ful.dat",
    "tbin_0.7_0.8" : "./mc/mc_full_model/mc0.7-0.8_ful.dat",
    "tbin_0.8_0.9" : "./mc/mc_full_model/mc0.8-0.9_ful.dat",
    "tbin_0.9_1.0" : "./mc/mc_full_model/mc0.9-1.0_ful.dat",
  }
  outputDirName  = "mc_full"
  outputTreeName = "PiPi"
  outputColumns  = ("cosTheta", "theta", "phi", "phiDeg", "mass", "minusT")
  frame          = "Hf"
  useJpacData    = True

  for tBinLabel, inputFileName in inputData.items():
    os.makedirs(f"{outputDirName}/{tBinLabel}", exist_ok = True)
    outputFileName = f"{outputDirName}/{tBinLabel}/data_flat.root"
    print(f"Writing data to tree '{outputTreeName}' in '{outputFileName}'")
    df = defineDataFrameColumns(
      df    = readDataJpac(inputFileName) if useJpacData else ROOT.RDataFrame("PiPi", "./data_labFrame_flat.root"),
      frame = frame,
      **(lorentzVectorsJpac() if useJpacData else lorentzVectorsTlv()),
    ).Snapshot(outputTreeName, outputFileName, outputColumns)
    # ).Snapshot(outputTreeName, outputFileName)  # write all columns
    print(f"ROOT DataFrame columns: {list(df.GetColumnNames())}")
    print(f"ROOT DataFrame entries: {df.Count().GetValue()}")
