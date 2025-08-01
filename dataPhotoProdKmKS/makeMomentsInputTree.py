#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
import functools
from scipy.special import cosdg, sindg
import os
from typing import Dict

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class BeamPolInfo:
  """Stores information about beam polarization for a specific orientation"""
  beamPol:       float  # photon-beam polarization magnitude
  beamPolPhiLab: float  # [deg] azimuthal angle of photon beam polarization in lab frame

# polarization values from Version 9 of `makePolVals` tool from https://halldweb.jlab.org/wiki-private/index.php/TPOL_Polarization
# beam polarization angles in lab frame taken from `Lab Phi` column of tables 2 to 5 in GlueX-doc-3977
BEAM_POL_INFOS: dict[str, dict[str, BeamPolInfo]] = {  # year_month : {orientation : BeamPolInfo(...)}
  "2017_01" : {  # polarization magnitudes obtained by running `.x makePolVals.C(17, 1, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      beamPol       = 0.3537,
      beamPolPhiLab = 1.8,
    ),
    "PERP_45" : BeamPolInfo(
      beamPol       = 0.3484,
      beamPolPhiLab = 47.9,
    ),
    "PERP_90" : BeamPolInfo(
      beamPol       = 0.3472,
      beamPolPhiLab = 94.5,
    ),
    "PARA_135" : BeamPolInfo(
      beamPol       = 0.3512,
      beamPolPhiLab = -41.6,
    ),
  },
  "2018_01" : {  # polarization magnitudes obtained by running `.x makePolVals.C(18, 1, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      beamPol       = 0.3420,
      beamPolPhiLab = 4.1,
    ),
    "PERP_45" : BeamPolInfo(
      beamPol       = 0.3474,
      beamPolPhiLab = 48.5,
    ),
    "PERP_90" : BeamPolInfo(
      beamPol       = 0.3478,
      beamPolPhiLab = 94.2,
    ),
    "PARA_135" : BeamPolInfo(
      beamPol       = 0.3517,
      beamPolPhiLab = -42.4,
    ),
  },
  "2018_08" : {  # polarization magnitudes obtained by running `.x makePolVals.C(18, 2, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      beamPol       = 0.3563,
      beamPolPhiLab = 3.3,
    ),
    "PERP_45" : BeamPolInfo(
      beamPol       = 0.3403,
      beamPolPhiLab = 48.3,
    ),
    "PERP_90" : BeamPolInfo(
      beamPol       = 0.3430,
      beamPolPhiLab = 92.9,
    ),
    "PARA_135" : BeamPolInfo(
      beamPol       = 0.3523,
      beamPolPhiLab = -42.1,
    ),
  },
}


# C++ function to calculate azimuthal angle of photon polarization vector
CPP_CODE_POLPHI = """
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


def setup(useRDataFrame: bool) -> None:
  """Setup the FSRoot environment and define cuts"""
  if ROOT.FSModeCollection.modeVector().size() != 0:
    return
  modeInfo = ROOT.FSModeInfo(fsModeString)
  modeInfo.display()
  ROOT.FSModeCollection.addModeInfo(fsModeString).addCategory(fsCategory)
  # ROOT.FSTree.addFriendTree("Chi2Rank")  # ranking trees have already been applied
  # ROOT.FSTree.showFriendTrees()
  if useRDataFrame:
    ROOT.FSHistogram.enableRDataFrame(False)  # false = delay filling of histograms until FSHistogram::executeRDataFrame() is called

  # define macros for azimuthal angle of photon polarization vector
  ROOT.gInterpreter.Declare(CPP_CODE_POLPHI)
  ROOT.FSTree.defineMacro("MYPOLPHI", 3,
    "bigPhi("
      "PxP[I],PyP[I],PzP[I],EnP[I],"
      "PxP[J],PyP[J],PzP[J],EnP[J],"
      "PxP[M]"
    ")"
  )
  ROOT.FSTree.defineMacro("POLPHI", 4,
    "FSMath::polphi("
      "PxP[I],PyP[I],PzP[I],EnP[I],"
      "PxP[J],PyP[J],PzP[J],"
      "PxP[M],PyP[M],PzP[M],EnP[M],"
      "PxP[N],PyP[N],PzP[N],EnP[N]"
    ")"
  )


def plotHistsAndWriteTree(
  inputFileName:   str,
  fsTreeName:      str,
  fsCategory:      str,
  cutString:       str,
  beamPol:         float,
  beamPolAngleLab: float,
  eventWeight:     float,
  useRDataFrame:   bool,
  useMCTruth:      bool,
  plotLabel:       str,
) -> None:
  """Plots histograms and writes friend tree with variables for moment analysis"""
  hists: Dict[str, ROOT.TH1F] = {}
  if useMCTruth:
    ROOT.FSTree.defineFourVector("MCGLUEXBEAM", "MCEnPB", "MCPxPB", "MCPyPB", "MCPzPB")
  MCLabel = "MC" if useMCTruth else ""

  # kinematic distributions
  massDef = f"{MCLabel}MASS"
  massKKDef = f"{massDef}([K-], [Ks])"
  massPiPiDef = f"{massDef}(3a, 3b)"  # 3a = pi+ from Ks, 3b = pi- from Ks
  momTransferDef = f"-{massDef}2(GLUEXTARGET, -[p+], -[pi+])"
  if not useMCTruth:
    hists[f"h{plotLabel}Chi2Ndf"           ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, "Chi2DOF",        "(100, 0, 25)",    cutString))
    hists[f"h{plotLabel}RfDeltaT"          ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, "RFDeltaT",       "(400, -20, 20)",  cutString))
  hists[f"h{MCLabel}{plotLabel}BeamEnergy" ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, f"{MCLabel}EnPB", "(400, 2, 12)",    cutString))
  hists[f"h{MCLabel}{plotLabel}MassKK"     ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, massKKDef,        "(100, 0.8, 2.5)", cutString))
  hists[f"h{MCLabel}{plotLabel}MassPiPi"   ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, massPiPiDef,      "(100, 0.4, 0.6)", cutString))
  hists[f"h{MCLabel}{plotLabel}MomTransfer"] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, momTransferDef,   "(100, 0, 1)",     cutString))

  # angular variables
  # for beam + target -> X + recoil and X -> a + b (see FSBasic/FSMath.h)
  # angles of particle a in X Gottfried-Jackson RF are calculated by
  #   GJCOSTHETA(a; b; beam)
  #   GJPHI(a; b; recoil; beam) [rad]
  GjCosThetaDef = f"{MCLabel}GJCOSTHETA([Ks]; [K-]; {MCLabel}GLUEXBEAM)"
  GjThetaDef    = f"acos({GjCosThetaDef})"
  GjPhiDef      = f"{MCLabel}GJPHI([Ks]; [K-]; [p+], [pi+]; {MCLabel}GLUEXBEAM)"
  GjPhiDegDef   = f"{GjPhiDef} * TMath::RadToDeg()"
  # angles of particle a in X helicity RF are calculated by
  #   HELCOSTHETA(a; b; recoil)
  #   HELPHI(a; b; recoil; beam) [rad]
  HfCosThetaDef = f"{MCLabel}HELCOSTHETA([Ks]; [K-]; [p+], [pi+])"
  HfThetaDef    = f"acos({HfCosThetaDef})"
  HfPhiDef      = f"{MCLabel}HELPHI([Ks]; [K-]; [p+], [pi+]; {MCLabel}GLUEXBEAM)"
  HfPhiDegDef   = f"{HfPhiDef} * TMath::RadToDeg()"
  ROOT.FSTree.defineFourVector("P0", "1000", f"{cosdg(beamPolAngleLab)}", f"{sindg(beamPolAngleLab)}", "0.0")  # vector representing beam polarization orientation in lab frame
  bigPhiDef     = f"{MCLabel}POLPHI([Ks], [K-]; P0; [p+], [pi+]; {MCLabel}GLUEXBEAM)"
  bigPhiDegDef  = f"{bigPhiDef} * TMath::RadToDeg()"
  print(f"Defined macro: {ROOT.FSTree.expandVariable(bigPhiDef)}")
  hists[f"h{MCLabel}{plotLabel}GjCosTheta"] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, GjCosThetaDef,  "(100, -1, +1)",     cutString))
  hists[f"h{MCLabel}{plotLabel}GjPhi"     ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, GjPhiDegDef,    "(100, -180, +180)", cutString))
  hists[f"h{MCLabel}{plotLabel}HfCosTheta"] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, HfCosThetaDef,  "(100, -1, +1)",     cutString))
  hists[f"h{MCLabel}{plotLabel}HfPhi"     ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, HfPhiDef,       "(100, -180, +180)", cutString))
  hists[f"h{MCLabel}{plotLabel}Phi"       ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, bigPhiDegDef,   "(100, -180, +180)", cutString))
  ROOT.FSTree.defineFourVector("BEAMPOLANGLELAB", "0.0", f"{beamPolAngleLab}", "0.0", "0.0")  # dummy vector representing beam polarization orientation in lab frame
  myBigPhiDef    = f"{MCLabel}MYPOLPHI([p+], [pi+]; {MCLabel}GLUEXBEAM; BEAMPOLANGLELAB)"
  myBigPhiDegDef = f"{myBigPhiDef} * TMath::RadToDeg()"
  print(f"Defined macro: {ROOT.FSTree.expandVariable(myBigPhiDef)}")
  hists[f"h{MCLabel}{plotLabel}MyPhi"     ] = (ROOT.FSModeHistogram.getTH1F(inputFileName, fsTreeName, fsCategory, myBigPhiDegDef, "(100, -180, +180)", cutString))

  # draw histograms
  if useRDataFrame:
    ROOT.FSHistogram.executeRDataFrame()
  for histName, hist in hists.items():
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.DrawCopy()
    canv.SaveAs(f"{histName}.pdf")

  # write root tree for moment analysis
  varDefs = ROOT.std.vector[ROOT.std.pair[str, str]]()
  varDefs.push_back(ROOT.std.pair[str, str]("beamPol",     f"{beamPol}"))
  varDefs.push_back(ROOT.std.pair[str, str]("beamPolPhi",  f"{beamPolAngleLab}"))
  varDefs.push_back(ROOT.std.pair[str, str]("cosTheta",    GjCosThetaDef))
  varDefs.push_back(ROOT.std.pair[str, str]("theta",       GjThetaDef))
  varDefs.push_back(ROOT.std.pair[str, str]("phiDeg",      GjPhiDegDef))
  varDefs.push_back(ROOT.std.pair[str, str]("phi",         GjPhiDef))
  # varDefs.push_back(ROOT.std.pair[str, str]("cosTheta",    HfCosThetaDef))
  # varDefs.push_back(ROOT.std.pair[str, str]("theta",       HfThetaDef))
  # varDefs.push_back(ROOT.std.pair[str, str]("phiDeg",      HfPhiDegDef))
  # varDefs.push_back(ROOT.std.pair[str, str]("phi",         HfPhiDef))
  varDefs.push_back(ROOT.std.pair[str, str]("Phi",         bigPhiDef))
  varDefs.push_back(ROOT.std.pair[str, str]("PhiDeg",      bigPhiDegDef))
  varDefs.push_back(ROOT.std.pair[str, str]("mass",        massKKDef))
  varDefs.push_back(ROOT.std.pair[str, str]("minusT",      momTransferDef))
  varDefs.push_back(ROOT.std.pair[str, str]("eventWeight", f"{eventWeight}"))
  print(f"Writing friend tree to file '{inputFileName}.angles'")
  ROOT.FSModeTree.createFriendTree(inputFileName, fsTreeName, fsCategory, "angles", varDefs)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")

  fsTreeName      = "ntFSGlueX_MODECODE"
  fsModeString    = "100_11100"
  # decoding of mode string:
  #  1  0  0   _ 1   1  1   0   0
  #  |  |  |     |   |  |   |   |
  #  p+ p- eta   K-  KS pi+ pi- pi0
  # (1)         (2) (3) (4)         <- indices in FSTree
  # (3a) = pi+ from KS, (3b) = pi- from KS
  fsCategory      = "m" + fsModeString
  useRDataFrame   = True

  # Fall 2018 data
  dataToProcess: tuple[tuple[str, str, float, bool, str], ...] = (  # (data directory, file name, event weight, use MC truth, plot label)
    ("data",       "pipkmks_100_11100_B4_M16_SIGNAL_SKIM_A2.root",        1.0,       False, "Sig"),
    ("data",       "pipkmks_100_11100_B4_M16_SIDEDBANDS_SKIM_A2.root",   -1.0 / 6.0, False, "Bkg"),
    ("phaseSpace", "pipkmks_100_11100_B4_M16_SIGNAL_SKIM_A2.root",        1.0,       False, "MCSig"),  #TODO is this correct? shouldn't the MC truth info be used for binning?
    ("phaseSpace", "pipkmks_100_11100_B4_M16_MCGEN_GENERAL_SKIM_A2.root", 1.0,       True,  "Gen"),
  )
  beamPol         = BEAM_POL_INFOS["2018_08"]["PARA_0"].beamPol
  beamPolAngleLab = BEAM_POL_INFOS["2018_08"]["PARA_0"].beamPolPhiLab
  #TODO Kevin: accidentals in MC data

  setup(useRDataFrame)
  ROOT.FSCut.defineCut("cutSet", "")  # all cuts are already applied
  cutString = "CUT(cutSet)"

  for dataDir, fileName, eventWeight, useMCTruth, plotLabel in dataToProcess:
    print(f"Processing data from file '{dataDir}/{fileName}' with event weight {eventWeight}")
    plotHistsAndWriteTree(
      inputFileName   = f"{dataDir}/{fileName}",
      fsTreeName      = fsTreeName,
      fsCategory      = fsCategory,
      cutString       = cutString,
      beamPol         = beamPol,
      beamPolAngleLab = beamPolAngleLab,
      eventWeight     = eventWeight,
      useRDataFrame   = useRDataFrame,
      useMCTruth      = useMCTruth,
      plotLabel       = plotLabel,
    )

  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogoff.FSROOT.C")
