#!/usr/bin/env python3

from dataclasses import dataclass
import functools
import os
from typing import (
  Dict,
  List,
  Tuple,
)

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class BeamPolInfo:
  """Stores info about beam polarization datasets"""
  datasetLabel:   str    # label used for dataset
  angle:          int    # beam polarization angle in lab frame [deg]
  polarization:   float  # average beam polarization
  ampScaleFactor: float  # fitted amplitude scaling factor for dataset

# see Eqs. (4.22)ff in Lawrence's thesis for polarization values and
# /w/halld-scshelf2101/malte/final_fullWaveset/nominal_fullWaveset_ReflIndiv_150rnd/010020/etapi_result_samePhaseD.fit
# for amplitude scaling factors
BEAM_POL_INFOS: Tuple[BeamPolInfo, ...] = (
  BeamPolInfo(datasetLabel = "000", angle =   0, polarization = 0.35062, ampScaleFactor = 1.0),
  BeamPolInfo(datasetLabel = "045", angle =  45, polarization = 0.34230, ampScaleFactor = 0.982204395837131),
  BeamPolInfo(datasetLabel = "090", angle =  90, polarization = 0.34460, ampScaleFactor = 0.968615883555624),
  BeamPolInfo(datasetLabel = "135", angle = 135, polarization = 0.35582, ampScaleFactor = 0.98383623655323),
)


# delcare C++ function to calculate azimuthal angle of photon polarization vector
CPP_CODE = """
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
	if (true) {
		// test code in https://github.com/lan13005/EtaPi-Analysis/blob/99c4c8045d75619bb2bfde6e800da72078723490/DSelector_etapi.C#L740
		const TLorentzVector target(0, 0, 0, 0.938272);
		const TLorentzRotation cmRestBoost(-(beam + target).BoostVector());
		const TLorentzVector beam_cm = cmRestBoost * beam;
		const TLorentzVector recoil_cm = cmRestBoost * recoil;
		const TVector3 y = (beam_cm.Vect().Unit().Cross(-recoil_cm.Vect().Unit())).Unit();
		const TVector3 eps2(TMath::Cos(polAngle * TMath::DegToRad()), TMath::Sin(polAngle * TMath::DegToRad()), 0);  // beam polarization vector
		double Phi2 = TMath::ATan2(y.Dot(eps2), beam_cm.Vect().Unit().Dot(eps2.Cross(y)));
		// ensure [-pi, +pi] range
		while (Phi2 > TMath::Pi()) {
			Phi2 -= TMath::TwoPi();
		}
		while (Phi2 < -TMath::Pi()) {
			Phi2 += TMath::TwoPi();
		}
		const double deltaPhi = Phi2 - Phi;
		if (std::abs(deltaPhi) > 1e-14) {
			std::cout << "Mismatch of Phi values: Phi2 = " << Phi2 << " - Phi = " << Phi << ": " << deltaPhi << std::endl;
		}
	}
	return Phi;
}
"""
ROOT.gInterpreter.Declare(CPP_CODE)


# declare C++ function to check equality of column values
CPP_CODE = """
struct checkValEqual {

  Double_t    _ref;
  std::string _varName;

  checkValEqual(
    const Double_t     ref     = 0.0,
    const std::string& varName = "values"
  ) : _ref    (ref),
      _varName(varName)
  { }

  void
  operator () (const Double_t v)
  {
    if (v != _ref) {
      std::cout << "Mismatch of " << _varName << ": " << v << " vs. " << _ref << std::endl;
    }
    return;
  }

};
"""
ROOT.gInterpreter.Declare(CPP_CODE)


def defineDataFrameColumns(
  data:                   ROOT.RDataFrame,
  coordSys:               str   = "hel",  # either "hel" for helicity frame or "gj" for Gottfried-Jackson frame
  beamPol:                float = 1.0,    # photon beam polarization
  beamPolAngle:           float = 0.0,    # photon beam polarization angle in lab [deg]
  beamPolAngleColumnName: str   = "BeamAngle",
  weightColumnName:       str   = "Weight",
  thrownData:             bool  = False,
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns for moments analysis"""
  columnSuffix = "_thrown" if thrownData else ""
  df = (
    # ensure that quantities used to calculate moments are doubles
    data.Define("beamPol",     f"(double){beamPol}")
        .Define("beamPolPhi",  f"(double){beamPolAngleColumnName}")
        .Define("cosTheta",    f"(double)cosTheta_eta_{coordSys}{columnSuffix}")
        .Define("theta",       f"(double)acos(cosTheta_eta_{coordSys}{columnSuffix})")
        .Define("t",           f"(double)mandelstam_t{columnSuffix}")
        .Define("mass",        f"(double)Mpi0eta{columnSuffix}")
        .Define("eventWeight", f"(double){weightColumnName}")
  )
  # check that beam polarization angle has correct value in tree
  # checkValsEqual = functools.partial(ROOT.checkValsEqual, flush = True)
  # df.Foreach(lambda beamPolPhi: None if beamPolPhi == beamPolAngle \
  #   else print(f"Mismatch of beam polarization angles; tree: {beamPolPhi=} vs. argument: {beamPolAngle=}"),
  #   ["beamPolPhi"])
  checkValEqual = ROOT.checkValEqual(beamPolAngle, "beam polarization angles")
  df.Foreach(checkValEqual, ["beamPolPhi"])
  bigPhiFunc = "bigPhi(Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0], Px_Beam, Py_Beam, Pz_Beam, E_Beam, beamPolPhi)"
  phiVarDef = f"(double)phi_eta_{coordSys}{columnSuffix}"
  if thrownData:
    df = (
      df.Define("phiDeg", f"{phiVarDef} * TMath::RadToDeg()")
        .Define("phi",    phiVarDef)
        .Define("Phi",    bigPhiFunc)
    )
  else:
    df = (
      df.Define  ("phiDeg", phiVarDef)
        .Define  ("phi",    f"{phiVarDef} * TMath::DegToRad()")
        .Redefine("Phi",    bigPhiFunc)
    )
  return df.Define("PhiDeg", "Phi * TMath::RadToDeg()")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)

  dataSets: Dict[str, List[Dict[str, str]]] = {
    beamPolInfo.datasetLabel : [
      {"dataSetLabel"         : "data",
       "inputFileNamePattern" : f"t010020_m080250_selectGenTandM_nominal/pol{beamPolInfo.datasetLabel}_t010020_m080250_selectGenTandM_DTOT_selected_nominal_acc_flat.root"},
      {"dataSetLabel"         : "phaseSpace_gen",
       "inputFileNamePattern" : f"t010020_m080250_selectGenTandM_nominal/pol{beamPolInfo.datasetLabel}_t010020_m080250_selectGenTandM_FTOT_gen_data_flat.root"},
      {"dataSetLabel"         : "phaseSpace_acc",
       "inputFileNamePattern" : f"t010020_m080250_selectGenTandM_nominal/pol{beamPolInfo.datasetLabel}_t010020_m080250_selectGenTandM_FTOT_selected_nominal_acc_flat.root"},
    ] for beamPolInfo in BEAM_POL_INFOS}
  inputTreeName    = "kin"
  outputTreeName   = "etaPi0"
  weightColumnName = "Weight"
  plotDirBaseName  = "./plots"

  for beamPolLabel, beamPolDataSets in dataSets.items():
    for dataSet in beamPolDataSets:
      dataSetLabel         = dataSet["dataSetLabel"]
      inputFileNamePattern = dataSet["inputFileNamePattern"]
      outputFileName       = f"./{dataSetLabel}_{beamPolLabel}_flat.root"
      recoData             = ((dataSetLabel == "data") or (dataSetLabel == "phaseSpace_acc"))

      # apply fiducial cuts
      data = ROOT.RDataFrame(inputTreeName, inputFileNamePattern)
      if recoData:
        data = data.Filter(  # see Tab. 3.5 in Lawrences's thesis
                "("
                  "((8.2 < E_Beam) && (E_Beam < 8.8))"  # [GeV]
                  "&& (proton_momentum > 0.3)"  # [GeV]
                  "&& ((52 < proton_z) && (proton_z < 78))"  # [cm]
                  #??? dE/dx CDC cut; related to pdEdxCDCProton in tree (always 1)?
                  "&& (photonE1 > 0.1)"  # [GeV]
                  "&& (photonE2 > 0.1)"  # [GeV]
                  "&& (photonE3 > 0.1)"  # [GeV]
                  "&& (photonE4 > 0.1)"  # [GeV]
                  "&& (((2.5 < photonTheta1) && (photonTheta1 < 10.3)) || (photonTheta1 > 11.9))"  # [deg]
                  "&& (((2.5 < photonTheta2) && (photonTheta2 < 10.3)) || (photonTheta2 > 11.9))"  # [deg]
                  "&& (((2.5 < photonTheta3) && (photonTheta3 < 10.3)) || (photonTheta3 > 11.9))"  # [deg]
                  "&& (((2.5 < photonTheta4) && (photonTheta4 < 10.3)) || (photonTheta4 > 11.9))"  # [deg]
                  "&& (unusedEnergy < 0.01)"  # [GeV]
                  "&& (abs(mmsq) < 0.05)"  # [GeV^2]
                  "&& (chiSq < 13.277)"
                  "&& (pVH > 0.5)"  #??? from where? relation to Eq. (4.3) in thesis?
                  # "&& ((0.8 < Mpi0eta_thrown) && (Mpi0eta_thrown < 2.0))"  # [GeV]
                ")"
              )

      # get BeamPolInfo for beamPolLabel
      beamPolInfo = next(info for info in BEAM_POL_INFOS if info.datasetLabel == beamPolLabel)
      data = defineDataFrameColumns(
        data,
        coordSys         = "hel",
        beamPol          = beamPolInfo.polarization,
        beamPolAngle     = beamPolInfo.angle,
        weightColumnName = weightColumnName,
        thrownData       = not recoData
      )
      # data.Describe().Print()

      # define background-subtracted histograms
      histDefs = []
      if recoData:
        histDefs += [
          # cut variables
          {"columnName" : "pVH",             "xAxisUnit" : "",        "yAxisTitle" : "Combos",                 "binning" : (100, -0.5, 1.5)},
          {"columnName" : "unusedEnergy",    "xAxisUnit" : "GeV",     "yAxisTitle" : "Combos / 1 MeV",         "binning" : (100, 0, 0.1)},
          {"columnName" : "chiSq",           "xAxisUnit" : "",        "yAxisTitle" : "Combos",                 "binning" : (100, 0, 15)},
          {"columnName" : "photonTheta1",    "xAxisUnit" : "deg",     "yAxisTitle" : "Combos / 0.1 deg",       "binning" : (200, 0, 20)},
          {"columnName" : "photonTheta2",    "xAxisUnit" : "deg",     "yAxisTitle" : "Combos / 0.1 deg",       "binning" : (200, 0, 20)},
          {"columnName" : "photonTheta3",    "xAxisUnit" : "deg",     "yAxisTitle" : "Combos / 0.1 deg",       "binning" : (200, 0, 20)},
          {"columnName" : "photonTheta4",    "xAxisUnit" : "deg",     "yAxisTitle" : "Combos / 0.1 deg",       "binning" : (200, 0, 20)},
          {"columnName" : "photonE1",        "xAxisUnit" : "GeV",     "yAxisTitle" : "Combos / 0.1 GeV",       "binning" : (90, 0, 9)},
          {"columnName" : "photonE2",        "xAxisUnit" : "GeV",     "yAxisTitle" : "Combos / 0.1 GeV",       "binning" : (90, 0, 9)},
          {"columnName" : "photonE3",        "xAxisUnit" : "GeV",     "yAxisTitle" : "Combos / 0.1 GeV",       "binning" : (90, 0, 9)},
          {"columnName" : "photonE4",        "xAxisUnit" : "GeV",     "yAxisTitle" : "Combos / 0.1 GeV",       "binning" : (90, 0, 9)},
          {"columnName" : "proton_momentum", "xAxisUnit" : "GeV",     "yAxisTitle" : "Combos / 2 MeV",         "binning" : (100, 0.2, 1.2)},
          {"columnName" : "proton_z",        "xAxisUnit" : "cm",      "yAxisTitle" : "Combos / 0.4 cm",        "binning" : (100, 40, 80)},
          {"columnName" : "mmsq",            "xAxisUnit" : "GeV^{2}", "yAxisTitle" : "Combos / 0.002 GeV^{2}", "binning" : (100, -0.1, 0.1)},
          # other kinematic variables
          {"columnName" : "Mpi0",           "xAxisUnit" : "GeV", "yAxisTitle" : "Combos / 2 MeV",  "binning" : (100, 0, 0.2)},
          {"columnName" : "Meta",           "xAxisUnit" : "GeV", "yAxisTitle" : "Combos / 3 MeV",  "binning" : (100, 0.4, 0.7)},
          {"columnName" : "Mpi0eta_thrown", "xAxisUnit" : "GeV", "yAxisTitle" : "Combos / 10 MeV", "binning" : (100, 0, 2.5)},
          {"columnName" : "rfTime",         "xAxisUnit" : "ns",  "yAxisTitle" : "Combos / 0.5 ns", "binning" : (100, -25, 25)},
          {"columnName" : "run",            "xAxisUnit" : "",    "yAxisTitle" : "Combos",          "binning" : (2200, 30000, 52000)},
          {"columnName" : "event",          "xAxisUnit" : "",    "yAxisTitle" : "Combos",          "binning" : (100000, 0, 500e6)},
        ]
      histDefs += [
        # moment variables
        {"columnName" : "beamPol",    "xAxisUnit" : "",        "yAxisTitle" : "Combos",            "binning" : (110, 0, 1.1)},
        {"columnName" : "beamPolPhi", "xAxisUnit" : "deg",     "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
        {"columnName" : "cosTheta",   "xAxisUnit" : "",        "yAxisTitle" : "Combos",            "binning" : (100, -1, 1)},
        {"columnName" : "theta",      "xAxisUnit" : "rad",     "yAxisTitle" : "Combos / 0.04 rad", "binning" : (100, 0, 4)},
        {"columnName" : "phiDeg",     "xAxisUnit" : "deg",     "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
        {"columnName" : "phi",        "xAxisUnit" : "rad",     "yAxisTitle" : "Combos / 0.08 rad", "binning" : (100, -4, 4)},
        {"columnName" : "PhiDeg",     "xAxisUnit" : "deg",     "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
        {"columnName" : "Phi",        "xAxisUnit" : "rad",     "yAxisTitle" : "Combos / 0.08 rad", "binning" : (100, -4, 4)},
        {"columnName" : "t",          "xAxisUnit" : "GeV^{2}", "yAxisTitle" : "Combos",            "binning" : (120, 0.09, 0.21)},
        {"columnName" : "mass",       "xAxisUnit" : "GeV",     "yAxisTitle" : "Combos / 10 MeV",   "binning" : (100, 0, 2.5)},
      ]
      hists = []
      for histDef in histDefs:
        cName = histDef["columnName"]
        unit  = histDef["xAxisUnit"]
        hists.append(data.Histo1D((f"h_{cName}", f";{cName}" + (f" [{unit}]" if unit else "") + f";{histDef['yAxisTitle']}",
                                  *histDef["binning"]), (cName, ), weightColumnName))

      # write root tree for moments analysis
      print(f"Writing skimmed tree to file '{outputFileName}'")
      data.Snapshot(outputTreeName, outputFileName,
                    ("beamPol", "beamPolPhi", "cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi", "mass", "eventWeight"))

      # fill and draw histograms
      plotDirName = f"{plotDirBaseName}_{beamPolLabel}"
      os.makedirs(plotDirName, exist_ok = True)
      ROOT.gStyle.SetOptStat(111111)
      for hist in hists:
        canv = ROOT.TCanvas(f"{hist.GetName()}.{dataSetLabel}")
        hist.SetMinimum(0)
        hist.Draw("HIST")
        canv.SaveAs(f"{plotDirName}/{canv.GetName()}.pdf")
