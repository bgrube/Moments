#!/usr/bin/env python3

import functools
from typing import (
  Dict,
  List,
)

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# delcare C++ function to calculate azimuthal angle of photon polarization vector
ROOT.gInterpreter.Declare(
"""
// returns azimuthal angle of photon polarization vector in lab frame [rad]
// for beam + target -> X + recoil and X -> a + b
//     D                    C
// code taken from https://github.com/JeffersonLab/halld_sim/blob/538677ee1347891ccefa5780e01b158e035b49b1/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
double
bigPhi(
	const double PxPC, const double PyPC, const double PzPC, const double EnPC,  // recoil
	const double PxPD, const double PyPD, const double PzPD, const double EnPD   // beam
) {
	const double polAngle = 0;  // polarization angle [deg]
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
		if (std::abs(deltaPhi) > 1e-15) {
			std::cout << "!!! Phi2 = " << Phi2 << " - Phi = " << Phi << ": " << deltaPhi << std::endl;
		}
	}
	return Phi;
}
""")


def defineDataFrameColumns(
  data:             ROOT.RDataFrame,
  coordSys:         str   = "hel",  # either "hel" for helicity frame or "gj" for Gottfried-Jackson frame
  beamPol:          float = 1.0,
  beamPolAngle:     float = 0,
  weightColumnName: str   = "Weight",
  thrownData:       bool  = False,
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns for moments analysis"""
  columnSuffix = "_thrown" if thrownData else ""
  df = (
    # ensure that quantities used to calculate moments are doubles
    data.Define("beamPol",     f"(double){beamPol}")
        .Define("beamPolPhi",  f"(double){beamPolAngle}")
        .Define("cosTheta",    f"(double)cosTheta_eta_{coordSys}{columnSuffix}")
        .Define("theta",       f"(double)acos(cosTheta_eta_{coordSys}{columnSuffix})")
        .Define("mass",        f"(double)Mpi0eta{columnSuffix}")
        .Define("eventWeight", f"(double){weightColumnName}")
  )
  bigPhiFunc = "bigPhi(Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0], Px_Beam, Py_Beam, Pz_Beam, E_Beam)"
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

  dataSets: List[Dict[str, str]] = [
    # {"dataSetLabel"         : "signal_acc",
    #  "inputFileNamePattern" : "a0a2_raw/a0a2_2bw_acc_flat.root"},
    # {"dataSetLabel"         : "signal_gen",
    #  "inputFileNamePattern" : "a0a2_raw/a0a2_2bw_gen_data_flat.root"},
    # {"dataSetLabel"         : "phaseSpace_acc",
    #  "inputFileNamePattern" : "a0a2_raw/a0a2_flat_acc_flat.root"},
    # {"dataSetLabel"         : "phaseSpace_gen",
    #  "inputFileNamePattern" : "a0a2_raw/a0a2_flat_gen_data_flat.root"},
    #
    {"dataSetLabel"         : "signal_acc",
     "inputFileNamePattern" : "SELECTED_a0a2_forBoris/a0a2_bin_10_acc_flat.root"},
    {"dataSetLabel"         : "signal_gen",
     "inputFileNamePattern" : "SELECTED_a0a2_forBoris/a0a2_bin10_gen_data_flat.root"},
    {"dataSetLabel"         : "phaseSpace_acc",
     "inputFileNamePattern" : "SELECTED_a0a2_forBoris/flat_acc_flat.root"},
    {"dataSetLabel"         : "phaseSpace_gen",
     "inputFileNamePattern" : "SELECTED_a0a2_forBoris/flat_gen_data_flat.root"},
  ]
  inputTreeName    = "kin"
  outputTreeName   = "etaPi0"
  weightColumnName = "Weight"
  beamPol          = 1.0
  beamPolAngle     = 0

  for dataSet in dataSets:
    dataSetLabel         = dataSet["dataSetLabel"]
    inputFileNamePattern = dataSet["inputFileNamePattern"]
    outputFileName       = f"./a0a2_{dataSetLabel}_flat.root"
    thrownData           = ((dataSetLabel == "signal_gen") or (dataSetLabel == "phaseSpace_gen"))

    # apply fiducial cuts
    data = ROOT.RDataFrame(inputTreeName, inputFileNamePattern)
    if not thrownData:
      data = data.Filter(
              "("
                "(pVH > 0.5)"
                "&& (unusedEnergy < 0.01)"  # [GeV]
                "&& (chiSq < 13.277)"
                "&& (((2.5 < photonTheta1) && (photonTheta1 < 10.3)) || (photonTheta1 > 11.9))"  # [deg]
                "&& (((2.5 < photonTheta2) && (photonTheta2 < 10.3)) || (photonTheta2 > 11.9))"  # [deg]
                "&& (((2.5 < photonTheta3) && (photonTheta3 < 10.3)) || (photonTheta3 > 11.9))"  # [deg]
                "&& (((2.5 < photonTheta4) && (photonTheta4 < 10.3)) || (photonTheta4 > 11.9))"  # [deg]
                "&& (photonE1 > 0.1)"  # [GeV]
                "&& (photonE2 > 0.1)"  # [GeV]
                "&& (photonE3 > 0.1)"  # [GeV]
                "&& (photonE4 > 0.1)"  # [GeV]
                "&& (proton_momentum > 0.3)"  # [GeV]
                "&& ((52 < proton_z) && (proton_z < 78))"  # [cm]
                "&& (abs(mmsq) < 0.05)"  # [GeV^2]
                "&& ((0.8 < Mpi0eta_thrown) && (Mpi0eta_thrown < 2.0))"  # [GeV]
                # "&& (run < 51400)"  # exclude region were signal and phase-space sample differ signficantly
              ")"
            )

    data = defineDataFrameColumns(data, coordSys = "hel",  beamPol = beamPol, beamPolAngle = beamPolAngle,
                                  weightColumnName = weightColumnName, thrownData = thrownData)

    # define background-subtracted histograms
    histDefs = []
    if not thrownData:
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
        {"columnName" : "run",            "xAxisUnit" : "",    "yAxisTitle" : "Combos",          "binning" : (1200, 50600, 51800)},
      ]
    histDefs += [
      # moment variables
      {"columnName" : "beamPol",    "xAxisUnit" : "",    "yAxisTitle" : "Combos",            "binning" : (110, 0, 1.1)},
      {"columnName" : "beamPolPhi", "xAxisUnit" : "deg", "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
      {"columnName" : "cosTheta",   "xAxisUnit" : "",    "yAxisTitle" : "Combos",            "binning" : (100, -1, 1)},
      {"columnName" : "theta",      "xAxisUnit" : "rad", "yAxisTitle" : "Combos / 0.04 rad", "binning" : (100, 0, 4)},
      {"columnName" : "phiDeg",     "xAxisUnit" : "deg", "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
      {"columnName" : "phi",        "xAxisUnit" : "rad", "yAxisTitle" : "Combos / 0.08 rad", "binning" : (100, -4, 4)},
      {"columnName" : "PhiDeg",     "xAxisUnit" : "deg", "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
      {"columnName" : "Phi",        "xAxisUnit" : "rad", "yAxisTitle" : "Combos / 0.08 rad", "binning" : (100, -4, 4)},
      {"columnName" : "mass",       "xAxisUnit" : "GeV", "yAxisTitle" : "Combos / 10 MeV",   "binning" : (100, 0, 2.5)},
    ]
    hists = []
    for histDef in histDefs:
      cName = histDef["columnName"]
      unit  = histDef["xAxisUnit"]
      hists.append(data.Histo1D((f"h_{cName}", f";{cName}" + (f" [{unit}]" if unit else "") + f";{histDef['yAxisTitle']}",
                                *histDef["binning"]), (cName,), weightColumnName))

    # write root tree for moments analysis
    print(f"Writing skimmed tree to file '{outputFileName}'")
    data.Snapshot(outputTreeName, outputFileName,
                  ("beamPol", "beamPolPhi", "cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi", "mass", "eventWeight"))

    # fill and draw histograms
    ROOT.gStyle.SetOptStat(111111)
    for hist in hists:
      canv = ROOT.TCanvas(f"{hist.GetName()}.{dataSetLabel}")
      hist.SetMinimum(0)
      hist.Draw("HIST")
      canv.SaveAs(".pdf")
