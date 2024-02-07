#!/usr/bin/env python3


import functools
import os
from typing import Dict

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)

  dataSet = "signal"
  inputFileNamePattern = "a0a2_raw/a0a2_2bw_acc_flat.root"
  # dataSet = "phaseSpace"
  # inputFileNamePattern = "t010020_m104180_selectGenTandM/pol000_t010020_m104180_selectGenTandM_F2017_1_selected_acc_flat.root"
  # outputFileName = f"./pol000_t010020_m104180_selectGenTandM_F2017_1_selected_acc_flat.{dataSet}.root"
  # inputFileNamePattern = "a0a2_raw/a0a2_flat_acc_flat.root"
  outputFileName = f"./a0a2_{dataSet}_acc_flat.root"
  inputTreeName = "kin"
  outputTreeName = "etaPi0"
  weightColumnName = "Weight"
  beamPol = 1.0
  beamPolAngle = 0

  # apply fiducial cuts
  data = ROOT.RDataFrame(inputTreeName, inputFileNamePattern).Filter(
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
           ")"
         )

  # define columns for moments analysis
  # ensure that quantities used to calculate moments are doubles
  # coordSys = "gj"
  coordSys = "hel"
  data = (
    data.Define  ("beamPol",     f"(double){beamPol}")
        .Define  ("beamPolPhi",  f"(double){beamPolAngle}")
        .Alias   ("cosTheta",    f"cosTheta_eta_{coordSys}")
        .Define  ("theta",       f"(double)acos(cosTheta_eta_{coordSys})")
        .Alias   ("phiDeg",      f"phi_eta_{coordSys}")
        .Define  ("phi",         f"(double)phi_eta_{coordSys} * TMath::DegToRad()")
        .Define  ("PhiDeg",      "Phi")  # cannot be an Alias because Redefine below would lead to infinite recursion
        .Redefine("Phi",         "(double)Phi * TMath::DegToRad()")
        .Alias   ("mass",        "Mpi0eta")
        # .Alias   ("mass",        "Mpi0eta_thrown")
        .Define  ("eventWeight", f"(double){weightColumnName}")
  )

  # define background-subtracted histograms
  histDefs = (
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
    # moment variables
    {"columnName" : "beamPol",    "xAxisUnit" : "",    "yAxisTitle" : "Combos",            "binning" : (110, 0, 1.1)},
    {"columnName" : "beamPolPhi", "xAxisUnit" : "deg", "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
    {"columnName" : "cosTheta",   "xAxisUnit" : "",    "yAxisTitle" : "Combos",            "binning" : (100, -1, 1)},
    {"columnName" : "theta",      "xAxisUnit" : "rad", "yAxisTitle" : "Combos / 0.04 rad", "binning" : (100, 0, 4)},
    {"columnName" : "phiDeg",     "xAxisUnit" : "deg", "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
    {"columnName" : "phi",        "xAxisUnit" : "rad", "yAxisTitle" : "Combos / 0.08 rad", "binning" : (100, -4, 4)},
    {"columnName" : "PhiDeg",     "xAxisUnit" : "deg", "yAxisTitle" : "Combos / 1 deg",    "binning" : (360, -180, 180)},
    {"columnName" : "Phi",        "xAxisUnit" : "rad", "yAxisTitle" : "Combos / 0.08 rad", "binning" : (100, -4, 4)},
    # other kinematic variables
    {"columnName" : "Mpi0",    "xAxisUnit" : "GeV", "yAxisTitle" : "Combos / 2 MeV",  "binning" : (100, 0, 0.2)},
    {"columnName" : "Meta",    "xAxisUnit" : "GeV", "yAxisTitle" : "Combos / 3 MeV",  "binning" : (100, 0.4, 0.7)},
    {"columnName" : "Mpi0eta", "xAxisUnit" : "GeV", "yAxisTitle" : "Combos / 10 MeV", "binning" : (100, 0, 2.5)},
    {"columnName" : "rfTime",  "xAxisUnit" : "ns",  "yAxisTitle" : "Combos / 0.5 ns", "binning" : (100, -25, 25)},
  )
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
    canv = ROOT.TCanvas(f"{hist.GetName()}.{dataSet}")
    hist.SetMinimum(0)
    hist.Draw("HIST")
    canv.SaveAs(".pdf")
