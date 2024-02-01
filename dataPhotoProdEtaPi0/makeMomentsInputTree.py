#!/usr/bin/env python3


import functools
import os
from typing import Dict

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)

  # dataSet = "signal"
  dataSet = "phaseSpace"
  inputFileNamePattern = "t010020_m104180_selectGenTandM/pol000_t010020_m104180_selectGenTandM_F2017_1_selected_acc_flat.root"
  skimFileName = f"./pol000_t010020_m104180_selectGenTandM_F2017_1_selected_acc_flat.{dataSet}.root"
  treeName = "kin"
  weightColumnName = "Weight"
  beamPol = 0.4
  beamPolAngle = 0

  # apply fiducial cuts
  data = ROOT.RDataFrame(treeName, inputFileNamePattern).Filter(
           "(pVH > 0.5)"
           "&& (unusedEnergy < 0.01)"
           "&& (chiSq < 13.277)"
           "&& (((2.5 < photonTheta1) && (photonTheta1 < 10.3)) || (photonTheta1 > 11.9))"
           "&& (((2.5 < photonTheta2) && (photonTheta2 < 10.3)) || (photonTheta2 > 11.9))"
           "&& (((2.5 < photonTheta3) && (photonTheta3 < 10.3)) || (photonTheta3 > 11.9))"
           "&& (((2.5 < photonTheta4) && (photonTheta4 < 10.3)) || (photonTheta4 > 11.9))"
           "&& (photonE1 > 0.1)"
           "&& (photonE2 > 0.1)"
           "&& (photonE3 > 0.1)"
           "&& (photonE4 > 0.1)"
           "&& (proton_momentum > 0.3)"
           "&& ((52 < proton_z) && (proton_z < 78))"
           "&& (abs(mmsq) < 0.05)"
         )

  # define columns for moments analysis
  data = (
    data.Define  ("beamPol",    f"{beamPol}")
        .Define  ("beamPolPhi", f"{beamPolAngle}")
        .Define  ("cosTheta",   "cosTheta_eta_gj")
        .Define  ("theta",      "acos(cosTheta_eta_gj)")
        .Define  ("phiDeg",     "phi_eta_gj")
        .Define  ("phi",        "phi_eta_gj * TMath::DegToRad()")
        .Define  ("PhiDeg",     "Phi")
        .Redefine("Phi",        "Phi * TMath::DegToRad()")
  )

  # define histograms
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
    {"columnName" : "proton_momentum", "xAxisUnit" : "GeV",     "yAxisTitle" : "Combos / 2 MeV",         "binning" : (100, 0.3, 0.5)},
    {"columnName" : "proton_z",        "xAxisUnit" : "cm",      "yAxisTitle" : "Combos / 0.4 cm",        "binning" : (100, 40, 80)},
    {"columnName" : "mmsq",            "xAxisUnit" : "GeV^{2}", "yAxisTitle" : "Combos / 0.002 GeV^{2}", "binning" : (100, -0.1, 0.1)},
    # moment variables
    {"columnName" : "beamPol",    "xAxisUnit" : "",    "yAxisTitle" : "Combos",            "binning" : (100, 0, 1)},
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
    {"columnName" : "Mpi0eta", "xAxisUnit" : "GeV", "yAxisTitle" : "Combos / 10 MeV", "binning" : (100, 1, 2)},
  )
  hists = []
  for histDef in histDefs:
    cName = histDef["columnName"]
    unit = histDef["xAxisUnit"]
    hists.append(data.Histo1D((f"h_{cName}", f";{cName}" + (f" [{unit}]" if unit else "") + f";{histDef['yAxisTitle']}",
                               *histDef["binning"]), (cName,), weightColumnName))

  # write root tree for moments analysis
  print(f"Writing skimmed tree to file '{skimFileName}'")
  data.Snapshot("etaPi0", skimFileName, ("beamPol", "beamPolPhi", "cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi"))

  # draw histograms
  ROOT.gStyle.SetOptStat(111111)
  for hist in hists:
    canv = ROOT.TCanvas(f"{hist.GetName()}.{dataSet}")
    hist.SetMinimum(0)
    hist.Draw("HIST")
    canv.SaveAs(".pdf")
