#!/usr/bin/env python3


import functools
import os
from typing import Dict

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def setup() -> None:
  if ROOT.FSModeCollection.modeVector().size() != 0:
    return
  ROOT.FSHistogram.readHistogramCache(fsRootCacheName)  # cache file names are: <FSRootCacheName>.cache.{dat,root}
  modeInfo = ROOT.FSModeInfo(fsModeString)
  modeInfo.display()
  ROOT.FSModeCollection.addModeInfo(fsModeString).addCategory(fsCategory)
  ROOT.FSTree.addFriendTree("Chi2Rank")
  ROOT.FSTree.showFriendTrees()

  # cuts for real data
  # ROOT.FSCut.defineCut("unusedTracks",   "NumUnusedTracks <= 1")
  # ROOT.FSCut.defineCut("unusedNeutrals", "NumNeutralHypos <= 4")  # # unused neutrals <= 2
  ROOT.FSCut.defineCut("chi2Ndf",        "Chi2DOF < 5")
  ROOT.FSCut.defineCut("beamEnergy",     "(8.2 < EnPB) && (EnPB < 8.8)")  # [GeV]
  # ROOT.FSCut.defineCut("prodVertZ",      "(52 < ProdVz) && (ProdVz < 78)")  # [cm]
  ROOT.FSCut.defineCut("tRange",         "-MASS2(GLUEXTARGET, -[p+]) > 0.1")  # [GeV^2]
  ROOT.FSCut.defineCut("rf",             "abs(RFDeltaT) < 2")  # [ns] in case chi^2 ranking is used
  ROOT.FSCut.defineCut("chi2Rank",       "Chi2Rank == 1")  # from Chi2Rank friend tree
  # ROOT.FSCut.defineCut("cosThetaGJPos",  GJCosThetaVar + " > 0")
  # ROOT.FSCut.defineCut("cosThetaGJNeg",  GJCosThetaVar + " < 0")
  # # sideband cuts
  # ROOT.FSCut.defineCut("rfSB",           "abs(RFDeltaT) < 2", "abs(RFDeltaT) > 2", 0.125)  # [ns]

  # cuts for MC truth
  # ROOT.FSTree.defineFourVector("MCGLUEXBEAM", "MCEnPB", "MCPxPB", "MCPyPB", "MCPzPB")
  # ROOT.FSCut.defineCut("mcBeamEnergy",     "(8.2 < MCEnPB) && (MCEnPB < 8.8)")  # [GeV]
  # # FSCut.defineCut("mcTRange",         "-MCMASS2(GLUEXTARGET, -1, -4) < 0.5")  # [0, 0.5] [GeV^2]
  # ROOT.FSCut.defineCut("mcTRange",         "0.1 < (-MCMASS2(GLUEXTARGET, -1, -4)) && (-MCMASS2(GLUEXTARGET, -1, -4) < 0.5)")  # [0.1, 0.5] [GeV^2]
  # ROOT.FSCut.defineCut("mcCosThetaGJPos",  GJCosThetaVarMC + " > 0")
  # ROOT.FSCut.defineCut("mcCosThetaGJNeg",  GJCosThetaVarMC + " < 0")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  print("Loading ./FSRootMacros.C...")
  assert ROOT.gROOT.LoadMacro("./FSRootMacros.C+") == 0, "Error loading './FSRootMacros.C'."

  useRDataFrame   = True
  fsRootCacheName = ".FSRootCache"
  fsTreeName      = "ntFSGlueX_MODECODE"
  fsModeString    = "100_110"
  fsCategory      = "m" + fsModeString

  dataSet = "signal"
  # dataSet = "phaseSpace"
  inputFileNamePattern = f"./{dataSet}_FSRoot/tree_pippim__B4_gen_amp_030994_???.root"
  beamPol = 0.4

  setup()
  if useRDataFrame:
    ROOT.FSHistogram.enableRDataFrame(False)  # false = delay filling of histograms until FSHistogram::executeRDataFrame() is called

  # ROOT.FSCut.defineCut("cutSet", "")
  # ROOT.FSCut.defineCut("cutSet", "CUT(beamEnergy, rf, chi2Rank)")
  ROOT.FSCut.defineCut("cutSet", "CUT(chi2Ndf, beamEnergy, rf, tRange, chi2Rank)")
  cutString = "CUT(cutSet)"

  hists: Dict[str, ROOT.TH1F] = {}
  hists["hMassPiPi"]    = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, "MASS([pi+], [pi-])",         "(100, 0, 1.5)",   cutString))
  hists["hChi2Ndf"]     = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, "Chi2DOF",                    "(100, 0, 25)",    cutString))
  hists["hChi2Rank"]    = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, "Chi2Rank",                   "(10, -1.5, 8.5)", cutString))
  hists["hBeamEnergy"]  = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, "EnPB",                       "(400, 2, 12)",    cutString))
  hists["hRfDeltaT"]    = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, "RFDeltaT",                   "(400, -20, 20)",  cutString))
  hists["hMomTransfer"] = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, "-MASS2(GLUEXTARGET, -[p+])", "(100, 0, 2)",     cutString))

  # angular variables
  # for beam + target -> X + recoil and X -> a + b (see FSBasic/FSMath.h)
  # angles of particle a in X Gottfried-Jackson RF are calculated by
  #   GJCOSTHETA(a; b; beam)
  #   GJPHI(a; b; recoil; beam) [rad]
  GjCosThetaDef = "GJCOSTHETA([pi+]; [pi-]; GLUEXBEAM)"
  GjThetaDef    = f"acos({GjCosThetaDef})"
  GjPhiDef      = "GJPHI([pi+]; [pi-]; [p+]; GLUEXBEAM)"
  GjPhiDegDef   = f"{GjPhiDef} * TMath::RadToDeg()"
  # ROOT.FSTree.defineMacro("MYGJPHI", 4, "MyFSMath::gjphi("
  #   "PxP[I],PyP[I],PzP[I],EnP[I],"
  #   "PxP[J],PyP[J],PzP[J],EnP[J],"
  #   "PxP[M],PyP[M],PzP[M],EnP[M],"
  #   "PxP[N],PyP[N],PzP[N],EnP[N])")
  # MyGjPhiDef    = "MYGJPHI([pi+]; [pi-]; [p+]; GLUEXBEAM)"
  # MyGjPhiDegDef = f"{MyGjPhiDef} * TMath::RadToDeg()"
  # angles of particle a in X helicity RF are calculated by
  #   HELCOSTHETA(a; b; recoil)
  #   HELPHI(a; b; recoil; beam) [rad]
  HfCosThetaDef = "HELCOSTHETA([pi+]; [pi-]; [p+])"
  HfThetaDef    = f"acos({HfCosThetaDef})"
  HfPhiDef      = "HELPHI([pi+]; [pi-]; [p+]; GLUEXBEAM)"
  HfPhiDegDef   = f"{HfPhiDef} * TMath::RadToDeg()"
  ROOT.FSTree.defineMacro("BIGPHI", 2, "MyFSMath::bigPhi("
    "PxP[I],PyP[I],PzP[I],EnP[I],"
    "PxP[J],PyP[J],PzP[J],EnP[J],"
    "(PolarizationAngle))")
  bigPhiDef    = "BIGPHI([p+]; GLUEXBEAM)"
  bigPhiDegDef = f"{bigPhiDef} * TMath::RadToDeg()"
  print(f"Defined macro: {ROOT.FSTree.expandVariable(bigPhiDef)}")
  hists["hGjCosTheta"] = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, GjCosThetaDef, "(100, -1, +1)",     cutString))
  hists["hGjPhi"]      = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, GjPhiDegDef,   "(100, -180, +180)", cutString))
  # hists["hMyGjPhi"]    = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, MyGjPhiDegDef, "(100, -180, +180)", cutString))
  hists["hHfCosTheta"] = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, HfCosThetaDef, "(100, -1, +1)",     cutString))
  hists["hHfPhi"]      = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, HfPhiDegDef,   "(100, -180, +180)", cutString))
  hists["hPhi"]        = (ROOT.FSModeHistogram.getTH1F(inputFileNamePattern, fsTreeName, fsCategory, bigPhiDegDef,  "(100, -180, +180)", cutString))

  # draw histograms and write tree
  if useRDataFrame:
    ROOT.FSHistogram.executeRDataFrame()
  for histName, hist in hists.items():
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.DrawCopy()
    canv.SaveAs(f"{histName}.{dataSet}.pdf")

  # root tree for moments analysis
  skimFileName = f"./tree_pippim__B4_gen_amp_030994.{dataSet}.root"
  print(f"Writing skimmed tree to file '{skimFileName}'")
  ROOT.FSModeTree.skimTree(inputFileNamePattern, fsTreeName, fsCategory, skimFileName, cutString)
  varDefs = ROOT.std.vector[ROOT.std.pair[str, str]]()
  varDefs.push_back(ROOT.std.pair[str, str]("beamPol",    f"{beamPol}"))
  varDefs.push_back(ROOT.std.pair[str, str]("beamPolPhi", "PolarizationAngle"))
  # varDefs.push_back(ROOT.std.pair[str, str]("cosTheta",   GjCosThetaDef))
  # varDefs.push_back(ROOT.std.pair[str, str]("theta",      GjThetaDef))
  # varDefs.push_back(ROOT.std.pair[str, str]("phiDeg",     GjPhiDegDef))
  # varDefs.push_back(ROOT.std.pair[str, str]("phi",        GjPhiDef))
  varDefs.push_back(ROOT.std.pair[str, str]("cosTheta",   HfCosThetaDef))
  varDefs.push_back(ROOT.std.pair[str, str]("theta",      HfThetaDef))
  varDefs.push_back(ROOT.std.pair[str, str]("phiDeg",     HfPhiDegDef))
  varDefs.push_back(ROOT.std.pair[str, str]("phi",        HfPhiDef))
  varDefs.push_back(ROOT.std.pair[str, str]("Phi",        bigPhiDef))
  varDefs.push_back(ROOT.std.pair[str, str]("PhiDeg",     bigPhiDegDef))
  # varDefs.push_back(ROOT.std.pair[str, str]("myphi",      MyGjPhiDef))
  ROOT.FSModeTree.createFriendTree(skimFileName, fsTreeName, fsCategory, "angles", varDefs)

  # write FSRoot cache
  ROOT.FSHistogram.dumpHistogramCache(fsRootCacheName)
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogoff.FSROOT.C")
