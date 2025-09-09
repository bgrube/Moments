#!/usr/bin/env python3


from __future__ import annotations

import ctypes
import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_MASSPAIR,
  CPP_CODE_MANDELSTAM_T,
  defineAngleFormulas,
  lorentzVectors,
)


# Alex' code to calculate helicity angles of A for beam + target -> X + recoil with X -> A + B
CPP_CODE_ALEX = """
TVector3
analyzerVectorHf(
  const double PxPA,     const double PyPA,     const double PzPA,     const double EPA,
  const double PxPB,     const double PyPB,     const double PzPB,     const double EPB,
  const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
  const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
  // boost all 4-vectors into the X rest frame
  TLorentzVector locAP4_XRF     (PxPA,     PyPA,     PzPA,     EPA);
  TLorentzVector locBP4_XRF     (PxPB,     PyPB,     PzPB,     EPB);
  TLorentzVector locTargetP4_XRF(0,        0,        0,        0.938271999359130859375);  // proton mass value from phase-space generator
  TLorentzVector locRecoilP4_XRF(PxRecoil, PyRecoil, PzRecoil, ERecoil);
  TLorentzVector locBeamP4_XRF  (PxBeam,   PyBeam,   PzBeam,   EBeam);
  const TLorentzVector XP4 = locAP4_XRF + locBP4_XRF;
  const TVector3 boostP3 = -XP4.BoostVector();
  locAP4_XRF.Boost     (boostP3);
  locBP4_XRF.Boost     (boostP3);
  locTargetP4_XRF.Boost(boostP3);
  locRecoilP4_XRF.Boost(boostP3);
  locBeamP4_XRF.Boost  (boostP3);

  // construct coordinate system
  const TVector3 locAP3_XRF      = locAP4_XRF.Vect();
  const TVector3 locTargetP3_XRF = locTargetP4_XRF.Vect();
  const TVector3 locRecoilP3_XRF = locRecoilP4_XRF.Vect();
  const TVector3 locBeamP3_XRF   = locBeamP4_XRF.Vect();
  // helicity frame: z-axis opposite to recoil in X rest frame
  const TVector3 z = -locRecoilP3_XRF.Unit();
  // // Gottfried-Jackson frame for meson resonances: z-axis along beam photon
  // const TVector3 z = locBeamP3_XRF.Unit();
  // // Gottfried-Jackson frame for baryon resonances: z-axis along target (see Delta++ SDME Eq. (2))
  // const TVector3 z = locTargetP3_XRF.Unit();
  // y-axis: normal to the production plane
  // flipping the y-axis leads to sign flip of odd-M moments
  const TVector3 y = (locRecoilP3_XRF.Cross(locBeamP3_XRF)).Unit();  // convention used in COMPASS; in Mathieu et al., PRD 100 (2019) 054017, Appendix A; in CLAS, PRD 80 (2009) 072005;and in GlueX, PRC 108 (2023) 055204
  // const TVector3 y = (locBeamP3_XRF.Cross(locRecoilP3_XRF)).Unit();  // convention used in Yu et al., PRC 96 (2017) 025208, Appendix A; in GlueX, PRC 105 (2022) 035201, Eq. (1); and in Delta++ SDMEs, Eqs. (2) and (A.1)
  // x-axis: right-handed coordinate system
  const TVector3 x = y.Cross(z);
  const TVector3 v(locAP3_XRF * x, locAP3_XRF * y, locAP3_XRF * z);
  return v;
}

double
cosTheta_Alex(
  const double PxPA,     const double PyPA,     const double PzPA,     const double EPA,
  const double PxPB,     const double PyPB,     const double PzPB,     const double EPB,
  const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
  const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
  const TVector3 v = analyzerVectorHf(
    PxPA,     PyPA,     PzPA,     EPA,
    PxPB,     PyPB,     PzPB,     EPB,
    PxRecoil, PyRecoil, PzRecoil, ERecoil,
    PxBeam,   PyBeam,   PzBeam,   EBeam
  );
  return v.CosTheta();
}

double
phiDeg_Alex(
  const double PxPA,     const double PyPA,     const double PzPA,     const double EPA,
  const double PxPB,     const double PyPB,     const double PzPB,     const double EPB,
  const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
  const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
  const TVector3 v = analyzerVectorHf(
    PxPA,     PyPA,     PzPA,     EPA,
    PxPB,     PyPB,     PzPB,     EPB,
    PxRecoil, PyRecoil, PzRecoil, ERecoil,
    PxBeam,   PyBeam,   PzBeam,   EBeam
  );
  return v.Phi() * TMath::RadToDeg();
}
"""


def convertGraphToHist(
  graph:     ROOT.TGraphErrors,
  binning:   tuple[int, float, float],
  histName:  str,
  histTitle: str = "",
) -> ROOT.TH1D:
  """Converts `TGraphErrors` to `TH1D` assuming equidistant binning"""
  hist = ROOT.TH1D(histName, histTitle, *binning)
  for pointIndex in range(graph.GetN()):
    x = ctypes.c_double(0.0)
    y = ctypes.c_double(0.0)
    graph.GetPoint(pointIndex, x, y)
    yErr = graph.GetErrorY(pointIndex)
    histBin = hist.FindFixBin(x)
    hist.SetBinContent(histBin, y)
    hist.SetBinError  (histBin, yErr)
  return hist


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_ALEX)

  tBinLabel = "tbin_0.4_0.5"
  dataSet               = "2017_01-ver04-70"
  dataSigRegionFileName = f"./{dataSet}/{tBinLabel}/amptools_tree_data_tbin1_ebin4.root"
  dataBkgRegionFileName = f"./{dataSet}/{tBinLabel}/amptools_tree_bkgnd_tbin1_ebin4.root"
  mcDataFileName        = f"./{dataSet}/{tBinLabel}/amptools_tree_accepted_tbin1_ebin4*.root"
  # dataSet               = "2018_08-ver02-05"
  # dataSigRegionFileName = f"./{dataSet}/{tBinLabel}/amptools_tree_2018-08_LE_signal.root"
  # dataBkgRegionFileName = f"./{dataSet}/{tBinLabel}/amptools_tree_2018-08_LE_bkgnd.root"
  # mcDataFileName        = f"./{dataSet}/{tBinLabel}/tree_amptools_recon.root"
  treeName      = "kin"
  outputDirName = f"./{dataSet}/{tBinLabel}/dataPlots"

  # create friend trees with correct weights
  os.makedirs(outputDirName, exist_ok = True)
  for dataFileName, weightFormula in [(dataSigRegionFileName, "Weight"), (dataBkgRegionFileName, "-Weight")]:
    friendFileName = f"{outputDirName}/{os.path.basename(dataFileName)}.weights"
    if os.path.exists(friendFileName):
      print(f"File '{friendFileName}' already exists, skipping creation of friend tree")
      continue
    print(f"Creating file '{friendFileName}' that contains friend tree with weights for file '{dataFileName}'")
    ROOT.RDataFrame(treeName, dataFileName) \
        .Define("eventWeight", weightFormula) \
        .Snapshot(treeName, friendFileName, ["eventWeight"])
  # attach friend trees to data tree
  dataTChain = ROOT.TChain(treeName)
  weightTChain = ROOT.TChain(treeName)
  for dataFileName in [dataSigRegionFileName, dataBkgRegionFileName]:
    dataTChain.Add(dataFileName)
    friendFileName = f"{outputDirName}/{os.path.basename(dataFileName)}.weights"
    weightTChain.Add(friendFileName)
  dataTChain.AddFriend(weightTChain)

  # create RDataFrame from real data in AmpTools format and define columns
  #!NOTE! coordinate system definitions for beam + target -> pi+ + pi- + recoil (all momenta in XRF):
  #    HF for pi+ pi- meson system:  use pi+  as analyzer and z_HF = -p_recoil and y_HF = p_recoil x p_beam
  #    HF for pi+- p  baryon system: use pi+- as analyzer and z_HF = -p_pi-+   and y_HF = p_beam   x p_pi-+
  #    GJ for pi+ pi- meson system:  use pi+  as analyzer and z_GJ = p_beam    and y_HF = p_recoil x p_beam
  #    GJ for pi+- p  baryon system: use pi+- as analyzer and z_GJ = p_target  and y_HF = p_beam   x p_pi-+
  lvBeamPhoton, lvTargetProton, lvRecoilProton, lvPip, lvPim = lorentzVectors(realData = True)
  df = ROOT.RDataFrame(dataTChain)
  for pairLabel, pairLvs, lvRecoil, lvBeamGJ, flipYAxis in (
    ("PiPi", (lvPip, lvPim         ), lvRecoilProton, lvBeamPhoton,   True),
    ("PipP", (lvPip, lvRecoilProton), lvPim,          lvTargetProton, False),
    ("PimP", (lvPim, lvRecoilProton), lvPip,          lvTargetProton, False),
  ):  # loop over two-body subsystems of pi+ pi- p final state
    for frame in ("Hf", "Gj"):  # loop over rest frame definitions
      df = defineAngleFormulas(
        df,
        lvBeamPhoton if frame == "Hf" else lvBeamGJ, lvRecoil, pairLvs[0], pairLvs[1],
        frame,
        flipYAxis,
        columnNames = {  # names of columns to define: key: column, value: name
          "cosThetaCol" : f"{frame}{pairLabel}CosTheta",
          "thetaCol"    : f"{frame}{pairLabel}Theta",
          "phiCol"      : f"{frame}{pairLabel}Phi",
        },
      )
    df = (
      df.Define(f"Mass{pairLabel}",   f"massPair({pairLvs[0]}, {pairLvs[1]})")
        .Define(f"Mass{pairLabel}Sq", f"std::pow(massPair({pairLvs[0]}, {pairLvs[1]}), 2)")
    )
  df = df.Define("minusT", f"-mandelstamT({lvTargetProton}, {lvRecoilProton})")
  df = (
    #                                                                [    two-body system    ]  [   "recoil"   ]  [   "beam"   ]
    df.Define("HfPiPiCosThetaDiff", f"HfPiPiCosTheta - cosTheta_Alex({lvPip}, {lvPim},          {lvRecoilProton}, {lvBeamPhoton})")
      .Define("HfPiPiPhiDegDiff",   f"HfPiPiPhiDeg   - phiDeg_Alex  ({lvPip}, {lvPim},          {lvRecoilProton}, {lvBeamPhoton})")
      .Define("HfPipPCosThetaDiff", f"HfPipPCosTheta - cosTheta_Alex({lvPip}, {lvRecoilProton}, {lvPim},          {lvBeamPhoton})")
      .Define("HfPipPPhiDegDiff",   f"HfPipPPhiDeg   - phiDeg_Alex  ({lvPip}, {lvRecoilProton}, {lvPim},          {lvBeamPhoton})")
      .Define("HfPimPCosThetaDiff", f"HfPimPCosTheta - cosTheta_Alex({lvPim}, {lvRecoilProton}, {lvPip},          {lvBeamPhoton})")
      .Define("HfPimPPhiDegDiff",   f"HfPimPPhiDeg   - phiDeg_Alex  ({lvPim}, {lvRecoilProton}, {lvPip},          {lvBeamPhoton})")
    # df.Define("GjPiPiCosThetaDiff", f"GjPiPiCosTheta - cosTheta_Alex({lvPip}, {lvPim},          {lvRecoilProton}, {lvBeamPhoton}  )")
    #   .Define("GjPiPiPhiDegDiff",   f"GjPiPiPhiDeg   - phiDeg_Alex  ({lvPip}, {lvPim},          {lvRecoilProton}, {lvBeamPhoton}  )")
    #   .Define("GjPipPCosThetaDiff", f"GjPipPCosTheta - cosTheta_Alex({lvPip}, {lvRecoilProton}, {lvPim},          {lvTargetProton})")
    #   .Define("GjPipPPhiDegDiff",   f"GjPipPPhiDeg   - phiDeg_Alex  ({lvPip}, {lvRecoilProton}, {lvPim},          {lvTargetProton})")
    #   .Define("GjPimPCosThetaDiff", f"GjPimPCosTheta - cosTheta_Alex({lvPim}, {lvRecoilProton}, {lvPip},          {lvTargetProton})")
    #   .Define("GjPimPPhiDegDiff",   f"GjPimPPhiDeg   - phiDeg_Alex  ({lvPim}, {lvRecoilProton}, {lvPip},          {lvTargetProton})")
  )

  # define real-data histograms applying RF-sideband subtraction
  yAxisLabel = "RF-Sideband Subtracted Combos"
  hists = [
    df.Histo1D(ROOT.RDF.TH1DModel("hDataEbeam",        ";E_{beam} [GeV];"          + yAxisLabel,  50, 3.55, 3.80), "E_Beam",   "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataPiPiMass",     ";m_{#pi#pi} [GeV];"        + yAxisLabel, 400, 0.28, 2.28), "MassPiPi", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataPiPiMassClas", ";m_{#pi#pi} [GeV];"        + yAxisLabel, 200, 0,    2),    "MassPiPi", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataPipPMass",     ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel, 400, 1,    5),    "MassPipP", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataPipPMassClas", ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel,  72, 1,    2.8),  "MassPipP", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataPimPMass",     ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel, 400, 1,    5),    "MassPimP", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataPimPMassClas", ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel,  72, 1,    2.8),  "MassPimP", "eventWeight"),
    df.Histo1D(ROOT.RDF.TH1DModel("hDataMinusT",       ";#minus t [GeV^{2}];"      + yAxisLabel, 100, 0,    1),    "minusT",   "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataPipPMassVsPiPiMassClas",   ";m_{#pi#pi} [GeV];m_{p#pi^{#plus}} [GeV]",                  100, 0.2,  1.8,  100,    1,  2.8), "MassPiPi",   "MassPipP",       "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz1",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#plus}}^{2} [GeV^{2}]",  100, 0,    3.5,  100,    1,  7.5), "MassPiPiSq", "MassPipPSq",     "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz2",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#minus}}^{2} [GeV^{2}]", 100, 0,    3.5,  100,    1,  7.5), "MassPiPiSq", "MassPimPSq",     "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataPiPiMassVsHfCosThetaAlex", ";m_{#pi#pi} [GeV];cos#theta_{HF}",                          100, 0.28, 2.28,  72,   -1,   +1), "MassPiPi",   "HfPiPiCosTheta", "eventWeight"),
    df.Histo2D(ROOT.RDF.TH2DModel("hDataPiPiMassVsHfPhiDegAlex",   ";m_{#pi#pi} [GeV];#phi_{HF}",                               100, 0.28, 2.28,  72, -180, +180), "MassPiPi",   "HfPiPiPhiDeg",   "eventWeight"),
  ]
  # add histograms specific to subsystems
  for pairLabel, massAxisTitle, massBinning in (
    ("PiPi", "m_{#pi#pi} [GeV]",        (56, 0.28, 1.40)),
    ("PipP", "m_{p#pi^{#plus}} [GeV]",  (72, 1,    2.8 )),
    ("PimP", "m_{p#pi^{#minus}} [GeV]", (72, 1,    2.8 )),
  ):
    hists += [
      df.Histo1D(ROOT.RDF.TH1DModel(f"hData{pairLabel}HfCosTheta_diff", ";#Delta cos#theta_{HF}",  1000, -3e-13, +3e-13), f"Hf{pairLabel}CosThetaDiff", "eventWeight"),
      df.Histo1D(ROOT.RDF.TH1DModel(f"hData{pairLabel}HfPhiDeg_diff",   ";#Delta #phi_{HF} [deg]", 1000, -1e-11, +1e-11), f"Hf{pairLabel}PhiDegDiff",   "eventWeight"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hData{pairLabel}AnglesGj",          ";cos#theta_{GJ};#phi_{GJ} [deg]",      50, -1,   +1, 50, -180, +180), f"Gj{pairLabel}CosTheta", f"Gj{pairLabel}PhiDeg",   "eventWeight"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hData{pairLabel}AnglesHf",          ";cos#theta_{HF};#phi_{HF} [deg]",      50, -1,   +1, 50, -180, +180), f"Hf{pairLabel}CosTheta", f"Hf{pairLabel}PhiDeg",   "eventWeight"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hData{pairLabel}MassVsGjCosTheta", f";{massAxisTitle}" + ";cos#theta_{GJ}", *massBinning, 72,   -1,   +1), f"Mass{pairLabel}",       f"Gj{pairLabel}CosTheta", "eventWeight"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hData{pairLabel}MassVsGjPhiDeg",   f";{massAxisTitle}" + ";#phi_{GJ}",      *massBinning, 72, -180, +180), f"Mass{pairLabel}",       f"Gj{pairLabel}PhiDeg",   "eventWeight"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hData{pairLabel}MassVsHfCosTheta", f";{massAxisTitle}" + ";cos#theta_{HF}", *massBinning, 72,   -1,   +1), f"Mass{pairLabel}",       f"Hf{pairLabel}CosTheta", "eventWeight"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hData{pairLabel}MassVsHfPhiDeg",   f";{massAxisTitle}" + ";#phi_{HF}",      *massBinning, 72, -180, +180), f"Mass{pairLabel}",       f"Hf{pairLabel}PhiDeg",   "eventWeight"),
    ]
  # create acceptance histograms for pi pi GJ and HF angles in bins of m_pipi
  massPiPiRange    = (0.28, 1.40)  # [GeV]
  massPiPiNmbBins  = 28
  massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
  for binIndex in range(0, massPiPiNmbBins):
    massPiPiBinMin    = massPiPiRange[0] + binIndex * massPiPiBinWidth
    massPiPiBinMax    = massPiPiBinMin + massPiPiBinWidth
    massPiPiBinFilter = f"({massPiPiBinMin} < MassPiPi) and (MassPiPi < {massPiPiBinMax})"
    histNameSuffix    = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
    hists += [
      df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hDataPiPiAnglesGj{histNameSuffix}", ";cos#theta_{GJ};#phi_{GJ} [deg]", 50, -1, +1, 50, -180, +180), "GjPiPiCosTheta", "GjPiPiPhiDeg"),
      df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hDataPiPiAnglesHf{histNameSuffix}", ";cos#theta_{HF};#phi_{HF} [deg]", 50, -1, +1, 50, -180, +180), "HfPiPiCosTheta", "HfPiPiPhiDeg"),
    ]

  # write real-data histograms to ROOT file and generate PDF plots
  outRootFileName = f"{outputDirName}/dataPlots.root"
  outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  outRootFile.cd()
  print(f"Writing histograms to '{outRootFileName}'")
  histNameLogScale = ("hDataPipPMassVsPiPiMassClas", )
  for hist in hists:
    canv = ROOT.TCanvas()
    if hist.GetName() in histNameLogScale:
      histType = hist.IsA().GetName()
      if histType.startswith("TH1"):
        canv.SetLogy()
      elif histType.startswith("TH2"):
        canv.SetLogz()
      else:
        pass # do nothing for other histogram types
    else:
      hist.SetMinimum(0)
    hist.Draw("COLZ")
    hist.Write()
    canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")

  # check against Alex' RF-sideband subtracted histograms
  if False:
    histFileNameAlex = "./plots_tbin1_ebin4.root"
    histNamesAlex = {  # map histogram names
      "hDataPiPiMass"                 : "M",
      "hDataPipPMass"                 : "Deltapp",
      "hDataPimPMass"                 : "Deltaz",
      "hDataPiPiMassVsHfCosThetaAlex" : "MassVsCosth",
      "hDataPiPiMassVsHfPhiDegAlex"   : "MassVsphi",
    }
    histFileAlex = ROOT.TFile.Open(histFileNameAlex, "READ")
    for hist in hists:
      if not hist.GetName() in histNamesAlex:
        continue
      histAlex = histFileAlex.Get(histNamesAlex[hist.GetName()])
      histDiff = hist.Clone(f"{hist.GetName()}_diff")
      histDiff.Add(histAlex, -1)
      canv = ROOT.TCanvas()
      histDiff.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{hist.GetName()}_diff.pdf")
    histFileAlex.Close()

  # overlay pipi mass distributions from data, accepted phase-space MC, and total acceptance-weighted intensity from PWA
  if True:
    lvPip = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # not clear whether correct index is 1 or 2
    lvPim = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # not clear whether correct index is 1 or 2
    dfMc = ROOT.RDataFrame(treeName, mcDataFileName) \
               .Define("MassPiPi", f"massPair({lvPip}, {lvPim})")
    histMassPiPiMc   = dfMc.Histo1D(ROOT.RDF.TH1DModel("Accepted Phase-Space MC", "", 90, 0.2, 2.0), "MassPiPi")
    histMassPiPiData = df.Histo1D  (ROOT.RDF.TH1DModel("RF-subtracted Data",      "", 90, 0.2, 2.0), "MassPiPi", "eventWeight")
    outRootFile.cd()
    canv = ROOT.TCanvas()
    histStack = ROOT.THStack("hPiPiMassDataAndMc", ";m_{#pi#pi} [GeV];Events / 20 MeV")
    histStack.Add(histMassPiPiMc.GetValue())
    histStack.Add(histMassPiPiData.GetValue())
    histMassPiPiMc.SetLineColor    (ROOT.kBlue  + 1)
    histMassPiPiMc.SetMarkerColor  (ROOT.kBlue  + 1)
    histMassPiPiData.SetLineColor  (ROOT.kRed   + 1)
    histMassPiPiData.SetMarkerColor(ROOT.kRed   + 1)
    if False:
      pwaPlotFile = ROOT.TFile.Open("./pwa_plots3.root", "READ")
      histMassPiPiPwa = convertGraphToHist(
        graph    = pwaPlotFile.Get("Total"),
        binning  = (56, 0.28, 1.40),
        histName = "PWA Total Intensity",
      )
      # histMassPiPiPwa.Scale(0.5)
      histStack.Add(histMassPiPiPwa)
      histMassPiPiPwa.SetLineColor   (ROOT.kGreen + 2)
      histMassPiPiPwa.SetMarkerColor (ROOT.kGreen + 2)
    histStack.Draw("NOSTACK")
    canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
    histStack.Write()
    canv.SaveAs(f"{outputDirName}/{histStack.GetName()}.pdf")

outRootFile.Close()
