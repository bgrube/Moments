#!/usr/bin/env python3


from __future__ import annotations

import os

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_MAKEPAIR,
  defineAngleFormulas,
  lorentzVectors,
)


# declare C++ function to calculate invariant mass of a particle
CPP_CODE_MASS = """
double
mass(const double Px, const double Py, const double Pz, const double E)
{
  const TLorentzVector p(Px, Py, Pz, E);
  return p.M();
}
"""


# declare C++ function to calculate momentum transfer squared
CPP_CODE_MOM_TRANSFER_SQ = """
double
momTransferSq(
  const double PxA, const double PyA, const double PzA, const double EA,
  const double PxB, const double PyB, const double PzB, const double EB
)	{
  const TLorentzVector pA(PxA, PyA, PzA, EA);
  const TLorentzVector pB(PxB, PyB, PzB, EB);
  return (pA - pB).M2();
}
"""


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C")
  ROOT.gROOT.LoadMacro("../rootlogon.C")
  ROOT.gStyle.SetOptStat("i")
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MAKEPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MASS)
  ROOT.gInterpreter.Declare(CPP_CODE_MOM_TRANSFER_SQ)

  tBinLabel      = "tbin_0.4_0.5"
  # mcDataFileName = "./amptools_tree_thrown_tbin1_ebin4_rho.root"
  # mcDataFileName = "./amptools_tree_accepted_tbin1_ebin4_rho.root"
  # mcDataFileName = "./amptools_tree_thrown_tbin1_ebin4*.root"
  mcDataFileName = "./amptools_tree_accepted_tbin1_ebin4*.root"
  treeName       = "kin"
  outputDirName  = f"{tBinLabel}/McPlots"

  # create RDataFrame from MC data in AmpTools format and define columns
  # see `plotDataTree.py` for definition of coordinate systems
  lvBeamPhoton, lvTargetProton, lvRecoilProton, lvPip, lvPim = lorentzVectors(realData = False)
  df = ROOT.RDataFrame(treeName, mcDataFileName)
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
  df = (
    df.Define("FsMassRecoil", f"mass({lvRecoilProton})")
      .Define("FsMassPip",    f"mass({lvPip})")
      .Define("FsMassPim",    f"mass({lvPim})")
      .Define("tAbs",         f"std::fabs(momTransferSq({lvTargetProton}, {lvRecoilProton}))")
  )

  # define MC histograms
  yAxisLabel = "Events"
  hists = [
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassRecoil",  ";m_{Recoil} [GeV];"        + yAxisLabel, 100, 0,    2),    "FsMassRecoil"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPip",     ";m_{#pi^{#plus}} [GeV];"   + yAxisLabel, 100, 0,    2),    "FsMassPip"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcFsMassPim",     ";m_{#pi^{#minus}} [GeV];"  + yAxisLabel, 100, 0,    2),    "FsMassPim"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcPiPiMassAlex",  ";m_{#pi#pi} [GeV];"        + yAxisLabel, 400, 0.28, 2.28), "MassPiPi"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcPipPMassAlex",  ";m_{p#pi^{#plus}} [GeV];"  + yAxisLabel, 400, 1,    5),    "MassPipP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcPimPMassAlex",  ";m_{p#pi^{#minus}} [GeV];" + yAxisLabel, 400, 1,    5),    "MassPimP"),
    df.Histo1D(ROOT.RDF.TH1DModel("hMcMomTransferSq", ";|t| [GeV^{2}];"           + yAxisLabel,  50, 0,    1),    "tAbs"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcDalitz1", ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#plus}}^{2} [GeV^{2}]",  100, 0, 3.5, 100, 1,  7.5), "MassPiPiSq", "MassPipPSq"),
    df.Histo2D(ROOT.RDF.TH2DModel("hMcDalitz2", ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#minus}}^{2} [GeV^{2}]", 100, 0, 3.5, 100, 1,  7.5), "MassPiPiSq", "MassPimPSq"),
  ]
  # add histograms specific to subsystems
  for pairLabel, massAxisTitle, massBinning in (
    ("PiPi", "m_{#pi#pi} [GeV]",        (56, 0.28, 1.40)),
    ("PipP", "m_{p#pi^{#plus}} [GeV]",  (72, 1,    2.8 )),
    ("PimP", "m_{p#pi^{#minus}} [GeV]", (72, 1,    2.8 )),
  ):
    hists += [
      df.Histo1D(ROOT.RDF.TH1DModel(f"hMc{pairLabel}Mass", f";{massAxisTitle};" + yAxisLabel, *massBinning), f"Mass{pairLabel}"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hMc{pairLabel}AnglesGj",         ";cos#theta_{GJ};#phi_{GJ} [deg]",       50, -1,   +1, 50, -180, +180), f"Gj{pairLabel}CosTheta", f"Gj{pairLabel}PhiDeg"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hMc{pairLabel}AnglesHf",         ";cos#theta_{HF};#phi_{HF} [deg]",       50, -1,   +1, 50, -180, +180), f"Hf{pairLabel}CosTheta", f"Hf{pairLabel}PhiDeg"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hMc{pairLabel}MassVsGjCosTheta", f";{massAxisTitle}" + ";cos#theta_{GJ}", *massBinning, 72,   -1,   +1), f"Mass{pairLabel}",       f"Gj{pairLabel}CosTheta"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hMc{pairLabel}MassVsGjPhiDeg",   f";{massAxisTitle}" + ";#phi_{GJ}",      *massBinning, 72, -180, +180), f"Mass{pairLabel}",       f"Gj{pairLabel}PhiDeg"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hMc{pairLabel}MassVsHfCosTheta", f";{massAxisTitle}" + ";cos#theta_{HF}", *massBinning, 72,   -1,   +1), f"Mass{pairLabel}",       f"Hf{pairLabel}CosTheta"),
      df.Histo2D(ROOT.RDF.TH2DModel(f"hMc{pairLabel}MassVsHfPhiDeg",   f";{massAxisTitle}" + ";#phi_{HF}",      *massBinning, 72, -180, +180), f"Mass{pairLabel}",       f"Hf{pairLabel}PhiDeg"),
    ]
  hists += [
    df.Histo2D(ROOT.RDF.TH2DModel(f"hMcPiPiMassVsGjCosThetaPipP", ";m_{#pi#pi} [GeV];cos#theta_{GJ}", 56, 0.28, 1.40, 72, -1, +1), f"MassPiPi", f"GjPipPCosTheta"),
    df.Histo2D(ROOT.RDF.TH2DModel(f"hMcPiPiMassVsGjCosThetaPimP", ";m_{#pi#pi} [GeV];cos#theta_{GJ}", 56, 0.28, 1.40, 72, -1, +1), f"MassPiPi", f"GjPimPCosTheta"),
    df.Filter("tAbs < 0.45").Histo2D(ROOT.RDF.TH2DModel(f"hMcPipPMassVsGjCosThetaCutT", ";m_{p#pi^{#plus}} [GeV];cos#theta_{GJ}", 72, 1, 2.8, 72, -1, +1), f"MassPipP", f"GjPipPCosTheta"),
  ]
  # create acceptance histograms for pi pi GJ and HF angles in bins of m_pipi
  massPiPiRange    = (0.28, 1.40)  # [GeV] binning used in PWA
  massPiPiNmbBins  = 28
  massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
  for binIndex in range(0, massPiPiNmbBins):
    massPiPiBinMin    = massPiPiRange[0] + binIndex * massPiPiBinWidth
    massPiPiBinMax    = massPiPiBinMin + massPiPiBinWidth
    massPiPiBinFilter = f"({massPiPiBinMin} < MassPiPi) and (MassPiPi < {massPiPiBinMax})"
    histNameSuffix    = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
    hists += [
      df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hMcPiPiAnglesGj{histNameSuffix}", ";cos#theta_{GJ};#phi_{GJ} [deg]", 50, -1, +1, 50, -180, +180), "GjPiPiCosTheta", "GjPiPiPhiDeg"),
      df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hMcPiPiAnglesHf{histNameSuffix}", ";cos#theta_{HF};#phi_{HF} [deg]", 50, -1, +1, 50, -180, +180), "HfPiPiCosTheta", "HfPiPiPhiDeg"),
    ]
  # # check that for accepted MC the columns used for real data contain the same info as the ones used for MC data
  # # also check that index 1 is pi+ and 2 pi-
  # df = (
  #   # df.Define("Delta_E",  "beam_p4_kin.Energy() -  E_Beam")
  #   #   .Define("Delta_Px", "beam_p4_kin.Px()     - Px_Beam")
  #   #   .Define("Delta_Py", "beam_p4_kin.Py()     - Py_Beam")
  #   #   .Define("Delta_Pz", "beam_p4_kin.Pz()     - Pz_Beam")
  #   # df.Define("Delta_E",  "p_p4_kin.Energy() -  E_FinalState[0]")
  #   #   .Define("Delta_Px", "p_p4_kin.Px()     - Px_FinalState[0]")
  #   #   .Define("Delta_Py", "p_p4_kin.Py()     - Py_FinalState[0]")
  #   #   .Define("Delta_Pz", "p_p4_kin.Pz()     - Pz_FinalState[0]")
  #   # df.Define("Delta_E",  "pip_p4_kin.Energy() -  E_FinalState[1]")  # confirms index assignment of 1 for pi+
  #   #   .Define("Delta_Px", "pip_p4_kin.Px()     - Px_FinalState[1]")
  #   #   .Define("Delta_Py", "pip_p4_kin.Py()     - Py_FinalState[1]")
  #   #   .Define("Delta_Pz", "pip_p4_kin.Pz()     - Pz_FinalState[1]")
  #   df.Define("Delta_E",  "pim_p4_kin.Energy() -  E_FinalState[2]")  # confirms index assignment of 2 for pi-
  #     .Define("Delta_Px", "pim_p4_kin.Px()     - Px_FinalState[2]")
  #     .Define("Delta_Py", "pim_p4_kin.Py()     - Py_FinalState[2]")
  #     .Define("Delta_Pz", "pim_p4_kin.Pz()     - Pz_FinalState[2]")
  # )
  # hists += [
  #   df.Histo1D(ROOT.RDF.TH1DModel("hMcDelta_E",  ";#Delta E [GeV];"     + yAxisLabel, 100, -1e-6, +1e-6), "Delta_E"),
  #   df.Histo1D(ROOT.RDF.TH1DModel("hMcDelta_Px", ";#Delta p_{x} [GeV];" + yAxisLabel, 100, -1e-6, +1e-6), "Delta_Px"),
  #   df.Histo1D(ROOT.RDF.TH1DModel("hMcDelta_Py", ";#Delta p_{y} [GeV];" + yAxisLabel, 100, -1e-6, +1e-6), "Delta_Py"),
  #   df.Histo1D(ROOT.RDF.TH1DModel("hMcDelta_Pz", ";#Delta p_{z} [GeV];" + yAxisLabel, 100, -1e-6, +1e-6), "Delta_Pz"),
  # ]

  # write MC histograms to ROOT file and generate PDF plots
  os.makedirs(outputDirName, exist_ok = True)
  outRootFileName = f"{outputDirName}/mcPlots.root"
  outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  outRootFile.cd()
  print(f"Writing histograms to '{outRootFileName}'")
  for hist in hists:
    print(f"Plotting histogram '{hist.GetName()}'")
    canv = ROOT.TCanvas()
    hist.SetMinimum(0)
    hist.Draw("COLZ")
    hist.Write()
    canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")

outRootFile.Close()
