#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
import os

import ROOT

from makeMomentsInputTree import (
  BeamPolInfo,
  BEAM_POL_INFOS,
  CPP_CODE_MASSPAIR,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_BEAM_POL_PHI,
  CPP_CODE_FLIPYAXIS,
  CPP_CODE_TRACKDISTFDC,
  CoordSysType,
  defineDataFrameColumns,
  getDataFrameWithCorrectEventWeights,
  InputDataType,
  InputDataFormat,
  SubSystemInfo,
  lorentzVectors,
)


def defineColumnsForPlots(
  df:                   ROOT.RDataFrame,
  inputDataFormat:      InputDataFormat,
  subsystem:            SubSystemInfo,
  beamPolInfo:          BeamPolInfo | None,
  additionalColumnDefs: dict[str, str] = {},  # additional columns to define
  additionalFilterDefs: list[str]      = [],  # additional filter conditions to apply
) -> ROOT.RDataFrame:
  """Defines RDataFrame columns for kinematic plots"""
  lvs = lorentzVectors(dataFormat = inputDataFormat)
  dfResult = df
  for frame in (CoordSysType.Hf, CoordSysType.Gj):
    #!NOTE! coordinate system definitions for beam + target -> pi+ + pi- + recoil (all momenta in XRF):
    #    HF for pi+ pi- meson system:  use pi+  as analyzer and z_HF = -p_recoil and y_HF = p_recoil x p_beam
    #    HF for pi+- p  baryon system: use pi+- as analyzer and z_HF = -p_pi-+   and y_HF = p_beam   x p_pi-+
    #    GJ for pi+ pi- meson system:  use pi+  as analyzer and z_GJ = p_beam    and y_HF = p_recoil x p_beam
    #    GJ for pi+- p  baryon system: use pi+- as analyzer and z_GJ = p_target  and y_HF = p_beam   x p_pi-+
    #  particle A is the analyzer, particle B is the other particle in the pair, and the recoil is the third particle in the final state
    dfResult = defineDataFrameColumns(
      df                   = dfResult,
      lvTarget             = lvs["target"],
      lvBeam               = lvs["beam"],  #TODO "beam" for GJ pi+- p baryon system is p_target
      lvRecoil             = lvs[subsystem.lvRecoilLabel],
      lvA                  = lvs[subsystem.lvALabel],
      lvB                  = lvs[subsystem.lvBLabel],
      beamPolInfo          = beamPolInfo,
      frame                = frame,
      flipYAxis            = (frame == CoordSysType.Hf) and subsystem.pairLabel == "PiPi",  # only flip y axis for pi+ pi- system in HF frame
      additionalColumnDefs = additionalColumnDefs,
      additionalFilterDefs = additionalFilterDefs,
      colNameSuffix        = subsystem.pairLabel,
    )
  # define additional columns that are independent of subsystem
  dfResult = (
    dfResult.Define(f"mass{subsystem.pairLabel}Sq", f"std::pow(mass{subsystem.pairLabel}, 2)")
            .Define("Ebeam",                        f"TLorentzVector({lvs['beam']}).E()")
            # track kinematics
            .Define("momLabP",        f"TLorentzVector({lvs['recoil']}).P()")
            .Define("momLabPip",      f"TLorentzVector({lvs['pip'   ]}).P()")
            .Define("momLabPim",      f"TLorentzVector({lvs['pim'   ]}).P()")
            .Define("thetaDegLabP",   f"TLorentzVector({lvs['recoil']}).Theta() * TMath::RadToDeg()")
            .Define("thetaDegLabPip", f"TLorentzVector({lvs['pip'   ]}).Theta() * TMath::RadToDeg()")
            .Define("thetaDegLabPim", f"TLorentzVector({lvs['pim'   ]}).Theta() * TMath::RadToDeg()")
  )
  print(f"!!! {df.GetDefinedColumnNames()=}")
  print(f"!!! {dfResult.GetDefinedColumnNames()=}")
  return dfResult


@dataclass
class HistogramDefinition:
  """Stores information needed to define a histogram"""
  name:       str  # name of the histogram
  title:      str  # title of the histogram
  binning:    tuple[int, float, float] | tuple[tuple[int, float, float], tuple[int, float, float]]
              # 1D binning: (number of bins, minimum value, maximum value)
              # 2D binning: ((number of x bins, minimum x value, maximum x value), (number of y bins, minimum y value, maximum y value))
  columnName: str  # name of RDataFrame column to plot


def bookHistograms(
  df:            ROOT.RDataFrame,
  inputDataType: InputDataType,
  subsystem:     SubSystemInfo,
) -> list[ROOT.TH1D | ROOT.TH2D | ROOT.TH3D]:
  """Books histograms for kinematic plots and returns the list of histograms"""
  applyWeights = inputDataType == InputDataType.realData and df.HasColumn("eventWeight")
  yAxisLabel = "RF-Sideband Subtracted Combos" if applyWeights else "Combos"
  # define 1D histograms
  histDefs: list[HistogramDefinition] = [
    HistogramDefinition("Ebeam",          ";E_{beam} [GeV];"                    + yAxisLabel, (100, 8,   9), "Ebeam"),
    HistogramDefinition("momLabP",        ";p_{p} [GeV];"                       + yAxisLabel, (100, 0,   1), "momLabP"),
    HistogramDefinition("momLabPip",      ";p_{#pi^{#plus}} [GeV];"             + yAxisLabel, (100, 0,  10), "momLabPip"),
    HistogramDefinition("momLabPim",      ";p_{#pi^{#minus}} [GeV];"            + yAxisLabel, (100, 0,  10), "momLabPim"),
    HistogramDefinition("thetaDegLabP",   ";#theta_{p}^{lab} [deg];"            + yAxisLabel, (100, 0, 100), "thetaDegLabP"),
    HistogramDefinition("thetaDegLabPip", ";#theta_{#pi^{#plus}}^{lab} [deg];"  + yAxisLabel, (100, 0,  30), "thetaDegLabPip"),
    HistogramDefinition("thetaDegLabPim", ";#theta_{#pi^{#minus}}^{lab} [deg];" + yAxisLabel, (100, 0,  30), "thetaDegLabPim"),
  ]
  # HistogramDefinition("minusT",   ";#minus t [GeV^{2}];" + yAxisLabel, (100, 0,    1),    "minusTPiPi"),
  # HistogramDefinition("massPiPi", ";m_{#pi#pi} [GeV];"   + yAxisLabel, (400, 0.28, 2.28), "massPiPi"),
  # book histograms
  hists = []
  for histDef in histDefs:
    if applyWeights:
      hists.append(df.Histo1D(ROOT.RDF.TH1DModel(histDef.name, histDef.title, *histDef.binning), histDef.columnName, "eventWeight"))
    else:
      hists.append(df.Histo1D(ROOT.RDF.TH1DModel(histDef.name, histDef.title, *histDef.binning), histDef.columnName))
  return hists


def makePlots(
  hists:         list[ROOT.TH1D | ROOT.TH2D | ROOT.TH3D],  # list of histograms to plot
  outputDirName: str,
) -> None:
  """Writes histograms to ROOT file and generates PDF plots"""
  os.makedirs(outputDirName, exist_ok = True)
  outRootFileName = f"{outputDirName}/plots.root"
  outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  outRootFile.cd()
  print(f"Writing histograms to '{outRootFileName}'")
  for hist in hists:
    print(f"Generating histogram '{hist.GetName()}'")
    canv = ROOT.TCanvas()
    if "TH2" in hist.ClassName() and str(hist.GetName()).startswith("mass"):
      canv.SetLogz(1)
    hist.SetMinimum(0)
    if "TH3" in hist.ClassName():
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
    else:
      hist.Draw("COLZ")
    hist.Write()
    canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")


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
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  dataDirName          = "./polarized"
  dataPeriods          = ("2018_08", )
  tBinLabels           = ("tbin_0.1_0.2", )
  # tBinLabels           = ("tbin_0.1_0.2", "tbin_0.2_0.3", "tbin_0.3_0.4", "tbin_0.4_0.5")
  additionalColumnDefs = {}
  additionalFilterDefs = []
  treeName             = "kin"
  inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in ampTools format
    InputDataType.realData : InputDataFormat.ampTools,
    # InputDataType.mcReco   : InputDataFormat.ampTools,
    # InputDataType.mcTruth  : InputDataFormat.ampTools,
  }
  subsystems: tuple[SubSystemInfo, ...] = (  # particle pairs to analyze; particle A is the analyzer
      SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim",    lvRecoilLabel = "recoil"),
      SubSystemInfo(pairLabel = "PipP", lvALabel = "pip", lvBLabel = "recoil", lvRecoilLabel = "pim"   ),
      # SubSystemInfo(pairLabel = "PimP", lvALabel = "pim", lvBLabel = "recoil", lvRecoilLabel = "pip"   ),
    )

  for dataPeriod in dataPeriods:
    print(f"Generating plots for data period '{dataPeriod}':")
    for tBinLabel in tBinLabels:
      print(f"Generating plots for t bin '{tBinLabel}':")
      inputDataDirName  = f"{dataDirName}/{dataPeriod}/{tBinLabel}/Alex"
      for inputDataType, inputDataFormat in inputDataFormats.items():
        print(f"Generating plots for input data type '{inputDataType}' in format '{inputDataFormat}'")
        for beamOrientation, beamPolInfo in BEAM_POL_INFOS[dataPeriod].items():
          print(f"Generating plots for beam orientation '{beamOrientation}'"
                + (f": pol = {beamPolInfo.pol:.4f}, PhiLab = {beamPolInfo.PhiLab:.1f} deg" if beamPolInfo is not None else ""))
          df = None
          if inputDataType == InputDataType.realData:
            # combine signal and background region data with correct event weights into one RDataFrame
            df = getDataFrameWithCorrectEventWeights(
              dataSigRegionFileNames  = (f"{inputDataDirName}/amptools_tree_signal_{beamOrientation}.root", ),
              dataBkgRegionFileNames  = (f"{inputDataDirName}/amptools_tree_bkgnd_{beamOrientation}.root",  ),
              treeName                = treeName,
              friendSigRegionFileName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/data_sig_{beamOrientation}.root.weights",
              friendBkgRegionFileName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/data_bkg_{beamOrientation}.root.weights",
            )
          elif inputDataType == InputDataType.mcReco:
            df = ROOT.RDataFrame(treeName, f"{inputDataDirName}/amptools_tree_accepted*.root")
          elif inputDataType == InputDataType.mcTruth:
            df = ROOT.RDataFrame(treeName, f"{inputDataDirName}/amptools_tree_thrown*.root")
          else:
            raise RuntimeError(f"Unsupported input data type '{inputDataType}'")
          for subsystem in subsystems:
            print(f"Generating plots for subsystem '{subsystem}':")
            dfSubsystem = defineColumnsForPlots(
              df                   = df,
              inputDataFormat      = inputDataFormat,
              subsystem            = subsystem,
              beamPolInfo          = beamPolInfo,
              additionalColumnDefs = additionalColumnDefs,
              additionalFilterDefs = additionalFilterDefs,
            )
            makePlots(
              hists = bookHistograms(
                df            = dfSubsystem,
                inputDataType = inputDataType,
                subsystem     = subsystem,
              ),
              outputDirName = f"{dataDirName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}/plots_{inputDataType.name}/{beamOrientation}",
            )

  # # create RDataFrame from real data in AmpTools format and define columns
  # lvs = lorentzVectors(realData = True)
  # print(f"Reading data from tree '{treeName}' in signal file(s) {dataSigRegionFileNames} and background file(s) '{dataBkgRegionFileNames}'")
  # df = (
  #   getDataFrameWithCorrectEventWeights(
  #     dataSigRegionFileNames  = dataSigRegionFileNames,
  #     dataBkgRegionFileNames  = dataBkgRegionFileNames,
  #     treeName                = treeName,
  #     friendSigRegionFileName = "data_sig.plot.root.weights",
  #     friendBkgRegionFileName = "data_bkg.plot.root.weights",
  #   ).Define("DistFdcPip",         f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvs['lvPip']})")
  #    .Define("DistFdcPim",         f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvs['lvPim']})")
  #   #  .Filter("(DistFdcPip > 4) and (DistFdcPim > 4)")  # require minimum distance of tracks at FDC position [cm]
  # )

  # # define real-data histograms applying RF-sideband subtraction
  # yAxisLabel = "RF-Sideband Subtracted Combos"
  # hists = [
  #   df.Histo1D(ROOT.RDF.TH1DModel("hDataEbeam",              ";E_{beam} [GeV];"                     + yAxisLabel, 100, 8,      9),    "E_Beam",      "eventWeight"),
  #   df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPi",           ";m_{#pi#pi} [GeV];"                   + yAxisLabel, 400, 0.28,   2.28), "MassPiPi",    "eventWeight"),
  #   df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPiPiPwa" ,       ";m_{#pi#pi} [GeV];"                   + yAxisLabel,  50, 0.28,   2.28), "MassPiPi",    "eventWeight"),
  #   df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPipP",           ";m_{p#pi^{#plus}} [GeV];"             + yAxisLabel, 400, 1,      5),    "MassPipP",    "eventWeight"),
  #   df.Histo1D(ROOT.RDF.TH1DModel("hDataMassPimP",           ";m_{p#pi^{#minus}} [GeV];"            + yAxisLabel, 400, 1,      5),    "MassPimP",    "eventWeight"),
  #   df.Histo1D(ROOT.RDF.TH1DModel("hDataMinusT",             ";#minus t [GeV^{2}];"                 + yAxisLabel, 100, 0,      1),    "minusT",      "eventWeight"),
  #   #
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesGjPiPi",             ";cos#theta_{GJ};#phi_{GJ} [deg]",     100, -1,   +1,     72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi",   "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataAnglesHfPiPi",             ";cos#theta_{HF};#phi_{HF} [deg]",     100, -1,   +1,     72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi",   "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsGjCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{GJ}",     50,  0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "GjCosThetaPiPi", "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsGjPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{GJ} [deg]",    50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "GjPhiDegPiPi",   "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsHfCosThetaPiPi", ";m_{#pi#pi} [GeV];cos#theta_{HF}",     50,  0.28, 2.28, 100,   -1,   +1), "MassPiPi",       "HfCosThetaPiPi", "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsHfPhiDegPiPi",   ";m_{#pi#pi} [GeV];#phi_{HF} [deg]",    50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "HfPhiDegPiPi",   "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsPhiDeg",         ";m_{#pi#pi} [GeV];#Phi [deg]",         50,  0.28, 2.28,  72, -180, +180), "MassPiPi",       "PhiDeg",         "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataMassPiPiVsMinusT",         ";m_{#pi#pi} [GeV];#minus t [GeV^{2}]", 50,  0.28, 2.28,  50,    0,    1), "MassPiPi",       "minusT",         "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz1",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#plus}}^{2} [GeV^{2}]",   100, 0,  6, 100, 0.5, 16.5), "MassPiPiSq", "MassPipPSq",  "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataDalitz2",                  ";m_{#pi#pi}^{2} [GeV^{2}];m_{p#pi^{#minus}}^{2} [GeV^{2}]",  100, 0,  6, 100, 0.5, 16.5), "MassPiPiSq", "MassPimPSq",  "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDatathetaDegLabVsMomLabPip",      ";p_{#pi^{#plus}} [GeV];#theta_{#pi^{#plus}}^{lab} [deg]",    100, 0, 10, 100, 0,   15),   "MomLabPip",  "thetaDegLabPip", "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDatathetaDegLabVsMomLabPim",      ";p_{#pi^{#minus}} [GeV];#theta_{#pi^{#minus}}^{lab} [deg]",  100, 0, 10, 100, 0,   15),   "MomLabPim",  "thetaDegLabPim", "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataDistFdcVsMomLabPip",       ";p_{#pi^{#plus}} [GeV];#Delta r_{#pi^{#plus}}^{FDC} [cm]",   100, 0, 10, 100, 0,   20),   "MomLabPip",  "DistFdcPip",  "eventWeight"),
  #   df.Histo2D(ROOT.RDF.TH2DModel("hDataDistFdcVsMomLabPim",       ";p_{#pi^{#minus}} [GeV];#Delta r_{#pi^{#minus}}^{FDC} [cm]", 100, 0, 10, 100, 0,   20),   "MomLabPim",  "DistFdcPim",  "eventWeight"),
  #   df.Histo3D(ROOT.RDF.TH3DModel("hDataPhiDegVsHfPhiDegPiPiVsHfCosThetaPiPi", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 25, -1, +1, 25, -180, +180, 25, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi", "PhiDeg", "eventWeight"),
  # ]
  # # create histograms for GJ and HF angles in m_pipi bins
  # massPiPiRange = (0.28, 2.28)  # [GeV]
  # massPiPiNmbBins = 50
  # massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
  # for binIndex in range(0, massPiPiNmbBins):
  #   massPiPiBinMin    = massPiPiRange[0] + binIndex * massPiPiBinWidth
  #   massPiPiBinMax    = massPiPiBinMin + massPiPiBinWidth
  #   massPiPiBinFilter = f"({massPiPiBinMin} < MassPiPi) and (MassPiPi < {massPiPiBinMax})"
  #   histNameSuffix    = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
  #   hists += [
  #     df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hDataAnglesGjPiPi{histNameSuffix}", ";cos#theta_{GJ};#phi_{GJ} [deg]", 100, -1, +1, 72, -180, +180), "GjCosThetaPiPi", "GjPhiDegPiPi"),
  #     df.Filter(massPiPiBinFilter).Histo2D(ROOT.RDF.TH2DModel(f"hDataAnglesHfPiPi{histNameSuffix}", ";cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "HfCosThetaPiPi", "HfPhiDegPiPi"),
  #   ]

  # # write real-data histograms to ROOT file and generate PDF plots
  # os.makedirs(outputDirName, exist_ok = True)
  # outRootFileName = f"{outputDirName}/dataPlots.root"
  # outRootFile = ROOT.TFile(outRootFileName, "RECREATE")
  # outRootFile.cd()
  # print(f"Writing histograms to '{outRootFileName}'")
  # for hist in hists:
  #   print(f"Generating histogram '{hist.GetName()}'")
  #   canv = ROOT.TCanvas()
  #   if "TH2" in hist.ClassName() and str(hist.GetName()).startswith("hDataMass"):
  #     canv.SetLogz(1)
  #   hist.SetMinimum(0)
  #   if "TH3" in hist.ClassName():
  #     hist.GetXaxis().SetTitleOffset(1.5)
  #     hist.GetYaxis().SetTitleOffset(2)
  #     hist.GetZaxis().SetTitleOffset(1.5)
  #     hist.Draw("BOX2Z")
  #   else:
  #     hist.Draw("COLZ")
  #   hist.Write()
  #   canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")

  # if True:
  #   # overlay pipi mass distributions from data and accepted phase-space MC
  #   lvPip = "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]"  # not clear whether correct index is 1 or 2
  #   lvPim = "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]"  # not clear whether correct index is 1 or 2
  #   dfMc = (
  #     ROOT.RDataFrame(treeName, mcDataFileNames)
  #         .Define("MassPiPi", f"massPair({lvPip}, {lvPim})")
  #   )
  #   histMassPiPiMc   = dfMc.Histo1D(ROOT.RDF.TH1DModel("Accepted Phase-Space MC", "", 50, 0.28, 2.28), "MassPiPi")
  #   histMassPiPiData = df.Histo1D  (ROOT.RDF.TH1DModel("RF-subtracted Data",      "", 50, 0.28, 2.28), "MassPiPi", "eventWeight")
  #   canv = ROOT.TCanvas()
  #   histStack = ROOT.THStack("hMassPiPiDataAndMc", ";m_{#pi#pi} [GeV];Events / 40 MeV")
  #   histStack.Add(histMassPiPiMc.GetValue())
  #   histStack.Add(histMassPiPiData.GetValue())
  #   histMassPiPiMc.SetLineColor    (ROOT.kBlue + 1)
  #   histMassPiPiMc.SetMarkerColor  (ROOT.kBlue + 1)
  #   histMassPiPiData.SetLineColor  (ROOT.kRed  + 1)
  #   histMassPiPiData.SetMarkerColor(ROOT.kRed  + 1)
  #   histStack.Draw("NOSTACK")
  #   canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
  #   histStack.Write()
  #   canv.SaveAs(f"{outputDirName}/{histStack.GetName()}.pdf")

  # outRootFile.Close()
