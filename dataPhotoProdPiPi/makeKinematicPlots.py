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
  name:        str  # name of the histogram
  title:       str  # title of the histogram
  binning:     tuple[tuple[int, float, float], ] | tuple[tuple[int, float, float], tuple[int, float, float]] | tuple[tuple[int, float, float], tuple[int, float, float], tuple[int, float, float]]
               # 1D binning: ((number of x bins, minimum x value, maximum x value), )
               # 2D binning: ((number of x bins, minimum x value, maximum x value), (number of y bins, minimum y value, maximum y value))
               # 3D binning: ((number of x bins, minimum x value, maximum x value), (number of y bins, minimum y value, maximum y value), (number of z bins, minimum z value, maximum z value))
  columnNames: tuple[str, ] | tuple[str, str] | tuple[str, str, str]  # name(s) of RDataFrame column(s) to plot
  filter:      str = ""  # optional filter condition to apply to the histogram


def bookHistogram(
  df:           ROOT.RDataFrame,
  histDef:      HistogramDefinition,
  applyWeights: bool,
) -> ROOT.TH1D | ROOT.TH2D | ROOT.TH3D:
  """Books a single histogram according to the given definition and returns it"""
  # apply optional filter
  dfHist = df.Filter(histDef.filter) if histDef.filter else df
  # get functions to book histogram of correct dimension
  histDimension = len(histDef.binning)
  if histDimension not in (1, 2, 3):
    raise NotImplementedError(f"Booking of {histDimension}D histograms is not implemented")
  THNDModelFunc = (
    ROOT.RDF.TH1DModel if histDimension == 1 else
    ROOT.RDF.TH2DModel if histDimension == 2 else
    ROOT.RDF.TH3DModel
  )
  dfHistoNDFunc = (
    dfHist.Histo1D if histDimension == 1 else
    dfHist.Histo2D if histDimension == 2 else
    dfHist.Histo3D
  )
  # flatten binning into single tuple
  binning = tuple(entry for binningTuple in histDef.binning for entry in binningTuple)
  # book histogram with or without event weights
  if applyWeights:
    return dfHistoNDFunc(THNDModelFunc(histDef.name, histDef.title, *binning), *histDef.columnNames, "eventWeight")
  else:
    return dfHistoNDFunc(THNDModelFunc(histDef.name, histDef.title, *binning), *histDef.columnNames)


def bookHistograms(
  df:            ROOT.RDataFrame,
  inputDataType: InputDataType,
  subsystem:     SubSystemInfo,
) -> list[ROOT.TH1D | ROOT.TH2D | ROOT.TH3D]:
  """Books histograms for kinematic plots and returns the list of histograms"""
  applyWeights = inputDataType == InputDataType.realData and df.HasColumn("eventWeight")
  yAxisLabel = "RF-Sideband Subtracted Combos" if applyWeights else "Combos"
  # define histograms that are independent of subsystem
  histDefs: list[HistogramDefinition] = [
    HistogramDefinition("Ebeam",          ";E_{beam} [GeV];"                    + yAxisLabel, ((100, 8,   9), ), ("Ebeam",          )),
    HistogramDefinition("momLabP",        ";p_{p} [GeV];"                       + yAxisLabel, ((100, 0,   1), ), ("momLabP",        )),
    HistogramDefinition("momLabPip",      ";p_{#pi^{#plus}} [GeV];"             + yAxisLabel, ((100, 0,  10), ), ("momLabPip",      )),
    HistogramDefinition("momLabPim",      ";p_{#pi^{#minus}} [GeV];"            + yAxisLabel, ((100, 0,  10), ), ("momLabPim",      )),
    HistogramDefinition("thetaDegLabP",   ";#theta_{p}^{lab} [deg];"            + yAxisLabel, ((100, 0, 100), ), ("thetaDegLabP",   )),
    HistogramDefinition("thetaDegLabPip", ";#theta_{#pi^{#plus}}^{lab} [deg];"  + yAxisLabel, ((100, 0,  30), ), ("thetaDegLabPip", )),
    HistogramDefinition("thetaDegLabPim", ";#theta_{#pi^{#minus}}^{lab} [deg];" + yAxisLabel, ((100, 0,  30), ), ("thetaDegLabPim", )),
    HistogramDefinition("thetaDegLabVsMomLabP",   ";p_{p} [GeV];#theta_{p}^{lab} [deg]",                       ((100, 0,  1), (100, 65, 80)), ("momLabP",   "thetaDegLabP"  )),
    HistogramDefinition("thetaDegLabVsMomLabPip", ";p_{#pi^{#plus}} [GeV];#theta_{#pi^{#plus}}^{lab} [deg]",   ((100, 0, 10), (100,  0, 30)), ("momLabPip", "thetaDegLabPip")),
    HistogramDefinition("thetaDegLabVsMomLabPim", ";p_{#pi^{#minus}} [GeV];#theta_{#pi^{#minus}}^{lab} [deg]", ((100, 0, 10), (100,  0, 30)), ("momLabPim", "thetaDegLabPim")),
  ]
  # define subsystem-dependent histograms
  pairLabel = subsystem.pairLabel
  pairTLatexLabel = subsystem.pairTLatexLabel
  histDefs += [
    HistogramDefinition(f"anglesGj{pairLabel}",                    f"{pairTLatexLabel};cos#theta_{{GJ}};#phi_{{GJ}} [deg]",                      ((100, -1,   +1  ), ( 72, -180, +180)), (f"cosThetaGj{pairLabel}", f"phiDegGj{pairLabel}")),
    HistogramDefinition(f"anglesHf{pairLabel}",                    f"{pairTLatexLabel};cos#theta_{{HF}};#phi_{{HF}} [deg]",                      ((100, -1,   +1  ), ( 72, -180, +180)), (f"cosThetaHf{pairLabel}", f"phiDegHf{pairLabel}")),
    HistogramDefinition(f"PhiDeg{pairLabel}VsPhiDegGj{pairLabel}VsCosThetaGj{pairLabel}", f"{pairTLatexLabel};cos#theta_{{Gj}};#phi_{{Gj}} [deg];#Phi [deg]", ((25, -1, +1), (25, -180, +180), (25, -180, +180)), (f"cosThetaGj{pairLabel}", f"phiDegGj{pairLabel}", f"PhiDeg{pairLabel}")),
    HistogramDefinition(f"PhiDeg{pairLabel}VsPhiDegHf{pairLabel}VsCosThetaHf{pairLabel}", f"{pairTLatexLabel};cos#theta_{{HF}};#phi_{{HF}} [deg];#Phi [deg]", ((25, -1, +1), (25, -180, +180), (25, -180, +180)), (f"cosThetaHf{pairLabel}", f"phiDegHf{pairLabel}", f"PhiDeg{pairLabel}")),
  ]
  if pairLabel == "PiPi":
    histDefs += [
      HistogramDefinition(f"mass{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];"              + yAxisLabel, ((400, 0.28, 2.28), ), (f"mass{pairLabel}",   )),
      HistogramDefinition(f"minusT{pairLabel}", f";#minus t_{{{pairTLatexLabel}}} [GeV^{{2}}];" + yAxisLabel, ((100, 0,    1),    ), (f"minusT{pairLabel}", )),
      HistogramDefinition(f"mass{pairLabel}VsCosThetaGj{pairLabel}", f";m_{{{pairTLatexLabel}}} [GeV];cos#theta_{{GJ}}",                           (( 50, 0.28, 2.28), (100,   -1,   +1)), (f"mass{pairLabel}",       f"cosThetaGj{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsPhiDegGj{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];#phi_{{GJ}}",                                (( 50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}",       f"phiDegGj{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsCosThetaHf{pairLabel}", f";m_{{{pairTLatexLabel}}} [GeV];cos#theta_{{HF}}",                           (( 50, 0.28, 2.28), (100,   -1,   +1)), (f"mass{pairLabel}",       f"cosThetaHf{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsPhiDegHf{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];#phi_{{HF}}",                                (( 50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}",       f"phiDegHf{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsPhiDeg",                f";m_{{{pairTLatexLabel}}} [GeV];#Phi",                                       (( 50, 0.28, 2.28), ( 72, -180, +180)), (f"mass{pairLabel}",       f"PhiDeg{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsMinusT{pairLabel}",     f";m_{{{pairTLatexLabel}}} [GeV];#minus t_{{{pairTLatexLabel}}} [GeV^{{2}}]", (( 50, 0.28, 2.28), ( 50,    0,    1)), (f"mass{pairLabel}",       f"minusT{pairLabel}")),
    ]
    # create histograms for GJ and HF angles in m_pipi bins
    massPiPiRange = (0.28, 2.28)  # [GeV]
    massPiPiNmbBins = 50
    massPiPiBinWidth = (massPiPiRange[1] - massPiPiRange[0]) / massPiPiNmbBins
    for binIndex in range(0, massPiPiNmbBins):
      massPiPiBinMin    = massPiPiRange[0] + binIndex * massPiPiBinWidth
      massPiPiBinMax    = massPiPiBinMin + massPiPiBinWidth
      massPiPiBinFilter = f"({massPiPiBinMin} < massPiPi) and (massPiPi < {massPiPiBinMax})"
      histNameSuffix    = f"_{massPiPiBinMin:.2f}_{massPiPiBinMax:.2f}"
      histDefs += [
        HistogramDefinition(f"anglesGj{pairLabel}{histNameSuffix}", f"{pairTLatexLabel};cos#theta_{{GJ}};#phi_{{GJ}} [deg]", ((100, -1, +1), (72, -180, +180)), (f"cosThetaGj{pairLabel}", f"phiDegGj{pairLabel}"), massPiPiBinFilter),
        HistogramDefinition(f"anglesHf{pairLabel}{histNameSuffix}", f"{pairTLatexLabel};cos#theta_{{HF}};#phi_{{HF}} [deg]", ((100, -1, +1), (72, -180, +180)), (f"cosThetaHf{pairLabel}", f"phiDegHf{pairLabel}"), massPiPiBinFilter),
      ]
  else:
    histDefs += [
      HistogramDefinition(f"mass{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];"              + yAxisLabel, ((400, 1, 5 ), ), (f"mass{pairLabel}",   )),
      HistogramDefinition(f"minusT{pairLabel}", f";#minus t_{{{pairTLatexLabel}}} [GeV^{{2}}];" + yAxisLabel, ((100, 0, 15), ), (f"minusT{pairLabel}", )),
      HistogramDefinition(f"mass{pairLabel}VsCosThetaGj{pairLabel}", f";m_{{{pairTLatexLabel}}} [GeV];cos#theta_{{GJ}}",                           (( 50, 1, 5), (100,   -1,   +1)), (f"mass{pairLabel}",       f"cosThetaGj{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsPhiDegGj{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];#phi_{{GJ}}",                                (( 50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}",       f"phiDegGj{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsCosThetaHf{pairLabel}", f";m_{{{pairTLatexLabel}}} [GeV];cos#theta_{{HF}}",                           (( 50, 1, 5), (100,   -1,   +1)), (f"mass{pairLabel}",       f"cosThetaHf{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsPhiDegHf{pairLabel}",   f";m_{{{pairTLatexLabel}}} [GeV];#phi_{{HF}}",                                (( 50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}",       f"phiDegHf{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsPhiDeg",                f";m_{{{pairTLatexLabel}}} [GeV];#Phi",                                       (( 50, 1, 5), ( 72, -180, +180)), (f"mass{pairLabel}",       f"PhiDeg{pairLabel}")),
      HistogramDefinition(f"mass{pairLabel}VsMinusT{pairLabel}",     f";m_{{{pairTLatexLabel}}} [GeV];#minus t_{{{pairTLatexLabel}}} [GeV^{{2}}]", (( 50, 1, 5), ( 50,    0,    1)), (f"mass{pairLabel}",       f"minusT{pairLabel}")),
    ]
  # book histograms
  hists = []
  for histDef in histDefs:
    hists.append(bookHistogram(df, histDef, applyWeights))
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
    ROOT.gStyle.SetOptStat("i")
    # ROOT.gStyle.SetOptStat(1111111)
    ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
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
  outRootFile.Close()


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  dataDirName          = "./polarized"
  dataPeriods          = ("2018_08", )
  tBinLabels           = ("tbin_0.1_0.2", "tbin_0.2_0.3", "tbin_0.3_0.4", "tbin_0.4_0.5")
  additionalColumnDefs = {}
  additionalFilterDefs = []
  treeName             = "kin"
  inputDataFormats: dict[InputDataType, InputDataFormat] = {  # all files in ampTools format
    InputDataType.realData : InputDataFormat.ampTools,
    InputDataType.mcReco   : InputDataFormat.ampTools,
    InputDataType.mcTruth  : InputDataFormat.ampTools,
  }
  subsystems: tuple[SubSystemInfo, ...] = (  # particle pairs to analyze; particle A is the analyzer
      SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim",    lvRecoilLabel = "recoil", pairTLatexLabel = "#pi#pi"),
      # SubSystemInfo(pairLabel = "PipP", lvALabel = "pip", lvBLabel = "recoil", lvRecoilLabel = "pim",    pairTLatexLabel = "p#pi^{#minus}"),
      # SubSystemInfo(pairLabel = "PimP", lvALabel = "pim", lvBLabel = "recoil", lvRecoilLabel = "pip",    pairTLatexLabel = "p#pi^{#plus}"),
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
