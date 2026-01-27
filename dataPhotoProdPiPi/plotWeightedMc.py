#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
  fields,
)
import functools
import os

import ROOT

from makeKinematicPlots import (
  bookHistogram,
  defineColumnsForPlots,
  HistListType,
  HistogramDefinition,
)
from makeMomentsInputTree import (
  BEAM_POL_INFOS,
  CPP_CODE_ANGLES_GLUEX_AMPTOOLS,
  CPP_CODE_BEAM_POL_PHI,
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_FLIPYAXIS,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  getDataFrameWithCorrectEventWeights,
  InputDataFormat,
  SubSystemInfo,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class DataToOverlay:
  """Stores data frames for data and weighted MC"""
  realData:   ROOT.RDataFrame
  weightedMc: ROOT.RDataFrame

  def Filter(
    self,
    filterExpr: str
  ) -> DataToOverlay:
    """Returns a new `DataToOverlay` instance with data frames filtered by `filterExpr`"""
    return DataToOverlay(
      realData   = self.realData.Filter  (filterExpr),
      weightedMc = self.weightedMc.Filter(filterExpr),
    )


@dataclass
class HistsToOverlay:
  """Stores histograms for data and weighted MC"""
  # tuples are assumed to contain histograms in identical order
  realData:   HistListType = field(default_factory = list)
  weightedMc: HistListType = field(default_factory = list)


def makePlots(
  dataToOverlay:     DataToOverlay,
  outputDirName:     str = ".",
  pairLabel:         str = "PiPi",
  pdfFileNameSuffix: str = "",
  yAxisLabel:        str = "RF-Sideband Subtracted Combos",
  colNameSuffix:     str = "",  # suffix appended to column names
) -> None:
  """Overlays 1D distributions and compares 2D distributions from real data and weighted Monte Carlo"""
  histsToOverlay = HistsToOverlay()
  # loop over members of `DataToOverlay` and book histograms for `realData` and `weightedMc`
  for dataToOverlayField in fields(dataToOverlay):
    # define histograms
    label = dataToOverlayField.name
    label = label[0].upper() + label[1:]  # make sure first character is upper case
    histNameSuffix = f"_{label}"  # i.e. "_RealData" or "_WeightedMc"
    histDefs: list[HistogramDefinition] = []
    if True:
    # if False:
      # distributions in X rest frame
      histDefs += [
        # 1D histograms
        HistogramDefinition(f"mass{histNameSuffix}",       ";m_{#pi#pi} [GeV];"                   + yAxisLabel, (( 50,    0.28,    2.28), ), (f"mass{colNameSuffix}",       )),
        HistogramDefinition(f"minusT{histNameSuffix}",     ";#minus t_{#pi#pi} [GeV^{2}];"        + yAxisLabel, ((100,    0,       1   ), ), (f"minusT{colNameSuffix}",     )),
        HistogramDefinition(f"cosThetaHF{histNameSuffix}", ";cos#theta_{HF};"                     + yAxisLabel, ((100,   -1,      +1   ), ), (f"cosThetaHF{colNameSuffix}", )),
        HistogramDefinition(f"phiDegHF{histNameSuffix}",   ";#phi_{HF} [deg];"                    + yAxisLabel, (( 72, -180,    +180   ), ), (f"phiDegHF{colNameSuffix}",   )),
        HistogramDefinition(f"PhiDeg{histNameSuffix}",     ";#Phi [deg];"                         + yAxisLabel, (( 72, -180,    +180   ), ), (f"PhiDeg{colNameSuffix}",     )),
        HistogramDefinition(f"PsiDegHF{histNameSuffix}",   ";#Psi = (#Phi#minus#phi_{HF}) [deg];" + yAxisLabel, (( 72, -180,    +180   ), ), (f"PsiDegHF{colNameSuffix}",   )),
        # 2D histograms
        HistogramDefinition(f"cosThetaHF{pairLabel}VsMass{histNameSuffix}",     ";m_{#pi#pi} [GeV];cos#theta_{HF}",  ((50, 0.28, 2.28), (50,   -1,   +1)), (f"mass{colNameSuffix}",       f"cosThetaHF{colNameSuffix}")),
        HistogramDefinition(f"phiDegHF{pairLabel}VsMass{histNameSuffix}",       ";m_{#pi#pi} [GeV];#phi_{HF} [deg]", ((50, 0.28, 2.28), (36, -180, +180)), (f"mass{colNameSuffix}",       f"phiDegHF{colNameSuffix}"  )),
        HistogramDefinition(f"PhiDeg{pairLabel}VsMass{histNameSuffix}",         ";m_{#pi#pi} [GeV];#Phi [deg]",      ((50, 0.28, 2.28), (36, -180, +180)), (f"mass{colNameSuffix}",       f"PhiDeg{colNameSuffix}"    )),
        HistogramDefinition(f"anglesHF{histNameSuffix}",                        ";cos#theta_{HF};#phi_{HF} [deg]",   ((50,   -1,   +1), (36, -180, +180)), (f"cosThetaHF{colNameSuffix}", f"phiDegHF{colNameSuffix}"  )),
        HistogramDefinition(f"PhiDegVsCosThetaHF{histNameSuffix}",              ";cos#theta_{HF};#Phi [deg]",        ((50,   -1,   +1), (36, -180, +180)), (f"cosThetaHF{colNameSuffix}", f"PhiDeg{colNameSuffix}"    )),
        HistogramDefinition(f"PhiDegVsPhiDegHF{histNameSuffix}",                ";#phi_{HF} [deg];#Phi [deg]",       ((36, -180, +180), (36, -180, +180)), (f"phiDegHF{colNameSuffix}",   f"PhiDeg{colNameSuffix}"    )),
        HistogramDefinition(f"PsiDegHF{pairLabel}VsCosThetaHF{histNameSuffix}", ";cos#theta_{HF};#Psi [deg]",        ((50,   -1,   +1), (36, -180, +180)), (f"cosThetaHF{colNameSuffix}", f"PsiDegHF{colNameSuffix}"  )),
        HistogramDefinition(f"PsiDegHF{pairLabel}VsPhiDegHF{histNameSuffix}",   ";#phi_{HF} [deg];#Psi [deg]",       ((36, -180, +180), (36, -180, +180)), (f"phiDegHF{colNameSuffix}",   f"PsiDegHF{colNameSuffix}"  )),
      ]
    if True:
    # if False:
      # distributions in lab frame
      for filter, title, histNameSuffix in [
        ("",                           "",              f"_{label}"                       ),  # all data
        (f"(phiDegHF{pairLabel} > 0)", "#phi_{HF} > 0", f"_phiDegHF{pairLabel}Pos_{label}"),
        (f"(phiDegHF{pairLabel} < 0)", "#phi_{HF} < 0", f"_phiDegHF{pairLabel}Neg_{label}"),
      ]:
        histDefs += [
          # 1D histograms
          HistogramDefinition(f"Ebeam{histNameSuffix}",          title + ";E_{beam} [GeV];"                    + yAxisLabel, ((100,    8,    9  ), ), ("Ebeam",          ), filter),
          HistogramDefinition(f"momLabP{histNameSuffix}",        title + ";p_{p} [GeV];"                       + yAxisLabel, ((100,    0,    1  ), ), ("momLabP",        ), filter),
          HistogramDefinition(f"momLabXP{histNameSuffix}",       title + ";p_{x}^{p} [GeV];"                   + yAxisLabel, ((100,   -0.5, +0.5), ), ("momLabXP",       ), filter),
          HistogramDefinition(f"momLabYP{histNameSuffix}",       title + ";p_{y}^{p} [GeV];"                   + yAxisLabel, ((100,   -0.5, +0.5), ), ("momLabYP",       ), filter),
          HistogramDefinition(f"momLabZP{histNameSuffix}",       title + ";p_{z}^{p} [GeV];"                   + yAxisLabel, ((100,    0,    0.5), ), ("momLabZP",       ), filter),
          HistogramDefinition(f"momLabPip{histNameSuffix}",      title + ";p_{#pi^{#plus}} [GeV];"             + yAxisLabel, ((100,    0,   10  ), ), ("momLabPip",      ), filter),
          HistogramDefinition(f"momLabXPip{histNameSuffix}",     title + ";p_{x}^{#pi^{#plus}} [GeV];"         + yAxisLabel, ((100,   -0.8, +0.8), ), ("momLabXPip",     ), filter),
          HistogramDefinition(f"momLabYPip{histNameSuffix}",     title + ";p_{y}^{#pi^{#plus}} [GeV];"         + yAxisLabel, ((100,   -0.8, +0.8), ), ("momLabYPip",     ), filter),
          HistogramDefinition(f"momLabZPip{histNameSuffix}",     title + ";p_{z}^{#pi^{#plus}} [GeV];"         + yAxisLabel, ((100,   -1,   +9  ), ), ("momLabZPip",     ), filter),
          HistogramDefinition(f"momLabPim{histNameSuffix}",      title + ";p_{#pi^{#minus}} [GeV];"            + yAxisLabel, ((100,    0,   10  ), ), ("momLabPim",      ), filter),
          HistogramDefinition(f"momLabXPim{histNameSuffix}",     title + ";p_{x}^{#pi^{#minus}} [GeV];"        + yAxisLabel, ((100,   -0.8, +0.8), ), ("momLabXPim",     ), filter),
          HistogramDefinition(f"momLabYPim{histNameSuffix}",     title + ";p_{y}^{#pi^{#minus}} [GeV];"        + yAxisLabel, ((100,   -0.8, +0.8), ), ("momLabYPim",     ), filter),
          HistogramDefinition(f"momLabZPim{histNameSuffix}",     title + ";p_{z}^{#pi^{#minus}} [GeV];"        + yAxisLabel, ((100,   -1,   +9  ), ), ("momLabZPim",     ), filter),
          HistogramDefinition(f"thetaDegLabP{histNameSuffix}",   title + ";#theta_{p}^{lab} [deg];"            + yAxisLabel, ((100,    0,   80  ), ), ("thetaDegLabP",   ), filter),
          HistogramDefinition(f"thetaDegLabPip{histNameSuffix}", title + ";#theta_{#pi^{#plus}}^{lab} [deg];"  + yAxisLabel, ((100,    0,   80  ), ), ("thetaDegLabPip", ), filter),
          HistogramDefinition(f"thetaDegLabPim{histNameSuffix}", title + ";#theta_{#pi^{#minus}}^{lab} [deg];" + yAxisLabel, ((100,    0,   80  ), ), ("thetaDegLabPim", ), filter),
          HistogramDefinition(f"phiDegLabP{histNameSuffix}",     title + ";#phi_{p}^{lab} [deg];"              + yAxisLabel, (( 72, -180, +180  ), ), ("phiDegLabP",     ), filter),
          HistogramDefinition(f"phiDegLabPip{histNameSuffix}",   title + ";#phi_{#pi^{#plus}}^{lab} [deg];"    + yAxisLabel, (( 72, -180, +180  ), ), ("phiDegLabPip",   ), filter),
          HistogramDefinition(f"phiDegLabPim{histNameSuffix}",   title + ";#phi_{#pi^{#minus}}^{lab} [deg];"   + yAxisLabel, (( 72, -180, +180  ), ), ("phiDegLabPim",   ), filter),
          # 2D histograms
          HistogramDefinition(f"momLabYPVsMomLabXP{histNameSuffix}",           title + ";p_{x}^{p} [GeV];p_{y}^{p} [GeV];",                                   ((50, -0.5, +0.5), (50,   -0.5,  +0.5)), ("momLabXP",       "momLabYP"      ), filter),
          HistogramDefinition(f"momLabYPipVsMomLabXPip{histNameSuffix}",       title + ";p_{x}^{#pi^{#plus}} [GeV];p_{y}^{#pi^{#plus}} [GeV];",               ((50, -0.8, +0.8), (50,   -0.8,  +0.8)), ("momLabXPip",     "momLabYPip"    ), filter),
          HistogramDefinition(f"momLabYPimVsMomLabXPim{histNameSuffix}",       title + ";p_{x}^{#pi^{#minus}} [GeV];p_{y}^{#pi^{#minus}} [GeV];",             ((50, -0.8, +0.8), (50,   -0.8,  +0.8)), ("momLabXPim",     "momLabYPim"    ), filter),
          HistogramDefinition(f"thetaDegLabPVsMomLabP{histNameSuffix}",        title + ";p_{p} [GeV];#theta_{p}^{lab} [deg]",                                 ((50,  0,    1  ), (50,   60,    80  )), ("momLabP",        "thetaDegLabP"  ), filter),
          HistogramDefinition(f"thetaDegLabPipVsMomLabPip{histNameSuffix}",    title + ";p_{#pi^{#plus}} [GeV];#theta_{#pi^{#plus}}^{lab} [deg]",             ((50,  0,   10  ), (50,    0,    30  )), ("momLabPip",      "thetaDegLabPip"), filter),
          HistogramDefinition(f"thetaDegLabPimVsMomLabPim{histNameSuffix}",    title + ";p_{#pi^{#minus}} [GeV];#theta_{#pi^{#minus}}^{lab} [deg]",           ((50,  0,   10  ), (50,    0,    30  )), ("momLabPim",      "thetaDegLabPim"), filter),
          HistogramDefinition(f"phiDegLabPVsThetaDegLabP{histNameSuffix}",     title + ";#theta_{p}^{lab} [deg];#phi_{p}^{lab} [deg];",                       ((50, 60,   80  ), (36, -180,  +180  )), ("thetaDegLabP",   "phiDegLabP"    ), filter),
          HistogramDefinition(f"phiDegLabPipVsThetaDegLabPip{histNameSuffix}", title + ";#theta_{#pi^{#plus}}^{lab} [deg];#phi_{#pi^{#plus}}^{lab} [deg];",   ((50,  0,   30  ), (36, -180,  +180  )), ("thetaDegLabPip", "phiDegLabPip"  ), filter),
          HistogramDefinition(f"phiDegLabPimVsThetaDegLabPim{histNameSuffix}", title + ";#theta_{#pi^{#minus}}^{lab} [deg];#phi_{#pi^{#minus}}^{lab} [deg];", ((50,  0,   30  ), (36, -180,  +180  )), ("thetaDegLabPim", "phiDegLabPim"  ), filter),
        ]
    # book histograms
    df = getattr(dataToOverlay, dataToOverlayField.name)
    hists = []
    for histDef in histDefs:
      hists.append(bookHistogram(df, histDef, applyWeights = (dataToOverlayField.name == "realData" and df.HasColumn("eventWeight"))))
    setattr(histsToOverlay, dataToOverlayField.name, hists)
  for histRealData, histWeightedMc in zip(histsToOverlay.realData, histsToOverlay.weightedMc):
    print(f"Comparing histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
    histRealData.SetTitle  ("Real data")
    histWeightedMc.SetTitle("Weighted MC")
    # normalize weighted MC to integral of real data
    weightedMcIntegral = histWeightedMc.Integral()
    if weightedMcIntegral != 0:
      histWeightedMc.Scale(histRealData.Integral() / weightedMcIntegral)
    else:
      print(f"??? Warning: weighted-MC histogram '{histWeightedMc.GetName()}' has zero integral, cannot normalize to real data!")
    histRealData.SetMinimum  (0)
    histWeightedMc.SetMinimum(0)  # needs to be set after Scale()
    # generate plots
    ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
    if histRealData.GetDimension() == 1:
      # generate 1D overlay plots
      print(f"Overlaying 1D histograms '{histRealData.GetName()}' and '{histWeightedMc.GetName()}'")
      print("(under-, overflow) fractions: "
            f"'{histRealData.GetName()  }' = ({histRealData.GetBinContent(0)                            / histRealData.Integral()}, "
                                            f"{histRealData.GetBinContent(histRealData.GetNbinsX() + 1) / histRealData.Integral()})"
            f" and '{histWeightedMc.GetName()}' = ({histWeightedMc.GetBinContent(0)                              / histWeightedMc.Integral()}, "
                                                 f"{histWeightedMc.GetBinContent(histWeightedMc.GetNbinsX() + 1) / histWeightedMc.Integral()})")
      ROOT.gStyle.SetOptStat(False)
      canv = ROOT.TCanvas()
      histStack = ROOT.THStack(histWeightedMc.GetName(), "")
      histRealData.SetLineColor  (ROOT.kRed + 1)
      histWeightedMc.SetLineColor(ROOT.kBlue + 1)
      histRealData.SetMarkerColor  (ROOT.kRed + 1)
      histWeightedMc.SetMarkerColor(ROOT.kBlue + 1)
      histWeightedMc.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
      histStack.Add(histWeightedMc.GetValue(), "HIST E")
      histStack.Add(histRealData.GetValue(),   "E")
      histStack.Draw("NOSTACK")
      histStack.SetMaximum(1.1 * max(histRealData.GetMaximum(), histWeightedMc.GetMaximum()))
      histStack.GetXaxis().SetTitle(histWeightedMc.GetXaxis().GetTitle())
      histStack.GetYaxis().SetTitle(histWeightedMc.GetYaxis().GetTitle())
      canv.BuildLegend(0.75, 0.85, 0.99, 0.99)
      nmbNonZeroBins = 0
      for binIndex in range(1, histRealData.GetNbinsX() + 1):
        # both histograms have same binning by construction
        if histRealData.GetBinContent(binIndex) > 0 or histWeightedMc.GetBinContent(binIndex) > 0:
          nmbNonZeroBins += 1
      chi2PerBin = histWeightedMc.Chi2Test(histRealData.GetValue(), "WW P CHI2") / nmbNonZeroBins
      label = ROOT.TLatex()
      label.SetNDC()
      label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignTop)
      label.DrawLatex(0.15, 0.89, f"#it{{#chi}}^{{2}}/bin = {chi2PerBin:.2f}")
      canv.SaveAs(f"{outputDirName}/{histStack.GetName()}{pdfFileNameSuffix}.pdf")
    elif histRealData.GetDimension() == 2:
      # generate 2D comparison plots
      print(f"Plotting real-data 2D histogram '{histRealData.GetName()}'")
      ROOT.gStyle.SetOptStat("i")
      canv = ROOT.TCanvas()
      maxZ = histRealData.GetMaximum()
      histRealData.SetMaximum(maxZ)
      histRealData.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{histRealData.GetName()}{pdfFileNameSuffix}.pdf")
      print(f"Plotting weighted-MC 2D histogram '{histWeightedMc.GetName()}'")
      canv = ROOT.TCanvas()
      histWeightedMc.SetMaximum(maxZ)
      histWeightedMc.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{histWeightedMc.GetName()}{pdfFileNameSuffix}.pdf")
      print(f"Plotting pulls of 2D histograms '{histRealData.GetName()}' - '{histWeightedMc.GetName()}'")
      ROOT.gStyle.SetOptStat(False)
      histPulls = histRealData.Clone(f"{histWeightedMc.GetName()}_pulls")
      histPulls.Add(histWeightedMc.GetValue(), -1)  # real data - weighted MC
      # divide each bin by its uncertainty to get pull
      for xBin in range(1, histPulls.GetNbinsX() + 1):
        for yBin in range(1, histPulls.GetNbinsY() + 1):
          binError = histPulls.GetBinError(xBin, yBin)
          if binError > 0:
            histPulls.SetBinContent(xBin, yBin, histPulls.GetBinContent(xBin, yBin) / binError)
          else:
            histPulls.SetBinContent(xBin, yBin, 0)
      histPulls.SetTitle("(Real data#minusWeighted MC)/#sigma_{Real data}")
      canv = ROOT.TCanvas()
      # draw pull plot with pos/neg color palette and symmetric z axis
      ROOT.gStyle.SetPalette(ROOT.kLightTemperature)
      zRange = max(abs(histPulls.GetMinimum()), abs(histPulls.GetMaximum()))
      histPulls.SetMinimum(-zRange)
      histPulls.SetMaximum(+zRange)
      histPulls.Draw("COLZ")
      canv.SaveAs(f"{outputDirName}/{histPulls.GetName()}{pdfFileNameSuffix}.pdf")
      ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette
    else:
      raise RuntimeError(f"Unsupported histogram type '{histRealData.ClassName()}'")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_ANGLES_GLUEX_AMPTOOLS)
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)

  dataPeriods = (
    # "2017_01",
    "2018_08",
  )
  tBinLabels = (
    "tbin_0.1_0.2",
    # "tbin_0.2_0.3",
    # "tbin_0.3_0.4",
    # "tbin_0.4_0.5",
  )
  beamPolLabels = (
    "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
  )
  maxL              = 4
  # maxL              = 8
  massMin           = 0.28  # [GeV]
  massBinWidth      = 0.04  # [GeV]
  nmbBins           = 50
  subSystem         = SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim", lvRecoilLabel = "recoil", pairTLatexLabel = "#pi#pi")
  useIntensityTerms = "allTerms"
  # useIntensityTerms = "parityConserving"
  # useIntensityTerms = "parityViolating"

  for dataPeriod in dataPeriods:
    print(f"Generating plots for data period '{dataPeriod}':")
    for tBinLabel in tBinLabels:
      print(f"Generating plots for t bin '{tBinLabel}':")
      for beamPolLabel in beamPolLabels:
        beamPolInfo = BEAM_POL_INFOS[dataPeriod][beamPolLabel]
        print(f"Generating plots for beam-polarization orientation '{beamPolLabel}'"
              + (f": pol = {beamPolInfo.pol:.4f}, PhiLab = {beamPolInfo.PhiLab:.1f} deg" if beamPolInfo is not None else ""))
        # load data in AMPTOOLS format
        dataDirName          = f"./polarized/{dataPeriod}/{tBinLabel}"
        weightedDataDirName  = f"{dataDirName}/{subSystem.pairLabel}/weightedMc.maxL_{maxL}/{beamPolLabel}"
        weightedDataFileName = f"{weightedDataDirName}/phaseSpace_acc_weighted_raw_{useIntensityTerms}.root"
        dataToOverlay = DataToOverlay(
          realData   = getDataFrameWithCorrectEventWeights(
            dataSigRegionFileNames  = (f"{dataDirName}/Alex/amptools_tree_signal_{beamPolLabel}.root", ),
            dataBkgRegionFileNames  = (f"{dataDirName}/Alex/amptools_tree_bkgnd_{beamPolLabel}.root",  ),
            treeName                = "kin",
            friendSigRegionFileName = f"{dataDirName}/data_sig_{beamPolLabel}.root.weights",
            friendBkgRegionFileName = f"{dataDirName}/data_bkg_{beamPolLabel}.root.weights",
          ),
          weightedMc = ROOT.RDataFrame(subSystem.pairLabel, weightedDataFileName),
        )
        print(f"Loaded weighted-MC data from '{weightedDataFileName}'")
        # loop over members of `DataToOverlay` and define columns needed for plotting for `realData` and `weightedMc`
        for member in fields(dataToOverlay):
          df = getattr(dataToOverlay, member.name)
          df = defineColumnsForPlots(
            df              = df,
            inputDataFormat = InputDataFormat.AMPTOOLS,
            subSystem       = subSystem,
            beamPolInfo     = BEAM_POL_INFOS[dataPeriod][beamPolLabel],
          )
          setattr(dataToOverlay, member.name, df)
        # plot overlays for full mass range and for individual mass bins
        plotDirName = f"{weightedDataDirName}/{useIntensityTerms}"
        print(f"Overlaying histograms for full mass range and writing plots into '{plotDirName}'")
        os.makedirs(plotDirName, exist_ok = True)
        makePlots(
          dataToOverlay = dataToOverlay,
          outputDirName = plotDirName,
          colNameSuffix = subSystem.pairLabel,
        )
        for massBinIndex in range(nmbBins):
          massBinMin = massMin + massBinIndex * massBinWidth
          massBinMax = massBinMin + massBinWidth
          print(f"Overlaying histograms for mass bin {massBinIndex} with range [{massBinMin:.2f}, {massBinMax:.2f}] GeV")
          massRangeFilter = f"(({massBinMin} < mass{subSystem.pairLabel}) && (mass{subSystem.pairLabel} < {massBinMax}))"
          makePlots(
            dataToOverlay     = dataToOverlay.Filter(massRangeFilter),
            outputDirName     = plotDirName,
            pdfFileNameSuffix = f"_{massBinMin:.2f}_{massBinMax:.2f}",
            colNameSuffix     = subSystem.pairLabel,
          )
