#!/usr/bin/env python3
"""
This module plots the results of the moment analysis of unpolarized pi+ pi- photoproduction data in the CLAS energy range.
The moment values are read from files produced by the script that calculates the moments.

Usage:
Run this module as a script to generate the output files.
"""


from __future__ import annotations

import functools
import glob
from io import StringIO
import math
import numpy as np
import pandas as pd

import ROOT
from wurlitzer import pipes, STDOUT

from calcMomentsPhotoProdPiPiUnpol import (
  AnalysisConfig,
  CFG,
)
import MomentCalculator
from MomentCalculator import (
  AcceptanceIntegralMatrix,
  constructMomentResultFrom,
  DataSet,
  KinematicBinningVariable,
  MomentIndices,
  MomentResult,
  MomentResultsKinematicBinning,
  MomentValue,
  QnMomentIndex,
)
from PlottingUtilities import (
  convertGraphToHist,
  makeMomentHistogram,
  MomentValueAndTruth,
  plotAngularDistr,
  plotComplexMatrix,
  plotMoments,
  plotMoments1D,
  plotMomentsBootstrapDiffInBin,
  plotMomentsBootstrapDistributions1D,
  plotMomentsBootstrapDistributions2D,
  plotMomentsCovMatrices,
  plotMomentsInBin,
  setupPlotStyle,
)
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def readMomentResultsClas(
    momentIndices:      MomentIndices,  # moment indices to read
    binVarMass:         KinematicBinningVariable,  # binning variable for mass bins
    tBinLabel:          str = "#: -T [GeV^2],,,0.4-0.5",
    # beamEnergyBinLabel: str = "#: E(P=1) [GeV],,,3.4-3.6",
    beamEnergyBinLabel: str = "#: E(P=1) [GeV],,,3.6-3.8",
    csvDirName:         str = "./dataPhotoProdPiPiUnpol/HEPData-ins825040-v1-csv",
) -> MomentResultsKinematicBinning:
  """Reads the moment values in the for the given moment indices and from the CLAS analysis in PRD 80 (2009) 072005 published at https://www.hepdata.net/record/ins825040"""
  csvFileNames = sorted(glob.glob(f"{csvDirName}/Table*.csv"))
  # each file contains the values for a given H(L, M) moment and a given t bin for all 4 beam-energy bins and all mass bins
  momentDfs: dict[QnMomentIndex, pd.DataFrame] = {}  # key: moment quantum numbers, value: Pandas dataframe with moment values in mass bins
  for qnMomentIndex in momentIndices.qnIndices:
    for csvFileName in csvFileNames:
      # first step: open file, read whole contents, and filter pick the file that contain the desired moment an the t bin
      with open(csvFileName, "r") as csvFile:
        csvData = csvFile.read()
        momentLabel = f"YLM(LM={qnMomentIndex.L}{qnMomentIndex.M},P=3_4) [MUB/GEV**3]"
        if (momentLabel in csvData) and (tBinLabel in csvData):
          print(f"Reading CLAS values for moment {qnMomentIndex.label} from file '{csvFileName}'")
          # within each file there are sub-tables for the 4 beam-energy bins
          # second step: extract sub-table for the desired beam-energy bin
          tableStartPos = csvData.find(beamEnergyBinLabel)
          assert tableStartPos >= 0, f"Could not find table for beam energy bin '{beamEnergyBinLabel}' in file '{csvFileName}'"
          tableEndPos = csvData.find("\n\n", tableStartPos)  # tables are separated by an empty line
          csvData = csvData[tableStartPos:tableEndPos if tableEndPos >= 0 else None]
          momentDf = pd.read_csv(
            StringIO(csvData),
            skiprows = 3,  # 2 rows with comments and 1 row with column names; the latter cannot be parsed by pandas
            names    = ["mass", "massLow", "massHigh", "moment", "uncertPlus", "uncertMinus"],
          )
          # scale moment values and their uncertainties by 1 / sqrt(2L + 1) to match normalization used in this analysis
          momentDf[["moment", "uncertPlus", "uncertMinus"]] /= np.sqrt(2 * qnMomentIndex.L + 1)
          momentDfs[qnMomentIndex] = momentDf
          break
  # ensure that mass bins are the same in all dataframes
  dfs = list(momentDfs.values())
  massColumn = dfs[0]["mass"]
  for df in dfs[1:]:
    assert df["mass"].equals(massColumn), f"Mass bins in dataframes differ:\n{df['mass']}\nvs.\n{massColumn}"
  # convert dataframes to MomentResultsKinematicBinning
  momentResults: list[MomentResult] = []
  for massBinCenter in massColumn:
    # loop over momentDfs and extract moment values for the given mass bin
    momentValues: list[MomentValue] = []
    for qnMomentIndex, momentDf in momentDfs.items():
      mask = (momentDf["mass"] == massBinCenter)
      moment, uncertPlus, uncertMinus = momentDf[mask][["moment", "uncertPlus", "uncertMinus"]].values[0]
      assert uncertPlus == -uncertMinus, f"Uncertainties are not symmetric: {uncertPlus} vs. {uncertMinus}"
      momentValues.append(
        MomentValue(
          qn         = qnMomentIndex,
          val        = moment,
          uncertRe   = uncertPlus,
          uncertIm   = 0,
          binCenters = {binVarMass: massBinCenter},
        )
      )
    momentResults.append(constructMomentResultFrom(momentIndices, momentValues))
  return MomentResultsKinematicBinning(momentResults)


def readMomentResultsJpac(
    momentIndices: MomentIndices,  # moment indices to read
    binVarMass:    KinematicBinningVariable,  # binning variable for mass bins
    tBinLabel:     str = "t=0.40-0.50",
    dataDirName:   str = "./dataPhotoProdPiPiUnpol/2406.08016",
) -> MomentResultsKinematicBinning:
  """Reads the moments values from the JPAC fit to the CLAS data in range 3.6 < E_gamma < 3.8 GeV as published in Figs. 6, 7, 13--16 in arXiv:2406.08016"""
  momentDfs: dict[QnMomentIndex, pd.DataFrame] = {}  # key: moment quantum numbers, value: Pandas dataframe with moment values in mass bins
  for qnMomentIndex in momentIndices.qnIndices:
    dataFileName = f"{dataDirName}/Y{qnMomentIndex.L}{qnMomentIndex.M}{tBinLabel}.dat"
    print(f"Reading JPAC values for moment {qnMomentIndex.label} from file '{dataFileName}'")
    try:
      momentDf = pd.read_csv(
        dataFileName,
        sep      = r"\s+",  # values are whitespace separated
        skiprows = 1,     # first row with column names
        names    = ["mass", "moment", "uncert"],
      )
      # scale moment values and their uncertainties by 1 / sqrt(2L + 1) to match normalization used in this analysis
      momentDf[["moment", "uncert"]] /= np.sqrt(2 * qnMomentIndex.L + 1)
      momentDfs[qnMomentIndex] = momentDf
    except FileNotFoundError as e:
        print(f"Warning: file '{dataFileName}' not found. Skipping moment {qnMomentIndex.label}.")
  # ensure that mass bins are the same in all dataframes
  dfs = list(momentDfs.values())
  massColumn = dfs[0]["mass"]
  for df in dfs[1:]:
    assert df["mass"].equals(massColumn), f"Mass bins in dataframes differ:\n{df['mass']}\nvs.\n{massColumn}"
  # convert dataframes to MomentResultsKinematicBinning
  momentResults: list[MomentResult] = []
  for massBinCenter in massColumn:
    # loop over momentDfs and extract moment values for the given mass bin
    momentValues: list[MomentValue] = []
    for qnMomentIndex, momentDf in momentDfs.items():
      mask = momentDf["mass"] == massBinCenter
      moment, uncert = momentDf[mask][["moment", "uncert"]].values[0]
      momentValues.append(
        MomentValue(
          qn         = qnMomentIndex,
          val        = moment,
          uncertRe   = uncert,
          uncertIm   = 0,
          binCenters = {binVarMass: massBinCenter},
        )
      )
    momentResults.append(constructMomentResultFrom(momentIndices, momentValues))
  return MomentResultsKinematicBinning(momentResults)


def makeAllPlots(
  cfg:   AnalysisConfig,
  timer: Utilities.Timer = Utilities.Timer(),
) -> None:
  """Generates all plots for the given analysis configuration"""
  # load moments from files
  momentIndices = MomentIndices(cfg.maxL)
  momentResultsFileBaseName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments"
  momentResultsMeas = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_meas.pkl")
  momentResultsPhys = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_phys.pkl")
  # momentResultsCompare      = readMomentResultsClas(momentIndices, cfg.binVarMass)
  # momentResultsCompareColor = ROOT.kGray + 1
  # momentResultsCompareLabel = "CLAS"
  momentResultsCompare      = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_pwa_SPD.pkl")
  momentResultsCompareLabel = "PWA #it{S} #plus #it{P} #plus #it{D}"
  # momentResultsCompare      = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_pwa_SPDF.pkl")
  # momentResultsCompareLabel = "PWA #it{S} #plus #it{P} #plus #it{D} #plus #it{F}"
  momentResultsCompareColor = ROOT.kBlue + 1
  momentResultsJpac         = readMomentResultsJpac(momentIndices, cfg.binVarMass)
  momentResultsJpacLabel    = "JPAC"
  # overlayMomentResultsJpac  = True
  overlayMomentResultsJpac  = False

  # normalize comparison and JPAC moments
  normalizeByIntegral = True  # if false comparison and JPAC moments are normalized to the maximum bin
  H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
  if normalizeByIntegral:
    # loop over mass bins and sum up H(0, 0) values
    H000Sum = H000SumComp = H000SumJpac = 0.0
    for HPhys, HComp, HJpac in zip(momentResultsPhys, momentResultsCompare, momentResultsJpac):
      H000Sum     += HPhys[H000Index].val.real
      H000SumComp += HComp[H000Index].val.real
      # H000SumJpac += HJpac[H000Index].val.real
    # momentResultsCompare.scaleBy(1 / (8 * math.pi))  # this works for PWA result
    print(f"!!! scale factor = {H000Sum / H000SumComp}")
    momentResultsCompare.scaleBy(H000Sum / H000SumComp)
    momentResultsJpac.scaleBy   (H000Sum / H000SumComp)  # use same factor as for comparison moments
    # momentResultsJpac.scaleBy(H000Sum / H000SumJpac)
  else:
    normMassBinIndex = 36  # corresponds to m_pipi = 0.765 GeV; in this bin H(0, 0) is maximal in CLAS and GlueX data
    H000Value = momentResultsPhys[normMassBinIndex][H000Index].val.real
    momentResultsCompare.scaleBy(H000Value / momentResultsCompare[normMassBinIndex][H000Index].val.real)
    momentResultsJpac.scaleBy   (H000Value / momentResultsCompare[normMassBinIndex][H000Index].val.real)  # use same factor as for comparison moments
    # momentResultsJpac.scaleBy(H000Value / momentResultsJpac[normMassBinIndex][H000Index].val.real)

  if True:
    with timer.timeThis(f"Time to plot results from analysis of real data"):
      # plot moments in each mass bin
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        HMeas = momentResultsMeas[massBinIndex]
        HComp = momentResultsCompare[massBinIndex]
        binLabel = MomentCalculator.binLabel(HPhys)
        binTitle = MomentCalculator.binTitle(HPhys)
        # print(f"True moments for kinematic bin {title}:\n{HTruth}")
        print(f"Measured moments of real data for kinematic bin {binTitle}:\n{HMeas}")
        print(f"Physical moments of real data for kinematic bin {binTitle}:\n{HPhys}")
        plotMomentsInBin(
          HData             = HPhys,
          normalizedMoments = cfg.normalizeMoments,
          HTruth            = HComp,
          pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_",
          legendLabels      = ("Moment", momentResultsCompareLabel),
          plotTruthUncert   = True,
          truthColor        = momentResultsCompareColor,
        )
        plotMomentsInBin(
          HData             = HMeas,
          normalizedMoments = cfg.normalizeMoments,
          HTruth            = None,
          pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_{binLabel}_",
          plotLegend        = False,
        )
        #TODO also plot correlation matrices
        plotMomentsCovMatrices(
          HData             = HPhys,
          pdfFileNamePrefix = f"{cfg.outFileDirName}/covMatrix_{binLabel}_",
          axisTitles        = ("Physical Moment Index", "Physical Moment Index"),
          plotTitle         = f"{binLabel}: ",
        )
        if cfg.nmbBootstrapSamples > 0:
          graphTitle = f"({binLabel})"
          plotMomentsBootstrapDistributions1D(
            HData             = HPhys,
            HTruth            = HComp,
            pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_",
            histTitle         = binTitle,
            HTruthLabel       = momentResultsCompareLabel,
          )
          plotMomentsBootstrapDistributions2D(
            HData             = HPhys,
            HTruth            = HComp,
            pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_",
            histTitle         = binTitle,
            HTruthLabel       = momentResultsCompareLabel,
          )
          plotMomentsBootstrapDiffInBin(
            HData             = HPhys,
            pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_",
            graphTitle        = binTitle,
          )

      # plot mass dependences of all moments
      for qnIndex in momentResultsPhys[0].indices.qnIndices:
        # get histogram with moment values from JPAC fit
        histJpac: ROOT.TH1D | None = None
        if overlayMomentResultsJpac:
          HValsJpac = tuple(MomentValueAndTruth(*HPhys[qnIndex]) for HPhys in momentResultsJpac)
          histJpac = makeMomentHistogram(
            HVals      = HValsJpac,
            momentPart = "Re",
            histName   = momentResultsJpacLabel,
            histTitle  = "",
            binning    = cfg.massBinning,
            plotTruth  = False,
            plotUncert = True,
          )
          if histJpac is not None:
            histJpacBand = histJpac.Clone(f"{histJpac.GetName()}_band")
            histJpac.SetLineColor(ROOT.kBlue + 1)
            histJpac.SetLineWidth(2)
            histJpacBand.SetFillColorAlpha(ROOT.kBlue + 1, 0.3)
        histPwaTotalIntensity = None
        if qnIndex == H000Index:
          plotFile = ROOT.TFile.Open("./dataPhotoProdPiPiUnpol/PWA_S_P_D/pwa_plots_weight1.root", "READ")
          histPwaTotalIntensity = convertGraphToHist(
            graph     = plotFile.Get("Total"),
            binning   = cfg.massBinning.astuple,
            histName  = "Total Intensity",
            histTitle = "",
          )
          histPwaTotalIntensity.SetLineColor(ROOT.kGreen + 2)
        plotMoments1D(
          momentResults     = momentResultsPhys,
          qnIndex           = qnIndex,
          binning           = cfg.massBinning,
          normalizedMoments = cfg.normalizeMoments,
          momentResultsTrue = momentResultsCompare,
          pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_",
          histTitle         = qnIndex.title,
          plotLegend        = True,
          legendLabels      = ("Moment", momentResultsCompareLabel),
          plotTruthUncert   = True,
          truthColor        = momentResultsCompareColor,
          histsToOverlay    = {} if histJpac is None else {  # dict: key = "Re" or "Im", list: tuple: (histogram, draw option, legend entry)
            "Re" : [
              (histJpac,     "HIST L", histJpac.GetName()),
              (histJpacBand,     "E3", ""),
            ],
          },
          # histsToOverlay    = {} if histPwaTotalIntensity is None else {  # dict: key = "Re" or "Im", list: tuple: (histogram, draw option, legend entry)
          #   "Re" : [
          #     (histPwaTotalIntensity, "HIST", histPwaTotalIntensity.GetName()),
          #   ],
          # },
        )
        plotMoments1D(
          momentResults     = momentResultsMeas,
          qnIndex           = qnIndex,
          binning           = cfg.massBinning,
          normalizedMoments = cfg.normalizeMoments,
          momentResultsTrue = None,
          pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_",
          histTitle         = qnIndex.title,
          plotLegend        = False,
        )

      # plot ratio of measured and physical value for Re[H_0(0, 0)]; estimates efficiency
      H000s = (
        tuple(MomentValueAndTruth(*HMeas[H000Index]) for HMeas in momentResultsMeas),
        tuple(MomentValueAndTruth(*HPhys[H000Index]) for HPhys in momentResultsPhys),
      )
      hists = (
        ROOT.TH1D(f"H000Meas", "", *cfg.massBinning.astuple),
        ROOT.TH1D(f"H000Phys", "", *cfg.massBinning.astuple),
      )
      for indexMeasPhys, H000 in enumerate(H000s):
        histIntensity = hists[indexMeasPhys]
        for HVal in H000:
          if (cfg.massBinning._var not in HVal.binCenters.keys()):
            continue
          y, yErr = HVal.part(True)
          binIndex = histIntensity.GetXaxis().FindBin(HVal.binCenters[cfg.massBinning.var])
          histIntensity.SetBinContent(binIndex, y)
          histIntensity.SetBinError  (binIndex, 1e-100 if yErr < 1e-100 else yErr)
      histRatio = hists[0].Clone("H000Ratio")
      histRatio.Divide(hists[1])
      canv = ROOT.TCanvas()
      histRatio.SetMarkerStyle(ROOT.kFullCircle)
      histRatio.SetMarkerSize(0.75)
      histRatio.SetXTitle(cfg.massBinning.axisTitle)
      histRatio.SetYTitle("#it{H}_{0}^{meas}(0, 0) / #it{H}_{0}^{phys}(0, 0)")
      histRatio.Draw("PEX0")
      canv.SaveAs(f"{cfg.outFileDirName}/{histRatio.GetName()}.pdf")

      # overlay H_0^meas(0, 0) and measured intensity distribution; must be identical
      histIntMeas = ROOT.RDataFrame(cfg.treeName, cfg.dataFileName) \
                        .Histo1D(
                          ROOT.RDF.TH1DModel("intensity_meas", f";{cfg.massBinning.axisTitle};Events", *cfg.massBinning.astuple), "mass", "eventWeight"
                        ).GetValue()
      for binIndex, H000Meas in enumerate(H000s[0]):
        H000Meas.truth = histIntMeas.GetBinContent(binIndex + 1)  # set truth values to measured intensity
      plotMoments(
        HVals             = H000s[0],
        binning           = cfg.massBinning,
        normalizedMoments = cfg.normalizeMoments,
        momentLabel       = H000Index.label,
        pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_intensity_",
        histTitle         = H000Index.title,
        legendLabels      = ("Measured Moment", "Measured Intensity"),
      )

  nmbPsGenEvents: list[int] = []  # number of generated phase-space events; needed to plot accepted phase-space moments
  if cfg.plotAngularDistributions:
    with timer.timeThis(f"Time to plot angular distributions"):
      print("Plotting angular distributions")
      # load all signal and phase-space data
      print(f"Loading real data from tree '{cfg.treeName}' in file '{cfg.dataFileName}'")
      data = ROOT.RDataFrame(cfg.treeName, cfg.dataFileName)
      print(f"Loading accepted phase-space data from tree '{cfg.treeName}' in file '{cfg.psAccFileName}'")
      dataPsAcc = ROOT.RDataFrame(cfg.treeName, cfg.psAccFileName)
      print(f"Loading generated phase-space data from tree '{cfg.treeName}' in file '{cfg.psGenFileName}'")
      dataPsGen = ROOT.RDataFrame(cfg.treeName, cfg.psGenFileName)
      # plot total angular distribution
      plotAngularDistr(
        dataPsAcc         = dataPsAcc,
        dataPsGen         = dataPsGen,
        dataSignalAcc     = data,
        dataSignalGen     = None,
        pdfFileNamePrefix = f"{cfg.outFileDirName}/angDistr_total_",
      )
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        # load data for mass bin
        massBinRange = cfg.massBinning.binValueRange(massBinIndex)
        binMassRangeFilter = f"(({massBinRange[0]} < {cfg.binVarMass.name}) && ({cfg.binVarMass.name} < {massBinRange[1]}))"
        dataInBin      = data.Filter     (binMassRangeFilter)
        dataPsAccInBin = dataPsAcc.Filter(binMassRangeFilter)
        dataPsGenInBin = dataPsGen.Filter(binMassRangeFilter)
        nmbPsGenEvents.append(dataPsGenInBin.Count().GetValue())
        # plot angular distributions for mass bin
        if cfg.plotAngularDistributions:
          plotAngularDistr(
            dataPsAcc         = dataPsAccInBin,
            dataPsGen         = dataPsGenInBin,
            dataSignalAcc     = dataInBin,
            dataSignalGen     = None,
            pdfFileNamePrefix = f"{cfg.outFileDirName}/angDistr_{MomentCalculator.binLabel(HPhys)}_"
          )

  if cfg.plotAccIntegralMatrices:
    with timer.timeThis(f"Time to plot acceptance integral matrices"):
      # plot acceptance integral matrices for all mass bins
      for HPhys in momentResultsPhys:
        # load integral matrix
        binLabel = MomentCalculator.binLabel(HPhys)
        accIntMatrix = AcceptanceIntegralMatrix(
          indices = HPhys.indices,
          dataSet = DataSet(
            data           = None,
            phaseSpaceData = None,
            nmbGenEvents   = 0,
            polarization   = None,
          ),
        )  # dummy matrix without dataset
        accIntMatrix.load(f"{cfg.outFileDirName}/integralMatrix_{binLabel}.npy")
        plotComplexMatrix(
          complexMatrix     = accIntMatrix.matrixNormalized,
          pdfFileNamePrefix = f"{cfg.outFileDirName}/accMatrix_{binLabel}_",
          axisTitles        = ("Physical Moment Index", "Measured Moment Index"),
          plotTitle         = f"{binLabel}: "r"$\mathrm{\mathbf{I}}_\text{acc}$, ",
          zRangeAbs         = 1.2,
          zRangeImag        = 0.05,
        )
        plotComplexMatrix(
          complexMatrix     = accIntMatrix.inverse,
          pdfFileNamePrefix = f"{cfg.outFileDirName}/accMatrixInv_{binLabel}_",
          axisTitles        = ("Measured Moment Index", "Physical Moment Index"),
          plotTitle         = f"{binLabel}: "r"$\mathrm{\mathbf{I}}_\text{acc}^{-1}$, ",
          zRangeAbs         = 5,
          zRangeImag        = 0.3,
        )

  if cfg.plotAccPsMoments:
    # load accepted phase-space moments
    try:
      momentResultsAccPsMeas = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_accPs_meas.pkl")
      momentResultsAccPsPhys = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_accPs_phys.pkl")
    except FileNotFoundError as e:
      print(f"Warning: File not found. Cannot plot accepted phase-space moments. {e}")
    else:
      with timer.timeThis(f"Time to plot accepted phase-space moments"):
        # plot mass dependences of all phase-space moments
        for qnIndex in momentIndices.qnIndices:
          HVals = tuple(MomentValueAndTruth(*momentResultInBin[qnIndex]) for momentResultInBin in momentResultsAccPsMeas)
          plotMoments(
            HVals             = HVals,
            binning           = cfg.massBinning,
            normalizedMoments = cfg.normalizeMoments,
            momentLabel       = qnIndex.label,
            pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{cfg.massBinning.var.name}_accPs_",
            histTitle         = qnIndex.title,
            plotLegend        = False,
          )
        # plot accepted phase-space moments in each mass bin
        for massBinIndex, HPhys in enumerate(momentResultsAccPsPhys):
          binLabel = MomentCalculator.binLabel(HPhys)
          binTitle = MomentCalculator.binTitle(HPhys)
          HMeas = momentResultsAccPsMeas[massBinIndex]
          print(f"Measured moments of accepted phase-space data for kinematic bin {binTitle}:\n{HMeas}")
          print(f"Physical moments of accepted phase-space data for kinematic bin {binTitle}:\n{HPhys}")
          # construct true moments for phase-space data
          HTruthPs = MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
          HTruthPs._valsFlatIndex[momentIndices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1 if cfg.normalizeMoments else nmbPsGenEvents[massBinIndex]  # ... except for H_0(0, 0)
          # set H_0^meas(0, 0) to 0 so that one can better see the other H_0^meas moments
          HMeas._valsFlatIndex[0] = 0
          # plot measured and physical moments; the latter should match the true moments exactly except for tiny numerical effects
          plotMomentsInBin(
            HData             = HMeas,
            normalizedMoments = cfg.normalizeMoments,
            HTruth            = None,
            pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_accPs_",
            plotLegend        = False,
          )
          plotMomentsInBin(
            HData             = HPhys,
            normalizedMoments = cfg.normalizeMoments,
            HTruth            = HTruthPs,
            pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_accPsCorr_"
          )


if __name__ == "__main__":
  # for maxL in (2, 4, 5, 8, 10, 12, 20):
  for maxL in (8, ):
    print(f"Plotting moments for L_max = {maxL}")
    CFG.maxL = maxL
    logFileName = f"{CFG.outFileDirName}/plotMomentsPhotoProdPiPiUnpol.log"
    print(f"Writing output to log file '{logFileName}'")
    with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
      Utilities.printGitInfo()
      timer = Utilities.Timer()
      ROOT.gROOT.SetBatch(True)
      setupPlotStyle()

      timer.start("Total execution time")

      makeAllPlots(CFG, timer)

      timer.stop("Total execution time")
      print(timer.summary)