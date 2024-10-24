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
import numpy as np
import pandas as pd

import ROOT

from calcMomentsPhotoProdPiPiUnpol import (
  AnalysisConfig,
  CFG,
)
from MomentCalculator import (
  binLabel,
  binTitle,
  constructMomentResultFrom,
  KinematicBinningVariable,
  MomentIndices,
  MomentResult,
  MomentResultsKinematicBinning,
  MomentValue,
  QnMomentIndex,
)
from PlottingUtilities import (
  makeMomentHistogram,
  MomentValueAndTruth,
  plotAngularDistr,
  plotMoments,
  plotMoments1D,
  plotMomentsBootstrapDiffInBin,
  plotMomentsBootstrapDistributions1D,
  plotMomentsBootstrapDistributions2D,
  plotMomentsCovMatrices,
  plotMomentsInBin,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads basisFunctions.C
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
  momentDfs: dict[QnMomentIndex, pd.DataFrame] = {}  # key: moment quantum numbers, value: Pandas data frame with moment values in mass bins
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
  # ensure that mass bins are the same in all data frames
  dfs = list(momentDfs.values())
  massColunm = dfs[0]["mass"]
  for df in dfs[1:]:
    assert df["mass"].equals(massColunm), f"Mass bins in data frames differ:\n{df['mass']}\nvs.\n{massColunm}"
  # convert data frames to MomentResultsKinematicBinning
  momentResults: list[MomentResult] = []
  for massBinCenter in massColunm:
    # loop over momentDfs and extract moment values for the given mass bin
    momentValues: list[MomentValue] = []
    for qnMomentIndex, momentDf in momentDfs.items():
      mask = momentDf["mass"] == massBinCenter
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
  momentDfs: dict[QnMomentIndex, pd.DataFrame] = {}  # key: moment quantum numbers, value: Pandas data frame with moment values in mass bins
  for qnMomentIndex in momentIndices.qnIndices:
    datFileName = f"{dataDirName}/Y{qnMomentIndex.L}{qnMomentIndex.M}{tBinLabel}.dat"
    print(f"Reading JPAC values for moment {qnMomentIndex.label} from file '{datFileName}'")
    try:
      momentDf = pd.read_csv(
        datFileName,
        sep      ='\s+',  # values are whitespace separated
        skiprows = 1,     # first row with column names
        names    = ["mass", "moment", "uncert"],
      )
      # scale moment values and their uncertainties by 1 / sqrt(2L + 1) to match normalization used in this analysis
      momentDf[["moment", "uncert"]] /= np.sqrt(2 * qnMomentIndex.L + 1)
      momentDfs[qnMomentIndex] = momentDf
    except FileNotFoundError as e:
        print(f"Warning: file '{datFileName}' not found. Skipping moment {qnMomentIndex.label}.")
  # ensure that mass bins are the same in all data frames
  dfs = list(momentDfs.values())
  massColunm = dfs[0]["mass"]
  for df in dfs[1:]:
    assert df["mass"].equals(massColunm), f"Mass bins in data frames differ:\n{df['mass']}\nvs.\n{massColunm}"
  # convert data frames to MomentResultsKinematicBinning
  momentResults: list[MomentResult] = []
  for massBinCenter in massColunm:
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


def makeAllPlots(cfg: AnalysisConfig) -> None:
  """Generates all plots for the given analysis configuration"""
  timer.start("Total execution time")

  if cfg.plotAngularDistributions:
    print("Plotting total angular distributions")
    # load all signal and phase-space data
    print(f"Loading real data from tree '{cfg.treeName}' in file '{cfg.dataFileName}'")
    data = ROOT.RDataFrame(cfg.treeName, cfg.dataFileName)
    print(f"Loading accepted phase-space data from tree '{cfg.treeName}' in file '{cfg.psAccFileName}'")
    dataPsAcc = ROOT.RDataFrame(cfg.treeName, cfg.psAccFileName)
    print(f"Loading generated phase-space data from tree '{cfg.treeName}' in file '{cfg.psGenFileName}'")
    dataPsGen = ROOT.RDataFrame(cfg.treeName, cfg.psGenFileName)
    plotAngularDistr(
      dataPsAcc         = dataPsAcc,
      dataPsGen         = dataPsGen,
      dataSignalAcc     = data,
      dataSignalGen     = None,
      pdfFileNamePrefix = f"{cfg.outFileDirName}/angDistr_total_",
    )

  # load moment results from files
  momentIndices = MomentIndices(cfg.maxL)
  momentResultsFileBaseName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments"
  momentResultsMeas = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_meas.pkl")
  momentResultsPhys = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_phys.pkl")
  momentResultsClas = readMomentResultsClas(momentIndices, cfg.binVarMass)
  momentResultsJpac = readMomentResultsJpac(momentIndices, cfg.binVarMass)

  # normalize CLAS and JPAC moments
  # TODO normalize by integral
  normMassBinIndex = 36  # corresponds to m_pipi = 0.765 GeV; in this bin H(0, 0) is maximal in CLAS and GlueX data
  H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
  H000Value = momentResultsPhys[normMassBinIndex][H000Index].val.real
  momentResultsClas.scaleBy(H000Value / momentResultsClas[normMassBinIndex][H000Index].val.real)
  momentResultsJpac.scaleBy(H000Value / momentResultsJpac[normMassBinIndex][H000Index].val.real)

  # plot moments in each kinematic bin
  for massBinIndex, HPhys in enumerate(momentResultsPhys):
    HMeas = momentResultsMeas[massBinIndex]
    HClas = momentResultsClas[massBinIndex]
    label = binLabel(HPhys)
    title = binTitle(HPhys)
    # print(f"True moments for kinematic bin {title}:\n{HTruth}")
    print(f"Measured moments of real data for kinematic bin {title}:\n{HMeas}")
    print(f"Physical moments of real data for kinematic bin {title}:\n{HPhys}")
    plotMomentsInBin(
      HData             = HPhys,
      normalizedMoments = cfg.normalizeMoments,
      HTruth            = HClas,
      pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{label}_",
      legendLabels      = ("Moment", "CLAS"),
      plotTruthUncert   = True,
      truthColor        = ROOT.kGray + 1,
    )
    plotMomentsInBin(
      HData             = HMeas,
      normalizedMoments = cfg.normalizeMoments,
      HTruth            = None,
      pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_{label}_",
      plotLegend        = False,
    )
    #TODO also plot correlation matrices
    plotMomentsCovMatrices(
      HData             = HPhys,
      pdfFileNamePrefix = f"{cfg.outFileDirName}/covMatrix_{label}_",
      axisTitles        = ("Physical Moment Index", "Physical Moment Index"),
      plotTitle         = f"{label}: ",
    )
    if cfg.nmbBootstrapSamples > 0:
      graphTitle = f"({label})"
      plotMomentsBootstrapDistributions1D(
        HData             = HPhys,
        HTruth            = HClas,
        pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{label}_",
        histTitle         = title,
        HTruthLabel       = "CLAS",
      )
      plotMomentsBootstrapDistributions2D(
        HData             = HPhys,
        HTruth            = HClas,
        pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{label}_",
        histTitle         = title,
        HTruthLabel       = "CLAS",
      )
      plotMomentsBootstrapDiffInBin(
        HData             = HPhys,
        pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{label}_",
        graphTitle        = title,
      )

  # plot kinematic dependences of all moments
  for qnIndex in momentResultsPhys[0].indices.qnIndices:
    # get histogram with moment values from JPAC fit
    #TODO do not plot JPAC moments for imaginary parts
    HValsJpac = tuple(MomentValueAndTruth(*HPhys[qnIndex]) for HPhys in momentResultsJpac)
    histJpac = makeMomentHistogram(
      HVals      = HValsJpac,
      momentPart = "Re",
      histName   = "JPAC",
      histTitle  = "",
      binning    = cfg.massBinning,
      plotTruth  = False,
      plotUncert = True,
    )
    histJpacBand = histJpac.Clone(f"{histJpac.GetName()}_band")
    histJpac.SetLineColor(ROOT.kBlue + 1)
    histJpac.SetLineWidth(2)
    histJpacBand.SetFillColorAlpha(ROOT.kBlue + 1, 0.3)
    plotMoments1D(
      momentResults     = momentResultsPhys,
      qnIndex           = qnIndex,
      binning           = cfg.massBinning,
      normalizedMoments = cfg.normalizeMoments,
      momentResultsTrue = momentResultsClas,
      pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_",
      histTitle         = qnIndex.title,
      plotLegend        = True,
      legendLabels      = ("Moment", "CLAS"),
      plotTruthUncert   = True,
      truthColor        = ROOT.kGray + 1,
      histsToOverlay    = [
        (histJpac,     "HIST L", histJpac.GetName()),
        (histJpacBand,     "E3", ""),
      ],
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
    for indexKinBin, HVal in enumerate(H000):
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
  for binIndex, HMeas in enumerate(H000s[0]):
    HMeas.truth = histIntMeas.GetBinContent(binIndex + 1)  # set truth values to measured intensity
  plotMoments(
    HVals             = H000s[0],
    binning           = cfg.massBinning,
    normalizedMoments = cfg.normalizeMoments,
    momentLabel       = H000Index.label,
    pdfFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_intensity_",
    histTitle         = H000Index.title,
    legendLabels      = ("Measured Moment", "Measured Intensity"),
  )

  timer.stop("Total execution time")
  print(timer.summary)


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  setupPlotStyle()

  makeAllPlots(CFG)
