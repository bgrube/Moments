#!/usr/bin/env python3
"""
This module plots the results of the moment analysis of unpolarized
and polarized pi+ pi- photoproduction data. The moment values are read
from files produced by the script `photoProdCalcMoments.py` that
calculates the moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
from enum import Enum, auto
import functools
import glob
from io import StringIO
import math
import numpy as np
import os
import pandas as pd

import ROOT
from wurlitzer import pipes, STDOUT

from AnalysisConfig import (
  AnalysisConfig,
  CFG_KEVIN,
  CFG_NIZAR,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_JPAC,
  CFG_UNPOLARIZED_PIPI_PWA,
  CFG_UNPOLARIZED_PIPP,
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
  HistAxisBinning,
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
          # convert moment values to complex numbers with zero imaginary part
          momentDf["moment"] = momentDf["moment"].astype(complex)
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
      # cannot use momentDf[mask][["moment", "uncertPlus", "uncertMinus"]].values because conversion to NumPy array would upcast all columns to complex
      # for some unknown reason, momentDf[mask].iloc[0] also converts all columns to complex, although it should not
      dfMassBin = momentDf[mask]  # mask selects exactly 1 row
      moment      = dfMassBin["moment"     ].iloc[0]
      uncertPlus  = dfMassBin["uncertPlus" ].iloc[0]
      uncertMinus = dfMassBin["uncertMinus"].iloc[0]
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
        skiprows = 1,       # first row with column names
        names    = ["mass", "moment", "uncert"],
      )
      # scale moment values and their uncertainties by 1 / sqrt(2L + 1) to match normalization used in this analysis
      momentDf[["moment", "uncert"]] /= np.sqrt(2 * qnMomentIndex.L + 1)
      # convert moment values to complex numbers with zero imaginary part
      momentDf["moment"] = momentDf["moment"].astype(complex)
      momentDfs[qnMomentIndex] = momentDf
    except FileNotFoundError as e:
        print(f"Warning: file '{dataFileName}' not found. Skipping moment {qnMomentIndex.label}.")
  # ensure that mass bins are the same in all dataframes
  dfs = list(momentDfs.values())
  massColumn = dfs[0]["mass"]
  for df in dfs[1:]:
    assert df["mass"].equals(massColumn), f"Mass bins in dataframes differ:\n{df['mass']}\nvs.\n{massColumn}"
  # convert dataframes to MomentResultsKinematicBinning
  momentResults = MomentResultsKinematicBinning([])
  for massBinCenter in massColumn:
    # loop over momentDfs and extract moment values for the given mass bin
    momentValues: list[MomentValue] = []
    for qnMomentIndex, momentDf in momentDfs.items():
      mask = momentDf["mass"] == massBinCenter
      dfMassBin = momentDf[mask]  # mask selects exactly 1 row
      moment = dfMassBin["moment"].iloc[0]
      uncert = dfMassBin["uncert"].iloc[0]
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
  return momentResults


class ComparisonMomentsType(Enum):
  """Enumerates moment data to compare to"""
  CLAS = auto()
  JPAC = auto()
  PWA  = auto()

def makeAllPlots(
  cfg:                         AnalysisConfig,
  timer:                       Utilities.Timer              = Utilities.Timer(),
  scaleFactorPhysicalMoments:  float                        = 1.0,    # optional scale factor for physical moments; can be used to convert number of events to cross section
  compareTo:                   ComparisonMomentsType | None = None,   # `None` means no comparison moments are plotted
  normalizeComparisonMoments:  bool                         = False,  # whether to scale comparison moments
  plotComparisonMomentsUncert: bool                         = False,  # whether to plot uncertainties of comparison moments
  outFileType:                 str                          = "pdf",  # "pdf" or "root"
  yAxisUnit:                   str                          = "",  # optional unit for moment values
) -> None:
  """Generates all plots for the given analysis configuration and writes them to output files"""
  # load moments from files
  momentIndices = MomentIndices(cfg.maxL)
  #TODO move this into AnalysisConfig?
  momentResultsFileBaseName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments"
  momentResultsMeas = None
  if os.path.exists(f"{momentResultsFileBaseName}_meas.pkl"):
    print(f"Reading measured moments from file '{momentResultsFileBaseName}_meas.pkl'")
    momentResultsMeas = MomentResultsKinematicBinning.loadPickle(f"{momentResultsFileBaseName}_meas.pkl")
  print(f"Reading physical moments from file '{momentResultsFileBaseName}_phys.pkl'")
  momentResultsPhys = MomentResultsKinematicBinning.loadPickle(f"{momentResultsFileBaseName}_phys.pkl")
  momentResultsPhys.scaleBy(scaleFactorPhysicalMoments)  # apply optional scale factor to physical moments
  if compareTo == ComparisonMomentsType.PWA:
    print(f"Reading PWA moments from file '{momentResultsFileBaseName}_pwa_SPD.pkl'")
  momentResultsCompare, momentResultsCompareLabel, momentResultsCompareColor = (
    (
      MomentResultsKinematicBinning.loadPickle(f"{momentResultsFileBaseName}_pwa_SPD.pkl"),
      "PWA #it{S} #plus #it{P} #plus #it{D}",
      # MomentResultsKinematicBinning.loadPickle(f"{momentResultsFileBaseName}_pwa_SPDF.pkl"),
      # "PWA #it{S} #plus #it{P} #plus #it{D} #plus #it{F}",
      ROOT.kBlue + 1,
    ) if compareTo == ComparisonMomentsType.PWA
    else
    (
      readMomentResultsClas(momentIndices, cfg.binVarMass),
      "CLAS",
      ROOT.kGray + 2,
    ) if compareTo == ComparisonMomentsType.CLAS
    else
    (
      readMomentResultsJpac(momentIndices, cfg.binVarMass),
      "JPAC",
      ROOT.kBlue + 1,
    ) if compareTo == ComparisonMomentsType.JPAC
    else
    (
      None,
      "",
      ROOT.kBlack,
    )
  )
  if compareTo == ComparisonMomentsType.JPAC and momentResultsCompare is not None:
    print(f"Writing JPAC moments to '{momentResultsFileBaseName}_JPAC.pkl'")
    momentResultsCompare.savePickle(f"{momentResultsFileBaseName}_JPAC.pkl")  # save copy of JPAC moments
  momentResultsJpac = None
  if momentResultsCompare is not None and not cfg.normalizeMoments:
    if compareTo == ComparisonMomentsType.CLAS:
      momentResultsJpac = readMomentResultsJpac(momentIndices, cfg.binVarMass)
      if normalizeComparisonMoments:
        normalizationFactorClas = momentResultsCompare.normalizeTo(
          momentResultsPhys,
          normBinIndex = None,  # normalize to integral over mass bins
          # normBinIndex = 36,    # normalize to mass bin, where H_0(0, 0) is maximal in CLAS and GlueX data; corresponds to m_pipi = 0.765 GeV
        )
        momentResultsJpac.scaleBy(normalizationFactorClas)  # use same factor as for CLAS moments
        print(f"Scaled CLAS and JPAC moments by factor {normalizationFactorClas}")
    if compareTo == ComparisonMomentsType.JPAC:
      if normalizeComparisonMoments:
        normalizationFactorJpac = momentResultsCompare.normalizeTo(
          momentResultsPhys,
          normBinIndex = None,  # normalize to integral over mass bins
          # normBinIndex = 36,    # normalize to mass bin, where H_0(0, 0) is maximal in CLAS and GlueX data; corresponds to m_pipi = 0.765 GeV
        )
        print(f"Scaled JPAC moments by factor {normalizationFactorJpac}")
    elif compareTo == ComparisonMomentsType.PWA:
      # scale moments from PWA result
      momentResultsCompare.scaleBy(1 / (8 * math.pi))  #TODO unclear where this factor comes from; could it be the kappa term in the intensity function?

  H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
  if True:
    with timer.timeThis(f"Time to plot results from analysis of real data"):
      # plot moments in each mass bin
      chi2ValuesInMassBins: list[list[dict[str, tuple[float, float] | tuple[None, None]]]] = [[]] * len(momentResultsPhys)  # index: mass-bin index; index: moment index; key: "Re"/"Im" for real and imaginary parts of moments; value: chi2 value w.r.t. to given true values and corresponding n.d.f.
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        binLabel = MomentCalculator.binLabel(HPhys)
        binTitle = MomentCalculator.binTitle(HPhys)
        HMeas = None if momentResultsMeas is None else momentResultsMeas[massBinIndex]
        if HMeas is not None:
          print(f"Measured moments of real data for kinematic bin {binTitle}:\n{HMeas}")
        print(f"Physical moments of real data for kinematic bin {binTitle}:\n{HPhys}")
        HComp = None if momentResultsCompare is None else momentResultsCompare[massBinIndex]
        if cfg.plotMomentsInBins:
          chi2ValuesInMassBins[massBinIndex] = plotMomentsInBin(
            HData             = HPhys,
            normalizedMoments = cfg.normalizeMoments,
            HTruth            = HComp,
            outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_phys_{binLabel}_",
            legendLabels      = ("Moment", momentResultsCompareLabel),
            plotTruthUncert   = plotComparisonMomentsUncert,
            truthColor        = momentResultsCompareColor,
            yAxisUnit         = yAxisUnit,
          )
          if cfg.plotMeasuredMoments and HMeas is not None:
            plotMomentsInBin(
              HData             = HMeas,
              normalizedMoments = cfg.normalizeMoments,
              HTruth            = None,
              outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_{binLabel}_",
              plotLegend        = False,
            )
        if cfg.plotCovarianceMatrices:
          #TODO also plot correlation matrices
          plotMomentsCovMatrices(
            HData             = HPhys,
            pdfFileNamePrefix = f"{cfg.outFileDirName}/covMatrix_{binLabel}_",
            axisTitles        = ("Physical Moment Index", "Physical Moment Index"),
            plotTitle         = f"{binLabel}: ",
          )
        if cfg.nmbBootstrapSamples > 0:
          plotMomentsBootstrapDistributions1D(
            HData             = HPhys,
            HTruth            = HComp,
            outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_",
            histTitle         = binTitle,
            HTruthLabel       = momentResultsCompareLabel,
          )
          plotMomentsBootstrapDistributions2D(
            HData             = HPhys,
            HTruth            = HComp,
            outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_",
            histTitle         = binTitle,
            HTruthLabel       = momentResultsCompareLabel,
          )
          plotMomentsBootstrapDiffInBin(
            HData             = HPhys,
            outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_",
            graphTitle        = binTitle,
          )
      if compareTo is not None and cfg.plotMomentsInBins:
        #TODO move to separate function?
        # plot average chi^2/ndf of all physical moments w.r.t. true values as a function of mass
        for momentIndex in range(momentResultsPhys[0].indices.momentIndexRange):
          for momentPart in ("Re", "Im"):
            _, ndf = chi2ValuesInMassBins[0][momentIndex][momentPart]  # assume that ndf is the same for all mass bins
            histChi2 = ROOT.TH1D(
              f"{cfg.outFileNamePrefix}_{cfg.massBinning.var.name}_chi2_H{momentIndex}_{momentPart}",
              f"#LT#it{{#chi}}^{{2}}/ndf#GT for all {momentPart}[#it{{H}}_{{{momentIndex}}}], #it{{L}}_{{max}} = {cfg.maxL};{cfg.massBinning.axisTitle};#it{{#chi}}^{{2}}/(ndf = {ndf})",
              *cfg.massBinning.astuple
            )
            for massBinIndex in range(len(chi2ValuesInMassBins)):
              chi2, ndf = chi2ValuesInMassBins[massBinIndex][momentIndex][momentPart]
              if chi2 is None or ndf is None:
                continue
              massBinCenter = momentResultsPhys.binCenters[massBinIndex][cfg.massBinning.var]
              histChi2.SetBinContent(histChi2.FindBin(massBinCenter), chi2 / ndf)
            canv = ROOT.TCanvas()
            histChi2.SetLineColor(ROOT.kBlue + 1)
            histChi2.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
            # histChi2.SetMaximum(30)
            histChi2.SetMaximum(5)
            histChi2.Draw("HIST")
            # add line at nominal chi2/ndf value to guide the eye
            line = ROOT.TLine()
            line.SetLineColor(ROOT.kGray + 1)
            line.SetLineStyle(ROOT.kDashed)
            line.DrawLine(cfg.massBinning.minVal, 1, cfg.massBinning.maxVal, 1)
            canv.SaveAs(f"{cfg.outFileDirName}/{histChi2.GetName()}.pdf")

      # plot mass dependences of all moments
      chi2ValuesForMoments: dict[QnMomentIndex, dict[str, tuple[float, float] | tuple[None, None]]] = {}  # key: quantum-number index of moment; key: "Re"/"Im" for real and imaginary parts of moments; value: chi2 value w.r.t. to given true values and corresponding n.d.f.
      for qnIndex in momentResultsPhys[0].indices.qnIndices:
        # get histogram with moment values from JPAC fit
        histsJpac:     dict[str, ROOT.TH1D] = {}
        histsBandJpac: dict[str, ROOT.TH1D] = {}
        if momentResultsJpac is not None:
          HValsJpac = tuple(MomentValueAndTruth(*HPhys[qnIndex]) for HPhys in momentResultsJpac)
          histsJpac = {momentPart : makeMomentHistogram(
            HVals      = HValsJpac,
            momentPart = momentPart,
            histName   = "JPAC",
            histTitle  = "",
            binning    = cfg.massBinning,
            plotTruth  = False,
            plotUncert = True,
          ) for momentPart in ("Re", "Im")}
          for momentPart, hist in histsJpac.items():
            hist.SetLineColor(ROOT.kBlue + 1)
            hist.SetLineWidth(2)
            histsBandJpac[momentPart] = hist.Clone(f"{hist.GetName()}_band")
            histsBandJpac[momentPart].SetFillColorAlpha(ROOT.kBlue + 1, 0.3)
        histPwaTotalIntensity = None
        if False and qnIndex == H000Index:
          plotFile = ROOT.TFile.Open("./dataPhotoProdPiPiUnpol/PWA_S_P_D/pwa_plots_weight1.root", "READ")
          # plotFile = ROOT.TFile.Open("./dataPhotoProdPiPiPol/PWA_S_P_D/pwa_plots_SPD.root", "READ")
          histPwaTotalIntensity = convertGraphToHist(
            graph     = plotFile.Get("Total"),
            binning   = cfg.massBinning.astuple,
            histName  = "Total Intensity",
            histTitle = "",
          )
          histPwaTotalIntensity.SetLineColor(ROOT.kGreen + 2)
        chi2ValuesForMoments[qnIndex] = plotMoments1D(
          momentResults     = momentResultsPhys,
          qnIndex           = qnIndex,
          binning           = cfg.massBinning,
          normalizedMoments = cfg.normalizeMoments,
          momentResultsTrue = momentResultsCompare,
          outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_phys_",
          outFileType       = outFileType,
          histTitle         = qnIndex.title,
          plotLegend        = True,
          legendLabels      = ("Moment", momentResultsCompareLabel),
          plotTruthUncert   = plotComparisonMomentsUncert,
          truthColor        = momentResultsCompareColor,
          yAxisUnit         = yAxisUnit,
          histsToOverlay    = {  # dict: key = "Re" or "Im", list: tuple: (histogram, draw option, legend entry)
            "Re" : [
              (histsJpac["Re"],     "HIST L", histsJpac["Re"].GetName()),
              (histsBandJpac["Re"], "E3",     ""),
            ],
            "Im" : [
              (histsJpac["Im"],     "HIST L", histsJpac["Im"].GetName()),
              (histsBandJpac["Im"], "E3",     ""),
            ],
          } if histsJpac else {},
          # histsToOverlay    = {} if histPwaTotalIntensity is None else {  # dict: key = "Re" or "Im", list: tuple: (histogram, draw option, legend entry)
          #   "Re" : [
          #     (histPwaTotalIntensity, "HIST", histPwaTotalIntensity.GetName()),
          #   ],
          # },
        )
        if cfg.plotMeasuredMoments and momentResultsMeas is not None:
          plotMoments1D(
            momentResults     = momentResultsMeas,
            qnIndex           = qnIndex,
            binning           = cfg.massBinning,
            normalizedMoments = cfg.normalizeMoments,
            momentResultsTrue = None,
            outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_",
            histTitle         = qnIndex.title,
            plotLegend        = False,
          )
      if compareTo is not None:
        # plot chi^2/ndf of each physical moments w.r.t. true value, averaged over the mass bins
        #TODO move to separate function?
        _, ndf = chi2ValuesForMoments[H000Index]["Re"]  # assume that ndf is the same for all moments; take value from Re[H_0(0, 0)]
        for momentIndex in range(momentResultsPhys[0].indices.momentIndexRange):
          for momentPart in ("Re", "Im"):
            # get chi^2 values for this momentIndex and momentPart
            chi2Values: dict[QnMomentIndex, tuple[float, float] | tuple[None, None]] = {qnIndex : value[momentPart] for qnIndex, value in chi2ValuesForMoments.items() if qnIndex.momentIndex == momentIndex}
            histChi2 = ROOT.TH1D(
              f"{cfg.outFileNamePrefix}_chi2_H{momentIndex}_{momentPart}",
              f"{momentPart}[#it{{H}}_{{{momentIndex}}}]: #LT#it{{#chi}}^{{2}}/ndf#GT for all {cfg.binVarMass.label}, #it{{L}}_{{max}} = {cfg.maxL};;#it{{#chi}}^{{2}}/(ndf = {ndf})",
              len(chi2Values), 0, len(chi2Values)
            )
            for binIndex, (qnIndex, (chi2, ndf)) in enumerate(chi2Values.items()):
              histChi2.GetXaxis().SetBinLabel(binIndex + 1, qnIndex.title)  # categorical x axis with moment labels
              if chi2 is None or ndf is None:
                continue
              histChi2.SetBinContent(binIndex + 1, chi2 / ndf)
            histChi2.GetXaxis().LabelsOption("V")
            canv = ROOT.TCanvas()
            histChi2.SetLineColor(ROOT.kBlue + 1)
            histChi2.SetFillColorAlpha(ROOT.kBlue + 1, 0.1)
            # histChi2.SetMaximum(10)
            histChi2.SetMaximum(3)
            histChi2.Draw("HIST")
            # add line at nominal chi2/ndf value to guide the eye
            line = ROOT.TLine()
            line.SetLineColor(ROOT.kGray + 1)
            line.SetLineStyle(ROOT.kDashed)
            line.DrawLine(0, 1, len(chi2Values), 1)
            canv.SaveAs(f"{cfg.outFileDirName}/{histChi2.GetName()}.pdf")

      # plot ratio of measured and physical value for Re[H_0(0, 0)]; estimates efficiency
      if momentResultsMeas is not None:
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
            if (cfg.massBinning.var not in HVal.binCenters.keys()):
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
        histRatio.SetTitle(f"#it{{L}}_{{max}} = {cfg.maxL};{cfg.massBinning.axisTitle};" + "#it{H}_{0}^{meas}(0, 0) / #it{H}_{0}^{phys}(0, 0)")
        # histRatio.SetMaximum(0.15)
        histRatio.Draw("PEX0")
        canv.SaveAs(f"{cfg.outFileDirName}/{histRatio.GetName()}.pdf")

      if not cfg.dataFileName or not os.path.exists(cfg.dataFileName):
        print(f"Warning: cannot find data file '{cfg.dataFileName=}'. Cannot overlay H_0^meas(0, 0) and measured distribution.")
      elif cfg.plotMeasuredMoments:
        print(f"Overlaying measured intensity distribution from file '{cfg.dataFileName}'")
        # overlay H_0^meas(0, 0) and measured intensity distribution; must be identical
        histIntMeas = cfg.loadData(AnalysisConfig.DataType.REAL_DATA).Histo1D(
          ROOT.RDF.TH1DModel("intensity_meas", f";{cfg.massBinning.axisTitle};Events", *cfg.massBinning.astuple), "mass", "eventWeight"
        ).GetValue()
        for binIndex, H000Meas in enumerate(H000s[0]):
          H000Meas.truth = histIntMeas.GetBinContent(binIndex + 1)  # set truth values to measured intensity
        plotMoments(
          HVals             = H000s[0],
          binning           = cfg.massBinning,
          normalizedMoments = cfg.normalizeMoments,
          momentLabel       = H000Index.label,
          outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_intensity_",
          histTitle         = H000Index.title,
          legendLabels      = ("Measured Moment", "Measured Intensity"),
        )

  if cfg.plotAngularDistributions:
    with timer.timeThis(f"Time to plot angular distributions"):
      print("Plotting angular distributions")
      # load all signal and phase-space data
      data      = cfg.loadData(AnalysisConfig.DataType.REAL_DATA)
      dataPsAcc = cfg.loadData(AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE)
      dataPsGen = cfg.loadData(AnalysisConfig.DataType.GENERATED_PHASE_SPACE)
      # plot total angular distribution
      plotAngularDistr(
        dataPsAcc         = dataPsAcc,
        dataPsGen         = dataPsGen,
        dataSignalAcc     = data,
        dataSignalGen     = None,
        outFileNamePrefix = f"{cfg.outFileDirName}/angDistr_total_",
      )
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        # load data for mass bin
        massBinFilter  = cfg.massBinning.binFilter(massBinIndex)
        dataInBin      = data.Filter     (massBinFilter)
        dataPsAccInBin = dataPsAcc.Filter(massBinFilter)
        # dataPsGenInBin = dataPsGen.Filter(massBinFilter)
        # plot angular distributions for mass bin
        if cfg.plotAngularDistributions:
          plotAngularDistr(
            dataPsAcc         = dataPsAccInBin,
            dataSignalAcc     = dataInBin,
            dataPsGen         = None,
            # dataPsGen         = dataPsGenInBin,
            dataSignalGen     = None,
            outFileNamePrefix = f"{cfg.outFileDirName}/angDistr_{MomentCalculator.binLabel(HPhys)}_",
            nmbBins2D         = 20,
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
            polarization   = "beamPol" if HPhys.indices.polarized else None,
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
      print(f"Reading measured moments for accepted phase-space MC from file '{momentResultsFileBaseName}_accPs_meas.pkl'")
      momentResultsAccPsMeas = MomentResultsKinematicBinning.loadPickle(f"{momentResultsFileBaseName}_accPs_meas.pkl")
      print(f"Reading physical moments for accepted phase-space MC from file '{momentResultsFileBaseName}_accPs_phys.pkl'")
      momentResultsAccPsPhys = MomentResultsKinematicBinning.loadPickle(f"{momentResultsFileBaseName}_accPs_phys.pkl")
    except FileNotFoundError as e:
      print(f"Warning: File not found. Cannot plot accepted phase-space moments. {e}")
    else:
      with timer.timeThis(f"Time to plot accepted phase-space moments"):
        if cfg.normalizeMoments:
          momentResultsAccPsMeas.normalize()
        # plot mass dependences of all phase-space moments
        for qnIndex in momentIndices.qnIndices:
          HVals = tuple(MomentValueAndTruth(*momentResultInBin[qnIndex]) for momentResultInBin in momentResultsAccPsMeas)
          plotMoments(
            HVals             = HVals,
            binning           = cfg.massBinning,
            normalizedMoments = cfg.normalizeMoments,
            momentLabel       = qnIndex.label,
            outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{cfg.massBinning.var.name}_accPs_",
            histTitle         = qnIndex.title,
            plotLegend        = False,
          )
        # plot accepted phase-space moments in each mass bin
        dataPsGen = cfg.loadData(AnalysisConfig.DataType.GENERATED_PHASE_SPACE)
        for massBinIndex, HPhys in enumerate(momentResultsAccPsPhys):
          binLabel = MomentCalculator.binLabel(HPhys)
          binTitle = MomentCalculator.binTitle(HPhys)
          HMeas = momentResultsAccPsMeas[massBinIndex]
          print(f"Measured moments of accepted phase-space data for kinematic bin {binTitle}:\n{HMeas}")
          print(f"Physical moments of accepted phase-space data for kinematic bin {binTitle}:\n{HPhys}")
          # plot measured moments
          plotMomentsInBin(
            HData             = HMeas,
            normalizedMoments = cfg.normalizeMoments,
            HTruth            = None,
            outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_accPs_",
            plotLegend        = False,
            forceYaxisRange   = (-0.1, +0.1) if cfg.normalizeMoments else (None, None),
          )
          if False:
            # construct true moments for phase-space data
            nmbPsGenEvents = None
            if not cfg.normalizeMoments:
              massBinFilter  = cfg.massBinning.binFilter(massBinIndex)
              dataPsGenInBin = dataPsGen.Filter(massBinFilter)
              nmbPsGenEvents = dataPsGenInBin.Count().GetValue()
            HTruthPs = MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
            HTruthPs._valsFlatIndex[momentIndices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1 if cfg.normalizeMoments else nmbPsGenEvents  # ... except for H_0(0, 0)
            HTruthPs.valid = True
            # plot physical moments; they should match the true moments exactly, i.e. all 0 except H_0(0, 0), modulo tiny numerical effects
            plotMomentsInBin(
              HData             = HPhys,
              normalizedMoments = cfg.normalizeMoments,
              HTruth            = HTruthPs,
              outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_accPsCorr_"
            )


if __name__ == "__main__":
  # compareTo = None
  compareTo = ComparisonMomentsType.CLAS
  # compareTo = ComparisonMomentsType.JPAC
  cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # compareTo = ComparisonMomentsType.PWA
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  # compareTo = ComparisonMomentsType.JPAC
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPP)  # perform analysis of unpolarized pi+ p data
  # cfg = deepcopy(CFG_NIZAR)  # perform analysis of Nizar's polarized eta pi0 data
  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarizedK- K_S Delta++ data
  plotCompareUncert = True
  # plotCompareUncert = False
  scaleFactorPhysicalMoments = 1.0
  # scaleFactorPhysicalMoments = 1.0 / (0.01 * 0.1 * 0.1305 * 1e6)  # [ub / GeV^3]; from 1 / (10 MeV * 0.1 GeV^2 * L) with L(Fall 2018) = 0.1305 pb^{-1}
  normalizeComparisonMoments = True  # whether to scale comparison moments to GlueX moments
  # normalizeComparisonMoments = False
  yAxisUnit = ""
  # yAxisUnit = " [#mub/GeV^{3}]"

  tBinLabels = (
    # "tbin_0.1_0.2",
    # "tbin_0.1_0.2.Hf.pi+",
    # "tbin_0.1_0.2.trackDistFdc",
    # "tbin_0.2_0.3",
    # "tbin_0.1_0.5",
    "tbin_0.4_0.5",
  )
  beamPolLabels = (
    # "PARA_0",
    # "PARA_0", "PARA_135", "PERP_45", "PERP_90",
    # "allOrient",
    "Unpol",
  )
  maxLs = (
    4,
    # 5,
    6,
    # 7,
    8,
  )
  # cfg.nmbBootstrapSamples = 10000  # number of bootstrap samples used for uncertainty estimation
  # cfg.massBinning         = HistAxisBinning(nmbBins = 10, minVal = 0.75, maxVal = 0.85)  # fit only rho region
  # cfg.polarization = None  # treat data as unpolarized

  outFileDirBaseNameCommon = cfg.outFileDirBaseName
  # outFileDirBaseNameCommon = f"{cfg.outFileDirBaseName}.ideal"
  for tBinLabel in tBinLabels:
    for beamPolLabel in beamPolLabels:
      # cfg.dataFileName       = f"./dataPhotoProdPiPiPol/{tBinLabel}/data_flat_{beamPolLabel}.root"
      # cfg.psAccFileName      = f"./dataPhotoProdPiPiPol/{tBinLabel}/phaseSpace_acc_flat_{beamPolLabel}.root"
      # cfg.psGenFileName      = f"./dataPhotoProdPiPiPol/{tBinLabel}/phaseSpace_gen_flat_{beamPolLabel}.root"
      cfg.outFileDirBaseName = f"{outFileDirBaseNameCommon}.{tBinLabel}/{beamPolLabel}"
      for maxL in maxLs:
        print(f"Plotting moments for t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
        cfg.maxL = maxL
        cfg.init()
        thisSourceFileName = os.path.basename(__file__)
        logFileName = f"{cfg.outFileDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
        print(f"Writing output to log file '{logFileName}'")
        with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
          Utilities.printGitInfo()
          timer = Utilities.Timer()
          ROOT.gROOT.SetBatch(True)
          setupPlotStyle()
          print(f"Using configuration:\n{cfg}")
          timer.start("Total execution time")
          makeAllPlots(
            cfg                         = cfg,
            timer                       = timer,
            scaleFactorPhysicalMoments  = scaleFactorPhysicalMoments,
            compareTo                   = compareTo,
            normalizeComparisonMoments  = normalizeComparisonMoments,
            plotComparisonMomentsUncert = plotCompareUncert,
            yAxisUnit                   = yAxisUnit,
          )
          timer.stop("Total execution time")
          print(timer.summary)
