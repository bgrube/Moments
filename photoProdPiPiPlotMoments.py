#!/usr/bin/env python3
"""
This module plots the results of the moment analysis of unpolarized
and polarized pi+ pi- photoproduction data. The moment values are read
from files produced by the script `photoProdPiPiCalcMoments.py` that
calculates the moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import glob
from io import StringIO
import math
import numpy as np
import os
import pandas as pd

import ROOT
from wurlitzer import pipes, STDOUT

from photoProdPiPiCalcMoments import (
  AnalysisConfig,
  CFG_NIZAR,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI,
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
  outFileType = "pdf"
  # outFileType = "root"
  # load moments from files``
  momentIndices = MomentIndices(cfg.maxL)
  #TODO move this into AnalysisConfig?
  momentResultsFileBaseName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments"
  print(f"Reading measured moments from file '{momentResultsFileBaseName}_meas.pkl'")
  momentResultsMeas = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_meas.pkl")
  print(f"Reading physical moments from file '{momentResultsFileBaseName}_phys.pkl'")
  momentResultsPhys = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_phys.pkl")
  compareTo = "CLAS"
  # compareTo = "PWA"
  # compareTo = None
  if compareTo == "PWA":
    print(f"Reading PWA moments from file '{momentResultsFileBaseName}_pwa_SPD.pkl'")
  momentResultsCompare, momentResultsCompareLabel, momentResultsCompareColor = (
    (
      readMomentResultsClas(momentIndices, cfg.binVarMass),
      "CLAS",
      ROOT.kGray + 2,
    ) if compareTo == "CLAS" else
    (
      MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_pwa_SPD.pkl"),
      "PWA #it{S} #plus #it{P} #plus #it{D}",
      # MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_pwa_SPDF.pkl"),
      # "PWA #it{S} #plus #it{P} #plus #it{D} #plus #it{F}",
      ROOT.kBlue + 1,
    ) if compareTo == "PWA" else
    (
      None,
      "",
      ROOT.kBlack,
    )
  )
  momentResultsJpac        = readMomentResultsJpac(momentIndices, cfg.binVarMass)
  momentResultsJpacLabel   = "JPAC"
  overlayMomentResultsJpac = True
  # overlayMomentResultsJpac = False

  H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
  if momentResultsCompare is not None and not cfg.normalizeMoments:
    if compareTo == "CLAS":
      # scale CLAS and JPAC moments to match GlueX data
      scaleFactorClas = 1.0
      normalizeByIntegral = True  # if false comparison and JPAC moments are normalized to the maximum bin
      if normalizeByIntegral:
        # loop over mass bins and sum up H(0, 0) values
        H000Sum = H000SumComp = 0.0
        for HPhys, HComp in zip(momentResultsPhys, momentResultsCompare):
          H000Sum     += HPhys[H000Index].val.real
          H000SumComp += HComp[H000Index].val.real
        scaleFactorClas = H000Sum / H000SumComp
      else:
        normMassBinIndex = 36  # corresponds to m_pipi = 0.765 GeV; in this bin H(0, 0) is maximal in CLAS and GlueX data
        H000Value = momentResultsPhys[normMassBinIndex][H000Index].val.real
        scaleFactorClas = H000Value / momentResultsCompare[normMassBinIndex][H000Index].val.real
      print(f"Scale CLAS moments by factor {scaleFactorClas}")
      momentResultsCompare.scaleBy(scaleFactorClas)
      momentResultsJpac.scaleBy   (scaleFactorClas)  # use same factor as for CLAS moments
    elif compareTo == "PWA":
      # scale moments from PWA result
      momentResultsCompare.scaleBy(1 / (8 * math.pi))  #TODO unclear where this factor comes from; could it be the kappa term in the intensity function?

  if True:
    with timer.timeThis(f"Time to plot results from analysis of real data"):
      # plot moments in each mass bin
      chi2ValuesInMassBins: list[list[dict[str, tuple[float, float] | tuple[None, None]]]] = [[]] * len(momentResultsPhys)  # index: mass-bin index; index: moment index; key: "Re"/"Im" for real and imaginary parts of moments; value: chi2 value w.r.t. to given true values and corresponding n.d.f.
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        HMeas = momentResultsMeas   [massBinIndex]
        HComp = None if momentResultsCompare is None else momentResultsCompare[massBinIndex]
        binLabel = MomentCalculator.binLabel(HPhys)
        binTitle = MomentCalculator.binTitle(HPhys)
        # print(f"True moments for kinematic bin {title}:\n{HTruth}")
        print(f"Measured moments of real data for kinematic bin {binTitle}:\n{HMeas}")
        print(f"Physical moments of real data for kinematic bin {binTitle}:\n{HPhys}")
        chi2ValuesInMassBins[massBinIndex] = plotMomentsInBin(
          HData             = HPhys,
          normalizedMoments = cfg.normalizeMoments,
          HTruth            = HComp,
          outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_phys_{binLabel}_",
          legendLabels      = ("Moment", momentResultsCompareLabel),
          plotTruthUncert   = True,
          truthColor        = momentResultsCompareColor,
        )
        if cfg.plotMeasuredMoments:
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
      # plot chi^2/ndf of physical moments w.r.t. true values as a function of mass
      for momentIndex in range(momentResultsPhys[0].indices.momentIndexRange):
        for momentPart in ("Re", "Im"):
          _, ndf = chi2ValuesInMassBins[0][momentIndex][momentPart]  # assume that ndf is the same for all mass bins
          histChi2 = ROOT.TH1D(
            f"{cfg.outFileNamePrefix}_{cfg.massBinning.var.name}_chi2_H{momentIndex}_{momentPart}",
            f"#it{{L}}_{{max}} = {cfg.maxL};{cfg.massBinning.axisTitle};#it{{#chi}}^{{2}}/(ndf = {ndf})",
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
          histChi2.SetMaximum(20)
          # histChi2.SetMaximum(3)
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
        if False and qnIndex == H000Index:
          #TODO add this info to AnalysisConfig
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
          plotTruthUncert   = True,
          truthColor        = momentResultsCompareColor,
          histsToOverlay    = {} if histJpac is None else {  # dict: key = "Re" or "Im", list: tuple: (histogram, draw option, legend entry)
            "Re" : [
              (histJpac,     "HIST L", histJpac.GetName()),
              (histJpacBand, "E3",     ""),
            ],
          },
          # histsToOverlay    = {} if histPwaTotalIntensity is None else {  # dict: key = "Re" or "Im", list: tuple: (histogram, draw option, legend entry)
          #   "Re" : [
          #     (histPwaTotalIntensity, "HIST", histPwaTotalIntensity.GetName()),
          #   ],
          # },
        )
        if cfg.plotMeasuredMoments:
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
      # plot chi^2/ndf of physical moments w.r.t. true values as a function of mass
      _, ndf = chi2ValuesForMoments[H000Index]["Re"]  # assume that ndf is the same for all moments; take value from Re[H_0(0, 0)]
      for momentIndex in range(momentResultsPhys[0].indices.momentIndexRange):
        for momentPart in ("Re", "Im"):
          # get chi^2 values for this momentIndex and momentPart
          chi2Values: dict[QnMomentIndex, tuple[float, float] | tuple[None, None]] = {qnIndex : value[momentPart] for qnIndex, value in chi2ValuesForMoments.items() if qnIndex.momentIndex == momentIndex}
          histChi2 = ROOT.TH1D(
            f"{cfg.outFileNamePrefix}_chi2_H{momentIndex}_{momentPart}",
            f"#it{{L}}_{{max}} = {cfg.maxL};;#it{{#chi}}^{{2}}/(ndf = {ndf})",
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
          histChi2.SetMaximum(10)
          # histChi2.SetMaximum(3)
          histChi2.Draw("HIST")
          # add line at nominal chi2/ndf value to guide the eye
          line = ROOT.TLine()
          line.SetLineColor(ROOT.kGray + 1)
          line.SetLineStyle(ROOT.kDashed)
          line.DrawLine(0, 1, len(chi2Values), 1)
          canv.SaveAs(f"{cfg.outFileDirName}/{histChi2.GetName()}.pdf")

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
      histRatio.SetMaximum(0.4)
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
        outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_meas_intensity_",
        histTitle         = H000Index.title,
        legendLabels      = ("Measured Moment", "Measured Intensity"),
      )

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
            polarization   = 0.0 if HPhys.indices.polarized else None,
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
      momentResultsAccPsMeas = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_accPs_meas.pkl")
      print(f"Reading physical moments for accepted phase-space MC from file '{momentResultsFileBaseName}_accPs_phys.pkl'")
      momentResultsAccPsPhys = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_accPs_phys.pkl")
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
        dataPsGen = ROOT.RDataFrame(cfg.treeName, cfg.psGenFileName)
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
            # plot physical moments; they should match the true moments exactly, i.e. all 0 except H_0(0, 0), modulo tiny numerical effects
            plotMomentsInBin(
              HData             = HPhys,
              normalizedMoments = cfg.normalizeMoments,
              HTruth            = HTruthPs,
              outFileNamePrefix = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_{binLabel}_accPsCorr_"
            )


if __name__ == "__main__":
  cfg = deepcopy(CFG_UNPOLARIZED_PIPI)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPP)  # perform analysis of unpolarized pi+ p data
  # cfg = deepcopy(CFG_NIZAR)  # perform analysis of Nizar's polarized eta pi0 data

  for maxL in (2, 4, 5, 6, 8, 10, 12, 14):
  # for maxL in (8, ):
    print(f"Plotting moments for L_max = {maxL}")
    cfg.maxL = maxL
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
      makeAllPlots(cfg, timer)
      timer.stop("Total execution time")
      print(timer.summary)
