#!/usr/bin/env python3
# performs moments analysis for pi+ pi- real-data events in CLAS energy range


from __future__ import annotations

import functools
import glob
from io import StringIO
import numpy as np
import pandas as pd
import threadpoolctl

import ROOT

from MomentCalculator import (
  binLabel,
  binTitle,
  constructMomentResultFrom,
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
  MomentResult,
  MomentResultsKinematicBinning,
  MomentValue,
  QnMomentIndex,
)
from PlottingUtilities import (
  HistAxisBinning,
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



if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
  with threadController.limit(limits = 4):
    print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of analysis
    treeName                 = "PiPi"
    dataFileName             = f"./dataPhotoProdPiPiUnpol/data_flat.root"
    psAccFileName            = f"./dataPhotoProdPiPiUnpol/phaseSpace_acc_flat.root"
    psGenFileName            = f"./dataPhotoProdPiPiUnpol/phaseSpace_gen_flat.root"
    maxL                     = 20  # define maximum L quantum number of moments
    outFileDirName           = Utilities.makeDirPath(f"./plotsPhotoProdPiPiUnpol.maxL_{maxL}")
    normalizeMoments         = False
    nmbBootstrapSamples      = 0
    # nmbBootstrapSamples      = 10000
    # plotAngularDistributions = True
    plotAngularDistributions = False
    # plotAccIntegralMatrices  = True
    plotAccIntegralMatrices  = False
    calcAccPsMoments         = True
    # calcAccPsMoments         = False
    # limitNmbPsAccEvents      = 0  # process all data
    limitNmbPsAccEvents      = 1000
    binVarMass               = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}", unit = "GeV/#it{c}^{2}", nmbDigits = 3)
    massBinning              = HistAxisBinning(nmbBins = 100, minVal = 0.4, maxVal = 1.4, _var = binVarMass)  # same binning as used by CLAS
    # massBinning              = HistAxisBinning(nmbBins = 1, minVal = 1.25, maxVal = 1.29, _var = binVarMass)  # f_2(1270) region
    nmbOpenMpThreads         = ROOT.getNmbOpenMpThreads()

    namePrefix = "norm" if normalizeMoments else "unnorm"

    # setup MomentCalculators for all mass bins
    momentIndices = MomentIndices(maxL)
    momentsInBins:  list[MomentCalculator] = []
    nmbPsGenEvents: list[int]              = []
    assert len(massBinning) > 0, f"Need at least one mass bin, but found {len(massBinning)}"
    with timer.timeThis(f"Time to load data and setup MomentCalculators for {len(massBinning)} bins"):
      for massBinIndex, massBinCenter in enumerate(massBinning):
        massBinRange = massBinning.binValueRange(massBinIndex)
        print(f"Preparing {binVarMass.name} bin [{massBinIndex} of {len(massBinning)}] at {massBinCenter} {binVarMass.unit} with range {massBinRange} {binVarMass.unit}")

        # load data for mass bin
        binMassRangeFilter = f"(({massBinRange[0]} < {binVarMass.name}) && ({binVarMass.name} < {massBinRange[1]}))"
        print(f"Loading real data from tree '{treeName}' in file '{dataFileName}' and applying filter {binMassRangeFilter}")
        dataInBin = ROOT.RDataFrame(treeName, dataFileName).Filter(binMassRangeFilter)
        print(f"Loaded {dataInBin.Count().GetValue()} data events; {dataInBin.Sum('eventWeight').GetValue()} background subtracted events")
        print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
        dataPsAccInBin = ROOT.RDataFrame(treeName, psAccFileName).Range(limitNmbPsAccEvents).Filter(binMassRangeFilter)
        print(f"Loading generated phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
        dataPsGenInBin = ROOT.RDataFrame(treeName, psGenFileName).Filter(binMassRangeFilter)
        nmbPsGenEvents.append(dataPsGenInBin.Count().GetValue())
        nmbPsAccEvents = dataPsAccInBin.Count().GetValue()
        print(f"Loaded phase-space events: number generated = {nmbPsGenEvents[-1]}; "
              f"number accepted = {nmbPsAccEvents}, "
              f" -> efficiency = {nmbPsAccEvents / nmbPsGenEvents[-1]:.3f}")

        # setup moment calculators for data
        dataSet = DataSet(
          data           = dataInBin,
          phaseSpaceData = dataPsAccInBin,
          nmbGenEvents   = nmbPsGenEvents[-1],
          polarization   = None,
        )
        momentsInBins.append(
          MomentCalculator(
            indices              = momentIndices,
            dataSet              = dataSet,
            binCenters           = {binVarMass : massBinCenter},
            integralFileBaseName = f"{outFileDirName}/integralMatrix",
          )
        )

        # plot angular distributions for mass bin
        if plotAngularDistributions:
          plotAngularDistr(
            dataPsAcc         = dataPsAccInBin,
            dataPsGen         = dataPsGenInBin,
            dataSignalAcc     = dataInBin,
            dataSignalGen     = None,
            pdfFileNamePrefix = f"{outFileDirName}/angDistr_{binLabel(momentsInBins[-1])}_"
          )
    moments = MomentCalculatorsKinematicBinning(momentsInBins)

    # calculate and plot integral matrix for all mass bins
    with timer.timeThis(f"Time to calculate integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating acceptance integral matrices for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateIntegralMatrices(forceCalculation = True)
      print(f"Acceptance integral matrix for first bin at {massBinning[0]} {binVarMass.unit}:\n{moments[0].integralMatrix}")
      eigenVals, _ = moments[0].integralMatrix.eigenDecomp
      print(f"Sorted eigenvalues of acceptance integral matrix for first bin at {massBinning[0]} {binVarMass.unit}:\n{np.sort(eigenVals)}")
      # plot acceptance integral matrices for all kinematic bins
      if plotAccIntegralMatrices:
        for momentResultInBin in moments:
          label = binLabel(momentResultInBin)
          plotComplexMatrix(
            complexMatrix     = momentResultInBin.integralMatrix.matrixNormalized,
            pdfFileNamePrefix = f"{outFileDirName}/accMatrix_{label}_",
            axisTitles        = ("Physical Moment Index", "Measured Moment Index"),
            plotTitle         = f"{label}: "r"$\mathrm{\mathbf{I}}_\text{acc}$, ",
            zRangeAbs         = 1.2,
            zRangeImag        = 0.05,
          )
          plotComplexMatrix(
            complexMatrix     = momentResultInBin.integralMatrix.inverse,
            pdfFileNamePrefix = f"{outFileDirName}/accMatrixInv_{label}_",
            axisTitles        = ("Measured Moment Index", "Physical Moment Index"),
            plotTitle         = f"{label}: "r"$\mathrm{\mathbf{I}}_\text{acc}^{-1}$, ",
            zRangeAbs         = 5,
            zRangeImag        = 0.3,
          )

    if calcAccPsMoments:
      # calculate moments of accepted phase-space data
      with timer.timeThis(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads"):
        print(f"Calculating moments of phase-space MC data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
        moments.calculateMoments(dataSource = MomentCalculator.MomentDataSource.ACCEPTED_PHASE_SPACE, normalize = normalizeMoments)
        #TODO move into separate plotting script
        # plot kinematic dependences of all phase-space moments
        for qnIndex in momentIndices.qnIndices:
          HVals = tuple(MomentValueAndTruth(*momentsInBin.HMeas[qnIndex]) for momentsInBin in moments)
          plotMoments(
            HVals             = HVals,
            binning           = massBinning,
            normalizedMoments = normalizeMoments,
            momentLabel       = qnIndex.label,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{massBinning.var.name}_accPs_",
            histTitle         = qnIndex.title,
            plotLegend        = False,
          )
        # plot accepted phase-space moments in each kinematic bin
        for massBinIndex, momentResultInBin in enumerate(moments):
          label = binLabel(momentResultInBin)
          title = binTitle(momentResultInBin)
          print(f"Measured moments of accepted phase-space data for kinematic bin {title}:\n{momentResultInBin.HMeas}")
          print(f"Physical moments of accepted phase-space data for kinematic bin {title}:\n{momentResultInBin.HPhys}")
          # construct true moments for phase-space data
          HTruePs = MomentResult(momentIndices, label = "true")  # all true phase-space moments are 0 ...
          HTruePs._valsFlatIndex[momentIndices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1 if normalizeMoments else nmbPsGenEvents[massBinIndex]  # ... except for H_0(0, 0)
          # set H_0^meas(0, 0) to 0 so that one can better see the other H_0^meas moments
          momentResultInBin.HMeas._valsFlatIndex[0] = 0
          # plot measured and physical moments; the latter should match the true moments exactly except for tiny numerical effects
          plotMomentsInBin(
            HData             = momentResultInBin.HMeas,
            normalizedMoments = normalizeMoments,
            HTrue             = None,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_accPs_",
            plotLegend        = False,
          )
          plotMomentsInBin(
            HData             = momentResultInBin.HPhys,
            normalizedMoments = normalizeMoments,
            HTrue             = HTruePs,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_accPsCorr_"
          )

    # calculate moments of real data and write them to files
    momentResultsFileBaseName = f"{outFileDirName}/{namePrefix}_moments"
    with timer.timeThis(f"Time to calculate moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateMoments(normalize = normalizeMoments, nmbBootstrapSamples = nmbBootstrapSamples)
      moments.momentResultsMeas.save(f"{momentResultsFileBaseName}_meas.pkl")
      moments.momentResultsPhys.save(f"{momentResultsFileBaseName}_phys.pkl")


    #TODO move into separate plotting script
    with timer.timeThis(f"Time to plot moments of real data"):
      if plotAngularDistributions:
        print("Plotting total angular distributions")
        # load all signal and phase-space data
        print(f"Loading real data from tree '{treeName}' in file '{dataFileName}'")
        data = ROOT.RDataFrame(treeName, dataFileName)
        print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}'")
        dataPsAcc = ROOT.RDataFrame(treeName, psAccFileName)
        print(f"Loading generated phase-space data from tree '{treeName}' in file '{psGenFileName}'")
        dataPsGen = ROOT.RDataFrame(treeName, psGenFileName)
        plotAngularDistr(
          dataPsAcc         = dataPsAcc,
          dataPsGen         = dataPsGen,
          dataSignalAcc     = data,
          dataSignalGen     = None,
          pdfFileNamePrefix = f"{outFileDirName}/angDistr_total_",
        )

      # load moment results from files
      momentResultsMeas = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_meas.pkl")
      momentResultsPhys = MomentResultsKinematicBinning.load(f"{momentResultsFileBaseName}_phys.pkl")
      momentResultsClas = readMomentResultsClas(momentIndices, binVarMass)
      # normalize CLAS moments
      H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
      normMassBinIndex = 36  # corresponds to m_pipi = 0.765 GeV; in this bin H(0, 0) is maximal in CLAS and GlueX data
      scaleFactor =   momentResultsPhys[normMassBinIndex][H000Index].val.real \
                    / momentResultsClas[normMassBinIndex][H000Index].val.real
      momentResultsClas.scaleBy(scaleFactor)

      # plot moments in each kinematic bin
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        HMeas = momentResultsMeas[massBinIndex]
        HClas = momentResultsClas[massBinIndex]
        label = binLabel(HPhys)
        title = binTitle(HPhys)
        # print(f"True moments for kinematic bin {title}:\n{HTrue}")
        print(f"Measured moments of real data for kinematic bin {title}:\n{HMeas}")
        print(f"Physical moments of real data for kinematic bin {title}:\n{HPhys}")
        plotMomentsInBin(
          HData             = HPhys,
          normalizedMoments = normalizeMoments,
          HTrue             = HClas,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
          legendLabels      = ("Moment", "CLAS"),
          plotHTrueUncert   = True,
        )
        plotMomentsInBin(
          HData             = HMeas,
          normalizedMoments = normalizeMoments,
          HTrue             = None,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_meas_{label}_",
          plotLegend        = False,
        )
        #TODO also plot correlation matrices
        plotMomentsCovMatrices(
          HData             = HPhys,
          pdfFileNamePrefix = f"{outFileDirName}/covMatrix_{label}_",
          axisTitles        = ("Physical Moment Index", "Physical Moment Index"),
          plotTitle         = f"{label}: ",
        )
        if nmbBootstrapSamples > 0:
          graphTitle = f"({label})"
          plotMomentsBootstrapDistributions1D(
            HData             = HPhys,
            HTrue             = HClas,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
            histTitle         = title,
            HTrueLabel        = "CLAS",
          )
          plotMomentsBootstrapDistributions2D(
            HData             = HPhys,
            HTrue             = HClas,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
            histTitle         = title,
            HTrueLabel        = "CLAS",
          )
          plotMomentsBootstrapDiffInBin(
            HData             = HPhys,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
            graphTitle        = title,
          )

      # plot kinematic dependences of all moments
      for qnIndex in momentResultsPhys[0].indices.qnIndices:
        plotMoments1D(
          momentResults     = momentResultsPhys,
          qnIndex           = qnIndex,
          binning           = massBinning,
          normalizedMoments = normalizeMoments,
          momentResultsTrue = momentResultsClas,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_",
          histTitle         = qnIndex.title,
          plotLegend        = True,
          legendLabels      = ("Moment", "CLAS"),
          plotHTrueUncert   = True,
        )
        plotMoments1D(
          momentResults     = momentResultsMeas,
          qnIndex           = qnIndex,
          binning           = massBinning,
          normalizedMoments = normalizeMoments,
          momentResultsTrue = None,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_meas_",
          histTitle         = qnIndex.title,
          plotLegend        = False,
        )

      # plot ratio of measured and physical value for Re[H_0(0, 0)]; estimates efficiency
      H000s = (
        tuple(MomentValueAndTruth(*HMeas[H000Index]) for HMeas in momentResultsMeas),
        tuple(MomentValueAndTruth(*HPhys[H000Index]) for HPhys in momentResultsPhys),
      )
      hists = (
        ROOT.TH1D(f"H000Meas", "", *massBinning.astuple),
        ROOT.TH1D(f"H000Phys", "", *massBinning.astuple),
      )
      for indexMeasPhys, H000 in enumerate(H000s):
        histIntensity = hists[indexMeasPhys]
        for indexKinBin, HVal in enumerate(H000):
          if (massBinning._var not in HVal.binCenters.keys()):
            continue
          y, yErr = HVal.realPart(True)
          binIndex = histIntensity.GetXaxis().FindBin(HVal.binCenters[massBinning.var])
          histIntensity.SetBinContent(binIndex, y)
          histIntensity.SetBinError  (binIndex, 1e-100 if yErr < 1e-100 else yErr)
      histRatio = hists[0].Clone("H000Ratio")
      histRatio.Divide(hists[1])
      canv = ROOT.TCanvas()
      histRatio.SetMarkerStyle(ROOT.kFullCircle)
      histRatio.SetMarkerSize(0.75)
      histRatio.SetXTitle(massBinning.axisTitle)
      histRatio.SetYTitle("#it{H}_{0}^{meas}(0, 0) / #it{H}_{0}^{phys}(0, 0)")
      histRatio.Draw("PEX0")
      canv.SaveAs(f"{outFileDirName}/{histRatio.GetName()}.pdf")

      # overlay H_0^meas(0, 0) and measured intensity distribution; must be identical
      histIntMeas = ROOT.RDataFrame(treeName, dataFileName) \
                        .Histo1D(
                          ROOT.RDF.TH1DModel("intensity_meas", f";{massBinning.axisTitle};Events", *massBinning.astuple), "mass", "eventWeight"
                        ).GetValue()
      for binIndex, HMeas in enumerate(H000s[0]):
        HMeas.truth = histIntMeas.GetBinContent(binIndex + 1)  # set truth values to measured intensity
      plotMoments(
        HVals             = H000s[0],
        binning           = massBinning,
        normalizedMoments = normalizeMoments,
        momentLabel       = H000Index.label,
        pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_meas_intensity_",
        histTitle         = H000Index.title,
        legendLabels      = ("Measured Moment", "Measured Intensity"),
      )

    timer.stop("Total execution time")
    print(timer.summary)
