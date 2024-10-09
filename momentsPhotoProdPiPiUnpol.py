#!/usr/bin/env python3
# performs moments analysis for pi+ pi- real-data events in CLAS energy range


from __future__ import annotations

from dataclasses import dataclass
import functools
import numpy as np
import threadpoolctl

import ROOT

from MomentCalculator import (
  binLabel,
  binTitle,
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
  MomentResult,
  MomentResultsKinematicBinning,
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
    outFileDirName           = Utilities.makeDirPath(f"./plotsPhotoProdPiPiUnpol")
    maxL                     = 5  # define maximum L quantum number of moments
    normalizeMoments         = False
    nmbBootstrapSamples      = 0
    # nmbBootstrapSamples      = 10000
    # plotAngularDistributions = True
    plotAngularDistributions = False
    # plotAccIntegralMatrices  = True
    plotAccIntegralMatrices  = False
    # calcAccPsMoments         = True
    calcAccPsMoments         = False
    binVarMass               = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
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
        dataPsAccInBin = ROOT.RDataFrame(treeName, psAccFileName).Filter(binMassRangeFilter)
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
        # plot accepted phase-space moments in each kinematic bin
        #TODO move into separate plotting script
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
          # plotMomentsInBin(
          #   HData             = momentResultInBin.HPhys,
          #   normalizedMoments = normalizeMoments,
          #   HTrue             = HTruePs,
          #   pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_accPsCorr_"
          # )
        # plot kinematic dependences of all phase-space moments
        for qnIndex in momentIndices.QnIndices():
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

    # calculate moments of real data and write them to files
    with timer.timeThis(f"Time to calculate moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateMoments(normalize = normalizeMoments, nmbBootstrapSamples = nmbBootstrapSamples)
      moments.momentResultsMeas.save(f"{outFileDirName}/{namePrefix}_moments_meas.pkl")
      moments.momentResultsPhys.save(f"{outFileDirName}/{namePrefix}_moments_phys.pkl")

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
      momentResultsMeas = MomentResultsKinematicBinning.load(f"{outFileDirName}/{namePrefix}_moments_meas.pkl")
      momentResultsPhys = MomentResultsKinematicBinning.load(f"{outFileDirName}/{namePrefix}_moments_phys.pkl")

      # plot moments in each kinematic bin
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        HMeas = momentResultsMeas[massBinIndex]
        label = binLabel(HPhys)
        title = binTitle(HPhys)
        # print(f"True moments for kinematic bin {title}:\n{HTrue}")
        print(f"Measured moments of real data for kinematic bin {title}:\n{HMeas}")
        print(f"Physical moments of real data for kinematic bin {title}:\n{HPhys}")
        plotMomentsInBin(
          HData             = HPhys,
          normalizedMoments = normalizeMoments,
          HTrue             = None,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
          legendLabels      = ("Moment", "PWA Result"),
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
            HTrue             = None,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
            histTitle         = title)
          plotMomentsBootstrapDistributions2D(
            HData             = HPhys,
            HTrue             = None,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
            histTitle         = title,
          )
          plotMomentsBootstrapDiffInBin(
            HData             = HPhys,
            pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
            graphTitle        = title,
          )

      # plot kinematic dependences of all moments
      for qnIndex in momentResultsPhys[0].indices.QnIndices():
        plotMoments1D(
          momentResults     = momentResultsPhys,
          qnIndex           = qnIndex,
          binning           = massBinning,
          normalizedMoments = normalizeMoments,
          momentResultsTrue = None,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_",
          histTitle         = qnIndex.title,
          legendLabels      = ("Moment", "PWA Result"),
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
      H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
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
