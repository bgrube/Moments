#!/usr/bin/env python3
# performs moments analysis for eta pi0 real-data events

from dataclasses import dataclass
import functools
import numpy as np
import pandas as pd
import threadpoolctl
from typing import (
  Dict,
  List,
  Tuple,
)

import ROOT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
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
  QnWaveIndex,
)
from PlottingUtilities import (
  HistAxisBinning,
  MomentValueAndTruth,
  plotAngularDistr,
  plotComplexMatrix,
  plotMoments,
  plotMoments1D,
  plotMomentsInBin,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads basisFunctions.C
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass
class BeamPolInfo:
  """Stores info about beam polarization datasets"""
  datasetLabel:   str    # label used for dataset
  angle:          int    # beam polarization angle in lab frame [deg]
  polarization:   float  # average beam polarization
  ampScaleFactor: float  # fitted amplitude scaling factor for dataset

# see Eqs. (4.22)ff in Lawrence's thesis for polarization values and
# /w/halld-scshelf2101/malte/final_fullWaveset/nominal_fullWaveset_ReflIndiv_150rnd/010020/etapi_result_samePhaseD.fit
# for amplitude scaling factors
BEAM_POL_INFOS: Tuple[BeamPolInfo, ...] = (
  BeamPolInfo(datasetLabel = "EtaPi0_000", angle =   0, polarization = 0.35062, ampScaleFactor = 1.0),
  BeamPolInfo(datasetLabel = "EtaPi0_045", angle =  45, polarization = 0.34230, ampScaleFactor = 0.982204395837131),
  BeamPolInfo(datasetLabel = "EtaPi0_090", angle =  90, polarization = 0.34460, ampScaleFactor = 0.968615883555624),
  BeamPolInfo(datasetLabel = "EtaPi0_135", angle = 135, polarization = 0.35582, ampScaleFactor = 0.98383623655323),
)


def readPartialWaveAmplitudes(
  csvFileName:       str,    # name of CSV file with partial-wave amplitudes
  massBinCenter:     float,  # [GeV]
  fitResultPlotDir:  str,    # directory with intensity plots generated from PWA fit result
  beamPolAngleLabel: str = "000",  # "total" for total data sample summed over polarization directions
) -> List[AmplitudeValue]:
  """Reads partial-wave amplitude values for given mass bin from CSV data"""
  print(f"Reading partial-wave amplitudes for mass bin at {massBinCenter} GeV from file '{csvFileName}'")
  df = pd.read_csv(csvFileName, index_col = [0]).astype({"mass": float})
  df = df.loc[np.isclose(df["mass"], massBinCenter)].drop(columns = ["mass"])  # select row for mass bin center
  # to calculate total amplitudes use amplitude values for 0 deg beam
  # polarization because for this dataset the scaling factor is 1
  angleLabel = "EtaPi0_000" if beamPolAngleLabel == "total" else f"EtaPi0_{beamPolAngleLabel}"
  df = df.filter(like = f"{angleLabel}::")   # select columns for beam polarization direction
  assert len(df) == 1, f"Expected exactly 1 row for mass-bin center {massBinCenter} GeV, but found {len(df)}"
  # Pandas cannot read-back complex values out of the box
  # there also seems to be no interest in fixing that; see <https://github.com/pandas-dev/pandas/issues/9379>
  # have to convert the respective column by hand
  ampSeries = df.astype('complex128').loc[df.index[0]]
  # normalize amplitudes to number of produced events
  # !Note! In principle, the normalized amplitude value is given by
  #   A^norm_i = A_i * ampScaleFactor_i * sqrt(I_{ii})
  # However, the diagonal elements of the integral matrix from the fit
  # result cannot be used here because they are valid only for the
  # whole fitted mass range.  The information for individual mass bins
  # is lost.  As a workaround, we extract the normalization factors
  # from the intensity plots generated from the PWA fit result.
  for key in ampSeries.index:
    waveName = key.split("::")[-1]
    plotFileName = f"{fitResultPlotDir}/etapi_plot_{waveName}.root"
    plotFile = ROOT.TFile.Open(plotFileName, "READ")
    waveIntensity = 0.0
    # for total intensity sum up intensities of all beam polarization directions
    for angleLabelHist in [beamPolInfo.datasetLabel for beamPolInfo in BEAM_POL_INFOS] if beamPolAngleLabel == "total" else [angleLabel]:
      intensityHistName = f"{angleLabelHist}_Metapi_40MeVBingen"  # acceptance-corrected intensity in units of produced events
      intensityHist = plotFile.Get(intensityHistName)
      waveIntensity += intensityHist.GetBinContent(intensityHist.FindBin(massBinCenter))
    #TODO intensity seems to be by a factor of 2 too large; need to check why
    normFactor = np.sqrt(waveIntensity / 2) / np.abs(ampSeries[key])  # normalize amplitude A such that |A|^2 == plotted intensity
    ampSeries[key] *= normFactor
    plotFile.Close()
  # add amplitudes of a_2(1320) and a_2(1700)
  a2AmpKeys = sorted([key for key in ampSeries.index if "::D" in key])  # keys of a_2(1320) amplitudes
  for a2AmpKey in a2AmpKeys:
    a2PrimeAmpKey = a2AmpKey.replace("::D", "::pD")  # key for a_2(1700) amplitude
    ampSeries[a2AmpKey] = ampSeries[a2AmpKey] + ampSeries[a2PrimeAmpKey]
    ampSeries.drop(a2PrimeAmpKey, inplace = True)
  print(f"Partial-wave amplitudes for mass bin at {massBinCenter} GeV:\n{ampSeries}")
  # construct list of AmplitudeValue objects
  partialWaveAmplitudes = [
    # negative-reflectivity waves
    AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val = ampSeries[f"{angleLabel}::NegativeRe::S0+-"]),  # S_0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -2), val = ampSeries[f"{angleLabel}::NegativeRe::D2--"]),  # D_-2^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val = ampSeries[f"{angleLabel}::NegativeRe::D1--"]),  # D_-1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val = ampSeries[f"{angleLabel}::NegativeRe::D0+-"]),  # D_0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val = ampSeries[f"{angleLabel}::NegativeRe::D1+-"]),  # D_+1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +2), val = ampSeries[f"{angleLabel}::NegativeRe::D2+-"]),  # D_+2^-
    # positive-reflectivity waves
    AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val = ampSeries[f"{angleLabel}::PositiveRe::S0++"]),  # S_0^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val = ampSeries[f"{angleLabel}::PositiveRe::D2-+"]),  # D_-2^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -1), val = ampSeries[f"{angleLabel}::PositiveRe::D1-+"]),  # D_-1^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m =  0), val = ampSeries[f"{angleLabel}::PositiveRe::D0++"]),  # D_0^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +1), val = ampSeries[f"{angleLabel}::PositiveRe::D1++"]),  # D_+1^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val = ampSeries[f"{angleLabel}::PositiveRe::D2++"]),  # D_+2^+
  ]
  return partialWaveAmplitudes


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
    treeName                 = "etaPi0"
    # beamPolAngleLabel        = "000"
    # beamPolAngleLabel        = "045"
    # beamPolAngleLabel        = "090"
    # beamPolAngleLabel        = "135"
    beamPolAngleLabel        = "total"
    fileNamePattern          = "*" if beamPolAngleLabel == "total" else beamPolAngleLabel
    dataFileName             = f"./dataPhotoProdEtaPi0/data_{fileNamePattern}_flat.root"
    psAccFileName            = f"./dataPhotoProdEtaPi0/phaseSpace_acc_{fileNamePattern}_flat.root"
    psGenFileName            = f"./dataPhotoProdEtaPi0/phaseSpace_gen_{fileNamePattern}_flat.root"
    pwAmpsFileName           = "./dataPhotoProdEtaPi0/evaluate_amplitude/evaluate_amplitude.csv"
    fitResultPlotDir         = "./dataPhotoProdEtaPi0/intensityPlots/010020"
    outFileDirName           = Utilities.makeDirPath(f"./plotsPhotoProdEtaPi0_{beamPolAngleLabel}")
    # maxL                     = 1  # define maximum L quantum number of moments
    maxL                     = 5  # define maximum L quantum number of moments
    normalizeMoments         = False
    # plotAngularDistributions = True
    plotAngularDistributions = False
    # plotAccIntegralMatrices  = True
    plotAccIntegralMatrices  = False
    # calcAccPsMoments         = True
    calcAccPsMoments         = False
    binVarMass               = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#eta#pi}^{0}}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    massBinning              = HistAxisBinning(nmbBins = 17, minVal = 1.04, maxVal = 1.72, _var = binVarMass)
    # massBinning              = HistAxisBinning(nmbBins = 1, minVal = 1.28, maxVal = 1.32, _var = binVarMass)
    nmbOpenMpThreads         = ROOT.getNmbOpenMpThreads()

    namePrefix = "norm" if normalizeMoments else "unnorm"

    # setup MomentCalculators for all mass bins
    momentIndices = MomentIndices(maxL)
    momentsInBins:    List[MomentCalculator] = []
    momentsInBinsPwa: List[MomentCalculator] = []
    nmbPsGenEvents:   List[int]              = []
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
        assert nmbPsGenEvents[-1] == dataPsGenInBin.Sum("eventWeight").GetValue(), f"Event number mismatch: {nmbPsGenEvents[-1]=} vs. {dataPsGenInBin.Sum('eventWeight').GetValue()=}"
        nmbPsAccEvents = dataPsAccInBin.Sum("eventWeight").GetValue()
        print(f"Loaded phase-space events: number generated = {nmbPsGenEvents[-1]}; "
              f"number accepted = {dataPsAccInBin.Count().GetValue()}, "
              f"{nmbPsAccEvents} events after background subtraction"
              f" -> efficiency = {nmbPsAccEvents / nmbPsGenEvents[-1]:.3f}")

        # calculate moments from PWA fits result
        amplitudeSet = AmplitudeSet(amps = readPartialWaveAmplitudes(pwAmpsFileName, massBinCenter, fitResultPlotDir, beamPolAngleLabel), tolerance = 1e-7)
        HPwa: MomentResult = amplitudeSet.photoProdMomentSet(maxL, normalize = normalizeMoments, printMomentFormulas = False)
        print(f"Moment values from partial-wave analysis:\n{HPwa}")

        # setup moment calculators for data
        dataSet = DataSet(
          data           = dataInBin,
          phaseSpaceData = dataPsAccInBin,
          nmbGenEvents   = nmbPsGenEvents[-1],
        )
        momentsInBins.append(MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix", binCenters = {binVarMass : massBinCenter}))
        # setup moment calculator to hold moment values from PWA result
        momentsInBinsPwa.append(MomentCalculator(momentIndices, dataSet, binCenters = {binVarMass : massBinCenter}, _HPhys = HPwa))

        # plot angular distributions for mass bin
        if plotAngularDistributions:
          plotAngularDistr(dataPsAccInBin, dataPsGenInBin, dataInBin, dataSignalGen = None, pdfFileNamePrefix = f"{outFileDirName}/angDistr_{binLabel(momentsInBins[-1])}_")
    moments    = MomentCalculatorsKinematicBinning(momentsInBins)
    momentsPwa = MomentCalculatorsKinematicBinning(momentsInBinsPwa)

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
          plotComplexMatrix(momentResultInBin.integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outFileDirName}/accMatrix_{label}_",
                            axisTitles = ("Physical Moment Index", "Measured Moment Index"), plotTitle = f"{label}: "r"$\mathrm{\mathbf{I}}_\text{acc}$, ",
                            zRangeAbs = 1.5, zRangeImag = 0.25)
          plotComplexMatrix(momentResultInBin.integralMatrix.inverse,          pdfFileNamePrefix = f"{outFileDirName}/accMatrixInv_{label}_",
                            axisTitles = ("Measured Moment Index", "Physical Moment Index"), plotTitle = f"{label}: "r"$\mathrm{\mathbf{I}}_\text{acc}^{-1}$, ",
                            zRangeAbs = 115, zRangeImag = 30)

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
          plotMomentsInBin(momentResultInBin.HMeas, normalizeMoments,                  pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_accPs_", plotLegend = False)
          # plotMomentsInBin(momentResultInBin.HPhys, normalizeMoments, HTrue = HTruePs, pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_accPsCorr_")
        # plot kinematic dependences of all phase-space moments
        for qnIndex in momentIndices.QnIndices():
          HVals = tuple(MomentValueAndTruth(*momentsInBin.HMeas[qnIndex]) for momentsInBin in moments)
          plotMoments(HVals, massBinning, normalizeMoments, momentLabel = qnIndex.label,
                      pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{massBinning.var.name}_accPs_", histTitle = qnIndex.title, plotLegend = False)

    # calculate moments of real data and write them to files
    with timer.timeThis(f"Time to calculate moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads"):
      print(f"Calculating moments of real data for {len(moments)} bins using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateMoments(normalize = normalizeMoments)
      momentsPwa.momentResultsPhys.save(f"{outFileDirName}/{namePrefix}_moments_true.pkl")
      moments.momentResultsMeas.save   (f"{outFileDirName}/{namePrefix}_moments_meas.pkl")
      moments.momentResultsPhys.save   (f"{outFileDirName}/{namePrefix}_moments_phys.pkl")

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
        plotAngularDistr(dataPsAcc, dataPsGen, data, dataSignalGen = None, pdfFileNamePrefix = f"{outFileDirName}/angDistr_total_")

      # load moment results from files
      momentResultsTrue = MomentResultsKinematicBinning.load(f"{outFileDirName}/{namePrefix}_moments_true.pkl")
      momentResultsMeas = MomentResultsKinematicBinning.load(f"{outFileDirName}/{namePrefix}_moments_meas.pkl")
      momentResultsPhys = MomentResultsKinematicBinning.load(f"{outFileDirName}/{namePrefix}_moments_phys.pkl")

      # plot moments in each kinematic bin
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        HTrue = momentResultsTrue[massBinIndex]
        HMeas = momentResultsMeas[massBinIndex]
        label = binLabel(HPhys)
        title = binTitle(HPhys)
        # print(f"True moments for kinematic bin {title}:\n{HTrue}")
        print(f"Measured moments of real data for kinematic bin {title}:\n{HMeas}")
        print(f"Physical moments of real data for kinematic bin {title}:\n{HPhys}")
        plotMomentsInBin(
          HData             = HPhys,
          normalizedMoments = normalizeMoments,
          HTrue             = HTrue,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_{label}_",
          legendLabels      = ("Moment", "PWA Result"),
        )
        plotMomentsInBin(
          HData             = HMeas,
          normalizedMoments = normalizeMoments,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_meas_{label}_",
          plotLegend        = False,
        )

      # plot kinematic dependences of all moments
      for qnIndex in momentResultsPhys[0].indices.QnIndices():
        plotMoments1D(
          momentResults     = momentResultsPhys,
          qnIndex           = qnIndex,
          binning           = massBinning,
          normalizedMoments = normalizeMoments,
          momentResultsTrue = momentResultsTrue,
          pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_",
          histTitle         = qnIndex.title,
          legendLabels      = ("Moment", "PWA Result"),
        )
        plotMoments1D(
          momentResults     = momentResultsMeas,
          qnIndex           = qnIndex,
          binning           = massBinning,
          normalizedMoments = normalizeMoments,
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
      histIntMeas = data.Histo1D(
        ROOT.RDF.TH1DModel("intensity_meas", f";{massBinning.axisTitle};Events", *massBinning.astuple), "mass", "eventWeight").GetValue()
      for binIndex, HMeas in enumerate(H000s[0]):
        HMeas.truth = histIntMeas.GetBinContent(binIndex + 1)
      plotMoments(
        HVals             = H000s[0],
        binning           = massBinning,
        normalizedMoments = normalizeMoments,
        momentLabel       = H000Index.label,
        pdfFileNamePrefix = f"{outFileDirName}/{namePrefix}_meas_",
        histTitle         = H000Index.title,
        legendLabels      = ("Measured Moment", "Measured Intensity"),
      )

      # plot ratio of measured and produced intensity; estimates efficiency
      plotFile = ROOT.TFile.Open(f"{fitResultPlotDir}/etapi_plot_S0+-_S0++_D2--_D1--_D0+-_D1+-_D2+-_D2-+_D1-+_D0++_D1++_D2++_pD2--_pD1--_pD0+-_pD1+-_pD2+-_pD2-+_pD1-+_pD0++_pD1++_pD2++.root", "READ")
      # sum up intensity distributions for all beam polarization directions
      angleLabel = "EtaPi0_000" if beamPolAngleLabel == "total" else f"EtaPi0_{beamPolAngleLabel}"
      histsIntensity = [None, None]  # 0 = acc, 1 = gen
      for angleLabelHist in [beamPolInfo.datasetLabel for beamPolInfo in BEAM_POL_INFOS] if beamPolAngleLabel == "total" else [angleLabel]:
        for labelIndex, label in enumerate(("acc", "gen")):
          histName = f"{angleLabelHist}_Metapi_40MeVBin{label}"
          if histsIntensity[labelIndex] is None:
            print(f"Reading histogram '{histName}'")
            histsIntensity[labelIndex] = plotFile.Get(histName)
          else:
            print(f"Adding histogram '{histName}'")
            histsIntensity[labelIndex].Add(plotFile.Get(histName))
      histRatioPwa = histsIntensity[0].Clone("PwaIntRatio")
      histRatioPwa.Divide(histsIntensity[1])
      canv = ROOT.TCanvas()
      # histRatioPwa.SetMarkerStyle(ROOT.kFullCircle)
      # histRatioPwa.SetMarkerSize(0.75)
      histRatioPwa.SetXTitle(massBinning.axisTitle)
      histRatioPwa.GetXaxis().SetRangeUser(1.04, 1.72)
      histRatioPwa.SetYTitle("Ratio of Measured and Produced Intensity")
      histRatioPwa.Draw("PEX0")
      canv.SaveAs(f"{outFileDirName}/{histRatioPwa.GetName()}.pdf")
      for histIntensity in histsIntensity:
        canv = ROOT.TCanvas()
        histIntensity.SetMarkerStyle(ROOT.kFullCircle)
        histIntensity.SetMarkerSize(0.75)
        histIntensity.SetXTitle(massBinning.axisTitle)
        histIntensity.GetXaxis().SetRangeUser(1.04, 1.72)
        histIntensity.SetYTitle("Intensity")
        histIntensity.Draw("PEX0")
        canv.SaveAs(f"{outFileDirName}/{histIntensity.GetName()}.pdf")
      # overlay efficiencies from moments and PWA intensities
      histEff = ROOT.THStack("efficiency", f";{histRatio.GetXaxis().GetTitle()};Efficiency")
      histRatio.SetLineColor(ROOT.kRed + 1)
      histRatio.SetMarkerColor(ROOT.kRed + 1)
      histRatio.SetMarkerStyle(ROOT.kFullCircle)
      histRatio.SetMarkerSize(0.75)
      histRatio.SetTitle("Moment Analysis")
      histEff.Add(histRatio, "PEX0")
      histRatioPwa.SetMarkerColor(ROOT.kBlue + 1)
      histRatioPwa.SetLineColor(ROOT.kBlue + 1)
      histRatioPwa.SetLineWidth(2)
      histRatioPwa.SetTitle("PWA")
      histEff.Add(histRatioPwa, "PE")
      canv = ROOT.TCanvas()
      histEff.Draw("NOSTACK")
      histEff.GetXaxis().SetLimits(1.04, 1.72)
      histEff.SetMaximum(0.18)
      canv.BuildLegend(0.7, 0.85, 0.99, 0.99)
      canv.SaveAs(f"{outFileDirName}/{histEff.GetName()}.pdf")

      # overlay intensity from partial-wave amplitudes and the total intensity from AmpTools; must be identical
      HVals = []
      for binIndex, HTrue in enumerate(momentResultsTrue):
        H000True = HTrue[H000Index]
        #FIXME binCenters is not set for momentResultsTrue
        # print(f"??? {H000True=}")
        HVal = MomentValueAndTruth(*H000True)
        HVal.binCenters = H000s[0][binIndex].binCenters  # workaround
        hist = histsIntensity[1]
        histBinIndex = hist.FindBin(HVal.binCenters[massBinning.var])
        HVal.label    = "phys"
        HVal.val      = hist.GetBinContent(histBinIndex)
        HVal.uncertRe = hist.GetBinError  (histBinIndex)
        HVal.truth    = H000True.val
        print(f"!!! {HVal=}")
        HVals.append(HVal)
      plotMoments(
        HVals             = HVals,
        binning           = massBinning,
        normalizedMoments = normalizeMoments,
        momentLabel       = H000Index.label,
        pdfFileNamePrefix = f"{outFileDirName}/amptools_intensity_",
        histTitle         = H000Index.title,
        legendLabels      = ("AmpTools Intensity", "My Intensity"),
      )

    timer.stop("Total execution time")
    print(timer.summary)
