#!/usr/bin/env python3
"""
This module weights data (usually generated or accepted phase-space
data) with the intensity distribution calculated from the results of
the moment analysis of unpolarized or polarized pi+ pi-
photoproduction data.  The moment values are read from files produced
by the script `photoProdCalcMoments.py` that calculates the moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import os
import subprocess
import tempfile

import ROOT
from wurlitzer import pipes, STDOUT

from AnalysisConfig import (
  AnalysisConfig,
  CFG_KEVIN,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_JPAC,
  CFG_UNPOLARIZED_PIPI_PWA,
  HistAxisBinning,
)
from MomentCalculator import (
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
from PlottingUtilities import setupPlotStyle
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def loadInputData(
  inputDataDef: AnalysisConfig.DataType | str | int,  # if `AnalysisConfig.DataType` instance, the file corresponding to `DataType` is loaded
                                                      # if `str`, a file name is expected
                                                      # if `int`, phase-space distribution in angles is generated with given number of events
  cfg:          AnalysisConfig,
  massBinIndex: int,  # index of mass bin to load/generate data for
) -> tuple[ROOT.RDataFrame, int]:
  """Loads data specified by `inputDataDef` and returns them as RDataFrame and the number of input events."""
  if isinstance(inputDataDef, AnalysisConfig.DataType) or isinstance(inputDataDef, str):
    dataToWeight = None
    if isinstance(inputDataDef, AnalysisConfig.DataType):
      print(f"Loading data of type '{inputDataDef}'")
      dataToWeight = cfg.loadData(inputDataDef)
    elif isinstance(inputDataDef, str):
      print(f"Loading data from file '{inputDataDef}'")
      dataToWeight = ROOT.RDataFrame(cfg.treeName, inputDataDef)
    assert dataToWeight is not None, f"Could not load data of type '{inputDataDef}'"
    kinematicBinFilter: str = cfg.massBinning.binFilter(massBinIndex)
    dataToWeight = dataToWeight.Filter(kinematicBinFilter)
    nmbInputEvents = dataToWeight.Count().GetValue()
    print(f"Input data contain {nmbInputEvents} events in bin '{kinematicBinFilter}'")
    return dataToWeight, nmbInputEvents
  elif isinstance(inputDataDef, int):
    nmbGenPsEvents = inputDataDef
    print(f"Generating phase-space distribution with {nmbGenPsEvents} events")
    kinematicBinRange: tuple[float, float] = cfg.massBinning.binValueRange(massBinIndex)
    dataToWeight = (
      ROOT.RDataFrame(nmbGenPsEvents)
          .Define("cosTheta", "(Double32_t)gRandom->Uniform(-1, +1)")
          .Define("theta",    "(Double32_t)std::acos(cosTheta)")
          .Define("phiDeg",   "(Double32_t)gRandom->Uniform(-180, +180)")
          .Define("phi",      "(Double32_t)phiDeg * TMath::DegToRad()")
          .Define("mass",    f"(Double32_t)gRandom->Uniform({kinematicBinRange[0]}, {kinematicBinRange[1]})")
          .Filter('if (rdfentry_ == 0) { cout << "Running event loop in weightAccPhaseSpaceWithIntensity()" << endl; } return true;')  # noop filter that just prints a log message when event loop is running
    )
    if cfg.polarization is not None:
      # polarized case: add Phi and polarization columns
      dataToWeight = (
        dataToWeight.Define("PhiDeg", "(Double32_t)gRandom->Uniform(-180, +180)")
                    .Define("Phi",    "(Double32_t)PhiDeg * TMath::DegToRad()")
      )
      if isinstance(cfg.polarization, float):
        dataToWeight = dataToWeight.Define("beamPol", f"(Double32_t){cfg.polarization}")
      elif isinstance(cfg.polarization, str):
        raise ValueError(f"Cannot read polarization from column '{cfg.polarization}'")
    #TODO is a snapshot necessary here to fill random columns only once?
    return dataToWeight, nmbGenPsEvents
  else:
    raise ValueError(f"Invalid {inputDataDef=}")


def weightDataWithIntensity(
  inputDataDef:     AnalysisConfig.DataType | str | int,  # if `AnalysisConfig.DataType` instance, the file corresponding to `DataType` is loaded
                                                          # if `str`, a file name is expected
                                                          # if `int`, phase-space distribution in angles is generated with given number of events
  massBinIndex:     int,  # index of mass bin to generate data for
  intensityFormula: str,  # formula for intensity function
  outFileName:      str,  # ROOT file to which weighted events are written
  cfg:              AnalysisConfig,
  seed:             int = 123456789,  # seed for rejection sampling and for generating phase-space events
) -> None:
  """Weight input data specified by `inputDataDef` with given intensity formula"""
  # get input data
  ROOT.gRandom.SetSeed(seed)
  # load input data
  dataToWeight, nmbInputEvents = loadInputData(
    inputDataDef = inputDataDef,
    cfg          = cfg,
    massBinIndex = massBinIndex,
  )
  # calculate intensity weight and random number in [0, 1] for each event
  print(f"Calculating weights using formula '{intensityFormula}'")
  dataToWeight = (
    dataToWeight.Define("intensityWeight",      f"(Double32_t){intensityFormula}")
                .Define("intensityWeightRndNmb", "(Double32_t)gRandom->Rndm()")  # random number in [0, 1] for each event
  )
  # write unweighted data to file and read data back to ensure that random columns are filled only once
  tmpFileName = f"{outFileName}.tmp.root"
  dataToWeight.Snapshot(cfg.treeName, tmpFileName)
  dataToWeight = ROOT.RDataFrame(cfg.treeName, tmpFileName)
  # determine maximum weight
  maxIntensityWeight = dataToWeight.Max("intensityWeight").GetValue()
  print(f"Maximum intensity is {maxIntensityWeight}")
  # apply weights by accepting each event with probability intensityWeight / maxIntensityWeight
  weightedData = (
    dataToWeight.Define("acceptEventIntensityWeight", f"(bool)(intensityWeightRndNmb < (intensityWeight / {maxIntensityWeight}))")
                .Filter("acceptEventIntensityWeight == true")
  )
  nmbWeightedEvents = weightedData.Count().GetValue()
  print(f"After weighting with the intensity function, the sample contains {nmbWeightedEvents} accepted events; "
        f"weighting efficiency is {nmbWeightedEvents / nmbInputEvents}")
  # write weighted data to file
  print(f"Writing data weighted with intensity function to file '{outFileName}'")
  weightedData.Snapshot(cfg.treeName, outFileName)  #TODO write out only essential columns
  subprocess.run(f"rm --force --verbose {tmpFileName}", shell = True)  # remove temporary file


def reweightData(
  dataToWeight: ROOT.RDataFrame,  # data to reweight
  treeName:     str,              # name of TTree holding the data
  variableName: str,              # column name corresponding to kinematic variable whose distribution is to be reweighted
  targetDistr:  ROOT.TH1D,        # histogram with target distribution
) -> ROOT.RDataFrame:
  """Generic function to reweight the data in an RDataFrame such that the distribution of the given variable matches the target distribution in the given histogram"""
  # get histogram of current distribution using same binning as targetDistribution
  currentDistr = dataToWeight.Histo1D(
    ROOT.RDF.TH1DModel(
      f"{variableName}Distr", f";{variableName};Count",
      targetDistr.GetNbinsX(), targetDistr.GetXaxis().GetXmin(), targetDistr.GetXaxis().GetXmax()
    ),
    variableName,
  ).GetValue()
  # # save plots of distributions
  # canv = ROOT.TCanvas()
  # currentDistr.Draw()
  # canv.SaveAs(f"{currentDistr.GetName()}.root")
  # canv = ROOT.TCanvas()
  # targetDistr.Draw()
  # canv.SaveAs(f"{targetDistr.GetName()}.root")
  # normalize histograms such that they represent the corresponding PDFs
  targetDistr.Scale (1.0 / targetDistr.Integral ())
  currentDistr.Scale(1.0 / currentDistr.Integral())
  # calculate PDF ratio that defines the weight histogram
  weightsHist = targetDistr.Clone("weightsHist")
  weightsHist.Divide(currentDistr)
  # add weights to input data
  RootUtilities.declareInCpp(weightsHist = weightsHist)  # use Python object in C++
  dataToWeight = (
    dataToWeight.Define("reweightingWeight", f"(Double32_t)PyVars::weightsHist.GetBinContent(PyVars::weightsHist.FindBin({variableName}))")
                .Define("reweightingRndNmb",  "(Double32_t)gRandom->Rndm()")
  )
  tmpFileName = tempfile.mktemp(dir = "./", prefix = "unweighted.", suffix = ".root")
  dataToWeight.Snapshot(treeName, tmpFileName)  # write unweighted data to file to ensure that random columns are filled only once
  dataToWeight = ROOT.RDataFrame(treeName, tmpFileName)  # read data back
  nmbEvents = dataToWeight.Count().GetValue()
  # determine maximum weight
  maxWeight = dataToWeight.Max("reweightingWeight").GetValue()
  print(f"Maximum weight is {maxWeight}")
  # apply weights by accepting each event with probability reweightingWeight / maxWeight
  reweightedData = (
    dataToWeight.Define("acceptEventReweight", f"(bool)(reweightingRndNmb < (reweightingWeight / {maxWeight}))")
                .Filter("acceptEventReweight == true")
  )
  nmbWeightedEvents = reweightedData.Count().GetValue()
  print(f"After reweighting, the sample contains {nmbWeightedEvents} accepted events; reweighting efficiency is {nmbWeightedEvents / nmbEvents}")
  subprocess.run(f"rm -fv {tmpFileName}", shell = True)
  return reweightedData


def reweightKinDistribution(
  dataToWeight:  ROOT.RDataFrame,  # data to reweight
  treeName:      str,              # name of TTree holding the data
  binning:       HistAxisBinning,  # binning of kinematic variable whose distribution is to be reweighted
  momentResults: MomentResultsKinematicBinning,  # moment values
  outFileName:   str,  # name of file to write data into
) -> None:
  """Reweight mass distribution of data according to mass dependence of H_0(0, 0)"""
  print(f"Reweighting {binning.var.name} dependence")
  # construct target distribution from H_0(0, 0) values in kinematic bins
  targetDistr = ROOT.TH1D(f"{binning.var.name}DistrTarget", f";{binning.axisTitle};Count", *binning.astuple)
  H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
  for momentResultsForBin in momentResults:
    massBinCenter = momentResultsForBin.binCenters[binning.var]
    targetDistr.SetBinContent(targetDistr.FindBin(massBinCenter), momentResultsForBin[H000Index].real[0])
  # reweight data
  reweightedData = reweightData(
    dataToWeight = dataToWeight,
    treeName     = treeName,
    variableName = binning.var.name,
    targetDistr  = targetDistr,
  )
  print(f"Writing reweighted data to file '{outFileName}'")
  reweightedData.Snapshot(treeName, outFileName)


if __name__ == "__main__":
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)   # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarized K- K_S Delta++ data

  inputDataDef = AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE
  # inputDatadef = AnalysisConfig.DataType.GENERATED_PHASE_SPACE
  # inputDatadef = 100000  # generate phase-space distribution in angles with given number of events
  dataBaseDirName = "./dataPhotoProdPiPi/polarized"
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
    # "Unpol",
  )
  maxLs = (
    4,
    # (4, 6),
    # (4, 8),
    # (4, 12),
    # (4, 16),
    # (4, 20),
    # 6,
    # 8,
    # (8, 16),
    # (8, 20),
    # 12,
    # 16,
    # 20,
  )
  reweightMassDistribution = False  # whether to reweight mass distribution after weighting with intensity function

  outFileDirBaseNameCommon = cfg.outFileDirBaseName
  for dataPeriod in dataPeriods:
    for tBinLabel in tBinLabels:
      for beamPolLabel in beamPolLabels:
        cfg.dataFileName       = f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/PiPi/data_flat_{beamPolLabel}.root"
        cfg.psAccFileName      = f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/PiPi/phaseSpace_acc_flat_{beamPolLabel}.root"
        cfg.psGenFileName      = f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/PiPi/phaseSpace_gen_flat_{beamPolLabel}.root"
        cfg.outFileDirBaseName = f"{outFileDirBaseNameCommon}/{dataPeriod}/{tBinLabel}/{beamPolLabel}"
        for maxL in maxLs:
          print(f"Generating weighted MC for data period '{dataPeriod}', t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
          cfg.maxL = maxL
          cfg.init()
          thisSourceFileName = os.path.basename(__file__)
          # create directory, into which weighted data will be written
          weightedDataDirName = f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/PiPi/weighted.maxL_{maxL}/{beamPolLabel}"
          Utilities.makeDirPath(weightedDataDirName)
          logFileName = f"{weightedDataDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
          print(f"Writing output to log file '{logFileName}'")
          with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
            Utilities.printGitInfo()
            timer = Utilities.Timer()
            ROOT.gROOT.SetBatch(True)
            setupPlotStyle()
            print(f"Using configuration:\n{cfg}")
            timer.start("Total execution time")
            momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_phys.pkl"
            # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_pwa_SPD.pkl"
            # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_JPAC.pkl"
            print(f"Reading moments from file '{momentResultsFileName}'")
            momentResults = MomentResultsKinematicBinning.loadPickle(momentResultsFileName)
            for momentResultsInBin in momentResults:
              massBinCenter = momentResultsInBin.binCenters[cfg.massBinning.var]
              massBinIndex  = cfg.massBinning.findBin(massBinCenter)
              assert massBinIndex is not None, f"Could not find bin for mass value of {massBinCenter} {cfg.massBinning.var.unit}"
              outFileName = f"{weightedDataDirName}/data_weighted_flat_bin_{massBinIndex}.root"
              print(f"Weighting events in mass bin {massBinIndex} at {massBinCenter:.{cfg.massBinning.var.nmbDigits}f} {cfg.massBinning.var.unit} by intensity function")
              weightDataWithIntensity(
                inputDataDef     = inputDataDef,
                massBinIndex     = massBinIndex,
                intensityFormula = momentResultsInBin.intensityFormula(
                  polarization                = cfg.polarization,
                  thetaFormula                = "theta",
                  phiFormula                  = "phi",
                  PhiFormula                  = "Phi",
                  printFormula                = False,
                  includeParityViolatingTerms = True,
                ),
                outFileName      = outFileName,
                cfg              = cfg,
                seed             = 12345 + massBinIndex,  # ensure rejection sampling and generated phase-space data in different mass bins are independent
              )

            # merge trees with weighted MC data for individual mass bins into single file
            mergedFileName  = f"{weightedDataDirName}/data_weighted_flat.root"
            nmbParallelJobs = 10
            with timer.timeThis(f"Time to merge ROOT files from all mass bins using hadd with {nmbParallelJobs} parallel jobs"):
              cmd = f"hadd -f -j {nmbParallelJobs} {mergedFileName} {weightedDataDirName}/data_weighted_flat_bin_*.root"
              print(f"Merging ROOT files from all mass bins: '{cmd}'")
              subprocess.run(cmd, shell = True)

            if reweightMassDistribution:
              # reweight mass distribution of merged file
              #TODO does not work for more than 1 data sample; call of RootUtilities.declareInCpp(weightsHist = weightsHist) crashes in ROOT
              reweightedFileName = f"{cfg.outFileDirName}/data_reweighted_flat.root"
              with timer.timeThis(f"Time to reweight mass distribution"):
                data = ROOT.RDataFrame(cfg.treeName, mergedFileName)
                reweightKinDistribution(
                  dataToWeight  = data,
                  treeName      = cfg.treeName,
                  binning       = cfg.massBinning,
                  momentResults = momentResults,
                  outFileName   = reweightedFileName,
                )

            timer.stop("Total execution time")
            print(timer.summary)
