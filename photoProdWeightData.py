#!/usr/bin/env python3
"""
This module weights accepted phase-space data with the intensity
calculated from the results of the moment analysis of unpolarized and
polarized pi+ pi- photoproduction data. The moment values are read
from files produced by the script `photoProdCalcMoments.py` that
calculates the moments.

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


def weightDataWithIntensity(
  intensityFormula: str,  # formula for intensity function
  massBinIndex:     int,  # index of mass bin to generate data for
  outFileName:      str,  # ROOT file to which weighted events are written
  cfg:              AnalysisConfig,
  inputDataType:    AnalysisConfig.DataType | None = None,    # if `None`, phase-space distribution in angles is generated
  nmbGenPsEvents:   int                            = 100000,  # number phase-space events to generate
  seed:             int                            = 123456789,
) -> None:
  """Weight phase-space MC data with given intensity formula"""
  kinematicBinFilter: str = cfg.massBinning.binFilter(massBinIndex)
  kinematicBinRange: tuple[float, float] = cfg.massBinning.binValueRange(massBinIndex)
  # get input data
  dataToWeight: ROOT.RDataFrame | None = None
  if inputDataType is None:
    print(f"Generating phase-space distribution with {nmbGenPsEvents} events")
    ROOT.gRandom.SetSeed(seed)
    dataToWeight = (
      ROOT.RDataFrame(nmbGenPsEvents)
          .Define("cosTheta", "(Double32_t)gRandom->Uniform(-1, +1)")
          .Define("theta",    "(Double32_t)std::acos(cosTheta)")
          .Define("phiDeg",   "(Double32_t)gRandom->Uniform(-180, +180)")
          .Define("phi",      "(Double32_t)phiDeg * TMath::DegToRad()")
          .Define("mass",    f"(Double32_t)gRandom->Uniform({kinematicBinRange[0]}, {kinematicBinRange[1]})")
          .Filter('if (rdfentry_ == 0) { cout << "Running event loop in weightAccPhaseSpaceWithIntensity()" << endl; } return true;')  # noop filter that just logs when event loop is running
    )
    if cfg.polarization is not None:
      # polarized case: add Phi and polarization columns
      dataToWeight = (
        dataToWeight.Define("PhiDeg",   "(Double32_t)gRandom->Uniform(-180, +180)")
                    .Define("Phi",      "(Double32_t)PhiDeg * TMath::DegToRad()")
      )
      if isinstance(cfg.polarization, float):
        dataToWeight = dataToWeight.Define("beamPol", f"(Double32_t){cfg.polarization}")
      elif isinstance(cfg.polarization, str):
        raise ValueError(f"Cannot read polarization from column '{cfg.polarization}'")
  else:
    print(f"Loading data of type '{inputDataType}'")
    dataToWeight = cfg.loadData(inputDataType)
    assert dataToWeight is not None, f"Could not load data of type '{inputDataType}'"
    dataToWeight = dataToWeight.Filter(kinematicBinFilter)
    nmbInputEvents = dataToWeight.Count().GetValue()
    print(f"Input data contain {nmbInputEvents} events in bin '{kinematicBinFilter}'")
  # calculate intensity weight and random number in [0, 1] for each event
  print(f"Calculating weights using formula '{intensityFormula}'")
  # ROOT.gRandom.SetSeed(seed)
  dataToWeight = (
    dataToWeight.Define("intensityWeight", f"(Double32_t){intensityFormula}")
                .Define("intensityRndNmb",  "(Double32_t)gRandom->Rndm()")  # random number in [0, 1] for each event
  )
  tmpFileName = f"{outFileName}.phaseSpace.root"
  dataToWeight.Snapshot(cfg.treeName, tmpFileName)  # write unweighted data to file to ensure that random columns are filled only once
  dataToWeight = ROOT.RDataFrame(cfg.treeName, tmpFileName)  # read data back
  # determine maximum weight
  maxIntensityWeight = dataToWeight.Max("intensityWeight").GetValue()
  print(f"Maximum intensity is {maxIntensityWeight}")
  # apply weights by accepting each event with probability intensityWeight / maxIntensityWeight
  weightedData = (
    dataToWeight.Define("acceptEventIntensityWeight", f"(bool)(intensityRndNmb < (intensityWeight / {maxIntensityWeight}))")
                .Filter("acceptEventIntensityWeight == true")
  )
  nmbWeightedEvents = weightedData.Count().GetValue()
  print(f"After weighting with the intensity function, the sample contains {nmbWeightedEvents} accepted events; generator efficiency is {nmbWeightedEvents / nmbGenPsEvents}")
  # write weighted data to file
  print(f"Writing data weighted with intensity function to file '{outFileName}'")
  weightedData.Snapshot(cfg.treeName, outFileName)  #TODO write out only essential columns
  subprocess.run(f"rm -fv {tmpFileName}", shell = True)


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
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)  # perform analysis of unpolarized pi+ pi- data
  cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarized K- K_S Delta++ data

  tBinLabels = (
    # "tbin_0.1_0.5",
    "tbin_0.4_0.5",
  )
  beamPolLabels = (
    # "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
    "Unpol",
  )
  maxLs = (
    4,
    # 5,
    # 6,
    # 7,
    # 8,
  )
  # cfg.psGenFileName = "./dataPhotoProdPiPiUnpolJPAC/ideal/phaseSpace_gen_ideal_flat.PiPi.root"

  outFileDirBaseNameCommon = cfg.outFileDirBaseName
  # outFileDirBaseNameCommon = f"{cfg.outFileDirBaseName}.ideal"
  for tBinLabel in tBinLabels:
    for beamPolLabel in beamPolLabels:
      cfg.outFileDirBaseName = f"{outFileDirBaseNameCommon}.{tBinLabel}/{beamPolLabel}"
      for maxL in maxLs:
        print(f"Generating weighted MC for t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
        cfg.maxLPhys = maxL
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
          outFileBaseName = f"{cfg.outFileDirName}/data_weighted_flat"
          # outFileBaseName = f"{cfg.outFileDirName}/data_weighted_pwa_SPD_flat"
          momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_phys.pkl"
          # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_pwa_SPD.pkl"
          # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_JPAC.pkl"
          print(f"Reading moments from file '{momentResultsFileName}'")
          momentResults = MomentResultsKinematicBinning.loadPickle(momentResultsFileName)
          for momentResultsForBin in momentResults:
            massBinCenter = momentResultsForBin.binCenters[cfg.massBinning.var]
            massBinIndex  = cfg.massBinning.findBin(massBinCenter)
            assert massBinIndex is not None, f"Could not find bin for mass value of {massBinCenter} GeV"
            outFileName = f"{outFileBaseName}_{massBinIndex}.root"
            print(f"Weighting events for bin {massBinIndex} at {massBinCenter:.2f} {cfg.massBinning.var.unit} weighted by intensity function")
            weightDataWithIntensity(
              intensityFormula = momentResultsForBin.intensityFormula(  #TODO include imaginary parts into intensity formula
                polarization = cfg.polarization,
                thetaFormula = "theta",
                phiFormula   = "phi",
                PhiFormula   = "Phi",
                printFormula = False,
              ),
              massBinIndex     = massBinIndex,
              outFileName      = outFileName,
              cfg              = cfg,
              # inputDataType    = AnalysisConfig.DataType.GENERATED_PHASE_SPACE,
              # inputDataType    = AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE,
              inputDataType    = None,  # generate phase-space distribution in angles
              nmbGenPsEvents   = 100000,
              seed             = 12345 + massBinIndex,  # ensure random data in mass bins are independent
            )

          # merge trees with weighted MC data from different mass bins into single file
          mergedFileName = f"{outFileBaseName}.root"
          nmbParallelJobs = 10
          with timer.timeThis(f"Time to merge ROOT files from all mass bins using hadd with {nmbParallelJobs} parallel jobs"):
            cmd = f"hadd -f -j {nmbParallelJobs} {mergedFileName} {outFileBaseName}_*.root"
            print(f"Merging ROOT files from all mass bins: '{cmd}'")
            subprocess.run(cmd, shell = True)

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
