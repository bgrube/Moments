#!/usr/bin/env python3
# calculates moments for eta pi0 real-data events

import functools
import threadpoolctl
from typing import (
  List,
)

import ROOT

from MomentCalculator import (
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
)
from PlottingUtilities import (
  HistAxisBinning,
  plotAngularDistr,
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
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 3):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of test case
    outFileDirName       = Utilities.makeDirPath("./plotsPhotoProdEtaPi0")
    treeName             = "etaPi0"
    dataFileName         = "./dataPhotoProdEtaPi0/data_flat.root"
    psAccFileName        = "./dataPhotoProdEtaPi0/phaseSpace_acc_flat.root"
    psGenFileName        = "./dataPhotoProdEtaPi0/phaseSpace_gen_flat.root"
    beamPolarization     = 0.4  #TODO get exact number
    # maxL                 = 1  # define maximum L quantum number of moments
    maxL                 = 5  # define maximum L quantum number of moments
    normalizeMoments     = False

    # load all signal and phase-space data
    print(f"Loading real data from tree '{treeName}' in file '{dataFileName}'")
    data = ROOT.RDataFrame(treeName, dataFileName)
    print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}'")
    dataPsAcc = ROOT.RDataFrame(treeName, psAccFileName)
    print(f"Loading generated phase-space data from tree '{treeName}' in file '{psGenFileName}'")
    dataPsGen = ROOT.RDataFrame(treeName, psGenFileName)
    # plot total angular distributions
    plotAngularDistr(dataPsAcc, dataPsGen, data, dataSignalGen = None, pdfFileNamePrefix = f"{outFileDirName}/angDistr_total_")

    # setup MomentCalculators for all mass bins
    momentIndices = MomentIndices(maxL)
    binVarMass    = KinematicBinningVariable(name = "mass", label = "#it{m}_{#it{#eta#pi}^{0}}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    massBinning   = HistAxisBinning(nmbBins = 17, minVal = 1.04, maxVal = 1.72, _var = binVarMass)
    momentsInBins:  List[MomentCalculator] = []
    nmbPsGenEvents: List[int]              = []
    assert len(massBinning) > 0, f"Need at least one mass bin, but found {len(massBinning)}"
    for massBinIndex, massBinCenter in enumerate(massBinning):
      massBinRange = massBinning.binValueRange(massBinIndex)
      print(f"Preparing {binVarMass.name} bin at {massBinCenter} {binVarMass.unit} with range {massBinRange} {binVarMass.unit}")

      # load data for mass bin
      binMassRangeFilter = f"(({massBinRange[0]} < {binVarMass.name}) && ({binVarMass.name} < {massBinRange[1]}))"
      print(f"Loading real data from tree '{treeName}' in file '{dataFileName}' and applying filter {binMassRangeFilter}")
      dataInBin = ROOT.RDataFrame(treeName, dataFileName).Filter(binMassRangeFilter)
      nmbDataEvents = dataInBin.Count().GetValue()
      print(f"Loaded {nmbDataEvents} data events")
      print(f"Loading accepted phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
      dataPsAccInBin = ROOT.RDataFrame(treeName, psAccFileName).Filter(binMassRangeFilter)
      nmbPsAccEvents = dataPsAccInBin.Count().GetValue()
      print(f"Loading generated phase-space data from tree '{treeName}' in file '{psAccFileName}' and applying filter {binMassRangeFilter}")
      dataPsGenInBin = ROOT.RDataFrame(treeName, psGenFileName).Filter(binMassRangeFilter)
      nmbPsGenEvents.append(dataPsGenInBin.Count().GetValue())
      print(f"Loaded phase-space events: number generated = {nmbPsGenEvents[-1]}, number accepted = {nmbPsAccEvents}"
            f" -> efficiency = {nmbPsAccEvents / nmbPsGenEvents[-1]:.3f}")

      # setup moment calculator
      dataSet = DataSet(beamPolarization, dataInBin, phaseSpaceData = dataPsAccInBin, nmbGenEvents = nmbPsGenEvents[-1])
      momentsInBins.append(MomentCalculator(momentIndices, dataSet, integralFileBaseName = f"{outFileDirName}/integralMatrix", binCenters = {binVarMass : massBinCenter}))

      # plot angular distributions for mass bin
      plotAngularDistr(dataPsAccInBin, dataPsGenInBin, dataInBin, dataSignalGen = None, pdfFileNamePrefix = f"{outFileDirName}/angDistr_{'_'.join(momentsInBins[-1].binLabels)}_")
    moments = MomentCalculatorsKinematicBinning(momentsInBins)

    timer.stop("Total execution time")
    print(timer.summary)
