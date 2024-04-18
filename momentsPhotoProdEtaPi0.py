#!/usr/bin/env python3
# calculates moments for eta pi0 real-data events

import functools
import threadpoolctl

import ROOT

from PlottingUtilities import (
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

    timer.stop("Total execution time")
    print(timer.summary)
