#!/usr/bin/env python3
"""
Test script that performs closure test for moment analysis that uses
extended maximum likelihood-fitting of the data.
"""


from __future__ import annotations

import functools
import numpy as np
import os
import threadpoolctl


import ROOT
import iminuit as im
from wurlitzer import pipes, STDOUT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  MomentResult,
  QnWaveIndex,
)
from PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
from testMomentsPhotoProd import genDataFromWaves
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  # set parameters of test case
  nmbPwaMcEvents   = 1000000 # number of "data" events to generate from partial-wave amplitudes
  nmbPsMcEvents    = 100000  # number of phase-space events to generate
  beamPolarization = 1.0     # polarization of photon beam
  maxL             = 4       # maximum L quantum number of moments
  outputDirName    = Utilities.makeDirPath("./plotsTestFitMoments")

  thisSourceFileName = os.path.basename(__file__)
  # logFileName = f"{outputDirName}/{os.path.splitext(thisSourceFileName)[0]}.log"
  # print(f"Writing output to log file '{logFileName}'")
  # with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
  if True:
    print(f"Using iminuit version {im.__version__}")
    Utilities.printGitInfo()
    timer = Utilities.Timer()
    ROOT.gROOT.SetBatch(True)
    ROOT.gRandom.SetSeed(1234567890)
    setupPlotStyle()
    threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
    print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
    with threadController.limit(limits = 4):
      print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")
      timer.start("Total execution time")

      # define angular distribution of signal
      partialWaveAmplitudesSig: tuple[AmplitudeValue, ...] = (  # set of all possible partial waves up to ell = 2
        # negative-reflectivity waves
        AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.4 + 0.1j),  # P_-1^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m =  0), val =  0.3 - 0.8j),  # P_0^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.8 + 0.7j),  # P_+1^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -2), val =  0.1 - 0.4j),  # D_-2^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val =  0.5 + 0.2j),  # D_-1^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val = -0.1 - 0.2j),  # D_ 0^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val =  0.2 - 0.1j),  # D_+1^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.2 + 0.3j),  # D_+2^-
        # positive-reflectivity waves
        AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = -1), val =  0.5 - 0.1j),  # P_-1^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.8 - 0.3j),  # P_0^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.6 + 0.3j),  # P_+1^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.2 + 0.1j),  # D_-2^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -1), val =  0.2 - 0.3j),  # D_-1^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m =  0), val =  0.1 - 0.2j),  # D_ 0^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +1), val =  0.2 + 0.5j),  # D_+1^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val = -0.3 - 0.1j),  # D_+2^+
      )
      amplitudeSetSig = AmplitudeSet(partialWaveAmplitudesSig)

      # calculate true moment values and generate data from partial-wave amplitudes
      HTruth: MomentResult = amplitudeSetSig.photoProdMomentSet(maxL)
      print(f"True moment values\n{HTruth}")
      with timer.timeThis("Time to generate MC data from partial waves"):
        dataPwaModel = genDataFromWaves(
          nmbEvents         = nmbPwaMcEvents,
          polarization      = beamPolarization,
          amplitudeSet      = amplitudeSetSig,
          efficiencyFormula = None,
          regenerateData    = True,
          outFileNamePrefix = f"{outputDirName}/",
        )
        # plot data generated from partial-wave amplitudes
        canv = ROOT.TCanvas()
        nmbBins = 25
        hist = dataPwaModel.Histo3D(
          ROOT.RDF.TH3DModel("hData", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
          "cosTheta", "phiDeg", "PhiDeg")
        hist.SetMinimum(0)
        hist.GetXaxis().SetTitleOffset(1.5)
        hist.GetYaxis().SetTitleOffset(2)
        hist.GetZaxis().SetTitleOffset(1.5)
        hist.Draw("BOX2Z")
        canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")

      # construct intensity formula with moments as parameters
      intensityFormula = HTruth.intensityFormula(
        polarization     = beamPolarization,
        thetaFormula     = "x",
        phiFormula       = "y",
        PhiFormula       = "z",
        printFormula     = True,
        useMomentSymbols = True,
      )
      intensityFcn = ROOT.TF3("intensityMoments", intensityFormula, 0, np.pi, -np.pi, +np.pi, -np.pi, +np.pi)
      for qnIndex in HTruth.indices.qnIndices:
        Hval = HTruth[qnIndex].val
        print(f"{qnIndex.label} = {Hval}")
        intensityFcn.SetParameter(qnIndex.label, Hval.imag if qnIndex.momentIndex == 2 else Hval.real)
      intensityFcn.SetTitle(";#theta [rad];#phi [rad];#Phi [rad]")
      intensityFcn.SetNpx(100)
      intensityFcn.SetNpy(100)
      intensityFcn.SetNpz(100)
      intensityFcn.SetMinimum(0)
      drawTF3(
        fcn         = intensityFcn,
        binnings    = (HistAxisBinning(25, 0, np.pi), HistAxisBinning(25, -np.pi, +np.pi), HistAxisBinning(25, -np.pi, +np.pi)),
        pdfFileName = f"{outputDirName}/{intensityFcn.GetName()}.pdf",
      )

      timer.stop("Total execution time")
      print(timer.summary)
