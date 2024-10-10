#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict
import functools
import json
import threadpoolctl
from typing import Any

import ROOT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  MomentResult,
  QnWaveIndex,
)
from PlottingUtilities import setupPlotStyle
import RootUtilities  # importing initializes OpenMP and loads basisFunctions.C
from testMomentsPhotoProd import (
  genAccepted2BodyPsPhotoProd,
  genDataFromWaves,
)
from Utilities import (
  makeDirPath,
  printGitInfo,
  Timer,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  printGitInfo()
  timer = Timer()
  ROOT.gROOT.SetBatch(True)
  ROOT.gRandom.SetSeed(1234567890)
  setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 5):  # use maximum of 5 threads
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of test case
    outFileDirName   = makeDirPath("./plotsTestNizar")
    nmbPwaMcEvents   = 10000    # number of "data" events to generate from partial-wave amplitudes
    nmbPsMcEvents    = 1000000  # number of phase-space events to generate
    beamPolarization = 1.0      # polarization of photon beam
    maxL             = 5        # maximum L quantum number of moments
    partialWaveAmplitudes: tuple[AmplitudeValue, ...] = (  # set of all possible partial waves up to ell = 2
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
    amplitudeSet: AmplitudeSet = AmplitudeSet(partialWaveAmplitudes)

    # calculate true moment values and generate data from partial-wave amplitudes
    t = timer.start("Time to generate MC data from partial waves")
    HTrue: MomentResult = amplitudeSet.photoProdMomentSet(maxL, normalize = True)
    print(f"True moment values\n{HTrue}")
    HTrueJsonFileName = f"{outFileDirName}/trueMomentValues.json"
    print(f"Writing true moment values to file {outFileDirName}/{HTrueJsonFileName}")
    # convert MomentResult objects to tuple of dictionaries and write to JSON file
    momentMemberVarsToRemove = ("binCenters", "label", "bsSamples", "uncertRe", "uncertIm")
    HTrueDicts: list[dict[str, Any]] = []
    for moment in HTrue.values:
      HTrueDict = {}
      for key, value in asdict(moment).items():
        if key not in momentMemberVarsToRemove:
          # export only selected member variables
          if isinstance(value, complex):
            # since JSON does not have builtin support for complex numbers,
            # expand complex numbers into real and imaginary parts
            HTrueDict[f"{key}_Re"] = value.real
            HTrueDict[f"{key}_Im"] = value.imag
          else:
            HTrueDict[key] = value
      HTrueDicts.append(HTrueDict)
    with open(HTrueJsonFileName, "w") as HTrueJsonFile:
      json.dump(HTrueDicts, HTrueJsonFile, indent = 4, default = str)
    dataPwaModel = genDataFromWaves(
      nmbEvents         = nmbPwaMcEvents,
      polarization      = beamPolarization,
      amplitudeSet      = amplitudeSet,
      efficiencyFormula = None,
      outFileNamePrefix = f"{outFileDirName}/",
      regenerateData    = True,
    )
    t.stop()

    # plot data generated from partial-wave amplitudes
    canv = ROOT.TCanvas()
    nmbBins = 25
    hist = dataPwaModel.Histo3D(
      ROOT.RDF.TH3DModel("hData", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180), "cosTheta", "phiDeg", "PhiDeg")
    hist.SetMinimum(0)
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
    canv.SaveAs(f"{outFileDirName}/{hist.GetName()}.pdf")

    # generate phase-space data for perfect acceptance
    t = timer.start("Time to generate phase-space MC data")
    dataAcceptedPs = genAccepted2BodyPsPhotoProd(
      nmbEvents         = nmbPsMcEvents,
      efficiencyFormula = None,
      outFileNamePrefix = f"{outFileDirName}/",
      regenerateData    = True
    )
    t.stop()

    timer.stop("Total execution time")
    print(timer.summary)
