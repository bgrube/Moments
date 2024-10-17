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


def weightAccPhaseSpaceWithIntensity(
  polarization:          float,         # photon-beam polarization
  partialWaveAmplitudes: AmplitudeSet,  # partial-wave amplitudes
  isSignal:              bool,          # flag to indicate whether signal or background intensity is being applied
  accPhaseSpaceFileName: str,           # name of file with accepted phase-space MC data
  accPhaseSpaceTreeName: str = "kin",   # name of tree with accepted phase-space MC data
):
  """Weight accepted phase-space MC data with intensity calculated from partial-wave amplitudes"""
  # read accepted phase-space MC data
  accPhaseSpaceData = ROOT.RDataFrame(accPhaseSpaceTreeName, accPhaseSpaceFileName)
  nmbAccPhaseSpaceEvents = accPhaseSpaceData.Count().GetValue()
  # construct formula for intensity calculation
  intensityFormula = partialWaveAmplitudes.intensityFormula(
    polarization = polarization,
    thetaFormula = "std::acos(cosTheta_eta_hel)",
    phiFormula   = "phi_eta_hel * TMath::DegToRad()",
    PhiFormula   = "Phi * TMath::DegToRad()",
    printFormula = True,
  )
  # calculate intensity weight and random number in [0, 1] for each event
  accPhaseSpaceData = accPhaseSpaceData.Define("intensityWeight", f"(double){intensityFormula}") \
                                       .Define("rndNmb",          "(double)gRandom->Rndm()")
  # determine maximum weight
  maxIntensityWeight = accPhaseSpaceData.Max("intensityWeight").GetValue()
  print(f"Maximum intensity is {maxIntensityWeight}")
  # accept each event with probability intensityWeight / maxIntensityWeight
  data = accPhaseSpaceData.Define("acceptEvent", f"(bool)(rndNmb < (intensityWeight / {maxIntensityWeight}))") \
                          .Filter("acceptEvent == true")
  nmbWeightedEvents = data.Count().GetValue()
  print(f"Sample contains {nmbWeightedEvents} events after intensity weighting; efficiency is {nmbWeightedEvents / nmbAccPhaseSpaceEvents}")
  # add columns with discriminating variables for sideband subtraction
  massPi0Mean   = 0.135881  # [GeV]
  massPi0Sigma  = 0.0076    # [GeV]
  massEtaMean   = 0.548625  # [GeV]
  massEtaSigma  = 0.0191    # [GeV]
  # the sideband boundaries are:
  #   pi0: from (massPi0Mean +- 3.0 * massPi0Sigma) to (massPi0Mean +- 5.5 * massPi0Sigma)
  #   eta: from (massEtaMean +- 3.0 * massEtaSigma) to (massEtaMean +- 6.0 * massEtaSigma)
  if isSignal:
    data = data.Define("massPi0", f"gRandom->Gaus({massPi0Mean}, {massPi0Sigma})") \
               .Define("massEta", f"gRandom->Gaus({massEtaMean}, {massEtaSigma})")
  else:
    data = data.Define("massPi0", f"gRandom->Uniform({massPi0Mean - 5.5 * massPi0Sigma}, {massPi0Mean + 5.5 * massPi0Sigma})") \
               .Define("massEta", f"gRandom->Uniform({massEtaMean - 6.0 * massEtaSigma}, {massEtaMean + 6.0 * massEtaSigma})")
  # write weighted data to file
  outFileName = f"{accPhaseSpaceFileName}.intensityWeighted.{'sig' if isSignal else 'bkg'}"
  print(f"Writing accepted phase-space data that were weighted with intensity function to file {outFileName}")
  data.Snapshot(accPhaseSpaceTreeName, f"{outFileName}")


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
    accPhaseSpaceFileName = "./dataTestNizar/F2017_1_selected_TestMomentFit_data_flat.root"
    outFileDirName        = makeDirPath("./plotsTestNizar")
    nmbPwaMcEvents        = 10000    # number of "data" events to generate from partial-wave amplitudes
    nmbPsMcEvents         = 1000000  # number of phase-space events to generate
    beamPolarization      = 1.0      # polarization of photon beam
    maxL                  = 5        # maximum L quantum number of moments
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
    # define angular distribution of background
    partialWaveAmplitudesBkg: tuple[AmplitudeValue, ...] = (  # set of all possible waves up to ell = 2
      # negative-reflectivity waves
      AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.9 + 0.7j),  # P_-1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m =  0), val = -0.6 + 0.4j),  # P_0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.9 - 0.8j),  # P_+1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -2), val = -1.0 - 0.7j),  # D_-2^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val = -0.8 - 0.7j),  # D_-1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val =  0.4 + 0.3j),  # D_ 0^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val = -0.6 - 0.1j),  # D_+1^-
      AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.1 - 0.9j),  # D_+2^-
      # positive-reflectivity waves
      AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = -1), val = -1.0 + 0.8j),  # P_-1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.2 + 0.2j),  # P_0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.0 - 0.3j),  # P_+1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.7 + 0.9j),  # D_-2^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -1), val = -0.4 - 0.5j),  # D_-1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m =  0), val = -0.3 + 0.2j),  # D_ 0^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +1), val = -1.0 - 0.4j),  # D_+1^+
      AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val =  0.5 - 0.2j),  # D_+2^+
    )
    amplitudeSetBkg = AmplitudeSet(partialWaveAmplitudesBkg)

    # calculate true moment values and generate data from partial-wave amplitudes
    t = timer.start("Time to generate MC data from partial waves")
    HTrue: MomentResult = amplitudeSetSig.photoProdMomentSet(maxL, normalize = True)
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
      amplitudeSet      = amplitudeSetSig,
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

    # weight accepted phase-space MC data with intensity calculated from partial-wave amplitudes
    t = timer.start("Time to weight accepted phase-space MC data")
    weightAccPhaseSpaceWithIntensity(
      polarization          = beamPolarization,
      partialWaveAmplitudes = amplitudeSetSig,
      isSignal              = True,
      accPhaseSpaceFileName = accPhaseSpaceFileName,
    )
    weightAccPhaseSpaceWithIntensity(
      polarization          = beamPolarization,
      partialWaveAmplitudes = amplitudeSetBkg,
      isSignal              = False,
      accPhaseSpaceFileName = accPhaseSpaceFileName,
    )
    t.stop()

    timer.stop("Total execution time")
    print(timer.summary)
