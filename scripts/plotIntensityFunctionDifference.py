#!/usr/bin/env python3
"""
This module plots the difference of two intensity distributions that
correspond to the moments estimated from data. The moment values are
read from files produced by the function defined in
`calculateMoments.py` that calculates the moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

# from copy import deepcopy
# import ctypes
import functools
# import numpy as np
# import os
# from scipy.optimize import minimize

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from moments.MomentCalculator import (
  MomentResult,
  MomentResultsKinematicBinning,
)
from scripts.plotIntensityFunctions import IntensityFunctor
from workflow.AnalysisConfig import (
#   AnalysisConfig,
  BeamPolInfo,
  BEAM_POL_INFOS,
#   CFG_POLARIZED_ETAPI0,
#   CFG_POLARIZED_PIPI,
)
from workflow.PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
  setupPlotStyle,
#   TF3toTH3,
)
from workflow import RootUtilities
from workflow import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def plotIntensityFunctionDifference(
  momentResults:     tuple[MomentResult, MomentResult],
  massBinIndex:      int,
  beamPolInfo:       BeamPolInfo | None,
  outputDirPath:     str,
  nmbBinsPerAxis:    int                             = 25,
  useIntensityTerms: MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.ALL,
  titleSuffix:       str                             = "",
  nameSuffix:        str                             = "",
  coordSysLabel:     str                             = "HF",
) -> None:
  """Plots difference of intensity functions that correspond to the given moment results in the given mass bin and writes plots to output directory"""
  print(f"Plotting difference of intensity functions for mass bin {massBinIndex} using {beamPolInfo} and intensity terms {useIntensityTerms.value}")
  # formula uses variables: x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
  intensityFormulas = tuple(
    momentResult.intensityFormula(
      polarization      = beamPolInfo.pol if beamPolInfo is not None else None,
      thetaFormula      = "std::acos(x)",
      phiFormula        = "TMath::DegToRad() * y",
      PhiFormula        = "TMath::DegToRad() * z",
      useIntensityTerms = useIntensityTerms,
    ) for momentResult in momentResults
  )
  ROOT.gStyle.SetImageScaling(3)  # improve bitmap rendering quality by tripling the resolution; default is 1
  intensityFcnDiff = ROOT.TF3(f"intensityFcnDiff_{useIntensityTerms.value}_bin_{massBinIndex}{nameSuffix}", f"({intensityFormulas[0]})-({intensityFormulas[1]})", -1, +1, -180, +180, -180, +180)
  binnings = (
    HistAxisBinning(nmbBinsPerAxis,   -1,   +1),  # cos(theta)
    HistAxisBinning(nmbBinsPerAxis, -180, +180),  # phi
    HistAxisBinning(nmbBinsPerAxis, -180, +180),  # Phi
  )
  drawTF3(
    fcn                = intensityFcnDiff,
    binnings           = binnings,
    outFilePath        = f"{outputDirPath}/{intensityFcnDiff.GetName()}.png",
    histTitle          = f"Intensity Difference{titleSuffix};cos#theta_{{{coordSysLabel}}};#phi_{{{coordSysLabel}}} [deg];#Phi [deg]",
    showNegativeValues = True,
  )


if __name__ == "__main__":
  Utilities.printGitInfo()
  ROOT.gROOT.SetBatch(True)
  RootUtilities.loadBasisFunctionsLibrary()  # initializes OpenMP and loads `cpp/basisFunctions.C`
  setupPlotStyle()

  momentResultsFilePaths = (
    f"./plots/PiPiPol/2017_01_ver05/tbin_0.100_0.114/PARA_0.maxL_4/unnorm_moments_phys_shifted.pkl",
    f"./plots/PiPiPol/2017_01_ver05/tbin_0.100_0.114/PARA_0.maxL_6/unnorm_moments_phys_shifted.pkl",
  )
  print(f"Reading moments from file '{momentResultsFilePaths}'")
  momentResults: tuple[MomentResultsKinematicBinning, MomentResultsKinematicBinning] = (
    MomentResultsKinematicBinning.loadPickle(momentResultsFilePaths[0]),
    MomentResultsKinematicBinning.loadPickle(momentResultsFilePaths[1]),
  )
  for massBinIndex in range(len(momentResults[0])):  # assume both momentResults have the same number of mass bins
    momentResultsInBin: tuple[MomentResult, MomentResult] = (
      momentResults[0][massBinIndex],
      momentResults[1][massBinIndex],
    )

    plotIntensityFunctionDifference(
      momentResults     = momentResultsInBin,
      massBinIndex      = massBinIndex,
      beamPolInfo       = BEAM_POL_INFOS["2017_01"]["PARA_0"],
      outputDirPath     = "./",
      nmbBinsPerAxis    = 25,
      useIntensityTerms = MomentResult.IntensityTermsType.PARITY_CONSERVING,
      titleSuffix       = ": L_{max} = 4 #minus L_{max} = 6",
      nameSuffix        = "_maxL_4_6",
      coordSysLabel     = "HF",
    )
