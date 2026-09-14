#!/usr/bin/env python3
"""
This module performs analysis of polarized pi+ pi- high-energy data.

Usage: Run this module as a script to perform the analysis.
"""


from __future__ import annotations

from copy import deepcopy
import functools

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from scripts.calculateMoments import calculateMoments
from scripts.convertInputData import convertInputData
from scripts.plotKinematicDistributions import plotKinematicDistributions
from scripts.plotMoments import plotMoments
from workflow.AnalysisConfig import (
  AnalysisConfig,
  CFG_POLARIZED_PIPI,
)
from workflow.DataConversionUtilities import (
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  CPP_CODE_TWO_BODY_ANGLES,
  lorentzVectors,
)
from workflow.PlottingUtilities import setupPlotStyle
from workflow.RootUtilities import loadBasisFunctionsLibrary
from workflow import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  timer = Utilities.Timer()
  timer.start("Total time for analysis")
  Utilities.printGitInfo()
  ROOT.gROOT.SetBatch(True)
  loadBasisFunctionsLibrary()  # initializes OpenMP and loads `cpp/basisFunctions.C`
  setupPlotStyle()

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)

  cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized gamma p -> (pi+ pi-) p data
  additionalColumnDefs: dict[AnalysisConfig.DataType, dict[str, str]] = {  # additional columns for each data type
    AnalysisConfig.DataType.REAL_DATA             : {},
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : {},
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : {},
  }
  additionalFilterDefs: dict[AnalysisConfig.DataType, list[str]] = {  # additional filters for each data type
    AnalysisConfig.DataType.REAL_DATA             : [],
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : [],
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : [],
  }

  if True:
  # if False:
    print("\n=== Step 1: convert input data into format required by `MomentCalculator` =====")
    if False:  # cut away forward tracks in reconstructed data
      for inputDataType in (AnalysisConfig.DataType.REAL_DATA, AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE):
        lvs = lorentzVectors(dataFormat = AnalysisConfig.DataFormat.ALEX)
        additionalColumnDefs[inputDataType] |= {
          "DistFdcPip": f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvs['pip']})",
          "DistFdcPim": f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvs['pim']})",
        }
        additionalFilterDefs[inputDataType] += ["(DistFdcPip > 4) and (DistFdcPim > 4)"]  # require minimum distance of tracks at FDC position [cm]
    convertInputData(
      cfg                  = cfg,
      additionalColumnDefs = additionalColumnDefs,
      additionalFilterDefs = additionalFilterDefs,
    )

  if True:
  # if False:
    print("\n=== Step 1A: plot kinematic distributions =====================================")
    ROOT.EnableImplicitMT()
    additionalFilterDefs = {  # kinematic range used in SDME analysis; for 2017_01_ver05 data
      AnalysisConfig.DataType.REAL_DATA             : ["(0.60 < massPiPi and massPiPi < 0.88)"],
      AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : ["(0.60 < massPiPi and massPiPi < 0.88)"],
      AnalysisConfig.DataType.GENERATED_PHASE_SPACE : ["(0.60 < massPiPi and massPiPi < 0.88)"],
    }
    # subsystemMassBinning = None  # do not generate plots in mass bins
    subsystemMassBinning = cfg.massBinning
    plotKinematicDistributions(
      cfg                  = cfg,
      additionalColumnDefs = additionalColumnDefs,
      additionalFilterDefs = additionalFilterDefs,
      subsystemMassBinning = subsystemMassBinning,
    )

  if True:
  # if False:
    print("\n=== Step 2: calculate moments using `MomentCalculator` ========================")
    calculateMoments(
      cfg                            = cfg,
      additionalColumnDefs           = {},
      additionalCuts                 = (
        # "(0.100 < minusT and minusT < 0.114)",
      ),
      forceIntegralMatrixCalculation = True,
    )

  if True:
  # if False:
    print("\n=== Step 3: plot moments ======================================================")
    # cfg.polarization = None  # treat data as unpolarized
    cfg.plotMomentsInBins = True
    # cfg.plotAccIntegralMatrices = True
    # cfg.plotMeasuredMoments = True
    plotMoments(
      cfg                         = cfg,
      scaleFactorPhysicalMoments  = 1.0,  # no scaling
      # compareTo                   = None,
      # compareTo                   = ComparisonMomentsType.PWA,
      compareTo                   = ("./plots/PiPiPol/2017_01_ver05/tbin_0.100_0.114/PARA_0.maxL_6/unnorm_moments_phys_shifted.pkl", "True Values"),
      normalizeComparisonMoments  = True,  # scale comparison moments to estimated moments
      # plotComparisonMomentsUncert = True,
      outFileType                 = "pdf",
      # outFileType                 = "root",
      yAxisUnit                   = "",
    )

  timer.stop("Total time for analysis")
  print(timer.summary)
