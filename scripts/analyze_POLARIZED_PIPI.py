#!/usr/bin/env python3
"""
This module performs analysis of polarized pi+ pi- high-energy data.

Usage: Run this module as a script to perform the analysis.
"""


from __future__ import annotations

from copy import deepcopy
import functools

import ROOT

from scripts.plotWeightedMc import plotWeightedMc
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from moments.MomentCalculator import MomentResult
from scripts.calculateMoments import calculateMoments
from scripts.convertInputData import convertInputData
from scripts.overlayMoments import (
  overlayMoments,
  ResultToOverlay,
)
from scripts.plotIntensityFunctions import plotIntensityFunctions
from scripts.plotKinematicDistributions import plotKinematicDistributions
from scripts.plotMoments import plotMoments
from scripts.weightDataWithMoments import weightDataWithMoments
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
    # ROOT.EnableImplicitMT()  #TODO multi-threading leads to non-reproducible results when calculating moments in step 2
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

  if True:
  # if False:
    print("\n=== Step 4: overlay moments ==================================================")
    normToFirstResult = True  # if set moments are normalized to H_0(0, 0) of first moment result
    # normToFirstResult = False
    for dataPeriod in cfg.dataPeriods:
      for tBinLabel in cfg.tBinLabels:
        resultsToOverlay: tuple[ResultToOverlay, ...] = (  # last moment result in this tuple defines, which moments are plotted
          ResultToOverlay("./plots/PiPiPol/2017_01_ver05/tbin_0.100_0.114/PARA_0.maxL_6/unnorm_moments_phys_shifted.pkl",                             "Truth L_{max} = 6"),
          ResultToOverlay(f"{cfg.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel = 'PARA_0', maxL = 4)}/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 4"),
          ResultToOverlay(f"{cfg.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel = 'PARA_0', maxL = 6)}/{cfg.outFileNamePrefix}_moments_phys.pkl", "L_{max} = 6"),
          ResultToOverlay("./plots/PiPiPol/2017_01_ver05/tbin_0.100_0.114/PARA_0.maxL_4/unnorm_moments_phys.pkl",                                     "Real data L_{max} = 4"),
        )
        outputDirPath = Utilities.makeDirPath(f"{cfg.outFileDirBasePath}/{dataPeriod}/{tBinLabel}.overlay")
        overlayMoments(
          cfg               = cfg,
          resultsToOverlay  = resultsToOverlay,
          outputDirPath     = outputDirPath,
          normToFirstResult = normToFirstResult,
        )

  if True:
  # if False:
    print("\n=== Step 5: plot intensity functions and make them positive definite ==========")
    scaleFactor = None
    # scaleFactor = 1.6112841143413135  # gen MC weighted with L_max = 4 and analyzed with L_max = 4, 6, 8
    # scaleFactor = 2.450175524066058   # acc MC weighted with L_max = 4 and analyzed with L_max = 4
    # scaleFactor = 2.4515044898120957  # acc MC weighted with L_max = 4 and analyzed with L_max = 6
    # scaleFactor = 2.441922028485739   # acc MC weighted with L_max = 4 and analyzed with L_max = 8
    #
    # scaleFactor = 1.682258789616807  # gen MC weighted with L_max = 6 and analyzed with L_max = 4, 6, 8
    # scaleFactor = 2.5108970501733427  # acc MC weighted with L_max = 6 and analyzed with L_max = 4
    # scaleFactor = 2.5735512097120283  # acc MC weighted with L_max = 6 and analyzed with L_max = 6
    # scaleFactor = 2.582279973210192   # acc MC weighted with L_max = 6 and analyzed with L_max = 8
    plotIntensityFunctions(
      cfg                      = cfg,
      momentType               = "phys",
      # makeIntensityPosDefinite = True,
      makeIntensityPosDefinite = False,
      overrideBeamPolInfo      = None,
      scaleFactor              = scaleFactor,
    )

  if True:
  # if False:
    print("\n=== Step 6: overlay weighted MC from shifted moments and real data ============")
    # weight accepted phase-space data in input format for generating kinematic plots in mass bins
    massBinningForWeighting = deepcopy(cfg.massBinning)  # same binning as for moment values
    massBinningForWeighting.nmbBins *= 10  # finer binning than for moment values
    weightDataWithMoments(
      cfg                       = cfg,
      momentsFileName           = "moments_phys_shifted.pkl",
      dataType                  = AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE,
      useIntensityTerms         = MomentResult.IntensityTermsType.PARITY_CONSERVING,
      weightInputData           = True,
      massBinningForWeighting   = massBinningForWeighting,
      reweightMassDistribution  = True,
      weightedDataDirPathSuffix = "_shifted",
    )
    # overlay weighted MC and real data
    ROOT.EnableImplicitMT()
    plotWeightedMc(
      cfg                       = cfg,
      useIntensityTerms         = MomentResult.IntensityTermsType.PARITY_CONSERVING,
      massBinning               = cfg.massBinning,
      weightedDataDirPathSuffix = "_shifted",
      nmbBinsAzim               = 72,
      nmbBinsOther              = 100,
      additionalColumnDefs      = {
        "realData"   : {},  # no additional columns to define for real data
        "weightedMc" : {},  # no additional columns to define for weighted MC
      },
      additionalFilterDefs      = [],
    )

  timer.stop("Total time for analysis")
  print(timer.summary)
