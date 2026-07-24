#!/usr/bin/env python3
"""
This module converts input data into the format expected by `MomentCalculator`.

Usage: Run this module as a script to convert input data files.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import os

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from moments.MomentCalculator import KinematicBinningVariable
from workflow.AnalysisConfig import (
  AnalysisConfig,
  BEAM_POL_INFOS,
  CFG_POLARIZED_ETAPI0,
  CFG_POLARIZED_KSKL,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_ETAPETA,
  CFG_UNPOLARIZED_PIPI_CLAS,
)
from workflow.DataConversionUtilities import (
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TRACKDISTFDC,
  CPP_CODE_TWO_BODY_ANGLES,
  defineDataFrameColumns,
  lorentzVectors,
)
from workflow.DataWeightingUtilities import reweightKinDistribution
from workflow.PlottingUtilities import (
  HistAxisBinning,
  setupPlotStyle,
)
from workflow.RootUtilities import loadBasisFunctionsLibrary
from workflow import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  loadBasisFunctionsLibrary()  # initializes OpenMP and loads `cpp/basisFunctions.C`
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  timer.start("Total execution time")
  ROOT.gROOT.SetBatch(True)
  # ROOT.EnableImplicitMT()
  setupPlotStyle()

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_TRACKDISTFDC)

  outputColumnsUnpolarized = ("theta", "phi", "mass", "minusT")
  outputColumnsPolarized   = ("beamPol", "beamPolPhiLabDeg", "Phi")
  #TODO move these two also into AnalysisConfig
  additionalColumnDefs     = {  # additional columns for each data type
    AnalysisConfig.DataType.REAL_DATA             : {},
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : {},
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : {},
  }
  additionalFilterDefs     = {  # additional filters for each data type
    AnalysisConfig.DataType.REAL_DATA             : [],
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : [],
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : [],
  }
  # reweightMinusTDistribution = True
  reweightAccPSMCMinusTDistribution = False

  cfg = deepcopy(CFG_POLARIZED_PIPI)  # polarized gamma p -> (pi+ pi-) p data
  if False:  # cut away forward tracks in reconstructed data
    for inputDataType in (AnalysisConfig.DataType.REAL_DATA, AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE):
      lvs = lorentzVectors(dataFormat = AnalysisConfig.DataFormat.ALEX)
      additionalColumnDefs[inputDataType] = {
        "DistFdcPip": f"(Double32_t)trackDistFdc(pip_x4_kin.Z(), {lvs['pip']})",
        "DistFdcPim": f"(Double32_t)trackDistFdc(pim_x4_kin.Z(), {lvs['pim']})",
      }
      additionalFilterDefs[inputDataType] = ["(DistFdcPip > 4) and (DistFdcPim > 4)"]  # require minimum distance of tracks at FDC position [cm]
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # unpolarized gamma p -> (pi+ pi-) p data in CLAS kinematic range
  # cfg = deepcopy(CFG_POLARIZED_ETAPI0)  # polarized gamma p -> (eta pi0) p data, with eta -> gamma gamma from Nizar's analysis
  # for inputDataType in additionalColumnDefs:
  #   additionalColumnDefs[inputDataType] = {
  #     "beamPol"          : "Pol",         # use this column for beam polarization degree
  #     "beamPolPhiLabDeg" : "BeamAngle",   # use this column for beam polarization angle in lab frame
  #   }
  # additionalColumnDefs[AnalysisConfig.DataType.REAL_DATA]["eventWeight"] = "weightASBS"  # use this column as event weights
  # cfg = deepcopy(CFG_UNPOLARIZED_ETAPETA)  # unpolarized gamma p -> (eta' eta) p data from Will's analysis
  # cfg = deepcopy(CFG_POLARIZED_KSKL)  # polarized gamma p -> (K_S K_L) p data from Gabriel's analysis
  # additionalColumnDefs[AnalysisConfig.DataType.REAL_DATA]["eventWeight"] = "Weight"  # use this column as event weight

  print(f"Using analysis configuration:\n{cfg}")
  print(f"Setting up subsystem '{cfg.subsystem}':")
  for dataPeriod in cfg.dataPeriods:
    print(f"Setting up data period '{dataPeriod}':")
    for tBinLabel in cfg.tBinLabels:
      print(f"Setting up t bin '{tBinLabel}':")
      os.makedirs(cfg.convertedDataDirBasePath(dataPeriod, tBinLabel), exist_ok = True)
      for beamPolLabel in cfg.beamPolLabels:
        beamPolInfo = BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel]
        print(f"Setting up beam-polarization orientation '{beamPolLabel}'")
        for inputDataType, inputDataFormat in cfg.inputDataFormats.items():
          print(f"Setting up input data type '{inputDataType}' with format '{inputDataFormat}':")
          inputFilePath = cfg.inputFilePath(inputDataType, dataPeriod, tBinLabel, beamPolLabel)
          df = ROOT.RDataFrame(cfg.inputTreeName, inputFilePath)  # real data must contains combined signal and background data with correct event weights
          print(f"Converting {inputDataType} data with {inputDataFormat} format for '{cfg.subsystem.pairLabel}' subsystem, "
                f"'{dataPeriod}' period, '{tBinLabel}' t bin, and {beamPolLabel or 'no'} beam polarization from file(s) {inputFilePath}")
          lvs = lorentzVectors(inputDataFormat)
          df = defineDataFrameColumns(
            df                   = df,
            lvTarget             = lvs["target"],
            lvBeam               = lvs["beam"],  #TODO "beam" for GJ pi+- p baryon system is p_target
            lvRecoil             = lvs[cfg.subsystem.lvRecoilLabel],
            lvA                  = lvs[cfg.subsystem.lvALabel],
            lvB                  = lvs[cfg.subsystem.lvBLabel],
            beamPolInfo          = beamPolInfo,
            frame                = cfg.frame,
            additionalColumnDefs = additionalColumnDefs[inputDataType],
            additionalFilterDefs = additionalFilterDefs[inputDataType],
          ).Filter(('if (rdfentry_ == 0) { std::cout << "Running event loop" << std::endl; } return true;'))  # no-op filter that logs when event loop is running
          outputFilePath = cfg.convertedFilePath(inputDataType, dataPeriod, tBinLabel, beamPolLabel)
          outputTreeName = cfg.subsystem.pairLabel
          outputColumns  = outputColumnsUnpolarized
          if beamPolInfo is not None:
            outputColumns += outputColumnsPolarized
          if df.HasColumn("eventWeight"):
            outputColumns += ("eventWeight", )
          if reweightAccPSMCMinusTDistribution and inputDataType == AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE:
            #TODO this is currently only implemented for the bin 0.1 < |t| < 0.2 GeV^2
            # reweight -t distribution to match that of real data
            reweightedFilePath = outputFilePath.replace(".root", ".reweighted_minusT.root")
            reweightKinDistribution(
              dataToWeight    = df,
              treeName        = outputTreeName,
              binning         = HistAxisBinning(
                nmbBins = 50, minVal = 0.1, maxVal = 0.2,
                _var = KinematicBinningVariable(name= "minusT", label = "#minus#it{t}", unit = "GeV^{2}/#it{c}^{2}", nmbDigits = 3),
              ),
              targetDistrFrom = f"{os.path.dirname(outputFilePath)}/data_flat_{beamPolLabel}.root",
              outFilePath     = reweightedFilePath,
              outputColumns   = outputColumns,
            )
          else:
            print(f"Writing columns {outputColumns} to tree '{outputTreeName}' in file '{outputFilePath}'")
            df.Snapshot(outputTreeName, outputFilePath, outputColumns)

  timer.stop("Total execution time")
  print(timer.summary)
