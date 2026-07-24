#!/usr/bin/env python3
"""
This module converts input data into the format expected by `MomentCalculator`.

Usage: Run this module as a script to convert input data files.
"""


from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import functools
import os
import subprocess
import tempfile

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`

from moments.MomentCalculator import (
  KinematicBinningVariable,
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
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
from workflow.PlottingUtilities import (
  HistAxisBinning,
  setupPlotStyle,
)
from workflow.RootUtilities import (
  declareInCpp,
  loadBasisFunctionsLibrary,
)
from workflow import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def reweightData(
  dataToWeight: ROOT.RDataFrame,  # data to reweight
  treeName:     str,              # name of TTree holding the data
  variableName: str,              # column name corresponding to kinematic variable whose distribution is to be reweighted
  targetDistr:  ROOT.TH1D,        # histogram with target distribution
) -> ROOT.RDataFrame:
  """Generic function that reweights data in given RDataFrame such that the distribution of the given variable matches the target distribution in the given histogram"""
  # get histogram of current distribution using same binning as targetDistribution
  currentDistr = dataToWeight.Histo1D(
    ROOT.RDF.TH1DModel(
      f"{variableName}Distr", f"Current Distribution;{variableName}",
      targetDistr.GetNbinsX(), targetDistr.GetXaxis().GetXmin(), targetDistr.GetXaxis().GetXmax()
    ),
    variableName,
  ).GetValue()
  # normalize target and current histograms such that they represent the corresponding PDFs
  targetDistr.Scale (1.0 / targetDistr.Integral() )
  currentDistr.Scale(1.0 / currentDistr.Integral())
  # calculate the weight as the ratio of target and current PDF
  weightsHist = targetDistr.Clone("weightsHist")
  weightsHist.SetTitle("Weights")
  weightsHist.Divide(currentDistr)
  # if True:
  if False:
    # save plots of distributions
    for hist in (currentDistr, targetDistr, weightsHist):
      canv = ROOT.TCanvas()
      hist.Draw()
      #TODO write files into correct output directory
      canv.SaveAs(f"{hist.GetName()}.root")
  # add columns for rejection sampling to input data
  declareInCpp(weightsHist = weightsHist)  # use Python TH1D object in C++  #TODO this can only be called once; otherwise this call crashes in ROOT
  dataToWeight = (
    dataToWeight.Define("reweightingWeight", f"(Double32_t)PyVars::weightsHist.GetBinContent(PyVars::weightsHist.FindBin({variableName}))")
                .Define("reweightingRndNmb",  "(Double32_t)gRandom->Rndm()")  # random number uniformly distributed in [0, 1]
  )
  tmpFilePath = tempfile.mktemp(dir = "./", prefix = "unweighted.", suffix = ".root")
  dataToWeight.Snapshot(treeName, tmpFilePath)  # write unweighted data to temporary file to ensure that random column is filled only once
  dataToWeight = ROOT.RDataFrame(treeName, tmpFilePath)  # read data back from temporary file
  nmbEvents = dataToWeight.Count().GetValue()  # number of events before reweighting
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
  # subprocess.run(f"rm --force --verbose {tmpFilePath}", shell = True)  #TODO this does not work as the RDataFrame based on this file is passed to the calling code
  return reweightedData


def reweightKinDistribution(
  dataToWeight:    ROOT.RDataFrame,  # data to reweight
  binning:         HistAxisBinning,  # binning of kinematic variable whose distribution is to be reweighted
  treeName:        str,              # name of TTree holding the data
  targetDistrFrom: str | MomentResultsKinematicBinning,  # construct target distribution from given data file name or from H_0(0, 0) in given moment results
  outFilePath:     str,  # name of file to write data into
  outputColumns:   Sequence[str] = (),  # columns to write into output file; if empty, all columns are written
) -> None:
  """Reweights distribution of given kinematic variable of given data according to the kinematic distribution of data in given file name or according to kinematic dependence of H_0(0, 0) in given moment results"""
  print(f"Reweighting {binning.var.name} dependence")
  targetDistr = None
  if isinstance(targetDistrFrom, str):
    # construct target distribution from real data
    print(f"Constructing target distribution from column '{binning.var.name}' in tree '{treeName}' in file '{targetDistrFrom}'")
    dataTarget = ROOT.RDataFrame(treeName, targetDistrFrom)
    targetDistr = dataTarget.Histo1D(
      ROOT.RDF.TH1DModel(f"{binning.var.name}DistrTarget", f"Target Distribution;{binning.axisTitle}", *binning.astuple),
      binning.var.name,
      "eventWeight",
    ).GetValue()
    # set under- and overflow bins to zero
    targetDistr.SetBinContent(0, 0.0)  # underflow bin
    targetDistr.SetBinContent(targetDistr.GetNbinsX() + 1, 0.0)  # overflow bin
  elif isinstance(targetDistrFrom, MomentResultsKinematicBinning):
    # construct target distribution from H_0(0, 0) values in kinematic bins
    targetDistr = ROOT.TH1D(f"{binning.var.name}DistrTarget", f"#it{{H}}_{{0}}(0, 0);{binning.axisTitle}", *binning.astuple)
    H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
    for momentResultsForBin in targetDistrFrom:
      binCenter = momentResultsForBin.binCenters[binning.var]
      targetDistr.SetBinContent(targetDistr.FindBin(binCenter), momentResultsForBin[H000Index].real[0])
  else:
    raise TypeError(f"Invalid {type(targetDistrFrom)=}. Must be str or MomentResultsKinematicBinning.")
  # reweight data
  originalColumns = list(dataToWeight.GetColumnNames())
  reweightedData = reweightData(
    dataToWeight = dataToWeight,
    treeName     = treeName,
    variableName = binning.var.name,
    targetDistr  = targetDistr,
  )
  print(f"Writing reweighted data to file '{outFilePath}'")
  reweightedData.Snapshot(treeName, outFilePath, originalColumns if not outputColumns else outputColumns)
  if True:
  # if False:
    # overlay target distribution and distribution after reweighting
    reweightedDistr = reweightedData.Histo1D(
      ROOT.RDF.TH1DModel(f"{binning.var.name}DistrReweighted", "Weighted MC", *binning.astuple),
      binning.var.name,
    ).GetValue()
    targetDistr.Scale(reweightedDistr.Integral() / targetDistr.Integral())
    histStack = ROOT.THStack(f"{binning.var.name}DataAndMc", f";{binning.axisTitle};Count")
    histStack.Add(targetDistr)
    histStack.Add(reweightedDistr)
    targetDistr.SetLineColor  (ROOT.kRed + 1)
    targetDistr.SetMarkerColor(ROOT.kRed + 1)
    reweightedDistr.SetLineColor  (ROOT.kBlue + 1)
    reweightedDistr.SetMarkerColor(ROOT.kBlue + 1)
    canv = ROOT.TCanvas()
    histStack.Draw("NOSTACK")
    canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
    canv.SaveAs(f"{outFilePath}.{binning.var.name}.pdf")


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
