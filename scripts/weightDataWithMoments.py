#!/usr/bin/env python3
"""
This module weights data (usually generated or accepted phase-space
data) with the intensity distribution calculated from the results of
the moment analysis of unpolarized or polarized pi+ pi-
photoproduction data.  The moment values are read from files produced
by the script `calculateMoments.py` that calculates the moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import os
import subprocess

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`
from wurlitzer import pipes, STDOUT

from moments.MomentCalculator import (
  MomentResult,
  MomentResultsKinematicBinning,
)
from workflow.DataConversionUtilities import (
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CPP_CODE_TWO_BODY_ANGLES,
)
from workflow.DataWeightingUtilities import (
  reweightKinDistribution,
  weightDataWithIntensityFormula,
)
from workflow.AnalysisConfig import (
  AnalysisConfig,
  BeamPolInfo,
  BEAM_POL_INFOS,
  CFG_KEVIN,
  CFG_POLARIZED_ETAPI0,
  CFG_POLARIZED_PIPI,
  CFG_POLARIZED_KSKL,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_JPAC,
  CFG_UNPOLARIZED_PIPI_PWA,
)
from workflow.PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
  setupPlotStyle,
)
from workflow import RootUtilities
from workflow import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


#TODO is this still needed? if yes, use function in `plotIntensityFunctions.py` instead
def plotIntensityFcn(
  momentResults:     MomentResult,
  massBinIndex:      int,
  beamPolInfo:       BeamPolInfo | None,
  outputDirPath:     str,
  nmbBinsPerAxis:    int                             = 25,
  useIntensityTerms: MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.ALL,
) -> None:
  """Draws intensity function in given mass bin and writes PDF to output directory"""
  print(f"Plotting intensity function for mass bin {massBinIndex}")
  polarization = beamPolInfo.pol if beamPolInfo is not None else None
  if True:
    # draw intensity function as 3D plot
    # formula uses variables: x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
    intensityFormula = momentResults.intensityFormula(
      polarization      = polarization,
      thetaFormula      = "std::acos(x)",
      phiFormula        = "TMath::DegToRad() * y",
      PhiFormula        = "TMath::DegToRad() * z",
      useIntensityTerms = useIntensityTerms,
    )
    intensityFcn = ROOT.TF3(f"intensityFcn_{useIntensityTerms.value}_bin_{massBinIndex}", intensityFormula, -1, +1, -180, +180, -180, +180)
    intensityFcn.SetMinimum(0)
    drawTF3(
      fcn         = intensityFcn,
      binnings    = (
        HistAxisBinning(nmbBinsPerAxis,   -1,   +1),
        HistAxisBinning(nmbBinsPerAxis, -180, +180),
        HistAxisBinning(nmbBinsPerAxis, -180, +180),
      ),
      outFilePath = f"{outputDirPath}/{intensityFcn.GetName()}.pdf",
      histTitle   = "Intensity Function;cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]",
    )
  if True:
    # draw intensity as function of phi_HF and Phi for fixed cos(theta)_HF value
    cosTheta = 0.0  # fixed value of cos(theta)_HF
    # formula uses variables: x = phi in [-180, +180] deg; y = Phi in [-180, +180] deg
    intensityFormulaFixedCosTheta = momentResults.intensityFormula(
      polarization      = polarization,
      thetaFormula      = f"std::acos({cosTheta})",
      phiFormula        = "TMath::DegToRad() * x",
      PhiFormula        = "TMath::DegToRad() * y",
      useIntensityTerms = useIntensityTerms,
    )
    intensityFcnFixedCosTheta = ROOT.TF2(f"intensityFcn_fixedCosTheta_{useIntensityTerms.value}_bin_{massBinIndex}", intensityFormulaFixedCosTheta, -180, +180, -180, +180)
    intensityFcnFixedCosTheta.SetTitle(f"Intensity Function for cos#theta_{{HF}} = {cosTheta};#phi_{{HF}} [deg];#Phi [deg]")
    intensityFcnFixedCosTheta.SetNpx(100)
    intensityFcnFixedCosTheta.SetNpy(100)
    intensityFcnFixedCosTheta.SetMinimum(0)
    canv = ROOT.TCanvas()
    intensityFcnFixedCosTheta.Draw("COLZ")
    canv.SaveAs(f"{outputDirPath}/{intensityFcnFixedCosTheta.GetName()}.pdf")


if __name__ == "__main__":
  RootUtilities.loadBasisFunctionsLibrary()  # initializes OpenMP and loads `cpp/basisFunctions.C`
  ROOT.gROOT.SetBatch(True)

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)

  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarizedK- K_S Delta++ data
  # cfg = deepcopy(CFG_POLARIZED_ETAPI0)  # perform analysis of Nizar's polarized eta pi0 data
  cfg = deepcopy(CFG_POLARIZED_KSKL)  # perform analysis of Gabriel's polarized K_S K_L data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)   # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  # BEAM_POL_INFOS["merged"]["All"] = BeamPolInfo(  # read beam polarization info from input tree
  #   pol    = "Pol",
  #   PhiLab = "BeamAngle",
  # )

  useIntensityTerms = MomentResult.IntensityTermsType.ALL                # include parity-conserving and parity-violating terms into formula
  # useIntensityTerms = MomentResult.IntensityTermsType.PARITY_CONSERVING  # include only parity-conserving terms
  # useIntensityTerms = MomentResult.IntensityTermsType.PARITY_VIOLATING   # include only parity-violating terms

  # weight accepted phase-space data in input format for generating kinematic plots in mass bins
  dataType                 = AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE
  weightInputFormat        = True
  weightedDataFileBaseName = f"phaseSpace_acc_weighted_input_{useIntensityTerms.value}"
  # # weight accepted phase-space data in converted format for input-output studies with acceptance correction
  # dataType                 = AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE
  # weightInputFormat        = False
  # weightedDataFileBaseName = f"phaseSpace_acc_weighted_flat_{useIntensityTerms.value}"
  # # weight generated phase-space data in converted format for input-output studies without acceptance correction
  # dataType                 = AnalysisConfig.DataType.GENERATED_PHASE_SPACE
  # weightInputFormat        = False
  # weightedDataFileBaseName = f"phaseSpace_gen_weighted_flat_{useIntensityTerms.value}"
  # limitNmbEventsTo         = 70000000  # limit number of events to read from input tree

  reweightMassDistribution = True
  limitNmbEventsTo         = None  # limit number of events to read from input tree
  # makeIntensityFcnPlots    = True  # draw intensity function in each mass bin
  makeIntensityFcnPlots    = False
  massBinningForWeighting  = deepcopy(cfg.massBinning)  # same binning as for moment values
  massBinningForWeighting.nmbBins *= 10  # finer binning than for moment values
  # massBinningForWeighting  = HistAxisBinning(nmbBins = 1, minVal = 0.72, maxVal = 0.76)  # rho(770) mass bin

  print(f"Generating weighted MC for subsystem '{cfg.subsystem}':")
  for dataPeriod in cfg.dataPeriods:
    for tBinLabel in cfg.tBinLabels:
      for beamPolLabel in cfg.beamPolLabels:
        for maxL in cfg.maxLs:
          print(f"Generating weighted MC for data period '{dataPeriod}', t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
          thisSourceFileName = os.path.basename(__file__)
          # create directory, into which weighted data will be written
          weightedDataDirPath = f"{cfg.dataDirBasePath}/{dataPeriod}/{tBinLabel}/{cfg.subsystem.pairLabel}/weightedMc.maxL_{maxL}/{beamPolLabel}"
          Utilities.makeDirPath(weightedDataDirPath)
          logFilePath = f"{weightedDataDirPath}/{os.path.splitext(thisSourceFileName)[0]}_{useIntensityTerms.value}.log"
          print(f"Writing output to log file '{logFilePath}'")
          with open(logFilePath, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
            Utilities.printGitInfo()
            timer = Utilities.Timer()
            setupPlotStyle()
            print(f"Using analysis configuration:\n{cfg}")
            timer.start("Total execution time")
            momentResultsFilePath = f"{cfg.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel, maxL)}/{cfg.outFileNamePrefix}_moments_phys.pkl"
            print(f"Reading moments from file '{momentResultsFilePath}'")
            momentResults = MomentResultsKinematicBinning.loadPickle(momentResultsFilePath)
            for massBinIndexForWeighting, massBinCenterForWeighting in enumerate(massBinningForWeighting):
              print(f"Weighting events at mass {massBinCenterForWeighting:.{cfg.massBinning.var.nmbDigits}f} {cfg.massBinning.var.unit}")
              # find moment result corresponding to mass-bin center
              massBinIndexForMoments = cfg.massBinning.findBin(massBinCenterForWeighting)
              assert massBinIndexForMoments is not None, f"Could not find bin for mass value of {massBinCenterForWeighting} {cfg.massBinning.var.unit}"
              momentResultsForBin = momentResults[massBinIndexForMoments]
              print(f"Weighting events with intensity function using moment values in mass bin {massBinIndexForMoments} at {momentResultsForBin.binCenters[cfg.massBinning.var]:.{cfg.massBinning.var.nmbDigits}f} {cfg.massBinning.var.unit}")
              dataFilePath = (
                cfg.inputFilePath    (dataType, dataPeriod, tBinLabel, beamPolLabel) if weightInputFormat else
                cfg.convertedFilePath(dataType, dataPeriod, tBinLabel, beamPolLabel)
              )
              treeName = cfg.inputTreeName if weightInputFormat else cfg.convertedTreeName
              print(f"Loading input data of type '{dataType}' in tree '{treeName}' from file '{dataFilePath}'")
              beamsPolInfo = BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel]
              weightDataWithIntensityFormula(
                inputDataDef         = (dataFilePath, treeName, cfg.inputDataFormats[dataType]),
                massBinning          = massBinningForWeighting,
                massBinIndex         = massBinIndexForWeighting,
                intensityFormula     = momentResultsForBin.intensityFormula(
                  polarization      = "beamPol" if beamsPolInfo is not None else None,
                  thetaFormula      = "theta",
                  phiFormula        = "phi",
                  PhiFormula        = "Phi",
                  useIntensityTerms = useIntensityTerms,
                ),
                weightedDataFilePath = f"{weightedDataDirPath}/{weightedDataFileBaseName}_bin_{massBinIndexForWeighting}.root",
                cfg                  = cfg,
                seed                 = 12345 + massBinIndexForWeighting,  # ensure rejection sampling and generated phase-space data in different mass bins are independent
                beamPolInfo          = beamsPolInfo,
                limitNmbEventsTo     = limitNmbEventsTo,
              )
              if makeIntensityFcnPlots:
                plotIntensityFcn(
                  momentResults     = momentResultsForBin,
                  massBinIndex      = massBinIndexForWeighting,
                  beamPolInfo       = beamsPolInfo,
                  outputDirPath     = weightedDataDirPath,
                  useIntensityTerms = useIntensityTerms,
                )

            # merge trees with weighted MC data for individual mass bins into single file
            mergedFilePath  = f"{weightedDataDirPath}/{weightedDataFileBaseName}.root"
            nmbParallelJobs = 10
            with timer.timeThis(f"Time to merge ROOT files from all mass bins using hadd with {nmbParallelJobs} parallel jobs"):
              cmd = f"hadd -f -j {nmbParallelJobs} {mergedFilePath} {weightedDataDirPath}/{weightedDataFileBaseName}_bin_*.root"
              print(f"Merging ROOT files from all mass bins: '{cmd}'")
              subprocess.run(cmd, shell = True)

            if reweightMassDistribution:
              # reweight mass distribution of merged file
              treeName           = cfg.inputTreeName if weightInputFormat else cfg.convertedTreeName
              reweightedFilePath = f"{weightedDataDirPath}/{weightedDataFileBaseName}_reweighted.root"
              with timer.timeThis(f"Time to reweight mass distribution"):
                reweightKinDistribution(
                  dataToWeight    = ROOT.RDataFrame(treeName, mergedFilePath),  # load merged data file created in step above
                  binning         = massBinningForWeighting,
                  treeName        = cfg.convertedTreeName,
                  targetDistrFrom = cfg.convertedFilePath(AnalysisConfig.DataType.REAL_DATA, dataPeriod, tBinLabel, beamPolLabel),  # match measured mass distribution
                  # targetDistrFrom = momentResults,  # match acceptance-corrected mass distribution given by H_0(0, 0)
                  outFilePath     = reweightedFilePath,
                )

            timer.stop("Total execution time")
            print(timer.summary)
