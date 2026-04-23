#!/usr/bin/env python3
"""
This module weights data (usually generated or accepted phase-space
data) with the intensity distribution calculated from the results of
the moment analysis of unpolarized or polarized pi+ pi-
photoproduction data.  The moment values are read from files produced
by the script `photoProdCalcMoments.py` that calculates the moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import os
import subprocess

import ROOT
from wurlitzer import pipes, STDOUT

from AnalysisConfig import (
  AnalysisConfig,
  CFG_KEVIN,
  CFG_POLARIZED_ETAPI0,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_JPAC,
  CFG_UNPOLARIZED_PIPI_PWA,
)
from makeMomentsInputTree import (
  BeamPolInfo,
  BEAM_POL_INFOS,
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_TWO_BODY_ANGLES,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CoordSysType,
  defineDataFrameColumns,
  InputDataFormat,
  lorentzVectors,
  reweightKinDistribution,
  SubSystemInfo,
)
from MomentCalculator import (
  MomentResult,
  MomentResultsKinematicBinning,
)
from PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def loadInputData(
  inputDataDef:     AnalysisConfig.DataType | tuple[str, str, SubSystemInfo, CoordSysType] | int,
    # if `AnalysisConfig.DataType` instance, the file corresponding to `DataType` is loaded
    # if `tuple`, a tuple (<tree name>, <file name>, <subsystem info>, <coordinate system type>) for raw data is expected
    # if `int`, phase-space distribution in angles is generated with given number of events
  cfg:              AnalysisConfig,
  massBinning:      HistAxisBinning,  # mass binning used for weighting
  massBinIndex:     int,              # index of mass bin to load/generate data for
  beamPolInfo:      BeamPolInfo | None = None,  # beam polarization information needed for raw data files
  limitNmbEventsTo: int | None         = None,  # if `int`, limits number of events to read from tree
) -> tuple[ROOT.RDataFrame, int, list[str]]:
  """Loads data specified by `inputDataDef` and returns them as RDataFrame and the number of input events."""
  dataToWeight = None
  if isinstance(inputDataDef, AnalysisConfig.DataType) or (isinstance(inputDataDef, tuple) and len(inputDataDef) == 4):
    if isinstance(inputDataDef, AnalysisConfig.DataType):
      print(f"Loading data of type '{inputDataDef}'")
      dataToWeight = cfg.loadData(inputDataDef)
    elif isinstance(inputDataDef, tuple):
      print(f"Loading raw data in tree '{inputDataDef[0]}' from file '{inputDataDef[1]}'")
      dataToWeight = ROOT.RDataFrame(inputDataDef[0], inputDataDef[1])
    assert dataToWeight is not None, f"Could not load data of type '{inputDataDef}'"
    if limitNmbEventsTo is not None:
      print(f"Limiting total number of input events (before binning) to {limitNmbEventsTo}")
      dataToWeight = dataToWeight.Range(0, limitNmbEventsTo)  # works only in single-thread mode
    originalColumns = list(dataToWeight.GetColumnNames())
    if isinstance(inputDataDef, tuple):
      # define columns needed to calculate intensity
      subsystem = inputDataDef[2]
      frame     = inputDataDef[3]
      assert beamPolInfo is not None, "Beam polarization information must be provided when loading raw data from file"
      lvs = lorentzVectors(dataFormat = InputDataFormat.AMPTOOLS)
      dataToWeight = defineDataFrameColumns(
        df          = dataToWeight,
        lvTarget    = lvs["target"],
        lvBeam      = lvs["beam"],
        lvRecoil    = lvs[subsystem.lvRecoilLabel],
        lvA         = lvs[subsystem.lvALabel],
        lvB         = lvs[subsystem.lvBLabel],
        beamPolInfo = beamPolInfo,
        frame       = frame,
      )
    kinematicBinFilter: str = massBinning.binFilter(massBinIndex)
    dataToWeight = dataToWeight.Filter(kinematicBinFilter)
    nmbInputEvents = dataToWeight.Count().GetValue()
    print(f"Input data contain {nmbInputEvents} events in bin '{kinematicBinFilter}'")
    return dataToWeight, nmbInputEvents, originalColumns
  elif isinstance(inputDataDef, int):
    nmbGenPsEvents = inputDataDef
    print(f"Generating phase-space distribution with {nmbGenPsEvents} events")
    kinematicBinRange: tuple[float, float] = massBinning.binValueRange(massBinIndex)
    dataToWeight = (
      ROOT.RDataFrame(nmbGenPsEvents)
          .Define("cosTheta", "(Double32_t)gRandom->Uniform(-1, +1)")
          .Define("theta",    "(Double32_t)std::acos(cosTheta)")
          .Define("phiDeg",   "(Double32_t)gRandom->Uniform(-180, +180)")
          .Define("phi",      "(Double32_t)(phiDeg * TMath::DegToRad())")
          .Define("mass",    f"(Double32_t)gRandom->Uniform({kinematicBinRange[0]}, {kinematicBinRange[1]})")
          # add no-op filter that just prints a log message when event loop is running
          .Filter('if (rdfentry_ == 0) { cout << "Running event loop in `loadInputData()`" << endl; } return true;')
    )
    if cfg.polarization is not None:
      # polarized case: add Phi and polarization columns
      dataToWeight = (
        dataToWeight.Define("PhiDeg", "(Double32_t)gRandom->Uniform(-180, +180)")
                    .Define("Phi",    "(Double32_t)(PhiDeg * TMath::DegToRad())")
      )
      if isinstance(cfg.polarization, float):
        dataToWeight = dataToWeight.Define("beamPol", f"(Double32_t){cfg.polarization}")
      elif isinstance(cfg.polarization, str):
        assert beamPolInfo is not None, f"Beam polarization information must be provided when generating phase-space data with {cfg.polarization=}"
        dataToWeight = dataToWeight.Define(cfg.polarization, f"(Double32_t){beamPolInfo.pol}")
    #TODO is a snapshot necessary here to fill random columns only once?
    return dataToWeight, nmbGenPsEvents, list(dataToWeight.GetColumnNames())
  else:
    raise ValueError(f"Invalid {inputDataDef=}")


def weightDataWithIntensityFormula(
  inputDataDef:         AnalysisConfig.DataType | tuple[str, str, SubSystemInfo, CoordSysType] | int,
    # if `AnalysisConfig.DataType` instance, the file corresponding to `DataType` is loaded
    # if `tuple`, a tuple (<tree name>, <file name>, <subsystem info>, <coordinate system type>) for raw data is expected
    # if `int`, phase-space distribution in angles is generated with given number of events
  massBinning:          HistAxisBinning,  # mass binning used for weighting
  massBinIndex:         int,  # index of mass bin to generate data for
  intensityFormula:     str,  # intensity formula as function of theta [rad] phi [rad], and Phi [rad] that defines distribution of events
  weightedDataFileName: str,  # ROOT file to which weighted events are written
  cfg:                  AnalysisConfig,
  seed:                 int                = 123456789,  # seed for rejection sampling and for generating phase-space events
  beamPolInfo:          BeamPolInfo | None = None,       # beam polarization information needed for raw data files
  limitNmbEventsTo:     int | None         = None,       # if `int`, limits number of events to read from tree
) -> ROOT.RDataFrame:
  """Weights input data specified by `inputDataDef` and `massBinIndex` with given intensity formula and writes data to `outFileName`"""
  ROOT.gRandom.SetSeed(seed)
  # load input data
  dataToWeight, nmbInputEvents, originalColumns = loadInputData(
    inputDataDef     = inputDataDef,
    cfg              = cfg,
    massBinning      = massBinning,
    massBinIndex     = massBinIndex,
    beamPolInfo      = beamPolInfo,
    limitNmbEventsTo = limitNmbEventsTo,
  )
  print(f"Calculating event weights using formula\n{intensityFormula}")
  # add columns for intensity weight and random number in [0, 1]
  dataToWeight = (
    dataToWeight.Define("intensityWeight",      f"(Double32_t){intensityFormula}")
                .Define("intensityWeightRndNmb", "(Double32_t)gRandom->Rndm()")  # random number in [0, 1] for each event
  )
  # write unweighted data to file and read data back to ensure that random columns are filled only once
  tmpFileName = f"{weightedDataFileName}.tmp"
  dataToWeight.Snapshot(cfg.treeName, tmpFileName)
  dataToWeight = ROOT.RDataFrame(cfg.treeName, tmpFileName)
  # determine range of weight values
  minIntensityWeight = dataToWeight.Min("intensityWeight").GetValue()
  maxIntensityWeight = dataToWeight.Max("intensityWeight").GetValue()
  print(f"Minimum intensity is {minIntensityWeight}")
  print(f"Maximum intensity is {maxIntensityWeight}")
  if minIntensityWeight < 0:
    print("WARNING: Intensity function is negative in some regions of phase space; "
          "this may lead to incorrect results during weighting!")
  # apply weights by accepting each event with probability intensityWeight / maxIntensityWeight
  weightedData = (
    dataToWeight.Define("acceptEventIntensityWeight", f"(bool)(intensityWeightRndNmb < (intensityWeight / {maxIntensityWeight}))")
                .Filter("acceptEventIntensityWeight == true")
  )
  nmbWeightedEvents = weightedData.Count().GetValue()
  print(f"After weighting with the intensity function, the sample contains {nmbWeightedEvents} accepted events; "
        f"weighting efficiency is {nmbWeightedEvents / nmbInputEvents}")
  # write weighted data to file
  print(f"Writing data weighted with intensity function to file '{weightedDataFileName}'")
  # weightedData.Snapshot(cfg.treeName, weightedDataFileName, originalColumns + ["intensityWeight"])
  weightedData.Snapshot(cfg.treeName, weightedDataFileName)  # write original columns + columns defined here
  subprocess.run(f"rm --force --verbose {tmpFileName}", shell = True)  # remove temporary file
  return ROOT.RDataFrame(cfg.treeName, weightedDataFileName)


#TODO ist this still needed? if yes, use function in `photoProdPlotIntensityFcn.py` instead
def plotIntensityFcn(
  momentResults:     MomentResult,
  massBinIndex:      int,
  beamPolInfo:       BeamPolInfo,
  outputDirName:     str,
  nmbBinsPerAxis:    int                             = 25,
  useIntensityTerms: MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.ALL,
) -> None:
  """Draws intensity function in given mass bin and writes PDF to output directory"""
  print(f"Plotting intensity function for mass bin {massBinIndex}")
  if True:
    # draw intensity function as 3D plot
    # formula uses variables: x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
    intensityFormula = momentResults.intensityFormula(
      polarization      = beamPolInfo.pol,
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
      outFileName = f"{outputDirName}/{intensityFcn.GetName()}.pdf",
      histTitle   = "Intensity Function;cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]",
    )
  if True:
    # draw intensity as function of phi_HF and Phi for fixed cos(theta)_HF value
    cosTheta = 0.0  # fixed value of cos(theta)_HF
    # formula uses variables: x = phi in [-180, +180] deg; y = Phi in [-180, +180] deg
    intensityFormulaFixedCosTheta = momentResults.intensityFormula(
      polarization      = beamPolInfo.pol,
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
    canv.SaveAs(f"{outputDirName}/{intensityFcnFixedCosTheta.GetName()}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("./rootlogon.C") == 0, "Error loading './rootlogon.C'"

  # declare C++ functions
  ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
  ROOT.gInterpreter.Declare(CPP_CODE_TWO_BODY_ANGLES)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)

  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarizedK- K_S Delta++ data
  cfg = deepcopy(CFG_POLARIZED_ETAPI0)  # perform analysis of Nizar's polarized eta pi0 data
  # cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)   # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data

  useIntensityTerms      = MomentResult.IntensityTermsType.ALL                # include parity-conserving and parity-violating terms into formula
  # useIntensityTerms    = MomentResult.IntensityTermsType.PARITY_CONSERVING  # include only parity-conserving terms
  # useIntensityTerms    = MomentResult.IntensityTermsType.PARITY_VIOLATING   # include only parity-violating terms
  # weight accepted phase-space data in raw data format for generating kinematic plots in mass bins
  weightedDataFileBaseName = f"phaseSpace_acc_weighted_raw_{useIntensityTerms.value}"
  reweightMassDistribution = True
  limitNmbEventsTo         = None  # limit number of events to read from input tree
  # # weight accepted phase-space data in MomentCalculator format for input-output studies with acceptance correction
  # inputDataDef             = AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE
  # weightedDataFileBaseName = f"phaseSpace_acc_weighted_flat_{useIntensityTerms.value}"
  # reweightMassDistribution = True  # reweight mass distribution after weighting with intensity function
  # limitNmbEventsTo         = None  # limit number of events to read from input tree
  # # weight generated phase-space data in MomentCalculator format for input-output studies without acceptance correction
  # # inputDataDef             = 100000  # generate phase-space distribution in angles with given number of events
  # inputDataDef             = AnalysisConfig.DataType.GENERATED_PHASE_SPACE
  # weightedDataFileBaseName = f"phaseSpace_gen_weighted_flat_{useIntensityTerms.value}"
  # reweightMassDistribution = True  # reweight mass distribution after weighting with intensity function
  # limitNmbEventsTo         = 70000000  # limit number of events to read from input tree

  # frame = CoordSysType.HF  # helicity frame, i.e. z_HF = -p_recoil
  # subsystem = SubSystemInfo(pairLabel = "PiPi", lvALabel = "pip", lvBLabel = "pim",    lvRecoilLabel = "recoil"),
  # inputDataDef: tuple[str, str, SubSystemInfo, CoordSysType] = ("kin", "Alex/amptools_tree_accepted*.root", subsystem, frame)
  frame = CoordSysType.GJ  # Gottfried-Jackson frame, i.e. z_GJ = p_beam
  subsystem = SubSystemInfo(pairLabel = "EtaPi0", lvALabel = "eta", lvBLabel = "pi0", lvRecoilLabel = "recoil")
  inputDataDef: tuple[str, str, SubSystemInfo, CoordSysType] = ("kin", "Nizar/amptools_tree_accepted_All.root", subsystem, frame)
  BEAM_POL_INFOS["merged"]["All"] = BeamPolInfo(  # read beam polarization info from input tree
    pol    = "Pol",
    PhiLab = "BeamAngle",
  )
  dataDirBaseName = f"./dataPhotoProd{subsystem.pairLabel}/polarized"
  dataPeriods = (
    "merged",
    # "2017_01",
    # "2018_08",
  )
  tBinLabels = (
    # "t010020",
    # "t020032",
    # "t032050",
    "t050075",
    # "t075100",
    # "tbin_0.1_0.2",
    # "tbin_0.2_0.3",
    # "tbin_0.3_0.4",
    # "tbin_0.4_0.5",
  )
  beamPolLabels = (
    "All",
    # "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
    # "AMO",
    # "Unpol",
  )
  cfg.polarization = None  # treat data as unpolarized
  maxLs = (
    4,
    # (4, 6),
    # (4, 8),
    # (4, 12),
    # (4, 16),
    # (4, 20),
    # 6,
    # 8,
    # (8, 16),
    # (8, 20),
    # 12,
    # 16,
    # 20,
  )
  # makeIntensityFcnPlots = True  # draw intensity function in each mass bin
  makeIntensityFcnPlots = False
  massBinningForWeighting = deepcopy(cfg.massBinning)  # same binning as for moment values
  massBinningForWeighting._var = deepcopy(cfg.binVarMass)
  # massBinningForWeighting = HistAxisBinning(nmbBins = 250, minVal = 0.28, maxVal = 2.28)  # finer binning than for moment values
  # massBinningForWeighting = HistAxisBinning(nmbBins = 1, minVal = 0.72, maxVal = 0.76)  # rho(770) mass bin
  # massBinningForWeighting.var = cfg.binVarMass

  outFileDirBaseNameCommon = cfg.outFileDirBaseName
  for dataPeriod in dataPeriods:
    for tBinLabel in tBinLabels:
      for beamPolLabel in beamPolLabels:
        cfg.dataFileName       = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}/data_flat_{beamPolLabel}.root"
        cfg.psAccFileName      = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}/phaseSpace_acc_flat_{beamPolLabel}.root"
        cfg.psGenFileName      = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}/phaseSpace_gen_flat_{beamPolLabel}.root"
        cfg.outFileDirBaseName = f"{outFileDirBaseNameCommon}/{dataPeriod}/{tBinLabel}/{beamPolLabel}"
        # cfg.outFileDirBaseName = f"{outFileDirBaseNameCommon}.reweighted_minusT/{dataPeriod}/{tBinLabel}/{beamPolLabel}"
        for maxL in maxLs:
          print(f"Generating weighted MC for data period '{dataPeriod}', t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
          cfg.maxL = maxL
          cfg.init()
          thisSourceFileName = os.path.basename(__file__)
          # create directory, into which weighted data will be written
          weightedDataDirName = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}/weightedMc.maxL_{maxL}/{beamPolLabel}"
          # weightedDataDirName = f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{subsystem.pairLabel}/weightedMc.maxL_{maxL}/{beamPolLabel}.reweighted_minusT"
          Utilities.makeDirPath(weightedDataDirName)
          logFileName = f"{weightedDataDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
          print(f"Writing output to log file '{logFileName}'")
          with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
            Utilities.printGitInfo()
            timer = Utilities.Timer()
            ROOT.gROOT.SetBatch(True)
            setupPlotStyle()
            print(f"Using configuration:\n{cfg}")
            timer.start("Total execution time")
            momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_phys.pkl"
            # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_pwa_SPD.pkl"
            # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_JPAC.pkl"
            print(f"Reading moments from file '{momentResultsFileName}'")
            momentResults = MomentResultsKinematicBinning.loadPickle(momentResultsFileName)
            for massBinIndexForWeighting, massBinCenterForWeighting in enumerate(massBinningForWeighting):
              print(f"Weighting events at mass {massBinCenterForWeighting:.{cfg.massBinning.var.nmbDigits}f} {cfg.massBinning.var.unit}")
              # find moment result corresponding to mass-bin center
              massBinIndexForMoments = cfg.massBinning.findBin(massBinCenterForWeighting)
              assert massBinIndexForMoments is not None, f"Could not find bin for mass value of {massBinCenterForWeighting} {cfg.massBinning.var.unit}"
              momentResultsForBin = momentResults[massBinIndexForMoments]
              print(f"Weighting events with intensity function using moment values in mass bin {massBinIndexForMoments} at {momentResultsForBin.binCenters[cfg.massBinning.var]:.{cfg.massBinning.var.nmbDigits}f} {cfg.massBinning.var.unit}")
              weightDataWithIntensityFormula(
                inputDataDef         = (
                  inputDataDef[0],
                  f"{dataDirBaseName}/{dataPeriod}/{tBinLabel}/{inputDataDef[1]}",
                  inputDataDef[2],
                  inputDataDef[3],
                ) if isinstance(inputDataDef, tuple) else inputDataDef,
                massBinning          = massBinningForWeighting,
                massBinIndex         = massBinIndexForWeighting,
                intensityFormula     = momentResultsForBin.intensityFormula(
                  polarization      = "beamPol",  # read polarization from tree column
                  thetaFormula      = "theta",
                  phiFormula        = "phi",
                  PhiFormula        = "Phi",
                  useIntensityTerms = useIntensityTerms,
                ),
                weightedDataFileName = f"{weightedDataDirName}/{weightedDataFileBaseName}_bin_{massBinIndexForWeighting}.root",
                cfg                  = cfg,
                seed                 = 12345 + massBinIndexForWeighting,  # ensure rejection sampling and generated phase-space data in different mass bins are independent
                beamPolInfo          = BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel] if isinstance(inputDataDef, tuple) else None,  #TODO is it correct to use no polarization info when weighting phase-space data?
                limitNmbEventsTo     = limitNmbEventsTo,
              )
              if makeIntensityFcnPlots:
                plotIntensityFcn(
                  momentResults     = momentResultsForBin,
                  massBinIndex      = massBinIndexForWeighting,
                  beamPolInfo       = BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel],  #TODO consistency of polarization value is only ensured for raw data
                  outputDirName     = weightedDataDirName,
                  useIntensityTerms = useIntensityTerms,
                )

            # merge trees with weighted MC data for individual mass bins into single file
            mergedFileName  = f"{weightedDataDirName}/{weightedDataFileBaseName}.root"
            nmbParallelJobs = 10
            with timer.timeThis(f"Time to merge ROOT files from all mass bins using hadd with {nmbParallelJobs} parallel jobs"):
              cmd = f"hadd -f -j {nmbParallelJobs} {mergedFileName} {weightedDataDirName}/{weightedDataFileBaseName}_bin_*.root"
              print(f"Merging ROOT files from all mass bins: '{cmd}'")
              subprocess.run(cmd, shell = True)

            if reweightMassDistribution:
              # reweight mass distribution of merged file
              reweightedFileName = f"{weightedDataDirName}/{weightedDataFileBaseName}_reweighted.root"
              with timer.timeThis(f"Time to reweight mass distribution"):
                reweightKinDistribution(
                  dataToWeight    = ROOT.RDataFrame(cfg.treeName, mergedFileName),  # load merged data file created in step above
                  treeName        = cfg.treeName,
                  binning         = massBinningForWeighting,
                  targetDistrFrom = cfg.dataFileName,
                  # targetDistrFrom = momentResults,
                  outFileName     = reweightedFileName,
                )

            timer.stop("Total execution time")
            print(timer.summary)
