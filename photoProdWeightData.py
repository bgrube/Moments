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
import tempfile

import ROOT
from wurlitzer import pipes, STDOUT

from AnalysisConfig import (
  AnalysisConfig,
  CFG_KEVIN,
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI_CLAS,
  CFG_UNPOLARIZED_PIPI_JPAC,
  CFG_UNPOLARIZED_PIPI_PWA,
  HistAxisBinning,
)
from dataPhotoProdPiPi.makeMomentsInputTree import (
  BeamPolInfo,
  BEAM_POL_INFOS,
  CPP_CODE_ANGLES_GLUEX_AMPTOOLS,
  CPP_CODE_BEAM_POL_PHI,
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
  CPP_CODE_MANDELSTAM_T,
  CPP_CODE_MASSPAIR,
  CoordSysType,
  defineDataFrameColumns,
  InputDataFormat,
  lorentzVectors,
)
from MomentCalculator import (
  MomentResult,
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
from PlottingUtilities import (
  drawTF3,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def loadInputData(
  inputDataDef:     AnalysisConfig.DataType | tuple[str, str] | int,  # if `AnalysisConfig.DataType` instance, the file corresponding to `DataType` is loaded
                                                                      # if `tuple[str, str]`, a tuple (<tree name>, <file name>) for raw data is expected
                                                                      # if `int`, phase-space distribution in angles is generated with given number of events
  cfg:              AnalysisConfig,
  massBinning:      HistAxisBinning,  # mass binning used for weighting
  massBinIndex:     int,              # index of mass bin to load/generate data for
  beamPolInfo:      BeamPolInfo | None = None,  # beam polarization information needed for raw data files
  limitNmbEventsTo: int | None         = None,  # if `int`, limits number of events to read from tree
) -> tuple[ROOT.RDataFrame, int, list[str]]:
  """Loads data specified by `inputDataDef` and returns them as RDataFrame and the number of input events."""
  if isinstance(inputDataDef, AnalysisConfig.DataType) or (isinstance(inputDataDef, tuple) and len(inputDataDef) == 2):
    dataToWeight = None
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
      assert beamPolInfo is not None, "Beam polarization information must be provided when loading raw data from file"
      lvs = lorentzVectors(dataFormat = InputDataFormat.AMPTOOLS)
      dataToWeight = defineDataFrameColumns(
        df          = dataToWeight,
        lvTarget    = lvs["target"],
        lvBeam      = lvs["beam"],
        lvRecoil    = lvs["recoil"],
        lvA         = lvs["pip"],
        lvB         = lvs["pim"],
        beamPolInfo = beamPolInfo,
        frame       = CoordSysType.HF,
        flipYAxis   = True,
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


def weightDataWithIntensity(
  inputDataDef:         AnalysisConfig.DataType | tuple[str, str] | int,  # if `AnalysisConfig.DataType` instance, the file corresponding to `DataType` is loaded
                                                                          # if `tuple[str, str]`, a tuple (<tree name>, <file name>) for raw data is expected
                                                                          # if `int`, phase-space distribution in angles is generated with given number of events
  massBinning:          HistAxisBinning,  # mass binning used for weighting
  massBinIndex:         int,              # index of mass bin to generate data for
  momentResults:        MomentResult,     # moment values used to construct intensity formula
  weightedDataFileName: str,              # ROOT file to which weighted events are written
  cfg:                  AnalysisConfig,
  seed:                 int                             = 123456789,  # seed for rejection sampling and for generating phase-space events
  beamPolInfo:          BeamPolInfo | None              = None,       # beam polarization information needed for raw data files
  useIntensityTerms:    MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.ALL,
  limitNmbEventsTo:     int | None                      = None,       # if `int`, limits number of events to read from tree
) -> None:
  """Weight input data specified by `inputDataDef` and `massBinIndex` with intensity formula from `momentResults` and write data to `outFileName`"""
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
  # construct intensity formula
  intensityFormula = momentResults.intensityFormula(
    polarization      = "beamPol",  # read polarization from tree column
    thetaFormula      = "theta",
    phiFormula        = "phi",
    PhiFormula        = "Phi",
    useIntensityTerms = useIntensityTerms,
  )
  print(f"Calculating weights using formula '{intensityFormula}'")
  # add columns for intensity weight and random number in [0, 1]
  dataToWeight = (
    dataToWeight.Define("intensityWeight",      f"(Double32_t){intensityFormula}")
                .Define("intensityWeightRndNmb", "(Double32_t)gRandom->Rndm()")  # random number in [0, 1] for each event
  )
  # write unweighted data to file and read data back to ensure that random columns are filled only once
  tmpFileName = f"{weightedDataFileName}.tmp"
  dataToWeight.Snapshot(cfg.treeName, tmpFileName)
  dataToWeight = ROOT.RDataFrame(cfg.treeName, tmpFileName)
  # determine maximum weight
  maxIntensityWeight = dataToWeight.Max("intensityWeight").GetValue()
  print(f"Maximum intensity is {maxIntensityWeight}")
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
  weightedData.Snapshot(cfg.treeName, weightedDataFileName, originalColumns)
  subprocess.run(f"rm --force --verbose {tmpFileName}", shell = True)  # remove temporary file


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
      f"{variableName}Distr", f";{variableName};Count",
      targetDistr.GetNbinsX(), targetDistr.GetXaxis().GetXmin(), targetDistr.GetXaxis().GetXmax()
    ),
    variableName,
  ).GetValue()
  if False:
    # save plots of current and target distributions
    canv = ROOT.TCanvas()
    currentDistr.Draw()
    canv.SaveAs(f"{currentDistr.GetName()}.root")
    canv = ROOT.TCanvas()
    targetDistr.Draw()
    canv.SaveAs(f"{targetDistr.GetName()}.root")
  # normalize histograms such that they represent the corresponding PDFs
  targetDistr.Scale (1.0 / targetDistr.Integral() )
  currentDistr.Scale(1.0 / currentDistr.Integral())
  # calculate the ratio of the target and the current PDF, that defines the weight histogram
  weightsHist = targetDistr.Clone("weightsHist")
  weightsHist.Divide(currentDistr)
  # add columns for rejection sampling to input data
  RootUtilities.declareInCpp(weightsHist = weightsHist)  # use Python TH1D object in C++  #TODO this can only be called once; otherwise this call crashes in ROOT
  dataToWeight = (
    dataToWeight.Define("reweightingWeight", f"(Double32_t)PyVars::weightsHist.GetBinContent(PyVars::weightsHist.FindBin({variableName}))")
                .Define("reweightingRndNmb",  "(Double32_t)gRandom->Rndm()")  # random number uniformly distributed in [0, 1]
  )
  tmpFileName = tempfile.mktemp(dir = "./", prefix = "unweighted.", suffix = ".root")
  dataToWeight.Snapshot(treeName, tmpFileName)  # write unweighted data to temporary file to ensure that random column is filled only once
  dataToWeight = ROOT.RDataFrame(treeName, tmpFileName)  # read data back from temporary file
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
  subprocess.run(f"rm --force --verbose {tmpFileName}", shell = True)
  return reweightedData


def reweightKinDistribution(
  dataToWeight:     ROOT.RDataFrame,  # data to reweight
  treeName:         str,              # name of TTree holding the data
  binning:          HistAxisBinning,  # binning of kinematic variable whose distribution is to be reweighted
  realDataFileName: str,              # name of file holding real data to construct target distribution from
  # momentResults:    MomentResultsKinematicBinning,  # moment values
  outFileName:      str,  # name of file to write data into
) -> None:
  """Reweights mass distribution of given data according to the mass dependence of H_0(0, 0)"""
  print(f"Reweighting {binning.var.name} dependence")
  # construct target distribution from real data
  print(f"Constructing target distribution from column '{binning.var.name}' in tree '{treeName}' in file '{realDataFileName}'")
  realData = ROOT.RDataFrame(treeName, realDataFileName)
  targetDistr = realData.Histo1D(
    ROOT.RDF.TH1DModel(f"{binning.var.name}DistrTarget", "Real data", *binning.astuple),
    binning.var.name,
    "eventWeight",
  ).GetValue()
  # # construct target distribution from H_0(0, 0) values in kinematic bins
  # targetDistr = ROOT.TH1D(f"{binning.var.name}DistrTarget", f";{binning.axisTitle};Count", *binning.astuple)
  # H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
  # for momentResultsForBin in momentResults:
  #   massBinCenter = momentResultsForBin.binCenters[binning.var]
  #   targetDistr.SetBinContent(targetDistr.FindBin(massBinCenter), momentResultsForBin[H000Index].real[0])
  # reweight data
  originalColumns = list(dataToWeight.GetColumnNames())
  reweightedData = reweightData(
    dataToWeight = dataToWeight,
    treeName     = treeName,
    variableName = binning.var.name,
    targetDistr  = targetDistr,
  )
  print(f"Writing reweighted data to file '{outFileName}'")
  reweightedData.Snapshot(treeName, outFileName, originalColumns)
  if True:
    # overlay target distribution and distribution after reweighting
    reweightedDistr = reweightedData.Histo1D(
      ROOT.RDF.TH1DModel(f"{binning.var.name}DistrReweighted", "Weighted MC", *binning.astuple),
      binning.var.name,
    ).GetValue()
    reweightedDistr.Scale(targetDistr.Integral() / reweightedDistr.Integral())
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
    canv.SaveAs(f"{outFileName}.{binning.var.name}.pdf")


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
  ROOT.gInterpreter.Declare(CPP_CODE_ANGLES_GLUEX_AMPTOOLS)
  ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)

  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_CLAS)  # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_PWA)   # perform analysis of unpolarized pi+ pi- data
  # cfg = deepcopy(CFG_UNPOLARIZED_PIPI_JPAC)  # perform analysis of unpolarized pi+ pi- data
  cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  # cfg = deepcopy(CFG_KEVIN)  # perform analysis of Kevin's polarized K- K_S Delta++ data

  dataBaseDirName          = "./dataPhotoProdPiPi/polarized"
  useIntensityTerms        = MomentResult.IntensityTermsType.ALL                # include parity-conserving and parity-violating terms into formula
  # useIntensityTerms        = MomentResult.IntensityTermsType.PARITY_CONSERVING  # include only parity-conserving terms
  # useIntensityTerms        = MomentResult.IntensityTermsType.PARITY_VIOLATING   # include only parity-violating terms
  # accepted phase-space data for generating kinematic plots in mass bins
  inputDataDef             = ("kin", "amptools_tree_accepted*.root")
  weightedDataFileBaseName = f"phaseSpace_acc_weighted_raw_{useIntensityTerms.value}"
  reweightMassDistribution = False
  limitNmbEventsTo         = None  # limit number of events to read from input tree
  # # accepted phase-space data for input-output studies with acceptance correction
  # inputDataDef             = AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE
  # weightedDataFileBaseName = f"phaseSpace_acc_weighted_flat_{useIntensityTerms.value}"
  # reweightMassDistribution = True  # reweight mass distribution after weighting with intensity function
  # limitNmbEventsTo         = None  # limit number of events to read from input tree
  # # generated phase-space data for input-output studies without acceptance correction
  # # inputDataDef             = 100000  # generate phase-space distribution in angles with given number of events
  # inputDataDef             = AnalysisConfig.DataType.GENERATED_PHASE_SPACE
  # weightedDataFileBaseName = f"phaseSpace_gen_weighted_flat_{useIntensityTerms.value}"
  # reweightMassDistribution = True  # reweight mass distribution after weighting with intensity function
  # limitNmbEventsTo         = 70000000  # limit number of events to read from input tree
  #
  dataPeriods = (
    # "2017_01",
    "2018_08",
  )
  tBinLabels = (
    "tbin_0.1_0.2",
    # "tbin_0.2_0.3",
    # "tbin_0.3_0.4",
    # "tbin_0.4_0.5",
  )
  beamPolLabels = (
    "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
    # "Unpol",
  )
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
  # makeIntensityFcnPlots   = True  # draw intensity function in each mass bin
  makeIntensityFcnPlots   = False
  massBinningForWeighting = deepcopy(cfg.massBinning)  # same binning as for moment values
  # massBinningForWeighting = HistAxisBinning(nmbBins = 250, minVal = 0.28, maxVal = 2.28)  # finer binning than for moment values
  massBinningForWeighting.var = cfg.binVarMass


  outFileDirBaseNameCommon = cfg.outFileDirBaseName
  for dataPeriod in dataPeriods:
    for tBinLabel in tBinLabels:
      for beamPolLabel in beamPolLabels:
        cfg.dataFileName       = f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/PiPi/data_flat_{beamPolLabel}.root"
        cfg.psAccFileName      = f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/PiPi/phaseSpace_acc_flat_{beamPolLabel}.root"
        cfg.psGenFileName      = f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/PiPi/phaseSpace_gen_flat_{beamPolLabel}.root"
        cfg.outFileDirBaseName = f"{outFileDirBaseNameCommon}/{dataPeriod}/{tBinLabel}/{beamPolLabel}"
        for maxL in maxLs:
          print(f"Generating weighted MC for data period '{dataPeriod}', t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
          cfg.maxL = maxL
          cfg.init()
          thisSourceFileName = os.path.basename(__file__)
          # create directory, into which weighted data will be written
          weightedDataDirName = f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/PiPi/weightedMc.maxL_{maxL}/{beamPolLabel}"
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
              weightDataWithIntensity(
                inputDataDef         = (inputDataDef[0], f"{dataBaseDirName}/{dataPeriod}/{tBinLabel}/Alex/{inputDataDef[1]}") \
                                       if isinstance(inputDataDef, tuple) else inputDataDef,
                massBinning          = massBinningForWeighting,
                massBinIndex         = massBinIndexForWeighting,
                momentResults        = momentResultsForBin,
                weightedDataFileName = f"{weightedDataDirName}/{weightedDataFileBaseName}_bin_{massBinIndexForWeighting}.root",
                cfg                  = cfg,
                seed                 = 12345 + massBinIndexForWeighting ,  # ensure rejection sampling and generated phase-space data in different mass bins are independent
                beamPolInfo          = BEAM_POL_INFOS[dataPeriod][beamPolLabel] if isinstance(inputDataDef, tuple) else None,
                useIntensityTerms    = useIntensityTerms,
                limitNmbEventsTo     = limitNmbEventsTo,
              )
              if makeIntensityFcnPlots:
                plotIntensityFcn(
                  momentResults     = momentResultsForBin,
                  massBinIndex      = massBinIndexForWeighting,
                  beamPolInfo       = BEAM_POL_INFOS[dataPeriod][beamPolLabel],  #TODO consistency of polarization value is only ensured for raw data
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
                  dataToWeight     = ROOT.RDataFrame(cfg.treeName, mergedFileName),  # load merged data file created in step above
                  treeName         = cfg.treeName,
                  binning          = massBinningForWeighting,
                  realDataFileName = cfg.dataFileName,
                  # momentResults    = momentResults,
                  outFileName      = reweightedFileName,
                )

            timer.stop("Total execution time")
            print(timer.summary)
