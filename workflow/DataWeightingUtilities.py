"""Module that provides utility functions for weighting data"""

from __future__ import annotations

from collections.abc import Sequence
import functools
import subprocess
import tempfile

import ROOT

from moments.MomentCalculator import (
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
from .AnalysisConfig import (
  AnalysisConfig,
  BeamPolInfo,
)
from .DataConversionUtilities import (
  defineDataFrameColumns,
  lorentzVectors,
)
from .PlottingUtilities import HistAxisBinning
from .RootUtilities import declareInCpp


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


def loadDataToWeight(
  inputDataDef:     tuple[str, str, AnalysisConfig.DataFormat] | int,
    # if `tuple`: a tuple (<file path>, <tree name>, <data format>) is expected
    # if `int`: phase-space distribution in angles is generated with given number of events
  cfg:              AnalysisConfig,
  massBinning:      HistAxisBinning,  # mass binning used for weighting
  massBinIndex:     int,              # index of mass bin to load/generate data for
  beamPolInfo:      BeamPolInfo | None = None,  # beam polarization information needed for raw data files
  limitNmbEventsTo: int | None         = None,  # if `int`, limits number of events to read from tree
) -> tuple[ROOT.RDataFrame, int, list[str]]:
  """Loads data specified by `inputDataDef` and returns them as RDataFrame and the number of input events."""
  dataToWeight = None
  if (isinstance(inputDataDef, tuple) and (len(inputDataDef) == 3)
      and isinstance(inputDataDef[0], str) and isinstance(inputDataDef[1], str) and isinstance(inputDataDef[2], AnalysisConfig.DataFormat)):
    filePath = inputDataDef[0]
    treeName = inputDataDef[1]
    dataToWeight = ROOT.RDataFrame(treeName, filePath)
    assert dataToWeight is not None, f"Could not load data defined by '{inputDataDef}'"
    if limitNmbEventsTo is not None:
      print(f"Limiting total number of input events (before binning) to {limitNmbEventsTo}")
      dataToWeight = dataToWeight.Range(0, limitNmbEventsTo)  # works only in single-thread mode
    originalColumns = list(dataToWeight.GetColumnNames())
    if isinstance(inputDataDef, tuple):
      # define columns needed to calculate intensity
      assert beamPolInfo is not None, "Beam polarization information must be provided when loading raw data from file"
      lvs = lorentzVectors(dataFormat = inputDataDef[2])
      dataToWeight = defineDataFrameColumns(
        df          = dataToWeight,
        lvTarget    = lvs["target"],
        lvBeam      = lvs["beam"],
        lvRecoil    = lvs[cfg.subsystem.lvRecoilLabel],
        lvA         = lvs[cfg.subsystem.lvALabel],
        lvB         = lvs[cfg.subsystem.lvBLabel],
        beamPolInfo = beamPolInfo,
        frame       = cfg.frame,
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
    if beamPolInfo is not None:
      # polarized case: add Phi and polarization columns
      dataToWeight = (
        dataToWeight.Define("PhiDeg", "(Double32_t)gRandom->Uniform(-180, +180)")
                    .Define("Phi",    "(Double32_t)(PhiDeg * TMath::DegToRad())")
      )
      dataToWeight = dataToWeight.Define("beamPol", f"(Double32_t){beamPolInfo.pol}")
    #TODO is a snapshot necessary here to fill random columns only once?
    return dataToWeight, nmbGenPsEvents, list(dataToWeight.GetColumnNames())
  else:
    raise ValueError(f"Invalid {inputDataDef=}")


def weightDataWithIntensityFormula(
  inputDataDef:         tuple[str, str, AnalysisConfig.DataFormat] | int,
    # if `tuple`: a tuple (<file path>, <tree name>, <data format>) is expected
    # if `int`: phase-space distribution in angles is generated with given number of events
  massBinning:          HistAxisBinning,  # mass binning used for weighting
  massBinIndex:         int,  # index of mass bin to generate data for
  intensityFormula:     str,  # intensity formula as function of theta [rad] phi [rad], and Phi [rad] that defines distribution of events
  weightedDataFilePath: str,  # ROOT file to which weighted events are written
  cfg:                  AnalysisConfig,
  seed:                 int                = 123456789,  # seed for rejection sampling and for generating phase-space events
  beamPolInfo:          BeamPolInfo | None = None,       # beam polarization information needed for raw data files
  limitNmbEventsTo:     int | None         = None,       # if `int`, limits number of events to read from tree
) -> ROOT.RDataFrame:
  """Weights input data specified by `inputDataDef` and `massBinIndex` with given intensity formula and writes data to `weightedDataFilePath`"""
  ROOT.gRandom.SetSeed(seed)
  # load input data
  dataToWeight, nmbInputEvents, originalColumns = loadDataToWeight(
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
    dataToWeight.Define("intensityWeight",      f"(Double32_t){intensityFormula}")  # intensity weight for each event
                .Define("intensityWeightRndNmb", "(Double32_t)gRandom->Rndm()")     # random number in [0, 1] for each event
  )
  # write unweighted data to file and read data back to ensure that random columns are filled only once
  tmpFilePath = f"{weightedDataFilePath}.tmp"
  treeName = cfg.inputTreeName if (isinstance(inputDataDef, tuple) and inputDataDef[1]) else cfg.convertedTreeName
  dataToWeight.Snapshot(treeName, tmpFilePath)
  dataToWeight = ROOT.RDataFrame(treeName, tmpFilePath)
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
  print(f"Writing data weighted with intensity function to file '{weightedDataFilePath}'")
  weightedData.Snapshot(treeName, weightedDataFilePath, originalColumns + ["intensityWeight", "mass"])  # write original columns and selected new columns
  # weightedData.Snapshot(treeName, weightedDataFilePath)  # write original columns + all columns defined here; !NOTE! the `phi` columns may trigger the ROOT bug https://github.com/root-project/root/issues/22295
  subprocess.run(f"rm --force --verbose {tmpFilePath}", shell = True)  # remove temporary file
  return ROOT.RDataFrame(treeName, weightedDataFilePath)
