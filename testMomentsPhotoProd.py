#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3


from __future__ import annotations

import functools
import numpy as np
import os
import threadpoolctl
from typing import (
  Callable,
  Iterable,
  TypedDict,
)

import ROOT

from AnalysisConfig import AnalysisConfig
from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  binLabel,
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentCalculatorsKinematicBinning,
  MomentIndices,
  MomentResult,
  MomentResultsKinematicBinning,
  QnWaveIndex,
)
from photoProdWeightData import weightDataWithIntensityFormula
from PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
  plotComplexMatrix,
  plotMoments1D,
  plotMomentsInBin,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# default TH3 plotting options for angular distributions
TH3_ANG_NMB_BINS = 25
TH3_ANG_BINNINGS = (
  HistAxisBinning(TH3_ANG_NMB_BINS,   -1,   +1),
  HistAxisBinning(TH3_ANG_NMB_BINS, -180, +180),
  HistAxisBinning(TH3_ANG_NMB_BINS, -180, +180),
)
TH3_ANG_TITLE = ";cos#theta;#phi [deg];#Phi [deg]"
class Th3PlotKwargsType(TypedDict):
  binnings:  tuple[HistAxisBinning, HistAxisBinning, HistAxisBinning]
  histTitle: str
TH3_ANG_PLOT_KWARGS: Th3PlotKwargsType = {"histTitle" : TH3_ANG_TITLE, "binnings" : TH3_ANG_BINNINGS}


def genDataFromIntensityFormula(
  nmbEvents:         int,  # number of events to generate
  intensityFormula:  str,  # intensity formula as function of x = cos(theta), y = phi [deg], z = Phi [deg] that defines distribution of events
  outFileBasePath:   str,  # path and base name of output files, i.e. .root and .pdf file
  regenerateData:    bool                      = False,   # if set data are regenerated although .root file exists
  treeName:          str                       = "data",  # name of the tree with generated data
  additionalColDefs: Iterable[tuple[str, str]] = (),      # additional column definitions to be added to the returned RDataFrame
) -> ROOT.RDataFrame:
  """Generates MC events according to the given intensity formula"""
  print(f"Generating events distributed according to intensity formula:\n{intensityFormula}")

  # construct and plot TF3 for intensity distribution
  baseName = os.path.basename(outFileBasePath)
  intensityFcn = ROOT.TF3(f"{baseName}_intensity", intensityFormula, -1, +1, -180, +180, -180, +180)
  intensityFcn.SetNpx(100)  # used in numeric integration performed by GetRandom3() used below
  intensityFcn.SetNpy(100)
  intensityFcn.SetNpz(100)
  # intensityFcn.SetMinimum(0)  # may be negative in pathological cases
  drawTF3(intensityFcn, **TH3_ANG_PLOT_KWARGS, outFileName = f"{outFileBasePath}_intensity.pdf")
  #TODO check for negative intensity values for wave set containing only P_+1^+ wave

  # if file with generated data already exists, read it and return RDataFrame
  fileName = f"{outFileBasePath}.root"
  if not regenerateData and os.path.exists(fileName):
    print(f"Reading generated MC data from existing file at '{fileName}'")
    df = ROOT.RDataFrame(treeName, fileName)
    nmbEvents = df.Count().GetValue()
    print(f"File '{fileName}' contains {nmbEvents} events")
    return df

  # else generate data and write to file
  print(f"Generating {nmbEvents} events and writing them to '{fileName}'")
  # Use TF3.GetRandom3() to generate random data points in angular space
  RootUtilities.declareInCpp(**{intensityFcn.GetName(): intensityFcn})  # use Python object in C++
  randomPointFcnCpp = f"""
    double cosTheta, phiDeg, PhiDeg;
    PyVars::{intensityFcn.GetName()}.GetRandom3(cosTheta, phiDeg, PhiDeg);
    const std::vector<double> dataPoint = {{cosTheta, phiDeg, PhiDeg}};
    return dataPoint;
  """  # C++ code that throws random point in angular space
  df = (
    ROOT.RDataFrame(nmbEvents)
        .Define("dataPoint", randomPointFcnCpp)
        .Define("cosTheta",  "(Double32_t)dataPoint[0]")
        .Define("theta",     "(Double32_t)std::acos(cosTheta)")
        .Define("phiDeg",    "(Double32_t)dataPoint[1]")
        .Define("phi",       "(Double32_t)(TMath::DegToRad() * phiDeg)")
        .Define("PhiDeg",    "(Double32_t)dataPoint[2]")
        .Define("Phi",       "(Double32_t)(TMath::DegToRad() * PhiDeg)")
        # add no-op filter that logs when event loop is running
        .Filter('if (rdfentry_ == 0) { cout << "Running event loop in `genDataFromIntensityFormula()`" << endl; } return true;')
  )  #!NOTE! for some reason, this is very slow
  columnsToWrite = ["cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi"]
  for colName, colDefinitions in additionalColDefs:
    print(f"Adding additional column '{colName}' with definitions '{colDefinitions}'")
    df = df.Define(colName, colDefinitions)
    columnsToWrite.append(colName)
  return df.Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](columnsToWrite))  # snapshot is needed or else the `dataPoint` column would be regenerated for every triggered loop


def genData(
  nmbEvents:         int,           # number of events to generate
  polarization:      float | None,  # photon-beam polarization
  inputData:         AmplitudeSet | MomentResult,  # generate data either from set of partial-wave amplitudes or from moment result
  outFileBasePath:   str,  # path and base name of output files, i.e. .root and .pdf file
  efficiencyFormula: str | None                = None,    # detection efficiency used to generate data
  regenerateData:    bool                      = False,   # if set data are regenerated although .root file exists
  treeName:          str                       = "data",  # name of the tree with generated data
  additionalColDefs: Iterable[tuple[str, str]] = (),      # additional column definitions to be added to the returned RDataFrame
) -> ROOT.RDataFrame:
  """Generates data according to given set of partial-wave amplitudes (assuming rank 1) or moment result and the given detection efficiency"""
  print(f"Generating {nmbEvents} events for photon-beam polarization {polarization} and efficiency {efficiencyFormula}")
  print(f"Events will be distributed according to intensity defined by\n{inputData}")

  # construct and draw efficiency function
  efficiencyFcn = ROOT.TF3(f"efficiencyGen", efficiencyFormula if efficiencyFormula else "1", -1, +1, -180, +180, -180, +180)
  efficiencyFcn.SetNpx(100)
  efficiencyFcn.SetNpy(100)
  efficiencyFcn.SetNpz(100)
  drawTF3(efficiencyFcn, **TH3_ANG_PLOT_KWARGS, maxVal = 1.0,
    outFileName = f"{outFileBasePath}_efficiency.pdf")

  # construct TF3 for intensity distribution in Eq. (171)
  # x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
  intensityFormula = inputData.intensityFormula(
    polarization = polarization,
    thetaFormula = "std::acos(x)",
    phiFormula   = "TMath::DegToRad() * y",
    PhiFormula   = "TMath::DegToRad() * z",
    printFormula = False,
  )

  # apply efficiency weighting to intensity formula
  if efficiencyFormula:
    intensityFormula = f"({intensityFormula}) * ({efficiencyFormula})"

  return genDataFromIntensityFormula(
    nmbEvents         = nmbEvents,
    intensityFormula  = intensityFormula,
    outFileBasePath   = outFileBasePath,
    regenerateData    = regenerateData,
    treeName          = treeName,
    additionalColDefs = additionalColDefs,
  )


BIN_CONTENT_3D_FUNCTOR_CPP = """
// Functor that returns the bin content of a 3D histogram at the given (x, y, z) point
class BinContent3DFunctor {
public:

	BinContent3DFunctor(TH3* hist)
	: _hist(hist)
	{ }

	double
	operator () (
		const double x,
		const double y,
		const double z
	) {
		if (
			   (x < _hist->GetXaxis()->GetBinLowEdge(1) or x > _hist->GetXaxis()->GetBinUpEdge(_hist->GetNbinsX()))
			or (y < _hist->GetYaxis()->GetBinLowEdge(1) or y > _hist->GetYaxis()->GetBinUpEdge(_hist->GetNbinsY()))
			or (z < _hist->GetZaxis()->GetBinLowEdge(1) or z > _hist->GetZaxis()->GetBinUpEdge(_hist->GetNbinsZ()))
		) {
			return 0;
		}
		return _hist->GetBinContent(_hist->FindBin(x, y, z));
	}

	protected:
	TH3* _hist;
};
"""


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  setupPlotStyle()
  ROOT.gStyle.SetOptStat(False)

  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gInterpreter.Declare(BIN_CONTENT_3D_FUNCTOR_CPP)
  ROOT.gRandom.SetSeed(1234567890)
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 4):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of test case
    # outputDirName         = Utilities.makeDirPath("./plotsTestPhotoProd")
    outputDirName         = Utilities.makeDirPath("./plotsTestPhotoProd.momentsRd.accEven.phys.detuneAccFull")
    # nmbDataEvents         = 1000
    # nmbAccPsEvents        = 1000000
    nmbDataEvents         = 1000000
    nmbAccPsEvents        = 10000000
    # beamPolarization      = 1.0
    beamPolarization      = 0.3563  # Fall 2018, PARA_0
    binVarMass            = KinematicBinningVariable(name = "mass", label = "#it{m}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
    massBinning           = HistAxisBinning(nmbBins = 1, minVal = 0.72, maxVal = 0.76, _var = binVarMass)
    maxL                  = 4  # define maximum L quantum number of moments
    partialWaveAmplitudes = [  # set of all possible waves up to ell = 2
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
    ]
    amplitudeSet = AmplitudeSet(partialWaveAmplitudes)

    # xVar = "x"
    # yVar = "y"
    # zVar = "z"
    xVar = "cosTheta"
    yVar = "phiDeg"
    zVar = "PhiDeg"
    # formulas for detection efficiency
    # x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
    # efficiencyFormula = "1"  # no_acc; perfect efficiency
    # efficiencyFormula = f"(1.5 - {xVar} * {xVar}) * (1.5 - {yVar} * {yVar} / (180 * 180)) * (1.5 - {zVar} * {zVar} / (180 * 180)) / pow(1.5, 3)"  # acc_1; even in all variables
    # efficiencyFormula = f"(0.75 + 0.25 * {xVar}) * (0.75 + 0.25 * ({yVar} / 180)) * (0.75 + 0.25 * ({zVar} / 180))"  # acc_2; odd in all variables
    # efficiencyFormula = f"(0.6 + 0.4 * {xVar}) * (0.6 + 0.4 * ({yVar} / 180)) * (0.6 + 0.4 * ({zVar} / 180))"  # acc_3; odd in all variables
    # accPsHistName = "accPs3D"  # accFull
    accPsHistName = "accPs3D_even"  # accEven
    accPsFileName = f"./dataPhotoProdPiPi/{accPsHistName}.root"
    accPsFileGen = ROOT.TFile.Open(accPsFileName, "READ")
    accPSHistGen = accPsFileGen[accPsHistName]
    print(f"Using 3D acceptance histogram '{accPSHistGen.GetName()}' in file '{accPsFileName}' to generate events: min = {accPSHistGen.GetMinimum()}, max = {accPSHistGen.GetMaximum()}")
    accPSHistGenFunctor = ROOT.BinContent3DFunctor(accPSHistGen)
    RootUtilities.declareInCpp(accPSHistGenFunctor = accPSHistGenFunctor)  # make Python object available to use in C++
    efficiencyFormula = f"(1 / {accPSHistGen.GetMaximum()}) * PyVars::accPSHistGenFunctor({xVar}, {yVar}, {zVar})"

    # define holes in efficiency
    efficiencyHoleGen = ""  # do not punch hole in efficiency when generating data
    # efficiencyHoleGen = f"!((0.3 < {xVar} && {xVar} < 0.7) && (-180 < {yVar} && {yVar} < -120))"  # hole in efficiency when generating data
    # efficiencyHoleGen = "!(" \
    #   f"   ((-0.9 < {xVar} && {xVar} < -0.2) && ( -30 < {yVar} && {yVar} <  +30))" \
    #   f"|| ((+0.2 < {xVar} && {xVar} < +0.9) && (-180 < {yVar} && {yVar} < -150))" \
    #   f"|| ((+0.2 < {xVar} && {xVar} < +0.9) && (+150 < {yVar} && {yVar} < +180))" \
    # ")"  # 3 holes in efficiency similar to 0.72-0.76 mass bin when generating data
    # efficiencyHoleGen = f"!((0 < {xVar} && {xVar} < 1) && (-180 < {yVar} && {yVar} < 0))"  # large hole (whole quadrant) in efficiency when generating data
    # efficiencyHoleGen = f"({yVar} > 0)"  # accept only upper half plane
    # efficiencyHoleGen = f"({yVar} < 0)"  # accept only lower half plane
    # efficiencyHoleReco = ""  # do not punch hole in efficiency when analyzing data
    efficiencyHoleReco = efficiencyHoleGen  # hole in efficiency when analyzing data chosen to be same as hole used when generating data
    # efficiencyHoleReco = f"!((0.35 < {xVar} && {xVar} < 0.65) && (-180 < {yVar} && {yVar} < -140))"  # hole in efficiency when analyzing data chosen to be smaller than hole used when generating data
    # efficiencyHoleReco = f"!((0.25 < {xVar} && {xVar} < 0.75) && (-180 < {yVar} && {yVar} < -100))"  # hole in efficiency when analyzing data chosen to be bigger than hole used when generating data

    # detune efficiency used to correct efficiency w.r.t. the one used to generate the data
    efficiencyFormulaDetune = ""
    # efficiencyFormulaDetune = f"0.1 * (1.5 - {yVar} * {yVar} / (180 * 180)) / 1.5"  # detuneEven1; detune by even terms in phi only
    # efficiencyFormulaDetune = f"0.1 * (1.5 - {xVar} * {xVar}) * (1.5 - {zVar} * {zVar} / (180 * 180)) / pow(1.5, 2)"  # detuneEven2; detune by even terms in cos(theta) and Phi
    # efficiencyFormulaDetune = f"0.1 * (1.5 - {xVar} * {xVar}) * (1.5 - {yVar} * {yVar} / (180 * 180)) * (1.5 - {zVar} * {zVar} / (180 * 180)) / pow(1.5, 3)"  # detuneEven3; detune by even terms in all variables
    # efficiencyFormulaDetune = f"0.1 * sin({yVar} * TMath::DegToRad())"  # detuneOdd1; detune by odd terms in phi only
    # efficiencyFormulaDetune = f"0.1 * (1 + sin({yVar} * TMath::DegToRad()))"  # detuneOdd2; detune by odd terms in phi only
    # efficiencyFormulaDetune = f"0.1 * (1 + {yVar} / 180)"  # detuneOdd3; detune by odd terms in phi only
    # # use 2D histogram in (cos theta, phi) plane to detune efficiency

    # accPsFile = ROOT.TFile.Open("./dataPhotoProdPiPi/anglesHFPiPi_0.72_0.76_odd.root", "READ")
    # accPSHist = accPsFile["anglesHFPiPi_0.72_0.76_odd"]
    # accPSHistFunctor = ROOT.BinContent2DFunctor(accPSHist)
    # RootUtilities.declareInCpp(accPSHistFunctor = accPSHistFunctor)  # make Python object available to use in C++
    # accPSHistRange = max(abs(accPSHist.GetMaximum()), abs(accPSHist.GetMinimum()))
    # efficiencyFormulaDetune = f"(0.15 / {accPSHistRange}) * PyVars::accPSHistFunctor({xVar}, {yVar})"  # detuneOdd4; detune by odd terms in phi only
    # efficiencyFormulaDetune = f"(0.35 + 0.15 * {xVar}) * (0.35 + 0.15 * ({yVar} / 180)) * (0.35 + 0.15 * ({zVar} / 180))"  # detuneOdd5; detune by odd terms in all variables
    accPsHistName = "accPs3D"  # detuneAccFull
    # accPsHistName = "accPs3D_even"  # detuneAccEven
    accPsFileName = f"./dataPhotoProdPiPi/{accPsHistName}.root"
    accPsFileReco = ROOT.TFile.Open(accPsFileName, "READ")
    accPSHistReco = accPsFileReco[accPsHistName]
    print(f"Using 3D acceptance histogram '{accPSHistReco.GetName()}' in file '{accPsFileName}' to analyze data: min = {accPSHistReco.GetMinimum()}, max = {accPSHistReco.GetMaximum()}")
    accPSHistRecoFunctor = ROOT.BinContent3DFunctor(accPSHistReco)
    RootUtilities.declareInCpp(accPSHistRecoFunctor = accPSHistRecoFunctor)  # make Python object available to use in C++
    efficiencyFormulaReco = f"(1 / {accPSHistReco.GetMaximum()}) * PyVars::accPSHistRecoFunctor({xVar}, {yVar}, {zVar})"

    # efficiencyFormulaGen = efficiencyFormulaReco = efficiencyFormula
    efficiencyFormulaGen = efficiencyFormula
    # insert efficiency holes
    if efficiencyHoleGen:
      efficiencyFormulaGen = f"(({efficiencyFormulaGen}) * ({efficiencyHoleGen}))"
    if efficiencyHoleReco:
      efficiencyFormulaReco = f"(({efficiencyFormulaReco}) * ({efficiencyHoleReco}))"
    # apply detuning to efficiency used when analyzing data
    if efficiencyFormulaDetune:
      efficiencyFormulaReco = f"(({efficiencyFormulaReco}) + ({efficiencyFormulaDetune}))"
    nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()

    # calculate true moment values and generate data from partial-wave amplitudes
    t = timer.start("Time to generate MC data from partial waves")
    # HTruth: MomentResult = amplitudeSet.photoProdMomentResult(maxL, normalize = False)
    momentResultsFileName = f"./plotsPhotoProdPiPiPol/2018_08/tbin_0.1_0.2/PARA_0.maxL_4/unnorm_moments_phys.pkl"
    print(f"Reading moments from file '{momentResultsFileName}'")
    HTruth = MomentResultsKinematicBinning.loadPickle(momentResultsFileName)[11]  # pick [0.72, 0.76] GeV bin
    if True:
    # if False:
      # set all unphysical parts of moment values to zero
      for flatIndex in range(len(HTruth._valsFlatIndex)):
        qnIndex = HTruth.indices[flatIndex]
        if qnIndex.momentIndex in (0, 1):
          HTruth._valsFlatIndex[flatIndex] = HTruth._valsFlatIndex[flatIndex].real
        elif qnIndex.momentIndex == 2:
          HTruth._valsFlatIndex[flatIndex] = 1j * HTruth._valsFlatIndex[flatIndex].imag
        else:
          raise ValueError(f"Unexpected moment index in {qnIndex=}")
    print(f"True moment values\n{HTruth}")
    if False:
      dataIntensity = genData(
        nmbEvents         = nmbDataEvents,
        polarization      = beamPolarization,
        # inputData         = amplitudeSet,  # must yield the same intensity distribution as MomentResult
        inputData         = HTruth,
        outFileBasePath   = f"{outputDirName}/data",
        efficiencyFormula = efficiencyFormulaGen,
        regenerateData    = False,
      )
    else:
      intensityFormula = HTruth.intensityFormula(
        polarization      = beamPolarization,
        # thetaFormula      = "std::acos(x)",
        # phiFormula        = "TMath::DegToRad() * y",
        # PhiFormula        = "TMath::DegToRad() * z",
        thetaFormula      = "theta",
        phiFormula        = "phi",
        PhiFormula        = "Phi",
        useIntensityTerms = MomentResult.IntensityTermsType.ALL,
      )
      dataIntensity = weightDataWithIntensityFormula(
        inputDataDef         = nmbDataEvents,
        massBinning          = massBinning,
        massBinIndex         = 0,
        intensityFormula     = f"({intensityFormula}) * ({efficiencyFormulaGen})",
        weightedDataFileName = f"{outputDirName}/data.root",
        cfg                  = AnalysisConfig(treeName = "data", polarization = beamPolarization),
        seed                 = 1234567890,
      )
    t.stop()

    # if True:
    if False:
      # plot intensity and efficiency functions used to generate data
      for label, formula in {
        "efficiencyFormulaGen"     : efficiencyFormulaGen,
        "efficiencyFormulaReco"    : efficiencyFormulaReco,
        "efficiencyFormulaDetune"  : efficiencyFormulaDetune,
        "intensityFormula"         : intensityFormula,
        "intensityFormulaAccepted" : f"({intensityFormula}) * ({efficiencyFormulaGen})",
      }.items():
        fcn = ROOT.TF3(label, formula, -1, +1, -180, +180, -180, +180)
        drawTF3(fcn, **TH3_ANG_PLOT_KWARGS, outFileName = f"{outputDirName}/{label}.pdf")
    # raise SystemExit("STOP")

    # generate accepted phase-space data
    t = timer.start("Time to generate phase-space MC data")
    if False:
      dataAccPs = genDataFromIntensityFormula(
        nmbEvents        = nmbAccPsEvents,
        intensityFormula = efficiencyFormulaReco,
        outFileBasePath  = f"{outputDirName}/acceptedPhaseSpace",
        regenerateData   = False,
      )
    else:
      dataAccPs = weightDataWithIntensityFormula(
        inputDataDef         = nmbAccPsEvents,
        massBinning          = massBinning,
        massBinIndex         = 0,
        intensityFormula     = efficiencyFormulaReco,
        weightedDataFileName = f"{outputDirName}/acceptedPhaseSpace.root",
        cfg                  = AnalysisConfig(treeName = "data", polarization = beamPolarization),
        seed                 = 1234567890,
      )
    t.stop()

    # plot data generated from partial-wave amplitudes and accepted phase-space data generated using `efficiencyFormulaReco`
    for data in (dataIntensity, dataAccPs):
      nmbBins = 25
      hists = (
        data.Histo2D(
          ROOT.RDF.TH2DModel(
            "data2D" if data is dataIntensity else "accPs2D", ";cos#theta;#phi [deg]",
            100, -1, +1, 72, -180, +180,
          ),
          "cosTheta", "phiDeg",
        ),
        data.Histo3D(
          ROOT.RDF.TH3DModel(
            "data3D" if data is dataIntensity else "accPs3D", ";cos#theta;#phi [deg];#Phi [deg]",
            nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180,
          ),
          "cosTheta", "phiDeg", "PhiDeg",
        ),
      )
      for hist in hists:
        canv = ROOT.TCanvas()
        hist.SetMinimum(0)
        hist.SetStats(False)
        if hist.GetDimension() == 2:
          hist.Draw("COLZ")
        elif hist.GetDimension() == 3:
          hist.GetXaxis().SetTitleOffset(1.5)
          hist.GetYaxis().SetTitleOffset(2)
          hist.GetZaxis().SetTitleOffset(1.5)
          hist.Draw("BOX2Z")
        canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")
        canv.SaveAs(f"{outputDirName}/{hist.GetName()}.root")

    # setup moment calculator
    momentIndices = MomentIndices(maxL)
    momentsInBins:      list[MomentCalculator] = []
    momentsInBinsTruth: list[MomentCalculator] = []
    for massBinCenter in massBinning:
      # dummy bins with identical data sets
      dataSet = DataSet(
        data           = dataIntensity,
        phaseSpaceData = dataAccPs,
        nmbGenEvents   = nmbAccPsEvents,
        polarization   = beamPolarization
      )  #TODO nmbPsMcEvents is not correct number to normalize integral matrix
      momentsInBins.append(
        MomentCalculator(
          indicesMeas          = momentIndices,
          indicesPhys          = momentIndices,
          dataSet              = dataSet,
          integralFileBaseName = f"{outputDirName}/integralMatrix",
          binCenters           = {binVarMass : massBinCenter},
        )
      )
    moments = MomentCalculatorsKinematicBinning(momentsInBins)

    # calculate integral matrices
    t = timer.start(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")
    moments.calculateIntegralMatrices(forceCalculation = False)
    # print acceptance integral matrix for first kinematic bin
    print(f"Acceptance integral matrix for first bin\n{moments[0].integralMatrix}")
    eigenVals, _ = moments[0].integralMatrix.eigenDecomp
    print(f"Eigenvalues of acceptance integral matrix for first bin\n{np.sort(eigenVals)}")
    # plot acceptance integral matrices for all kinematic bins
    for HData in moments:
      label = binLabel(HData)
      plotComplexMatrix(moments[0].integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outputDirName}/I_acc_{label}")
      plotComplexMatrix(moments[0].integralMatrix.inverse,          pdfFileNamePrefix = f"{outputDirName}/I_inv_{label}")
    t.stop()

    # calculate moments of data generated from partial-wave amplitudes
    t = timer.start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
    moments.calculateMoments(normalize = False)
    # print all moments for first kinematic bin
    print(f"Measured moments of data generated according to partial-wave amplitudes for first kinematic bin\n{moments[0].HMeas}")
    print(f"Physical moments of data generated according to partial-wave amplitudes for first kinematic bin\n{moments[0].HPhys}")

    # plot moments in each kinematic bin
    momentResultsPhys  = moments.momentResultsPhys
    momentResultsTruth = MomentResultsKinematicBinning([HTruth] * len(moments))  # dummy truth values; identical for all bins
    normalizationFactor = momentResultsTruth.normalizeTo(momentResultsPhys)  # normalize true moments to data
    for massBinIndex, HPhys in enumerate(momentResultsPhys):
      plotMomentsInBin(
        HData             = HPhys,
        normalizedMoments = False,
        HTruth            = momentResultsTruth[massBinIndex],
        outFileNamePrefix = f"{outputDirName}/h{binLabel(HPhys)}_",
      )
    if False:
      # plot kinematic dependences of all moments #TODO normalize H_0(0, 0) to total number of events
      for qnIndex in momentIndices.qnIndices:
        plotMoments1D(
          momentResults     = momentResultsPhys,
          qnIndex           = qnIndex,
          binning           = massBinning,
          normalizedMoments = False,
          momentResultsTrue = momentResultsTruth,
          outFileNamePrefix = f"{outputDirName}/h",
          histTitle         = qnIndex.title,
        )
    t.stop()

    timer.stop("Total execution time")
    print(timer.summary)
