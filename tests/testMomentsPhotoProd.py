#!/usr/bin/env python3

"""Performs input-output test for moment calculation for linearly polarized photoproduction"""

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3


from __future__ import annotations

from collections.abc import (
  Callable,
  Iterable,
)
from copy import deepcopy
import functools
import numpy as np
import os
import threadpoolctl
from typing import TypedDict

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`
from wurlitzer import pipes, STDOUT

from moments.MomentCalculator import (
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
from scripts.calculateMoments import CPP_CODE_IS_IN_EFFICIENCY_HOLES
from workflow.DataWeightingUtilities import (
  CPP_CODE_RDF_RANDOM_NMB_GENERATOR,
  weightDataWithIntensityFormula,
)
from workflow.AnalysisConfig import (
  BeamPolInfo,
  BEAM_POL_INFOS,
  CFG_POLARIZED_PIPI,
)
from workflow.PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
  plotComplexMatrix,
  plotMomentsKinVar,
  plotMomentsInBin,
  setupPlotStyle,
)
from workflow import RootUtilities
from workflow import Utilities


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

PARTIAL_WAVE_AMPLITUDES = [  # set of all possible waves up to ell = 2
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
  drawTF3(intensityFcn, **TH3_ANG_PLOT_KWARGS, outFilePath = f"{outFileBasePath}_intensity.pdf")
  #TODO check for negative intensity values for wave set containing only P_+1^+ wave

  # if file with generated data already exists, read it and return RDataFrame
  filePath = f"{outFileBasePath}.root"
  if not regenerateData and os.path.exists(filePath):
    print(f"Reading generated MC data from existing file at '{filePath}'")
    df = ROOT.RDataFrame(treeName, filePath)
    nmbEvents = df.Count().GetValue()
    print(f"File '{filePath}' contains {nmbEvents} events")
    return df

  # else generate data and write to file
  print(f"Generating {nmbEvents} events and writing them to '{filePath}'")
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
  return df.Snapshot(treeName, filePath, ROOT.std.vector[ROOT.std.string](columnsToWrite))  # snapshot is needed or else the `dataPoint` column would be regenerated for every triggered loop


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
    outFilePath = f"{outFileBasePath}_efficiency.pdf")

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


CPP_CODE_BIN_CONTENT_3D_FUNCTOR = r"""
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


CPP_CODE_SMEAR_PARTICLE_DIRECTION = r"""
#include <cmath>

#include "Math/Vector3D.h"

using ROOT::Math::Polar3DVector;
using ROOT::Math::XYZVector;


// Smears the direction of a particle given its polar and azimuthal
// angles (theta, phi) by Gaussian displacements in the plane tangent
// to the particle direction and along the directions defined by these
// angles, i.e. the smearing is independent of the chosen coordinate
// axes.  The approximation is valid for small displacement angles,
// where the tangent-plane displacement can be interpreted as a
// rotation along a great circle.  For large sigmas, the smeared
// direction should be sampled from a Kent distribution
// <https://en.wikipedia.org/wiki/Kent_distribution>.  The smeared
// particle direction is returned as a Polar3DVector.
Polar3DVector
smearParticleDirection(
	const std::size_t slot,
	const double      theta,              // polar angle of the particle direction [rad]
	const double      phi,                // azimuthal angle of the particle direction [rad]
	const double      sigmaThetaTangent,  // standard deviation of the polar angle in the plane transverse to the particle direction [rad]
	const double      sigmaPhiTangent     // standard deviation of the azimuthal angle in the plane transverse to the particle direction [rad]
) {
  const Polar3DVector particleDirectionPolar(1, theta, phi);  // unit vector of the particle direction in polar coordinates
	const XYZVector particleDirection(particleDirectionPolar);  // unit vector of the particle direction in Cartesian coordinates
	const XYZVector unitVectorTheta(
		 std::cos(theta) * std::cos(phi),
		 std::cos(theta) * std::sin(phi),
		-std::sin(theta)
	);  // unit vector along the polar angle direction (theta)
	const XYZVector unitVectorPhi(
		-std::sin(phi),
		 std::cos(phi),
		 0
	);  // unit vector along the azimuthal angle direction (phi)

	// throw Gaussian displacements along the theta and phi directions
  // Note: this assumes that rdfRng is defined and initialized by the calling code
	const double deltaTheta = rdfRng::gaussian(slot, 0.0, sigmaThetaTangent);
	const double deltaPhi   = rdfRng::gaussian(slot, 0.0, sigmaPhiTangent);

	const double delta = std::hypot(deltaTheta, deltaPhi);  // total displacement in the tangent plane, interpreted as a rotation angle in radians
	if (delta <= 1e-15) {
		return particleDirectionPolar;
	}

	const XYZVector tangentDirection = (deltaTheta * unitVectorTheta + deltaPhi * unitVectorPhi) / delta;  // unit vector of displacement in the tangent plane
	// The tangent displacement vector and the particle direction vector
	// are perpendicular unit vectors and define the plane of rotation
	// for the displacement angle.  Hence, we can use the two vectors as
	// coordinate axes to express the rotation by the displacement angle.
	const XYZVector smearedParticleDirection = std::cos(delta) * particleDirection + std::sin(delta) * tangentDirection;
	return Polar3DVector(smearedParticleDirection);
}
"""


def smearAngles(
  dataToSmear:          ROOT.RDataFrame,
  sigmaThetaTangent:    float,
  sigmaPhiTangent:      float,
  sigmaPhiPolarization: float,
) -> ROOT.RDataFrame:
  """Smears the particle direction by Gaussian displacements in the plane transverse to the particle direction and the polarization plane angle by a Gaussian"""
  dataToSmear = (
    dataToSmear.Define("particleDirectionSmeared", f"smearParticleDirection(rdfslot_, theta, phi, {sigmaThetaTangent}, {sigmaPhiTangent})")
               .Redefine("theta", "particleDirectionSmeared.theta()")
               .Redefine("phi",   "particleDirectionSmeared.phi()")
               .Redefine("Phi",  f"rdfRng::gaussian(rdfslot_, Phi, {sigmaPhiPolarization})")
  )
  return dataToSmear


# taken from https://root-forum.cern.ch/t/can-we-shift-histogram-for-several-channels/12050/2
#TODO move to more suitable module
def scaleHistAxes(
  axes:     Iterable[ROOT.TAxis],
  scaleFcn: Callable[[float], float],
) -> None:
  """Scales the given axes with the given linear scaling function"""
  for axis in axes:
    if (axis.IsVariableBinSize()):  # the axis has non-equidistant bins
      #!NOTE! bins must remain in increasing order, hence the `scaleFcn` function
      # must be strictly (monotonically) increasing
      xBins = axis.GetXbins();  # get TArrayD with current x-bin borders
      for i in range(xBins.GetSize()):
        xBins[i] = scaleFcn(xBins[i]);  # scale x-bin borders
      axis.Set(xBins.GetSize() - 1, xBins.GetArray());  # set new x-bin borders
    else:  # the axis has equidistant bins
      #!NOTE! we modify Xmin and Xmax only, hence the `scaleFcn` function
      # must be linear (and Xmax must remain greater than Xmin)
      axis.Set(
        axis.GetNbins(),
        scaleFcn(axis.GetXmin()),  # new Xmin
        scaleFcn(axis.GetXmax())   # new Xmax
      )


if __name__ == "__main__":
  RootUtilities.loadBasisFunctionsLibrary(enableOpenMp = True)  # initializes OpenMP and loads `cpp/basisFunctions.C`
  ROOT.gROOT.SetBatch(True)
  nmbThreads = 4
  # ROOT.EnableImplicitMT(nmbThreads)  # enable multi-threading with the specified number of threads
  setupPlotStyle()
  # ROOT.gStyle.SetOptStat(False)

  # set parameters of test case
  cfg                       = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized gamma p -> (pi+ pi-) p data
  maxL                      = 4  # define maximum L quantum number of moments
  # maxL                      = 6
  # maxL                      = (6, 12)
  # maxL                      = 8
  outputDirPath             = Utilities.makeDirPath("./plots/testPhotoProd.maxL_4")
  # nmbDataEvents             = 1000
  # nmbAccPsEvents            = 100000
  nmbDataEvents             = 10000000
  nmbAccPsEvents            = 100000000
  beamPolInfo: BeamPolInfo  = BEAM_POL_INFOS["2017_01"]["PARA_0"]
  binVarMass                = KinematicBinningVariable(name = "mass", label = "#it{m}", unit = "GeV/#it{c}^{2}", nmbDigits = 2)
  massBinning               = HistAxisBinning(nmbBins = 1, minVal = 0.72, maxVal = 0.76, _var = binVarMass)  # single bin at rho(770) peak position
  momentResultsMassBinIndex = 3  # index of the mass bin in the moment results
  generateDataFromPwa       = False  # is set to `False`, data are generated from moments
  simulateResolution        = False  # if True, particle directions and polarization-plane angle are smeared bhy Gaussians
  sigmaThetaTangent         = 1  # standard deviation of the polar angle in the plane transverse to the particle direction [rad]
  sigmaPhiTangent           = 1  # standard deviation of the azimuthal angle in the plane transverse to the particle direction [rad]
  sigmaPhiPolarization      = 1  # standard deviation of the polarization-plane angle [rad]

  # redirect all console output below into log file
  thisSourceFileName = os.path.basename(__file__)
  logFilePath = f"{outputDirPath}/{os.path.splitext(thisSourceFileName)[0]}.log"
  print(f"Writing output to log file '{logFilePath}'")
  with open(logFilePath, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):
    Utilities.printGitInfo()
    timer = Utilities.Timer()
    ROOT.gInterpreter.Declare(CPP_CODE_BIN_CONTENT_3D_FUNCTOR)
    ROOT.gInterpreter.Declare(CPP_CODE_IS_IN_EFFICIENCY_HOLES)
    ROOT.gInterpreter.Declare(CPP_CODE_RDF_RANDOM_NMB_GENERATOR)
    ROOT.rdfRng.initialize(1234567890)
    ROOT.gInterpreter.Declare(CPP_CODE_SMEAR_PARTICLE_DIRECTION)
    threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
    print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
    with threadController.limit(limits = nmbThreads):
      print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
      timer.start("Total execution time")
      Utilities.printResourceUsage()
      ROOT.gROOT.GetListOfFiles().Print()

      xVar = "cosTheta"
      yVar = "phiDeg"
      zVar = "PhiDeg"
      # formulas for detection efficiency
      # x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
      # efficiencyFormula = "1"  # no_acc; perfect efficiency
      # efficiencyFormula = f"(1.5 - {xVar} * {xVar}) * (1.5 - {yVar} * {yVar} / (180 * 180)) * (1.5 - {zVar} * {zVar} / (180 * 180)) / pow(1.5, 3)"  # acc_1; even in all variables
      # efficiencyFormula = f"(0.75 + 0.25 * {xVar}) * (0.75 + 0.25 * ({yVar} / 180)) * (0.75 + 0.25 * ({zVar} / 180))"  # acc_2; odd in all variables
      # efficiencyFormula = f"(0.6 + 0.4 * {xVar}) * (0.6 + 0.4 * ({yVar} / 180)) * (0.6 + 0.4 * ({zVar} / 180))"  # acc_3; odd in all variables
      accPsHistNameGen = "accPs3D"  # full 3D acceptance histogram
      # accPsHistNameGen = "accPs3D_even"  # phi_HF-even component of the 3D acceptance histogram
      accPsFilePathGen = f"./data/PiPi/{accPsHistNameGen}.root"
      accPsFileGen = ROOT.TFile.Open(accPsFilePathGen, "READ")
      accPSHistGen = accPsFileGen[accPsHistNameGen]
      print(f"Using 3D acceptance histogram '{accPSHistGen.GetName()}' in file '{accPsFilePathGen}' to generate events: min = {accPSHistGen.GetMinimum()}, max = {accPSHistGen.GetMaximum()}")
      accPSHistGenFunctor = ROOT.BinContent3DFunctor(accPSHistGen)
      RootUtilities.declareInCpp(accPSHistGenFunctor = accPSHistGenFunctor)  # make Python object available to use in C++
      efficiencyFormula = f"(1 / {accPSHistGen.GetMaximum()}) * PyVars::accPSHistGenFunctor({xVar}, {yVar}, {zVar})"

      # define holes in efficiency when generating data
      efficiencyHoleGen = ""  # do not punch hole
      # efficiencyHoleGen = f"!isInEfficiencyHoles({xVar}, {yVar})"  # holes
      # efficiencyHoleGen = f"!((0.3 < {xVar} && {xVar} < 0.7) && (-180 < {yVar} && {yVar} < -120))"  # small hole
      # efficiencyHoleGen = "!(" \
      #   f"   ((-0.9 < {xVar} && {xVar} < -0.2) && ( -30 < {yVar} && {yVar} <  +30))" \
      #   f"|| ((+0.2 < {xVar} && {xVar} < +0.9) && (-180 < {yVar} && {yVar} < -150))" \
      #   f"|| ((+0.2 < {xVar} && {xVar} < +0.9) && (+150 < {yVar} && {yVar} < +180))" \
      # ")"  # holes; similar to 0.72-0.76 mass bin
      # efficiencyHoleGen = f"!((0 < {xVar} && {xVar} < 1) && (-180 < {yVar} && {yVar} < 0))"  # large hole (whole quadrant) in efficiency when generating data
      # half-plane holes
      # efficiencyHoleGen = f"({xVar} < 0)"  # cosThetaNeg
      # efficiencyHoleGen = f"({xVar} > 0)"  # cosThetaPos
      # efficiencyHoleGen = f"({yVar} < 0)"  # phiNeg
      # efficiencyHoleGen = f"({yVar} > 0)"  # phiPos
      # efficiencyHoleGen = f"({zVar} < 0)"  # PhiNeg
      # efficiencyHoleGen = f"({zVar} > 0)"  # PhiPos
      # cut out spike regions in phi-odd component of efficiency
      # efficiencyHoleGen = f"(-0.75 < {xVar} && {xVar} < +0.75)"  # cosThetaNoSpikeOdd
      # efficiencyHoleGen = f"((-150 < {yVar} && {yVar} < -30) || (+30 < {yVar} && {yVar} < +150))"  # phiNoSpikeOdd
      # efficiencyHoleGen = "!(" \
      #   f"   ((-1 < {xVar} && {xVar} < -0.75) && (-180 < {yVar} && {yVar} < -150))" \
      #   f"|| ((-1 < {xVar} && {xVar} < -0.75) && (+150 < {yVar} && {yVar} < +180))" \
      #   f"|| ((+0.75 < {xVar} && {xVar} < +1) && ( -30 < {yVar} && {yVar} <  +30))" \
      # ")"  # noSpikeOdd
      # define holes in efficiency when analyzing data
      # efficiencyHoleReco = ""  # do not punch hole
      efficiencyHoleReco = efficiencyHoleGen  # same as hole as used when generating data
      # efficiencyHoleReco = f"!((0.35 < {xVar} && {xVar} < 0.65) && (-180 < {yVar} && {yVar} < -140))"  # 1 hole smaller than the one used when generating data
      # efficiencyHoleReco = f"!((0.25 < {xVar} && {xVar} < 0.75) && (-180 < {yVar} && {yVar} < -100))"  # 1 hole bigger than the one used when generating data

      # detune efficiency used to correct efficiency w.r.t. the one used to generate the data
      efficiencyFormulaDetune = ""
      # efficiencyFormulaDetune = f"0.1 * (1.5 - {yVar} * {yVar} / (180 * 180)) / 1.5"  # detuneEven1; detune by even terms in phi only
      # efficiencyFormulaDetune = f"0.1 * (1.5 - {xVar} * {xVar}) * (1.5 - {zVar} * {zVar} / (180 * 180)) / pow(1.5, 2)"  # detuneEven2; detune by even terms in cos(theta) and Phi
      # efficiencyFormulaDetune = f"0.1 * (1.5 - {xVar} * {xVar}) * (1.5 - {yVar} * {yVar} / (180 * 180)) * (1.5 - {zVar} * {zVar} / (180 * 180)) / pow(1.5, 3)"  # detuneEven3; detune by even terms in all variables
      # efficiencyFormulaDetune = f"0.1 * sin({yVar} * TMath::DegToRad())"  # detuneOdd1; detune by odd terms in phi only
      # efficiencyFormulaDetune = f"0.1 * (1 + sin({yVar} * TMath::DegToRad()))"  # detuneOdd2; detune by odd terms in phi only
      # efficiencyFormulaDetune = f"0.1 * (1 + {yVar} / 180)"  # detuneOdd3; detune by odd terms in phi only
      # # use 2D histogram in (cos theta, phi) plane to detune efficiency
      # accPsFile = ROOT.TFile.Open("./data/PiPi/anglesHFPiPi_0.72_0.76_odd.root", "READ")
      # accPSHist = accPsFile["anglesHFPiPi_0.72_0.76_odd"]
      # accPSHistFunctor = ROOT.BinContent2DFunctor(accPSHist)
      # RootUtilities.declareInCpp(accPSHistFunctor = accPSHistFunctor)  # make Python object available to use in C++
      # accPSHistRange = max(abs(accPSHist.GetMaximum()), abs(accPSHist.GetMinimum()))
      # efficiencyFormulaDetune = f"(0.15 / {accPSHistRange}) * PyVars::accPSHistFunctor({xVar}, {yVar})"  # detuneOdd4; detune by odd terms in phi only
      # efficiencyFormulaDetune = f"(0.35 + 0.15 * {xVar}) * (0.35 + 0.15 * ({yVar} / 180)) * (0.35 + 0.15 * ({zVar} / 180))"  # detuneOdd5; detune by odd terms in all variables
      # accPsHistNameReco = "accPs3D"  # detuneAccFull
      # # accPsHistNameReco = "accPs3D_even"  # detuneAccEven
      # accPsFilePathReco = f"./data/PiPi/{accPsHistNameReco}.root"
      # accPsFileReco = ROOT.TFile.Open(accPsFilePathReco, "READ")
      # accPSHistReco = accPsFileReco[accPsHistNameReco]
      # print(f"Using 3D acceptance histogram '{accPSHistReco.GetName()}' in file '{accPsFilePathReco}' to analyze data: min = {accPSHistReco.GetMinimum()}, max = {accPSHistReco.GetMaximum()}")
      # accPSHistRecoFunctor = ROOT.BinContent3DFunctor(accPSHistReco)
      # RootUtilities.declareInCpp(accPSHistRecoFunctor = accPSHistRecoFunctor)  # make Python object available to use in C++
      # efficiencyFormulaReco = f"(1 / {accPSHistReco.GetMaximum()}) * PyVars::accPSHistRecoFunctor({xVar}, {yVar}, {zVar})"

      efficiencyFormulaGen = efficiencyFormulaReco = efficiencyFormula
      # efficiencyFormulaGen = efficiencyFormula
      # insert efficiency holes
      if efficiencyHoleGen:
        efficiencyFormulaGen = f"(({efficiencyFormulaGen}) * ({efficiencyHoleGen}))"
      if efficiencyHoleReco:
        efficiencyFormulaReco = f"(({efficiencyFormulaReco}) * ({efficiencyHoleReco}))"
      # apply detuning to efficiency used when analyzing data
      if efficiencyFormulaDetune:
        efficiencyFormulaReco = f"(({efficiencyFormulaReco}) + ({efficiencyFormulaDetune}))"
      nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()
      Utilities.printResourceUsage()
      ROOT.gROOT.GetListOfFiles().Print()

      # calculate true moment values and generate data
      t = timer.start("Time to generate MC data")
      HTruth: MomentResult | None = None
      dataIntensity: ROOT.RDataFrame | None = None
      if generateDataFromPwa:
        print(f"Using PWA amplitudes to generate MC data")
        amplitudeSet = AmplitudeSet(PARTIAL_WAVE_AMPLITUDES)
        HTruth = amplitudeSet.photoProdMomentResult(maxL, normalize = False)
      else:
        momentResultsFilePath = f"plots/PiPiPol/2017_01_ver05/tbin_0.100_0.114/PARA_0.maxL_6/unnorm_moments_phys_shifted.pkl"
        print(f"Using moments from file '{momentResultsFilePath}' to generate MC data")
        HTruth = MomentResultsKinematicBinning.loadPickle(momentResultsFilePath)[momentResultsMassBinIndex]  # pick [0.72, 0.76] GeV bin
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
      print(f"Generating {nmbDataEvents} data events from true moment values for bin {HTruth.binCenters}\n{HTruth}")
      # if True:
      if False:
        # generate data by randomly drawing from the product of the intensity
        # function defined by MomentResult and the efficiency function
        dataIntensity = genData(
          nmbEvents         = nmbDataEvents,
          polarization      = beamPolInfo.pol,
          # inputData         = amplitudeSet,  # must yield the same intensity distribution as MomentResult
          inputData         = HTruth,
          outFileBasePath   = f"{outputDirPath}/data",
          efficiencyFormula = efficiencyFormulaGen,  #TODO transform variables to (x, y, z)
          regenerateData    = False,
        )
      else:
        # generate data using rejection sampling from a phase-space
        # sample with weights given by intensity and efficiency formula
        intensityFormula = HTruth.intensityFormula(
          polarization      = beamPolInfo.pol,
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
          weightedDataFilePath = f"{outputDirPath}/data.root",
          cfg                  = cfg,
          seed                 = 1234567890,
          beamPolInfo          = beamPolInfo,
        )
      t.stop()
      Utilities.printResourceUsage()
      ROOT.gROOT.GetListOfFiles().Print()

      # simulate resolution effects by smearing angles of generated data
      if simulateResolution:
        t = timer.start("Time to smear angles")
        print("Smearing angles of generated data")
        dataIntensity = smearAngles(
          dataToSmear          = dataIntensity,
          sigmaThetaTangent    = sigmaThetaTangent,
          sigmaPhiTangent      = sigmaPhiTangent,
          sigmaPhiPolarization = sigmaPhiPolarization,
        )
        t.stop()
        Utilities.printResourceUsage()
        ROOT.gROOT.GetListOfFiles().Print()

      # plot intensity and efficiency functions used to generate data
      if True:
      # if False:
        #!NOTE! this does not work well for histogram-based efficiencies because binning used in plotting is coarser than histogram binning
        #       to fix this plotting should use the same binning and then a Rebin3D has to be applied
        for label, formula in {
          "efficiencyFormulaGen"     : efficiencyFormulaGen,
          "efficiencyFormulaReco"    : efficiencyFormulaReco,
          # "efficiencyFormulaDetune"  : efficiencyFormulaDetune,
          "intensityFormula"         : intensityFormula,
          "intensityFormulaAccepted" : f"({intensityFormula}) * ({efficiencyFormulaGen})",
        }.items():
          # transform variables to (x, y, z) so that formula can be used to construct a TF3
          for old, new in (("cosTheta", "x"), ("phiDeg", "y"), ("PhiDeg", "z"),
                          ("theta", "std::acos(x)"), ("phi", "TMath::DegToRad() * y"), ("Phi", "TMath::DegToRad() * z")):
            formula = formula.replace(old, new)
          fcn = ROOT.TF3(label, formula, -1, +1, -180, +180, -180, +180)
          drawTF3(fcn, **TH3_ANG_PLOT_KWARGS, outFilePath = f"{outputDirPath}/{label}.pdf")
        Utilities.printResourceUsage()
        ROOT.gROOT.GetListOfFiles().Print()
      # raise SystemExit("STOP")

      # generate accepted phase-space data
      print(f"Generating {nmbAccPsEvents} accepted phase-space events")
      t = timer.start("Time to generate phase-space MC data")
      if False:
        dataAccPs = genDataFromIntensityFormula(
          nmbEvents        = nmbAccPsEvents,
          intensityFormula = efficiencyFormulaReco,
          outFileBasePath  = f"{outputDirPath}/acceptedPhaseSpace",
          regenerateData   = False,
        )
      else:
        dataAccPs = weightDataWithIntensityFormula(
          inputDataDef         = nmbAccPsEvents,
          massBinning          = massBinning,
          massBinIndex         = 0,
          intensityFormula     = efficiencyFormulaReco,
          weightedDataFilePath = f"{outputDirPath}/acceptedPhaseSpace.root",
          cfg                  = cfg,
          seed                 = 1234567890,
          beamPolInfo          = beamPolInfo,
        )
      t.stop()
      Utilities.printResourceUsage()
      ROOT.gROOT.GetListOfFiles().Print()

      # plot data and accepted phase-space data
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
          canv.SaveAs(f"{outputDirPath}/{hist.GetName()}.pdf")
          canv.SaveAs(f"{outputDirPath}/{hist.GetName()}.root")
      Utilities.printResourceUsage()
      ROOT.gROOT.GetListOfFiles().Print()

      # setup moment calculator
      momentsInBins:      list[MomentCalculator] = []
      momentsInBinsTruth: list[MomentCalculator] = []
      for massBinCenter in massBinning:
        # dummy bins with identical data sets
        dataSet = DataSet(
          data           = dataIntensity,
          phaseSpaceData = dataAccPs,
          nmbGenEvents   = nmbAccPsEvents,  #TODO nmbAccPsEvents is not correct number to normalize integral matrix
          polarization   = beamPolInfo.pol
        )
        momentsInBins.append(
          MomentCalculator(
            indicesMeas          = MomentIndices(maxL if isinstance(maxL, int) else maxL[1]),
            indicesPhys          = MomentIndices(maxL if isinstance(maxL, int) else maxL[0]),
            dataSet              = dataSet,
            integralFileBaseName = f"{outputDirPath}/integralMatrix",
            binCenters           = {binVarMass : massBinCenter},
          )
        )
      moments = MomentCalculatorsKinematicBinning(momentsInBins)
      Utilities.printResourceUsage()
      ROOT.gROOT.GetListOfFiles().Print()

      # calculate integral matrices
      t = timer.start(f"Time to calculate integral matrices using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateIntegralMatrices(forceCalculation = False)
      # print acceptance integral matrix for first kinematic bin
      print(f"Acceptance integral matrix for first bin\n{moments[0].integralMatrix}")
      if isinstance(maxL, int):
        eigenVals, _ = moments[0].integralMatrix.eigenDecomp
        print(f"Eigenvalues of acceptance integral matrix for first bin\n{np.sort(eigenVals)}")
      # plot acceptance integral matrices for all kinematic bins
      for HData in moments:
        label = binLabel(HData)
        if isinstance(maxL, int):
          plotComplexMatrix(moments[0].integralMatrix.matrixNormalized, pdfFileNamePrefix = f"{outputDirPath}/I_acc_norm_{label}")
        else:
          plotComplexMatrix(moments[0].integralMatrix.matrix, pdfFileNamePrefix = f"{outputDirPath}/I_acc_{label}")
        plotComplexMatrix(moments[0].integralMatrix.inverse, pdfFileNamePrefix = f"{outputDirPath}/I_inv_{label}")
      t.stop()
      Utilities.printResourceUsage()
      ROOT.gROOT.GetListOfFiles().Print()

      # calculate moments from data
      t = timer.start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
      moments.calculateMoments(normalize = False)
      # print all moments for first kinematic bin
      print(f"Measured moments:\n{moments[0].HMeas}")
      print(f"Physical moments:\n{moments[0].HPhys}")
      Utilities.printResourceUsage()
      ROOT.gROOT.GetListOfFiles().Print()

      # plot moments in each kinematic bin
      ROOT.gStyle.SetOptStat(False)
      momentResultsPhys  = moments.momentResultsPhys
      momentResultsTruth = MomentResultsKinematicBinning([HTruth] * len(moments))  # dummy truth values; identical for all bins
      normalizationFactor = momentResultsTruth.normalizeTo(momentResultsPhys)  # normalize true moments to data
      for massBinIndex, HPhys in enumerate(momentResultsPhys):
        plotMomentsInBin(
          HData             = HPhys,
          normalizedMoments = False,
          HTruth            = momentResultsTruth[massBinIndex],
          outFileNamePrefix = f"{outputDirPath}/h{binLabel(HPhys)}_",
        )
      if False:
        # plot kinematic dependences of all moments #TODO normalize H_0(0, 0) to total number of events
        for qnIndex in momentResultsPhys[0].indices.qnIndices:
          plotMomentsKinVar(
            momentResults     = momentResultsPhys,
            qnIndex           = qnIndex,
            binning           = massBinning,
            normalizedMoments = False,
            momentResultsTrue = momentResultsTruth,
            outFileNamePrefix = f"{outputDirPath}/h",
            histTitle         = qnIndex.title,
          )
      t.stop()

      timer.stop("Total execution time")
      print(timer.summary)
      Utilities.printResourceUsage()
