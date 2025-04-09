#!/usr/bin/env python3
"""
Test script that performs closure test for moment analysis that uses
extended maximum likelihood-fitting of the data.
"""


from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import functools
import numpy as np
import nptyping as npt
import os
import textwrap
import threadpoolctl
from typing import Sequence


import ROOT
import iminuit as im
import iminuit.cost as cost
from iminuit.typing import (
  Cost,
  Model,
)
from wurlitzer import pipes, STDOUT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  MomentIndices,
  MomentResult,
  QnMomentIndex,
  QnWaveIndex,
)
from PlottingUtilities import (
  drawTF3,
  plotMomentsInBin,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
from testMomentsPhotoProd import TH3_ANG_PLOT_KWARGS
from testMomentsPhotoProdWeighted import genSigAndBkgDataFromWaves
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# TINY_FLOAT = np.finfo(dtype = np.double).tiny
TINY_FLOAT = np.finfo(dtype = float).tiny


def genAccepted2BodyPs(
  nmbGenEvents:      int,  # number of phase-space events to generate
  efficiencyFormula: str | None = None,   # detection efficiency used for acceptance correction
  regenerateData:    bool       = False,  # if set data are regenerated although .root file exists
  outFileNamePrefix: str        = "./",   # name prefix for output files
  columnsToWrite:    list[str]  = ["theta", "phi", "Phi"],  # columns to write to file
) -> ROOT.RDataFrame:
  """Generates RDataFrame with two-body phase-space distribution weighted by given detection efficiency"""
  print("Drawing efficiency function")
  efficiencyFcn = ROOT.TF3("efficiency", efficiencyFormula if efficiencyFormula else "1", -1, +1, -180, +180, -180, +180)
  drawTF3(efficiencyFcn, **TH3_ANG_PLOT_KWARGS, pdfFileName = f"{outFileNamePrefix}{efficiencyFcn.GetName()}Reco.pdf", nmbPoints = 100, maxVal = 1.0)

  # don't regenerate data if file already exists
  treeName = "data"
  outFileNameAccPs = f"{outFileNamePrefix}acceptedPhaseSpace.root"
  if os.path.exists(outFileNameAccPs) and not regenerateData:
    print(f"Reading accepted phase-space MC data from '{outFileNameAccPs}'")
    return ROOT.RDataFrame(treeName, outFileNameAccPs)

  print(f"Generating {nmbGenEvents} events distributed according to two-body phase-space")
  # generate isotropic distributions in cos theta, phi, and Phi
  outFileNamePs = f"{outFileNamePrefix}phaseSpace.root"
  # C++ code that throws random point in angular space
  pointFcn = """
    const double cosTheta = gRandom->Uniform(-1, +1);
    const double phi      = gRandom->Uniform(-TMath::Pi(), +TMath::Pi());  // [rad]
    const double Phi      = gRandom->Uniform(-TMath::Pi(), +TMath::Pi());  // [rad]
    const std::vector<double> point = {cosTheta, phi, Phi};
    return point;
  """
  psData = (
    ROOT.RDataFrame(nmbGenEvents)
        .Define("point",    pointFcn)
        .Define("cosTheta", "point[0]")
        .Define("theta",    "std::acos(cosTheta)")
        .Define("phi",      "point[1]")
        .Define("Phi",      "point[2]")
        .Filter('if (rdfentry_ == 0) { std::cout << "Running event loop in genData2BodyPs()" << std::endl; } return true;')  # noop filter that logs when event loop is running
        .Snapshot(treeName, outFileNamePs, ROOT.std.vector[ROOT.std.string](columnsToWrite))  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
  )

  print(f"Weighting phase-space events with efficiency function '{efficiencyFormula}'")
  RootUtilities.declareInCpp(efficiencyFcn = efficiencyFcn)  # use Python object in C++
  psAccData = (
    psData.Define("efficiencyWeight", f"(Double32_t)PyVars::efficiencyFcn.Eval(cos(theta), TMath::RadToDeg() * phi, TMath::RadToDeg() * Phi)")
          .Define("rndNmb",            "(Double32_t)gRandom->Rndm()")
  )
  # determine maximum weight
  maxEfficiencyWeight = psAccData.Max("efficiencyWeight").GetValue()
  print(f"Maximum efficiency weight is {maxEfficiencyWeight}")
  print(f"Weighting phase-space events with efficiency and writing accepted phase-space events to file '{outFileNameAccPs}'")
  psAccData = (
    psAccData.Define("acceptEvent", f"(bool)(rndNmb < (efficiencyWeight / {maxEfficiencyWeight}))")
             .Filter("acceptEvent == true")  # accept each event with probability efficiencyWeight / maxEfficiencyWeight
             .Snapshot(treeName, outFileNameAccPs, ROOT.std.vector[ROOT.std.string](columnsToWrite))
  )
  return psAccData


def defineIntensityFcnVectorizedCpp(intensityFormula: str) -> None:
  """Defines the vectorized C++ intensity function using TFormula and OpenMP"""
  ROOT.gInterpreter.Declare(f'TFormula intensityFormula = TFormula("intensity", "{intensityFormula}");')
  ROOT.gInterpreter.Declare(
    """
    std::vector<double>
    intensityFcnVectorized(
      const std::vector<double>& thetas,  // polar angles of analyzer [rad]
      const std::vector<double>& phis,    // azimuthal angles of analyzer [rad]
      const std::vector<double>& Phis,    // azimuthal angle of beam-polarization vector w.r.t. production plane [rad]
      const std::vector<double>& moments
    ) {
      const size_t nmbEvents = thetas.size();
      assert(phis.size() == nmbEvents);
      assert(Phis.size() == nmbEvents);
      auto intensities = std::vector<double>(nmbEvents);
      // multi-threaded loop over events using OpenMP
      #pragma omp parallel for
      for (size_t i = 0; i < nmbEvents; ++i) {
        const double angles[3] = {thetas[i], phis[i], Phis[i]};
        intensities[i] = intensityFormula.EvalPar(angles, moments.data());
      }
      return intensities;
    }
    """
  )


def intensityFcnVectorized(
  dataPoints: tuple[
    npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],  # theta values
    npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],  # phi values
    npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],  # Phi values
  ],
  moments: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64],
) -> tuple[float, npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]]:
  """Wrapper function that calls the vectorized C++ intensity function"""
  thetas, phis, Phis = dataPoints
  intensities = np.array(ROOT.intensityFcnVectorized(
    np.ascontiguousarray(thetas),
    np.ascontiguousarray(phis),
    np.ascontiguousarray(Phis),
    moments,
  ))
  # this is a bit hacky; before calling this function we need to set the phase-space efficiency from MC data
  integral = moments[0] * intensityFcnVectorized.efficiency
  return (integral, intensities)


ROOT.gInterpreter.Declare(
"""
// basis functions for physical moments; Eq. (XXX)
// scalar version that calculates function value for a single event
double
f_basis(
	const int    momentIndex,  // 0, 1, or 2
	const int    L,
	const int    M,
	const double theta,  // [rad]
	const double phi,    // [rad]
	const double Phi,    // [rad]
	const double polarization
) {
	const double commonTerm = std::sqrt((2 * L + 1) / (4 * TMath::Pi())) * ((M == 0) ? 1 : 2) * ylm(L, M, theta);
	switch (momentIndex) {
	case 0:
		return  commonTerm * std::cos(M * phi);
	case 1:
		return  commonTerm * std::cos(M * phi) * polarization * std::cos(2 * Phi);
	case 2:
		return -commonTerm * std::sin(M * phi) * polarization * std::sin(2 * Phi);
	default:
		throw std::domain_error("f_basis() unknown moment index.");
	}
}
"""
)

ROOT.gInterpreter.Declare(
"""
// vector version that calculates basis-function value for each entry
// in the input vectors for a constant polarization value
std::vector<double>
f_basis(
	const int                  momentIndex,  // 0, 1, or 2
	const int                  L,
	const int                  M,
	const std::vector<double>& theta,  // [rad]
	const std::vector<double>& phi,    // [rad]
	const std::vector<double>& Phi,    // [rad]
	const double               polarization
) {
	// assume that theta, phi, and Phi have the same length
	const size_t nmbEvents = theta.size();
	std::vector<double> fcnValues(nmbEvents);
	// multi-threaded loop over events using OpenMP
	#pragma omp parallel for
	for (size_t i = 0; i < nmbEvents; ++i) {
		fcnValues[i] = f_basis(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
	}
	return fcnValues;
}
"""
)


@dataclass
class IntensityFcnVectorized:
  """Functor that calculates intensities from real data and integrals from accepted phase-space MC data"""
  indices:           MomentIndices  # indices that define set of moments to fit
  beamPol:           float  # polarization of photon beam
  # arrays with real-data values for theta, phi, and Phi angles
  _thetas:           npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] | None             = None
  _phis:             npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] | None             = None
  _Phis:             npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] | None             = None
  # arrays with accepted phase-space values for theta, phi, and Phi angles
  _thetasAccPs:      npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] | None             = None
  _phisAccPs:        npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] | None             = None
  _PhisAccPs:        npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] | None             = None
  _baseFcnVals:      npt.NDArray[npt.Shape["nmbMoments, nmbEvents"], npt.Float64] | None = None  # precalculated real-data values of basis functions
  _baseFcnIntegrals: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64] | None            = None  # precalculated accepted phase-space integrals of basis functions

  def precalcBasisFcnValues(
    self,
    thetas: npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
    phis:   npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
    Phis:   npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
  ) -> None:
    """Copies real data and precalculates values of basis functions for all real-data events"""
    self._thetas = np.ascontiguousarray(thetas).copy()
    self._phis   = np.ascontiguousarray(phis).copy()
    self._Phis   = np.ascontiguousarray(Phis).copy()
    # check that all data arrays have the same length
    nmbEvents = len(self._thetas)
    assert nmbEvents == len(self._phis) == len(self._Phis), f"Data arrays must have same length; but got {len(self._thetas)=} vs. {len(self._phis)=} vs. {len(self._Phis)=}"
    nmbMoments = len(self.indices)
    # precalculate real-data values of basis functions
    self._baseFcnVals = np.zeros((nmbMoments, nmbEvents), dtype = np.double)
    for flatIndex in self.indices.flatIndices:
      qnIndex = self.indices[flatIndex]
      self._baseFcnVals[flatIndex] = np.asarray(ROOT.f_basis(
        qnIndex.momentIndex, qnIndex.L, qnIndex.M,
        self._thetas,
        self._phis,
        self._Phis,
        self.beamPol if self.beamPol is not None else 0.0,  #TODO add signature with event-by-event polarization
      ))

  def precalcBasisFcnAccPsIntegrals(
    self,
    thetas:       npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64],
    phis:         npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64],
    Phis:         npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64],
    nmbGenEvents: int,  # number of generated phase-space events
  ) -> None:
    """Stores reference to accepted phase-space data and precalculates accepted phase-space integrals for all basis functions"""
    #TODO do we really need these member variables?
    self._thetasAccPs = np.ascontiguousarray(thetas)
    self._phisAccPs   = np.ascontiguousarray(phis)
    self._PhisAccPs   = np.ascontiguousarray(Phis)
    # check that all data arrays have the same length
    nmbAccEvents = len(self._thetasAccPs)
    assert nmbAccEvents == len(self._phisAccPs) == len(self._PhisAccPs), f"Data arrays must have same length; but got {len(self._thetasAccPs)=} vs. {len(self._phisAccPs)=} vs. {len(self._PhisAccPs)=}"
    nmbMoments = len(self.indices)
    self._baseFcnIntegrals = np.zeros((nmbMoments, ), dtype = np.double)
    for flatIndex in self.indices.flatIndices:
      # calculate basis-functions value for each accepted phase-space event
      qnIndex = self.indices[flatIndex]
      baseFcnValsAccPs = np.asarray(ROOT.f_basis(
        qnIndex.momentIndex, qnIndex.L, qnIndex.M,
        self._thetasAccPs,
        self._phisAccPs,
        self._PhisAccPs,
        self.beamPol if self.beamPol is not None else 0.0,  #TODO add signature with event-by-event polarization
      ))
      # calculate accepted phase-space integral by summing basis-functions values over accepted phase-space events
      #TODO add event weights
      self._baseFcnIntegrals[flatIndex] = (4 * np.pi / nmbGenEvents) * np.sum(np.sort(baseFcnValsAccPs))  # sort values to make sum more accurate  #TODO why not 8 * np.pi**2?
    print(f"!!! {self._baseFcnIntegrals=}")

  def __call__(
    self,
    dataPoints: tuple[
      npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],  # theta values
      npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],  # phi values
      npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],  # Phi values
    ],
    moments: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64]
  ) -> tuple[np.float64, npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]]:
    """Wrapper function that returns the normalization integral of intensity function and the intensities for each event"""
    thetas, phis, Phis = dataPoints
    # precalculate the basis-function values if input data are not set or if input data change
    if (   (self._baseFcnVals is None)
        or (self._thetas      is None)
        or (self._phis        is None)
        or (self._Phis        is None)
        or (not np.array_equal(thetas, self._thetas))
        or (not np.array_equal(phis,   self._phis))
        or (not np.array_equal(Phis,   self._Phis))):
      print("Warning: Input data where not yet set or they changed. Precalculating real-data values of basis functions. This should happen only once at the start of the minimization.")
      self.precalcBasisFcnValues(thetas, phis, Phis)
    assert not np.may_share_memory(thetas, self._thetas), f"Argument and cached array for `theta` must not share memory"
    assert not np.may_share_memory(phis,   self._phis),   f"Argument and cached array for `phi` must not share memory"
    assert not np.may_share_memory(Phis,   self._Phis),   f"Argument and cached array for `Phi` must not share memory"
    # calculate intensities for all events
    intensities: npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] = np.dot(moments, self._baseFcnVals)
    # calculate integral value
    assert self._baseFcnIntegrals is not None, "Need to call `IntensityFcnVectorized.precalcBasisFcnAccPsIntegrals()` before calling the functor"
    integral = np.dot(moments, self._baseFcnIntegrals)
    return (integral, intensities)


@dataclass
class ExtendedUnbinnedWeightedNLL:
  """Negative log-likelihood function for extended unbinned maximum likelihood fit with weighted events; !NOTE! the sum of weights must be identical to the number of signal events"""
  intensityFcn: Model  # callable that returns the normalization integral of intensity function and the intensities for each event
  thetas:       npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64]
  phis:         npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64]
  Phis:         npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64]
  eventWeights: npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64]

  def __call__(
    self,
    moments: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64]
  ) -> np.float64:
    """Negative log-likelihood function for intensity as a function of moment parameters"""
    integral, intensities = intensityFcn(dataPoints = (self.thetas, self.phis, self.Phis), moments = moments)
    # nonPositiveIntensities = intensities[intensities <= 0]
    # nonPositiveIndices = np.where(intensities <= 0)[0]
    # if nonPositiveIntensities.size > 0:
    #   print(f"Warning: ignoring non-positive intensities: {nonPositiveIntensities}")
    #   print(f"!!! {self.eventWeights[nonPositiveIndices]=}")
    weightedLogIntensities = self.eventWeights * np.log(intensities + TINY_FLOAT)  # protect against 0 intensities
    return -(np.sum(np.sort(weightedLogIntensities)).item() - integral)  # sort summands to make sum more accurate; use item() to ensure scalar quantity


def convertIminuitToMomentResult(
  minuit:  im.Minuit,      # iminuit object containing fit result
  indices: MomentIndices,  # indices that define set of moments that was fit
) -> MomentResult:
    """Converts iminuit result into `MomentResult` object"""
    HPhys = MomentResult(indices)
    if indices.polarized:
      # construct quantum-number index ranges for purely real and purely imaginary moments
      reIndexRange = (
        QnMomentIndex(momentIndex = 0, L = 0,            M = 0),
        QnMomentIndex(momentIndex = 1, L = indices.maxL, M = indices.maxL),
      )  # all H_0 and H_1 moments are real-valued
      imIndexRange = (
        QnMomentIndex(momentIndex = 2, L = 1,            M = 1),
        QnMomentIndex(momentIndex = 2, L = indices.maxL, M = indices.maxL)
      )  # all H_2 moments are purely imaginary
      # convert to flat-index ranges
      reSlice = slice(indices[reIndexRange[0]], indices[reIndexRange[1]] + 1)
      imSlice = slice(indices[imIndexRange[0]], indices[imIndexRange[1]] + 1)
      # copy values
      par = np.array(minuit.values)
      HPhys._valsFlatIndex[reSlice] = par[reSlice]
      HPhys._valsFlatIndex[imSlice] = par[imSlice] * 1j  # convert to purely imaginary
      # copy covariance matrix
      cov = minuit.covariance
      HPhys._V_ReReFlatIndex[reSlice, reSlice] = np.array(cov[reSlice, reSlice])
      HPhys._V_ImImFlatIndex[imSlice, imSlice] = np.array(cov[imSlice, imSlice])
      HPhys._V_ReImFlatIndex[reSlice, imSlice] = np.array(cov[reSlice, imSlice])
    else:
      # copy values
      HPhys._valsFlatIndex[:] = np.array(minuit.values)
      # copy covariance matrix
      HPhys._V_ReReFlatIndex[:] = np.array(minuit.covariance[:])
    return HPhys


def fitSuccess(
  minuit:  im.Minuit,
  verbose: bool = False,
) -> bool:
  if verbose:
    print(textwrap.dedent(
      f"""
      Fit success:
          {minuit.valid=}
          {minuit.fmin.has_covariance=}
          {minuit.fmin.has_accurate_covar=}
          {minuit.fmin.has_posdef_covar=}
          {minuit.fmin.has_made_posdef_covar=}
          {minuit.fmin.hesse_failed=}
          {minuit.fmin.is_above_max_edm=}
          {minuit.fmin.has_reached_call_limit=}
          {minuit.fmin.has_parameters_at_limit=}
      """
    ))
  if (
              minuit.valid
      and     minuit.fmin.has_covariance
      and     minuit.fmin.has_accurate_covar
      and     minuit.fmin.has_posdef_covar
      and not minuit.fmin.has_made_posdef_covar
      and not minuit.fmin.hesse_failed
      and not minuit.fmin.is_above_max_edm
      and not minuit.fmin.has_reached_call_limit
      and not minuit.fmin.has_parameters_at_limit
  ):
    return True
  return False


def performFitAttempt(
  nll:          Cost,
  startValues:  npt.NDArray[npt.Shape["nmbMoments"], npt.Float64],
  momentLabels: Sequence[str],
) -> im.Minuit:
  """Performs fit attempt and returns minimizer"""
  minuit = im.Minuit(nll, startValues, name = momentLabels)
  minuit.errordef = im.Minuit.LIKELIHOOD
  minuit.migrad()
  minuit.hesse()
  return minuit


if __name__ == "__main__":
  # set parameters of test case
  nmbPwaMcEventsSig       = 10000   # number of signal "data" events to generate from partial-wave amplitudes
  nmbPwaMcEventsBkg       = 10000   # number of background "data" events to generate from partial-wave amplitudes
  nmbPsMcEvents           = 100000  # number of phase-space events to generate
  # beamPolarization        = None    # unpolarized photon beam
  beamPolarization        = 1.0     # polarization of photon beam
  maxL                    = 4       # maximum L quantum number of moments
  nmbFitAttempts          = 1000    # number of fit attempts with random start values
  nmbParallelFitProcesses = 200     # number of parallel processes to use for fitting
  outputDirName           = Utilities.makeDirPath("./plotsTestFitMoments")
  randomSeed              = 123456789
  # formulas for detection efficiency: x = cos(theta); y = phi in [-180, +180] deg
  # efficiencyFormula = "1"  # perfect acceptance
  efficiencyFormula = "(1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / 1.5**3"  # acceptance even in all variables

  # define angular distribution of signal
  partialWaveAmplitudesSig: tuple[AmplitudeValue, ...] = (  # set of all possible partial waves up to ell = 2
    # # amplitudes for unpolarized photon beam
    # AmplitudeValue(QnWaveIndex(refl = None, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 1, m = -1), val = -0.4 + 0.1j),  # P_-1^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 1, m =  0), val =  0.3 - 0.8j),  # P_0^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 1, m = +1), val = -0.8 + 0.7j),  # P_+1^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m = -2), val =  0.1 - 0.4j),  # D_-2^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m = -1), val =  0.5 + 0.2j),  # D_-1^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m =  0), val = -0.1 - 0.2j),  # D_ 0^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m = +1), val =  0.2 - 0.1j),  # D_+1^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m = +2), val = -0.2 + 0.3j),  # D_+2^-
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
  )
  # define angular distribution of background
  partialWaveAmplitudesBkg: tuple[AmplitudeValue, ...] = (  # set of all possible partial waves up to ell = 2
    # # amplitudes for unpolarized photon beam
    # AmplitudeValue(QnWaveIndex(refl = None, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 1, m = -1), val = -0.9 + 0.7j),  # P_-1^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 1, m =  0), val = -0.6 + 0.4j),  # P_0^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 1, m = +1), val = -0.9 - 0.8j),  # P_+1^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m = -2), val = -1.0 - 0.7j),  # D_-2^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m = -1), val = -0.8 - 0.7j),  # D_-1^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m =  0), val =  0.4 + 0.3j),  # D_ 0^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m = +1), val = -0.6 - 0.1j),  # D_+1^-
    # AmplitudeValue(QnWaveIndex(refl = None, l = 2, m = +2), val = -0.1 - 0.9j),  # D_+2^-
    # negative-reflectivity waves
    AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.9 + 0.7j),  # P_-1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m =  0), val = -0.6 + 0.4j),  # P_0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.9 - 0.8j),  # P_+1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -2), val = -1.0 - 0.7j),  # D_-2^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val = -0.8 - 0.7j),  # D_-1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val =  0.4 + 0.3j),  # D_ 0^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val = -0.6 - 0.1j),  # D_+1^-
    AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.1 - 0.9j),  # D_+2^-
    # positive-reflectivity waves
    AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = -1), val = -1.0 + 0.8j),  # P_-1^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.2 + 0.2j),  # P_0^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.0 - 0.3j),  # P_+1^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.7 + 0.9j),  # D_-2^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -1), val = -0.4 - 0.5j),  # D_-1^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m =  0), val = -0.3 + 0.2j),  # D_ 0^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +1), val = -1.0 - 0.4j),  # D_+1^+
    AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val =  0.5 - 0.2j),  # D_+2^+
  )

  thisSourceFileName = os.path.basename(__file__)
  logFileName = f"{outputDirName}/{os.path.splitext(thisSourceFileName)[0]}.log"
  print(f"Writing output to log file '{logFileName}'")
  with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
  # if True:
    print(f"Using iminuit version {im.__version__}")
    Utilities.printGitInfo()
    timer = Utilities.Timer()
    ROOT.gROOT.SetBatch(True)
    setupPlotStyle()
    threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
    print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
    with threadController.limit(limits = 1):
      print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")
      timer.start("Total execution time")

      print(f"Generating accepted phase-space MC events from {nmbPsMcEvents} phase-space events")
      timer.start("Time to generate accepted phase-space MC data")
      ROOT.gRandom.SetSeed(randomSeed)
      dataAcceptedPs = genAccepted2BodyPs(
        nmbGenEvents      = nmbPsMcEvents,
        efficiencyFormula = efficiencyFormula,
        outFileNamePrefix = f"{outputDirName}/",
        # regenerateData    = True,
        regenerateData    = False,
      )
      nmbAcceptedPsEvents  = dataAcceptedPs.Count().GetValue()
      phaseSpaceEfficiency = nmbAcceptedPsEvents / nmbPsMcEvents
      print(f"The accepted phase-space sample contains {nmbAcceptedPsEvents} accepted events corresponding to an efficiency of {phaseSpaceEfficiency:.3f}")
      timer.stop("Time to generate accepted phase-space MC data")

      print("Calculating true moment values and generating data from partial-wave amplitudes")
      timer.start("Time to generate MC data from partial waves")
      amplitudeSetSig = AmplitudeSet(partialWaveAmplitudesSig)
      amplitudeSetBkg = AmplitudeSet(partialWaveAmplitudesBkg)
      ROOT.gRandom.SetSeed(randomSeed)
      dataPwaModel, dataPwaModelSig, dataPwaModelBkg = genSigAndBkgDataFromWaves(
        nmbEventsSig      = nmbPwaMcEventsSig,
        nmbEventsBkg      = nmbPwaMcEventsBkg,
        amplitudeSetSig   = amplitudeSetSig,
        amplitudeSetBkg   = amplitudeSetBkg,
        polarization      = beamPolarization,
        outputDirName     = outputDirName,
        efficiencyFormula = efficiencyFormula,
        # regenerateData    = True,
        regenerateData    = False,
      )
      # dataPwaModel = dataPwaModel.Filter("eventWeight == 1.0")  # select only signal region
      # dataPwaModel = dataPwaModel.Filter("eventWeight == -0.5")  # select only sideband regions
      timer.stop("Time to generate MC data from partial waves")

      # normalize true moments to acceptance-corrected number of signal and background events, respectively
      HTruthSig: MomentResult = amplitudeSetSig.photoProdMomentSet(maxL, normalize = nmbPwaMcEventsSig / phaseSpaceEfficiency)  #!NOTE! this normalization is slightly off because it does not take into account the events that are cut away by selecting the signal region
      HTruthBkg: MomentResult = amplitudeSetBkg.photoProdMomentSet(maxL, normalize = (1.2 / 2.0) * nmbPwaMcEventsBkg / phaseSpaceEfficiency)  # takes into account the sideband widths
      print(f"True moment values for signal:\n{HTruthSig}")
      print(f"True moment values for background:\n{HTruthBkg}")

      intensityTF3s: dict[str, ROOT.TF3] = {}
      for name, label, HTruth in (("signal", "Sig", HTruthSig), ("background", "Bkg", HTruthBkg)):
        print(f"Drawing true intensity function for {name}")
        # formula uses variables: x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
        intensityFormula = HTruth.intensityFormula(
          polarization     = beamPolarization,
          thetaFormula     = "std::acos(x)",
          phiFormula       = "TMath::DegToRad() * y",
          PhiFormula       = "TMath::DegToRad() * z",
          printFormula     = True,
          useMomentSymbols = True,
        )
        intensityTF3 = ROOT.TF3(f"intensityMoments{label}", intensityFormula, -1, +1, -180, +180, -180, +180)
        for qnIndex in HTruth.indices.qnIndices:
          Hval = HTruth[qnIndex].val
          intensityTF3.SetParameter(qnIndex.label, Hval.imag if qnIndex.momentIndex == 2 else Hval.real)
        intensityTF3.SetNpx(100)
        intensityTF3.SetNpy(100)
        intensityTF3.SetNpz(100)
        intensityTF3.SetMinimum(0)
        drawTF3(intensityTF3, **TH3_ANG_PLOT_KWARGS, pdfFileName = f"{outputDirName}/{intensityTF3.GetName()}.pdf")
        intensityTF3s[label] = intensityTF3

      timer.start("Time to construct model functions")
      print("Constructing vectorized intensity function")
      # # formula uses variables: x = theta in [0, pi] rad; y = phi in [-pi, +pi] rad; z = Phi in [-pi, +pi] rad
      # intensityFormula = HTruth.intensityFormula(
      #   polarization     = beamPolarization,
      #   thetaFormula     = "x",
      #   phiFormula       = "y",
      #   PhiFormula       = "z",
      #   printFormula     = True,
      #   useMomentSymbols = True,
      # )
      # defineIntensityFcnVectorizedCpp(intensityFormula)
      # intensityFcn = intensityFcnVectorized
      # intensityFcn.efficiency = phaseSpaceEfficiency  # needed for integral calculation
      intensityFcn = IntensityFcnVectorized(HTruthSig.indices, beamPolarization)
      intensityFcn.precalcBasisFcnAccPsIntegrals(
        thetas       = dataAcceptedPs.AsNumpy(columns = ["theta", ])["theta"],
        phis         = dataAcceptedPs.AsNumpy(columns = ["phi",   ])["phi"],
        Phis         = dataAcceptedPs.AsNumpy(columns = ["Phi",   ])["Phi"],
        nmbGenEvents = nmbPsMcEvents,  #TODO this works only for perfect acceptance
      )
      momentValuesTruth = np.array([HTruthSig[qnIndex].val.real if qnIndex.momentIndex < 2 else HTruthSig[qnIndex].val.imag for qnIndex in HTruthSig.indices.qnIndices])  # make all moment values real-valued
      momentLabels = tuple(qnIndex.label for qnIndex in HTruthSig.indices.qnIndices)
      thetas = np.array([0,    1,    2],    dtype = np.double)
      phis   = np.array([0.5,  1.5,  2.5],  dtype = np.double)
      Phis   = np.array([0.75, 1.75, 2.75], dtype = np.double)
      intensities = intensityFcn(dataPoints = (thetas, phis, Phis), moments = momentValuesTruth)
      intensitiesTF3 = []
      for theta, phi, Phi in zip(thetas, phis, Phis):
        intensitiesTF3.append(intensityTF3s["Sig"].Eval(np.cos(theta), np.rad2deg(phi), np.rad2deg(Phi)))
      print(f"!!! {intensities=} vs. {intensitiesTF3=}, delta = {np.array(intensitiesTF3) - intensities[1]}")
      timer.stop("Time to construct functions")

      print("Loading data and setting up iminuit's extended unbinned likelihood function and minimizer")
      thetas = dataPwaModel.AsNumpy(columns = ["theta", ])["theta"]
      phis   = dataPwaModel.AsNumpy(columns = ["phi",   ])["phi"]
      Phis   = dataPwaModel.AsNumpy(columns = ["Phi",   ])["Phi"]
      extUnbinnedNllFcn = cost.ExtendedUnbinnedNLL(
        data       = (thetas, phis, Phis),
        scaled_pdf = intensityFcn,
        verbose    = 0,
      )
      # minuit = im.Minuit(extUnbinnedNllFcn, 0.99 * momentValuesTruth, name = momentLabels)
      # print(f"Fitting {len(thetas)} events")
      # with timer.timeThis("Time needed by MIGRAD"):
      #   minuit.migrad()
      # with timer.timeThis("Time needed by HESSE"):
      #   minuit.hesse()
      # # print(minuit)
      # print(minuit.fmin)
      # print(minuit.params)
      # print(minuit.merrors)
      # # with timer.timeThis("Time needed by draw_mnmatrix"):
      # #   figure, axes = minuit.draw_mnmatrix()
      # #   figure.savefig(f"{outputDirName}/minuit_mnmatrix.pdf")

      # print("Plotting fit results")
      # HPhys = convertIminuitToMomentResult(minuit, HTruthSig.indices)
      # plotMomentsInBin(
      #   HData             = HPhys,
      #   normalizedMoments = False,
      #   HTruth            = HTruthSig,
      #   legendLabels      = ("Moment", "Truth"),
      #   outFileNamePrefix = f"{outputDirName}/unnorm_phys_",
      #   plotTruthUncert   = True,
      #   truthColor        = ROOT.kBlue + 1,
      # )

      print("Setting up custom extended unbinned weighted likelihood function and iminuit's minimizer")
      # eventWeights = np.ones_like(thetas)
      eventWeights = dataPwaModel.AsNumpy(columns = ["eventWeight", ])["eventWeight"]
      nll = ExtendedUnbinnedWeightedNLL(
        intensityFcn = intensityFcn,
        thetas       = thetas,
        phis         = phis,
        Phis         = Phis,
        eventWeights = eventWeights,
      )
      print(f"!!! {2 * nll(momentValuesTruth)=} - {extUnbinnedNllFcn(momentValuesTruth)=} = {2 * nll(momentValuesTruth) - extUnbinnedNllFcn(momentValuesTruth)}")
      print(f"Performing {nmbFitAttempts} fir attempts of {len(thetas)} events using custom NLL function and {nmbParallelFitProcesses} processes")
      with timer.timeThis(f"Time needed for performing {nmbFitAttempts} fit attempts running {nmbParallelFitProcesses} fits in parallel"):
        # generate random start values for all attempts
        startValueSets: list[npt.NDArray[npt.Shape["nmbMoments"], npt.Float64]] = []
        np.random.seed(randomSeed)
        nmbEventsSigCorr = np.sum(eventWeights) / phaseSpaceEfficiency  # number of acceptance-corrected signal events
        for _ in range(nmbFitAttempts):
          # startValues    = np.zeros_like(momentValuesTruth)
          startValues    = np.random.normal(loc = 0, scale = 0.02 * nmbEventsSigCorr, size = len(momentValuesTruth))  #!NOTE! convergence rate is very sensitive to scale parameter
          # startValues   += momentValuesTruth  # randomly perturb true moment values
          startValues[0] = nmbEventsSigCorr  # set H_0(0, 0) to number of acceptance-corrected signal events
          startValueSets.append(startValues)
        # perform fit attempts in parallel
        def increaseNiceLevel() -> None:  # need to define separate function to set `initializer` argument of `ProcessPoolExecutor`
          """Increases nice level of process that calls this function"""
          os.nice(19)
        with ProcessPoolExecutor(max_workers = nmbParallelFitProcesses, initializer = increaseNiceLevel) as executor:
          futures = [
            executor.submit(performFitAttempt, nll = nll, startValues = startValueSets[fitAttemptIndex], momentLabels = momentLabels)
            for fitAttemptIndex in range(nmbFitAttempts)
          ]
          minuits = [future.result() for future in futures]
        # filter out successful fits
        minuitsSuccess: list[im.Minuit] = []
        for fitAttemptIndex, minuit in enumerate(minuits):
          print(minuit.fmin)
          print(minuit.params)
          print(minuit.merrors)
          success = fitSuccess(minuit)
          print(f"Fit [{fitAttemptIndex + 1} of {nmbFitAttempts}] was {'' if success else 'NOT '}successful")
          if success:
            minuitsSuccess.append(minuit)

      if len(minuitsSuccess) == 0:
        print("Warning: No fit was successful. Exiting.")
      else:
        print(f"{len(minuitsSuccess)} out of {nmbFitAttempts} fit attempts were successful")
        print(f"!!! {sorted([minuit.fmin.fval for minuit in minuitsSuccess])=}")
        print("Plotting fit results")
        HPhys2 = convertIminuitToMomentResult(minuitsSuccess[0], HTruthSig.indices)
        plotMomentsInBin(
          HData             = HPhys2,
          normalizedMoments = False,
          HTruth            = HTruthSig,
          # HTruth            = HTruthBkg,
          legendLabels      = ("Moment", "Truth"),
          # HTruth            = HPhys,
          # legendLabels      = ("Custom NLL", "iminuit NLL"),
          outFileNamePrefix = f"{outputDirName}/unnorm_phys2_",
          plotTruthUncert   = True,
          truthColor        = ROOT.kBlue + 1,
        )

      timer.stop("Total execution time")
      print(timer.summary)
