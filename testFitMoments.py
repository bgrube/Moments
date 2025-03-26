#!/usr/bin/env python3
"""
Test script that performs closure test for moment analysis that uses
extended maximum likelihood-fitting of the data.
"""


from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
)
import functools
import numpy as np
import nptyping as npt
import os
import threadpoolctl


import ROOT
import iminuit as im
import iminuit.cost as cost
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
from testMomentsPhotoProd import (
  genDataFromWaves,
  TH3_ANG_PLOT_KWARGS,
)
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# TINY_FLOAT = np.finfo(dtype = np.double).tiny
TINY_FLOAT = np.finfo(dtype = float).tiny


def defineIntensityFcnVectorizedCpp(intensityFormula: str) -> None:
  """Defines the vectorized C++ intensity function using TFormula and OpenMP"""
  ROOT.gInterpreter.ProcessLine(f'TFormula intensityFormula = TFormula("intensity", "{intensityFormula}");')
  ROOT.gInterpreter.ProcessLine(
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
    npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
    npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
    npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
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
  # for perfect acceptance H_0(0, 0) is predicted number of measured events
  integral = moments[0] * thetas.shape[0]  # normalize integral such that parameters can be directly compared to true values
  return (integral, intensities)


ROOT.gInterpreter.ProcessLine(
"""
// basis functions for physical moments; Eq. (XXX)
// scalar version that calculates function value for a single event
double
f_base(
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
		throw std::domain_error("f_base() unknown moment index.");
	}
}
"""
)

ROOT.gInterpreter.ProcessLine(
"""
// vector version that calculates basis-function value for each entry
// in the input vectors for a constant polarization value
std::vector<double>
f_base(
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
		fcnValues[i] = f_base(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
	}
	return fcnValues;
}
"""
)


@dataclass
class IntensityFcnVectorized:
  """Functor that calculates intensities from real data and integrals from accepted phase-space MC data"""
  indices:      MomentIndices  # indices that define set of moments to fit
  beamPol:      float  # polarization of photon beam
  # arrays with real-data values for theta, phi, and Phi angles
  thetas:       npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]
  phis:         npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]
  Phis:         npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]
  _baseFcnVals: npt.NDArray[npt.Shape["nmbMoments, nmbEvents"], npt.Float64] = field(init = False)  # precalculated values of basis functions for all events

  def setRealData(
    self,
    thetas: npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
    phis:   npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
    Phis:   npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
  ) -> None:
    """Sets real data member variales and precalulates values and integrals of basis functions"""
    self.thetas = np.ascontiguousarray(thetas).copy()
    self.phis   = np.ascontiguousarray(phis).copy()
    self.Phis   = np.ascontiguousarray(Phis).copy()
    # check that all data arrays have the same length
    nmbEvents  = len(self.thetas)
    assert nmbEvents == len(self.phis) == len(self.Phis), f"Data arrays must have same length; but got {len(self.thetas)=} vs. {len(self.phis)=} vs. {len(self.Phis)=}"
    nmbMoments = len(self.indices)
    # calculate values of basis functions for all events
    self._baseFcnVals = np.zeros((nmbMoments, nmbEvents), dtype = np.double)
    for flatIndex in self.indices.flatIndices:
      qnIndex = self.indices[flatIndex]
      self._baseFcnVals[flatIndex] = np.asarray(ROOT.f_base(
        qnIndex.momentIndex, qnIndex.L, qnIndex.M,
        self.thetas,
        self.phis,
        self.Phis,
        self.beamPol,
      ))

  def __post_init__(self) -> None:
    self.setRealData(self.thetas, self.phis, self.Phis)

  def __call__(
    self,
    dataPoints: tuple[
      npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
      npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
      npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
    ],  #!Note! this is an unused dummy argument kept for interface compatibility; the real data need to be passed already in the constructor or using `self.setRealData()` before calling this function
    #TODO change the logic such that precalculation is performed at first call when new data is passed?
    moments: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64]
  ) -> tuple[float, npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]]:
    """Wrapper function that calculates intensities for each event and normalization integral of intensity function"""
    thetas, phis, Phis = dataPoints
    # ensure that data passed to function are identical to the data used to precalculate the basis-function values
    assert not np.may_share_memory(thetas, self.thetas), f"Argument and cached array for theta must not share memory"
    assert not np.may_share_memory(phis,   self.phis),   f"Argument and cached array for phi must not share memory"
    assert not np.may_share_memory(Phis,   self.Phis),   f"Argument and cached array for Phi must not share memory"
    assert np.array_equal(thetas, self.thetas), f"Argument and cached array for theta must be identical"
    assert np.array_equal(phis,   self.phis),   f"Argument and cached array for phi must be identical"
    assert np.array_equal(Phis,   self.Phis),   f"Argument and cached array for Phi must be identical"
    intensities = np.dot(moments, self._baseFcnVals)  # calculate intensities for all events
    # for perfect acceptance H_0(0, 0) is predicted number of measured events
    integral = moments[0] * thetas.shape[0]  # normalize integral such that parameters can be directly compared to true values
    return (integral, intensities)


def convertIminuitToMomentResult(
  minuit:  im.Minuit,      # iminuit object containing fit result
  indices: MomentIndices,  # moment indices for which the fit was performed
) -> MomentResult:
    """Converts iminuit result into `MomentResult` object"""
    HPhys = MomentResult(indices)
    # construct index ranges for purely real and purely imaginary moments
    reIndexRange = (
      QnMomentIndex(momentIndex = 0, L = 0,            M = 0),
      QnMomentIndex(momentIndex = 1, L = indices.maxL, M = indices.maxL),
    )  # all H_0 and H_1 moments are real-valued
    imIndexRange = (
      QnMomentIndex(momentIndex = 2, L = 1,            M = 1),
      QnMomentIndex(momentIndex = 2, L = indices.maxL, M = indices.maxL)
    )  # all H_2 moments are purely imaginary
    # convert to flat indices
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
    return HPhys


if __name__ == "__main__":
  # set parameters of test case
  nmbPwaMcEvents   = 10000   # number of "data" events to generate from partial-wave amplitudes
  nmbPsMcEvents    = 100000  # number of phase-space events to generate
  beamPolarization = 1.0     # polarization of photon beam
  maxL             = 4       # maximum L quantum number of moments
  outputDirName    = Utilities.makeDirPath("./plotsTestFitMoments")

  thisSourceFileName = os.path.basename(__file__)
  logFileName = f"{outputDirName}/{os.path.splitext(thisSourceFileName)[0]}.log"
  print(f"Writing output to log file '{logFileName}'")
  with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
  # if True:
    print(f"Using iminuit version {im.__version__}")
    Utilities.printGitInfo()
    timer = Utilities.Timer()
    ROOT.gROOT.SetBatch(True)
    ROOT.gRandom.SetSeed(123456789)
    setupPlotStyle()
    threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
    print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
    with threadController.limit(limits = 4):
      print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")
      timer.start("Total execution time")

      # define angular distribution of signal
      partialWaveAmplitudesSig: tuple[AmplitudeValue, ...] = (  # set of all possible partial waves up to ell = 2
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
      amplitudeSetSig = AmplitudeSet(partialWaveAmplitudesSig)

      print("Calculating true moment values and generating data from partial-wave amplitudes")
      HTruth: MomentResult = amplitudeSetSig.photoProdMomentSet(maxL)
      print(f"True moment values\n{HTruth}")
      timer.start("Time to generate MC data from partial waves")
      dataPwaModel = genDataFromWaves(
        nmbEvents         = nmbPwaMcEvents,
        polarization      = beamPolarization,
        amplitudeSet      = amplitudeSetSig,
        efficiencyFormula = None,
        # regenerateData    = True,
        regenerateData    = False,
        outFileNamePrefix = f"{outputDirName}/",
      )
      print("Plotting data generated from partial-wave amplitudes")
      canv = ROOT.TCanvas()
      nmbBins = 25
      hist = dataPwaModel.Histo3D(
        ROOT.RDF.TH3DModel("hData", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
        "cosTheta", "phiDeg", "PhiDeg")
      hist.SetMinimum(0)
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      hist.Draw("BOX2Z")
      canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")
      timer.stop("Time to generate MC data from partial waves")

      timer.start("Time to construct functions")
      print("Constructing intensity function with moments as parameters from formula")
      # formula uses variables: x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
      intensityFormula = HTruth.intensityFormula(
        polarization     = beamPolarization,
        thetaFormula     = "std::acos(x)",
        phiFormula       = "TMath::DegToRad() * y",
        PhiFormula       = "TMath::DegToRad() * z",
        printFormula     = True,
        useMomentSymbols = True,
      )
      intensityTF3 = ROOT.TF3("intensityMoments", intensityFormula, -1, +1, -180, +180, -180, +180)
      for qnIndex in HTruth.indices.qnIndices:
        Hval = HTruth[qnIndex].val
        print(f"!!! {qnIndex.label} = {Hval}")
        intensityTF3.SetParameter(qnIndex.label, Hval.imag if qnIndex.momentIndex == 2 else Hval.real)
      print("Drawing intensity function")
      intensityTF3.SetNpx(100)
      intensityTF3.SetNpy(100)
      intensityTF3.SetNpz(100)
      intensityTF3.SetMinimum(0)
      drawTF3(intensityTF3, **TH3_ANG_PLOT_KWARGS, pdfFileName = f"{outputDirName}/{intensityTF3.GetName()}.pdf")

      print("Constructing vectorized intensity function using TFormula and OpenMP")
      # formula uses variables: x = theta in [0, pi] rad; y = phi in [-pi, +pi] rad; z = Phi in [-pi, +pi] rad
      intensityFormula = HTruth.intensityFormula(
        polarization     = beamPolarization,
        thetaFormula     = "x",
        phiFormula       = "y",
        PhiFormula       = "z",
        printFormula     = True,
        useMomentSymbols = True,
      )
      momentValues = np.array([HTruth[qnIndex].val.imag if qnIndex.momentIndex == 2 else HTruth[qnIndex].val.real for qnIndex in HTruth.indices.qnIndices])
      momentLabels = tuple(qnIndex.label for qnIndex in HTruth.indices.qnIndices)
      thetas = np.array([0,    1,    2],    dtype = np.double)
      phis   = np.array([0.5,  1.5,  2.5],  dtype = np.double)
      Phis   = np.array([0.75, 1.75, 2.75], dtype = np.double)
      # defineIntensityFcnVectorizedCpp(intensityFormula)
      # intensityFcn = intensityFcnVectorized
      intensityFcn = IntensityFcnVectorized(
        indices = HTruth.indices,
        beamPol = beamPolarization,
        thetas  = thetas,
        phis    = phis,
        Phis    = Phis,
      )
      intensitiesFcn = intensityFcn(dataPoints = (thetas, phis, Phis), moments = momentValues)
      intensitiesTF3 = []
      for theta, phi, Phi in zip(thetas, phis, Phis):
        intensitiesTF3.append(intensityTF3.Eval(np.cos(theta), np.rad2deg(phi), np.rad2deg(Phi)))
      print(f"!!! {intensitiesFcn=} vs. {intensitiesTF3=}, delta = {np.array(intensitiesTF3) - intensitiesFcn[1]}")
      timer.stop("Time to construct functions")

      timer.start("Time to load data and construct NLL function")
      print("Loading data and setting up unbinned likelihood function and minimizer")
      thetas = dataPwaModel.AsNumpy(columns = ["theta", ])["theta"]
      phis   = dataPwaModel.AsNumpy(columns = ["phi",   ])["phi"]
      Phis   = dataPwaModel.AsNumpy(columns = ["Phi",   ])["Phi"]
      intensityFcn.setRealData(thetas, phis, Phis)
      extUnbinnedNllFcn = cost.ExtendedUnbinnedNLL(
        data       = (thetas, phis, Phis),
        scaled_pdf = intensityFcn,
        verbose    = 0,
      )
      minuit = im.Minuit(extUnbinnedNllFcn, 1.01 * momentValues, name = momentLabels)
      timer.stop("Time to load data and construct NLL function")

      print(f"Fitting {len(thetas)} events")
      with timer.timeThis("Time needed by MIGRAD"):
        minuit.migrad()
      with timer.timeThis("Time needed by HESSE"):
        minuit.hesse()
      # print(minuit)
      print(minuit.fmin)
      print(minuit.params)
      print(minuit.merrors)
      # with timer.timeThis("Time needed by draw_mnmatrix"):
      #   figure, axes = minuit.draw_mnmatrix()
      #   figure.savefig(f"{outputDirName}/minuit_mnmatrix.pdf")

      print("Plotting fit results")
      HPhys = convertIminuitToMomentResult(minuit, HTruth.indices)
      plotMomentsInBin(
        HData             = HPhys,
        normalizedMoments = False,
        HTruth            = HTruth,
        outFileNamePrefix = f"{outputDirName}/unnorm_phys_",
        legendLabels      = ("Moment", "Truth"),
        plotTruthUncert   = True,
        truthColor        = ROOT.kBlue + 1,
      )

      if False:
        # perform same fit using own function for the negative log-likelihood (NLL)
        def nll(moments: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64]) -> float:
          """Negative log-likelihood function for intensity as a function of moment parameters"""
          integral, intensities = intensityFcnVectorized(dataPoints = (thetas, phis, Phis), moments = moments)
          nonPositiveIntensities = intensities[intensities <= 0]
          if nonPositiveIntensities.size > 0:
            print(f"!!! non-positive intensities: {nonPositiveIntensities}")
          return -(np.sum(np.log(intensities)) - integral)
          #TODO why the options below lead to "INVALID Minimum", "ABOVE EDM threshold"? the same calculations are performed by iminuit's ExtendedUnbinnedNLL
          # return -(np.sum(np.sort(np.log(intensities))) - integral)  # sort logs of intensities to make sum more accurate
          # return -(np.sum(np.sort(np.log(intensities + TINY_FLOAT))) - integral)  # sort logs of intensities to make sum more accurate and protect against 0 intensities
        print(f"!!! {2 * nll(momentValues)=} - {extUnbinnedNllFcn(momentValues)=} = {2 * nll(momentValues) - extUnbinnedNllFcn(momentValues)}")
        print(f"Fitting {len(thetas)} events using custom NLL function")
        minuit2 = im.Minuit(nll, 1.01 * momentValues, name = momentLabels)
        minuit2.errordef = im.Minuit.LIKELIHOOD
        with timer.timeThis("Time needed by MIGRAD2"):
          minuit2.migrad()
        with timer.timeThis("Time needed by HESSE2"):
          minuit2.hesse()
        # print(minuit2)
        print(minuit2.fmin)
        print(minuit2.params)
        print(minuit2.merrors)

        print("Plotting fit results")
        HPhys2 = convertIminuitToMomentResult(minuit2, HTruth.indices)
        plotMomentsInBin(
          HData             = HPhys2,
          normalizedMoments = False,
          HTruth            = HPhys,
          outFileNamePrefix = f"{outputDirName}/unnorm_phys2_",
          legendLabels      = ("iminuit NLL", "Custom NLL"),
          plotTruthUncert   = True,
          truthColor        = ROOT.kBlue + 1,
        )

      timer.stop("Total execution time")
      print(timer.summary)
