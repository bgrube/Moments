#!/usr/bin/env python3
"""
This module plots the intensity distributions that correspond to the
moments estimated from data. The moment values are read from files
produced by the function defined in `calculateMoments.py` that
calculates the moments.

Usage: Run this module as a script to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
import ctypes
import functools
import numpy as np
import os
from scipy.optimize import minimize

import ROOT
ROOT.PyConfig.DisableRootLogon = True  # prevent loading of `~/.rootlogon.C`
from wurlitzer import pipes, STDOUT

from moments.MomentCalculator import (
  MomentResult,
  MomentResultsKinematicBinning,
  QnMomentIndex,
)
from workflow.AnalysisConfig import (
  AnalysisConfig,
  BeamPolInfo,
  BEAM_POL_INFOS,
  CFG_POLARIZED_ETAPI0,
  CFG_POLARIZED_PIPI,
)
from workflow.PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
  setupPlotStyle,
  TF3toTH3,
)
from workflow import RootUtilities
from workflow import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


class IntensityFunctor:
  """Functor that calculates the intensity function for physical parts of moments"""

  def __init__(
    self,
    momentResults: MomentResult,
    beamPol:       float = 0.0,
    onlyNegValues: bool  = False,  # if True, return only negative values of intensity function
    invertSign:    bool  = False,  # if True, invert sign of intensity function
  ) -> None:
    self.momentResults = momentResults
    self.nmbMoments    = len(self.momentResults)
    self.beamPol       = beamPol
    self.onlyNegValues = onlyNegValues
    self.invertSign    = invertSign
    self.baseFcnValues = np.zeros((self.nmbMoments, ), dtype = np.float64)  # array for basis function values
    # get moment values as flat, real-valued array
    # construct quantum-number index ranges that correspond to purely real and purely imaginary moments, respectively
    indices = self.momentResults.indices
    maxL = indices.maxL
    reIndexRange = (
      QnMomentIndex(momentIndex = 0, L = 0,    M = 0),
      QnMomentIndex(momentIndex = 1, L = maxL, M = maxL),
    )  # all H_0 and H_1 moments are real-valued
    imIndexRange = (
      QnMomentIndex(momentIndex = 2, L = 1,    M = 1),
      QnMomentIndex(momentIndex = 2, L = maxL, M = maxL)
    )  # all H_2 moments are purely imaginary; all H_2(L, 0) are 0
    # convert to flat-index ranges
    self.reSlice = slice(indices[reIndexRange[0]], indices[reIndexRange[1]] + 1)
    self.imSlice = slice(indices[imIndexRange[0]], indices[imIndexRange[1]] + 1)
    # copy moment values
    self.momentValues = np.zeros((self.nmbMoments, ), dtype = np.float64)
    self.momentValues[self.reSlice] = np.real(self.momentResults._valsFlatIndex[self.reSlice])
    self.momentValues[self.imSlice] = np.imag(self.momentResults._valsFlatIndex[self.imSlice])

  def calcBaseFcnValues(
    self,
    args: np.ndarray,  # 3 arguments: <cos(theta)>, <phi [deg]>, <Phi [deg]>
  ) -> None:
    """Calculates basis function values for all moments"""
    indices = self.momentResults.indices
    cosTheta = args[0]
    # convert azimuthal angles from degrees to radians
    phi      = args[1] * ROOT.TMath.DegToRad()
    Phi      = args[2] * ROOT.TMath.DegToRad()
    for flatIndex in indices.flatIndices:
      qnIndex = indices[flatIndex]
      self.baseFcnValues[flatIndex] = ROOT.f_basis(
        qnIndex.momentIndex, qnIndex.L, qnIndex.M,
        np.arccos(cosTheta),
        phi,
        Phi,
        self.beamPol,
    )

  def __call__(
    self,
    args: np.ndarray,  # 3 arguments: <cos(theta)>, <phi [deg]>, <Phi [deg]>
    _:    np.ndarray,  # unused argument; required by ROOT
  ) -> float:
    """Calculates intensity function"""
    self.calcBaseFcnValues(args)  # calculate basis functions for all moments
    # calculate intensity
    intensity = float(self.momentValues @ self.baseFcnValues)
    if self.onlyNegValues and intensity > 0:
      return 0.0
    return -intensity if self.invertSign else intensity


class IntensityUncertFunctor:
  """Functor that calculates the uncertainty of the intensity function value for physical parts of moments"""

  def __init__(
    self,
    momentResults: MomentResult,
    beamPol:       float = 0.0,
    onlyNegValues: bool  = False,  # if True, return only negative values of intensity function
    invertSign:    bool  = False,  # if True, invert sign of significance function
  ) -> None:
    self.intensityFunctor = IntensityFunctor(
      momentResults = momentResults,
      beamPol       = beamPol,
      onlyNegValues = onlyNegValues,
      invertSign    = invertSign,
    )
    momentResults = self.intensityFunctor.momentResults
    nmbMoments    = self.intensityFunctor.nmbMoments
    # copy covariance matrix
    self.covMatrix = np.zeros((nmbMoments, nmbMoments), dtype = np.float64)
    reSlice = self.intensityFunctor.reSlice
    imSlice = self.intensityFunctor.imSlice
    self.covMatrix[reSlice, reSlice] = momentResults._V_ReReFlatIndex[reSlice, reSlice]
    self.covMatrix[imSlice, imSlice] = momentResults._V_ImImFlatIndex[imSlice, imSlice]
    self.covMatrix[reSlice, imSlice] = momentResults._V_ReImFlatIndex[reSlice, imSlice]
    self.covMatrix[imSlice, reSlice] = momentResults._V_ReImFlatIndex[reSlice, imSlice].T

  def __call__(
    self,
    args: np.ndarray,  # 3 arguments: <cos(theta)>, <phi [deg]>, <Phi [deg]>
    _:    np.ndarray,  # unused argument required by ROOT
  ) -> float:
    """Calculates uncertainty function"""
    # calculate intensity value
    self.intensityFunctor.calcBaseFcnValues(args)
    baseFcnValues = self.intensityFunctor.baseFcnValues
    # calculate standard deviation of intensity function
    standardDev = float(np.sqrt(baseFcnValues @ self.covMatrix @ baseFcnValues))  # since baseFcnValues has shape (nmbMoments, ) it does not need to be transposed
    return standardDev


class IntensitySignificanceFunctor(IntensityUncertFunctor):
  """Functor that calculates the significance of the deviation of the intensity function from 0 for physical parts of moments"""

  def __call__(
    self,
    args: np.ndarray,  # 3 arguments: <cos(theta)>, <phi [deg]>, <Phi [deg]>
    _:    np.ndarray,  # unused argument required by ROOT
  ) -> float:
    """Calculates significance function"""
    # calculate intensity value
    intensity = self.intensityFunctor(args, _)
    baseFcnValues = self.intensityFunctor.baseFcnValues
    # calculate standard deviation of intensity function
    standardDev = float(np.sqrt(baseFcnValues @ self.covMatrix @ baseFcnValues))  # since baseFcnValues has shape (nmbMoments, ) it does not need to be transposed
    significance = intensity / standardDev if standardDev > 0 else 0.0
    return significance


class IntensityIntegralFunctor:
  """Functor that calculates the integral of the intensity function for physical parts of moments"""

  def __init__(
    self,
    intensityFunctor: IntensityFunctor,
    nmbBinsPerAxis:   int = 25,  # number of bins per axis for numerical integration
  ) -> None:
    self.intensityFunctor = intensityFunctor
    self.intensityFcn     = ROOT.TF3(f"intensityFcn", self.intensityFunctor, -1, +1, -180, +180, -180, +180)
    self.nmbBinsPerAxis   = nmbBinsPerAxis

  def __call__(
    self,
    momentValues: np.ndarray,  # flat array of moment values
  ) -> float:
    """Calculates integral of intensity function"""
    self.intensityFunctor.momentValues = momentValues  # set moment values used by intensity function
    # integral = float(self.intensityFcn.Integral(-1, +1, -180, +180, -180, +180))  # does not work
    # The adaptive quadrature integration used by `TF3.Integral()`
    # does not work, because it does not find the regions with
    # negative intensity and hence returns zero. This is also true
    # when using `TF1.IntegralMultiple()` with increased number of
    # maximum function evaluations. Using
    # `ROOT.Math.AdaptiveIntegratorMultiDim` directly and increasing
    # the minimum number of points using `SetMinPts()` yields a
    # non-zero integral but the algorithm fails to converge for
    # reasonable values of the maximum number of points and the
    # computed value is too small indicating the algorithm does not
    # find all negative regions.

    # timer = Utilities.Timer()

    # # use Monte Carlo integration
    # # for comparable run times they don't seem to give much better results than simple grid integration
    # ROOT.Math.IntegratorMultiDimOptions.SetDefaultIntegrator("VEGAS")  # default is "ADAPTIVE"
    # # ROOT.Math.IntegratorMultiDimOptions.SetDefaultIntegrator("MISER")  # default is "ADAPTIVE"
    # maxpts = int(1e4)  # maximum number of function evaluations
    # epsrel = 1e-6      # relative target accuracy
    # epsabs = 1e-6      # absolute target accuracy
    # relerr = ctypes.c_double(0.0)  # estimate of relative accuracy of integral
    # nfnevl = ctypes.c_int(0)       # number of function evaluations
    # ifail  = ctypes.c_int(0)       # error flag
    # timer.start("VEGAS")
    # integral = float(self.intensityFcn.IntegralMultiple(3, np.array([-1, -180, -180], dtype = np.float64), np.array([+1, +180, +180], dtype = np.float64), maxpts, epsrel, epsabs, relerr, nfnevl, ifail))
    # print(f"!!! VEGAS {integral=}, {nfnevl=}, {relerr=}, {ifail=}")
    # timer.stop("VEGAS")
    # ROOT.Math.IntegratorMultiDimOptions.SetDefaultIntegrator("MISER")  # default is "ADAPTIVE"
    # timer.start("MISER")
    # integral = float(self.intensityFcn.IntegralMultiple(3, np.array([-1, -180, -180], dtype = np.float64), np.array([+1, +180, +180], dtype = np.float64), maxpts, epsrel, epsabs, relerr, nfnevl, ifail))
    # print(f"!!! MISER {integral=}, {nfnevl=}, {relerr=}, {ifail=}")
    # timer.stop("MISER")

    # use simple grid integration
    # timer.start("HIST")
    intensityFcnHist = TF3toTH3(
      fcn      = self.intensityFcn,
      binnings = (
        HistAxisBinning(self.nmbBinsPerAxis,   -1,   +1),  # cos(theta)
        HistAxisBinning(self.nmbBinsPerAxis, -180, +180),  # phi
        HistAxisBinning(self.nmbBinsPerAxis, -180, +180),  # Phi
      ),
      histName = self.intensityFcn.GetName(),
    )
    integral = float(intensityFcnHist.Integral("width"))
    # print(f"!!! {integral=}")
    # timer.stop("HIST")
    # print(timer.summary)
    return integral
    #TODO try vegas package
    # benchmarks:
    # 1e4:  VEGAS integral=-3992222.0601890543, nfnevl=c_int(0), relerr=c_double(0.004470002848448925),   ifail=c_int(0); wall time = 5.584 sec, CPU time = 5.561 sec
    # 1e4:  MISER integral=-3714687.1934114816, nfnevl=c_int(0), relerr=c_double(0.05762639926726086),    ifail=c_int(0); wall time = 1.146 sec, CPU time = 1.143 sec
    #  25 points: integral=-4114769.2088862327; wall time = 1.835 sec, CPU time = 1.831 sec
    # 1e5:  VEGAS integral=-3961930.3445391045, nfnevl=c_int(0), relerr=c_double(0.0006438089454992076),  ifail=c_int(0); wall time = 52.97 sec, CPU time = 52.79 sec
    # 1e5:  MISER integral=-3928096.1729773143, nfnevl=c_int(0), relerr=c_double(0.015646596362202617),   ifail=c_int(0); wall time = 11.45 sec, CPU time = 11.42 sec
    #  50 points: integral=-3999756.605608042;  wall time = 15.12 sec, CPU time = 15.08 sec
    # 1e6:  VEGAS integral=-3969148.325157645,  nfnevl=c_int(0), relerr=c_double(0.00016875431791790187), ifail=c_int(0); wall time = 587.3 sec, CPU time = 585.7 sec
    # 1e6:  MISER integral=-3979578.4439491937, nfnevl=c_int(0), relerr=c_double(0.004340729467225029),   ifail=c_int(0); wall time = 116.8 sec, CPU time = 116.5 sec
    # 100 points: integral=-3969229.6918768394; wall time = 116.8 sec, CPU time = 116.4 sec
    # 1e7:  VEGAS integral=-3968043.144606392,  nfnevl=c_int(0), relerr=c_double(2.1594798389239337e-05), ifail=c_int(0); wall time = 3853 sec,  CPU time = 3843 sec
    # 1e7:  MISER integral=-3965868.0687554656, nfnevl=c_int(0), relerr=c_double(0.0014723495766031775),  ifail=c_int(0); wall time = 1151 sec,  CPU time = 1148 sec


def makeIntensityPositiveDefinite(
  momentResults:    MomentResult,
  beamPol:          float = 0.0,
  # relMargin:        float = -1e-4,  # allow for small relative negative values of intensity integral to keep integral in a region, where its derivatives are still defined
  relMargin:        float = 0.0,
  # L_max = 4
  relTolerance:     float = 5e-5,  # relative tolerance for violation of constraint that integral of negative part of intensity function is 0
  maxNmbIterations: int   = 2000,  # maximum number of iterations for minimization
  # # L_max = 6
  # relTolerance:     float = 5e-4,
  # maxNmbIterations: int   = 7000,
  # # L_max = 8
  # relTolerance:     float = 5e-3,
  # maxNmbIterations: int   = 20000,
) -> tuple[MomentResult, float]:
  """Performs minimal shift of moment values to make intensity function positive definite and returns shifted moments and the chi^2 of the shift"""
  print(f"Making intensity function positive definite by shifting moment values")
  negIntensitySignificanceFunctor = IntensitySignificanceFunctor(  # significance of negative part of intensity function
    momentResults = momentResults,
    beamPol       = beamPol,
    onlyNegValues = True,
  )
  negIntensityFunctor = negIntensitySignificanceFunctor.intensityFunctor  # negative part of intensity function
  H: np.ndarray = negIntensityFunctor.momentValues  # nominal moment values
  V: np.ndarray = negIntensitySignificanceFunctor.covMatrix  # covariance matrix of moment values
  # determine minimal shift delta of moments values H such that
  # intensity function is positive definite, i.e.
  # min_delta[delta^T V^-1 delta] such that g(H + delta) >= 0
  # Cholesky-decompose the covariance matrix, i.e. V = L L^T
  try:
    L = np.linalg.cholesky(V)
  except np.linalg.LinAlgError:
    print("Warning: Cholesky decomposition of covariance failed, adding small diagonal term")
    L = np.linalg.cholesky(V + 1e-12 * np.eye(len(V)))
  # perform minimization in whitened space, i.e. define new parameters
  # H_w = L^-1 H that have unit covariance matrix, i.e. V(H_w) = I
  # therefore, the whitened parameter difference is delta_w = L^-1 delta
  # and minimizing the objective function delta^T V^-1 delta is
  # equivalent to minimizing delta_w^T delta_w, i.e. the Euclidean norm of delta_w
  negIntensityIntegralFcn = IntensityIntegralFunctor(negIntensityFunctor)  # integral of negative part of intensity function
  # Unfortunately, at a value of g(H + delta) 0, which is the goal for
  # the minimizer, the negative part of the intensity function is not
  # differentiable with respect to the moment values, so the
  # minimization does not converge properly. Still, the minimization
  # result leads to intensity functions with much smaller negative
  # parts than the original moment values, so it is still useful.
  integral = negIntensityIntegralFcn(H)
  margin = relMargin * abs(integral)  # absolute margin for constraint g(H + delta) >= 0
  catol  = relTolerance * abs(integral)  # absolute tolerance for violation of constraint g(H + delta) >= 0
  print(f"Running minimizer with absolute constraint margin = {margin}, absolute tolerance = {catol}, and a maximum of {maxNmbIterations} iterations")
  result = minimize(
    fun         = lambda delta_w: float(delta_w @ delta_w),  # objective function to be minimized is Euclidean norm in whitened space
    x0          = np.zeros(len(H)),  # start values for delta_w
    method      = 'COBYLA',          # use the Constrained Optimization BY Linear Approximation (COBYLA) algorithm
    options     = {  # options for 'COBYLA' method
      "rhobeg"  : 0.1,               # reasonable initial changes to delta_w
      "maxiter" : maxNmbIterations,  # maximum number of function evaluations
      "catol"   : catol,             # absolute tolerance for violation of constraint g(H + delta) >= 0
      "disp"    : True,              # display convergence messages
    },
    constraints = [{  # constraints for minimization
      "fun"  : lambda delta_w: negIntensityIntegralFcn(H + L @ delta_w) - margin,  # function g(H + delta) defining the constraint with delta = L @ delta_w; subtract margin to allow for small deviations from g = 0
      "type" : "ineq",                                                             # inequality constraint, i.e. g(H + delta) >= 0
    }],
  )
  # get result from minimization and transform back from whitened to original space
  print(f"Minimization finished with result:\n{result}")
  delta_w = result.x
  delta = L @ delta_w
  print(f"Absolute shifts of moment values:\n{delta}")
  uncertainties = np.sqrt(np.diag(V))
  deltaSignificances = delta / uncertainties
  print(f"Significances of shifts:\n{deltaSignificances}")
  H_shifted = H + delta
  print(f"Shifted moment values:\n{H_shifted}")
  print(f"Integral of negative intensity for original moment values = {integral}")
  print(f"Integral of negative intensity for shifted  moment values = {negIntensityIntegralFcn(H_shifted)}; ratio = {negIntensityIntegralFcn(H_shifted) / integral if integral != 0 else float('nan')}")
  # construct new MomentResult object with shifted moment values
  momentResultsShifted = deepcopy(momentResults)
  reSlice = negIntensityFunctor.reSlice
  imSlice = negIntensityFunctor.imSlice
  momentResultsShifted._valsFlatIndex[reSlice] = H_shifted[reSlice]
  momentResultsShifted._valsFlatIndex[imSlice] = H_shifted[imSlice] * 1j  # convert to purely imaginary
  return momentResultsShifted, result.fun


def plotIntensityFcn(
  momentResults:            MomentResult,
  massBinIndex:             int,
  beamPolInfo:              BeamPolInfo | None,
  outputDirPath:            str,
  nmbBinsPerAxis:           int                             = 25,
  useIntensityTerms:        MomentResult.IntensityTermsType = MomentResult.IntensityTermsType.ALL,
  coordSysLabel:            str                             = "HF",
  makeIntensityPosDefinite: bool                            = False,  # if True, shift moment values such that intensity function is positive definite
  scaleFactor:              float | None                    = None,   # scale intensity function with this factor
) -> MomentResult | None:  # return moments shifted such that intensity function is positive definite
  """Draw intensity function in given mass bin and save PDF to output directory"""
  print(f"Plotting intensity function for mass bin {massBinIndex} using {beamPolInfo} and intensity terms {useIntensityTerms.value}")
  if scaleFactor is not None:
    print(f"Scaling moments by factor {scaleFactor}")
    momentResults = deepcopy(momentResults)
    momentResults.scaleBy(scaleFactor)
  # draw intensity function as 3D plot
  # formula uses variables: x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
  intensityFormula = momentResults.intensityFormula(
    polarization      = beamPolInfo.pol if beamPolInfo is not None else None,
    thetaFormula      = "std::acos(x)",
    phiFormula        = "TMath::DegToRad() * y",
    PhiFormula        = "TMath::DegToRad() * z",
    useIntensityTerms = useIntensityTerms,
  )
  ROOT.gStyle.SetImageScaling(3)  # improve bitmap rendering quality by tripling the resolution; default is 1
  intensityFcn = ROOT.TF3(f"intensityFcn_{useIntensityTerms.value}_bin_{massBinIndex}", intensityFormula, -1, +1, -180, +180, -180, +180)
  binnings = (
    HistAxisBinning(nmbBinsPerAxis,   -1,   +1),  # cos(theta)
    HistAxisBinning(nmbBinsPerAxis, -180, +180),  # phi
    HistAxisBinning(nmbBinsPerAxis, -180, +180),  # Phi
  )
  histFcn, minVal, maxVal = drawTF3(
    fcn                = intensityFcn,
    binnings           = binnings,
    outFilePath        = f"{outputDirPath}/{intensityFcn.GetName()}.png",
    histTitle          = f"Intensity Function;cos#theta_{{{coordSysLabel}}};#phi_{{{coordSysLabel}}} [deg];#Phi [deg]",
    showNegativeValues = True,
  )
  if minVal < 0:
    print(f"WARNING: Intensity function for mass bin {massBinIndex} has negative values: minimum = {minVal}, maximum = {maxVal}")
  # draw negative part of intensity function (if any)
  intensityFormulaNeg = f"-({intensityFormula})"
  intensityFcnNeg = ROOT.TF3(f"{intensityFcn.GetName()}_neg", intensityFormulaNeg, -1, +1, -180, +180, -180, +180)
  histFcnNeg, _, _ = drawTF3(
    fcn                = intensityFcnNeg,
    binnings           = binnings,
    outFilePath        = f"{outputDirPath}/{intensityFcnNeg.GetName()}.png",
    histTitle          = f"Intensity Function, Negative Part;cos#theta_{{{coordSysLabel}}};#phi_{{{coordSysLabel}}} [deg];#Phi [deg]",
    showNegativeValues = False,
  )
  # draw statistical uncertainty of intensity function
  beamPol = beamPolInfo.pol if beamPolInfo is not None else 0.0
  intensityUncertFunctor = IntensityUncertFunctor(
    momentResults = momentResults,
    beamPol       = beamPol,
  )
  intensityFcnUncert = ROOT.TF3(f"intensityFcnUncert_{useIntensityTerms.value}_bin_{massBinIndex}", intensityUncertFunctor, -1, +1, -180, +180, -180, +180)
  drawTF3(
    fcn         = intensityFcnUncert,
    binnings    = binnings,
    outFilePath = f"{outputDirPath}/{intensityFcnUncert.GetName()}.png",
    histTitle   = f"Intensity Uncertainty;cos#theta_{{{coordSysLabel}}};#phi_{{{coordSysLabel}}} [deg];#Phi [deg]",
  )
  # draw statistical significance of negative part of intensity function (if any)
  intensitySignificanceFunctor = IntensitySignificanceFunctor(
    momentResults = momentResults,
    beamPol       = beamPol,
    onlyNegValues = True,  # only show negative part of intensity function
    invertSign    = True,  # invert sign of significance function to make negative part of intensity function positive
  )
  intensityFcnSignificance = ROOT.TF3(f"intensityFcnSignificance_{useIntensityTerms.value}_bin_{massBinIndex}_neg", intensitySignificanceFunctor, -1, +1, -180, +180, -180, +180)
  drawTF3(
    fcn         = intensityFcnSignificance,
    binnings    = binnings,
    outFilePath = f"{outputDirPath}/{intensityFcnSignificance.GetName()}.png",
    histTitle   = f"Intensity Significance;cos#theta_{{{coordSysLabel}}};#phi_{{{coordSysLabel}}} [deg];#Phi [deg]",
  )
  momentsShifted = None
  if makeIntensityPosDefinite and useIntensityTerms == MomentResult.IntensityTermsType.PARITY_CONSERVING:
    #TODO this code works only for parity-conserving moments
    # make intensity function positive definite by shifting moment values and draw negative part to confirm
    momentsShifted, chi2 = makeIntensityPositiveDefinite(momentResults, beamPol = beamPol)
    # plot intensity function for shifted moment values
    intensityFunctorShifted = IntensityFunctor(
      momentResults = momentsShifted,
      beamPol       = beamPol,
    )
    intensityFcnShifted = ROOT.TF3(f"intensityFcn_shifted_{useIntensityTerms.value}_bin_{massBinIndex}", intensityFunctorShifted, -1, +1, -180, +180, -180, +180)
    drawTF3(
      fcn         = intensityFcnShifted,
      binnings    = binnings,
      outFilePath = f"{outputDirPath}/{intensityFcnShifted.GetName()}.png",
      histTitle   = f"Intensity, Shifted #chi^{{2}} = {chi2:.2g};cos#theta_{{{coordSysLabel}}};#phi_{{{coordSysLabel}}} [deg];#Phi [deg]",
    )
    # plot negative part of intensity function for shifted moment values
    intensityFunctorShifted.onlyNegValues = True  # only show negative part of intensity function
    intensityFunctorShifted.invertSign    = True  # invert sign of significance function to make negative part of intensity function positive
    intensityFcnShiftedNeg = ROOT.TF3(f"intensityFcn_shifted_{useIntensityTerms.value}_bin_{massBinIndex}_neg", intensityFunctorShifted, -1, +1, -180, +180, -180, +180)
    drawTF3(
      fcn         = intensityFcnShiftedNeg,
      binnings    = binnings,
      outFilePath = f"{outputDirPath}/{intensityFcnShiftedNeg.GetName()}.png",
      histTitle   = f"Intensity, Negative Part, Shifted #chi^{{2}} = {chi2:.2g};cos#theta_{{{coordSysLabel}}};#phi_{{{coordSysLabel}}} [deg];#Phi [deg]",
    )
  # ROOT.gStyle.SetCanvasDefH(600)  # revert back to default resolution
  # ROOT.gStyle.SetCanvasDefW(600)
  # draw projections of intensity function onto (cos(theta), phi) plane
  histProj = histFcn.Project3D("yx")  #!NOTE! "yx" gives y = phi vs. x = cos(theta)
  canv = ROOT.TCanvas()
  ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # draw 2D plot with pos/neg color palette and symmetric z axis
  histProj.SetTitle(f"Intensity Function Projection;{histFcn.GetXaxis().GetTitle()};{histFcn.GetYaxis().GetTitle()}")
  # zRange = abs(histProj.GetMinimum()) if histProj.GetMinimum() < 0 else 10.0  # choose z range to see negative values; but avoid zero range in case function positive
  # histProj.SetMinimum(-zRange)
  # histProj.SetMaximum(+zRange)
  histProj.Draw("COLZ")
  canv.SaveAs(f"{outputDirPath}/{histProj.GetName()}.pdf")
  ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette
  histProjNeg = histFcnNeg.Project3D("yx")  #!NOTE! "yx" gives y = phi vs. x = cos(theta)
  canv = ROOT.TCanvas()
  histProjNeg.SetTitle(f"Intensity Function Projection, Negative Part;{histFcnNeg.GetXaxis().GetTitle()};{histFcnNeg.GetYaxis().GetTitle()}")
  histProjNeg.SetMinimum(0)
  histProjNeg.Draw("COLZ")
  canv.SaveAs(f"{outputDirPath}/{histProjNeg.GetName()}.pdf")
  if False:
    # draw intensity as function of phi and Phi for fixed cos(theta) value
    cosTheta = 0.0  # fixed value of cos(theta)
    # formula uses variables: x = phi in [-180, +180] deg; y = Phi in [-180, +180] deg
    intensityFormulaFixedCosTheta = momentResults.intensityFormula(
      polarization      = beamPolInfo.pol,
      thetaFormula      = f"std::acos({cosTheta})",
      phiFormula        = "TMath::DegToRad() * x",
      PhiFormula        = "TMath::DegToRad() * y",
      useIntensityTerms = useIntensityTerms,
    )
    intensityFcnFixedCosTheta = ROOT.TF2(f"intensityFcn_fixedCosTheta_{useIntensityTerms.value}_bin_{massBinIndex}", intensityFormulaFixedCosTheta, -180, +180, -180, +180)
    intensityFcnFixedCosTheta.SetTitle(f"Intensity Function for cos#theta_{{{coordSysLabel}}} = {cosTheta};#phi_{{{coordSysLabel}}} [deg];#Phi [deg]")
    intensityFcnFixedCosTheta.SetNpx(100)
    intensityFcnFixedCosTheta.SetNpy(100)
    intensityFcnFixedCosTheta.SetMinimum(0)
    canv = ROOT.TCanvas()
    intensityFcnFixedCosTheta.Draw("COLZ")
    canv.SaveAs(f"{outputDirPath}/{intensityFcnFixedCosTheta.GetName()}.pdf")
  return momentsShifted


def plotIntensityFunctions(
  cfg:                      AnalysisConfig,
  momentType:               str,                        # type of moments to read from file: "phys" or "meas"
  makeIntensityPosDefinite: bool,                       # if True, perform minimal shift of moment values to make the intensity functions positive definite
  overrideBeamPolInfo:      BeamPolInfo | None = None,  # if provided, this beam polarization info will override the one in BEAM_POL_INFOS
  scaleFactor:              float | None       = None,  # scale moments by this factor before plotting intensity functions
) -> None:
  """Plots the intensity distributions that correspond to the moments
  estimated from data. The moment values are read from files produced
  by the function defined in `calculateMoments.py` that calculates the
  moments."""
  print(f"Plotting intensity functions for subsystem '{cfg.subsystem}':")
  for dataPeriod in cfg.dataPeriods:
    for tBinLabel in cfg.tBinLabels:
      for beamPolLabel in cfg.beamPolLabels:
        for maxL in cfg.maxLs:
          fitResultDirPath = cfg.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel, maxL)
          thisSourceFileName = os.path.basename(__file__)
          logFilePath = f"{fitResultDirPath}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
          print(f"Writing output to log file '{logFilePath}'")
          with open(logFilePath, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
            Utilities.printGitInfo()
            timer = Utilities.Timer()
            timer.start("Total execution time")
            print(f"Plotting intensity functions for data period '{dataPeriod}', t bin '{tBinLabel}', beam-polarization orientation '{beamPolLabel}', and L_max = {maxL}")
            momentResultsFilePath = f"{fitResultDirPath}/{cfg.outFileNamePrefix}_moments_{momentType}.pkl"
            print(f"Reading moments from file '{momentResultsFilePath}'")
            momentResults = MomentResultsKinematicBinning.loadPickle(momentResultsFilePath)
            for useIntensityTerms in (
              # MomentResult.IntensityTermsType.ALL,
              MomentResult.IntensityTermsType.PARITY_CONSERVING,
              # MomentResult.IntensityTermsType.PARITY_VIOLATING,
            ):
              momentsShifted = []
              for massBinIndex, momentResultsForBin in enumerate(momentResults):
                print(f"Plotting intensity function for {momentResultsForBin.binCenters=}")
                momentsShifted.append(
                  plotIntensityFcn(
                    momentResults            = momentResultsForBin,
                    massBinIndex             = massBinIndex,
                    beamPolInfo              = overrideBeamPolInfo if overrideBeamPolInfo is not None else BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel],
                    outputDirPath            = fitResultDirPath,
                    nmbBinsPerAxis           = 50,
                    useIntensityTerms        = useIntensityTerms,
                    coordSysLabel            = cfg.frame.name,
                    makeIntensityPosDefinite = makeIntensityPosDefinite,
                    scaleFactor              = scaleFactor,
                  )
                )
              # save shifted moments to file
              if momentsShifted and all(m is not None for m in momentsShifted):
                momentResultsShifted = MomentResultsKinematicBinning(momentsShifted)
                momentResultsShifted.savePickle(momentResultsFilePath.replace(".pkl", f"_shifted.pkl"))
            timer.stop("Total execution time")
            print(timer.summary)


if __name__ == "__main__":
  Utilities.printGitInfo()
  ROOT.gROOT.SetBatch(True)
  RootUtilities.loadBasisFunctionsLibrary()  # initializes OpenMP and loads `cpp/basisFunctions.C`
  setupPlotStyle()

  # cfg = deepcopy(CFG_POLARIZED_ETAPI0)  # perform analysis of Nizar's polarized eta pi0 data
  # overrideBeamPolInfo = BEAM_POL_INFOS["2018_08"]["PARA_0"]  # force beam polarization
  cfg = deepcopy(CFG_POLARIZED_PIPI)  # perform analysis of polarized pi+ pi- data
  overrideBeamPolInfo = None
  scaleFactor = None
  # scaleFactor = 1.6112841143413135  # gen MC weighted with L_max = 4 and analyzed with L_max = 4, 6, 8
  # scaleFactor = 2.450175524066058   # acc MC weighted with L_max = 4 and analyzed with L_max = 4
  # scaleFactor = 2.4515044898120957  # acc MC weighted with L_max = 4 and analyzed with L_max = 6
  # scaleFactor = 2.441922028485739   # acc MC weighted with L_max = 4 and analyzed with L_max = 8
  #
  # scaleFactor = 1.682258789616807  # gen MC weighted with L_max = 6 and analyzed with L_max = 4, 6, 8
  # scaleFactor = 2.5108970501733427  # acc MC weighted with L_max = 6 and analyzed with L_max = 4
  # scaleFactor = 2.5735512097120283  # acc MC weighted with L_max = 6 and analyzed with L_max = 6
  # scaleFactor = 2.582279973210192   # acc MC weighted with L_max = 6 and analyzed with L_max = 8

  plotIntensityFunctions(
    cfg                      = cfg,
    # momentType               = "meas",
    momentType               = "phys",
    # makeIntensityPosDefinite = True,
    makeIntensityPosDefinite = False,
    overrideBeamPolInfo      = overrideBeamPolInfo,
    scaleFactor              = scaleFactor,
  )
