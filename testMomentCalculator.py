#!/usr/bin/env python3

from __future__ import annotations

import bidict as bd
import copy
import functools
import numpy as np
from typing import TYPE_CHECKING
import sys

import ROOT

from makeMomentsInputTree import (
  CPP_CODE_ANGLES_GLUEX_AMPTOOLS,
  CPP_CODE_BEAM_POL_PHI,
  CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE,
)
from MomentCalculator import (
  AcceptanceIntegralMatrix,
  AmplitudeSet,
  AmplitudeValue,
  DataSet,
  KinematicBinningVariable,
  MomentCalculator,
  MomentIndices,
  MomentResult,
  QnMomentIndex,
  QnWaveIndex,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  if False:
    for polarized in (True, False):
      print(f"MomentIndices for {'' if polarized else 'un'}polarized moments:")
      momentIndices = MomentIndices(maxL = 5, polarized = polarized)
      for i in momentIndices.flatIndices:
        print(f"{i} : {momentIndices[i]}")
      for i in momentIndices.qnIndices:
        print(f"{i} : {momentIndices[i]}")
      qnIndex = QnMomentIndex(0, 0, 0)
      print(f"{type(momentIndices._qnIndexByFlatIndex[0])=}")
      print(f"{momentIndices[0]=} vs. {momentIndices[qnIndex]=}")
      print(f"__str__()  = '{momentIndices}'")
      print(f"__repr__() = '{repr(momentIndices)}'")
      momentIndices2 = MomentIndices(maxL = 5, polarized = polarized)
      print(f"equality: {momentIndices == momentIndices2=}")
      if TYPE_CHECKING:
        a = bd.bidict({0 : qnIndex})
        reveal_type(a)
      qnIndex = QnMomentIndex(3, 0, 0)
      try:
        print(f"{momentIndices[qnIndex]}")
      except KeyError as e:
        print(f"KeyError: {e}: {qnIndex} does not exist in momentIndices")

    print("Integral matrix:")
    integralMatrix = AcceptanceIntegralMatrix(
      indicesMeas = MomentIndices(maxL = 4, polarized = True),
      indicesPhys = MomentIndices(maxL = 4, polarized = True),
      dataSet     = DataSet(data = None, phaseSpaceData = None, nmbGenEvents = 0, polarization = 1.0),  # dummy data set
    )
    # integralMatrix.load("./plotsPhotoProdPiPiPol/2018_08/tbin_0.1_0.2/PARA_0.maxL_4/integralMatrix_mass_0.460.npy")
    integralMatrix.load("./plotsPhotoProdPiPiPol.allH2/2018_08/tbin_0.1_0.2/PARA_0.maxL_4/integralMatrix_mass_0.460.npy")
    indexMeas = QnMomentIndex(momentIndex = 2, L = 0, M = 0)
    for indexPhys in integralMatrix.indicesPhys.qnIndices:
      print(f"Integral[{indexMeas=}, {indexPhys=}] = {integralMatrix[indexMeas, indexPhys]}")

    print("MomentResult for polarized moments:")
    momentResult = MomentResult(MomentIndices(maxL = 5, polarized = True))
    momentResult.valid = True
    print(f"{momentResult=}")
    print(f"{momentResult._valsFlatIndex=}")
    print(f"{momentResult[0]=}")
    print(f"{momentResult[56]=}")
    qnIndex = QnMomentIndex(2, 5, 5)
    print(f"{momentResult[qnIndex].val=}")
    print(f"{momentResult[qnIndex].uncertRe=}")
    print(f"{momentResult[qnIndex].uncertIm=}")

  amplitudes = {
    True : [  # polarized photoproduction
      #                         refl J   M    amplitude
      # negative-reflectivity waves
      AmplitudeValue(QnWaveIndex(-1, 0,  0),  1.0 + 0.0j),  # S_0^-
      AmplitudeValue(QnWaveIndex(-1, 1, -1), -0.4 + 0.1j),  # P_-1^-
      AmplitudeValue(QnWaveIndex(-1, 1,  0),  0.3 - 0.8j),  # P_0^-
      AmplitudeValue(QnWaveIndex(-1, 1, +1), -0.8 + 0.7j),  # P_+1^-
      AmplitudeValue(QnWaveIndex(-1, 2, -2),  0.1 - 0.4j),  # D_-2^-
      AmplitudeValue(QnWaveIndex(-1, 2, -1),  0.5 + 0.2j),  # D_-1^-
      AmplitudeValue(QnWaveIndex(-1, 2,  0), -0.1 - 0.2j),  # D_ 0^-
      AmplitudeValue(QnWaveIndex(-1, 2, +1),  0.2 - 0.1j),  # D_+1^-
      AmplitudeValue(QnWaveIndex(-1, 2, +2), -0.2 + 0.3j),  # D_+2^-
      # positive-reflectivity waves
      AmplitudeValue(QnWaveIndex(+1, 0,  0),  0.5 + 0.0j),  # S_0^+
      AmplitudeValue(QnWaveIndex(+1, 1, -1),  0.5 - 0.1j),  # P_-1^+
      AmplitudeValue(QnWaveIndex(+1, 1,  0), -0.8 - 0.3j),  # P_0^+
      AmplitudeValue(QnWaveIndex(+1, 1, +1),  0.6 + 0.3j),  # P_+1^+
      AmplitudeValue(QnWaveIndex(+1, 2, -2),  0.2 + 0.1j),  # D_-2^+
      AmplitudeValue(QnWaveIndex(+1, 2, -1),  0.2 - 0.3j),  # D_-1^+
      AmplitudeValue(QnWaveIndex(+1, 2,  0),  0.1 - 0.2j),  # D_ 0^+
      AmplitudeValue(QnWaveIndex(+1, 2, +1),  0.2 + 0.5j),  # D_+1^+
      AmplitudeValue(QnWaveIndex(+1, 2, +2), -0.3 - 0.1j),  # D_+2^+
    ],
    False : [  # unpolarized production
      #                          refl  J   M    amplitude
      AmplitudeValue(QnWaveIndex(None, 0,  0),  1.0 + 0.0j),  # S_0
      AmplitudeValue(QnWaveIndex(None, 1, -1), -0.4 + 0.1j),  # P_-1
      AmplitudeValue(QnWaveIndex(None, 1,  0),  0.3 - 0.8j),  # P_0
      AmplitudeValue(QnWaveIndex(None, 1, +1), -0.8 + 0.7j),  # P_+1
      AmplitudeValue(QnWaveIndex(None, 2, -2),  0.1 - 0.4j),  # D_-2
      AmplitudeValue(QnWaveIndex(None, 2, -1),  0.5 + 0.2j),  # D_-1
      AmplitudeValue(QnWaveIndex(None, 2,  0), -0.1 - 0.2j),  # D_ 0
      AmplitudeValue(QnWaveIndex(None, 2, +1),  0.2 - 0.1j),  # D_+1
      AmplitudeValue(QnWaveIndex(None, 2, +2), -0.2 + 0.3j),  # D_+2
    ],
  }
  if False:
    for polarized, amps in amplitudes.items():
      print(f"AmplitudeSet for {'' if polarized else 'un'}polarized moments:")
      ampSet = AmplitudeSet(amps, tolerance = 1e-13)
      qnIndex = QnWaveIndex(+1 if polarized else None, 2, +2)
      print(f"{ampSet[qnIndex]=}")
      ampSet[qnIndex] = 100 + 100j
      print(f"{ampSet[qnIndex]=}")
      for amp in ampSet.amplitudes():
        print(f"{amp=}")
      print(f"{ampSet.maxSpin=}")

      for refl in (-1, +1) if polarized else (None, ):
        for amp1 in ampSet.amplitudes(onlyRefl = refl):
          l1 = amp1.qn.l
          m1 = amp1.qn.m
          for amp2 in ampSet.amplitudes(onlyRefl = refl):
            l2 = amp2.qn.l
            m2 = amp2.qn.m
            rhos = ampSet.photoProdSpinDensElements(refl, l1, l2, m1, m2)
            print(f"rho {refl=}; ({l1=}, {m1=}); ({l2=}, {m2=}) = {rhos}")

      print(f"MomentResult for {'' if polarized else 'un'}polarized moments:")
      H:  MomentResult = ampSet.photoProdMomentResult(maxL = 4, normalize = False, printMomentFormulas = True)
      print(f"H =\n{H}")
      H2 = copy.deepcopy(H)
      H2._valsFlatIndex[0] += 1e-6
      print(f"equality: {H == H2=}, {H2._valsFlatIndex[0]=}")
      H2._valsFlatIndex[0] += 1e-5
      print(f"equality: {H == H2=}, {H2._valsFlatIndex[0]=}")

      print(f"Intensity formula for {'' if polarized else 'un'}polarized moments:")
      ampSet.intensityFormula(
        polarization = 1.0,
        thetaFormula = "theta",
        phiFormula   = "phi",
        PhiFormula   = "Phi",
        printFormula = True,
      )

  if True:
    # test intensity formula from moments
    ampSet = AmplitudeSet(amplitudes[True], tolerance = 1e-13)
    H: MomentResult = ampSet.photoProdMomentResult(maxL = 4, normalize = False, printMomentFormulas = False)
    # H.indices.setPolarized(False)
    print(f"H =\n{H}")
    intensityFormula = H.intensityFormula(
      polarization      = "beamPol",
      # polarization      = None,
      thetaFormula      = "theta",
      phiFormula        = "phi",
      PhiFormula        = "Phi",
      printFormula      = True,
      useMomentSymbols  = True,
      useIntensityTerms = MomentResult.IntensityTermsType.ALL,
      # useIntensityTerms = MomentResult.IntensityTermsType.PARITY_CONSERVING,
      # useIntensityTerms = MomentResult.IntensityTermsType.PARITY_VIOLATING,
    )

    # test whether intensity formula and intensity function for likelihood fit give identical values
    # generate 3D grid for angular variables
    # define bin edges
    nmbBinsPerAxis = 25
    thetaBinEdges = np.linspace(0,       np.pi, nmbBinsPerAxis + 1)  # theta from 0 to pi
    phiBinEdges   = np.linspace(-np.pi, +np.pi, nmbBinsPerAxis + 1)  # phi from -pi to pi
    PhiBinEdges   = np.linspace(-np.pi, +np.pi, nmbBinsPerAxis + 1)  # Phi from -pi to pi
    # define bin centers
    thetaBinCenters = (thetaBinEdges[:-1] + thetaBinEdges[1:]) / 2
    phiBinCenters   = (phiBinEdges  [:-1] + phiBinEdges  [1:]) / 2
    PhiBinCenters   = (PhiBinEdges  [:-1] + PhiBinEdges  [1:]) / 2
    # create the full grid (Cartesian product) in a memory-efficient manner
    thetas, phis, Phis = np.meshgrid(thetaBinCenters, phiBinCenters, PhiBinCenters, indexing = "ij", copy = False)
    # dictionary of NumPy arrays with all grid values
    gridData = {
      "theta" : thetas.ravel(),
      "phi"   : phis.ravel(),
      "Phi"   : Phis.ravel(),
    }
    # convert to ROOT data frame
    rdf = ROOT.RDF.FromNumpy(gridData)
    # rdf.Describe().Print()
    # setup a moment calculator and construct negative log-likelihood function using data generated above
    momentCalculator = MomentCalculator(
      indicesMeas = H.indices,  # dummy, not used in this test
      indicesPhys = H.indices,
      dataSet     = DataSet(
        data           = rdf,
        phaseSpaceData = None,
        nmbGenEvents   = 0,
        polarization   = 1.0,
      ),
      binCenters  = {
        KinematicBinningVariable(
          name      = "mass",
          label     = "#it{m}_{#it{#pi}^{#plus}#it{p}}",
          unit      = "GeV/#it{c}^{2}",
          nmbDigits = 3,
        ) : 0.74,
      },  # dummy bin
    )
    negativeLogLikelihoodFcn = momentCalculator.negativeLogLikelihoodFcn()
    # prepare moment values vector for intensity function
    reIndexRange = (
      QnMomentIndex(momentIndex = 0, L = 0,              M = 0),
      QnMomentIndex(momentIndex = 1, L = H.indices.maxL, M = H.indices.maxL),
    )  # all H_0 and H_1 moments are real-valued
    imIndexRange = (
      QnMomentIndex(momentIndex = 2, L = 1,              M = 1),
      QnMomentIndex(momentIndex = 2, L = H.indices.maxL, M = H.indices.maxL)
    )  # all H_2 moments are purely imaginary; all H_2(L, 0) are 0
    # convert to flat-index ranges
    reSlice = slice(H.indices[reIndexRange[0]], H.indices[reIndexRange[1]] + 1)
    imSlice = slice(H.indices[imIndexRange[0]], H.indices[imIndexRange[1]] + 1)
    # print(f"{reSlice=}, {imSlice=}")
    print(f"!!! {H._valsFlatIndex=}")
    momentValues = np.concatenate((np.real(H._valsFlatIndex[reSlice]), np.imag(H._valsFlatIndex[imSlice])))
    print(f"!!! {momentValues.dtype=}\n{momentValues=}")
    # get intensities evaluated at data points
    _, intensitiesFcnNll = negativeLogLikelihoodFcn._intensityFcn(momentValues)
    print(f"!!! {intensitiesFcnNll.dtype=}\n{intensitiesFcnNll=}")

    # compare with intensity formula from `MomentResult`
    # declare C++ functions
    ROOT.gInterpreter.Declare(CPP_CODE_FIX_AZIMUTHAL_ANGLE_RANGE)
    ROOT.gInterpreter.Declare(CPP_CODE_ANGLES_GLUEX_AMPTOOLS)
    ROOT.gInterpreter.Declare(CPP_CODE_BEAM_POL_PHI)
    intensityFcnMoments = ROOT.TF3(
      f"intensityFcnMoments",
      H.intensityFormula(
        polarization      = 1.0,
        thetaFormula      = "x",
        phiFormula        = "y",
        PhiFormula        = "z",
        useIntensityTerms = MomentResult.IntensityTermsType.ALL,  # must agree with other intensity values
        # useIntensityTerms = MomentResult.IntensityTermsType.PARITY_CONSERVING,  # must agree with other intensity values
        # useIntensityTerms = MomentResult.IntensityTermsType.PARITY_VIOLATING,  # will not agree with other intensity values
      ),
      0, np.pi, -np.pi, +np.pi, -np.pi, +np.pi,
    )
    intensityFcnMomentsVect = np.vectorize(intensityFcnMoments.Eval)
    intensitiesFcnMoments = intensityFcnMomentsVect(gridData["theta"], gridData["phi"], gridData["Phi"])
    deltaFcnNll = intensitiesFcnMoments - intensitiesFcnNll
    print(f"Maximum deviation between intensity formula from MomentResult and intensity function from negative log-likelihood: {np.max(np.abs(deltaFcnNll))}")

    # compare with intensity formula from `AmplitudeSet`
    intensityFcnAmpSet = ROOT.TF3(
      f"intensityFcnAmpSet",
      ampSet.intensityFormula(
        polarization = 1.0,
        thetaFormula = "x",
        phiFormula   = "y",
        PhiFormula   = "z",
        printFormula = True,
      ),
      0, np.pi, -np.pi, +np.pi, -np.pi, +np.pi,
    )
    intensityFcnAmpSetVect = np.vectorize(intensityFcnAmpSet.Eval)
    intensitiesFcnAmpSet = intensityFcnAmpSetVect(gridData["theta"], gridData["phi"], gridData["Phi"])
    deltaFcnAmpSet = intensitiesFcnMoments - intensitiesFcnAmpSet
    print(f"Maximum deviation between intensity formula from MomentResult and from amplitude set: {np.max(np.abs(deltaFcnAmpSet))}")
    # with np.printoptions(threshold = sys.maxsize):
    #   print(f"!!! {deltaFcnAmpSet=}")

# polarized case:

# Intensity formula = (0.28209479177387814 * std::real(complexT([ReH0_0_0], [ImH0_0_0]) * Ylm(0, 0, theta, phi)) + 0.4886025119029199 * std::real(complexT([ReH0_1_0], [ImH0_1_0]) * Ylm(1, 0, theta, phi)) + 0.9772050238058398 * std::real(complexT([ReH0_1_1], [ImH0_1_1]) * Ylm(1, 1, theta, phi))) - ((-0.28209479177387814 * std::real(complexT([ReH1_0_0], [ImH1_0_0]) * Ylm(0, 0, theta, phi))) + (-0.4886025119029199 * std::real(complexT([ReH1_1_0], [ImH1_1_0]) * Ylm(1, 0, theta, phi))) + (-0.9772050238058398 * std::real(complexT([ReH1_1_1], [ImH1_1_1]) * Ylm(1, 1, theta, phi)))) * beamPol * std::cos(2 * Phi) - ((-0.9772050238058398 * std::real(complexT([ReH2_1_1], [ImH2_1_1]) * Ylm(1, 1, theta, phi)))) * beamPol * std::sin(2 * Phi)

# Intensity formula =
#   (
#      0.28209479177387814 * std::real(complexT([ReH0_0_0], [ImH0_0_0]) * Ylm(0, 0, theta, phi))
#    + 0.4886025119029199  * std::real(complexT([ReH0_1_0], [ImH0_1_0]) * Ylm(1, 0, theta, phi))
#    + 0.9772050238058398  * std::real(complexT([ReH0_1_1], [ImH0_1_1]) * Ylm(1, 1, theta, phi))
#   )
# - (
#      (-0.28209479177387814 * std::real(complexT([ReH1_0_0], [ImH1_0_0]) * Ylm(0, 0, theta, phi)))
#    + (-0.4886025119029199  * std::real(complexT([ReH1_1_0], [ImH1_1_0]) * Ylm(1, 0, theta, phi)))
#    + (-0.9772050238058398  * std::real(complexT([ReH1_1_1], [ImH1_1_1]) * Ylm(1, 1, theta, phi)))
#   ) * beamPol * std::cos(2 * Phi)
# - (
#    (-0.9772050238058398 * std::real(complexT([ReH2_1_1], [ImH2_1_1]) * Ylm(1, 1, theta, phi)))
#   ) * beamPol * std::sin(2 * Phi)


# Intensity formula = (0.28209479177387814 * [H0_0_0] * ReYlm(0, 0, theta, phi) + 0.4886025119029199 * [H0_1_0] * ReYlm(1, 0, theta, phi) + 0.9772050238058398 * [H0_1_1] * ReYlm(1, 1, theta, phi)) - ((-0.28209479177387814 * [H1_0_0] * ReYlm(0, 0, theta, phi)) + (-0.4886025119029199 * [H1_1_0] * ReYlm(1, 0, theta, phi)) + (-0.9772050238058398 * [H1_1_1] * ReYlm(1, 1, theta, phi))) * beamPol * std::cos(2 * Phi) - (0.9772050238058398 * [H2_1_1] * ImYlm(1, 1, theta, phi)) * beamPol * std::sin(2 * Phi)

# Intensity formula =
#   (
#      0.28209479177387814 * [H0_0_0] * ReYlm(0, 0, theta, phi)
#    + 0.4886025119029199  * [H0_1_0] * ReYlm(1, 0, theta, phi)
#    + 0.9772050238058398  * [H0_1_1] * ReYlm(1, 1, theta, phi)
#   )
# - (
#      (-0.28209479177387814 * [H1_0_0] * ReYlm(0, 0, theta, phi))
#    + (-0.4886025119029199  * [H1_1_0] * ReYlm(1, 0, theta, phi))
#    + (-0.9772050238058398  * [H1_1_1] * ReYlm(1, 1, theta, phi))
#   ) * beamPol * std::cos(2 * Phi)
# - (0.9772050238058398 * [H2_1_1] * ImYlm(1, 1, theta, phi)) * beamPol * std::sin(2 * Phi)


# unpolarized case:

# Intensity formula = (0.28209479177387814 * [H0_0_0] * ReYlm(0, 0, theta, phi) + 0.4886025119029199 * [H0_1_0] * ReYlm(1, 0, theta, phi) + 0.9772050238058398 * [H0_1_1] * ReYlm(1, 1, theta, phi))

# Intensity formula = (0.28209479177387814 * std::real(complexT([ReH0_0_0], [ImH0_0_0]) * Ylm(0, 0, theta, phi)) + 0.4886025119029199 * std::real(complexT([ReH0_1_0], [ImH0_1_0]) * Ylm(1, 0, theta, phi)) + 0.9772050238058398 * std::real(complexT([ReH0_1_1], [ImH0_1_1]) * Ylm(1, 1, theta, phi)))
