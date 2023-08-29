#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from scipy.stats import random_correlation


NMB_VARS = 4

# setup global random number generator
RNG = np.random.default_rng(seed = 12345)


def getRandomCovarianceReal(
  n:   int,
  rng: np.random.Generator,
) -> npt.NDArray:
  """Generates random R^{n x n} covariance matrix"""
  # generate random correlation matrix
  eigenValues = rng.random(n)
  eigenValues = eigenValues * n / np.sum(eigenValues)  # rescale such that sum of eigenvalues == nmbVars
  rho = random_correlation.rvs(eigs = eigenValues, random_state = rng)
  # generate random standard deviations
  stdDevs = rng.random(n)
  S = np.diag(rng.random(n))
  return S @ rho @ S


A = RNG.random((NMB_VARS, NMB_VARS))
def realFunc(x: npt.NDArray) -> npt.NDArray:
  """Function R^n -> R^n for which to perform uncertainty propagation"""
  return x
  # return 2 * x
  # return A @ x


def realFuncJacobian(x: npt.NDArray) -> npt.NDArray:
  """Returns R^{n x n} Jacobian matrix of R^n -> R^n function evaluated at given point"""
  return np.identity(x.shape[0])
  # return 2 * np.identity(x.shape[0])
  # return A


def testRealVectorCase(
  xMeans:  npt.NDArray,
  xCovMat: npt.NDArray,
) -> None:
  """Tests uncertainty propagation for R^n -> R^n function"""
  # perform Monte Carlo uncertainty propagation
  print(f"in: mu = {xMeans}, V = \n{xCovMat}")
  print(f"A = \n{A}")
  # generate samples from multi-variate Gaussian
  nmbSamples = 1000000
  samples = RNG.multivariate_normal(mean = xMeans, cov = xCovMat, size = nmbSamples)
  print(samples.shape, samples[0])
  # calculate function values for each sample
  ySamples = np.array([realFunc(x) for x in samples])
  print(ySamples.shape, ySamples[0])
  # calculate means and covariance matrix from function values
  yMeansMc  = np.mean(ySamples, axis = 0)
  yCovMatMc = np.cov(ySamples, rowvar = False)
  print(f"MC: mu = {yMeansMc}")
  print(f"V = \n{yCovMatMc}")
  print(f"ratio = \n{yCovMatMc / xCovMat}")

  # perform analytic uncertainty propagation
  yMeans  = realFunc(xMeans)
  J       = realFuncJacobian(xMeans)
  yCovMat = J @ (xCovMat @ J.T)  #!Note! @ is left-associative
  print(f"analytic: mu = {yMeans}, V = \n{yCovMat}")
  print(f"ratio = \n{yCovMat / yCovMatMc}")


def realVecToComplexVec(xReal: npt.NDArray) -> npt.NDArray[np.complex128]:
  """transforms R^2n vector of form [Re_0, Im_0, Re_1, Im_1, ...] to C^n vector [Re_0 + j Im_0, Re_1 + j Im_1, ...]"""
  return xReal[0::2] + 1j * xReal[1::2]


def realCovToComplexCov(
  covReal:      npt.NDArray,
  pseudoCovMat: bool = False,
) -> npt.NDArray[np.complex128]:
  """transforms R^{2n x 2n} covariance of form
  [[V[Re_0],         cov[Re_0, Im_0], ... ]
   [cov[Im_0, Re_0], V[Im_0],         ... ]
    ... ]
  to either the Hermitian covariance matrix or the pseudo-covariance matrix, both being C^{n x n}
  """
  # see https://www.wikiwand.com/en/Complex_random_vector#Covariance_matrix_and_pseudo-covariance_matrix
  # and https://www.wikiwand.com/en/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts
  V_Re_Re = covReal[0::2, 0::2]
  V_Im_Im = covReal[1::2, 1::2]
  V_Re_Im = covReal[0::2, 1::2]
  V_Im_Re = covReal[1::2, 0::2]
  if pseudoCovMat:
    return V_Re_Re - V_Im_Im + 1j * (V_Im_Re + V_Re_Im)
  else:
    return V_Re_Re + V_Im_Im + 1j * (V_Im_Re - V_Re_Im)


def complexCovToRealCov(
  covHermit: npt.NDArray[np.complex128],
  covPseudo: npt.NDArray[np.complex128],
) -> npt.NDArray:
  """transforms the C^{n x n} Hermitian and pseudo-covariance matrices into a R^{2n x 2n} covariance matrix with block form
  [[V(Re, Re),   V(Re, Im)]
   [V(Re, Im)^T, V(Im, Im)]]
  """
  n = covHermit.shape[0]
  covReal = np.zeros((2 * n, 2 * n))
  covReal[0::2, 0::2] = (np.real(covHermit) + np.real(covPseudo)) / 2  # V_Re_Re
  covReal[1::2, 1::2] = (np.real(covHermit) - np.real(covPseudo)) / 2  # V_Im_Im
  covReal[0::2, 1::2] = (np.imag(covPseudo) - np.imag(covHermit)) / 2  # V_Re_Im
  covReal[1::2, 0::2] = (np.imag(covPseudo) + np.imag(covHermit)) / 2  # V_Im_Re
  return covReal


def realCovToComplexCov2(
  covReal:      npt.NDArray,
  pseudoCovMat: bool = False,
) -> npt.NDArray[np.complex128]:
  """transforms R^{2n x 2n} covariance with block form
  [[V(Re, Re),   V(Re, Im)]
   [V(Re, Im)^T, V(Im, Im)]]
  to either the Hermitian covariance matrix or the pseudo-covariance matrix, both being C^{n x n}
  """
  # see https://www.wikiwand.com/en/Complex_random_vector#Covariance_matrix_and_pseudo-covariance_matrix
  # and https://www.wikiwand.com/en/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts
  n = covReal.shape[0] // 2
  V_Re_Re = covReal[:n, :n]
  V_Im_Im = covReal[n:, n:]
  V_Re_Im = covReal[:n, n:]
  V_Im_Re = covReal[n:, :n]
  if pseudoCovMat:
    return V_Re_Re - V_Im_Im + 1j * (V_Im_Re + V_Re_Im)
  else:
    return V_Re_Re + V_Im_Im + 1j * (V_Im_Re - V_Re_Im)


def complexCovToRealCov2(
  covHermit: npt.NDArray[np.complex128],
  covPseudo: npt.NDArray[np.complex128],
) -> npt.NDArray:
  """transforms the C^{n x n} Hermitian and pseudo-covariance matrices into a R^{2n x 2n} covariance matrix with block form
  [[V(Re, Re),   V(Re, Im)]
   [V(Re, Im)^T, V(Im, Im)]]
  """
  V_Re_Re = (np.real(covHermit) + np.real(covPseudo)) / 2
  V_Im_Im = (np.real(covHermit) - np.real(covPseudo)) / 2
  V_Re_Im = (np.imag(covPseudo) - np.imag(covHermit)) / 2
  V_Im_Re = (np.imag(covPseudo) + np.imag(covHermit)) / 2
  return np.block([
    [V_Re_Re, V_Re_Im],
    [V_Im_Re, V_Im_Im],
  ])


def covariance(
  x:    npt.NDArray,
  y:    npt.NDArray,
  xSum: complex,
  ySum: complex,
) -> npt.NDArray:
  """Computes covariance of data samples of random variables x and y"""
  N = x.shape[0]
  xMean = xSum / N
  yMean = ySum / N
  return (1 / (N - 1)) * ((x - xMean) @ np.asmatrix(y - yMean).H)


def autoCovMatrix(z: npt.NDArray) -> npt.NDArray:
  """Computes auto-covariance matrix for n-dim vector x of random variables; identical to np.cov(z)"""
  # see https://github.com/numpy/numpy/blob/v1.25.0/numpy/lib/function_base.py#L2530-L2749
  # columns of z represent variables; rows are observations
  Z = z.T
  deltaZ = Z - Z.mean(1, keepdims = True)
  return (deltaZ @ np.asmatrix(deltaZ).H) / (Z.shape[1] - 1)


def crossCovMatrix(
  x: npt.NDArray,
  y: npt.NDArray,
) -> npt.NDArray:
  """Computes cross-covariance matrix for n-dim vectors x and y of random variables; identical to np.cov(x, y)[:n, n:]"""
  # columns of x and y represent variables; rows are observations
  X = x.T
  Y = y.T
  deltaX = X - X.mean(1, keepdims = True)
  deltaY = Y - Y.mean(1, keepdims = True)
  return (deltaX @ np.asmatrix(deltaY).H) / (X.shape[1] - 1)


Acomplex = A[0::2, 0::2] + 1j * A[1::2, 1::2]
def complexFunc(z: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  """Function for which to perform uncertainty propagation"""
  return z
  # return np.conjugate(z)
  # return 2 * z
  # return (2 + 2j) * z
  # return Acomplex @ z
  # return z * z
  # return z * np.conjugate(z)


def complexFuncJacobian(z: npt.NDArray[np.complex128]) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
  """Returns Jacobian matrix of function evaluated at given point"""
  # f(z) = z
  J     = np.identity(z.shape[0], dtype = np.complex128)
  Jconj = np.zeros((z.shape[0], z.shape[0]), dtype = np.complex128)
  # # f(z) = z^*
  # J     = np.zeros((z.shape[0], z.shape[0]), dtype = np.complex128)
  # Jconj = np.identity(z.shape[0], dtype = np.complex128)
  # # f(z) = 2 * z
  # J = 2 * np.identity(z.shape[0], dtype = np.complex128)
  # Jconj = np.zeros((z.shape[0], z.shape[0]), dtype = np.complex128)
  # # f(z) = (1 + 2j) * z
  # J = (2 + 2j) * np.identity(z.shape[0], dtype = np.complex128)
  # Jconj = np.zeros((z.shape[0], z.shape[0]), dtype = np.complex128)
  # # f(z) = Acomplex @ z
  # J = Acomplex
  # Jconj = np.zeros((z.shape[0], z.shape[0]), dtype = np.complex128)
  # # f(z) = z * z
  # J = 2 * z * np.identity(z.shape[0], dtype = np.complex128)
  # Jconj = np.zeros((z.shape[0], z.shape[0]), dtype = np.complex128)
  # # f(z) = z * np.conjugate(z)
  # J = np.conjugate(z) * np.identity(z.shape[0], dtype = np.complex128)
  # Jconj = z * np.identity(z.shape[0], dtype = np.complex128)
  return J, Jconj


if __name__ == "__main__":
  # define means and covariance matrix of input values
  xMeans = 10 + RNG.random(NMB_VARS)
  xCovMat = getRandomCovarianceReal(NMB_VARS, RNG)

  # test real-valued vectors
  # testRealVectorCase(xMeans, xCovMat)

  # test complex-valued vectors
  # perform Monte Carlo uncertainty propagation
  xMeansComplex = realVecToComplexVec(xMeans)
  print(f"in: mu = {xMeans} = {xMeansComplex}, V = \n{xCovMat}")
  print(f"A = \n{Acomplex}")
  # generate samples from multi-variate Gaussian
  nmbSamples = 1000000
  samples = RNG.multivariate_normal(mean = xMeans, cov = xCovMat, size = nmbSamples)
  print(samples.shape, samples[0])

  # Hermitian and pseudo-covariance matrices
  # calculate function values for each sample
  ySamples = np.array([complexFunc(realVecToComplexVec(x)) for x in samples])
  n = ySamples.shape[1]
  print(ySamples.shape, ySamples[0])
  # calculate means and covariance matrices from function values
  yMeansMc = np.mean(ySamples, axis = 0)
  yCovMatHermitMc   = autoCovMatrix(ySamples)
  yCovMatHermitMcNp = np.cov(ySamples, rowvar = False)
  # yCovMatHermitMc   = crossCovMatrix(ySamples, ySamples)
  # yCovMatHermitMcNp = np.cov(ySamples, ySamples, rowvar = False)[:n, n:]
  yCovMatPseudoMc   = crossCovMatrix(ySamples, np.conjugate(ySamples))
  # surprisingly, instead of calculating the cross covariance V[x, y]
  # np.cov(x, y) stacks x on top of y calculates the full covariance matrix
  # with the block form:
  # [[V[x],    V[x, y]]
  #  [V[y, x], V[y]]]
  # see https://github.com/numpy/numpy/issues/2623
  yCovMatPseudoMcNp = np.cov(ySamples, np.conjugate(ySamples), rowvar = False)[:n, n:]
  xCovMatHermit     = realCovToComplexCov(xCovMat)
  xCovMatPseudo     = realCovToComplexCov(xCovMat, pseudoCovMat = True)
  print(f"MC: mu = {yMeansMc}")
  print(f"V_y_Hermit = \n{yCovMatHermitMc}")
  print(f"vs. \n{yCovMatHermitMcNp}")
  print(f"ratio = \n{np.real_if_close(yCovMatHermitMc / yCovMatHermitMcNp)}")
  print(f"vs. \n{xCovMatHermit}")
  print(f"ratio = \n{yCovMatHermitMc / xCovMatHermit}")
  print(f"V_y_pseudo = \n{yCovMatPseudoMc}")
  print(f"vs. \n{yCovMatPseudoMcNp}")
  print(f"ratio = \n{np.real_if_close(yCovMatPseudoMc / yCovMatPseudoMcNp)}")
  print(f"vs. \n{xCovMatPseudo}")
  print(f"ratio = \n{yCovMatPseudoMc / xCovMatPseudo}")
  # test conversion routines
  xCovMatReal = complexCovToRealCov(xCovMatHermit, xCovMatPseudo)
  print(f"complexCovToRealCov() ratio = \n{np.real_if_close(xCovMatReal / xCovMat)}")
  print(f"realCovToComplexCov(Hermitian) ratio = \n{np.real_if_close(xCovMatHermit / realCovToComplexCov(xCovMatReal))}")
  print(f"realCovToComplexCov(Pseudo) ratio = \n{np.real_if_close(xCovMatPseudo / realCovToComplexCov(xCovMatReal, pseudoCovMat = True))}")

  # augmented vectors and matrices
  ySamplesAug   = np.block([ySamples, np.conjugate(ySamples)])
  yCovMatAugMc  = np.cov(ySamplesAug, rowvar = False)
  yCovMatAugMc2 = np.cov(ySamples, np.conjugate(ySamples), rowvar = False)
  yCovMatAugMc3 = np.block([
    [yCovMatHermitMc, yCovMatPseudoMc],
    [np.conjugate(yCovMatPseudoMc), np.conjugate(yCovMatHermitMc)],
  ])
  print(f"V_y_Aug_MC = \n{yCovMatAugMc}")
  print(f"vs. \n{yCovMatAugMc2}")
  print(f"ratio = \n{np.real_if_close(yCovMatAugMc / yCovMatAugMc2)}")
  print(f"vs. \n{yCovMatAugMc3}")
  print(f"ratio = \n{np.real_if_close(yCovMatAugMc / yCovMatAugMc3)}")

  # perform analytic uncertainty propagation
  yMeans = complexFunc(realVecToComplexVec(xMeans))
  J, Jconj = complexFuncJacobian(realVecToComplexVec(xMeans))
  Jaug = np.block([
    [J, Jconj],
    [np.conjugate(Jconj), np.conjugate(J)],
  ])
  xCovMatAug = np.block([
    [xCovMatHermit, xCovMatPseudo],
    [np.conjugate(xCovMatPseudo), np.conjugate(xCovMatHermit)],
  ])
  print(f"J_aug = \n{Jaug}")
  yCovMatAug = Jaug @ (xCovMatAug @ np.asmatrix(Jaug).H)  #!Note! @ is left-associative
  print(f"analytic: mu = {yMeans}")
  print(f"ratio = \n{yMeans / yMeansMc}")
  print(f"V_y_Aug_Prop = \n{yCovMatAug}")
  print(f"ratio = \n{yCovMatAug / yCovMatAugMc}")
