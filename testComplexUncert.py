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
  '''Generates random R^{n x n} covariance matrix'''
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
  '''Function R^n -> R^n for which to perform uncertainty propagation'''
  return x
  # return 2 * x
  # return A @ x


def realFuncJacobian(x: npt.NDArray) -> npt.NDArray:
  '''Returns R^{n x n} Jacobian matrix of R^n -> R^n function evaluated at given point'''
  return np.identity(x.shape[0])
  # return 2 * np.identity(x.shape[0])
  # return A


def testRealVectorCase(
  xMeans:  npt.NDArray,
  xCovMat: npt.NDArray,
) -> None:
  '''Tests uncertainty propagation for R^n -> R^n function'''
  # perform Monte Carlo uncertainty propagation
  print(f"in: mu = {xMeans}, V = \n{xCovMat}")
  print(f"A = \n{A}")
  # generate samples from multi-variate Gaussian
  nmbSamples = 1000000
  samples = RNG.multivariate_normal(mean = xMeans, cov = xCovMat, size = nmbSamples)
  print(samples.shape, samples[0])
  # function values for each sample
  ySamples = np.array([realFunc(x) for x in samples])
  print(ySamples.shape, ySamples[0])
  # means and covariance matrix of function values
  yMeansMc  = np.mean(ySamples, axis = 0)
  yCovMatMc = np.cov(ySamples, rowvar = False)
  print(f"MC: mu = {yMeansMc}, V = \n{yCovMatMc}")
  print(f"factor = \n{np.divide(yCovMatMc, xCovMat)}")

  # perform analytic uncertainty propagation
  yMeans  = realFunc(xMeans)
  J       = realFuncJacobian(xMeans)
  yCovMat = J @ (xCovMat @ J.T)  #!Note! @ is left-associative
  print(f"analytic: mu = {yMeans}, V = \n{yCovMat}")
  print(f"factor = \n{np.divide(yCovMat, yCovMatMc)}")


def complexFromRealVector(xReal: npt.NDArray) -> npt.NDArray[np.complex128]:
  '''transforms R^2n vector of form [Re_0, Im_0, Re_1, Im_1, ...] to C^n vector [Re_0 + j Im_0, Re_1 + j Im_1, ...]'''
  return xReal[0::2] + 1j * xReal[1::2]


def complexFromRealCov(
  covReal:      npt.NDArray,
  pseudoCovMat: bool = False,
) -> npt.NDArray[np.complex128]:
  '''transforms R^{2n x 2n} covariance of form
  [[V[Re_0],         cov[Re_0, Im_0], ... ]
   [cov[Im_0, Re_0], V[Im_0],         ... ]
    ... ]
  to either the Hermitian covariance matrix or the pseudo-covariance matrix, both being C^{n x n}
  '''
  # see https://www.wikiwand.com/en/Complex_random_vector#Covariance_matrix_and_pseudo-covariance_matrix
  # and https://www.wikiwand.com/en/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts
  n = covReal.shape[0] // 2
  if pseudoCovMat:
    return covReal[:n, :n] - covReal[n:, n:] + 1j * (covReal[:n, n:] + covReal[n:, :n])
  else:
    return covReal[:n, :n] + covReal[n:, n:] + 1j * (covReal[:n, n:] - covReal[n:, :n])


def complexFunc(x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  '''Function for which to perform uncertainty propagation'''
  return x


def complexFuncJacobian(x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  '''Returns Jacobian matrix of function evaluated at given point'''
  return np.identity(x.shape[0], dtype = np.complex128)


def covariance(
  x:    npt.NDArray,
  y:    npt.NDArray,
  xSum: float,
  ySum: float,
) -> npt.NDArray:
  '''Computes covariance of data samples of random variables x and y'''
  N = x.shape[0]
  # xySum = x.T @ y
  # return (1 / (N - 1)) * (xySum - xSum * ySum / N)
  return (1 / (N - 1)) * ((x - xSum / N).T @ (y - ySum / N))


def covMatrixReal(x: npt.NDArray) -> npt.NDArray:
  '''Computes covariance matrix for R^n vector x of random variables; identical to np.cov()'''
  n = x.shape[1]
  cov = np.zeros((n, n))
  xSums = np.sum(x, axis = 0)
  for i in range(n):
    for j in range(n):
      cov[i, j] = covariance(x[:, i], x[:, j], xSums[i], xSums[j])
  return cov


def covMatrixComplex(
  x:            npt.NDArray[np.complex128],
  pseudoCovMat: bool = False,
) -> npt.NDArray[np.complex128]:
  '''Computes Hermitian or pseudo-covariance matrix for C^n vector x of random variables'''
  n = x.shape[1]
  cov = np.zeros((n, n), dtype = np.complex128)
  xSums = np.sum(x, axis = 0)
  for i in range(n):
    for j in range(n):
      cov[i, j] = covariance(x[:, i], x[:, j], xSums[i], xSums[j]) if pseudoCovMat \
                  else covariance(x[:, i], np.conjugate(x[:, j]), xSums[i], np.conjugate(xSums[j]))  # identical to np.cov()
  return cov


if __name__ == "__main__":
  # define means and covariance matrix of input values
  xMeans = RNG.random(NMB_VARS)
  xCovMat = getRandomCovarianceReal(NMB_VARS, RNG)

  # test real-valued vectors
  # testRealVectorCase(xMeans, xCovMat)

  # test complex-valued vectors
  # perform Monte Carlo uncertainty propagation
  xMeansComplex = complexFromRealVector(xMeans)
  print(f"in: mu = {xMeans} = {xMeansComplex}, V = \n{xCovMat}")
  # print(f"A = \n{A}")
  # generate samples from multi-variate Gaussian
  nmbSamples = 1000000
  samples = RNG.multivariate_normal(mean = xMeans, cov = xCovMat, size = nmbSamples)
  print(samples.shape, samples[0])
  # function values for each sample
  ySamples = np.array([complexFunc(complexFromRealVector(x)) for x in samples])
  print(ySamples.shape, ySamples[0])
  # means and covariance matrix of function values
  yMeansMc = np.mean(ySamples, axis = 0)
  yCovMatHermitMc = covMatrixComplex(ySamples)
  yCovMatPseudoMc = covMatrixComplex(ySamples, pseudoCovMat = True)
  # print(f"factor = \n{np.divide(yCovMatMc, xCovMat)}")
  xCovMatHermit = complexFromRealCov(xCovMat)
  xCovMatPseudo = complexFromRealCov(xCovMat, pseudoCovMat = True)
  # xCovMatPseudo = complexFromRealCov(xCovMat, pseudoCovMat = True)
  print(f"MC: mu = {yMeansMc}\nV_y_Hermit = \n{yCovMatHermitMc}")
  print(f"V_x_Hermit = \n{xCovMatHermit}")
  print(f"foo = \n{np.divide(yCovMatHermitMc, np.cov(ySamples, rowvar = False))}")
  print(f"V_y_pseudo = \n{yCovMatPseudoMc}")
  print(f"V_x_pseudo = \n{xCovMatPseudo}")
