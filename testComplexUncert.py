#!/usr/bin/env python3

import numpy as np
from scipy.stats import random_correlation
# from typing import Any, Collection, Dict, List, Optional, Tuple


def getRandomCovariance(
  nmbVars: int,
  rng:     np.random.Generator,
) -> np.ndarray:
  '''Generates random covariance matrix'''
  # generate random correlation matrix
  eigenValues = rng.random(nmbVars)
  eigenValues = eigenValues * nmbVars / np.sum(eigenValues)  # rescale such that sum of eigenvalues == nmbVars
  rho = random_correlation.rvs(eigs = eigenValues, random_state = rng)
  # generate random standard deviations
  stdDevs = rng.random(nmbVars)
  S = np.diag(rng.random(nmbVars))
  return S @ rho @ S


def func(x: np.ndarray) -> np.ndarray:
  '''Function for which to perform uncertainty propagation'''
  # return x
  return 2 * x


def funcJacobian(x: np.ndarray) -> np.ndarray:
  '''Returns Jacobian matrix of function evaluated at given point'''
  # return np.identity(x.shape[0])
  return 2 * np.identity(x.shape[0])


if __name__ == "__main__":
  # define means and covariance matrix of input values
  rng = np.random.default_rng(seed = 12345)
  nmbVars = 4
  xMeans = rng.random(nmbVars)
  xCovMat = getRandomCovariance(nmbVars, rng)

  # perform Monte Carlo uncertainty propagation
  print(f"in: mu = {xMeans}, V = \n{xCovMat}")
  # generate samples from multi-variate Gaussian
  nmbSamples = 1000000
  samples = rng.multivariate_normal(mean = xMeans, cov = xCovMat, size = nmbSamples)
  print(samples.shape, samples[0])
  # function values for each sample
  ySamples = np.array([func(x) for x in samples])
  print(ySamples.shape, ySamples[0])
  # means and covariance matrix of function values
  yMeansMc  = np.mean(ySamples, axis = 0)
  yCovMatMc = np.cov(ySamples, rowvar = False)
  print(f"MC: mu = {yMeansMc}, V = \n{yCovMatMc}")
  print(f"factor = \n{np.divide(yCovMatMc, xCovMat)}")

  # perform analytic uncertainty propagation
  yMeans  = func(xMeans)
  J = funcJacobian(xMeans)
  yCovMat = J.T @ xCovMat @ J
  print(f"analytic: mu = {yMeans}, V = \n{yCovMat}")
  print(f"factor = \n{np.divide(yCovMat, xCovMat)}")
