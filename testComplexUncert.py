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
  '''function for which to perform uncertainty propagation'''
  return 2 * x


if __name__ == "__main__":
  rng = np.random.default_rng(seed = 12345)
  nmbVars = 4
  means = rng.random(nmbVars)
  covMat = getRandomCovariance(nmbVars, rng)
  print(f"in: mu = {means}, V = \n{covMat}")
  # generate Gaussian samples
  nmbSamples = 1000000
  samples = rng.multivariate_normal(mean = means, cov = covMat, size = nmbSamples)
  print(samples.shape, samples[0])

  funcVals = np.array([func(x) for x in samples])
  print(funcVals.shape, funcVals[0])
  meansSample  = np.mean(funcVals, axis = 0)
  covMatSample = np.cov(funcVals, rowvar = False)
  print(f"out: mu = {meansSample}, V = \n{covMatSample}")
  # print(f"delta V = \n{covMat - covMatSample}")
  print(f"factor = \n{np.divide(covMatSample, covMat)}")
