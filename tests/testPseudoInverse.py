#!/usr/bin/env python3

import functools
import numpy as np
import scipy
import threadpoolctl

import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  dim = 100
  nmbIterations = 10000
  rng = np.random.default_rng(42)

  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  with threadController.limit(limits = 1):
    print(f"State of ThreadpoolController:\n{threadController.info()}")
    timer = Utilities.Timer()
    # generate random matrices
    with timer.timeThis(f"Generating {nmbIterations} matrices of dimension {dim} times {dim}"):
      matrices = [rng.random((dim, dim))  # uniform floats in [0,1)
                  for _ in range(nmbIterations)]
    # time calculation of pseudo-inverse using numpy
    with timer.timeThis(f"Calculating pseudo-inverse using numpy.linalg.pinv() with {threadController.info()[0].get('num_threads', 'unknown number of')} threads"):
      invMatricesNp = []
      for matrix in matrices:
        invMatricesNp.append(np.linalg.pinv(matrix))
    # time calculation of pseudo-inverse using scipy
    with timer.timeThis(f"Calculating pseudo-inverse using scipy.linalg.pinv() with {threadController.info()[0].get('num_threads', 'unknown number of')} threads"):
      invMatricesSp = []
      for matrix in matrices:
        invMatricesSp.append(scipy.linalg.pinv(matrix))
    with timer.timeThis(f"Calculating differences"):
      diffMatrices = []
      for invNp, invSp in zip(invMatricesNp, invMatricesSp):
        diffMatrices.append(np.abs(invNp - invSp).max())
      print(f"{max(diffMatrices)=}")
    print(timer.summary)
