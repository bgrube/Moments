#!/usr/bin/env python3

import numpy as np

import ROOT


CPP_CODE = """
#include <complex>
#include <vector>

std::vector<std::complex<double>>
getVector()
{
	const size_t n = 10;
	std::vector<std::complex<double>> vec(n);
	for (size_t i = 0; i < n; ++i) {
		vec[i] = std::complex<double>(i, i);
	}
	return vec;
}
"""


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gInterpreter.Declare(CPP_CODE)

  vec = ROOT.getVector()
  print(f"{type(vec)=}, {type(vec[0])=}, {vec[0]=}")
  npVec = np.asarray(vec)  # with ROOT 6.32.08 this gives `AttributeError: 'complex<double>' object has no attribute '__array__'`
