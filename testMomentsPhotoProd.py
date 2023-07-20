#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=2

import functools
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Any, Collection, Dict, List, Optional, Tuple

import py3nj
from uncertainties import UFloat, ufloat

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# # see https://root-forum.cern.ch/t/tf1-eval-as-a-function-in-rdataframe/50699/3
# def declareInCpp(**kwargs: Any) -> None:
#   '''Creates C++ variables (names = keys of kwargs) for PyROOT objects (values of kwargs) in PyVars:: namespace'''
#   for key, value in kwargs.items():
#     ROOT.gInterpreter.Declare(  # type: ignore
# f'''
# namespace PyVars
# {{
#   auto& {key} = *reinterpret_cast<{type(value).__cpp_name__}*>({ROOT.addressof(value)});
# }}
# ''')


# set of all possible waves up to ell = 2
PROD_AMPS: Dict[int, Dict[Tuple[int, int,], complex]] = {
  # negative-reflectivity waves
  -1 : {
    #J   M    amplitude
    (0,  0) :  1   + 0j,    # S_0^-
    (1, -1) : -0.4 + 0.1j,  # P_-1^-
    (1,  0) :  0.3 - 0.8j,  # P_0^-
    (1, +1) : -0.8 + 0.7j,  # P_+1^-
    (2, -2) :  0.1 - 0.4j,  # D_-2^-
    (2, -1) :  0.5 + 0.2j,  # D_-1^-
    (2,  0) : -0.1 - 0.2j,  # D_ 0^-
    (2, +1) :  0.2 - 0.1j,  # D_+1^-
    (2, +2) : -0.2 + 0.3j,  # D_+2^-
  },
  # positive-reflectivity waves
  +1 : {
    #J   M    amplitude
    (0,  0) :  0.5 + 0j,    # S_0^+
    (1, -1) :  0.5 - 0.1j,  # P_-1^+
    (1,  0) : -0.8 - 0.3j,  # P_0^+
    (1, +1) :  0.6 + 0.3j,  # P_+1^+
    (2, -2) :  0.2 + 0.1j,  # D_-2^+
    (2, -1) :  0.2 - 0.3j,  # D_-1^+
    (2,  0) :  0.1 - 0.2j,  # D_ 0^+
    (2, +1) :  0.2 + 0.5j,  # D_+1^+
    (2, +2) : -0.3 - 0.1j,  # D_+2^+
  },
}


# # C++ implementation of RDataFrame custom action that calculates covariance between two columns
# ROOT.gROOT.LoadMacro("./Covariance.C++")  # type: ignore
# # C++ implementation of (complex conjugated) Wigner D function
# # also provides complexT typedef for std::complex<double>
# ROOT.gROOT.LoadMacro("./wignerD.C++")  # type: ignore


# def theta(m: int) -> float:
#   '''Calculates normalization factor in reflectivity basis'''
#   # see Eq. (19)
#   if m > 0:
#     return 1 / math.sqrt(2)
#   elif m == 0:
#     return 1 / 2
#   else:
#     return 0


def calcMomentSetFromWaves(
  prodAmps: Dict[int, Dict[Tuple[int, int,], complex]],
  L:        int,
  M:        int,
) -> Tuple[complex, complex, complex]:
  '''Calculates values of (H_0, H_1, H_2) with L and M from given production amplitudes'''
  # Eqs. (152) to (154) assuming that rank is 1
  moments: List[complex] = 3 * [0 + 0j]
  for refl in (-1, +1):
    for wave1 in prodAmps[refl]:
      ell1:         int     = wave1[0]
      m1:           int     = wave1[1]
      prodAmp1:     complex = prodAmps[refl][wave1]
      prodAmp1NegM: complex = prodAmps[refl][(ell1, -m1)]
      for wave2 in prodAmps[refl]:
        ell2:         int     = wave2[0]
        m2:           int     = wave2[1]
        prodAmp2:     complex = prodAmps[refl][wave2]
        prodAmp2NegM: complex = prodAmps[refl][(ell2, -m2)]
        term = math.sqrt((2 * ell2 + 1) / (2 * ell1 + 1)) * (
            py3nj.clebsch_gordan(2 * ell2, 2 * L, 2 * ell1, 0,      0,     0,      ignore_invalid = True)  # (ell_2 0,    L 0 | ell_1 0  )
          * py3nj.clebsch_gordan(2 * ell2, 2 * L, 2 * ell1, 2 * m2, 2 * M, 2 * m1, ignore_invalid = True)  # (ell_2 m_2,  L M | ell_1 m_1)
        )
        moments[0] +=            term * (prodAmp1 * prodAmp2.conjugate()                + (-1)**(m1 - m2) * prodAmp1NegM * prodAmp2NegM.conjugate())  # H_0; Eq. (152)
        moments[1] +=            term * ((-1)**m1 * prodAmp1NegM * prodAmp2.conjugate() + (-1)**m2 * prodAmp1 * prodAmp2NegM.conjugate())             # H_1; Eq. (153)
        moments[2] += (0 + 1j) * term * ((-1)**m1 * prodAmp1NegM * prodAmp2.conjugate() - (-1)**m2 * prodAmp1 * prodAmp2NegM.conjugate())             # H_2; Eq. (154)
  return tuple(moments)


def getMaxSpin(prodAmps: Dict[int, Dict[Tuple[int, int,], complex]]) -> int:
  '''Gets maximum spin from set of production amplitudes'''
  maxSpin = 0
  for refl in (-1, +1):
    for wave in prodAmps[refl]:
      ell = wave[0]
      maxSpin = ell if ell > maxSpin else maxSpin
  return maxSpin


def calcAllMomentsFromWaves(prodAmps: Dict[int, Dict[Tuple[int, int,], complex]]) -> List[Tuple[complex, complex, complex]]:
  '''Calculates moments for given production amplitudes'''
  moments: List[Tuple[complex, complex, complex]] = []
  maxSpin = getMaxSpin(prodAmps)
  norm = 1
  index = 0
  for L in range(2 * maxSpin + 2):  # calculate moments 2 units above maximum L (should be zero)
    for M in range(L + 1):          # calculate moments 1 unit above maximum M (should be zero)
      momentSet: List[complex] = list(calcMomentSetFromWaves(prodAmps, L, M))
      if ((abs(momentSet[0].imag) > 1e-15) or (abs(momentSet[1].imag) > 1e-15) or (abs(momentSet[2].real) > 1e-15)):
        print(f"Warning: expect (Im[H_0({L} {M})], Im[H_1({L} {M})], Re[H_2({L} {M})]) = (0, 0, 0)  found ({momentSet[0].imag}, {momentSet[1].imag}, {momentSet[2].imag})")
      else:
        # set respective real and imaginary parts exactly to zero.
        momentSet[0] = momentSet[0].real + 0j
        momentSet[1] = momentSet[1].real + 0j
        momentSet[2] = 0 + momentSet[2].imag * 1j
      if (M == 0 and momentSet[2] != 0):
        print(f"Warning: expect H_2({L} {M}) = 0  found {momentSet[2].imag}")
      # normalize to H_0(0, 0)
      norm = momentSet[0] if index == 0 else norm
      moments.append(tuple(moment / norm for moment in momentSet))
      index += 1
  index = 0
  for L in range(2 * maxSpin + 2):
    for M in range(L + 1):
      print(f"(H_0({L} {M}), H_1({L} {M}), H_2({L} {M})) = {moments[index]}")
      index += 1
  return moments


def setupPlotStyle():
  #TODO remove dependency from external file or add file to repo
  ROOT.gROOT.LoadMacro("~/rootlogon.C")  # type: ignore
  ROOT.gROOT.ForceStyle()  # type: ignore
  ROOT.gStyle.SetCanvasDefW(600)  # type: ignore
  ROOT.gStyle.SetCanvasDefH(600)  # type: ignore
  ROOT.gStyle.SetPalette(ROOT.kBird)  # type: ignore
  # ROOT.gStyle.SetPalette(ROOT.kViridis)  # type: ignore
  ROOT.gStyle.SetLegendFillColor(ROOT.kWhite)  # type: ignore
  ROOT.gStyle.SetLegendBorderSize(1)  # type: ignore
  # ROOT.gStyle.SetOptStat("ni")  # type: ignore  # show only name and integral
  # ROOT.gStyle.SetOptStat("i")  # type: ignore  # show only integral
  ROOT.gStyle.SetOptStat("")  # type: ignore
  ROOT.gStyle.SetStatFormat("8.8g")  # type: ignore
  ROOT.gStyle.SetTitleColor(1, "X")  # type: ignore  # fix that for some mysterious reason x-axis titles of 2D plots and graphs are white
  ROOT.gStyle.SetTitleOffset(1.35, "Y")  # type: ignore


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore
  ROOT.gRandom.SetSeed(1234567890)  # type: ignore
  # ROOT.EnableImplicitMT(10)  # type: ignore
  setupPlotStyle()
  ROOT.gBenchmark.Start("Total execution time")  # type: ignore

  # get data
  nmbEvents = 1000
  nmbMcEvents = 100000
  # formulas for acceptance: x = cos(theta), y = phi in [-180, +180] deg
  acceptanceFormula = "1"  # acc_perfect

  # partial-wave decomposition
  inputMoments: List[Tuple[complex, complex, complex]] = calcAllMomentsFromWaves(PROD_AMPS)

  ROOT.gBenchmark.Show("Total execution time")  # type: ignore
