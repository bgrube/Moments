#!/usr/bin/env python3

from __future__ import annotations

import bidict as bd
import copy
import functools
from typing import TYPE_CHECKING

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  MomentIndices,
  MomentResult,
  QnMomentIndex,
  QnWaveIndex,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
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

  print(f"MomentResult for polarized moments:")
  momentResult = MomentResult(MomentIndices(maxL = 5, polarized = True))
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
    H:  MomentResult = ampSet.photoProdMomentSet(maxL = 4, normalize = False, printMomentFormulas = True)
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
