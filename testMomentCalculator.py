#!/usr/bin/env python3

import bidict as bd
import functools
from typing import TYPE_CHECKING

import MomentCalculator


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  momentIndices = MomentCalculator.MomentIndices(maxL = 5, photoProd = True)
  for i in momentIndices.flatIndices():
    print(f"{i} : {momentIndices.indexMap.QnIndex_for[i]}")
  for i in momentIndices.QnIndices():
    print(f"{i} : {momentIndices.indexMap.flatIndex_for[i]}")
  print(f"{type(momentIndices.indexMap[0])}, {momentIndices.indexMap[0]} vs. {momentIndices[0]}")
  print(f"__str__ = '{momentIndices}' vs. __repr__ = '{repr(momentIndices)}'")
  momentIndices2 = MomentCalculator.MomentIndices(maxL = 5, photoProd = True)
  print(f"equality = {momentIndices == momentIndices2}")
  if TYPE_CHECKING:
    a = bd.namedbidict(typename = 'QnIndexByFlatIndexBidict', keyname = 'flatIndex', valname = 'QnIndex')
    reveal_type(a)

  momentResult = MomentCalculator.MomentResult(momentIndices)
  print(momentResult)
  print(momentResult._valsFlatIndex)
  print(momentResult[0])
  print(momentResult[56])
  i = MomentCalculator.QnMomentIndex(2, 5, 5)
  print(momentResult[i].val)
  print(momentResult[i].uncertRe)
  print(momentResult[i].uncertIm)

  amps = [
    #                                                           refl J   M    amplitude
    # negative-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 0,  0),  1.0 + 0.0j),  # S_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 1, -1), -0.4 + 0.1j),  # P_-1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 1,  0),  0.3 - 0.8j),  # P_0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 1, +1), -0.8 + 0.7j),  # P_+1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2, -2),  0.1 - 0.4j),  # D_-2^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2, -1),  0.5 + 0.2j),  # D_-1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2,  0), -0.1 - 0.2j),  # D_ 0^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2, +1),  0.2 - 0.1j),  # D_+1^-
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2, +2), -0.2 + 0.3j),  # D_+2^-
    # positive-reflectivity waves
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 0,  0),  0.5 + 0.0j),  # S_0^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 1, -1),  0.5 - 0.1j),  # P_-1^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 1,  0), -0.8 - 0.3j),  # P_0^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 1, +1),  0.6 + 0.3j),  # P_+1^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2, -2),  0.2 + 0.1j),  # D_-2^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2, -1),  0.2 - 0.3j),  # D_-1^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2,  0),  0.1 - 0.2j),  # D_ 0^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2, +1),  0.2 + 0.5j),  # D_+1^+
    MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2, +2), -0.3 - 0.1j),  # D_+2^+
  ]
  ampSet = MomentCalculator.AmplitudeSet(amps)
  qnIndex = MomentCalculator.QnWaveIndex(+1, 2, +2)
  print(f"ampSet[{qnIndex}] = {ampSet[qnIndex]}")
  ampSet[qnIndex] = 100 + 100j
  print(f"ampSet[{qnIndex}] = {ampSet[qnIndex]}")
  for amp in ampSet.amplitudes():
    print(amp)
  print(f"getMaxSpin() = {ampSet.maxSpin()}")

  for refl in (-1, +1):
    for amp1 in ampSet.amplitudes(onlyRefl = refl):
      l1 = amp1.qn.l
      m1 = amp1.qn.m
      for amp2 in ampSet.amplitudes(onlyRefl = refl):
        l2 = amp2.qn.l
        m2 = amp2.qn.m
        rhos = ampSet.photoProdSpinDensElements(refl, l1, l2, m1, m2)
        print(f"rho {refl}; ({l1}, {m1}); ({l2}, {m2}) = {rhos}")
