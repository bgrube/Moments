#!/usr/bin/env python3

import bidict as bd
from typing import TYPE_CHECKING

import MomentCalculator


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
  i = MomentCalculator.QnIndex(2, 5, 5)
  print(momentResult[i].val)
  print(momentResult[i].uncertRe)
  print(momentResult[i].uncertIm)
