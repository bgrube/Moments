#!/usr/bin/env python3

import bidict as bd
from typing import TYPE_CHECKING

import MomentCalculator


if __name__ == "__main__":
  momentIndex = MomentCalculator.MomentIndex(maxL = 5, photoProd = True)
  for i in momentIndex.flatIndices():
    print(f"{i} : {momentIndex.indexMap.QnIndex_for[i]}")
  for i in momentIndex.QnIndices():
    print(f"{i} : {momentIndex.indexMap.flatIndex_for[i]}")
  if TYPE_CHECKING:
    a = bd.namedbidict(typename = 'QnIndexByFlatIndexBidict', keyname = 'flatIndex', valname = 'QnIndex')
    reveal_type(a)

  momentResult = MomentCalculator.MomentResult(momentIndex)
  print(momentResult._HFlatIndex)
  print(momentResult[0])
  print(momentResult[56])
  i = MomentCalculator.QnIndex(2, 5, 5)
  print(momentResult[i].val)
  print(momentResult[i].uncertRe)
  print(momentResult[i].uncertIm)
  print(momentResult[i].covReIm)
  if TYPE_CHECKING:
    reveal_type(momentResult._HFlatIndex)
    reveal_type(momentResult._HFlatIndex[0])
