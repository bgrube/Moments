#!/usr/bin/env python3


import MomentCalculator


if __name__ == "__main__":
  momentIndex = MomentCalculator.MomentIndex(maxL = 5, photoProd = True)
  for i in momentIndex.flatIndices():
    print(f"{i} : {momentIndex.indexMap.QnIndex_for[i]}")
  for i in momentIndex.QnIndices():
    print(f"{i} : {momentIndex.indexMap.flatIndex_for[i]}")
