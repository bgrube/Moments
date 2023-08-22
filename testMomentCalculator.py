#!/usr/bin/env python3


import MomentCalculator


if __name__ == "__main__":
  momentIndex = MomentCalculator.MomentIndex(5)
  for i in momentIndex.flatIndices():
    print(f"{i} : {momentIndex.QnIndexByFlatIndex.QnIndex_for[i]}")
  for i in momentIndex.QnIndices():
    print(f"{i} : {momentIndex.QnIndexByFlatIndex.flatIndex_for[i]}")
