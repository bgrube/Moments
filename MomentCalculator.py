#!/usr/bin/env python3


import bidict as bd
from dataclasses import dataclass
import math
import numpy as np
import numpy.typing as npt
from typing import Dict, Generator, Optional, Tuple

import ROOT


class MomentIndex:
  '''Provides mapping between moment index schemes and iterators for moment indices'''
  def __init__(
    self,
    maxL:      int,          # maximum L quantum number of moments
    photoProd: bool = True,  # switches between diffraction and photoproduction mode
  ) -> None:
    self.maxL = maxL
    # create new bidict subclass
    QnIndexByFlatIndexBidict = bd.namedbidict(typename = 'QnIndexByFlatIndexBidict', keyname = 'flatIndex', valname = 'QnIndex')
    # instantiate bidict subclass
    self.QnIndexByFlatIndex: bd.BidictBase[int, Tuple[int, int, int]] = QnIndexByFlatIndexBidict()
    flatIndex = 0
    for momentIndex in range(3 if photoProd else 1):
      for L in range(maxL + 1):
        for M in range(L + 1):
          if momentIndex == 2 and M == 0:
            continue  # H_2(L, 0) are always zero and would lead to a singular acceptance integral matrix
          QnIndex = (momentIndex, L, M)
          self.QnIndexByFlatIndex.QnIndex_for[flatIndex] = QnIndex
          flatIndex += 1

  def flatIndices(self) -> Generator[int, None, None]:
    '''Generates flat indices'''
    for flatIndex in range(len(self.QnIndexByFlatIndex)):
      yield flatIndex

  def QnIndices(self) -> Generator[Tuple[int, int, int], None, None]:
    '''Generates quantum number indices of the form (moment index, L, M)'''
    for flatIndex in range(len(self.QnIndexByFlatIndex)):
      yield self.QnIndexByFlatIndex.QnIndex_for[flatIndex]


@dataclass
class DataSet:
  '''Stores information about a single dataset'''
  polarization:   float            # photon-beam polarization
  data:           ROOT.RDataFrame  # data from which to calculate moments
  phaseSpaceData: ROOT.RDataFrame  # (accepted) phase-space data
  nmbGenEvents:   int              # number of generated events
