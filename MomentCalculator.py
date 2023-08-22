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


@dataclass
class AcceptanceIntegralMatrix:
  '''Calculates and provides access to acceptance integral matrix'''
  index:    MomentIndex # index mapping and iterators
  dataSet:  DataSet     # access to data samples
  IQnIndex: Optional[Dict[Tuple[int, int, int, int, int, int], complex]] = None  # integral matrix indexed by quantum numbers
  #TODO convert storage to NumPy array with flat index and make IQnIndex a property
  #TODO add possibility to write to/load from file

  def calculateMatrix(self) -> None:
    '''Calculates integral matrix of basis functions from (accepted) phase-space data'''
    # get phase-space data data as NumPy arrays
    thetas = self.dataSet.phaseSpaceData.AsNumpy(columns = ["theta"])["theta"]
    phis   = self.dataSet.phaseSpaceData.AsNumpy(columns = ["phi"]  )["phi"]
    Phis   = self.dataSet.phaseSpaceData.AsNumpy(columns = ["Phi"]  )["Phi"]
    nmbAccEvents = len(thetas)
    assert thetas.shape == (nmbAccEvents,) and thetas.shape == phis.shape == Phis.shape, (
      f"Not all NumPy arrays with input data have shape ({nmbAccEvents},): theta: {thetas.shape} vs. phi: {phis.shape} vs. Phi: {Phis.shape}")
    # calculate basis-function values for physical and measured moments; Eqs. (175) and (176); defined in `wignerD.C`
    fMeasValues: Dict[Tuple[int, int, int], npt.NDArray[np.complex128]] = {}
    fPhysValues: Dict[Tuple[int, int, int], npt.NDArray[np.complex128]] = {}
    for momentIndex in range(3):
      for L in range(self.index.maxL + 1):
        for M in range(L + 1):
          if momentIndex == 2 and M == 0:
            continue  # H_2(L, 0) are always zero and would lead to a singular acceptance integral matrix
          fMeasValues[(momentIndex, L, M)] = np.asarray(ROOT.f_meas(momentIndex, L, M, thetas, phis, Phis, self.dataSet.polarization))
          fPhysValues[(momentIndex, L, M)] = np.asarray(ROOT.f_phys(momentIndex, L, M, thetas, phis, Phis, self.dataSet.polarization))
    # calculate integral-matrix elements; Eq. (178)
    self.IQnIndex = {}
    for indices_meas, f_meas in fMeasValues.items():
      for indices_phys, f_phys in fPhysValues.items():
        self.IQnIndex[indices_meas + indices_phys] = 8 * math.pi**2 / self.dataSet.nmbGenEvents * np.dot(f_meas, f_phys)
