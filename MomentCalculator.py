#!/usr/bin/env python3


import bidict as bd
from dataclasses import dataclass
import math
import numpy as np
import numpy.typing as npt
from typing import Dict, Generator, Optional, Tuple

import ROOT


#TODO add dataclass for Qn Index


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

  @property
  def nmbMoments(self) -> int:
    '''Returns total number of moments'''
    return len(self.QnIndexByFlatIndex)

  def flatIndices(self) -> Generator[int, None, None]:
    '''Generates flat indices'''
    for flatIndex in range(self.nmbMoments):
      yield flatIndex

  def QnIndices(self) -> Generator[Tuple[int, int, int], None, None]:
    '''Generates quantum number indices of the form (moment index, L, M)'''
    for flatIndex in range(self.nmbMoments):
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
  index:      MomentIndex  # index mapping and iterators
  dataSet:    DataSet      # access to data samples
  IFlatIndex: Optional[npt.NDArray[np.complex128]] = None  # integral matrix with flat indices
  #TODO add possibility to write to/load from file
  #TODO add [] accessor for flat and Qn indices

  def calculateMatrix(self) -> None:
    '''Calculates integral matrix of basis functions from (accepted) phase-space data'''
    # get phase-space data data as NumPy arrays
    thetas = self.dataSet.phaseSpaceData.AsNumpy(columns = ["theta"])["theta"]
    phis   = self.dataSet.phaseSpaceData.AsNumpy(columns = ["phi"]  )["phi"]
    Phis   = self.dataSet.phaseSpaceData.AsNumpy(columns = ["Phi"]  )["Phi"]
    nmbAccEvents = len(thetas)
    nmbMoments   = self.index.nmbMoments
    assert thetas.shape == (nmbAccEvents,) and thetas.shape == phis.shape == Phis.shape, (
      f"Not all NumPy arrays with input data have shape ({nmbAccEvents},): theta: {thetas.shape} vs. phi: {phis.shape} vs. Phi: {Phis.shape}")
    # calculate basis-function values for physical and measured moments; Eqs. (175) and (176); defined in `wignerD.C`
    fMeas: npt.NDArray[np.complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    fPhys: npt.NDArray[np.complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    for flatIndex in self.index.flatIndices():
      momentIndex, L, M = self.index.QnIndexByFlatIndex.QnIndex_for[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(momentIndex, L, M, thetas, phis, Phis, self.dataSet.polarization))
      fPhys[flatIndex] = np.asarray(ROOT.f_phys(momentIndex, L, M, thetas, phis, Phis, self.dataSet.polarization))
    # calculate integral-matrix elements; Eq. (178)
    self.IFlatIndex = np.empty((nmbMoments, nmbMoments), dtype = np.complex128)
    for flatIndexMeas in self.index.flatIndices():
      for flatIndexPhys in self.index.flatIndices():
        self.IFlatIndex[flatIndexMeas, flatIndexPhys] = 8 * math.pi**2 / self.dataSet.nmbGenEvents * np.dot(fMeas[flatIndexMeas], fPhys[flatIndexPhys])

  def IQnIndex(
    self,
    QnIndexMeas: Tuple[int, int, int],
    QnIndexPhys: Tuple[int, int, int],
  ) -> Optional[complex]:
    '''Returns integral matrix elements for quantum-number indices'''
    if self.IFlatIndex is None:
      return None
    else:
      flatIndexMeas: int = self.index.QnIndexByFlatIndex.flatIndex_for[QnIndexMeas]
      flatIndexPhys: int = self.index.QnIndexByFlatIndex.flatIndex_for[QnIndexPhys]
      return self.IFlatIndex[flatIndexMeas, flatIndexPhys]
