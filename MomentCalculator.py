#!/usr/bin/env python3


import bidict as bd
from dataclasses import dataclass
import math
import numbers
import numpy as np
import nptyping as npt
from typing import Dict, Generator, Optional, Tuple, Union

import ROOT


@dataclass(frozen = True)  # immutable
class QnIndex:
  '''Stores information about quantum-number indices of moments'''
  momentIndex: int  # subscript of photoproduction moments
  L:           int  # angular momentum
  M:           int  # projection quantum number of L


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
    self.indexMap: bd.BidictBase[int, QnIndex] = QnIndexByFlatIndexBidict()
    flatIndex = 0
    for momentIndex in range(3 if photoProd else 1):
      for L in range(maxL + 1):
        for M in range(L + 1):
          if momentIndex == 2 and M == 0:
            continue  # H_2(L, 0) are always zero and would lead to a singular acceptance integral matrix
          self.indexMap.QnIndex_for[flatIndex] = QnIndex(momentIndex, L, M)
          flatIndex += 1

  @property
  def nmbMoments(self) -> int:
    '''Returns total number of moments'''
    return len(self.indexMap)

  def flatIndices(self) -> Generator[int, None, None]:
    '''Generates flat indices'''
    for flatIndex in range(self.nmbMoments):
      yield flatIndex

  def QnIndices(self) -> Generator[QnIndex, None, None]:
    '''Generates quantum-number indices of the form QnIndex(moment index, L, M)'''
    for flatIndex in range(self.nmbMoments):
      yield self.indexMap.QnIndex_for[flatIndex]


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
  _IFlatIndex: Optional[npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]] = None  # integral matrix with flat indices

  def __getitem__(
    self,
    indices: Tuple[Union[numbers.Integral, QnIndex, slice], Union[numbers.Integral, QnIndex, slice]],
  ) -> Optional[Union[complex, npt.NDArray[npt.Shape["*"], npt.Complex128], npt.NDArray[npt.Shape["*, *"], npt.Complex128]]]:
    '''Returns integral matrix elements for any combination of flat and quantum-number indices'''
    if self._IFlatIndex is None:
      return None
    else:
      # turn quantum-number indices to flat indices
      flatIndexMeas: Union[numbers.Integral, slice] = self.index.indexMap.flatIndex_for[indices[0]] if isinstance(indices[0], QnIndex) else indices[0]
      flatIndexPhys: Union[numbers.Integral, slice] = self.index.indexMap.flatIndex_for[indices[1]] if isinstance(indices[1], QnIndex) else indices[1]
      return self._IFlatIndex[flatIndexMeas, flatIndexPhys]

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
    nmbMoments = self.index.nmbMoments
    fMeas: npt.NDArray[npt.Shape["*"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    fPhys: npt.NDArray[npt.Shape["*"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    for flatIndex in self.index.flatIndices():
      qnIndex = self.index.indexMap.QnIndex_for[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, self.dataSet.polarization))
      fPhys[flatIndex] = np.asarray(ROOT.f_phys(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, self.dataSet.polarization))
    # calculate integral-matrix elements; Eq. (178)
    self._IFlatIndex = np.empty((nmbMoments, nmbMoments), dtype = np.complex128)
    for flatIndexMeas in self.index.flatIndices():
      for flatIndexPhys in self.index.flatIndices():
        self._IFlatIndex[flatIndexMeas, flatIndexPhys] = (8 * math.pi**2 / self.dataSet.nmbGenEvents) * np.dot(fMeas[flatIndexMeas], fPhys[flatIndexPhys])

  def isValid(self) -> bool:
    return (self._IFlatIndex is not None) and self._IFlatIndex.shape == (self.index.nmbMoments, self.index.nmbMoments)

  def saveMatrix(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    '''Saves NumPy array that holds the integral matrix to file with given name'''
    if self._IFlatIndex is not None:
      print(f"Saving integral matrix to file '{fileName}'.")
      np.save(fileName, self._IFlatIndex)

  def loadMatrix(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    '''Loads NumPy array that holds the integral matrix from file with given name'''
    print(f"Loading integral matrix from file '{fileName}'.")
    array = np.load(fileName)
    if not self.isValid():
      raise IndexError(f"Integral loaded from file '{fileName}' has wrong shape. Expected {(self.index.nmbMoments, self.index.nmbMoments)}, got {array.shape}.")
    self._IFlatIndex = array

  def loadOrCalculateMatrix(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    '''Loads NumPy array that holds the integral matrix from file with given name; and calculates the integral matrix if loading failed'''
    try:
      self.loadMatrix(fileName)
    except:
      print(f"Could not load integral matrix from file '{fileName}'; calculating matrix instead.")
      self.calculateMatrix()
