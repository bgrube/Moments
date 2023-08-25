from __future__ import annotations

import bidict as bd
from dataclasses import dataclass, field
import numpy as np
import nptyping as npt
from typing import (
  overload,
  Any,
  Generator,
  List,
  Optional,
  Tuple,
  Union,
)
#TODO switch from int for indices to SupportsIndex; but this requires Python 3.8+

import ROOT


@dataclass(frozen = True)  # immutable
class QnIndex:
  '''Stores information about quantum-number indices of moments'''
  momentIndex: int  # subscript of photoproduction moments
  L:           int  # angular momentum
  M:           int  # projection quantum number of L


@dataclass
class MomentIndices:
  '''Provides mapping between moment index schemes and iterators for moment indices'''
  maxL:      int          # maximum L quantum number of moments
  photoProd: bool = True  # switches between diffraction and photoproduction mode
  indexMap:  bd.BidictBase[int, QnIndex] = field(init = False)  # bidirectional map for flat index <-> quantum-number index conversion

  def __post_init__(self) -> None:
    # create new bidict subclass
    QnIndexByFlatIndexBidict = bd.namedbidict(typename = 'QnIndexByFlatIndexBidict', keyname = 'flatIndex', valname = 'QnIndex')
    # instantiate bidict subclass
    self.indexMap = QnIndexByFlatIndexBidict()
    flatIndex = 0
    for momentIndex in range(3 if self.photoProd else 1):
      for L in range(self.maxL + 1):
        for M in range(L + 1):
          if momentIndex == 2 and M == 0:
            continue  # H_2(L, 0) are always zero and would lead to a singular acceptance integral matrix
          self.indexMap[flatIndex] = QnIndex(momentIndex, L, M)
          flatIndex += 1

  def __len__(self) -> int:
    '''Returns total number of moments'''
    return len(self.indexMap)

  def __getitem__(
    self,
    subscript: int,
  ) -> QnIndex:
    '''Returns QnIndex that correspond to given flat index'''
    return self.indexMap[subscript]

  def flatIndices(self) -> Generator[int, None, None]:
    '''Generates flat indices'''
    for flatIndex in range(len(self)):
      yield flatIndex

  def QnIndices(self) -> Generator[QnIndex, None, None]:
    '''Generates quantum-number indices of the form QnIndex(moment index, L, M)'''
    for flatIndex in range(len(self)):
      yield self[flatIndex]


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
  indices:     MomentIndices  # index mapping and iterators
  dataSet:     DataSet        # info on data samples
  _IFlatIndex: Optional[npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]] = None  # integral matrix with flat indices

  @overload
  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnIndex], Union[int, QnIndex]],
  ) -> Optional[complex]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[slice, Union[int, QnIndex]],
  ) -> Optional[npt.NDArray[npt.Shape["*"], npt.Complex128]]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnIndex], slice],
  ) -> Optional[npt.NDArray[npt.Shape["*"], npt.Complex128]]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[slice, slice],
  ) -> Optional[npt.NDArray[npt.Shape["*, *"], npt.Complex128]]: ...

  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnIndex, slice], Union[int, QnIndex, slice]],
  ) -> Optional[Union[complex, npt.NDArray[npt.Shape["*"], npt.Complex128], npt.NDArray[npt.Shape["*, *"], npt.Complex128]]]:
    '''Returns integral matrix elements for any combination of flat and quantum-number indices'''
    if self._IFlatIndex is None:
      return None
    else:
      # turn quantum-number indices to flat indices
      flatIndexMeas: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript[0]] if isinstance(subscript[0], QnIndex) else subscript[0]
      flatIndexPhys: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript[1]] if isinstance(subscript[1], QnIndex) else subscript[1]
      return self._IFlatIndex[flatIndexMeas, flatIndexPhys]

  def __str__(self) -> str:
    if self._IFlatIndex is None:
      return "None"
    else:
      return np.array2string(self._IFlatIndex, precision = 3, suppress_small = True, max_line_width = 150)

  def calculate(self) -> None:
    '''Calculates integral matrix of basis functions from (accepted) phase-space data'''
    # get phase-space data data as NumPy arrays
    thetas = self.dataSet.phaseSpaceData.AsNumpy(columns = ["theta"])["theta"]
    phis   = self.dataSet.phaseSpaceData.AsNumpy(columns = ["phi"]  )["phi"]
    Phis   = self.dataSet.phaseSpaceData.AsNumpy(columns = ["Phi"]  )["Phi"]
    print(f"Phase-space data column: {type(thetas)}; {thetas.shape}; {thetas.dtype}; {thetas.dtype.type}")
    nmbAccEvents = len(thetas)
    assert thetas.shape == (nmbAccEvents,) and thetas.shape == phis.shape == Phis.shape, (
      f"Not all NumPy arrays with input data have the correct shape. Expected ({nmbAccEvents},) but got theta: {thetas.shape}, phi: {phis.shape}, and Phi: {Phis.shape}")
    # calculate basis-function values for physical and measured moments; Eqs. (175) and (176); defined in `wignerD.C`
    nmbMoments = len(self.indices)
    fMeas: npt.NDArray[npt.Shape["*"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    fPhys: npt.NDArray[npt.Shape["*"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    for flatIndex in self.indices.flatIndices():
      qnIndex = self.indices[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, self.dataSet.polarization))
      fPhys[flatIndex] = np.asarray(ROOT.f_phys(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, self.dataSet.polarization))
    # calculate integral-matrix elements; Eq. (178)
    self._IFlatIndex = np.empty((nmbMoments, nmbMoments), dtype = np.complex128)
    for flatIndexMeas in self.indices.flatIndices():
      for flatIndexPhys in self.indices.flatIndices():
        self._IFlatIndex[flatIndexMeas, flatIndexPhys] = (8 * np.pi**2 / self.dataSet.nmbGenEvents) * np.dot(fMeas[flatIndexMeas], fPhys[flatIndexPhys])

  def isValid(self) -> bool:
    return (self._IFlatIndex is not None) and self._IFlatIndex.shape == (len(self.indices), len(self.indices))

  def save(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    '''Saves NumPy array that holds the integral matrix to file with given name'''
    if self._IFlatIndex is not None:
      print(f"Saving integral matrix to file '{fileName}'.")
      np.save(fileName, self._IFlatIndex)

  def load(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    '''Loads NumPy array that holds the integral matrix from file with given name'''
    print(f"Loading integral matrix from file '{fileName}'.")
    array = np.load(fileName)
    if not array.shape == (len(self.indices), len(self.indices)):
      raise IndexError(f"Integral loaded from file '{fileName}' has wrong shape. Expected {(len(self.indices), len(self.indices))} but got {array.shape}.")
    self._IFlatIndex = array

  def loadOrCalculate(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    '''Loads NumPy array that holds the integral matrix from file with given name; and calculates the integral matrix if loading failed'''
    try:
      self.load(fileName)
    except Exception as e:
      print(f"Could not load integral matrix from file '{fileName}': {e} Calculating matrix instead.")
      self.calculate()


@dataclass
class MomentValue:
  '''Stores and provides access to single moment value'''
  qn:       QnIndex   # quantum numbers
  val:      complex   # moment value
  uncertRe: float     # uncertainty of real part
  uncertIm: float     # uncertainty of imaginary part
  label:    str = ""  # label used for printing

  def __str__(self) -> str:
    result = ""
    momentSymbol = f"H{'^' + self.label if self.label else ''}_{self.qn.momentIndex}(L = {self.qn.L}, M = {self.qn.M})"
    result += (f"Re[{momentSymbol}] = {self.val.real} +- {self.uncertRe}\n"
               f"Im[{momentSymbol}] = {self.val.imag} +- {self.uncertIm}")
    return result


@dataclass
class MomentResult:
  '''Stores and provides access to moment values'''
  indices:           MomentIndices  # index mapping and iterators
  label:             str = ""       # label used for printing
  _valsFlatIndex:    npt.NDArray[npt.Shape["*"], npt.Complex128]     = field(init = False)  # flat array with moment values
  _covReReFlatIndex: npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64] = field(init = False)  # covariance matrix of real parts of moment values with flat indices
  _covImImFlatIndex: npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64] = field(init = False)  # covariance matrix of imaginary parts of moment values with flat indices
  _covReImFlatIndex: npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64] = field(init = False)  # covariance matrix of real and imaginary parts of moment values with flat indices

  def __post_init__(self) -> None:
    nmbMoments = len(self.indices)
    self._valsFlatIndex    = np.zeros((nmbMoments, ), dtype = npt.Complex128)
    self._covReReFlatIndex = np.zeros((nmbMoments, nmbMoments), dtype = npt.Float64)
    self._covImImFlatIndex = np.zeros((nmbMoments, nmbMoments), dtype = npt.Float64)
    self._covReImFlatIndex = np.zeros((nmbMoments, nmbMoments), dtype = npt.Float64)

  @overload
  def __getitem__(
    self,
    subscript: Union[int, QnIndex],
  ) -> MomentValue: ...

  @overload
  def __getitem__(
    self,
    subscript: slice,
  ) -> List[MomentValue]: ...

  def __getitem__(
    self,
    subscript: Union[int, QnIndex, slice],
  ) -> Union[MomentValue, List[MomentValue]]:
    '''Returns moment value and corresponding uncertainties at the given flat or quantum-number index'''
    # turn quantum-number index to flat index
    flatIndex: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript] if isinstance(subscript, QnIndex) else subscript
    if isinstance(flatIndex, slice):
      return [
        MomentValue(
          qn       = self.indices[i],
          val      = self._valsFlatIndex[i],
          uncertRe = np.sqrt(self._covReReFlatIndex[i, i]),
          uncertIm = np.sqrt(self._covImImFlatIndex[i, i]),
          label    = self.label,
        ) for i in range(*flatIndex.indices(len(self.indices)))
      ]
    elif isinstance(flatIndex, int):
      return MomentValue(
        qn       = self.indices[flatIndex],
        val      = self._valsFlatIndex[flatIndex],
        uncertRe = np.sqrt(self._covReReFlatIndex[flatIndex, flatIndex]),
        uncertIm = np.sqrt(self._covImImFlatIndex[flatIndex, flatIndex]),
        label    = self.label,
      )
    else:
      raise TypeError(f"Invalid subscript type {type(flatIndex)}.")

  def __str__(self) -> str:
    result = (str(self[flatIndex]) for flatIndex in self.indices.flatIndices())
    return "\n".join(result)

  def copyFrom(
    self,
    other: MomentResult,  # instance from which data are copied
  ) -> None:
    '''Copies all values from given MomentResult instance but leaves `label` untouched'''
    self.indices           = other.indices
    self._valsFlatIndex    = other._valsFlatIndex
    self._covReReFlatIndex = other._covReReFlatIndex
    self._covImImFlatIndex = other._covImImFlatIndex
    self._covReImFlatIndex = other._covReImFlatIndex


@dataclass
class MomentCalculator:
  '''Calculates and provides access to moments'''
  indices:        MomentIndices  # index mapping and iterators
  dataSet:        DataSet        # info on data samples
  integralMatrix: Optional[AcceptanceIntegralMatrix] = None  # acceptance integral matrix
  HMeas:          Optional[MomentResult]             = None  # calculated measured moments
  HPhys:          Optional[MomentResult]             = None  # calculated physical moments

  def _calcReImCovMatrices(
    self,
    V_aug: npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128],  # augmentented covariance matrix
  ) -> Tuple[npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64], npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64], npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64]]:
    '''Calculates covariance matrices for real parts, for imaginary parts, and for real and imaginary parts from augmented covariance matrix'''
    nmbMoments = len(self.indices)
    V_Hermit = V_aug[:nmbMoments, :nmbMoments]  # Hermitian covariance matrix; Eq. (88)
    V_pseudo = V_aug[:nmbMoments, nmbMoments:]  # pseudo-covariance matrix; Eq. (88)
    V_ReRe = (np.real(V_Hermit) + np.real(V_pseudo)) / 2  # Eq. (91)
    V_ImIm = (np.real(V_Hermit) - np.real(V_pseudo)) / 2  # Eq. (92)
    V_ReIm = (np.imag(V_pseudo) - np.imag(V_Hermit)) / 2  # Eq. (93)
    return (V_ReRe, V_ImIm, V_ReIm)

  def calculate(self) -> None:
    '''Calculates photoproduction moments and their covariances'''
    # get input data as NumPy arrays
    thetas = self.dataSet.data.AsNumpy(columns = ["theta"])["theta"]
    phis   = self.dataSet.data.AsNumpy(columns = ["phi"]  )["phi"]
    Phis   = self.dataSet.data.AsNumpy(columns = ["Phi"]  )["Phi"]
    print(f"Input data column: {type(thetas)}; {thetas.shape}; {thetas.dtype}; {thetas.dtype.type}")
    nmbEvents = len(thetas)
    assert thetas.shape == (nmbEvents,) and thetas.shape == phis.shape == Phis.shape, (
      f"Not all NumPy arrays with input data have the correct shape. Expected ({nmbEvents},) but got theta: {thetas.shape}, phi: {phis.shape}, and Phi: {Phis.shape}")
    # get number of moments (the poor-man's way)
    nmbMoments = len(self.indices)
    # calculate basis-function values and values of measured moments
    fMeas  = np.empty((nmbMoments, nmbEvents), dtype = npt.Complex128)
    self.HMeas = MomentResult(self.indices, label = "meas")
    for flatIndex in self.indices.flatIndices():
      qnIndex = self.indices[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, self.dataSet.polarization))  # Eq. (176)
      self.HMeas._valsFlatIndex[flatIndex] = 2 * np.pi * np.sum(fMeas[flatIndex])  # Eq. (179)
    # calculate covariances; Eqs. (88), (180), and (181)
    V_meas_aug = (2 * np.pi)**2 * nmbEvents * np.cov(fMeas, np.conjugate(fMeas))  # augmented covariance matrix
    self.HMeas._covReReFlatIndex, self.HMeas._covImImFlatIndex, self.HMeas._covReImFlatIndex = self._calcReImCovMatrices(V_meas_aug)
    print(self.HMeas)
    self.HPhys = MomentResult(self.indices, label = "phys")
    V_phys_aug = np.empty(V_meas_aug.shape, dtype = npt.Complex128)
    if self.integralMatrix is None:
      # ideal detector: physical moments are identical to measured moments
      self.HPhys._valsFlatIndex = self.HMeas._valsFlatIndex
      V_phys_aug = V_meas_aug
    else:
      # get acceptance integral matrix
      assert self.integralMatrix._IFlatIndex is not None, "Integral matrix is None."
      #TODO move to user code
      print(f"Acceptance integral matrix = \n{self.integralMatrix}")
      I_acc: npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128] = self.integralMatrix._IFlatIndex
      eigenVals, eigenVecs = np.linalg.eig(I_acc)
      print(f"I_acc eigenvalues = {eigenVals}")
      # print(f"I_acc eigenvectors = {eigenVecs}")
      # print(f"I_acc determinant = {np.linalg.det(I_acc)}")
      # print(f"I_acc = \n{np.array2string(I_acc, precision = 3, suppress_small = True, max_line_width = 150)}")
      #TODO move to utilities module
      # plotComplexMatrix(I_acc, fileNamePrefix = "I_acc")
      I_inv = np.linalg.inv(I_acc)
      # eigenVals, eigenVecs = np.linalg.eig(I_inv)
      # print(f"I^-1 eigenvalues = {eigenVals}")
      # print(f"I^-1 = \n{np.array2string(I_inv, precision = 3, suppress_small = True, max_line_width = 150)}")
      # plotComplexMatrix(I_inv, fileNamePrefix = "I_inv")
      # calculate physical moments, i.e. correct for detection efficiency
      self.HPhys._valsFlatIndex = I_inv @ self.HMeas._valsFlatIndex  # Eq. (83)
      # perform linear uncertainty propagation
      J = I_inv  # Jacobian of efficiency correction; Eq. (101)
      J_conj = np.zeros((nmbMoments, nmbMoments), dtype = npt.Complex128)  # conjugate Jacobian; Eq. (101)
      J_aug = np.block([
        [J,                    J_conj],
        [np.conjugate(J_conj), np.conjugate(J)],
      ])  # augmented Jacobian; Eq. (98)
      V_phys_aug = J_aug @ (V_meas_aug @ np.asmatrix(J_aug).H)  #!Note! @ is left-associative; Eq. (85)
    # normalize moments such that H_0(0, 0) = 1
    norm = self.HPhys[0].val
    self.HPhys._valsFlatIndex /= norm
    V_phys_aug /= norm**2
    self.HPhys._covReReFlatIndex, self.HPhys._covImImFlatIndex, self.HPhys._covReImFlatIndex = self._calcReImCovMatrices(V_phys_aug)
