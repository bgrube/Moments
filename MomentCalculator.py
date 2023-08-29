"""Module that provides classes and functions to calculate moments for photoproduction of two-(pseudo)scalar mesons"""
# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3

from __future__ import annotations

import bidict as bd
from dataclasses import dataclass, field, fields, InitVar
import numpy as np
import nptyping as npt
from typing import (
  overload,
  Any,
  Dict,
  Generator,
  List,
  Optional,
  Sequence,
  Tuple,
  Union,
)
#TODO switch from int for indices to SupportsIndex; but this requires Python 3.8+

import py3nj
import ROOT


@dataclass(frozen = True)  # immutable
class QnWaveIndex:
  """Stores information about quantum-number indices of two-pseudocalar partial-waves"""
  refl: int  # reflectivity
  l:    int  # orbital angular momentum
  m:    int  # projection quantum number of l


@dataclass
class AmplitudeValue:
  """Stores and provides access to single amplitude value"""
  qn:  QnWaveIndex  # quantum numbers
  val: complex      # amplitude value


@dataclass
class AmplitudeSet:
  """Stores partial-wave amplitudes and makes them accessible by quantum numbers"""
  amps: InitVar[Sequence[AmplitudeValue]]
  _amps: Tuple[Dict[Tuple[int, int], complex], Dict[Tuple[int, int], complex]] = field(init = False)  # internal storage for amplitudes split by positive and negative reflectivity

  def __post_init__(
    self,
    amps: Sequence[AmplitudeValue],
  ) -> None:
    """Constructs object from list"""
    self._amps = ({}, {})
    for amp in amps:
      self[amp.qn] = amp.val

  def __getitem__(
    self,
    subscript: QnWaveIndex,
  ) -> AmplitudeValue:
    """Returns partial-wave amplitude for given quantum numbers; returns 0 for non-existing amplitudes"""
    assert abs(subscript.refl) == 1, f"Reflectivity quantum number can only be +-1; got {subscript.refl}."
    reflIndex = 0 if subscript.refl == +1 else 1
    return AmplitudeValue(subscript, self._amps[reflIndex].get((subscript.l, subscript.m), 0))

  def __setitem__(
    self,
    subscript: QnWaveIndex,
    amp:       complex,
  ) -> None:
    """Returns partial-wave amplitude for given quantum numbers"""
    assert abs(subscript.refl) == 1, f"Reflectivity quantum number can only be +-1; got {subscript.refl}."
    reflIndex = 0 if subscript.refl == +1 else 1
    self._amps[reflIndex][(subscript.l, subscript.m)] = amp

  def amplitudes(
    self,
    onlyRefl: Optional[int] = None,  # if set to +-1 only waves with the corresponding reflectivities
  ) -> Generator[AmplitudeValue, None, None]:
    """Generates amplitude values; optionally filtered by reflectivity"""
    assert onlyRefl is None or abs(onlyRefl) == 1, f"Invalid reflectivity value f{onlyRefl}; expect +1, -1, or None"
    reflIndices = (0, 1)
    if onlyRefl == +1:
      reflIndices = (0, )
    elif onlyRefl == -1:
      reflIndices = (1, )
    for reflIndex in reflIndices:
      for (l, m), val in self._amps[reflIndex].items():
        yield AmplitudeValue(QnWaveIndex(+1 if reflIndex == 0 else -1, l, m), self._amps[reflIndex][(l, m)])

  def maxSpin(self) -> int:
    """Returns maximum spin of wave set ignoring 0 amplitudes"""
    maxSpin = 0
    for amp in self.amplitudes():
      l = amp.qn.l
      if amp.val != 0:
        maxSpin = max(l, maxSpin)
    return maxSpin

  def spinDensElementSet(
    self,
    refl: int,  # reflectivity
    l1:   int,  # l
    l2:   int,  # l'
    m1:   int,  # m
    m2:   int,  # m'
  ) -> Tuple[complex, complex, complex]:
    """Returns elements of spin-density matrix components (0^rho^ll'_mm', 1^rho^ll'_mm', 2^rho^ll'_mm') with given quantum numbers calculated from partial-wave amplitudes assuming rank 1"""
    qn1     = QnWaveIndex(refl, l1,  m1)
    qn1NegM = QnWaveIndex(refl, l1, -m1)
    qn2     = QnWaveIndex(refl, l2,  m2)
    qn2NegM = QnWaveIndex(refl, l2, -m2)
    rhos: List[complex] = 3 * [0 + 0j]
    rhos[0] =                    (           self[qn1    ].val * self[qn2].val.conjugate() + (-1)**(m1 - m2) * self[qn1NegM].val * self[qn2NegM].val.conjugate())  # Eq. (150)
    rhos[1] =            -refl * ((-1)**m1 * self[qn1NegM].val * self[qn2].val.conjugate() + (-1)**m2        * self[qn1    ].val * self[qn2NegM].val.conjugate())  # Eq. (151)
    rhos[2] = -(0 + 1j) * refl * ((-1)**m1 * self[qn1NegM].val * self[qn2].val.conjugate() - (-1)**m2        * self[qn1    ].val * self[qn2NegM].val.conjugate())  # Eq. (152)
    return (rhos[0], rhos[1], rhos[2])

  def momentSet(
    self,
    L: int,  # angular momentum
    M: int,  # projection quantum number of L
  ) -> Tuple[complex, complex, complex]:
    """Returns moments (H_0, H_1, H_2) with given quantum numbers calculated from partial-wave amplitudes assuming rank 1"""
    # Eqs. (154) to (156) assuming that rank is 1
    moments: List[complex] = 3 * [0 + 0j]
    for refl in (-1, +1):
      for amp1 in self.amplitudes(onlyRefl = refl):
        l1 = amp1.qn.l
        m1 = amp1.qn.m
        for amp2 in self.amplitudes(onlyRefl = refl):
          l2 = amp2.qn.l
          m2 = amp2.qn.m
          term = np.sqrt((2 * l2 + 1) / (2 * l1 + 1)) * (
              py3nj.clebsch_gordan(2 * l2, 2 * L, 2 * l1, 0,      0,     0,      ignore_invalid = True)  # (l_2 0,    L 0 | l_1 0  )
            * py3nj.clebsch_gordan(2 * l2, 2 * L, 2 * l1, 2 * m2, 2 * M, 2 * m1, ignore_invalid = True)  # (l_2 m_2,  L M | l_1 m_1)
          )
          if term == 0:  # unphysical Clebsch-Gordan
            continue
          rhos: Tuple[complex, complex, complex] = self.spinDensElementSet(refl, l1, l2, m1, m2)
          moments[0] +=  term * rhos[0]  # H_0; Eq. (124)
          moments[1] += -term * rhos[1]  # H_1; Eq. (125)
          moments[2] += -term * rhos[2]  # H_2; Eq. (125)
    return (moments[0], moments[1], moments[2])

  def allMoments(
    self,
    maxL: int,  # maximum L quantum number of moments
  ) -> MomentResult:
    """Returns moments calculated from partial-wave amplitudes assuming rank 1; the H_2(L, 0) are omitted"""
    momentIndices = MomentIndices(maxL)
    momentsFlatIndex = np.zeros((len(momentIndices), ), dtype = npt.Complex128)
    norm = 1.0
    for L in range(maxL + 1):
      for M in range(L + 1):
        # get all moments for given (L, M)
        moments: List[complex] = list(self.momentSet(L, M))
        # ensure that moments are real-valued or purely imaginary, respectively
        tolerance = 1e-15
        assert (abs(moments[0].imag) < tolerance) and (abs(moments[1].imag) < tolerance) and (abs(moments[2].real) < tolerance), (
          f"expect (Im[H_0({L} {M})], Im[H_1({L} {M})], and Re[H_2({L} {M})]) < {tolerance} but found ({moments[0].imag}, {moments[1].imag}, {moments[2].real})")
        # set respective real and imaginary parts exactly to zero
        moments[0] = moments[0].real + 0j
        moments[1] = moments[1].real + 0j
        moments[2] = 0 + moments[2].imag * 1j
        # ensure that H_2(L, 0) is zero
        assert M != 0 or (M == 0 and moments[2] == 0), f"expect H_2({L} {M}) == 0 but found {moments[2].imag}"
        # normalize to H_0(0, 0)
        if L == M == 0:
          norm = moments[0].real  # H_0(0, 0)
        for momentIndex, moment in enumerate(moments[:2 if M == 0 else 3]):
          qnIndex   = QnMomentIndex(momentIndex, L, M)
          flatIndex = momentIndices.indexMap.flatIndex_for[qnIndex]
          momentsFlatIndex[flatIndex] = moment / norm
    HTrue = MomentResult(momentIndices, label = "true")
    HTrue._valsFlatIndex = momentsFlatIndex
    return HTrue


@dataclass(frozen = True)  # immutable
class QnMomentIndex:
  """Stores information about quantum-number indices of moments"""
  momentIndex: int  # subscript of photoproduction moments
  L:           int  # angular momentum
  M:           int  # projection quantum number of L


@dataclass
class MomentIndices:
  """Provides mapping between moment index schemes and iterators for moment indices"""
  maxL:      int          # maximum L quantum number of moments
  photoProd: bool = True  # switches between diffraction and photoproduction mode
  indexMap:  bd.BidictBase[int, QnMomentIndex] = field(init = False)  # bidirectional map for flat index <-> quantum-number index conversion

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
          self.indexMap[flatIndex] = QnMomentIndex(momentIndex, L, M)
          flatIndex += 1

  def __len__(self) -> int:
    """Returns total number of moments"""
    return len(self.indexMap)

  def __getitem__(
    self,
    subscript: int,
  ) -> QnMomentIndex:
    """Returns QnIndex that correspond to given flat index"""
    return self.indexMap[subscript]

  def flatIndices(self) -> Generator[int, None, None]:
    """Generates flat indices"""
    for flatIndex in range(len(self)):
      yield flatIndex

  def QnIndices(self) -> Generator[QnMomentIndex, None, None]:
    """Generates quantum-number indices of the form QnIndex(moment index, L, M)"""
    for flatIndex in range(len(self)):
      yield self[flatIndex]


@dataclass
class DataSet:
  """Stores information about a single dataset"""
  polarization:   float            # photon-beam polarization
  data:           ROOT.RDataFrame  # data from which to calculate moments
  phaseSpaceData: ROOT.RDataFrame  # (accepted) phase-space data
  nmbGenEvents:   int              # number of generated events


@dataclass
class AcceptanceIntegralMatrix:
  """Calculates and provides access to acceptance integral matrix"""
  indices:     MomentIndices  # index mapping and iterators
  dataSet:     DataSet        # info on data samples
  _IFlatIndex: Optional[npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]] = None  # integral matrix with flat indices

  @overload
  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnMomentIndex], Union[int, QnMomentIndex]],
  ) -> Optional[complex]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[slice, Union[int, QnMomentIndex]],
  ) -> Optional[npt.NDArray[npt.Shape["*"], npt.Complex128]]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnMomentIndex], slice],
  ) -> Optional[npt.NDArray[npt.Shape["*"], npt.Complex128]]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[slice, slice],
  ) -> Optional[npt.NDArray[npt.Shape["*, *"], npt.Complex128]]: ...

  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnMomentIndex, slice], Union[int, QnMomentIndex, slice]],
  ) -> Optional[Union[complex, npt.NDArray[npt.Shape["*"], npt.Complex128], npt.NDArray[npt.Shape["*, *"], npt.Complex128]]]:
    """Returns integral matrix elements for any combination of flat and quantum-number indices"""
    if self._IFlatIndex is None:
      return None
    else:
      # turn quantum-number indices to flat indices
      flatIndexMeas: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript[0]] if isinstance(subscript[0], QnMomentIndex) else subscript[0]
      flatIndexPhys: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript[1]] if isinstance(subscript[1], QnMomentIndex) else subscript[1]
      return self._IFlatIndex[flatIndexMeas, flatIndexPhys]

  def __str__(self) -> str:
    if self._IFlatIndex is None:
      return "None"
    else:
      return np.array2string(self._IFlatIndex, precision = 3, suppress_small = True, max_line_width = 150)

  def calculate(self) -> None:
    """Calculates integral matrix of basis functions from (accepted) phase-space data"""
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
    """Saves NumPy array that holds the integral matrix to file with given name"""
    if self._IFlatIndex is not None:
      print(f"Saving integral matrix to file '{fileName}'.")
      np.save(fileName, self._IFlatIndex)

  def load(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    """Loads NumPy array that holds the integral matrix from file with given name"""
    print(f"Loading integral matrix from file '{fileName}'.")
    array = np.load(fileName)
    if not array.shape == (len(self.indices), len(self.indices)):
      raise IndexError(f"Integral loaded from file '{fileName}' has wrong shape. Expected {(len(self.indices), len(self.indices))} but got {array.shape}.")
    self._IFlatIndex = array

  def loadOrCalculate(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    """Loads NumPy array that holds the integral matrix from file with given name; and calculates the integral matrix if loading failed"""
    try:
      self.load(fileName)
    except Exception as e:
      print(f"Could not load integral matrix from file '{fileName}': {e} Calculating matrix instead.")
      self.calculate()


@dataclass
class MomentValue:
  """Stores and provides access to single moment value"""
  qn:       QnMomentIndex   # quantum numbers
  val:      complex   # moment value
  uncertRe: float     # uncertainty of real part
  uncertIm: float     # uncertainty of imaginary part
  label:    str = ""  # label used for printing

  def __iter__(self):
    """Returns iterator over shallow copy of fields"""
    return iter(tuple(getattr(self, field.name) for field in fields(self)))

  def __str__(self) -> str:
    result = ""
    momentSymbol = f"H{'^' + self.label if self.label else ''}_{self.qn.momentIndex}(L = {self.qn.L}, M = {self.qn.M})"
    result += (f"Re[{momentSymbol}] = {self.val.real} +- {self.uncertRe}\n"
               f"Im[{momentSymbol}] = {self.val.imag} +- {self.uncertIm}")
    return result

  @property
  def real(self) -> Tuple[float, float]:
    """Returns real part with uncertainty"""
    return (self.val.real, self.uncertRe)

  @property
  def imag(self) -> Tuple[float, float]:
    """Returns imaginary part with uncertainty"""
    return (self.val.imag, self.uncertIm)

  def realOrImag(
    self,
    realPart: bool,  # switched between real part (True) and imaginary part (False)
  ) -> Tuple[float, float]:
    """Returns real or imaginary part with corresponding uncertainty according to given flag"""
    if realPart:
      return self.real
    else:
      return self.imag


@dataclass
class MomentValueAndTruth(MomentValue):
  """Stores and provides access to single moment value and provides truth value"""
  truth: Optional[complex] = None  # true moment value


@dataclass(eq = False)
class MomentResult:
  """Stores and provides access to moment values"""
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

  def __eq__(
    self,
    other: MomentResult,
  )-> bool:
    # custom equality check needed because of NumPy arrays
    if not isinstance(other, MomentResult):
      return NotImplemented
    return (
      self.indices == other.indices
      and np.array_equal(self._valsFlatIndex,    other._valsFlatIndex)
      and np.array_equal(self._covReReFlatIndex, other._covReReFlatIndex)
      and np.array_equal(self._covImImFlatIndex, other._covImImFlatIndex)
      and np.array_equal(self._covReImFlatIndex, other._covReImFlatIndex)
    )

  @overload
  def __getitem__(
    self,
    subscript: Union[int, QnMomentIndex],
  ) -> MomentValue: ...

  @overload
  def __getitem__(
    self,
    subscript: slice,
  ) -> List[MomentValue]: ...

  def __getitem__(
    self,
    subscript: Union[int, QnMomentIndex, slice],
  ) -> Union[MomentValue, List[MomentValue]]:
    """Returns moment value and corresponding uncertainties at the given flat or quantum-number index/indices"""
    # turn quantum-number index to flat index
    flatIndex: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript] if isinstance(subscript, QnMomentIndex) else subscript
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

  #TODO just for backwards compatibility; remove
  def print(self) -> None:
    for flatIndex in self.indices.flatIndices():
      HVal = self[flatIndex]
      momentSymbol = f"H{'^' + HVal.label if HVal.label else ''}_{HVal.qn.momentIndex}(L = {HVal.qn.L}, M = {HVal.qn.M})"
      print(f"{momentSymbol} = {HVal.val}")

  def copyFrom(
    self,
    other: MomentResult,  # instance from which data are copied
  ) -> None:
    """Copies all values from given MomentResult instance but leaves `label` untouched"""
    self.indices           = other.indices
    self._valsFlatIndex    = other._valsFlatIndex
    self._covReReFlatIndex = other._covReReFlatIndex
    self._covImImFlatIndex = other._covImImFlatIndex
    self._covReImFlatIndex = other._covReImFlatIndex


@dataclass
class MomentCalculator:
  """Calculates and provides access to moments"""
  indices:        MomentIndices  # index mapping and iterators
  dataSet:        DataSet        # info on data samples
  integralMatrix: Optional[AcceptanceIntegralMatrix] = None  # acceptance integral matrix
  HMeas:          Optional[MomentResult]             = None  # calculated measured moments
  HPhys:          Optional[MomentResult]             = None  # calculated physical moments

  def _calcReImCovMatrices(
    self,
    V_aug: npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128],  # augmentented covariance matrix
  ) -> Tuple[npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64], npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64], npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64]]:
    """Calculates covariance matrices for real parts, for imaginary parts, and for real and imaginary parts from augmented covariance matrix"""
    nmbMoments = len(self.indices)
    V_Hermit = V_aug[:nmbMoments, :nmbMoments]  # Hermitian covariance matrix; Eq. (88)
    V_pseudo = V_aug[:nmbMoments, nmbMoments:]  # pseudo-covariance matrix; Eq. (88)
    V_ReRe = (np.real(V_Hermit) + np.real(V_pseudo)) / 2  # Eq. (91)
    V_ImIm = (np.real(V_Hermit) - np.real(V_pseudo)) / 2  # Eq. (92)
    V_ReIm = (np.imag(V_pseudo) - np.imag(V_Hermit)) / 2  # Eq. (93)
    return (V_ReRe, V_ImIm, V_ReIm)

  def calculate(self) -> None:
    """Calculates photoproduction moments and their covariances"""
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
    self.HMeas._covReReFlatIndex, self.HMeas._covImImFlatIndex, self.HMeas._covReImFlatIndex = self._calcReImCovMatrices(V_meas_aug)  # type: ignore
    self.HPhys = MomentResult(self.indices, label = "phys")
    V_phys_aug = np.empty(V_meas_aug.shape, dtype = npt.Complex128)
    if self.integralMatrix is None:
      # ideal detector: physical moments are identical to measured moments
      np.copyto(self.HPhys._valsFlatIndex, self.HMeas._valsFlatIndex)
      np.copyto(V_phys_aug, V_meas_aug)
    else:
      # get acceptance integral matrix
      assert self.integralMatrix._IFlatIndex is not None, "Integral matrix is None."
      I_acc: npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128] = self.integralMatrix._IFlatIndex
      #TODO move to matrix class
      import PlottingUtilities  # avoid circular dependency
      PlottingUtilities.plotComplexMatrix(I_acc, pdfFileNamePrefix = "I_acc")
      I_inv = np.linalg.inv(I_acc)
      PlottingUtilities.plotComplexMatrix(I_inv, pdfFileNamePrefix = "I_inv")
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
