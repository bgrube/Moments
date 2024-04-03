"""Module that provides classes and functions to calculate moments for photoproduction of two-(pseudo)scalar mesons"""
# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=4
#TODO update equation numbers

from __future__ import annotations

import bidict as bd
import dataclasses
from dataclasses import (
  dataclass,
  field,
  fields,
  InitVar,
)
from enum import Enum
import functools
import numpy as np
import nptyping as npt
from typing import (
  Any,
  ClassVar,
  Dict,
  Generator,
  Iterator,
  List,
  Optional,
  overload,
  Sequence,
  Tuple,
  Union,
)
#TODO switch from int for indices to SupportsIndex; requires Python 3.8+

import py3nj
import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@dataclass(frozen = True)  # immutable
class QnWaveIndex:
  """Immutable container class that stores information about quantum-number indices of two-pseudocalar partial-waves in reflectivity basis"""
  refl: int  # reflectivity
  l:    int  # orbital angular momentum
  m:    int  # projection quantum number of l

  @property
  def label(self) -> str:
    """Returns string to construct TObject or file names"""
    return f"[{self.l}]_{self.m}_" + ("p" if self.refl > 0 else "m")

  @property
  def title(self) -> str:
    """Returns TLatex string for titles"""
    return f"[{self.l}]_{{{self.m}}}" + "^{(" + ("#plus" if self.refl > 0 else "#minus") + ")}"


@dataclass
class AmplitudeValue:
  """Container class that stores and provides access to single amplitude value"""
  qn:  QnWaveIndex  # quantum numbers
  val: complex      # amplitude value


@dataclass
class AmplitudeSet:
  """Container class that stores partial-wave amplitudes, makes them accessible by quantum numbers, and calculates spin-density matrix elements and moments"""
  amps:      InitVar[Sequence[AmplitudeValue]]
  tolerance: float = 1e-15  # used when checking whether that moments are either real-valued or purely imaginary
  _amps:     Tuple[Dict[Tuple[int, int], complex], Dict[Tuple[int, int], complex]] = field(init = False)  # internal storage for amplitudes split by positive and negative reflectivity

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
    return AmplitudeValue(subscript, self._amps[reflIndex].get((subscript.l, subscript.m), 0 + 0j))

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
    """Returns all amplitude values up maximum spin; optionally filtered by reflectivity"""
    assert onlyRefl is None or abs(onlyRefl) == 1, f"Invalid reflectivity value f{onlyRefl}; expect +1, -1, or None"
    reflIndices: Tuple[int, ...] = (0, 1)
    if onlyRefl == +1:
      reflIndices = (0, )
    elif onlyRefl == -1:
      reflIndices = (1, )
    for reflIndex in reflIndices:
      for l in range(self.maxSpin + 1):
        for m in range(-l, l + 1):
          yield self[QnWaveIndex(+1 if reflIndex == 0 else -1, l, m)]

  @property
  def maxSpin(self) -> int:
    """Returns maximum spin of wave set ignoring 0 amplitudes"""
    maxl = 0
    for reflIndex in (0, 1):
      for (l, _), val in self._amps[reflIndex].items():
        if val != 0:
          maxl = max(l, maxl)
    return maxl

  def photoProdSpinDensElements(
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

  def photoProdMoments(
    self,
    L: int,  # angular momentum
    M: int,  # projection quantum number of L
    printMoments: bool = False,
  ) -> Tuple[complex, complex, complex]:
    """Returns moments (H_0, H_1, H_2) with given quantum numbers calculated from partial-wave amplitudes assuming rank 1"""
    # Eqs. (154) to (156) assuming that rank is 1
    moments:        List[complex] = 3 * [0 + 0j]
    momentFormulas: List[str]     = [f"H_{i}({L}, {M}) =" for i in range(3)]
    for refl in (-1, +1):
      for amp1 in self.amplitudes(onlyRefl = refl):
        l1 = amp1.qn.l
        m1 = amp1.qn.m
        for amp2 in self.amplitudes(onlyRefl = refl):
          l2 = amp2.qn.l
          m2 = amp2.qn.m
          term = np.sqrt((2 * l2 + 1) / (2 * l1 + 1)) * (
              py3nj.clebsch_gordan(2 * l2, 2 * L, 2 * l1, 0,      0,     0,      ignore_invalid = True)  # (l_2, 0;    L, 0 | l_1, 0  )
            * py3nj.clebsch_gordan(2 * l2, 2 * L, 2 * l1, 2 * m2, 2 * M, 2 * m1, ignore_invalid = True)  # (l_2, m_2;  L, M | l_1, m_1)
          )
          if term == 0:  # unphysical Clebsch-Gordan
            continue
          rhos: Tuple[complex, complex, complex] = self.photoProdSpinDensElements(refl, l1, l2, m1, m2)
          moments[0] +=  term * rhos[0]  # H_0; Eq. (124)
          moments[1] += -term * rhos[1]  # H_1; Eq. (125)
          moments[2] += -term * rhos[2]  # H_2; Eq. (125)
          if printMoments:
            momentFormulas[0] += "" if np.isclose(rhos[0], 0) else f" + {term} * rho_0(refl = {refl}, l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[0]}"
            momentFormulas[1] += "" if np.isclose(rhos[1], 0) else f" - {term} * rho_1(refl = {refl}, l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[1]}"
            momentFormulas[2] += "" if np.isclose(rhos[2], 0) else f" - {term} * rho_2(refl = {refl}, l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[2]}"
    if printMoments:
      print(f"Moment formulas:"
            + ("" if np.isclose(moments[0], 0) else f"\n    {momentFormulas[0]} = {moments[0]}")
            + ("" if np.isclose(moments[1], 0) else f"\n    {momentFormulas[1]} = {moments[1]}")
            + ("" if np.isclose(moments[2], 0) else f"\n    {momentFormulas[2]} = {moments[2]}"))
    return (moments[0], moments[1], moments[2])

  def photoProdMomentSet(
    self,
    maxL:         int,  # maximum L quantum number of moments
    normalize:    Union[bool, int] = True,  # if set to true, moment values are normalized to H_0(0, 0)
                                            # if set to # of events, moments are normalized such that H_0(0, 0) = # of events
    printMoments: bool = False,  # if set formulas for calculation of moments in terms of spin-density matrix elements are printed
  ) -> MomentResult:
    """Returns moments calculated from partial-wave amplitudes assuming rank-1 spin-density matrix; the moments H_2(L, 0) are omitted"""
    momentIndices = MomentIndices(maxL)
    momentsFlatIndex = np.zeros((len(momentIndices), ), dtype = npt.Complex128)
    norm: float = 1.0
    for L in range(maxL + 1):
      for M in range(L + 1):
        # get all moments for given (L, M)
        moments: List[complex] = list(self.photoProdMoments(L, M, printMoments))
        # ensure that moments are either real-valued or purely imaginary
        assert (abs(moments[0].imag) < self.tolerance) and (abs(moments[1].imag) < self.tolerance) and (abs(moments[2].real) < self.tolerance), (
          f"expect (Im[H_0({L} {M})], Im[H_1({L} {M})], and Re[H_2({L} {M})]) < {self.tolerance} but found ({moments[0].imag}, {moments[1].imag}, {moments[2].real})")
        # set respective real and imaginary parts exactly to zero
        moments[0] = moments[0].real + 0j
        moments[1] = moments[1].real + 0j
        moments[2] = 0 + moments[2].imag * 1j
        # ensure that H_2(L, 0) is zero
        assert M != 0 or (M == 0 and moments[2] == 0), f"expect H_2({L} {M}) == 0 but found {moments[2].imag}"
        if normalize and L == M == 0:
          if isinstance(normalize, bool):
            # normalize all moments to H_0(0, 0)
            norm = moments[0].real  # Re[H_0(0, 0)]
          elif isinstance(normalize, int) and normalize > 0:
            # normalize all moments such that H_0(0, 0) = given number of events
            norm = moments[0].real / float(normalize)
        for momentIndex, moment in enumerate(moments[:2 if M == 0 else 3]):
          qnIndex   = QnMomentIndex(momentIndex, L, M)
          flatIndex = momentIndices.indexMap.flatIndex_for[qnIndex]
          momentsFlatIndex[flatIndex] = moment / norm
    HTrue = MomentResult(momentIndices, label = "true")
    HTrue._valsFlatIndex = momentsFlatIndex
    return HTrue


@dataclass(frozen = True)  # immutable
class QnMomentIndex:
  """Immutable container class that stores information about quantum-number indices of moments"""
  momentIndex:  int  # subscript of photoproduction moments
  L:            int  # angular momentum
  M:            int  # projection quantum number of L
  momentSymbol: ClassVar[str] = "H"  # symbol used to construct labels and titles

  @property
  def label(self) -> str:
    """Returns string to construct TObject or file names"""
    return f"{QnMomentIndex.momentSymbol}{self.momentIndex}_{self.L}_{self.M}"

  @property
  def title(self) -> str:
    """Returns TLatex string for titles"""
    return f"#it{{{QnMomentIndex.momentSymbol}}}_{{{self.momentIndex}}}({self.L}, {self.M})"


@dataclass
class MomentIndices:
  """Provides mapping between moment index schemes and iterators for moment indices"""
  maxL:      int          # maximum L quantum number of moments
  photoProd: bool = True  # switches between diffraction and photoproduction mode
  indexMap:  bd.BidictBase[int, QnMomentIndex] = field(init = False)  # bidirectional map for flat index <-> quantum-number index conversion

  def __post_init__(self) -> None:
    # create new bidict subclass
    #TODO bidict 0.23 will remove namedbidict; see https://github.com/jab/bidict/issues/290 for workaround
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
  """Container class that stores information about a single dataset"""
  polarization:   float            # photon-beam polarization
  data:           ROOT.RDataFrame  # data from which to calculate moments
  phaseSpaceData: ROOT.RDataFrame  # (accepted) phase-space data  #TODO make optional
  nmbGenEvents:   int              # number of generated events


@dataclass(frozen = True)  # immutable
class KinematicBinningVariable:
  """Immutable container class that stores information to define a binning variable"""
  name:      str  # name of variable; used e.g. for filenames
  label:     str  # TLatex expression used for plotting
  unit:      str  # TLatex expression used for plotting
  nmbDigits: Optional[int] = None  # number of digits after decimal point to use when converting value to string

  @property
  def axisTitle(self) -> str:
    """Returns axis title"""
    return f"{self.label} [{self.unit}]"


@dataclass
class AcceptanceIntegralMatrix:
  """Container class that calculates, stores, and provides access to acceptance integral matrix"""
  indices:     MomentIndices  # index mapping and iterators
  dataSet:     DataSet        # info on data samples
  _IFlatIndex: Optional[npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]] = None  # acceptance integral matrix with flat indices; must either be given or set be calling load() or calculate()

  # accessor that guarantees existence of optional field
  @property
  def matrix(self) -> npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]:
    """Returns acceptance integral matrix"""
    assert self._IFlatIndex is not None, "self._IFlatIndex must not be None"
    return self._IFlatIndex

  @property
  def matrixNormalized(self) -> npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]:
    """Returns acceptance integral matrix normalized to its diagonal elements"""
    norm = np.diag(np.reciprocal(np.sqrt(np.diag(self.matrix))))  # diagonal matrix with 1 / sqrt(self_ii)
    return norm @ self.matrix @ norm

  @property
  def inverse(self) -> npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]:
    """Returns inverse of acceptance integral matrix"""
    return np.linalg.inv(self.matrix)

  @property
  def eigenDecomp(self) -> Tuple[npt.NDArray[npt.Shape["Dim"], npt.Complex128], npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]]:
    """Returns eigenvalues and eigenvectors of acceptance integral matrix"""
    return np.linalg.eig(self.matrix)

  @overload
  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnMomentIndex], Union[int, QnMomentIndex]],
  ) -> Optional[complex]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[slice, Union[int, QnMomentIndex]],
  ) -> Optional[npt.NDArray[npt.Shape["Slice"], npt.Complex128]]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnMomentIndex], slice],
  ) -> Optional[npt.NDArray[npt.Shape["Slice"], npt.Complex128]]: ...

  @overload
  def __getitem__(
    self,
    subscript: Tuple[slice, slice],
  ) -> Optional[npt.NDArray[npt.Shape["Slice1, Slice2"], npt.Complex128]]: ...

  def __getitem__(
    self,
    subscript: Tuple[Union[int, QnMomentIndex, slice], Union[int, QnMomentIndex, slice]],
  ) -> Optional[Union[complex, npt.NDArray[npt.Shape["Slice"], npt.Complex128], npt.NDArray[npt.Shape["Slice1, Slice2"], npt.Complex128]]]:
    """Returns acceptance integral matrix elements for any combination of flat and quantum-number indices"""
    if self._IFlatIndex is None:
      return None
    else:
      # turn quantum-number indices to flat indices
      flatIndexMeas: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript[0]] if isinstance(subscript[0], QnMomentIndex) else subscript[0]
      flatIndexPhys: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript[1]] if isinstance(subscript[1], QnMomentIndex) else subscript[1]
      return self._IFlatIndex[flatIndexMeas, flatIndexPhys]

  def __str__(self) -> str:
    if self._IFlatIndex is None:
      return str(None)
    else:
      return np.array2string(self._IFlatIndex, precision = 3, suppress_small = True, max_line_width = 150)

  def calculate(self) -> None:
    """Calculates integral matrix of basis functions from (accepted) phase-space data"""
    # get phase-space data data as std::vectors
    thetas = ROOT.std.vector["double"](self.dataSet.phaseSpaceData.AsNumpy(columns = ["theta"])["theta"])
    phis   = ROOT.std.vector["double"](self.dataSet.phaseSpaceData.AsNumpy(columns = ["phi"  ])["phi"  ])
    Phis   = ROOT.std.vector["double"](self.dataSet.phaseSpaceData.AsNumpy(columns = ["Phi"  ])["Phi"  ])
    print(f"Phase-space data column: {type(thetas)}; {thetas.size()}; {thetas.value_type}")
    nmbAccEvents = thetas.size()
    assert thetas.size() == phis.size() == Phis.size(), (
      f"Not all std::vectors with input data have the correct size. Expected {nmbAccEvents} but got theta: {thetas.size()}, phi: {phis.size()}, and Phi: {Phis.size()}")
    if "eventWeight" in self.dataSet.phaseSpaceData.GetColumnNames():
      print("Using weights in 'eventWeight' column to calculate acceptance integral matrix")
      # !Note! event weights must be normalized such that sum_i event_i = number of background-subtracted events (see Eq. (63))
      eventWeights = self.dataSet.phaseSpaceData.AsNumpy(columns = ["eventWeight"])["eventWeight"]
    else:
      # all events have weight 1
      eventWeights = np.ones(nmbAccEvents, dtype = npt.Float64)
    assert eventWeights.shape == (nmbAccEvents,), f"NumPy arrays with event weights does not have the correct shape. Expected ({nmbAccEvents},) but got {eventWeights.shape}"
    # calculate basis-function values for physical and measured moments; Eqs. (175) and (176); defined in `basisFunctions.C`
    nmbMoments = len(self.indices)
    fMeas: npt.NDArray[npt.Shape["nmbMoments, nmbAccEvents"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = npt.Complex128)
    fPhys: npt.NDArray[npt.Shape["nmbMoments, nmbAccEvents"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = npt.Complex128)
    for flatIndex in self.indices.flatIndices():
      qnIndex = self.indices[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, self.dataSet.polarization))
      fPhys[flatIndex] = np.asarray(ROOT.f_phys(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, self.dataSet.polarization))
    # calculate integral-matrix elements; Eq. (178)
    self._IFlatIndex = np.empty((nmbMoments, nmbMoments), dtype = npt.Complex128)
    for flatIndexMeas in self.indices.flatIndices():
      for flatIndexPhys in self.indices.flatIndices():
        self._IFlatIndex[flatIndexMeas, flatIndexPhys] = ((8 * np.pi**2 / self.dataSet.nmbGenEvents)
          * np.dot(np.multiply(eventWeights, fMeas[flatIndexMeas]), fPhys[flatIndexPhys]))
    assert self.isValid(), f"Acceptance integral matrix does not exist or has wrong shape"

  def isValid(self) -> bool:
    """Returns whether acceptance integral matrix exists and has correct shape"""
    return (self._IFlatIndex is not None) and self._IFlatIndex.shape == (len(self.indices), len(self.indices))

  def save(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    """Saves NumPy array that holds the acceptance integral matrix to file with given name"""
    assert self.isValid(), f"Acceptance integral matrix does not exist or has wrong shape"
    if self._IFlatIndex is not None:
      print(f"Saving acceptance integral matrix to file '{fileName}'.")
      np.save(fileName, self._IFlatIndex)

  def load(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    """Loads NumPy array that holds the acceptance integral matrix from file with given name"""
    print(f"Loading acceptance integral matrix from file '{fileName}'.")
    array = np.load(fileName)
    if not array.shape == (len(self.indices), len(self.indices)):
      raise IndexError(f"NumPy array loaded from file '{fileName}' has wrong shape. Expected {(len(self.indices), len(self.indices))} but got {array.shape}.")
    self._IFlatIndex = array
    assert self.isValid(), f"Integral matrix data are inconsistent"

  def loadOrCalculate(
    self,
    fileName: str = "./integralMatrix.npy",
  ) -> None:
    """Tries to load NumPy array that holds the acceptance integral matrix from file with given name; if loading failed the acceptance integral matrix is calculated"""
    try:
      self.load(fileName)
    except Exception as e:
      print(f"Could not load acceptance integral matrix from file '{fileName}': {e} Calculating matrix instead.")
      self.calculate()


@dataclass
class MomentValue:
  """Container class that stores and provides access to single moment value"""
  qn:        QnMomentIndex  # quantum numbers
  val:       complex   # moment value
  uncertRe:  float     # uncertainty of real part
  uncertIm:  float     # uncertainty of imaginary part
  label:     str = ""  # label used for printing
  bsSamples: npt.NDArray[npt.Shape["nmbBootstrapSamples"], npt.Complex128] = np.zeros((0, ), dtype = npt.Complex128)  # array with moment values for each bootstrap sample; array is empty if bootstrapping is disabled

  def __iter__(self):
    """Returns iterator over shallow copy of fields"""
    return iter(tuple(getattr(self, field.name) for field in fields(self)))

  def __str__(self) -> str:
    momentSymbol = f"H{'^' + self.label if self.label else ''}_{self.qn.momentIndex}(L = {self.qn.L}, M = {self.qn.M})"
    result = (f"Re[{momentSymbol}] = {self.val.real} +- {self.uncertRe}\n"
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

  def realPart(
    self,
    realPart: bool,  # switches between real part (True) and imaginary part (False)
  ) -> Tuple[float, float]:
    """Returns real or imaginary part with corresponding uncertainty according to given flag"""
    if realPart:
      return self.real
    else:
      return self.imag

  @property
  def hasBootstrapSamples(self) -> bool:
    """Returns whether bootstrap samples exist"""
    return self.bsSamples.size > 0

  def bootstrapEstimate(
    self,
    realPart: bool,  # switches between real part (True) and imaginary part (False)
  ) -> Tuple[float, float]:
    """Returns bootstrap estimate and its uncertainty for real or imaginary part"""
    assert self.hasBootstrapSamples, "No bootstrap samples available"
    bsSamples = self.bsSamples.real if realPart else self.bsSamples.imag
    bsVal     = float(np.mean(bsSamples))
    bsUncert  = float(np.std(bsSamples, ddof = 1))
    return (bsVal, bsUncert)


@dataclass(eq = False)
class MomentResult:
  """Container class that stores and provides access to moment values"""
  indices:             MomentIndices  # index mapping and iterators
  label:               str = ""  # label used for printing
  nmbBootstrapSamples: int = 0   # number of bootstrap samples
  bootstrapSeed:       int = 0   # seed for random number generator used for bootstrap samples
  _valsFlatIndex:      npt.NDArray[npt.Shape["nmbMoments"],                      npt.Complex128] = field(init = False)  # flat array with moment values
  _covReReFlatIndex:   npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # covariance matrix of real parts of moment values with flat indices
  _covImImFlatIndex:   npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # covariance matrix of imaginary parts of moment values with flat indices
  _covReImFlatIndex:   npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # covariance matrix of real and imaginary parts of moment values with flat indices; !Note! this matrix is _not_ symmetric
  _bsSamplesFlatIndex: npt.NDArray[npt.Shape["nmbMoments, nmbBootstrapSamples"], npt.Complex128] = field(init = False)  # flat array with moment values for each bootstrap sample; array is empty if bootstrapping is disabled

  def __post_init__(self) -> None:
    nmbMoments = len(self.indices)
    self._valsFlatIndex      = np.zeros((nmbMoments, ),                         dtype = npt.Complex128)
    self._covReReFlatIndex   = np.zeros((nmbMoments, nmbMoments),               dtype = npt.Float64)
    self._covImImFlatIndex   = np.zeros((nmbMoments, nmbMoments),               dtype = npt.Float64)
    self._covReImFlatIndex   = np.zeros((nmbMoments, nmbMoments),               dtype = npt.Float64)
    self._bsSamplesFlatIndex = np.zeros((nmbMoments, self.nmbBootstrapSamples), dtype = npt.Complex128)

  def __eq__(
    self,
    other: object,
  )-> bool:
    # custom equality check needed because of NumPy arrays
    if not isinstance(other, MomentResult):
      return NotImplemented
    return (
      self.indices == other.indices
      #TODO shouldn't we use allclose() (+ check for identical shape) instead of array_equal()?
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
    """Returns moment values and corresponding uncertainties at the given flat or quantum-number index/indices"""
    # turn quantum-number index to flat index
    flatIndex: Union[int, slice] = self.indices.indexMap.flatIndex_for[subscript] if isinstance(subscript, QnMomentIndex) else subscript
    if isinstance(flatIndex, slice):
      return [
        MomentValue(
          qn        = self.indices[i],
          val       = self._valsFlatIndex[i],
          uncertRe  = np.sqrt(self._covReReFlatIndex[i, i]),
          uncertIm  = np.sqrt(self._covImImFlatIndex[i, i]),
          bsSamples = self._bsSamplesFlatIndex[i],
          label     = self.label,
        ) for i in range(*flatIndex.indices(len(self.indices)))
      ]
    elif isinstance(flatIndex, int):
      return MomentValue(
        qn        = self.indices[flatIndex],
        val       = self._valsFlatIndex[flatIndex],
        uncertRe  = np.sqrt(self._covReReFlatIndex[flatIndex, flatIndex]),
        uncertIm  = np.sqrt(self._covImImFlatIndex[flatIndex, flatIndex]),
        bsSamples = self._bsSamplesFlatIndex[flatIndex],
        label     = self.label,
      )
    else:
      raise TypeError(f"Invalid subscript type {type(flatIndex)}.")

  def __str__(self) -> str:
    result = (str(self[flatIndex]) for flatIndex in self.indices.flatIndices())
    return "\n".join(result)

  def covariance(
    self,
    subscripts: Tuple[Union[int, QnMomentIndex], Union[int, QnMomentIndex]],  # indices of the two moments
    realParts:  Tuple[bool, bool],  # switches between real part (True) and imaginary part (False) of the two moments
  ) -> npt.NDArray[npt.Shape["2, 2"], npt.Float64]:
    """Returns 2 x 2 covariance submatrix of real or imaginary parts of two moments given by flat or quantum-number indices"""
    assert len(subscripts) == 2, f"Expect exactly two moment indices; got {len(subscripts)} instead"
    assert len(realParts) == 2, f"Expect exactly two flags for real/imag part; got {len(realParts)} instead"
    flatIndices: Tuple[int, int] = tuple(self.indices.indexMap.flatIndex_for[subscript] if isinstance(subscript, QnMomentIndex) else subscript
                                         for subscript in subscripts)
    if realParts == (True, True):
      return np.array([
        [self._covReReFlatIndex[flatIndices[0], flatIndices[0]], self._covReReFlatIndex[flatIndices[0], flatIndices[1]]],
        [self._covReReFlatIndex[flatIndices[1], flatIndices[0]], self._covReReFlatIndex[flatIndices[1], flatIndices[1]]],
      ])
    elif realParts == (False, False):
      return np.array([
        [self._covImImFlatIndex[flatIndices[0], flatIndices[0]], self._covImImFlatIndex[flatIndices[0], flatIndices[1]]],
        [self._covImImFlatIndex[flatIndices[1], flatIndices[0]], self._covImImFlatIndex[flatIndices[1], flatIndices[1]]],
      ])
    elif realParts == (True, False):
      return np.array([
        [self._covReReFlatIndex[flatIndices[0], flatIndices[0]], self._covReImFlatIndex[flatIndices[0], flatIndices[1]]],
        [self._covReImFlatIndex[flatIndices[0], flatIndices[1]], self._covImImFlatIndex[flatIndices[1], flatIndices[1]]],
      ])
    elif realParts == (False, True):
      return np.array([
        [self._covImImFlatIndex[flatIndices[0], flatIndices[0]], self._covReImFlatIndex[flatIndices[1], flatIndices[0]]],
        [self._covReImFlatIndex[flatIndices[1], flatIndices[0]], self._covReReFlatIndex[flatIndices[1], flatIndices[1]]],
      ])
    else:
      raise ValueError(f"Invalid realParts tuple {realParts}; must be tuple of 2 bools")

  @property
  def compositeCovarianceMatrix(self) -> npt.NDArray[npt.Shape["2 * nmbMoments, 2 * nmbMoments"], npt.Float64]:
    """Returns real-valued composite covariance matrix for all moments"""
    # Eq. (11) in https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6125&version=2
    return np.block([
      [self._covReReFlatIndex,   self._covReImFlatIndex],
      [self._covReImFlatIndex.T, self._covImImFlatIndex],
    ])

  @property
  def hermitianAndPseudoCovarianceMatrix(self) \
    -> Tuple[npt.NDArray[npt.Shape["nmbMoments, nmbMoments"], npt.Complex128],
             npt.NDArray[npt.Shape["nmbMoments, nmbMoments"], npt.Complex128]]:
    """Returns tuple with complex-valued Hermitian covariance matrix and pseudo-covariance matrix for all moments"""
    # Eqs. (101) and (102)
    return (self._covReReFlatIndex + self._covImImFlatIndex + 1j * (self._covReImFlatIndex.T - self._covReImFlatIndex),  # Hermitian covariance matrix
            self._covReReFlatIndex - self._covImImFlatIndex + 1j * (self._covReImFlatIndex.T + self._covReImFlatIndex))  # pseudo-covariance matrix

  @property
  def augmentedCovarianceMatrix(self) -> npt.NDArray[npt.Shape["2 * nmbMoments, 2 * nmbMoments"], npt.Complex128]:
    """Returns augmented covariance matrix for all moments"""
    covHermit, covPseudo = self.hermitianAndPseudoCovarianceMatrix
    # Eq. (95)
    return np.block([
      [covPseudo,               covPseudo              ],
      [np.conjugate(covPseudo), np.conjugate(covPseudo)],
    ])

  @property
  def hasBootstrapSamples(self) -> bool:
    """Returns whether bootstrap samples exist"""
    return self._bsSamplesFlatIndex.size > 0


@dataclass
class BootstrapIndices:
  """Generator class for bootstrap indices"""
  nmbEvents:  int  # number of events in data sample
  nmbSamples: int  # number of bootstrap samples
  seed:       int  # seed for random number generator

  def __iter__(self) -> Generator[npt.NDArray[npt.Shape["nmbEvents"], npt.Complex128], None, None]:
    """Generates bootstrap samples and returns data and weights for each sample"""
    rng = np.random.default_rng(self.seed)
    for _ in range(self.nmbSamples):
      yield rng.choice(self.nmbEvents, size = self.nmbEvents, replace = True)


@dataclass
class MomentCalculator:
  """Container class that holds all information needed to calculate moments for a single kinematic bin"""
  indices:              MomentIndices  # index mapping and iterators
  dataSet:              DataSet  # info on data samples
  integralFileBaseName: str  = "./integralMatrix"  # naming scheme for integral files is '<integralFileBaseName>_[<binning var>_<bin center>_...].npy'
  _integralMatrix:      Optional[AcceptanceIntegralMatrix] = None  # if None no acceptance correction is performed; must either be given or calculated by calling calculateIntegralMatrix()
  _HMeas:               Optional[MomentResult] = None  # measured moments; must either be given or calculated by calling calculateMoments()
  _HPhys:               Optional[MomentResult] = None  # physical moments; must either be given or calculated by calling calculateMoments()
  _binCenters:          Optional[Dict[KinematicBinningVariable, float]] = None  # dictionary with bin centers  #TODO make public member and delete properties

  # accessors that guarantee existence of optional fields
  @property
  def integralMatrix(self) -> AcceptanceIntegralMatrix:
    """Returns acceptance integral matrix"""
    assert self._integralMatrix is not None, "self._integralMatrix must not be None"
    return self._integralMatrix

  @property
  def HMeas(self) -> MomentResult:
    """Returns physical moments"""
    assert self._HMeas is not None, "self._HMeas must not be None"
    return self._HMeas

  @property
  def HPhys(self) -> MomentResult:
    """Returns physical moments"""
    assert self._HPhys is not None, "self._HPhys must not be None"
    return self._HPhys

  @property
  def binCenters(self) -> Dict[KinematicBinningVariable, float]:
    """Returns dictionary with kinematic variables and bin centers"""
    assert self._binCenters is not None, "self._binCenters must not be None"
    return self._binCenters

  @binCenters.setter
  def binCenters(
    self,
    centers: Dict[KinematicBinningVariable, float],
  ) -> None:
    """Sets dictionary with kinematic variables and bin centers"""
    self._binCenters = centers

  @property
  def binLabels(self) -> List[str]:
    """Returns list of bin labels; naming scheme of entries is '<binning var>_<bin center>_...'"""
    if self._binCenters is None:
      return []
    return [f"{var.name}_" + (f"{center:.{var.nmbDigits}f}" if var.nmbDigits is not None else f"{center}") for var, center in self.binCenters.items()]

  @property
  def binTitles(self) -> List[str]:
    """Returns list of TLatex expressions for bin centers; scheme of entries is '<binning var> = <bin center> <unit>'"""
    if self._binCenters is None:
      return []
    return [f"{var.label} = " + (f"{center:.{var.nmbDigits}f}" if var.nmbDigits is not None else f"{center}") + f" {var.unit}" for var, center in self.binCenters.items()]

  @property
  def integralFileName(self) -> str:
    """Returns file name used to save acceptance integral matrix; naming scheme is '<integralFileBaseName>_[<binning var>_<bin center>_...].npy'"""
    return "_".join([self.integralFileBaseName, ] + self.binLabels) + ".npy"

  def calculateIntegralMatrix(
    self,
    forceCalculation: bool = False,
  ) -> None:
    """Calculates acceptance integral matrix"""
    self._integralMatrix = AcceptanceIntegralMatrix(self.indices, self.dataSet)
    if forceCalculation:
      self._integralMatrix.calculate()
    else:
      self._integralMatrix.loadOrCalculate(self.integralFileName)
    self._integralMatrix.save(self.integralFileName)

  def _calcReImCovMatrices(
    self,
    V_aug: npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128],  # augmented covariance matrix
  ) -> Tuple[npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64], npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64], npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64]]:
    """Calculates covariance matrices for real parts, for imaginary parts, and for real and imaginary parts from augmented covariance matrix"""
    nmbMoments = len(self.indices)
    V_Hermit = V_aug[:nmbMoments, :nmbMoments]  # Hermitian covariance matrix; Eq. (88)
    V_pseudo = V_aug[:nmbMoments, nmbMoments:]  # pseudo-covariance matrix; Eq. (88)
    V_ReRe = (np.real(V_Hermit) + np.real(V_pseudo)) / 2  # Eq. (91)
    V_ImIm = (np.real(V_Hermit) - np.real(V_pseudo)) / 2  # Eq. (92)
    V_ReIm = (np.imag(V_pseudo) - np.imag(V_Hermit)) / 2  # Eq. (93)
    return (V_ReRe, V_ImIm, V_ReIm)

  MomentDataSource = Enum("MomentDataSource", ("DATA", "ACCEPTED_PHASE_SPACE", "ACCEPTED_PHASE_SPACE_CORR"))

  def calculateMoments(
    self,
    dataSource:          MomentDataSource = MomentDataSource.DATA,  # type of data to calculate moments from
    normalize:           bool             = True,   # if set physical moments are normalized to H_0(0, 0)
    nmbBootstrapSamples: int              = 0,      # number of bootstrap samples; 0 means no bootstrapping
    bootstrapSeed:       int              = 12345,  # seed for random number generator used for bootstrap samples
  ) -> None:
    """Calculates photoproduction moments and their covariances using given data source"""
    # define dataset and integral matrix to use for moment calculation
    dataSet        = None
    integralMatrix = None
    if dataSource == self.MomentDataSource.DATA:
      # calculate moments of data
      dataSet        = self.dataSet
      integralMatrix = self._integralMatrix
    elif dataSource == self.MomentDataSource.ACCEPTED_PHASE_SPACE:
      #TODO this also seems to apply acceptance correction; check!
      # calculate moments of acceptance function
      dataSet        = dataclasses.replace(self.dataSet, data = self.dataSet.phaseSpaceData)
    elif dataSource == self.MomentDataSource.ACCEPTED_PHASE_SPACE_CORR:
      # calculate moments of acceptance-corrected phase space; should all be 0 except H_0(0, 0)
      dataSet        = dataclasses.replace(self.dataSet, data = self.dataSet.phaseSpaceData)
      integralMatrix = self._integralMatrix
    else:
      raise ValueError(f"Unknown data source '{dataSource}'")
    # get input data as std::vectors
    thetas = ROOT.std.vector["double"](dataSet.data.AsNumpy(columns = ["theta"])["theta"])
    phis   = ROOT.std.vector["double"](dataSet.data.AsNumpy(columns = ["phi"  ])["phi"  ])
    Phis   = ROOT.std.vector["double"](dataSet.data.AsNumpy(columns = ["Phi"  ])["Phi"  ])
    print(f"Input data column: {type(thetas)}; {thetas.size()}; {thetas.value_type}")
    nmbEvents = thetas.size()
    assert thetas.size() == phis.size() == Phis.size(), (
      f"Not all std::vectors with input data have the correct size. Expected {nmbEvents} but got theta: {thetas.size()}, phi: {phis.size()}, and Phi: {Phis.size()}")
    if "eventWeight" in dataSet.data.GetColumnNames():
      print("Using weights in 'eventWeight' column to calculate moments")
      # !Note! event weights must be normalized such that sum_i event_i = number of background-subtracted events (see Eq. (63))
      eventWeights = dataSet.data.AsNumpy(columns = ["eventWeight"])["eventWeight"]
    else:
      # all events have weight 1
      eventWeights = np.ones(nmbEvents, dtype = npt.Float64)
    assert eventWeights.shape == (nmbEvents,), f"NumPy arrays with event weights does not have the correct shape. Expected ({nmbEvents},) but got {eventWeights.shape}"
    # sumOfWeights        = np.sum(eventWeights)
    # sumOfSquaredWeights = np.sum(np.square(eventWeights))
    # calculate basis-function values and values of measured moments
    nmbMoments = len(self.indices)
    fMeas:      npt.NDArray[npt.Shape["nmbMoments, nmbEvents"], npt.Complex128] = np.empty((nmbMoments, nmbEvents), dtype = npt.Complex128)
    # fMeasMeans: npt.NDArray[npt.Shape["nmbMoments"],            npt.Complex128] = np.empty((nmbMoments,),           dtype = npt.Complex128)  # weighted means of fMeas values
    bootstrapIndices = BootstrapIndices(nmbEvents, nmbBootstrapSamples, bootstrapSeed)
    self._HMeas = MomentResult(self.indices, label = "meas", nmbBootstrapSamples = nmbBootstrapSamples, bootstrapSeed = bootstrapSeed)
    for flatIndex in self.indices.flatIndices():
      qnIndex = self.indices[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, dataSet.polarization))  # Eq. (176)
      weightedSum = eventWeights.dot(fMeas[flatIndex])
      self._HMeas._valsFlatIndex[flatIndex] = 2 * np.pi * weightedSum  # Eq. (179)
      # fMeasMeans[flatIndex] = weightedSum / sumOfWeights
      # perform bootstrapping of H_meas
      for bsSampleIndex, bsDataIndices in enumerate(bootstrapIndices):  # loop over same set of random data indices for each flatIndex
        # resample data
        fMeasBsSample        = fMeas[flatIndex][bsDataIndices]
        eventWeightsBsSample = eventWeights    [bsDataIndices]
        # calculate bootstrap sample for H_meas
        self._HMeas._bsSamplesFlatIndex[flatIndex, bsSampleIndex] = 2 * np.pi * eventWeightsBsSample.dot(fMeasBsSample)
    #TODO update: calculate covariance matrices for measured moments; Eqs. (88), (180), and (181)
    # # unfortunately, np.cov() does not accept negative weights
    # # reimplement code from https://github.com/numpy/numpy/blob/d35cd07ea997f033b2d89d349734c61f5de54b0d/numpy/lib/function_base.py#L2530-L2749
    # # delta_fMeas = fMeas - fMeasMeans[:, None]
    # # delta_fMeas_aug = np.concatenate((delta_fMeas, np.conjugate(delta_fMeas)), axis = 0)  # augmented vector, i.e. delta_fMeas stacked on top of delta_fMeas^*
    # # see https://juliastats.org/StatsBase.jl/stable/weights/#Implementations and https://juliastats.org/StatsBase.jl/stable/cov/
    # # for a sample of ~1000 background-subtracted events the uncertainty estimates using the various Bessel corrections differ only in the 4th decimal place
    # besselCorrection = 1 / (sumOfWeights - 1)  # assuming frequency weights, i.e. the sum of weights is the number of background-subtracted events
    # # besselCorrection = 1 / (sumOfWeights - sumOfSquaredWeights / sumOfWeights)  # assuming analytic weights that describe importance of each measurement
    # # nmbNonZeroWeights = np.count_nonzero(eventWeights)
    # # besselCorrection = nmbNonZeroWeights / ((nmbNonZeroWeights - 1) * sumOfWeights)  # assuming probability weights that represent the inverse of the sampling probability for each observation
    # V_meas_aug = (2 * np.pi)**2 * sumOfSquaredWeights * besselCorrection * ((eventWeights * delta_fMeas_aug) @ np.asmatrix(delta_fMeas_aug).H)
    fMeasWeighted = eventWeights * fMeas
    V_meas_aug = (2 * np.pi)**2 * nmbEvents * np.cov(fMeasWeighted, np.conjugate(fMeasWeighted), ddof = 1)
    self._HMeas._covReReFlatIndex, self._HMeas._covImImFlatIndex, self._HMeas._covReImFlatIndex = self._calcReImCovMatrices(V_meas_aug)
    # calculate physical moments and propagate uncertainty
    self._HPhys = MomentResult(self.indices, label = "phys", nmbBootstrapSamples = nmbBootstrapSamples, bootstrapSeed = bootstrapSeed)
    V_phys_aug = np.empty(V_meas_aug.shape, dtype = npt.Complex128)
    if integralMatrix is None:
      # ideal detector: physical moments are identical to measured moments
      np.copyto(self._HPhys._valsFlatIndex, self._HMeas._valsFlatIndex)
      np.copyto(V_phys_aug, V_meas_aug)
      np.copyto(self._HPhys._bsSamplesFlatIndex, self._HMeas._bsSamplesFlatIndex)
    else:
      # get inverse of acceptance integral matrix
      I_inv = integralMatrix.inverse
      # calculate physical moments, i.e. correct for detection efficiency
      self._HPhys._valsFlatIndex = I_inv @ self._HMeas._valsFlatIndex  # Eq. (83)
      # calculate bootstrap samples for H_phys
      for bsSampleIndex in range(nmbBootstrapSamples):
        self._HPhys._bsSamplesFlatIndex[:, bsSampleIndex] = I_inv @ self._HMeas._bsSamplesFlatIndex[:, bsSampleIndex]
      # perform linear uncertainty propagation
      J = I_inv  # Jacobian of efficiency correction; Eq. (101)
      J_conj = np.zeros((nmbMoments, nmbMoments), dtype = npt.Complex128)  # conjugate Jacobian; Eq. (101)
      J_aug = np.block([
        [J,                    J_conj],
        [np.conjugate(J_conj), np.conjugate(J)],
      ])  # augmented Jacobian; Eq. (98)
      V_phys_aug = J_aug @ (V_meas_aug @ np.asmatrix(J_aug).H)  #!Note! @ is left-associative; Eq. (85)
    if normalize:
      # normalize moments such that H_0(0, 0) = 1
      #TODO ensure that accesses to indexMap are read-only
      norm: complex = self._HPhys[self.indices.indexMap.flatIndex_for[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]].val
      self._HPhys._valsFlatIndex /= norm
      V_phys_aug /= norm**2
      # normalize bootstrap samples for H_phys
      for bsSampleIndex in range(nmbBootstrapSamples):
        norm: complex = self._HPhys._bsSamplesFlatIndex[self.indices.indexMap.flatIndex_for[QnMomentIndex(momentIndex = 0, L = 0, M = 0)], bsSampleIndex]
        self._HPhys._bsSamplesFlatIndex[:, bsSampleIndex] /= norm
    self._HPhys._covReReFlatIndex, self._HPhys._covImImFlatIndex, self._HPhys._covReImFlatIndex = self._calcReImCovMatrices(V_phys_aug)


@dataclass
class MomentCalculatorsKinematicBinning:
  """Container class that holds all information needed to calculate moments for several kinematic bins"""
  moments: List[MomentCalculator]  # data for all bins of the kinematic binning

  def __len__(self) -> int:
    """Returns number of kinematic bins"""
    return len(self.moments)

  def __getitem__(
    self,
    subscript: int,
  ) -> MomentCalculator:
    """Returns MomentCalculator that correspond to given bin index"""
    return self.moments[subscript]

  def __iter__(self) -> Iterator[MomentCalculator]:
    """Iterates over MomentCalculators in kinematic bins"""
    return iter(self.moments)

  def calculateIntegralMatrices(
    self,
    forceCalculation: bool = False,
  ) -> None:
    """Calculates acceptance integral matrices for all kinematic bins"""
    for momentsInBin in self:
      print(f"Calculating acceptance integral matrix for kinematic bin {momentsInBin.binCenters}")
      momentsInBin.calculateIntegralMatrix(forceCalculation)

  def calculateMoments(
    self,
    dataSource:          MomentCalculator.MomentDataSource = MomentCalculator.MomentDataSource.DATA,  # type of data to calculate moments from
    normalize:           bool = True,   # if set physical moments are normalized to H_0(0, 0)
    nmbBootstrapSamples: int  = 0,      # number of bootstrap samples; 0 means no bootstrapping
    bootstrapSeed:       int  = 12345,  # seed for random number generator used for bootstrap samples
  ) -> None:
    """Calculates moments for all kinematic bins using given data source"""
    for kinBinIndex, momentsInBin in enumerate(self):
      momentsInBin.calculateMoments(dataSource, normalize, nmbBootstrapSamples, bootstrapSeed + kinBinIndex)
