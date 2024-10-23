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
import pickle
from typing import (
  Any,
  ClassVar,
  Generator,
  Iterator,
  overload,
  Sequence,
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
  _amps:     tuple[dict[tuple[int, int], complex], dict[tuple[int, int], complex]] = field(init = False)  # internal storage for amplitudes split by positive and negative reflectivity

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
    onlyRefl: int | None = None,  # if set to +-1 only waves with the corresponding reflectivities
  ) -> Generator[AmplitudeValue, None, None]:
    """Returns all amplitude values up maximum spin; optionally filtered by reflectivity"""
    assert onlyRefl is None or abs(onlyRefl) == 1, f"Invalid reflectivity value f{onlyRefl}; expect +1, -1, or None"
    reflIndices: tuple[int, ...] = (0, 1)
    if onlyRefl == +1:
      reflIndices = (0, )
    elif onlyRefl == -1:
      reflIndices = (1, )
    for reflIndex in reflIndices:
      for l in range(self.maxSpin + 1):
        for m in range(-l, l + 1):
          yield self[QnWaveIndex(refl = (+1 if reflIndex == 0 else -1), l = l, m = m)]

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
  ) -> tuple[complex, complex, complex]:
    """Returns elements of spin-density matrix components (0^rho^ll'_mm', 1^rho^ll'_mm', 2^rho^ll'_mm') with given quantum numbers calculated from partial-wave amplitudes assuming rank 1"""
    qn1     = QnWaveIndex(refl, l1,  m1)
    qn1NegM = QnWaveIndex(refl, l1, -m1)
    qn2     = QnWaveIndex(refl, l2,  m2)
    qn2NegM = QnWaveIndex(refl, l2, -m2)
    rhos: list[complex] = 3 * [0 + 0j]
    rhos[0] =                    (           self[qn1    ].val * self[qn2].val.conjugate() + (-1)**(m1 - m2) * self[qn1NegM].val * self[qn2NegM].val.conjugate())  # Eq. (150)
    rhos[1] =            -refl * ((-1)**m1 * self[qn1NegM].val * self[qn2].val.conjugate() + (-1)**m2        * self[qn1    ].val * self[qn2NegM].val.conjugate())  # Eq. (151)
    rhos[2] = -(0 + 1j) * refl * ((-1)**m1 * self[qn1NegM].val * self[qn2].val.conjugate() - (-1)**m2        * self[qn1    ].val * self[qn2NegM].val.conjugate())  # Eq. (152)
    return (rhos[0], rhos[1], rhos[2])

  def photoProdMoments(
    self,
    L:                   int,  # angular momentum
    M:                   int,  # projection quantum number of L
    printMomentFormulas: bool = False,  # if set formulas for calculation of moments in terms of spin-density matrix elements are printed
  ) -> tuple[complex, complex, complex]:
    """Returns moments (H_0, H_1, H_2) with given quantum numbers calculated from partial-wave amplitudes assuming rank 1"""
    #TODO this function seems to be extremely slow; check
    # Eqs. (154) to (156) assuming that rank is 1
    moments:        list[complex] = 3 * [0 + 0j]
    momentFormulas: list[str]     = [f"H_{i}({L}, {M}) =" for i in range(3)]
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
          rhos: tuple[complex, complex, complex] = self.photoProdSpinDensElements(refl, l1, l2, m1, m2)
          moments[0] +=  term * rhos[0]  # H_0; Eq. (124)
          moments[1] += -term * rhos[1]  # H_1; Eq. (125)
          moments[2] += -term * rhos[2]  # H_2; Eq. (125)
          if printMomentFormulas:
            momentFormulas[0] += "" if np.isclose(rhos[0], 0) else f" + {term} * rho_0(refl = {refl}, l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[0]}"
            momentFormulas[1] += "" if np.isclose(rhos[1], 0) else f" - {term} * rho_1(refl = {refl}, l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[1]}"
            momentFormulas[2] += "" if np.isclose(rhos[2], 0) else f" - {term} * rho_2(refl = {refl}, l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[2]}"
    if printMomentFormulas:
      print(f"Moment formulas:"
            + ("" if np.isclose(moments[0], 0) else f"\n    {momentFormulas[0]} = {moments[0]}")
            + ("" if np.isclose(moments[1], 0) else f"\n    {momentFormulas[1]} = {moments[1]}")
            + ("" if np.isclose(moments[2], 0) else f"\n    {momentFormulas[2]} = {moments[2]}"))
    return (moments[0], moments[1], moments[2])

  def photoProdMomentSet(
    self,
    maxL:                int,  # maximum L quantum number of moments
    normalize:           bool | int = True,  # if set to true, moment values are normalized to H_0(0, 0)
                                             # if set to # of events, moments are normalized such that H_0(0, 0) = # of events
    printMomentFormulas: bool = False,  # if set formulas for calculation of moments in terms of spin-density matrix elements are printed
  ) -> MomentResult:
    """Returns moments calculated from partial-wave amplitudes assuming rank-1 spin-density matrix; the moments H_2(L, 0) are omitted"""
    momentIndices = MomentIndices(maxL, polarized = True)
    momentsFlatIndex = np.zeros((len(momentIndices), ), dtype = np.complex128)
    norm: float = 1.0
    for L in range(maxL + 1):
      for M in range(L + 1):
        # get all moments for given (L, M)
        moments: list[complex] = list(self.photoProdMoments(L, M, printMomentFormulas))
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
          flatIndex = momentIndices[qnIndex]
          momentsFlatIndex[flatIndex] = moment / norm
    HTrue = MomentResult(
      indices = momentIndices,
      label   = "true",
    )
    HTrue._valsFlatIndex = momentsFlatIndex
    return HTrue

  def intensityFormula(
    self,
    polarization: float,         # photon-beam polarization
    thetaFormula: str,           # formula for polar angle theta [rad]
    phiFormula:   str,           # formula for azimuthal angle phi [rad]
    PhiFormula:   str,           # formula for angle Phi between photon polarization and production plane[rad]
    printFormula: bool = False,  # if set formula for calculation of intensity is printed
  ) -> str:
    """Returns formula for intensity calculated from partial-wave amplitudes assuming rank-1 spin-density matrix"""
    # constructed formula uses functions defined in `basisFunctions.C
    intensityComponentTerms: list[tuple[str, str, str]] = []  # summands in Eq. (161) separated by intensity component
    for refl in (-1, +1):
      for amp1 in self.amplitudes(onlyRefl = refl):
        l1 = amp1.qn.l
        m1 = amp1.qn.m
        decayAmp1 = f"Ylm({l1}, {m1}, {thetaFormula}, {phiFormula})"
        for amp2 in self.amplitudes(onlyRefl = refl):
          l2 = amp2.qn.l
          m2 = amp2.qn.m
          decayAmp2 = f"Ylm({l2}, {m2}, {thetaFormula}, {phiFormula})"
          rhos: tuple[complex, complex, complex] = self.photoProdSpinDensElements(refl, l1, l2, m1, m2)
          terms = tuple(
            f"{decayAmp1} * complexT({rho.real}, {rho.imag}) * std::conj({decayAmp2})"  # summand in Eq. (161)
            if rho != 0 else "" for rho in rhos
          )
          intensityComponentTerms.append((terms[0], terms[1], terms[2]))
    # sum all terms for each intensity component
    intensityComponentsFormula = []
    for iComponent in range(3):
      intensityComponentsFormula.append(f"({' + '.join(filter(None, (term[iComponent] for term in intensityComponentTerms)))})")
    # sum intensity components
    intensityFormula = (
      f"std::real({intensityComponentsFormula[0]} "
      f"- {intensityComponentsFormula[1]} * {polarization} * std::cos(2 * {PhiFormula}) "
      f"- {intensityComponentsFormula[2]} * {polarization} * std::sin(2 * {PhiFormula}))"
    )  # Eq. (120)
    if printFormula:
      print(f"Intensity formula = {intensityFormula}")
    return intensityFormula


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
  maxL:                int  # maximum L quantum number of moments
  polarized:           bool = False  # switches between unpolarized production and polarized photoproduction mode
  _qnIndexByFlatIndex: bd.bidict[int, QnMomentIndex] = field(init = False)  # bidirectional map for flat index <-> quantum-number index conversion

  def regenerateIndexMaps(self) -> None:
    self._qnIndexByFlatIndex = bd.bidict()
    flatIndex = 0
    for momentIndex in range(self.momentIndexRange):
      for L in range(self.maxL + 1):
        for M in range(L + 1):
          if momentIndex == 2 and M == 0:
            continue  # H_2(L, 0) are always zero and would lead to a singular acceptance integral matrix
          self._qnIndexByFlatIndex[flatIndex] = QnMomentIndex(momentIndex, L, M)
          flatIndex += 1

  def __post_init__(self) -> None:
    self.regenerateIndexMaps()

  def __len__(self) -> int:
    """Returns total number of moments"""
    return len(self._qnIndexByFlatIndex)

  @overload
  def __getitem__(
    self,
    subscript: int,
  ) -> QnMomentIndex: ...

  @overload
  def __getitem__(
    self,
    subscript: QnMomentIndex,
  ) -> int: ...

  def __getitem__(
    self,
    subscript: int | QnMomentIndex,
  ) -> QnMomentIndex | int:
    """Returns QnIndex that correspond to given flat index and vice versa"""
    if isinstance(subscript, int):
      return self._qnIndexByFlatIndex[subscript]
    elif isinstance(subscript, QnMomentIndex):
      return self._qnIndexByFlatIndex.inverse[subscript]
    else:
      raise TypeError(f"Invalid subscript type {type(subscript)}.")

  @property
  def momentIndexRange(self) -> int:
    """Returns range of moment indices"""
    return 3 if self.polarized else 1

  @property
  def flatIndices(self) -> Generator[int, None, None]:
    """Generates flat indices"""
    for flatIndex in range(len(self)):
      yield flatIndex

  @property
  def qnIndices(self) -> Generator[QnMomentIndex, None, None]:
    """Generates quantum-number indices of the form QnIndex(moment index, L, M)"""
    for flatIndex in range(len(self)):
      yield self[flatIndex]


@dataclass
class DataSet:
  """Container class that stores information about a single dataset"""
  data:           ROOT.RDataFrame  # data from which to calculate moments
  phaseSpaceData: ROOT.RDataFrame | None  # (accepted) phase-space data; None corresponds to perfect acceptance
  nmbGenEvents:   int  # number of generated events
  polarization:   float | None = None  # photon-beam polarization; 0.0 = read from tree


@dataclass(frozen = True)  # immutable
class KinematicBinningVariable:
  """Immutable container class that stores information to define a binning variable"""
  name:      str  # name of variable; used e.g. for filenames
  label:     str  # TLatex expression used for plotting
  unit:      str  # TLatex expression used for plotting
  nmbDigits: int | None = None  # number of digits after decimal point to use when converting value to string

  @property
  def axisTitle(self) -> str:
    """Returns axis title"""
    return f"{self.label} [{self.unit}]"


@dataclass
class AcceptanceIntegralMatrix:
  """Container class that calculates, stores, and provides access to acceptance integral matrix"""
  indices:     MomentIndices  # index mapping and iterators
  dataSet:     DataSet        # info on data samples
  _IFlatIndex: npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128] | None = None  # acceptance integral matrix with flat indices; first index is for measured moments, second index is for physical moments; must either be given or set be calling load() or calculate()

  def __post_init__(self) -> None:
    # set polarized moments case of `indices` according to info provided by `dataSet`
    if self.dataSet.polarization is None:
      self.indices.polarized = False
    else:
      self.indices.polarized = True

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
  def eigenDecomp(self) -> tuple[npt.NDArray[npt.Shape["Dim"], npt.Complex128], npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128]]:
    """Returns eigenvalues and eigenvectors of acceptance integral matrix"""
    return np.linalg.eig(self.matrix)

  @overload
  def __getitem__(
    self,
    subscript: tuple[int | QnMomentIndex, int | QnMomentIndex],
  ) -> complex | None: ...

  @overload
  def __getitem__(
    self,
    subscript: tuple[slice, int | QnMomentIndex],
  ) -> npt.NDArray[npt.Shape["Slice"], npt.Complex128] | None: ...

  @overload
  def __getitem__(
    self,
    subscript: tuple[int | QnMomentIndex, slice],
  ) -> npt.NDArray[npt.Shape["Slice"], npt.Complex128] | None: ...

  @overload
  def __getitem__(
    self,
    subscript: tuple[slice, slice],
  ) -> npt.NDArray[npt.Shape["Slice1, Slice2"], npt.Complex128] | None: ...

  def __getitem__(
    self,
    subscript: tuple[int | QnMomentIndex | slice, int | QnMomentIndex | slice],
  ) -> complex | npt.NDArray[npt.Shape["Slice"], npt.Complex128] | npt.NDArray[npt.Shape["Slice1, Slice2"], npt.Complex128] | None:
    """Returns acceptance integral matrix elements for any combination of flat and quantum-number indices"""
    if self._IFlatIndex is None:
      return None
    else:
      # turn quantum-number indices to flat indices
      flatIndexMeas: int | slice = self.indices[subscript[0]] if isinstance(subscript[0], QnMomentIndex) else subscript[0]
      flatIndexPhys: int | slice = self.indices[subscript[1]] if isinstance(subscript[1], QnMomentIndex) else subscript[1]
      return self._IFlatIndex[flatIndexMeas, flatIndexPhys]

  def __str__(self) -> str:
    if self._IFlatIndex is None:
      return str(None)
    else:
      return np.array2string(self._IFlatIndex, precision = 3, suppress_small = True, max_line_width = 150)

  def calculate(self) -> None:
    """Calculates integral matrix of basis functions from (accepted) phase-space data"""
    if self.dataSet.phaseSpaceData is None:
      print("Warning: no phase-space data; using perfect acceptance")
      self._IFlatIndex = np.eye(len(self.indices), dtype = np.complex128)
      return
    # get photon-beam polarization
    beamPol: float | ROOT.std.vector["double"] | None = None
    if self.dataSet.polarization is None:
      # unpolarized case
      print("Calculating acceptance integral matrix for unpolarized production")
      beamPol = 0.0  # for unpolarized production the basis functions are independent of beamPol; set value to zero
    elif self.dataSet.polarization == 0:
      # polarized case: read polarization value from tree
      assert "beamPol" in self.dataSet.phaseSpaceData.GetColumnNames(), "No 'beamPol' column found in phase-space data"
      print("Reading photon-beam polarization from 'beamPol' column when calculating the acceptance integral matrix")
      beamPol = ROOT.std.vector["double"](self.dataSet.phaseSpaceData.AsNumpy(columns = ["beamPol"])["beamPol"])
    else:
      # polarized case: use polarization value defined for data set
      beamPol = self.dataSet.polarization
      print(f"Using photon-beam polarization of {beamPol} when calculating the acceptance integral matrix")
    # get phase-space data data as std::vectors
    thetas = ROOT.std.vector["double"](self.dataSet.phaseSpaceData.AsNumpy(columns = ["theta"])["theta"])
    phis   = ROOT.std.vector["double"](self.dataSet.phaseSpaceData.AsNumpy(columns = ["phi"  ])["phi"  ])
    Phis   = ROOT.std.vector["double"](self.dataSet.phaseSpaceData.AsNumpy(columns = ["Phi"  ])["Phi"  ]) if self.dataSet.polarization is not None else \
             ROOT.std.vector["double"](np.zeros(len(thetas)))  # for unpolarized production the basis functions are independent of Phi; set values to zero
    print(f"Phase-space data column: type = {type(thetas)}; length = {thetas.size()}; value type = {thetas.value_type}")
    nmbAccEvents = thetas.size()
    assert thetas.size() == phis.size() == Phis.size(), (
      f"Not all std::vectors with input data have the correct size. Expected {nmbAccEvents} but got theta: {thetas.size()}, phi: {phis.size()}, and Phi: {Phis.size()}")
    # get event weights
    eventWeights: npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64] = np.empty(nmbAccEvents, dtype = np.float64)
    if "eventWeight" in self.dataSet.phaseSpaceData.GetColumnNames():
      print("Applying weights from 'eventWeight' column in calculation of acceptance integral matrix")
      # !Note! event weights must be normalized such that sum_i event_i = number of background-subtracted events (see Eq. (63))
      eventWeights = self.dataSet.phaseSpaceData.AsNumpy(columns = ["eventWeight"])["eventWeight"]
      assert eventWeights.shape == (nmbAccEvents,), f"NumPy arrays with event weights does not have the correct shape. Expected ({nmbAccEvents},) but got {eventWeights.shape}"
    else:
      # all events have weight 1
      eventWeights = np.ones(nmbAccEvents, dtype = np.float64)
    # calculate basis-function values for physical and measured moments; Eqs. (175) and (176); defined in `basisFunctions.C`
    nmbMoments = len(self.indices)
    fMeas: npt.NDArray[npt.Shape["nmbMoments, nmbAccEvents"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    fPhys: npt.NDArray[npt.Shape["nmbMoments, nmbAccEvents"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    for flatIndex in self.indices.flatIndices:
      qnIndex = self.indices[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, beamPol))
      fPhys[flatIndex] = np.asarray(ROOT.f_phys(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, beamPol))
    # calculate integral-matrix elements; Eq. (178)
    self._IFlatIndex = np.empty((nmbMoments, nmbMoments), dtype = np.complex128)
    for flatIndexMeas in self.indices.flatIndices:
      for flatIndexPhys in self.indices.flatIndices:
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
  qn:         QnMomentIndex  # quantum numbers
  val:        complex  # moment value
  uncertRe:   float    # uncertainty of real part
  uncertIm:   float    # uncertainty of imaginary part
  binCenters: dict[KinematicBinningVariable, float]                         = field(default_factory = dict)  # center values of variables that define kinematic bin
  label:      str                                                           = ""  # label used for printing
  bsSamples:  npt.NDArray[npt.Shape["nmbBootstrapSamples"], npt.Complex128] = np.zeros((0, ), dtype = np.complex128)  # array with moment values for each bootstrap sample; array is empty if bootstrapping is disabled

  def __iter__(self) -> Iterator[Any]:
    """Returns iterator over shallow copy of fields"""
    return iter(tuple(getattr(self, field.name) for field in fields(self)))

  def __str__(self) -> str:
    momentSymbol = f"H{'^' + self.label if self.label else ''}_{self.qn.momentIndex}(L = {self.qn.L}, M = {self.qn.M})"
    result = (f"Re[{momentSymbol}] = {self.val.real} +- {self.uncertRe}\n"
              f"Im[{momentSymbol}] = {self.val.imag} +- {self.uncertIm}")
    return result

  @property
  def real(self) -> tuple[float, float]:
    """Returns real part with uncertainty"""
    return (self.val.real, self.uncertRe)

  @property
  def imag(self) -> tuple[float, float]:
    """Returns imaginary part with uncertainty"""
    return (self.val.imag, self.uncertIm)

  def part(
    self,
    real: bool,  # switches between real part (True) and imaginary part (False)
  ) -> tuple[float, float]:
    """Returns real or imaginary part with corresponding uncertainty according to given flag"""
    if real:
      return self.real
    else:
      return self.imag

  @property
  def hasBootstrapSamples(self) -> bool:
    """Returns whether bootstrap samples exist"""
    return self.bsSamples.size > 0

  def bootstrapSamplesPart(
    self,
    real: bool,  # switches between real part (True) and imaginary part (False)
  ) -> npt.NDArray[npt.Shape["nmbBootstrapSamples"], npt.Float64]:
    """Returns real or imaginary part of bootstrap samples according to given flag"""
    assert self.hasBootstrapSamples, "No bootstrap samples available"
    if real:
      return self.bsSamples.real
    else:
      return self.bsSamples.imag

  def bootstrapEstimatePart(
    self,
    real: bool,  # switches between real part (True) and imaginary part (False)
  ) -> tuple[float, float]:
    """Returns bootstrap estimate and its uncertainty for real or imaginary part"""
    bsSamples = self.bootstrapSamplesPart(real)
    bsVal     = float(np.mean(bsSamples))
    bsUncert  = float(np.std(bsSamples, ddof = 1))
    return (bsVal, bsUncert)


@dataclass(eq = False)
class MomentResult:
  """Container class that stores and provides access to moment values for single kinematic bin"""
  indices:             MomentIndices  # index mapping and iterators
  binCenters:          dict[KinematicBinningVariable, float]                                     = field(default_factory = dict)  # center values of variables that define kinematic bin
  label:               str                                                                       = ""  # label used for printing
  nmbBootstrapSamples: int                                                                       = 0   # number of bootstrap samples
  bootstrapSeed:       int                                                                       = 0   # seed for random number generator used for bootstrap samples
  _valsFlatIndex:      npt.NDArray[npt.Shape["nmbMoments"],                      npt.Complex128] = field(init = False)  # flat array with moment values
  _covReReFlatIndex:   npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # covariance matrix of real parts of moment values with flat indices
  _covImImFlatIndex:   npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # covariance matrix of imaginary parts of moment values with flat indices
  _covReImFlatIndex:   npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # covariance matrix of real and imaginary parts of moment values with flat indices; !Note! this matrix is _not_ symmetric
  _bsSamplesFlatIndex: npt.NDArray[npt.Shape["nmbMoments, nmbBootstrapSamples"], npt.Complex128] = field(init = False)  # flat array with moment values for each bootstrap sample; array is empty if bootstrapping is disabled

  def __post_init__(self) -> None:
    nmbMoments = len(self)
    self._valsFlatIndex      = np.zeros((nmbMoments,),                          dtype = np.complex128)
    self._covReReFlatIndex   = np.zeros((nmbMoments, nmbMoments),               dtype = np.float64)
    self._covImImFlatIndex   = np.zeros((nmbMoments, nmbMoments),               dtype = np.float64)
    self._covReImFlatIndex   = np.zeros((nmbMoments, nmbMoments),               dtype = np.float64)
    self._bsSamplesFlatIndex = np.zeros((nmbMoments, self.nmbBootstrapSamples), dtype = np.complex128)

  def __eq__(
    self,
    other: object,
  )-> bool:
    # custom equality check needed because of NumPy arrays
    if not isinstance(other, MomentResult):
      return NotImplemented
    return (
      (self.indices == other.indices)
      and (self.binCenters == other.binCenters)
      and (self._valsFlatIndex.shape    == other._valsFlatIndex.shape   )
      and (self._covReReFlatIndex.shape == other._covReReFlatIndex.shape)
      and (self._covImImFlatIndex.shape == other._covImImFlatIndex.shape)
      and (self._covReImFlatIndex.shape == other._covReImFlatIndex.shape)
      and np.allclose(self._valsFlatIndex,    other._valsFlatIndex   )
      and np.allclose(self._covReReFlatIndex, other._covReReFlatIndex)
      and np.allclose(self._covImImFlatIndex, other._covImImFlatIndex)
      and np.allclose(self._covReImFlatIndex, other._covReImFlatIndex)
    )

  def __len__(self) -> int:
    """Returns total number of moments"""
    return len(self.indices)

  @overload
  def __getitem__(
    self,
    subscript: int | QnMomentIndex,
  ) -> MomentValue: ...

  @overload
  def __getitem__(
    self,
    subscript: slice,
  ) -> list[MomentValue]: ...

  def __getitem__(
    self,
    subscript: int | QnMomentIndex | slice,
  ) -> MomentValue | list[MomentValue]:
    """Returns moment values and corresponding uncertainties at the given flat or quantum-number index/indices"""
    # turn quantum-number index to flat index
    flatIndex: int | slice = self.indices[subscript] if isinstance(subscript, QnMomentIndex) else subscript
    if isinstance(flatIndex, slice):
      return [
        MomentValue(
          qn         = self.indices[i],
          val        = self._valsFlatIndex[i],
          uncertRe   = np.sqrt(self._covReReFlatIndex[i, i]),
          uncertIm   = np.sqrt(self._covImImFlatIndex[i, i]),
          binCenters = self.binCenters,
          label      = self.label,
          bsSamples  = self._bsSamplesFlatIndex[i],
        ) for i in range(*flatIndex.indices(len(self.indices)))
      ]
    elif isinstance(flatIndex, int):
      return MomentValue(
        qn         = self.indices[flatIndex],
        val        = self._valsFlatIndex[flatIndex],
        uncertRe   = np.sqrt(self._covReReFlatIndex[flatIndex, flatIndex]),
        uncertIm   = np.sqrt(self._covImImFlatIndex[flatIndex, flatIndex]),
        binCenters = self.binCenters,
        label      = self.label,
        bsSamples  = self._bsSamplesFlatIndex[flatIndex],
      )
    else:
      raise TypeError(f"Invalid subscript type {type(flatIndex)}.")

  @property
  def values(self) -> Generator[MomentValue, None, None]:
    """Generator that yields moment values"""
    for flatIndex in self.indices.flatIndices:
      yield self[flatIndex]

  def __contains__(
    self,
    subscript: int | QnMomentIndex
  ) -> bool:
    """Returns whether moment with given flat index or quantum-number index exists"""
    try:
      flatIndex: int = self.indices[subscript] if isinstance(subscript, QnMomentIndex) else subscript
    except KeyError:
      return False
    try:
      self._valsFlatIndex[flatIndex]
      return True
    except IndexError:
      return False

  def __str__(self) -> str:
    result = (str(moment) for moment in self.values)
    return "\n".join(result)

  def covariance(
    self,
    momentIndexPair: tuple[int | QnMomentIndex, int | QnMomentIndex],  # indices of the two moments
    realParts:       tuple[bool, bool],  # switches between real part (True) and imaginary part (False) of the two moments
  ) -> npt.NDArray[npt.Shape["2, 2"], npt.Float64]:
    """Returns 2 x 2 covariance matrix of real or imaginary parts of two moments given by flat or quantum-number indices"""
    assert len(momentIndexPair) == 2, f"Expect exactly two moment indices; got {len(momentIndexPair)} instead"
    assert len(realParts) == 2, f"Expect exactly two flags for real/imag part; got {len(realParts)} instead"
    flatIndexPair: tuple[int, int] = tuple(
      self.indices[momentIndex] if isinstance(momentIndex, QnMomentIndex) else momentIndex
      for momentIndex in momentIndexPair
    )
    if realParts == (True, True):
      return np.array([
        [self._covReReFlatIndex[flatIndexPair[0], flatIndexPair[0]], self._covReReFlatIndex[flatIndexPair[0], flatIndexPair[1]]],
        [self._covReReFlatIndex[flatIndexPair[1], flatIndexPair[0]], self._covReReFlatIndex[flatIndexPair[1], flatIndexPair[1]]],
      ])
    elif realParts == (False, False):
      return np.array([
        [self._covImImFlatIndex[flatIndexPair[0], flatIndexPair[0]], self._covImImFlatIndex[flatIndexPair[0], flatIndexPair[1]]],
        [self._covImImFlatIndex[flatIndexPair[1], flatIndexPair[0]], self._covImImFlatIndex[flatIndexPair[1], flatIndexPair[1]]],
      ])
    elif realParts == (True, False):
      return np.array([
        [self._covReReFlatIndex[flatIndexPair[0], flatIndexPair[0]], self._covReImFlatIndex[flatIndexPair[0], flatIndexPair[1]]],
        [self._covReImFlatIndex[flatIndexPair[0], flatIndexPair[1]], self._covImImFlatIndex[flatIndexPair[1], flatIndexPair[1]]],
      ])
    elif realParts == (False, True):
      return np.array([
        [self._covImImFlatIndex[flatIndexPair[0], flatIndexPair[0]], self._covReImFlatIndex[flatIndexPair[1], flatIndexPair[0]]],
        [self._covReImFlatIndex[flatIndexPair[1], flatIndexPair[0]], self._covReReFlatIndex[flatIndexPair[1], flatIndexPair[1]]],
      ])
    else:
      raise ValueError(f"Invalid realParts tuple {realParts}; must be tuple of 2 bools")

  def covarianceBootstrap(
    self,
    momentIndexPair: tuple[int | QnMomentIndex, int | QnMomentIndex],  # indices of the two moments
    realParts:       tuple[bool, bool],  # switches between real part (True) and imaginary part (False) of the two moments
  ) -> npt.NDArray[npt.Shape["2, 2"], npt.Float64]:
    """Returns bootstrap estimate of 2 x 2 covariance matrix of real or imaginary parts of two moments given by flat or quantum-number indices"""
    assert len(momentIndexPair) == 2, f"Expect exactly two moment indices; got {len(momentIndexPair)} instead"
    assert len(realParts      ) == 2, f"Expect exactly two flags for real/imag part; got {len(realParts)} instead"
    flatIndexPair: tuple[int, int] = tuple(
      self.indices[momentIndex] if isinstance(momentIndex, QnMomentIndex) else momentIndex
      for momentIndex in momentIndexPair
    )
    # get bootstrap samples of moments
    HVals = (self[flatIndexPair[0]], self[flatIndexPair[1]])
    assert all(HVal.hasBootstrapSamples for HVal in HVals), "Bootstrap samples must be present for both moments"
    assert len(HVals[0].bsSamples) == len(HVals[1].bsSamples), "Number of bootstrap samples must be the same for both moments"
    momentSamplesBs = ((HVals[0].bsSamples.real if realParts[0] else HVals[0].bsSamples.imag,
                        HVals[1].bsSamples.real if realParts[1] else HVals[1].bsSamples.imag))
    return np.cov(momentSamplesBs[0], momentSamplesBs[1], ddof = 1)

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
    -> tuple[npt.NDArray[npt.Shape["nmbMoments, nmbMoments"], npt.Complex128],
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

  def scaleBy(
    self,
    factor: float,
  ) -> None:
    """Scales moment values and uncertainties by given factor"""
    self._valsFlatIndex    *= factor
    self._covReReFlatIndex *= factor**2
    self._covImImFlatIndex *= factor**2
    self._covReImFlatIndex *= factor**2
    if self.hasBootstrapSamples:
      self._bsSamplesFlatIndex *= factor

  def save(
    self,
    pickleFileName: str,
  ) -> None:
    """Saves MomentResult to pickle file"""
    with open(pickleFileName, "wb") as file:
      pickle.dump(self, file)

  @classmethod
  def load(
    cls,
    pickleFileName: str,
  ) -> MomentResult:
    """Loads MomentResult from pickle file"""
    with open(pickleFileName, "rb") as file:
      return pickle.load(file)


def constructMomentResultFrom(
  indices:      MomentIndices,  # index mapping and iterators
  momentValues: Sequence[MomentValue],
) -> MomentResult:
  # ensure that all moment values have the same bin centers, labels, and number of bootstrap samples
  assert all(momentValues[0].binCenters         == momentValue.binCenters         for momentValue in momentValues[1:]), "Moment values must have the same bin centers"
  assert all(momentValues[0].label              == momentValue.label              for momentValue in momentValues[1:]), "Moment values must have the same label"
  assert all(momentValues[0].bsSamples.shape[0] == momentValue.bsSamples.shape[0] for momentValue in momentValues[1:]), "Moment values must have the same number of boostrap samples"
  # ensure that quantum numbers of moment values are unique
  assert len(set(momentValue.qn for momentValue in momentValues)) == len(momentValues), "Moment values must have unique quantum numbers"
  momentResult = MomentResult(
    indices             = indices,
    binCenters          = momentValues[0].binCenters,
    label               = momentValues[0].label,
    nmbBootstrapSamples = momentValues[0].bsSamples.shape[0],
  )
  # loop over moment quantum numbers
  for qnIndex in indices.qnIndices:
    # find MomentValue with same qnIndex
    foundMomentValue = False
    for momentValue in momentValues:
      if momentValue.qn == qnIndex:
        foundMomentValue = True
        # copy values to MomentResult
        flatIndex = indices[qnIndex]
        momentResult._valsFlatIndex[flatIndex]               = momentValue.val
        momentResult._covReReFlatIndex[flatIndex, flatIndex] = momentValue.uncertRe**2
        momentResult._covImImFlatIndex[flatIndex, flatIndex] = momentValue.uncertIm**2
        if momentResult.nmbBootstrapSamples:
          momentResult._bsSamplesFlatIndex[flatIndex] = momentValue.bsSamples
        break
    if not foundMomentValue:
      print(f"Warning: no moment value found for {qnIndex}. Assuming value of 0.")
  return momentResult

@dataclass
class MomentResultsKinematicBinning:
  """Container class that stores and provides access to moment values for several kinematic bins"""
  moments: list[MomentResult]  # data for all bins of the kinematic binning

  # make MomentResultsKinematicBinning behave like a list of MomentResults
  def __len__(self) -> int:
    """Returns number of kinematic bins"""
    return len(self.moments)

  def __getitem__(
    self,
    subscript: int,
  ) -> MomentResult:
    """Returns MomentResult that correspond to given bin index"""
    return self.moments[subscript]

  def __iter__(self) -> Iterator[MomentResult]:
    """Iterates over MomentResults in kinematic bins"""
    return iter(self.moments)

  @property
  def binCenters(self) -> tuple[dict[KinematicBinningVariable, float], ...]:
    """Returns tuple with bin centers of all moments"""
    return tuple(momentResult.binCenters for momentResult in self)

  def scaleBy(
    self,
    factor: float,
  ) -> None:
    """Scales MomentResults by given factor"""
    for moment in self:
      moment.scaleBy(factor)

  def save(
    self,
    pickleFileName: str,
  ) -> None:
    """Saves MomentResultsKinematicBinning to pickle file"""
    with open(pickleFileName, "wb") as file:
      pickle.dump(self, file)

  @classmethod
  def load(
    cls,
    pickleFileName: str,
  ) -> MomentResultsKinematicBinning:
    """Loads MomentResultsKinematicBinning from pickle file"""
    with open(pickleFileName, "rb") as file:
      return pickle.load(file)


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
  binCenters:           dict[KinematicBinningVariable, float]  # dictionary with center values of kinematic variables that define bin
  integralFileBaseName: str = "./integralMatrix"  # naming scheme for integral files is '<integralFileBaseName>_[<binning var>_<bin center>_...].npy'
  _integralMatrix:      AcceptanceIntegralMatrix | None = None  # if None no acceptance correction is performed; must either be given or calculated by calling calculateIntegralMatrix()
  _HMeas:               MomentResult | None = None  # measured moments; must either be given or calculated by calling calculateMoments()
  _HPhys:               MomentResult | None = None  # physical moments; must either be given or calculated by calling calculateMoments()

  def __post_init__(self) -> None:
    # set polarized moments case of `indices` according to info provided by `dataSet`
    if self.dataSet.polarization is None:
      self.indices.polarized = False
    else:
      self.indices.polarized = True

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
  def integralFileName(self) -> str:
    """Returns file name used to save acceptance integral matrix; naming scheme is '<integralFileBaseName>_[<binning var>_<bin center>_...].npy'"""
    return "_".join([self.integralFileBaseName, ] + binLabels(self)) + ".npy"

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
  ) -> tuple[npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64], npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64], npt.NDArray[npt.Shape["Dim, Dim"], npt.Float64]]:
    """Calculates covariance matrices for real parts, for imaginary parts, and for real and imaginary parts from augmented covariance matrix"""
    nmbMoments = len(self.indices)
    V_Hermit = V_aug[:nmbMoments, :nmbMoments]  # Hermitian covariance matrix; Eq. (88)
    V_pseudo = V_aug[:nmbMoments, nmbMoments:]  # pseudo-covariance matrix; Eq. (88)
    V_ReRe = (np.real(V_Hermit) + np.real(V_pseudo)) / 2  # Eq. (91)
    V_ImIm = (np.real(V_Hermit) - np.real(V_pseudo)) / 2  # Eq. (92)
    V_ReIm = (np.imag(V_pseudo) - np.imag(V_Hermit)) / 2  # Eq. (93)
    return (V_ReRe, V_ImIm, V_ReIm)

  MomentDataSource = Enum("MomentDataSource", ("DATA", "ACCEPTED_PHASE_SPACE"))

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
    integralMatrix = self._integralMatrix
    if dataSource == self.MomentDataSource.DATA:
      # calculate moments of data
      dataSet = self.dataSet
    elif dataSource == self.MomentDataSource.ACCEPTED_PHASE_SPACE:
      # calculate moments of acceptance-corrected phase space; physical moments should all be 0 except H_0(0, 0)
      dataSet = dataclasses.replace(self.dataSet, data = self.dataSet.phaseSpaceData)
    else:
      raise ValueError(f"Unknown data source '{dataSource}'")
    # get photon-beam polarization
    beamPol: float | ROOT.std.vector["double"] | None = None
    if dataSet.polarization is None:
      # unpolarized case
      print("Calculating moments for unpolarized production")
      beamPol = 0.0  # for unpolarized production the basis functions are independent of beamPol; set value to zero
    elif dataSet.polarization == 0:
      # polarized case: read polarization value from tree
      assert "beamPol" in dataSet.data.GetColumnNames(), "No 'beamPol' column found in data"
      print("Reading photon-beam polarization from 'beamPol' column when calculating the moments")
      beamPol = ROOT.std.vector["double"](dataSet.data.AsNumpy(columns = ["beamPol"])["beamPol"])
    else:
      # polarized case: use polarization value defined for data set
      beamPol = dataSet.polarization
      print(f"Using photon-beam polarization of {beamPol} when calculating the moments")
    # get input data as std::vectors
    thetas = ROOT.std.vector["double"](dataSet.data.AsNumpy(columns = ["theta"])["theta"])
    phis   = ROOT.std.vector["double"](dataSet.data.AsNumpy(columns = ["phi"  ])["phi"  ])
    Phis   = ROOT.std.vector["double"](dataSet.data.AsNumpy(columns = ["Phi"  ])["Phi"  ]) if dataSet.polarization is not None else \
             ROOT.std.vector["double"](np.zeros(len(thetas)))  # for unpolarized production the basis functions are independent of Phi; set values to zero
    print(f"Input data column: type = {type(thetas)}; length = {thetas.size()}; value type = {thetas.value_type}")
    nmbEvents = thetas.size()
    assert thetas.size() == phis.size() == Phis.size(), (
      f"Not all std::vectors with input data have the correct size. Expected {nmbEvents} but got theta: {thetas.size()}, phi: {phis.size()}, and Phi: {Phis.size()}")
    # get event weights
    eventWeights: npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] = np.empty(nmbEvents, dtype = np.float64)
    if "eventWeight" in dataSet.data.GetColumnNames():
      print("Applying weights from 'eventWeight' column in calculation of moments")
      # !Note! event weights must be normalized such that sum_i event_i = number of background-subtracted events (see Eq. (63))
      eventWeights = dataSet.data.AsNumpy(columns = ["eventWeight"])["eventWeight"]
      assert eventWeights.shape == (nmbEvents,), f"NumPy arrays with event weights does not have the correct shape. Expected ({nmbEvents},) but got {eventWeights.shape}"
    else:
      # all events have weight 1
      eventWeights = np.ones(nmbEvents, dtype = np.float64)
    # calculate basis-function values and values of measured moments
    nmbMoments = len(self.indices)
    fMeas: npt.NDArray[npt.Shape["nmbMoments, nmbEvents"], npt.Complex128] = np.empty((nmbMoments, nmbEvents), dtype = np.complex128)
    bootstrapIndices = BootstrapIndices(nmbEvents, nmbBootstrapSamples, bootstrapSeed)
    self._HMeas = MomentResult(
      indices             = self.indices,
      binCenters          = self.binCenters,
      label               = "meas",
      nmbBootstrapSamples = nmbBootstrapSamples,
      bootstrapSeed       = bootstrapSeed,
    )
    for flatIndex in self.indices.flatIndices:
      qnIndex = self.indices[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(qnIndex.momentIndex, qnIndex.L, qnIndex.M, thetas, phis, Phis, beamPol))  # Eq. (176)
      weightedSum = eventWeights.dot(fMeas[flatIndex])
      self._HMeas._valsFlatIndex[flatIndex] = 2 * np.pi * weightedSum  # Eq. (179)
      # perform bootstrapping of HMeas
      for bsSampleIndex, bsDataIndices in enumerate(bootstrapIndices):  # loop over same set of random data indices for each flatIndex
        # resample data
        fMeasBsSample        = fMeas[flatIndex][bsDataIndices]
        eventWeightsBsSample = eventWeights    [bsDataIndices]
        # calculate bootstrap sample
        self._HMeas._bsSamplesFlatIndex[flatIndex, bsSampleIndex] = 2 * np.pi * eventWeightsBsSample.dot(fMeasBsSample)
    # calculate covariance matrices for measured moments; Eqs. (88), (180), and (181) #TODO update eq numbers
    fMeasWeighted = eventWeights * fMeas
    V_meas_aug = (2 * np.pi)**2 * nmbEvents * np.cov(fMeasWeighted, np.conjugate(fMeasWeighted), ddof = 1)
    self._HMeas._covReReFlatIndex, self._HMeas._covImImFlatIndex, self._HMeas._covReImFlatIndex = self._calcReImCovMatrices(V_meas_aug)
    # calculate physical moments and propagate uncertainty
    self._HPhys = MomentResult(
      indices             = self.indices,
      binCenters          = self.binCenters,
      label               = "phys",
      nmbBootstrapSamples = nmbBootstrapSamples,
      bootstrapSeed       = bootstrapSeed,
    )
    V_phys_aug = np.empty(V_meas_aug.shape, dtype = np.complex128)
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
      J_conj = np.zeros((nmbMoments, nmbMoments), dtype = np.complex128)  # conjugate Jacobian; Eq. (101)
      J_aug = np.block([
        [J,                    J_conj],
        [np.conjugate(J_conj), np.conjugate(J)],
      ])  # augmented Jacobian; Eq. (98)
      V_phys_aug = J_aug @ (V_meas_aug @ np.asmatrix(J_aug).H)  #!Note! @ is left-associative; Eq. (85)
    if normalize:
      # normalize moments such that H_0(0, 0) = 1
      norm: complex = self._HPhys[self.indices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]].val
      self._HPhys._valsFlatIndex /= norm
      V_phys_aug /= norm**2
      # normalize bootstrap samples for H_phys
      for bsSampleIndex in range(nmbBootstrapSamples):
        norm: complex = self._HPhys._bsSamplesFlatIndex[self.indices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)], bsSampleIndex]
        self._HPhys._bsSamplesFlatIndex[:, bsSampleIndex] /= norm
    self._HPhys._covReReFlatIndex, self._HPhys._covImImFlatIndex, self._HPhys._covReImFlatIndex = self._calcReImCovMatrices(V_phys_aug)


# functions that read bin labels and titles from MomentCalculator or MomentResult
def binLabels(obj: MomentCalculator | MomentResult) -> list[str]:
  """Returns list of bin labels; naming scheme of entries is '<binning var>_<bin center>'"""
  return [f"{var.name}_" + (f"{center:.{var.nmbDigits}f}" if var.nmbDigits is not None else f"{center}") for var, center in obj.binCenters.items()]

def binLabel(obj: MomentCalculator | MomentResult) -> str:
  """Returns label for bin; scheme is '<binning var 0>_<bin center 0>_ ...'"""
  return "_".join(binLabels(obj))

def binTitles(obj: MomentCalculator | MomentResult) -> list[str]:
  """Returns list of TLatex expressions for bin centers; scheme of entries is '<binning var> = <bin center> <unit>'"""
  return [f"{var.label} = " + (f"{center:.{var.nmbDigits}f}" if var.nmbDigits is not None else f"{center}") + f" {var.unit}" for var, center in obj.binCenters.items()]

def binTitle(obj: MomentCalculator | MomentResult) -> str:
  """Returns TLatex expressions for kinematic bin centers; scheme is '<binning var 0> = <bin center 0> <unit>, ...'"""
  return ", ".join(binTitles(obj))


@dataclass
class MomentCalculatorsKinematicBinning:
  """Container class that holds all information needed to calculate moments for several kinematic bins"""
  moments: list[MomentCalculator]  # data for all bins of the kinematic binning

  # make MomentCalculatorsKinematicBinning behave like a list of MomentCalculators
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
      print(f"Calculating moments for kinematic bin {momentsInBin.binCenters}")
      momentsInBin.calculateMoments(dataSource, normalize, nmbBootstrapSamples, bootstrapSeed + kinBinIndex)

  @property
  def momentResultsMeas(self) -> MomentResultsKinematicBinning:
    """Returns MomentResultsKinematicBinning with all measured moments"""
    return MomentResultsKinematicBinning([momentsInBin.HMeas for momentsInBin in self])

  @property
  def momentResultsPhys(self) -> MomentResultsKinematicBinning:
    """Returns MomentResultsKinematicBinning with all physical moments"""
    return MomentResultsKinematicBinning([momentsInBin.HPhys for momentsInBin in self])
