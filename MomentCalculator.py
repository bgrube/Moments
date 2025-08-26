"""Module that provides classes and functions to calculate moments for photoproduction of two-(pseudo)scalar mesons"""
# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=4
#TODO update equation numbers

from __future__ import annotations

import bidict as bd
from concurrent.futures import ProcessPoolExecutor
import copy
import dataclasses
from dataclasses import (
  dataclass,
  field,
  fields,
  InitVar,
)
from enum import Enum, auto
import functools
import json
import math
import numpy as np
import nptyping as npt
import os
import pickle
import textwrap
from typing import (
  Any,
  ClassVar,
  overload,
)
from collections.abc import (
  Sequence,
  Iterator,
  Generator,
)

import iminuit as im
from iminuit.typing import Cost
import spherical
import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@functools.cache
def cachedClebschGordan(
  l1: int,
  m1: int,
  l2: int,
  m2: int,
  L:  int,
  M:  int,
) -> float:
  """Cached function that returns Clebsch-Gordan coefficient <l1, m1; l2, m2 | L, M>; works only for integer quantum numbers"""
  #!NOTE! `spherical` only supports integer angular-momentum quantum numbers
  #       if half-integer quantum numbers are needed, use `py3nj.clebsch_gordan` or `sympy.physics.quantum.cg.CG`
  return spherical.clebsch_gordan(l1, m1, l2, m2, L, M)


@dataclass(frozen = True)  # immutable
class QnWaveIndex:
  """Immutable container class that stores information about quantum-number indices of two-pseudoscalar partial-waves in reflectivity basis"""
  refl: int | None  # reflectivity: +-1 for polarized photoproduction; `None` for unpolarized production
  l:    int  # orbital angular momentum
  m:    int  # projection quantum number of l

  @property
  def label(self) -> str:
    """Returns string to construct `TObject` or file names"""
    return f"[{self.l}]_{self.m}" + (
      ""   if self.refl is None else
      "_p" if self.refl > 0 else
      "_m"
    )

  @property
  def title(self) -> str:
    """Returns `TLatex` string for titles"""
    return f"[{self.l}]_{{{self.m}}}" + (
      ""           if self.refl is None else
      "^{(#plus)}" if self.refl > 0 else
      "^{(#minus)}"
    )


@dataclass
class AmplitudeValue:
  """Container class that stores and provides access to single amplitude value"""
  qn:  QnWaveIndex  # quantum numbers
  val: complex      # amplitude value


@dataclass
class AmplitudeSet:
  """Container class that stores partial-wave amplitudes, makes them accessible by quantum numbers, and calculates spin-density matrix elements and moments"""
  amps:       InitVar[Sequence[AmplitudeValue]]
  tolerance:  float = 1e-15  # used when checking whether that moments are either real-valued or purely imaginary
  _amps:      tuple[dict[tuple[int, int], complex], dict[tuple[int, int], complex]] = field(init = False)  # internal storage for amplitudes tuple: positive/ negative reflectivity, dict: key = (l, m), value = moment
  _polarized: bool = field(init = False)  # indicates whether amplitudes are for polarized photoproduction or unpolarized production

  def __post_init__(
    self,
    amps: Sequence[AmplitudeValue],
  ) -> None:
    """Constructs object from list"""
    self._amps = ({}, {})
    # check that reflectivities are consistent
    if all(amp.qn.refl is None for amp in amps):
      self._polarized = False
    elif all(amp.qn.refl is not None and abs(amp.qn.refl) == 1 for amp in amps):
      self._polarized = True
    else:
      raise ValueError("Reflectivity quantum numbers of partial-wave amplitudes can either be all None (unpolarized production) or all +-1 (polarized photoproduction)")
    for amp in amps:
      self[amp.qn] = amp.val

  @property
  def polarized(self) -> bool:
    """Returns `True` if amplitudes are for polarized photoproduction and `False` for unpolarized production"""
    return self._polarized

  def _checkRefl(
    self,
    refl: int | None,
  ) -> None:
    """Checks whether given reflectivity quantum number matches the partial-wave amplitudes"""
    if self.polarized:
      assert refl is not None and abs(refl) == 1, f"Reflectivity quantum number must be +-1 for polarized photoproduction; got {refl}"
    else:
      assert refl is None, f"Reflectivity quantum number must be `None` for unpolarized production; got {refl}"

  def _reflIndex(
    self,
    refl: int | None,
  ) -> int:
    """Returns index to access `_amps` tuple"""
    self._checkRefl(refl)
    return 0 if (refl is None or refl == +1) else 1

  def __getitem__(
    self,
    subscript: QnWaveIndex,
  ) -> AmplitudeValue:
    """Returns partial-wave amplitude for given quantum numbers; returns 0 for non-existing amplitudes"""
    return AmplitudeValue(subscript, self._amps[self._reflIndex(subscript.refl)].get((subscript.l, subscript.m), 0 + 0j))

  def __setitem__(
    self,
    subscript: QnWaveIndex,
    amp:       complex,
  ) -> None:
    """Sets partial-wave amplitude for given quantum numbers"""
    self._amps[self._reflIndex(subscript.refl)][(subscript.l, subscript.m)] = amp

  def amplitudes(
    self,
    onlyRefl: int | None = None,  # if set to +-1 only waves with the corresponding reflectivities are returned; for all other values all amplitudes are returned
  ) -> Generator[AmplitudeValue, None, None]:
    """Returns all amplitude values up maximum spin; optionally filtered by reflectivity"""
    reflIndices: tuple[int, ...] = (0, 1) if self.polarized else (0, )
    if self.polarized:
      if onlyRefl is not None:
        self._checkRefl(onlyRefl)
      if (onlyRefl == +1):
        reflIndices = (0, )
      elif onlyRefl == -1:
        reflIndices = (1, )
    for reflIndex in reflIndices:
      for l in range(self.maxSpin + 1):
        for m in range(-l, l + 1):
          yield self[QnWaveIndex(
            refl = (
              None if not self.polarized else
              +1   if reflIndex == 0 else
              -1
            ),
            l = l, m = m
          )]

  @property
  def maxSpin(self) -> int:
    """Returns maximum spin of wave set ignoring 0 amplitudes"""
    maxl = 0
    for reflIndex in (0, 1) if self.polarized else (0, ):
      for (l, _), val in self._amps[reflIndex].items():
        if val != 0:
          maxl = max(l, maxl)
    return maxl

  def photoProdSpinDensElements(
    self,
    refl: int | None,  # reflectivity
    l1:   int,  # l
    l2:   int,  # l'
    m1:   int,  # m
    m2:   int,  # m'
  ) -> tuple[complex, complex, complex]:
    """Returns elements of spin-density matrix components (0^rho^ll'_mm', 1^rho^ll'_mm', 2^rho^ll'_mm') with given quantum numbers calculated from partial-wave amplitudes assuming rank 1"""
    self._checkRefl(refl)
    qn1     = QnWaveIndex(refl, l1,  m1)
    qn1NegM = QnWaveIndex(refl, l1, -m1)
    qn2     = QnWaveIndex(refl, l2,  m2)
    qn2NegM = QnWaveIndex(refl, l2, -m2)
    rhos: list[complex] = 3 * [0 + 0j]
    if self.polarized:
      rhos[0] =                    (           self[qn1    ].val * self[qn2].val.conjugate() + (-1)**(m1 - m2) * self[qn1NegM].val * self[qn2NegM].val.conjugate())  # Eq. (150)
      rhos[1] =            -refl * ((-1)**m1 * self[qn1NegM].val * self[qn2].val.conjugate() + (-1)**m2        * self[qn1    ].val * self[qn2NegM].val.conjugate())  # Eq. (151)
      rhos[2] = -(0 + 1j) * refl * ((-1)**m1 * self[qn1NegM].val * self[qn2].val.conjugate() - (-1)**m2        * self[qn1    ].val * self[qn2NegM].val.conjugate())  # Eq. (152)
    else:
      rhos[0] = self[qn1].val * self[qn2].val.conjugate() + (-1)**(m1 - m2) * self[qn1NegM].val * self[qn2NegM].val.conjugate()
      rhos[1] = 0 + 0j
      rhos[2] = 0 + 0j
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
    for refl in (-1, +1) if self.polarized else (None, ):
      for amp1 in self.amplitudes(onlyRefl = refl):
        l1 = amp1.qn.l
        m1 = amp1.qn.m
        for amp2 in self.amplitudes(onlyRefl = refl):
          l2 = amp2.qn.l
          m2 = amp2.qn.m
          term =  np.sqrt((2 * l2 + 1) / (2 * l1 + 1)) * (
                cachedClebschGordan(l2, 0,  L, 0, l1, 0 )  # <l_2, 0;    L, 0 | l_1, 0  >
              * cachedClebschGordan(l2, m2, L, M, l1, m1)  # <l_2, m_2;  L, M | l_1, m_1>
          )
          if term == 0:  # unphysical combination of angular-momentum quantum numbers -> zero Clebsch-Gordan coefficient
            continue
          rhos: tuple[complex, complex, complex] = self.photoProdSpinDensElements(refl, l1, l2, m1, m2)
          moments[0] += term * rhos[0]  # H_0; Eqs. (29) and (124)
          if self.polarized:
            moments[1] -= term * rhos[1]  # H_1; Eq. (125)
            moments[2] -= term * rhos[2]  # H_2; Eq. (125)
          if printMomentFormulas:
            momentFormulas[0] += "" if np.isclose(rhos[0], 0) else f" + {term} * rho_0(" + ("" if refl is None else f"refl = {refl}, ") + f"l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[0]}"
            if self.polarized:
              momentFormulas[1] += "" if np.isclose(rhos[1], 0) else f" - {term} * rho_1(refl = {refl}, l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[1]}"
              momentFormulas[2] += "" if np.isclose(rhos[2], 0) else f" - {term} * rho_2(refl = {refl}, l1 = {l1}, m1 = {m1}, l2 = {l2}, m2 = {m2})] = {rhos[2]}"
    if printMomentFormulas:
      print("Moment(s) in terms of spin-density matrix elements:"
            + ("" if np.isclose(moments[0], 0) else f"\n    {momentFormulas[0]} = {moments[0]}")
            + ("" if np.isclose(moments[1], 0) else f"\n    {momentFormulas[1]} = {moments[1]}")
            + ("" if np.isclose(moments[2], 0) else f"\n    {momentFormulas[2]} = {moments[2]}"))
    return (moments[0], moments[1], moments[2])

  def photoProdMomentResult(
    self,
    maxL:                int,  # maximum L quantum number of moments
    normalize:           bool | float = True,   # if set to true, moment values are normalized to H_0(0, 0)
                                                # if set to # of events, moments are normalized such that H_0(0, 0) = # of events
    printMomentFormulas: bool         = False,  # if set formulas for calculation of moments in terms of spin-density matrix elements are printed
    binCenters:          dict[KinematicBinningVariable, float] = {}  # center values of variables that define kinematic bin of `MomentResult`
  ) -> MomentResult:
    """Returns moments calculated from partial-wave amplitudes assuming rank-1 spin-density matrix; since they are 0 the moments H_2(L, 0) are omitted"""
    momentIndices = MomentIndices(maxL, polarized = self.polarized)
    momentsFlatIndex = np.zeros((len(momentIndices), ), dtype = np.complex128)
    norm: float = 1.0
    for L in range(maxL + 1):
      for M in range(L + 1):
        # get all moments for given (L, M)
        moments: list[complex] = list(self.photoProdMoments(L, M, printMomentFormulas))
        # ensure that moments are either real-valued or purely imaginary
        zeroMomentParts = (moments[0].imag, moments[1].imag, moments[2].real)
        assert all(np.isclose(momentPart, 0, atol = self.tolerance) for momentPart in zeroMomentParts), \
          f"Expect (Im[H_0({L} {M})], Im[H_1({L} {M})], and Re[H_2({L} {M})]) < {self.tolerance} but found {zeroMomentParts}"
        # set respective real and imaginary parts exactly to zero
        moments[0] = moments[0].real + 0j
        moments[1] = moments[1].real + 0j
        moments[2] = 0 + moments[2].imag * 1j
        # ensure that H_2(L, 0) is zero
        assert M != 0 or (M == 0 and np.isclose(moments[2], 0, atol = self.tolerance)), f"expect H_2({L} {M}) == 0 but found {moments[2]}"
        if normalize and L == M == 0:
          if isinstance(normalize, bool):
            # normalize all moments to H_0(0, 0) including H_0(0, 0) itself (i.e. H_0(0, 0) = 1 after normalization)
            norm = moments[0].real  # Re[H_0(0, 0)]
          elif isinstance(normalize, float) and normalize > 0:
            # normalize all moments such that H_0(0, 0) = given number of events
            norm = moments[0].real / normalize
        momentsRange = (
          1 if not self.polarized else
          2 if M == 0 else
          3
        )
        for momentIndex, moment in enumerate(moments[:momentsRange]):
          qnIndex   = QnMomentIndex(momentIndex, L, M)
          flatIndex = momentIndices[qnIndex]
          momentsFlatIndex[flatIndex] = moment / norm
    HTruth = MomentResult(
      indices    = momentIndices,
      binCenters = binCenters,
      label      = "true",
    )
    HTruth._valsFlatIndex = momentsFlatIndex
    HTruth.valid = True
    return HTruth

  def intensityFormula(
    self,
    polarization: float | str | None,  # photon-beam polarization; None = unpolarized photoproduction; polarized photoproduction: either polarization value or name of polarization variable
    thetaFormula: str,  # formula for polar angle theta [rad]
    phiFormula:   str,  # formula for azimuthal angle phi [rad]
    PhiFormula:   str,  # formula for angle Phi between photon polarization and production plane[rad]
    printFormula: bool = False,  # if set formula for calculation of intensity is printed
  ) -> str:
    """Returns formula for intensity calculated from partial-wave amplitudes assuming rank-1 spin-density matrix"""
    # constructed formula uses functions defined in `basisFunctions.C`
    intensityComponentTerms: list[tuple[str, str, str]] = []  # summands in Eq. (161) separated by intensity component
    for refl in (-1, +1) if self.polarized else (None, ):
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
    intensityFormula = f"std::real({intensityComponentsFormula[0]}"
    if self.polarized:  # Eq. (120)
      assert polarization is not None, f"For polarized photoproduction, `polarization` must not be `None`"
      intensityFormula += f" - {intensityComponentsFormula[1]} * {polarization} * std::cos(2 * {PhiFormula})"
      intensityFormula += f" - {intensityComponentsFormula[2]} * {polarization} * std::sin(2 * {PhiFormula})"
    else:
      assert polarization is None, f"For unpolarized photoproduction, `polarization` must be `None`"
    intensityFormula += ")"
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
    """Returns string to construct `TObject` or file names"""
    return f"{QnMomentIndex.momentSymbol}{self.momentIndex}_{self.L}_{self.M}"

  @property
  def title(self) -> str:
    """Returns `TLatex` string for titles"""
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

  def setMaxL(
    self,
    value: int
  ) -> None:
    """Sets maximum L quantum number of moments and regenerates index maps"""
    self.maxL = value
    self.regenerateIndexMaps()

  def setPolarized(
    self,
    value: bool
  ) -> None:
    """Sets flag that switches between unpolarized production and polarized photoproduction mode and regenerates index maps"""
    if value != self.polarized:
      self.polarized = value
      self.regenerateIndexMaps()

  def __eq__(
    self,
    other: MomentIndices,
  ) -> bool:
    """Returns whether two `MomentIndices` objects are equal"""
    if not isinstance(other, MomentIndices):
      return NotImplemented
    return (
          self.maxL                == other.maxL
      and self.polarized           == other.polarized
      and self._qnIndexByFlatIndex == other._qnIndexByFlatIndex
    )

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
    """Returns `QnIndex` that correspond to given flat index and vice versa"""
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
    """Generates quantum-number indices of the form 'QnIndex(moment index, L, M)'"""
    for flatIndex in range(len(self)):
      yield self[flatIndex]


@dataclass
class DataSet:
  """Container class that stores information about a single dataset"""
  data:           ROOT.RDataFrame            # data from which to calculate moments
  phaseSpaceData: ROOT.RDataFrame | None     # (accepted) phase-space data; None corresponds to perfect acceptance
  nmbGenEvents:   int                        # number of generated events
  polarization:   float | str | None = None  # photon-beam polarization; None = unpolarized photoproduction; polarized photoproduction: either polarization value or name of polarization variable


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


def getStdVectorFromRdfColumn(
  data:       ROOT.RDataFrame,
  columnName: str,
) -> ROOT.std.vector["double"]:
  """Returns given column of `RDataFrame` as `std::vector<double>`"""
  columnType = data.GetColumnType(columnName)
  assert columnType in ("double", "Double_t"), \
    f"Data column '{columnName}' must be of type 'double', 'Double_t', or 'Double32_t' but is of type '{columnType}'"
  return ROOT.std.vector["double"](data.AsNumpy(columns = [columnName, ])[columnName])


def readInputData(
  polarization:      float | str | None,  # photon-beam polarization; None = unpolarized photoproduction; polarized photoproduction: either polarization value or name of polarization variable
  data:              ROOT.RDataFrame,
  flipSignOfWeights: bool = False,  # if set, weights of real-data event are sign-flipped
) -> tuple[
  float | ROOT.std.vector["double"] | None,         # beam polarization
  ROOT.std.vector["double"],                        # theta values
  ROOT.std.vector["double"],                        # phi values
  ROOT.std.vector["double"],                        # Phi values
  npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]  # event weights
]:
  """Reads input data needed to calculate moments or acceptance integral matrix from given `RDataFrame`"""
  # get photon-beam polarization
  beamPol: float | ROOT.std.vector["double"] | None = None
  if polarization is None:
    # unpolarized case
    print("Setting beam polarization to 0 for unpolarized production")
    beamPol = 0.0  # for unpolarized production the basis functions are independent of beamPol; set value to zero
  elif isinstance(polarization, str):
    # polarized case: read polarization value from column of ROOT tree
    assert polarization in data.GetColumnNames(), f"No '{polarization}' column found in data"
    print(f"Reading photon-beam polarization from '{polarization}' column")
    beamPol = getStdVectorFromRdfColumn(data = data, columnName = polarization)
  elif isinstance(polarization, float):
    # polarized case: use given polarization value for whole data set
    beamPol = polarization
    print(f"Using photon-beam polarization of {beamPol}")
  else:
    raise TypeError(f"Invalid type of `polarization` argument: {type(polarization)}; expected float, str, or None")
  # get input data as std::vectors
  if polarization is None:
    print("Reading values of decay angles from 'theta' and 'phi'")
  else:
    print("Reading values of decay angles from 'theta', 'phi', and 'Phi' columns")
  thetas = getStdVectorFromRdfColumn(data = data, columnName = "theta")
  phis   = getStdVectorFromRdfColumn(data = data, columnName = "phi")
  Phis   = getStdVectorFromRdfColumn(data = data, columnName = "Phi") if polarization is not None else \
    ROOT.std.vector["double"](np.zeros(len(thetas), dtype = np.float64))  # for unpolarized production the basis functions are independent of Phi; set values to zero
  print(f"Input data column: type = {type(thetas)}; length = {thetas.size()}; value type = {thetas.value_type}")
  nmbEvents = thetas.size()
  assert thetas.size() == phis.size() == Phis.size(), (
    f"Not all std::vectors with input data have the correct size. Expected {nmbEvents} but got theta: {thetas.size()}, phi: {phis.size()}, and Phi: {Phis.size()}")

  # get event weights
  eventWeights: npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] = np.empty(nmbEvents, dtype = np.float64)
  if "eventWeight" in data.GetColumnNames():
    print("Applying weights from 'eventWeight' column")
    #!NOTE! event weights must be normalized such that sum_i event_i = number of background-subtracted events (see Eq. (63))
    eventWeights = data.AsNumpy(columns = ["eventWeight"])["eventWeight"]
    assert eventWeights.shape == (nmbEvents, ), f"NumPy array with event weights does not have the correct shape. Expected ({nmbEvents}, ) but got {eventWeights.shape}"
    if flipSignOfWeights:
      print("Inverting sign of event weights")
      eventWeights = -eventWeights
  else:
    # all events have weight 1
    eventWeights = np.ones(nmbEvents, dtype = np.float64)

  return beamPol, thetas, phis, Phis, eventWeights


@dataclass
class AcceptanceIntegralMatrix:
  """Container class that calculates, stores, and provides access to acceptance integral matrix"""
  indices:     MomentIndices  # index mapping and iterators
  dataSet:     DataSet        # info on data samples
  _IFlatIndex: npt.NDArray[npt.Shape["Dim, Dim"], npt.Complex128] | None = None  # acceptance integral matrix with flat indices; first index is for measured moments, second index is for physical moments; must either be given or set be calling load() or calculate()

  def __post_init__(self) -> None:
    # set polarized moments case of `indices` according to info provided by `dataSet`
    if self.dataSet.polarization is None:
      self.indices.setPolarized(False)
    else:
      self.indices.setPolarized(True)

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
    print("Reading phase-space data for the calculation of the acceptance integral matrix")
    beamPol, thetas, phis, Phis, eventWeights = readInputData(
      polarization = self.dataSet.polarization,
      data         = self.dataSet.phaseSpaceData,
    )
    nmbAccEvents = thetas.size()
    # calculate basis-function values for physical and measured moments; Eqs. (175) and (176); defined in `basisFunctions.C`
    nmbMoments = len(self.indices)
    fMeas: npt.NDArray[npt.Shape["nmbMoments, nmbAccEvents"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    fPhys: npt.NDArray[npt.Shape["nmbMoments, nmbAccEvents"], npt.Complex128] = np.empty((nmbMoments, nmbAccEvents), dtype = np.complex128)
    for flatIndex in self.indices.flatIndices:
      qnIndex = self.indices[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(
        qnIndex.momentIndex, qnIndex.L, qnIndex.M,
        thetas,
        phis,
        Phis,
        beamPol,
      ))
      fPhys[flatIndex] = np.asarray(ROOT.f_phys(
        qnIndex.momentIndex, qnIndex.L, qnIndex.M,
        thetas,
        phis,
        Phis,
        beamPol,
      ))
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
    result = (f"Re[{momentSymbol}] = {self.val.real} +- {self.uncertRe}; "
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

  def toJsonDict(self) -> dict:
    """Returns dictionary with moment value and uncertainty that can be serialized to JSON using `json.dumps(MomentValue.toJsonDict())`"""
    return {
      "momentIndex" : self.qn.momentIndex,
      "L"           : self.qn.L,
      "M"           : self.qn.M,
      "valRe"       : self.val.real,
      "uncertRe"    : self.uncertRe,
      "valIm"       : self.val.imag,
      "uncertIm"    : self.uncertIm,
      "binCenters"  : {kinVar.name : binCenter for kinVar, binCenter in self.binCenters.items()},
    }

  def toJsonStr(self) -> str:
    """Returns JSON string with moment value and uncertainty"""
    return json.dumps(self.toJsonDict(), indent = 2)


@dataclass(eq = False)
class MomentResult:
  """Container class that stores and provides access to moment values for single kinematic bin"""
  indices:              MomentIndices  # index mapping and iterators
  binCenters:           dict[KinematicBinningVariable, float]                                     = field(default_factory = dict)  # center values of variables that define kinematic bin
  label:                str                                                                       = ""  # label used for printing
  _nmbBootstrapSamples: int                                                                       = 0   # number of bootstrap samples
  bootstrapRandomSeed:  int                                                                       = 0   # seed for random number generator used for bootstrap samples
  _valsFlatIndex:       npt.NDArray[npt.Shape["nmbMoments"],                      npt.Complex128] = field(init = False)  # flat array with moment values
  _V_ReReFlatIndex:     npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # autocovariance matrix of real parts of moment values with flat indices
  _V_ImImFlatIndex:     npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # autocovariance matrix of imaginary parts of moment values with flat indices
  _V_ReImFlatIndex:     npt.NDArray[npt.Shape["nmbMoments, nmbMoments"],          npt.Float64]    = field(init = False)  # cross-covariance matrix of real and imaginary parts of moment values with flat indices; !NOTE! this matrix is _not_ symmetric
  _bsSamplesFlatIndex:  npt.NDArray[npt.Shape["nmbMoments, nmbBootstrapSamples"], npt.Complex128] = field(init = False)  # flat array with moment values for each bootstrap sample; array is empty if bootstrapping is disabled
  _valid:               bool                                                                      = field(init = False)  # indicates whether instance contains valid values

  def __post_init__(self) -> None:
    nmbMoments = len(self)
    self._valsFlatIndex      = np.zeros((nmbMoments, ),                         dtype = np.complex128)
    self._V_ReReFlatIndex    = np.zeros((nmbMoments, nmbMoments),               dtype = np.float64)
    self._V_ImImFlatIndex    = np.zeros((nmbMoments, nmbMoments),               dtype = np.float64)
    self._V_ReImFlatIndex    = np.zeros((nmbMoments, nmbMoments),               dtype = np.float64)
    self._bsSamplesFlatIndex = np.zeros((nmbMoments, self.nmbBootstrapSamples), dtype = np.complex128)
    self._valid = False

  @property
  def nmbBootstrapSamples(self) -> int:
    """Returns number of bootstrap samples that is calculated"""
    return self._nmbBootstrapSamples

  @nmbBootstrapSamples.setter
  def nmbBootstrapSamples(
    self,
    value: int,
  ) -> None:
    """Sets number of bootstrap samples to be calculated and clears internal array that stores the bootstrap samples"""
    self._nmbBootstrapSamples = value
    nmbMoments = len(self)
    self._bsSamplesFlatIndex = np.zeros((nmbMoments, self.nmbBootstrapSamples), dtype = np.complex128)

  @property
  def valid(self) -> bool:
    """Returns whether this `MomentResult` instance contains valid values"""
    return self._valid

  @valid.setter
  def valid(
    self,
    value: bool,
  ) -> None:
    """Sets whether this `MomentResult` instance contains valid values"""
    self._valid = value

  def __bool__(self) -> bool:
    """Defines truthiness by returning whether this `MomentResult` instance contains valid values"""
    return self.valid

  def hasSameMomentIndicesAndBinCenters(
    self,
    other: MomentResult,
  ) -> bool:
    """Returns whether two `MomentResult` objects have the same indices and bin centers"""
    if not isinstance(other, MomentResult):
      raise TypeError(f"Expect 'other' to be of type 'MomentResults'; got '{type(other)}' instead")
    return (
      (self.indices == other.indices)
      and (self.binCenters == other.binCenters)
      and (self._valsFlatIndex.shape   == other._valsFlatIndex.shape)
      and (self._V_ReReFlatIndex.shape == other._V_ReReFlatIndex.shape)
      and (self._V_ImImFlatIndex.shape == other._V_ImImFlatIndex.shape)
      and (self._V_ReImFlatIndex.shape == other._V_ReImFlatIndex.shape)
    )

  def __eq__(
    self,
    other: MomentResult,
  ) -> bool:
    """Returns whether two `MomentResult` objects are equal in the sense that they have the same indices, bin centers, moment values, and uncertainties"""
    if not isinstance(other, MomentResult):
      return NotImplemented
    return (
      self.hasSameMomentIndicesAndBinCenters(other)
      and np.allclose(self._valsFlatIndex,   other._valsFlatIndex)
      and np.allclose(self._V_ReReFlatIndex, other._V_ReReFlatIndex)
      and np.allclose(self._V_ImImFlatIndex, other._V_ImImFlatIndex)
      and np.allclose(self._V_ReImFlatIndex, other._V_ReImFlatIndex)
    )

  def scaleBy(
    self,
    factor: float,
  ) -> None:
    """Scales moment values and uncertainties by given factor"""
    self._valsFlatIndex   *= factor
    self._V_ReReFlatIndex *= factor**2
    self._V_ImImFlatIndex *= factor**2
    self._V_ReImFlatIndex *= factor**2
    if self.hasBootstrapSamples:
      self._bsSamplesFlatIndex *= factor

  # special methods to implement *=, *, +=, +, -=, and - operators
  def __imul__(
    self,
    scalar: int | float,
  ) -> MomentResult:
    """Multiplication of this with a scalar"""
    if not isinstance(scalar, (int, float)):
      return NotImplemented
    self.label = f"{scalar}*{self.label}"
    self.scaleBy(float(scalar))
    return self

  def __mul__(
    self,
    scalar: int | float,
  ) -> MomentResult:
    """Multiplication with a scalar from the right and returning a new `MomentResult`"""
    product = copy.deepcopy(self)
    product *= scalar
    return product

  def __rmul__(
    self,
    scalar: int | float,
  ) -> MomentResult:
    """Multiplication with a scalar from the left and returning a new `MomentResult`"""
    return self.__mul__(scalar)

  def __iadd__(
    self,
    other: MomentResult,
  ) -> MomentResult:
    """Combines this and another `MomentResult` assuming they were estimated from independent data samples, i.e. by summing the moment values and (co)variances"""
    if not isinstance(other, MomentResult):
      return NotImplemented
    # ensure that `other` has the same indices and bin centers
    assert self.hasSameMomentIndicesAndBinCenters(other), "Moment results must have the same moment indices and bin centers"
    if self.hasBootstrapSamples or other.hasBootstrapSamples:
      print(f"Warning: bootstrap samples are not added.")
    self.label += f"+{other.label}"
    # sum moment values and (co)variances; see Eq. (220)
    # relies on arrays being initialized with zeros
    self._valsFlatIndex   += other._valsFlatIndex
    self._V_ReReFlatIndex += other._V_ReReFlatIndex
    self._V_ImImFlatIndex += other._V_ImImFlatIndex
    self._V_ReImFlatIndex += other._V_ReImFlatIndex
    self._valid = self._valid and other._valid  # only valid if both summands are valid
    return self

  def __add__(
    self,
    other: MomentResult,
  ) -> MomentResult:
    """Combines this and another `MomentResult` into a new `MomentResult` by summing the moment values and (co)variances"""
    sum = copy.deepcopy(self)
    sum += other
    return sum

  def __isub__(
    self,
    other: MomentResult,
  ) -> MomentResult:
    """Combines this and another `MomentResult` into one `MomentResult` by subtracting the moment values and adding the (co)variances"""
    if not isinstance(other, MomentResult):
      return NotImplemented
    self += (-1 * other)
    return self

  def __sub__(
    self,
    other: MomentResult,
  ) -> MomentResult:
    """Combines this and another `MomentResult` into one `MomentResult` by subtracting the moment values and adding the (co)variances"""
    if not isinstance(other, MomentResult):
      return NotImplemented
    return self + (-1 * other)

  # special methods to implement Collection protocol
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
    if not self:
      raise ValueError("MomentResult is not valid; cannot access moment values")
    # turn quantum-number index to flat index
    flatIndex: int | slice = self.indices[subscript] if isinstance(subscript, QnMomentIndex) else subscript
    if isinstance(flatIndex, slice):
      return [
        MomentValue(
          qn         = self.indices[i],
          val        = self._valsFlatIndex[i],
          uncertRe   = np.sqrt(self._V_ReReFlatIndex[i, i]),
          uncertIm   = np.sqrt(self._V_ImImFlatIndex[i, i]),
          binCenters = self.binCenters,
          label      = self.label,
          bsSamples  = self._bsSamplesFlatIndex[i],
        ) for i in range(*flatIndex.indices(len(self.indices)))
      ]
    elif isinstance(flatIndex, int):
      return MomentValue(
        qn         = self.indices[flatIndex],
        val        = self._valsFlatIndex[flatIndex],
        uncertRe   = np.sqrt(self._V_ReReFlatIndex[flatIndex, flatIndex]),
        uncertIm   = np.sqrt(self._V_ImImFlatIndex[flatIndex, flatIndex]),
        binCenters = self.binCenters,
        label      = self.label,
        bsSamples  = self._bsSamplesFlatIndex[flatIndex],
      )
    else:
      raise TypeError(f"Invalid subscript type {type(flatIndex)}.")

  def __contains__(
    self,
    subscript: int | QnMomentIndex
  ) -> bool:
    """Returns whether moment with given flat index or quantum-number index exists"""
    if not self:
      raise ValueError("MomentResult is not valid; cannot access moment values")
    try:
      flatIndex: int = self.indices[subscript] if isinstance(subscript, QnMomentIndex) else subscript
    except KeyError:
      return False
    try:
      self._valsFlatIndex[flatIndex]
      return True
    except IndexError:
      return False

  @property
  def values(self) -> Generator[MomentValue, None, None]:
    """Generator that yields moment values"""
    if not self:
      raise ValueError("MomentResult is not valid; cannot access moment values")
    for flatIndex in self.indices.flatIndices:
      yield self[flatIndex]

  def __str__(self) -> str:
    result = (str(moment) for moment in self.values)
    return "\n".join(result)

  def covariance(
    self,
    momentIndexPair: tuple[int | QnMomentIndex, int | QnMomentIndex],  # indices of the two moments
    realParts:       tuple[bool, bool],  # switches between real part (True) and imaginary part (False) of the two moments
  ) -> npt.NDArray[npt.Shape["2, 2"], npt.Float64]:
    """Returns 2 x 2 covariance matrix of real or imaginary parts of two moments given by flat or quantum-number indices"""
    if not self:
      raise ValueError("MomentResult is not valid; cannot access covariance values")
    assert len(momentIndexPair) == 2, f"Expect exactly two moment indices; got {len(momentIndexPair)} instead"
    assert len(realParts) == 2, f"Expect exactly two flags for real/imag part; got {len(realParts)} instead"
    flatIndexPair: tuple[int, int] = tuple(
      self.indices[momentIndex] if isinstance(momentIndex, QnMomentIndex) else momentIndex
      for momentIndex in momentIndexPair
    )
    if realParts == (True, True):
      return np.array([
        [self._V_ReReFlatIndex[flatIndexPair[0], flatIndexPair[0]], self._V_ReReFlatIndex[flatIndexPair[0], flatIndexPair[1]]],
        [self._V_ReReFlatIndex[flatIndexPair[1], flatIndexPair[0]], self._V_ReReFlatIndex[flatIndexPair[1], flatIndexPair[1]]],
      ])
    elif realParts == (False, False):
      return np.array([
        [self._V_ImImFlatIndex[flatIndexPair[0], flatIndexPair[0]], self._V_ImImFlatIndex[flatIndexPair[0], flatIndexPair[1]]],
        [self._V_ImImFlatIndex[flatIndexPair[1], flatIndexPair[0]], self._V_ImImFlatIndex[flatIndexPair[1], flatIndexPair[1]]],
      ])
    elif realParts == (True, False):
      return np.array([
        [self._V_ReReFlatIndex[flatIndexPair[0], flatIndexPair[0]], self._V_ReImFlatIndex[flatIndexPair[0], flatIndexPair[1]]],
        [self._V_ReImFlatIndex[flatIndexPair[0], flatIndexPair[1]], self._V_ImImFlatIndex[flatIndexPair[1], flatIndexPair[1]]],
      ])
    elif realParts == (False, True):
      return np.array([
        [self._V_ImImFlatIndex[flatIndexPair[0], flatIndexPair[0]], self._V_ReImFlatIndex[flatIndexPair[1], flatIndexPair[0]]],
        [self._V_ReImFlatIndex[flatIndexPair[1], flatIndexPair[0]], self._V_ReReFlatIndex[flatIndexPair[1], flatIndexPair[1]]],
      ])
    else:
      raise ValueError(f"Invalid realParts tuple {realParts}; must be tuple of 2 bools")

  @property
  def hasBootstrapSamples(self) -> bool:
    """Returns whether bootstrap samples exist"""
    return self._bsSamplesFlatIndex.size > 0

  def covarianceBootstrap(
    self,
    momentIndexPair: tuple[int | QnMomentIndex, int | QnMomentIndex],  # indices of the two moments
    realParts:       tuple[bool, bool],  # switches between real part (True) and imaginary part (False) of the two moments
  ) -> npt.NDArray[npt.Shape["2, 2"], npt.Float64]:
    """Returns bootstrap estimate of 2 x 2 covariance matrix of real or imaginary parts of two moments given by flat or quantum-number indices"""
    if not self:
      raise ValueError("MomentResult is not valid; cannot access covariance values")
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
    """Returns real-valued composite covariance matrix for all moments indexed by flat index"""
    # Eq. (11) in https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6125&version=2
    if not self:
      raise ValueError("MomentResult is not valid; cannot access covariance values")
    return np.block([
      [self._V_ReReFlatIndex,   self._V_ReImFlatIndex],
      [self._V_ReImFlatIndex.T, self._V_ImImFlatIndex],
    ])

  @property
  def hermitianAndPseudoCovarianceMatrix(self) \
    -> tuple[npt.NDArray[npt.Shape["nmbMoments, nmbMoments"], npt.Complex128],
             npt.NDArray[npt.Shape["nmbMoments, nmbMoments"], npt.Complex128]]:
    """Returns tuple with complex-valued Hermitian covariance matrix and pseudo-covariance matrix for all moments indexed by flat index"""
    # Eqs. (101) and (102)
    if not self:
      raise ValueError("MomentResult is not valid; cannot access covariance values")
    return (self._V_ReReFlatIndex + self._V_ImImFlatIndex + 1j * (self._V_ReImFlatIndex.T - self._V_ReImFlatIndex),  # Hermitian covariance matrix
            self._V_ReReFlatIndex - self._V_ImImFlatIndex + 1j * (self._V_ReImFlatIndex.T + self._V_ReImFlatIndex))  # pseudo-covariance matrix

  @property
  def augmentedCovarianceMatrix(self) -> npt.NDArray[npt.Shape["2 * nmbMoments, 2 * nmbMoments"], npt.Complex128]:
    """Returns augmented covariance matrix for all moments indexed by flat index"""
    if not self:
      raise ValueError("MomentResult is not valid; cannot access covariance values")
    V_Hermit, V_pseudo = self.hermitianAndPseudoCovarianceMatrix
    # Eq. (95)
    return np.block([
      [V_Hermit,               V_pseudo              ],
      [np.conjugate(V_pseudo), np.conjugate(V_Hermit)],
    ])

  def setCovarianceMatricesFrom(
    self,
    V_aug: npt.NDArray[npt.Shape["2 * nmbMoments, 2 * nmbMoments"], npt.Complex128],
  ) -> None:
    """Set internal covariance matrices for real and imaginary parts from augmented covariance matrix"""
    nmbMoments = len(self.indices)
    V_Hermit = V_aug[:nmbMoments, :nmbMoments]  # Hermitian covariance matrix; Eq. (88)
    V_pseudo = V_aug[:nmbMoments, nmbMoments:]  # pseudo-covariance matrix; Eq. (88)
    self._V_ReReFlatIndex = (np.real(V_Hermit) + np.real(V_pseudo)) / 2  # Eq. (91)
    self._V_ImImFlatIndex = (np.real(V_Hermit) - np.real(V_pseudo)) / 2  # Eq. (92)
    self._V_ReImFlatIndex = (np.imag(V_pseudo) - np.imag(V_Hermit)) / 2  # Eq. (93)

  def normalize(self) -> None:
    """Scales all moment values by common factor such that H_0(0, 0) = 1"""
    if not self:
      raise ValueError("MomentResult is not valid; cannot normalize")
    norm: complex = self[self.indices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]].val  # get value of H_0(0, 0)
    self._valsFlatIndex /= norm
    # since H_0(0, 0) might be complex-valued, we have to
    # i) convert covariance matrices to an augmented covariance matrix
    V_aug = self.augmentedCovarianceMatrix
    # ii) scale the augmented covariance matrix with the complex-valued normalization factor
    #!NOTE! this is not 100% correct because H_0(0, 0) actually
    #  becomes a constant and its variance should actually be removed
    #  from the covariance matrix; cross-check with the bootstrap
    #  estimates to see whether this effect is negligible
    V_aug /= norm**2
    # iii) convert the augmented covariance matrix back to the covariance matrices for real and imaginary parts
    self.setCovarianceMatricesFrom(V_aug)
    # normalize each bootstrap sample
    for bsSampleIndex in range(self.nmbBootstrapSamples):
      norm: complex = self._bsSamplesFlatIndex[self.indices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)], bsSampleIndex]
      self._bsSamplesFlatIndex[:, bsSampleIndex] /= norm

  def savePickle(
    self,
    pickleFileName: str,
  ) -> None:
    """Saves `MomentResult` to pickle file"""
    with open(pickleFileName, "wb") as file:
      pickle.dump(self, file)

  @classmethod
  def loadPickle(
    cls,
    pickleFileName: str,
  ) -> MomentResult:
    """Loads `MomentResult` from pickle file"""
    with open(pickleFileName, "rb") as file:
      return pickle.load(file)

  def toJsonStr(self) -> str:
    """Returns JSON string with moment values for all moments in kinematic bin"""
    return json.dumps([momentValue.toJsonDict() for momentValue in self.values], indent = 2)

  def intensityFormula(
    self,
    polarization:                float | str | None,  # photon-beam polarization; None = unpolarized photoproduction; polarized photoproduction: either polarization value or name of polarization variable
    thetaFormula:                str,  # formula for polar angle theta [rad]
    phiFormula:                  str,  # formula for azimuthal angle phi [rad]
    PhiFormula:                  str,  # formula for angle Phi between photon polarization and production plane[rad]
    printFormula:                bool = False,  # if True, formula for calculation of intensity is printed
    useMomentSymbols:            bool = False,  # if True, insert TFormula parameter names "[Hi_L_M]" instead of moment values into formula
    includeParityViolatingTerms: bool = True,   # if True, include parity-violating terms into formula
  ) -> str:
    """Returns formula for intensity calculated from moment values"""
    # constructed formula uses functions defined in `basisFunctions.C`
    intensityComponentTerms: tuple[list[str], list[str], list[str]] = ([], [], [])  # summands in Eqs. (150) to (152) separated by intensity component
    for qnIndex in self.indices.qnIndices:
      momentIndex = qnIndex.momentIndex
      L = qnIndex.L
      M = qnIndex.M
      HLM = self[QnMomentIndex(momentIndex, L, M)].val
      YLM = f"Ylm({L}, {M}, {thetaFormula}, {phiFormula})"
      term = f"{np.sqrt((2 * L + 1) / (4 * math.pi)) * (1 if M == 0 else 2)} "  # normalization factor
      if includeParityViolatingTerms:
        term += "* complexT("  # complexT is a typedef for std::complex<double> in `basisFunctions.C`
        term += (f"[Re{qnIndex.label}], [Im{qnIndex.label}]" if useMomentSymbols else f"({HLM.real}), ({HLM.imag})")
        term +=  ")"
        term += f"* {YLM}"
      else:
        term += "* " + (f"[{qnIndex.label}] " if useMomentSymbols else f"({HLM.imag if momentIndex == 2 else HLM.real}) ")  # real or imaginary part of moment value
        term += f"* {'Im' if momentIndex == 2 else 'Re'}{YLM}"  # real or imaginary part of spherical harmonic
      intensityComponentTerms[momentIndex].append(term)
    # sum all terms for each intensity component
    intensityComponentsFormula = [""] * 3
    for momentIndex, terms in enumerate(intensityComponentTerms):
      intensityComponentsFormula[momentIndex] = f"({' + '.join(filter(None, terms))})"
    if self.indices.polarized:
      intensityComponentsFormula[1] = f"(-{intensityComponentsFormula[1]})"
    # sum intensity components
    intensityFormula = f"{intensityComponentsFormula[0]}"
    if self.indices.polarized:  # Eq. (120)
      assert polarization is not None, f"For polarized photoproduction, `polarization` must not be `None`"
      intensityFormula += f" - {intensityComponentsFormula[1]} * {polarization} * std::cos(2 * {PhiFormula})"
      intensityFormula += f" - {intensityComponentsFormula[2]} * {polarization} * std::sin(2 * {PhiFormula})"
    else:
      assert polarization is None, f"For unpolarized photoproduction, `polarization` must be `None`"
    if includeParityViolatingTerms:
      intensityFormula = f"std::real({intensityFormula})"
    if printFormula:
      print(f"Intensity formula = {intensityFormula}")
    return intensityFormula

def constructMomentResultFrom(
  indices:      MomentIndices,  # index mapping and iterators
  momentValues: Sequence[MomentValue],
) -> MomentResult:
  """Constructs `MomentResult` instance from given moment values; !Note! only diagonal elements of covariance matrix are set"""
  # ensure that all moment values have the same bin centers, labels, and number of bootstrap samples
  assert all(momentValues[0].binCenters         == momentValue.binCenters         for momentValue in momentValues[1:]), "Moment values must have the same bin centers"
  assert all(momentValues[0].label              == momentValue.label              for momentValue in momentValues[1:]), "Moment values must have the same label"
  assert all(momentValues[0].bsSamples.shape[0] == momentValue.bsSamples.shape[0] for momentValue in momentValues[1:]), "Moment values must have the same number of bootstrap samples"
  # ensure that quantum numbers of moment values are unique
  assert len(set(momentValue.qn for momentValue in momentValues)) == len(momentValues), "Moment values must have unique quantum numbers"
  momentResult = MomentResult(
    indices              = indices,
    binCenters           = momentValues[0].binCenters,
    label                = momentValues[0].label,
    _nmbBootstrapSamples = momentValues[0].bsSamples.shape[0],
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
        momentResult._valsFlatIndex[flatIndex] = momentValue.val
        #!NOTE! covariance information is not contained in `MomentValue`; only diagonal elements are set
        momentResult._V_ReReFlatIndex[flatIndex, flatIndex] = momentValue.uncertRe**2
        momentResult._V_ImImFlatIndex[flatIndex, flatIndex] = momentValue.uncertIm**2
        if momentResult.nmbBootstrapSamples > 0:
          momentResult._bsSamplesFlatIndex[flatIndex] = momentValue.bsSamples
        break
    if not foundMomentValue:
      print(f"Warning: no moment value found for {qnIndex}. Assuming value of 0.")
  momentResult.valid = True
  return momentResult

#TODO add member function to calculate chi2 of another MomentResult w.r.t. self


@dataclass
class MomentResultsKinematicBinning:
  """Container class that stores and provides access to moment values for several kinematic bins"""
  moments: list[MomentResult]  # data for all bins of the kinematic binning

  def hasSameMomentIndicesAndBinCenters(
    self,
    other: MomentResultsKinematicBinning,
  ) -> bool:
    """Returns whether two `MomentResultsKinematicBinning` objects have the same indices and the same bin centers in the same order"""
    if not isinstance(other, MomentResultsKinematicBinning):
      raise TypeError(f"Expect 'other' to be of type 'MomentResultsKinematicBinning'; got '{type(other)}' instead")
    return all(self[binIndex].hasSameMomentIndicesAndBinCenters(other[binIndex]) for binIndex in range(len(self)))

  def __eq__(
    self,
    other: MomentResultsKinematicBinning,
  ) -> bool:
    """Returns whether two `MomentResultsKinematicBinning` objects are equal in the sense that they have the same indices, bin centers moment values, and uncertainties"""
    if not isinstance(other, MomentResultsKinematicBinning):
      return NotImplemented
    if len(self) != len(other):
      return False
    return all(self[binIndex] == other[binIndex] for binIndex in range(len(self)))

  def scaleBy(
    self,
    factor: float,
  ) -> None:
    """Scales all `MomentResults` by given factor"""
    for moment in self:
      moment.scaleBy(factor)

  def normalizeTo(
    self,
    other:        MomentResultsKinematicBinning,
    normBinIndex: int | None = None,  # if `None` moments are normalized to the integral of H_0(0, 0) over all bins, otherwise the given bin is used
  ) -> float:
    """Normalizes these moments in all bins such that H_0(0, 0) is equal to H_0(0, 0) in the other moments either in the given bin or integrated over all bins; returns normalization factor"""
    H000Index = QnMomentIndex(momentIndex = 0, L = 0, M =0)
    normalizationFactor = 1.0
    if normBinIndex is None:
      # normalize to integral over all bins
      # calculate integral by looping over all bins and summing up H_0(0, 0) values
      H000SumThis = H000SumOther = 0.0
      for HThis, HOther in zip(self, other):
        H000SumThis  += HThis [H000Index].val.real
        H000SumOther += HOther[H000Index].val.real
      normalizationFactor = H000SumOther / H000SumThis
    else:
      # normalize to given bin
      H000ValueThis  = self [normBinIndex][H000Index].val.real
      H000ValueOther = other[normBinIndex][H000Index].val.real
      normalizationFactor = H000ValueOther / H000ValueThis
    self.scaleBy(normalizationFactor)
    return normalizationFactor

  # special methods to implement *=, *, +=, +, -=, and - operators
  def __imul__(
    self,
    scalar: int | float,
  ) -> MomentResultsKinematicBinning:
    """Multiplication of all `MomentResults` with a scalar"""
    for momentResult in self:
      momentResult *= scalar
    return self

  def __mul__(
    self,
    scalar: int | float,
  ) -> MomentResultsKinematicBinning:
    """Multiplication with a scalar from the right and returning a new `MomentResultsKinematicBinning` instance"""
    product = copy.deepcopy(self)
    product *= scalar
    return product

  def __rmul__(
    self,
    scalar: int | float,
  ) -> MomentResultsKinematicBinning:
    """Multiplication with a scalar from the left and returning a new `MomentResultsKinematicBinning` instance"""
    return self.__mul__(scalar)

  def __iadd__(
    self,
    other: MomentResultsKinematicBinning,
  ) -> MomentResultsKinematicBinning:
    """Combines this and another `MomentResultsKinematicBinning` instance by summing the `MomentResults` for each kinematic bin"""
    if not isinstance(other, MomentResultsKinematicBinning):
      return NotImplemented
    # ensure that `other` has the same kinematic bins
    assert self.hasSameMomentIndicesAndBinCenters(other), "Moment results must have the same moment indices and bin centers in the same order"
    for binIndex, momentResult in enumerate(self):
      momentResult += other[binIndex]
    return self

  def __add__(
    self,
    other: MomentResultsKinematicBinning,
  ) -> MomentResultsKinematicBinning:
    """Combines this and another `MomentResultsKinematicBinning` into a new `MomentResultsKinematicBinning` by summing the `MomentResults` for each kinematic bin"""
    sum = copy.deepcopy(self)
    sum += other
    return sum

  def __isub__(
    self,
    other: MomentResultsKinematicBinning,
  ) -> MomentResultsKinematicBinning:
    """Combines this and another `MomentResultsKinematicBinning` into one `MomentResultsKinematicBinning` by subtracting the `MomentResults` for each kinematic bin"""
    if not isinstance(other, MomentResultsKinematicBinning):
      return NotImplemented
    self += (-1 * other)
    return self

  def __sub__(
    self,
    other: MomentResultsKinematicBinning,
  ) -> MomentResultsKinematicBinning:
    """Combines this and another `MomentResult` into one `MomentResult` by subtracting the the `MomentResults` for each kinematic bin"""
    if not isinstance(other, MomentResultsKinematicBinning):
      return NotImplemented
    return self + (-1 * other)

  # make MomentResultsKinematicBinning behave like a list of MomentResults
  def __len__(self) -> int:
    """Returns number of kinematic bins"""
    return len(self.moments)

  def __getitem__(
    self,
    subscript: int,
  ) -> MomentResult:
    """Returns `MomentResult` that correspond to given bin index"""
    return self.moments[subscript]

  def __setitem__(
    self,
    subscript: int,
    value:     MomentResult,
  ) -> None:
    """Sets `MomentResult` at given bin index"""
    self.moments[subscript] = value

  def __iter__(self) -> Iterator[MomentResult]:
    """Iterates over `MomentResults` in kinematic bins"""
    return iter(self.moments)

  def append(
    self,
    momentResult: MomentResult,
  ) -> None:
    """Appends a `MomentResult` to the list of kinematic bins"""
    self.moments.append(momentResult)

  @property
  def binCenters(self) -> tuple[dict[KinematicBinningVariable, float], ...]:
    """Returns tuple with bin centers of all moments"""
    return tuple(momentResult.binCenters for momentResult in self)

  def normalize(self) -> None:
    """Scales all `MomentResults` such that in each bin H_0(0, 0) = 1"""
    for moment in self:
      moment.normalize()

  def savePickle(
    self,
    pickleFileName: str,
  ) -> None:
    """Saves `MomentResultsKinematicBinning` to pickle file"""
    with open(pickleFileName, "wb") as file:
      pickle.dump(self, file)
    print(f"Wrote moment results to file '{pickleFileName}'")

  @classmethod
  def loadPickle(
    cls,
    pickleFileName: str,
  ) -> MomentResultsKinematicBinning:
    """Loads `MomentResultsKinematicBinning` from pickle file"""
    with open(pickleFileName, "rb") as file:
      return pickle.load(file)
    print(f"Loaded moment results from file '{pickleFileName}'")

  def toJsonStr(self) -> str:
    """Returns JSON string with valid moment values in all kinematic bins"""
    return json.dumps(
      [
        momentValue.toJsonDict() for momentResult in self if momentResult
                                 for momentValue in momentResult.values
      ],
      indent = 2,
    )


@dataclass
class BootstrapIndices:
  """Generator class for bootstrap indices"""
  nmbEvents:  int  # number of events in data sample
  nmbSamples: int  # number of bootstrap samples
  _seed:      int  # seed for random number generator
  _rng:       np.random.Generator = field(init = False, repr = False)

  def __post_init__(self) -> None:
    """Initializes the random number generator with the given seed"""
    self.seed = self._seed

  @property
  def seed(self) -> int:
    """Returns the seed for the random number generator"""
    return self._seed

  @seed.setter
  def seed(
    self,
    value: int,
  ) -> None:
    """Sets the seed for the random number generator and reinitializes the internal random number generator"""
    self._seed = value
    self._rng = np.random.default_rng(self._seed)

  def __iter__(self) -> Generator[npt.NDArray[npt.Shape["nmbEvents"], npt.Int64], None, None]:
    """Generates randomized event indices for each bootstrap sample"""
    for _ in range(self.nmbSamples):
      indices = self._rng.choice(self.nmbEvents, size = self.nmbEvents, replace = True)
      indices.sort()  # sort to improve performance of memory access when indices are applied to arrays
      yield indices


@dataclass
class MomentCalculator:
  """Container class that holds all information needed to calculate moments for a single kinematic bin"""
  indices:              MomentIndices  # index mapping and iterators
  dataSet:              DataSet  # info on data samples
  binCenters:           dict[KinematicBinningVariable, float]  # dictionary with center values of kinematic variables that define bin
  integralFileBaseName: str                             = "./integralMatrix"  # naming scheme for integral files is '<integralFileBaseName>_[<binning var>_<bin center>_...].npy'
  flipSignOfWeights:    bool                            = False  # if set, weights of real-data event are sign-flipped
  _HMeas:               MomentResult | None             = None   # measured moments; must either be given or calculated by calling calculateMoments()
  _HPhys:               MomentResult | None             = None   # physical moments; must either be given or calculated by calling calculateMoments()
  _integralMatrix:      AcceptanceIntegralMatrix | None = None   # if None no acceptance correction is performed; must either be given or calculated by calling calculateIntegralMatrix()

  def __post_init__(self) -> None:
    # set polarized moments case of `indices` according to info provided by `dataSet`
    if self.dataSet.polarization is None:
      self.indices.setPolarized(False)
    else:
      self.indices.setPolarized(True)
    # initialize _HMeas and _HPhys to empty `MomentResults` if None
    if self._HMeas is None:
      self._HMeas = MomentResult(
        indices    = self.indices,
        binCenters = self.binCenters,
        label      = "meas",
      )
    if self._HPhys is None:
      self._HPhys = MomentResult(
        indices    = self.indices,
        binCenters = self.binCenters,
        label      = "phys",
      )

  # accessors that guarantee existence of optional fields
  @property
  def HMeas(self) -> MomentResult:
    assert self._HMeas is not None, "HMeas must not be None"
    return self._HMeas

  @HMeas.setter
  def HMeas(self, value: MomentResult) -> None:
    self._HMeas = value

  @property
  def HPhys(self) -> MomentResult:
    assert self._HPhys is not None, "HPhys must not be None"
    return self._HPhys

  @HPhys.setter
  def HPhys(self, value: MomentResult) -> None:
    self._HPhys = value

  @property
  def integralMatrix(self) -> AcceptanceIntegralMatrix:
    """Returns acceptance integral matrix"""
    assert self._integralMatrix is not None, "self._integralMatrix must not be None"
    return self._integralMatrix

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

  class DataSourceType(Enum):
    REAL_DATA            = auto()
    ACCEPTED_PHASE_SPACE = auto()

  def calculateMoments(
    self,
    dataSourceType:            DataSourceType = DataSourceType.REAL_DATA,  # type of data to calculate moments from
    normalize:                 bool           = True,   # if set, physical moments are normalized to H_0(0, 0)
    applyCovPoissonCorrection: bool           = True,   # if set, Poissonian fluctuation of number of events is taken into account  #TODO add this to `AnalysisConfig`?
    nmbBootstrapSamples:       int            = 0,      # number of bootstrap samples; 0 means no bootstrapping
    bootstrapRandomSeed:       int            = 12345,  # seed used for random number generator used for bootstrap samples
  ) -> None:
    """Calculates photoproduction moments and their covariances using given data source"""
    # define dataset and integral matrix to use for moment calculation
    dataSet        = None
    integralMatrix = self._integralMatrix
    if dataSourceType == self.DataSourceType.REAL_DATA:
      # calculate moments of data
      dataSet = self.dataSet
    elif dataSourceType == self.DataSourceType.ACCEPTED_PHASE_SPACE:
      # calculate moments of acceptance-corrected phase space; physical moments should all be 0 except H_0(0, 0)
      dataSet = dataclasses.replace(self.dataSet, data = self.dataSet.phaseSpaceData)
    else:
      raise ValueError(f"Unknown data source '{dataSourceType}'")
    print("Reading input data for moment calculation")
    beamPol, thetas, phis, Phis, eventWeights = readInputData(
      polarization      = dataSet.polarization,
      data              = dataSet.data,
      flipSignOfWeights = self.flipSignOfWeights,
    )
    nmbEvents = thetas.size()
    # calculate basis-function values and values of measured moments
    nmbMoments = len(self.indices)
    fMeas: npt.NDArray[npt.Shape["nmbMoments, nmbEvents"], npt.Complex128] = np.empty((nmbMoments, nmbEvents), dtype = np.complex128)
    bootstrapIndices = BootstrapIndices(nmbEvents, nmbBootstrapSamples, bootstrapRandomSeed)
    self.HMeas.nmbBootstrapSamples = nmbBootstrapSamples
    self.HMeas.bootstrapRandomSeed = bootstrapRandomSeed
    for flatIndex in self.indices.flatIndices:
      qnIndex = self.indices[flatIndex]
      fMeas[flatIndex] = np.asarray(ROOT.f_meas(
        qnIndex.momentIndex, qnIndex.L, qnIndex.M,
        thetas,
        phis,
        Phis,
        beamPol,
      ))  # Eq. (176)
      weightedSum = eventWeights.dot(fMeas[flatIndex])
      self.HMeas._valsFlatIndex[flatIndex] = 2 * np.pi * weightedSum  # Eq. (179)
      # perform bootstrapping of HMeas
      if nmbBootstrapSamples > 0:
        print(f"Calculating {nmbBootstrapSamples} bootstrap samples for measured moment {self.indices[flatIndex].label}")
      for bsSampleIndex, bsDataIndices in enumerate(bootstrapIndices):  # loop over same set of random data indices for each flatIndex
        # resample data
        fMeasBsSample        = fMeas[flatIndex][bsDataIndices]
        eventWeightsBsSample = eventWeights    [bsDataIndices]
        # calculate bootstrap sample
        self.HMeas._bsSamplesFlatIndex[flatIndex, bsSampleIndex] = 2 * np.pi * eventWeightsBsSample.dot(fMeasBsSample)
    # calculate covariance matrices for measured moments; Eqs. (88), (180), and (181) #TODO update eq numbers
    fMeasWeighted = eventWeights * fMeas
    Vmeas_aug = (2 * np.pi)**2 * nmbEvents * np.cov(fMeasWeighted, np.conjugate(fMeasWeighted), ddof = 1)
    if applyCovPoissonCorrection:
      # add correction that accounts for Poissonian fluctuations of number of events; Eqs. (123) and (124)
      fMeasWeightedMean = np.mean(fMeasWeighted, axis = 1)
      fMeasWeightedMean_aug = np.block([fMeasWeightedMean, np.conjugate(fMeasWeightedMean)])
      Vmeas_aug += (2 * np.pi)**2 * nmbEvents * np.outer(fMeasWeightedMean_aug, np.conjugate(fMeasWeightedMean_aug))
    self.HMeas.setCovarianceMatricesFrom(Vmeas_aug)
    # calculate physical moments and propagate uncertainty
    self.HPhys.nmbBootstrapSamples = nmbBootstrapSamples
    self.HPhys.bootstrapRandomSeed = bootstrapRandomSeed
    Vphys_aug = np.empty(Vmeas_aug.shape, dtype = np.complex128)
    if integralMatrix is None:
      # ideal detector: physical moments are identical to measured moments
      np.copyto(self.HPhys._valsFlatIndex, self.HMeas._valsFlatIndex)
      np.copyto(Vphys_aug, Vmeas_aug)
      np.copyto(self.HPhys._bsSamplesFlatIndex, self.HMeas._bsSamplesFlatIndex)
    else:
      # get inverse of acceptance integral matrix
      Iinv = integralMatrix.inverse
      # calculate physical moments, i.e. correct for detection efficiency
      self.HPhys._valsFlatIndex = Iinv @ self.HMeas._valsFlatIndex  # Eq. (83)
      # calculate bootstrap samples for H_phys
      if nmbBootstrapSamples > 0:
        print(f"Calculating {nmbBootstrapSamples} bootstrap samples for physical moments")
      for bsSampleIndex in range(nmbBootstrapSamples):
        self.HPhys._bsSamplesFlatIndex[:, bsSampleIndex] = Iinv @ self.HMeas._bsSamplesFlatIndex[:, bsSampleIndex]
      # perform linear uncertainty propagation
      J = Iinv  # Jacobian of efficiency correction; Eq. (101)
      Jconj = np.zeros((nmbMoments, nmbMoments), dtype = np.complex128)  # conjugate Jacobian; Eq. (101)
      J_aug = np.block([
        [J,                   Jconj],
        [np.conjugate(Jconj), np.conjugate(J)],
      ])  # augmented Jacobian; Eq. (98)
      Vphys_aug = J_aug @ (Vmeas_aug @ np.asmatrix(J_aug).H)  #!NOTE! @ is left-associative; Eq. (85)
    self.HPhys.setCovarianceMatricesFrom(Vphys_aug)
    if normalize:
      self.HPhys.normalize()
    self.HMeas.valid = True
    self.HPhys.valid = True


  @dataclass
  class _ExtendedUnbinnedWeightedNLL:
    """Picklable functor that calculates negative log-likelihood function for extended unbinned maximum likelihood fit with weighted events"""
    momentCalculator:            InitVar[MomentCalculator]  # pass instance of outer class as temporary argument
    printNonPositiveIntensities: bool                                                         = False
    _eventWeights:               npt.NDArray[npt.Shape["nmbEvents"],             npt.Float64] = field(init = False)  # weights of real-data events
    _phaseSpaceEfficiency:       float                                                        = field(init = False)  # efficiency averaged over phase space
    _baseFcnVals:                npt.NDArray[npt.Shape["nmbMoments, nmbEvents"], npt.Float64] = field(init = False)  # precalculated real-data values of basis functions
    _integralVector:             npt.NDArray[npt.Shape["nmbMoments"],            npt.Float64] = field(init = False)  # precalculated acceptance integral vector
    flipSignOfWeights:           bool                                                         = False  # if set, real-data event weights are sign-flipped

    def _calculateBasisFcnValues(
      self,
      momentCalculator: MomentCalculator,  # instance of outer class
    ) -> None:
      """Calculates values of basis functions for all real-data events"""
      print("Reading real data for moment fit")
      beamPol, thetas, phis, Phis, self._eventWeights = readInputData(
        polarization      = momentCalculator.dataSet.polarization,
        data              = momentCalculator.dataSet.data,
        flipSignOfWeights = self.flipSignOfWeights,
      )
      nmbEvents  = len(thetas)
      nmbMoments = len(momentCalculator.indices)
      print(f"Calculating values of basis functions for {nmbMoments} moments and {nmbEvents} real-data events")
      self._baseFcnVals = np.zeros((nmbMoments, nmbEvents), dtype = np.double)
      for flatIndex in momentCalculator.indices.flatIndices:
        qnIndex = momentCalculator.indices[flatIndex]
        self._baseFcnVals[flatIndex] = np.asarray(ROOT.f_basis(
          qnIndex.momentIndex, qnIndex.L, qnIndex.M,
          thetas,
          phis,
          Phis,
          beamPol,
        ))

    def _calculateIntegralVector(
      self,
      momentCalculator: MomentCalculator,  # instance of outer class
    ) -> None:
      """Calculates integrals of all basis functions from accepted phase-space events"""
      nmbMoments = len(momentCalculator.indices)
      if momentCalculator.dataSet.phaseSpaceData is None:
        print("Warning: no phase-space data; using perfect acceptance")
        self._integralVector = np.zeros((nmbMoments, ), dtype = np.double)
        H000Index: int = momentCalculator.indices[QnMomentIndex(momentIndex = 0, L = 0, M = 0)]
        self._integralVector[H000Index] = 1.0
        self._phaseSpaceEfficiency      = 1.0
        return
      print("Reading phase-space data for the calculation of the acceptance integral vector")
      #TODO add case with perfect acceptance when accepted phase-space data are not provided
      beamPolAccPs, thetasAccPs, phisAccPs, PhisAccPs, eventWeightsAccPs = readInputData(
        polarization = momentCalculator.dataSet.polarization,
        data         = momentCalculator.dataSet.phaseSpaceData,
      )
      nmbEventsAccPs = len(thetasAccPs)
      self._phaseSpaceEfficiency = nmbEventsAccPs / momentCalculator.dataSet.nmbGenEvents  # efficiency averaged over phase space
      print(f"Calculating acceptance integral vector for {nmbMoments} moments from {nmbEventsAccPs} accepted phase-space events")
      self._integralVector = np.zeros((nmbMoments, ), dtype = np.double)
      for flatIndex in momentCalculator.indices.flatIndices:
        # calculate basis-functions value for each accepted phase-space event
        qnIndex = momentCalculator.indices[flatIndex]
        baseFcnValsAccPs: npt.NDArray[npt.Shape["nmbAccEvents"], npt.Float64] = np.asarray(ROOT.f_basis(
          qnIndex.momentIndex, qnIndex.L, qnIndex.M,
          thetasAccPs,
          phisAccPs,
          PhisAccPs,
          beamPolAccPs,
        ))
        # calculate accepted phase-space integral by summing basis-functions values over accepted phase-space events
        #TODO why prefactor is not 8 * np.pi**2 / N?
        self._integralVector[flatIndex] = ((4 * np.pi / momentCalculator.dataSet.nmbGenEvents)
          * np.sum(np.sort(np.multiply(eventWeightsAccPs, baseFcnValsAccPs))))  # sorting values reduces rounding errors

    def __post_init__(
      self,
      momentCalculator: MomentCalculator,
    ) -> None:
      """Loads input data and precalculates basis-function values and acceptance integral vector"""
      # precalculate the basis-function values and acceptance integral vector
      self._calculateBasisFcnValues(momentCalculator)
      self._calculateIntegralVector(momentCalculator)

    def _intensityFcn(
      self,
      moments: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64]  # fit parameters = all non-zero moment values: [{Re H_0}, {Re H_1}, {Im H_2}]
    ) -> tuple[np.float64, npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]]:
      """Returns the integral of the intensity function and the intensities for each event"""
      # calculate intensities for all events
      intensities: npt.NDArray[npt.Shape["nmbEvents"], npt.Float64] = np.dot(moments, self._baseFcnVals)
      # calculate integral value
      integral = np.dot(moments, self._integralVector)
      return (integral, intensities)

    def __call__(
      self,
      moments: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64]  # fit parameters = all non-zero moment values: [{Re H_0}, {Re H_1}, {Im H_2}]
    ) -> np.float64:
      """Returns negative log-likelihood function for given moment parameters"""
      integral, intensities = self._intensityFcn(moments)
      if self.printNonPositiveIntensities:
        nonPositiveIntensities = intensities[intensities <= 0]
        nonPositiveIndices     = np.where(intensities <= 0)[0]
        if nonPositiveIntensities.size > 0:
          print(f"Warning: ignoring non-positive intensities: {nonPositiveIntensities} with event weights {self._eventWeights[nonPositiveIndices]}")
      weightedLogIntensities = self._eventWeights * np.log(intensities + np.finfo(dtype = float).tiny)  # protect against 0 intensities
      return -(np.sum(np.sort(weightedLogIntensities)).item() - integral)  # sorting summands reduces rounding errors; use item() to ensure that sum is scalar quantity

    @property
    def nmbSignalEvents(self) -> float:
      """Returns number of signal events"""
      return np.sum(self._eventWeights)

    @property
    def phaseSpaceEfficiency(self) -> float:
      """Returns efficiency averaged over phase space"""
      return self._phaseSpaceEfficiency

  def negativeLogLikelihoodFcn(self) -> _ExtendedUnbinnedWeightedNLL:
    """Constructs negative log-likelihood function for extended unbinned maximum likelihood fit with weighted events; should be called only when data change"""
    return self._ExtendedUnbinnedWeightedNLL(self, flipSignOfWeights = self.flipSignOfWeights)

  @classmethod
  def fitMoments(
    cls,
    negativeLogLikelihoodFcn: Cost,  # function to minimize
    startValues:              npt.NDArray[npt.Shape["nmbMoments"], npt.Float64],  # initial values for fit parameters
    indices:                  MomentIndices,  # indices that define set of moments to fit
    minuit:                   im.Minuit | None              = None,   # use provided Minuit object for reentrant fitting; if None, a new Minuit object is created
    fixMomentsToZero:         Sequence[int | QnMomentIndex] = [],     # list of moment indices, for which the fit parameters are fixed to zero
  ) -> im.Minuit:
    """Estimates photoproduction moments and their covariances by fitting intensity model to data from the given source"""
    if minuit is None:
      # setup minimizer
      momentLabels = tuple(qnIndex.label for qnIndex in indices.qnIndices)
      minuit = im.Minuit(negativeLogLikelihoodFcn, startValues, name = momentLabels)
      minuit.errordef = im.Minuit.LIKELIHOOD
    if fixMomentsToZero:
      # fix parameters corresponding to moment indices to 0
      momentLabelsToFix = [
        index.label if isinstance(index, QnMomentIndex) else indices[index].label
        for index in fixMomentsToZero
      ]
      for label in momentLabelsToFix:
        minuit.fixto(label, 0)
    # perform fit
    minuit.migrad()
    if minuit.valid:
      minuit.hesse()  # spend time on HESSE only for converged fits
    return minuit

  @classmethod
  def fitSuccess(
    cls,
    minuit:  im.Minuit,
    verbose: bool = False,
  ) -> bool:
    """Returns whether fit was successful"""
    if verbose:
      print(textwrap.dedent(
        f"""
        Fit success:
            {minuit.valid=}
            {minuit.fmin.has_covariance=}
            {minuit.fmin.has_accurate_covar=}
            {minuit.fmin.has_posdef_covar=}
            {minuit.fmin.has_made_posdef_covar=}
            {minuit.fmin.hesse_failed=}
            {minuit.fmin.is_above_max_edm=}
            {minuit.fmin.has_reached_call_limit=}
            {minuit.fmin.has_parameters_at_limit=}
        """
      ))
    if (
                minuit.valid
        and     minuit.fmin.has_covariance
        and     minuit.fmin.has_accurate_covar
        and     minuit.fmin.has_posdef_covar
        and not minuit.fmin.has_made_posdef_covar
        and not minuit.fmin.hesse_failed
        and not minuit.fmin.is_above_max_edm
        and not minuit.fmin.has_reached_call_limit
        and not minuit.fmin.has_parameters_at_limit
    ):
      return True
    return False

  @classmethod
  def selectSuccessfulFits(
    cls,
    fitMinuits:      list[im.Minuit],  # list of Minuit objects containing the fit results
    printFitResults: bool = True,
  ) -> list[im.Minuit]:
    """Selects successful fits from list of Minuit objects"""
    # select successful fits
    successfulFits = []
    nmbFitAttempts = len(fitMinuits)
    for fitAttemptIndex, minuit in enumerate(fitMinuits):
      if printFitResults:
        print(minuit.fmin)
        print(minuit.params)
        print(minuit.merrors)
      success = MomentCalculator.fitSuccess(minuit)
      print(f"Fit [{fitAttemptIndex + 1} of {nmbFitAttempts}] was {'' if success else 'NOT '}successful")
      if success:
        successfulFits.append(minuit)
    print(f"{len(successfulFits)} out of {nmbFitAttempts} fit attempts were successful")
    return successfulFits

  def _convertIminuitToMomentResult(
    self,
    minuit:    im.Minuit,    # iminuit object containing the fit result
    normalize: bool = True,  # if set physical moments are normalized to H_0(0, 0)
  ) -> None:
    """Fills `MomentResult` object for physical moments from iminuit object"""
    # construct empty `MomentResult` object
    # fill `HPhys` with values from `minuit`
    if not self.indices.polarized:
      # copy parameter values
      self.HPhys._valsFlatIndex  [:] = np.array(minuit.values)
      # copy covariance matrix
      self.HPhys._V_ReReFlatIndex[:] = np.array(minuit.covariance)
    else:
      # construct quantum-number index ranges that correspond to purely real and purely imaginary moments, respectively
      reIndexRange = (
        QnMomentIndex(momentIndex = 0, L = 0,                 M = 0),
        QnMomentIndex(momentIndex = 1, L = self.indices.maxL, M = self.indices.maxL),
      )  # all H_0 and H_1 moments are real-valued
      imIndexRange = (
        QnMomentIndex(momentIndex = 2, L = 1,                 M = 1),
        QnMomentIndex(momentIndex = 2, L = self.indices.maxL, M = self.indices.maxL)
      )  # all H_2 moments are purely imaginary; all H_2(L, 0) are 0
      # convert to flat-index ranges
      reSlice = slice(self.indices[reIndexRange[0]], self.indices[reIndexRange[1]] + 1)
      imSlice = slice(self.indices[imIndexRange[0]], self.indices[imIndexRange[1]] + 1)
      # copy values
      parameters = np.array(minuit.values)
      self.HPhys._valsFlatIndex[reSlice] = parameters[reSlice]
      self.HPhys._valsFlatIndex[imSlice] = parameters[imSlice] * 1j  # convert to purely imaginary
      # copy covariance matrix
      covMatrix = np.array(minuit.covariance)
      self.HPhys._V_ReReFlatIndex[reSlice, reSlice] = covMatrix[reSlice, reSlice]
      self.HPhys._V_ImImFlatIndex[imSlice, imSlice] = covMatrix[imSlice, imSlice]
      self.HPhys._V_ReImFlatIndex[reSlice, imSlice] = covMatrix[reSlice, imSlice]
    if normalize:
      self.HPhys.normalize()
    self.HPhys.valid = True

  def fitMomentsMultipleAttempts(
    self,
    nmbFitAttempts:          int,            # number of attempts to fit the moments
    nmbParallelFitProcesses: int,            # number of fit processes to run in parallel
    normalize:               bool  = False,  # if set physical moments are normalized to H_0(0, 0)
    startValueRandomSeed:    int   = 12345,  # seed used for random start values
    startValueStdDevScale:   float = 0.02,   # standard deviation of random start values is (scale parameter * number of acceptance-corrected signal events)
    processNiceLevel:        int   = 19,     # run processes with this nice level
  ) -> list[im.Minuit]:
    """Performs several attempts to fit the moments with random start values in parallel, stores best fit result in `HPhys`, and returns all fit results"""
    # construct NLL
    negativeLogLikelihoodFcn = self.negativeLogLikelihoodFcn()
    # generate random start values for each fit attempt
    startValueSets: list[npt.NDArray[npt.Shape["nmbMoments"], npt.Float64]] = []
    np.random.seed(startValueRandomSeed)
    nmbEventsSigCorr = negativeLogLikelihoodFcn.nmbSignalEvents / negativeLogLikelihoodFcn.phaseSpaceEfficiency  # number of acceptance-corrected signal events
    nmbMoments = len(self.indices)
    for _ in range(nmbFitAttempts):
      # startValues = np.zeros(nmbMoments, dtype = np.float64)  # set all start values to 0
      # generate start values that are Gaussian-distributed around 0
      startValues = np.random.normal(loc = 0, scale = startValueStdDevScale * nmbEventsSigCorr, size = nmbMoments)  #!NOTE! convergence rate is very sensitive to scale parameter
      startValues[0] = nmbEventsSigCorr  # set H_0(0, 0) to number of acceptance-corrected signal events
      startValueSets.append(startValues)
    # run fit attempts in parallel
    def increaseNiceLevel() -> None:  # need to define separate function to set `initializer` argument of `ProcessPoolExecutor`
      """Increases nice level of process that calls this function"""
      if processNiceLevel > 0:
        os.nice(processNiceLevel)
    with ProcessPoolExecutor(max_workers = nmbParallelFitProcesses, initializer = increaseNiceLevel) as executor:
      futures = [
        executor.submit(
          MomentCalculator.fitMoments,
          negativeLogLikelihoodFcn = negativeLogLikelihoodFcn,
          startValues              = startValueSets[fitAttemptIndex],
          indices                  = self.indices,
          minuit                   = None,
        )
        for fitAttemptIndex in range(nmbFitAttempts)
      ]
      fitMinuits = [future.result() for future in futures]
    # select successful fits
    successfulFitMinuits = MomentCalculator.selectSuccessfulFits(fitMinuits)
    # use successful fit with lowest NLL value to fill `HPhys`
    if successfulFitMinuits:
      # bestFitMinuit = sorted(successfulFitMinuits, lambda minuit: minuit.fmin.fval)[0]  # select fit with lowest NLL
      bestFitMinuit = successfulFitMinuits[0]  #TODO clarify why taking the first successful fit is okay
      self._convertIminuitToMomentResult(bestFitMinuit, normalize = normalize)
    else:
      print(f"Warning: No successful fits found. Cannot set physical moment values for kinematic bin {self.binCenters}.")
    return fitMinuits


# functions that read bin labels and titles from MomentCalculator or MomentResult
def binLabels(obj: MomentCalculator | MomentResult) -> list[str]:
  """Returns list of bin labels; naming scheme of entries is '<binning var>_<bin center>'"""
  return [f"{var.name}_" + (f"{center:.{var.nmbDigits}f}" if var.nmbDigits is not None else f"{center}") for var, center in obj.binCenters.items()]

def binLabel(obj: MomentCalculator | MomentResult) -> str:
  """Returns label for bin; scheme is '<binning var 0>_<bin center 0>_ ...'"""
  return "_".join(binLabels(obj))

def binTitles(obj: MomentCalculator | MomentResult) -> list[str]:
  """Returns list of `TLatex` expressions for bin centers; scheme of entries is '<binning var> = <bin center> <unit>'"""
  return [f"{var.label} = " + (f"{center:.{var.nmbDigits}f}" if var.nmbDigits is not None else f"{center}") + f" {var.unit}" for var, center in obj.binCenters.items()]

def binTitle(obj: MomentCalculator | MomentResult) -> str:
  """Returns `TLatex` expressions for kinematic bin centers; scheme is '<binning var 0> = <bin center 0> <unit>, ...'"""
  return ", ".join(binTitles(obj))


@dataclass
class MomentCalculatorsKinematicBinning:
  """Container class that holds all information needed to calculate moments for several kinematic bins"""
  calculators: list[MomentCalculator]  # data for all bins of the kinematic binning

  # make MomentCalculatorsKinematicBinning behave like a list of MomentCalculators
  def __len__(self) -> int:
    """Returns number of kinematic bins"""
    return len(self.calculators)

  def __getitem__(
    self,
    subscript: int,
  ) -> MomentCalculator:
    """Returns `MomentCalculator` that correspond to given bin index"""
    return self.calculators[subscript]

  def __iter__(self) -> Iterator[MomentCalculator]:
    """Iterates over `MomentCalculators` in kinematic bins"""
    return iter(self.calculators)

  def append(
    self,
    calculator: MomentCalculator
  ) -> None:
    """Appends a new `MomentCalculator`"""
    self.calculators.append(calculator)

  def calculateIntegralMatrices(
    self,
    forceCalculation: bool = False,
  ) -> None:
    """Calculates acceptance integral matrices for all kinematic bins"""
    for kinBinIndex, momentsInBin in enumerate(self):
      print(f"Calculating acceptance integral matrix for kinematic bin [{kinBinIndex + 1} of {len(self)}] at {momentsInBin.binCenters}")
      momentsInBin.calculateIntegralMatrix(forceCalculation)

  def calculateMoments(
    self,
    dataSourceType:      MomentCalculator.DataSourceType = MomentCalculator.DataSourceType.REAL_DATA,  # type of data to calculate moments from
    normalize:           bool                            = True,   # if set physical moments are normalized to H_0(0, 0)
    nmbBootstrapSamples: int                             = 0,      # number of bootstrap samples; 0 means no bootstrapping
    bootstrapRandomSeed: int                             = 12345,  # seed used for random number generator used for bootstrap samples
  ) -> None:
    """Calculates moments for all kinematic bins using given data source"""
    for kinBinIndex, momentsInBin in enumerate(self):
      print(f"Calculating moments for kinematic bin [{kinBinIndex + 1} of {len(self)}] at {momentsInBin.binCenters}")
      momentsInBin.calculateMoments(
        dataSourceType      = dataSourceType,
        normalize           = normalize,
        nmbBootstrapSamples = nmbBootstrapSamples,
        bootstrapRandomSeed = bootstrapRandomSeed + kinBinIndex,
      )

  def fitMomentsMultipleAttempts(
    self,
    nmbFitAttempts:          int,            # number of attempts to fit the moments
    nmbParallelFitProcesses: int,            # number of fit processes to run in parallel
    normalize:               bool  = False,  # if set physical moments are normalized to H_0(0, 0)
    startValueRandomSeed:    int   = 12345,  # seed used for random start values
    startValueStdDevScale:   float = 0.02,   # standard deviation of random start values is (scale parameter * number of acceptance-corrected signal events)
    processNiceLevel:        int   = 19,     # run processes with this nice level
  ) -> list[list[im.Minuit]]:  # Minuit objects for [<kinematic bin>][<fit attempts>]
    """Estimates photoproduction moments by fitting intensity model to each kinematic bin by running several attempts with random start values in parallel; stores the best fit result in `self` and returns fit results for all attempts"""
    fitMinuits: list[list[im.Minuit]] = [[] for _ in self]  # empty list for each kinematic bin
    for kinBinIndex, momentsInBin in enumerate(self):
      print(f"Fitting moments for kinematic bin [{kinBinIndex + 1} of {len(self)}] at {momentsInBin.binCenters}")
      fitMinuits[kinBinIndex] = momentsInBin.fitMomentsMultipleAttempts(
        nmbFitAttempts          = nmbFitAttempts,
        nmbParallelFitProcesses = nmbParallelFitProcesses,
        normalize               = normalize,
        startValueRandomSeed    = startValueRandomSeed + kinBinIndex,
        startValueStdDevScale   = startValueStdDevScale,
        processNiceLevel        = processNiceLevel,
      )
    return fitMinuits

  @property
  def momentResultsMeas(self) -> MomentResultsKinematicBinning:
    """Returns `MomentResultsKinematicBinning` with all measured moments"""
    return MomentResultsKinematicBinning([momentsInBin.HMeas for momentsInBin in self])

  @property
  def momentResultsPhys(self) -> MomentResultsKinematicBinning:
    """Returns `MomentResultsKinematicBinning` with all physical moments"""
    return MomentResultsKinematicBinning([momentsInBin.HPhys for momentsInBin in self])
