"""
Module that provides utility functions converting `MomentCalculator`
classes to and from JSON dictionaries that can be serialized to JSON
strings using `json.dumps()`.
"""

from __future__ import annotations

import functools
import json
import nptyping as npt
import numpy as np
from typing import (
  Any,
  Callable,
)

from moments.MomentCalculator import (
  KinematicBinningVariable,
  MomentIndices,
  MomentResult,
  MomentResultsKinematicBinning,
  MomentValue,
  QnMomentIndex,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# use type-based dispatch to convert objects to JSON-serializable dictionaries
@functools.singledispatch
def toJsonDict(obj: Any) -> Any:
  """Converts an object to something JSON-serializable. Raise TypeError if type of obj unknown"""
  raise TypeError(f"No JSON serializer registered for type {type(obj).__name__}")


# per-type serialization logic
@toJsonDict.register
def _(obj: QnMomentIndex) -> dict[str, Any]:
  """Returns dictionary for moment quantum-number index that can be serialized to JSON"""
  return {
    "type"        : "QnMomentIndex",
    "momentIndex" : obj.momentIndex,
    "L"           : obj.L,
    "M"           : obj.M,
  }


@toJsonDict.register
def _(obj: MomentIndices) -> dict[str, Any]:
  """Returns dictionary with moment indices that can be serialized to JSON"""
  return {
    "type"      : "MomentIndices",
    "maxL"      : obj.maxL,
    "polarized" : obj.polarized,
    "qnIndices" : [toJsonDict(qn) for qn in obj.qnIndices],  # store list of quantum-number indices to preserve order
  }


# single dispatch does not work for types in containers like dict, so we need a separate function for binCenters
def _toJsonDict_binCenters(binCenters: dict[KinematicBinningVariable, float]) -> dict[str, Any]:
  """Returns dictionary with kinematic bin centers that can be serialized to JSON"""
  jsonDict: dict[str, Any] = {"type" : "binCenters"}
  jsonDict |= {
    "binCenters" : [
      {
        "varName"     : kinVar.name,
        "centerValue" : binCenter,
        "label"       : kinVar.label,
        "unit"        : kinVar.unit,
        "nmbDigits"   : kinVar.nmbDigits,
      }
    ] for kinVar, binCenter in binCenters.items()
  }
  return jsonDict


@toJsonDict.register
def _(obj: MomentValue) -> dict[str, Any]:
  """Returns dictionary with moment value and uncertainty that can be serialized to JSON"""
  return {
    "type"        : "MomentValue",
    "qn"          : toJsonDict(obj.qn),
    "valRe"       : obj.val.real,
    "uncertRe"    : obj.uncertRe,
    "valIm"       : obj.val.imag,
    "uncertIm"    : obj.uncertIm,
    "binCenters"  : _toJsonDict_binCenters(obj.binCenters),
  }


def verifyNdarray(
  a:     npt.NDArray[npt.Shape["*, ..."], Any],
  nDim:  int,
  dtype: np.dtype[Any],
) -> bool:
  """Verifies whether the given array is a NumPy ndarray with the specified number of dimensions and data type, and contains only finite values"""
  if not isinstance(a, np.ndarray):
    return False
  if a.ndim != nDim:
    return False
  if a.dtype != dtype:
    return False
  if not np.isfinite(a).all():
    return False
  return True


def _toJsonDict_ndarray1DComplex(obj: npt.NDArray[npt.Shape["*"], npt.Complex128]) -> dict[str, Any]:
  """
  Converts given 1D NumPy ndarray of dtype complex128 to a dictionary that can be serialized to JSON.

  JSON layout:
  {
    "type"  : "ndarray1DComplex",
    "dtype" : "complex128",
    "order" : "row-major",
    "shape" : [rows],
    "data"  : [{"re": ..., "im": ...}, ...]  # list of complex numbers represented as objects
  }

  Raises:
    ValueError: if the array is not 1D or contains NaN/Inf (strict JSON disallows them).
  """
  if not verifyNdarray(obj, nDim = 1, dtype = np.complex128):
    raise ValueError(f"Expected a 1D NumPy array of complex128, but got = {repr(obj)}")
  return {
    "type"  : "ndarray1DComplex",
    "dtype" : obj.dtype.name,
    "order" : "row-major",
    "shape" : [int(obj.shape[0])],
    "data"  : [{"re": float(value.real), "im": float(value.imag)} for value in obj],
  }


def _toJsonDict_ndarray2DFloat(obj: npt.NDArray[npt.Shape["*, *"], npt.Float64]) -> dict[str, Any]:
  """
  Converts given 2D NumPy ndarray of dtype float64 to a dictionary that can be serialized to JSON.

  JSON layout:
  {
    "type"  : "ndarray2DFloat",
    "dtype" : "float64",
    "order" : "row-major",
    "shape" : [rows, cols],
    "data"  : [[...], [...], ...]  # nested lists (rows of doubles)
  }

  Raises:
    ValueError: if the array is not 2D or contains NaN/Inf (strict JSON disallows them).
  """
  if not verifyNdarray(obj, nDim = 2, dtype = np.float64):
    raise ValueError(f"Expected a 2D NumPy array of float64, but got = {repr(obj)}")
  return {
    "type"  : "ndarray2DFloat",
    "dtype" : obj.dtype.name,
    "order" : "row-major",
    "shape" : [int(obj.shape[0]), int(obj.shape[1])],
    "data"  : obj.tolist(),  # nested lists of Python floats
  }


@toJsonDict.register
def _(obj: MomentResult) -> dict[str, Any]:
  """Returns dictionary for moment result that can be serialized to JSON"""
  return {
    "type"             : "MomentResult",
    "indices"          : toJsonDict(obj.indices),
    "binCenters"       : _toJsonDict_binCenters(obj.binCenters),
    "_valsFlatIndex"   : _toJsonDict_ndarray1DComplex(obj._valsFlatIndex),
    "_V_ReReFlatIndex" : _toJsonDict_ndarray2DFloat(obj._V_ReReFlatIndex),
    "_V_ImImFlatIndex" : _toJsonDict_ndarray2DFloat(obj._V_ImImFlatIndex),
    "_V_ReImFlatIndex" : _toJsonDict_ndarray2DFloat(obj._V_ReImFlatIndex),
    "valid"            : obj.valid,
  }


@toJsonDict.register
def _(obj: MomentResultsKinematicBinning) -> dict[str, Any]:
  """Returns dictionary with list moment results in all kinematic bins, which can be serialized to JSON"""
  return {
    "type"    : "MomentResultsKinematicBinning",
    "results" : [toJsonDict(momentResult) for momentResult in obj],
  }


# generic wrapper encoder that uses toJsonDict() to convert objects to JSON-serializable dictionaries
class _MyEncoder(json.JSONEncoder):
  """
  A JSONEncoder that asks toJsonDict() to convert objects. If
  toJsonDict() knows the type, we return a JSON-serializable value;
  otherwise, we let the base encoder raise.
  """
  def default(
    self,
    obj: Any,
  ) -> Any:
    try:
      return toJsonDict(obj)
    except TypeError:
      # defer to the standard behavior (which will raise)
      return super().default(obj)


def toJsonStr(
  obj: Any,
  **kwargs,
) -> str:
  """Convenience wrapper around json.dumps() using _MyEncoder"""
  return json.dumps(obj, cls = _MyEncoder, **kwargs)


# use value-based dispatch to convert JSON-serializable dictionaries back to objects
# deserializer registry keyed by the "type" tag
_FROM_JSON_DICT_FUNCS: dict[str, Callable[[dict], Any]] = {}

def registerFromJsonDictFunc(tag: str):
  """Decorator to register a deserializer function for a tagged dictionary"""
  def deco(func: Callable[[dict], Any]) -> Callable[[dict], Any]:
    _FROM_JSON_DICT_FUNCS[tag] = func
    return func
  return deco


# per-type reconstruction logic
@registerFromJsonDictFunc("QnMomentIndex")
def _fromJsonDict_QnMomentIndex(jsonDict: dict[str, Any]) -> QnMomentIndex:
  """Returns moment quantum numbers constructed from a JSON-serializable dictionary"""
  return QnMomentIndex(
    momentIndex = jsonDict["momentIndex"],
    L           = jsonDict["L"],
    M           = jsonDict["M"],
  )


@registerFromJsonDictFunc("MomentIndices")
def _fromJsonDict_MomentIndices(jsonDict: dict[str, Any]) -> MomentIndices:
  """Returns moment quantum numbers constructed from a JSON-serializable dictionary"""
  return MomentIndices(
    maxL      = jsonDict["maxL"],
    polarized = jsonDict["polarized"],
  )


@registerFromJsonDictFunc("binCenters")
def _fromJsonDict_binCenters(jsonDict: dict[str, Any]) -> dict[KinematicBinningVariable, float]:
  """Returns bin centers constructed from a JSON-serializable dictionary"""
  return {
    KinematicBinningVariable(
      name     = binCenter["varName"],
      label    = binCenter["label"],
      unit     = binCenter["unit"],
      nmbDigits= binCenter["nmbDigits"],
    ) : binCenter["centerValue"]
    for binCenter in jsonDict["binCenters"]
  }


@registerFromJsonDictFunc("MomentValue")
def _fromJsonDict_MomentValue(jsonDict: dict[str, Any]) -> MomentValue:
  """Returns moment value constructed from a JSON-serializable dictionary"""
  return MomentValue(
    qn         = jsonDict["qn"],  #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict
    val        = complex(jsonDict["valRe"], jsonDict["valIm"]),
    uncertRe   = jsonDict["uncertRe"],
    uncertIm   = jsonDict["uncertIm"],
    binCenters = jsonDict["binCenters"],  #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict
  )


def checkNdarrayJsonDict(
  jsonDict: dict[str, Any],
  nDim:     int,
  dtype:    np.dtype[Any],
) -> bool:
  """Checks whether given JSON dictionary represents a serialized NumPy ndarray with the specified number of dimensions and data type"""
  if not isinstance(jsonDict, dict) or not all(isinstance(key, str) for key in jsonDict):
    return False
  if not all(field in jsonDict for field in ("dtype", "order", "shape", "data")):
    return False
  if jsonDict["dtype"] != np.dtype(dtype).name:
    return False
  shape = jsonDict["shape"]
  if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape) or len(shape) != nDim:
    return False
  data  = jsonDict["data"]
  if not isinstance(data, list) or len(data) != shape[0]:
    return False
  return True


@registerFromJsonDictFunc("ndarray1DComplex")
def _fromJsonDict_ndarray1DComplex(jsonDict: dict[str, Any]) -> npt.NDArray[npt.Shape["*"], npt.Complex128]:
  """Returns 1D NumPy ndarray of complex128 constructed from a JSON-serializable dictionary"""
  wellFormed = checkNdarrayJsonDict(jsonDict, nDim = 1, dtype = np.complex128)
  if not wellFormed or (
    wellFormed and not all(
      (
        isinstance(value, dict)
        and "re" in value
        and "im" in value
      ) for value in jsonDict["data"])
    ):
    raise ValueError(f"Invalid JSON dictionary for 1D complex128 ndarray:\n{jsonDict}")
  array = np.empty(jsonDict["shape"], dtype = np.complex128)
  for index, entry in enumerate(jsonDict["data"]):
    array[index] = complex(float(entry["re"]), float(entry["im"]))
  return array


@registerFromJsonDictFunc("ndarray2DFloat")
def _fromJsonDict_ndarray2DFloat(jsonDict: dict[str, Any]) -> npt.NDArray[npt.Shape["*, *"], npt.Float64]:
  """Returns 2D NumPy ndarray of float64 constructed from a JSON-serializable dictionary"""
  wellFormed = checkNdarrayJsonDict(jsonDict, nDim = 2, dtype = np.float64)
  if not wellFormed or (
    wellFormed and not all(
      (
        isinstance(row, list)
        and len(row) == jsonDict["shape"][1]
        and all(isinstance(value, float) for value in row)
      ) for row in jsonDict["data"])
    ):
    raise ValueError(f"Invalid JSON dictionary for 2D float64 ndarray:\n{jsonDict}")
  array = np.empty(jsonDict["shape"], dtype = np.float64)
  for indexRow, row in enumerate(jsonDict["data"]):
    for indexCol, value in enumerate(row):
      array[indexRow, indexCol] = float(value)
  return array


@registerFromJsonDictFunc("MomentResult")
def _fromJsonDict_MomentResult(jsonDict: dict[str, Any]) -> MomentResult:
  """Returns moment result constructed from a JSON-serializable dictionary"""
  result = MomentResult(
    indices    = jsonDict["indices"],     #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict
    binCenters = jsonDict["binCenters"],  #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict
  )
  result._valsFlatIndex   = jsonDict["_valsFlatIndex"]    #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict
  result._V_ReReFlatIndex = jsonDict["_V_ReReFlatIndex"]  #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict
  result._V_ImImFlatIndex = jsonDict["_V_ImImFlatIndex"]  #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict
  result._V_ReImFlatIndex = jsonDict["_V_ReImFlatIndex"]  #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict
  result.valid            = jsonDict["valid"]
  return result


@registerFromJsonDictFunc("MomentResultsKinematicBinning")
def _fromJsonDict_MomentResultsKinematicBinning(jsonDict: dict[str, Any]) -> MomentResultsKinematicBinning:
  """Returns MomentResultsKinematicBinning constructed from a JSON-serializable dictionary"""
  return MomentResultsKinematicBinning([result for result in jsonDict["results"]])  #!NOTE! this only works when called through MyDecoder, which will recursively call fromJsonDict() on the nested dict


# generic function called for every dict encountered during loads
def _fromJsonDict(jsonDict: dict) -> Any:
  tag = jsonDict.get("type")
  if tag:
    fromJsonDictFunc = _FROM_JSON_DICT_FUNCS.get(tag)
    if fromJsonDictFunc is not None:
      return fromJsonDictFunc(jsonDict)
  # no tag or unknown tag: leave as plain dict
  return jsonDict


# generic wrapper decoder that uses fromJsonDict() to convert JSON-serializable dictionaries to objects
class _MyDecoder(json.JSONDecoder):
  """JSONDecoder that reconstructs tagged objects via deserializer registry"""
  def __init__(
    self,
    *args,
    **kwargs,
  ) -> None:
    # object_hook runs bottom‑up: inner dicts are processed first, then outer ones—--so nested objects rebuild correctly
    super().__init__(object_hook = _fromJsonDict, *args, **kwargs)


def fromJsonStr(
  jsonStr: str | bytes,
  **kwargs,
) -> Any:
  """Convenience wrapper around json.loads() using _MyDecoder"""
  return json.loads(jsonStr, cls = _MyDecoder, **kwargs)
