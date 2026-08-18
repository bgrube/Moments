"""
Module that provides utility functions converting `MomentCalculator`
classes to and from JSON dictionaries that can be serialized to JSON
strings using `json.dumps()`.
"""

from __future__ import annotations

import functools
import json
from typing import (
  Any,
  Callable,
)

from moments.MomentCalculator import (
  KinematicBinningVariable,
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


# single dispatch does not work for types in containers like dict, so we need a separate function for binCenters
def _toJsonDict_binCenters(binCenters: dict[KinematicBinningVariable, float]) -> dict[str, Any]:
  """Returns dictionary with kinematic bin centers that can be serialized to JSON"""
  jsonDict: dict[str, Any] = {"type" : "binCenters"}
  jsonDict |= {
    kinVar.name : {
      "binCenter" : binCenter,
      "label"     : kinVar.label,
      "unit"      : kinVar.unit,
      "nmbDigits" : kinVar.nmbDigits,
    } for kinVar, binCenter in binCenters.items()
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


@toJsonDict.register
def _(obj: MomentResult) -> list[dict[str, Any]]:
  """Returns list of dictionaries with moment values and uncertainties that can be serialized to JSON"""
  return [toJsonDict(momentValue) for momentValue in obj.values]


@toJsonDict.register
def _(obj: MomentResultsKinematicBinning) -> list[dict[str, Any]]:
  """Returns list of dictionaries with valid moment values and uncertainties in all kinematic bins that can be serialized to JSON"""
  return [
    toJsonDict(momentValue) for momentResult in obj if momentResult
                            for momentValue  in momentResult.values
  ]


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


@registerFromJsonDictFunc("binCenters")
def _fromJsonDict_binCenters(jsonDict: dict[str, Any]) -> dict[KinematicBinningVariable, float]:
  """Returns bin centers constructed from a JSON-serializable dictionary"""
  return {
    KinematicBinningVariable(
      name     = name,
      label    = value["label"],
      unit     = value["unit"],
      nmbDigits= value["nmbDigits"],
    ) : value["binCenter"]
    for name, value in jsonDict.items() if name != "type"
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
