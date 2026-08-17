"""
Module that provides utility functions converting `MomentCalculator`
classes to and from JSON dictionaries that can be serialized to JSON
strings using `json.dumps()`.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from moments.MomentCalculator import (
  MomentResult,
  MomentResultsKinematicBinning,
  MomentValue,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


@functools.singledispatch
def toJsonDict(obj: Any) -> Any:
  """Converts an object to something JSON-serializable. Raise TypeError if type of obj unknown."""
  raise TypeError(f"No JSON serializer registered for type {type(obj).__name__}")


@toJsonDict.register
def _(obj: MomentValue) -> dict[str, Any]:
  """Returns dictionary with moment value and uncertainty that can be serialized to JSON"""
  return {
    "momentIndex" : obj.qn.momentIndex,
    "L"           : obj.qn.L,
    "M"           : obj.qn.M,
    "valRe"       : obj.val.real,
    "uncertRe"    : obj.uncertRe,
    "valIm"       : obj.val.imag,
    "uncertIm"    : obj.uncertIm,
    "binCenters"  : {kinVar.name : binCenter for kinVar, binCenter in obj.binCenters.items()},
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
  """Convenience wrapper around json.dumps using _MyEncoder"""
  return json.dumps(obj, cls = _MyEncoder, **kwargs)
