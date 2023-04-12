#!/usr/bin/env python3


import math
from typing import Any, Collection, List, Tuple

from uncertainties import UFloat, ufloat

import ROOT


# see https://root-forum.cern.ch/t/tf1-eval-as-a-function-in-rdataframe/50699/3
def declareInCpp(**kwargs: Any) -> None:
  '''Creates C++ variables (names given by keys) for PyROOT objects (given values) in PyVars:: namespace'''
  for key, value in kwargs.items():
    ROOT.gInterpreter.Declare(  # type: ignore
f'''
namespace PyVars
{{
  auto& {key} = *reinterpret_cast<{type(value).__cpp_name__}*>({ROOT.addressof(value)});
}}
''')


def generateDataLegPolLC(
  nmbEvents:  int,
  maxDegree:  int,
  parameters: Collection[float] = (0.5, 0.5, 0.25, -0.25, -0.125, 0.125),  # make sure that resulting linear combination is positive definite
) -> Tuple[str, str]:
  '''Generates data according to linear combination of Legendre polynomials'''
  assert len(parameters) >= maxDegree + 1, f"Need at least {maxDegree + 1} parameters; only {len(parameters)} given: {parameters}"
  # linear combination of legendre polynomials up to given degree
  terms = tuple(f"[{degree}] * ROOT::Math::legendre({degree}, x)" for degree in range(maxDegree + 1))
  legendrePolLC = ROOT.TF1("legendrePolLC", " + ".join(terms), -1, +1)  # type: ignore
  legendrePolLC.SetNpx(1000)  # used in numeric integration performed by GetRandom()
  for index, parameter in enumerate(parameters):
    legendrePolLC.SetParameter(index, parameter)
  legendrePolLC.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()  # type: ignore
  legendrePolLC.Draw()
  canv.SaveAs(f"{legendrePolLC.GetName()}.pdf")

  # generate random data that follow linear combination of legendre polynomials
  ROOT.gRandom.SetSeed(1234567890)  # type: ignore
  declareInCpp(legendrePolLC = legendrePolLC)
  treeName = "data"
  fileName = f"{legendrePolLC.GetName()}.root"
  df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  df.Define("val", "PyVars::legendrePolLC.GetRandom()") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `val` column would be regenerated for every triggered loop
                                   # use noop filter to log when event loop is running
  return treeName, fileName


def calculateLegMoments(
  dataFrame: Any,
  maxDegree: int,
) -> List[UFloat]:
  nmbEvents = dataFrame.Count().GetValue()
  moments: List[UFloat] = []
  for degree in range(maxDegree + 5):
    # unnormalized moments
    dfMoment = dataFrame.Define("legendrePol", f"ROOT::Math::legendre({degree}, val)")
    momentVal = dfMoment.Sum("legendrePol").GetValue()
    momentErr = math.sqrt(nmbEvents) * dfMoment.StdDev("legendrePol").GetValue()  # iid events: Var[sum_i^N f(x_i)] = sum_i^N Var[f] = N * Var[f]
    # normalize moments
    legendrePolIntegral = 2 / (2 * degree + 1)  # = int_-1^+1
    norm = 1 / (nmbEvents * legendrePolIntegral)
    moments.append(norm * ufloat(momentVal, momentErr))  # type: ignore
  print(moments)
  for degree, moment in enumerate(moments):
    print(f"Moment degree {degree} = {moment}")
  return moments


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore

  # get data
  nmbEvents = 100000
  maxDegree = 5
  treeName, fileName = generateDataLegPolLC(nmbEvents, maxDegree)
  ROOT.EnableImplicitMT(10)  # type: ignore
  dataFrame = ROOT.RDataFrame(treeName, fileName)  # type: ignore

  # plot data
  canv = ROOT.TCanvas()  # type: ignore
  hist = dataFrame.Histo1D(ROOT.RDF.TH1DModel("data", "", 100, -1, +1), "val")  # type: ignore
  hist.SetMinimum(0)
  hist.Draw()
  canv.SaveAs(f"{hist.GetName()}.pdf")
  # print("!!!", dfData.AsNumpy())

  # calculate moments
  calculateLegMoments(dataFrame, maxDegree)
