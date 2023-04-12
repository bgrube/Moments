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


# see e.g. LHCb, PRD 92 (2015) 112009
def generateDataLegPolLC(
  nmbEvents:  int,
  maxDegree:  int,
  parameters: Collection[float],
) -> Tuple[str, str]:
  '''Generates data according to linear combination of Legendre polynomials'''
  assert len(parameters) >= maxDegree + 1, f"Need {maxDegree + 1} parameters; only {len(parameters)} were given: {parameters}"
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


# see e.g. Chung, PRD56 (1997) 7299; Suh-Urk's note Techniques of Amplitude Analysis for Two-pseudoscalar Systems; or E852, PRD 60 (1999) 092001
# see also https://en.wikipedia.org/wiki/Spherical_harmonics#Spherical_harmonics_expansion
def generateDataSphHarmLC(
  nmbEvents:  int,
  maxL:       int,
  parameters: Collection[float],  # make sure that resulting linear combination is positive definite
) -> Tuple[str, str]:
  '''Generates data according to linear combination of spherical harmonics'''
  nmbTerms = (maxL + 1) * (2 * maxL + 1) - ((maxL - 1) * (2 * maxL - 1))  # see Eqs. (15) to (17)
  assert len(parameters) >= nmbTerms, f"Need {nmbTerms} parameters; only {len(parameters)} were given: {parameters}"
  # linear combination of spherical harmonics up to given maximum orbital angular momentum
  # using Eq. (12) in Eq. (6): I = sum_L (2 L + 1 ) / (4pi)  H(L 0) (D_00^L)^* + sum_{M = 1}^L H(L M) 2 Re[D_M0^L]
  # and (D_00^L)^* = D_00^L = sqrt(4 pi / (2 L + 1)) (Y_L^0)^* = Y_L^0
  # and Re[D_M0^L] = d_M0^L cos(M phi) = Re[sqrt(4 pi / (2 L + 1)) (Y_L^M)^*] = Re[sqrt(4 pi / (2 L + 1)) Y_L^M]
  # i.e. Eq. (13) becomes: I = sum_L sqrt((2 L + 1 ) / (4pi)) sum_{M = 0}^L tau(M) H(L M) Re[Y_L^M]
  terms = []
  termIndex = 0
  for L in range(2 * maxL + 1):
    for M in range(min(L, 2) + 1):
      terms.append(f"[{termIndex}] * ROOT::Math::sph_legendre({L}, {M}, std::acos(x)) * std::cos({M} * TMath::DegToRad() * y)")  # ROOT defines this as function of theta (not cos(theta)!); sigh
      termIndex += 1
  # print("!!! LC =", " + ".join(terms))
  sphericalHarmLC = ROOT.TF2("sphericalHarmlLC", " + ".join(terms), -1, +1, -180, +180)  # type: ignore
  sphericalHarmLC.SetNpx(100)  # used in numeric integration performed by GetRandom()
  sphericalHarmLC.SetNpy(100)
  for index, parameter in enumerate(parameters):
    sphericalHarmLC.SetParameter(index, parameter)
  sphericalHarmLC.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()  # type: ignore
  sphericalHarmLC.Draw("COLZ")
  canv.SaveAs(f"{sphericalHarmLC.GetName()}.pdf")

  # generate random data that follow linear combination of legendre polynomials
  # ROOT.gRandom.SetSeed(1234567890)  # type: ignore
  # declareInCpp(legendrePolLC = legendrePolLC)
  treeName = "data"
  fileName = f"{sphericalHarmLC.GetName()}.root"
  # df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  # df.Define("val", "PyVars::legendrePolLC.GetRandom()") \
  #   .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
  #   .Snapshot(treeName, fileName)  # snapshot is needed or else the `val` column would be regenerated for every triggered loop
  #                                  # use noop filter to log when event loop is running
  return treeName, fileName


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore

  # get data
  nmbEvents = 100000
  maxOrder = 5
  treeName, fileName = generateDataLegPolLC(nmbEvents,  maxDegree = maxOrder, parameters = (0.5, 0.5, 0.25, -0.25, -0.125, 0.125))  # make sure that resulting linear combination is positive definite
  # maxOrder = 1
  # treeName, fileName = generateDataSphHarmLC(nmbEvents, maxL = maxOrder, parameters = (1, 0.3, 0.25, -0.15, 0.125, -0.1))  # make sure that resulting linear combination is positive definite
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
  calculateLegMoments(dataFrame, maxDegree = maxOrder)
