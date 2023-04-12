#!/usr/bin/env python3


from typing import Any

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


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore
  ROOT.EnableImplicitMT()    # type: ignore

  # linear combination of legendre polynomials up to fiven degree
  degree = 5
  terms = tuple(f"[{i}] * ROOT::Math::legendre({i}, x)" for i in range(degree + 1))
  legendrePolLC = ROOT.TF1("legendrePolLC", " + ".join(terms), -1, +1)  # type: ignore
  legendrePolLC.SetNpx(1000)  # used in numeric integration performed by GetRandom()
  legendrePolLC.SetParameters(0.5, 0.5, 0.25, -0.25, -0.125, 0.125)
  legendrePolLC.SetMinimum(0)

  canv = ROOT.TCanvas()  # type: ignore
  legendrePolLC.Draw()
  canv.SaveAs(f"{legendrePolLC.GetName()}.pdf")

  # generate data according to linear combination of legendre polynomials
  ROOT.gRandom.SetSeed(1234567890)  # type: ignore
  declareInCpp(legendrePolLC = legendrePolLC)
  df = ROOT.RDataFrame(100000)  # type: ignore
  dfData = df.Define("val", "PyVars::legendrePolLC.GetRandom()")
  # dfData.Snapshot("test", "test.root")

  # plot data
  hist = dfData.Histo1D(ROOT.RDF.TH1DModel(f"{legendrePolLC.GetName()}_hist", "", 100, -1, +1), "val")  # type: ignore
  hist.SetMinimum(0)
  hist.Draw()
  canv.SaveAs(f"{hist.GetName()}.pdf")
