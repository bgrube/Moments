#!/usr/bin/env python3


import numpy as np

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore

  # linear combination of legendre polynomials up to degree 5
  legendrePolLC = ROOT.TF1(
    f"legendrePolLC",
    "  [0] * ROOT::Math::legendre(0, x)"
    "+ [1] * ROOT::Math::legendre(1, x)"
    "+ [2] * ROOT::Math::legendre(2, x)"
    "+ [3] * ROOT::Math::legendre(3, x)"
    "+ [4] * ROOT::Math::legendre(4, x)"
    "+ [5] * ROOT::Math::legendre(5, x)",
    -1, +1
  )
  legendrePolLC.SetNpx(1000)  # used in numeric integration performed by GetRandom()
  legendrePolLC.SetParameters(0.5, 0.5, 0.25, -0.25, -0.125, 0.125)
  legendrePolLC.SetMinimum(0)

  canv = ROOT.TCanvas()
  legendrePolLC.Draw()
  canv.SaveAs(f"{legendrePolLC.GetName()}.pdf")

  # generate data according to linear combination of legendre polynomials
  data = np.array([legendrePolLC.GetRandom() for index in range(100000)])

  # plot data
  hist = ROOT.TH1F(f"{legendrePolLC.GetName()}_hist", "", 100, -1, +1)
  for val in data:
    hist.Fill(val)
  hist.SetMinimum(0)
  hist.Draw()
  canv.SaveAs(f"{hist.GetName()}.pdf")
