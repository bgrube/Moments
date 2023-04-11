#!/usr/bin/env python3


import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore

  # linear combinaton of legendre polynomials up to degree 5
  legendrePolLC = ROOT.TF1(
    f"legendrePolLC",
    "  [0] * ROOT::Math::legendre(0, x)"
    "+ [1] * ROOT::Math::legendre(1, x)"
    "+ [2] * ROOT::Math::legendre(2, x)"
    "+ [3] * ROOT::Math::legendre(3, x)"
    "+ [4] * ROOT::Math::legendre(4, x)"
    "+ [5] * ROOT::Math::legendre(5, x)",
    -1, 1
  )
  legendrePolLC.SetParameters(0.5, 0.5, 0.25, -0.25, -0.125, 0.125)
  # legendrePolLC.SetLineStyle(1)
  # legendrePolLC.SetLineWidth(2)
  # legendrePolLC.SetMaximum(1)
  # legendrePolLC.SetMinimum(-1)

  canv = ROOT.TCanvas()
  legendrePolLC.Draw()
  canv.SaveAs(f"{legendrePolLC.GetName()}.pdf")
