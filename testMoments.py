#!/usr/bin/env python3


import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore


  legendrePols = []
  for order in range(5):
      func = ROOT.TF1(f"legPol_{order}", "ROOT::Math::legendre([0],x)", -1, 1)
      func.SetParameter(0, order)
      func.SetLineStyle(1)
      func.SetLineWidth(2)
      func.SetLineColor(order + 1)
      legendrePols.append(func)

  legendrePols[0].SetMaximum(1)
  legendrePols[0].SetMinimum(-1)
  legendrePols[0].SetTitle("Legendre polynomials")

  canv = ROOT.TCanvas("legendre")
  legend = ROOT.TLegend(0.4, 0.7, 0.6, 0.89)
  for order, legendrePol in enumerate(legendrePols):
      legend.AddEntry(legendrePol, f" LegendrePol(x; {order})", "l")
      if order == 0:
          legendrePol.Draw()
      else:
          legendrePol.Draw("SAME")

  legend.Draw("SAME")

  canv.SaveAs("foo.pdf")
