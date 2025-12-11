#!/usr/bin/env python3


from __future__ import annotations

import functools

import ROOT

from PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
)


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


FCN_CPP = """
// Functor that returns the bin content of a given 3D histogram
// can be used to convert a TH3 into a TF3
class histogram3DFunctor {
public:

	histogram3DFunctor(TH3* hist)
	 : _hist(hist)
	{ }

	double
	operator () (
		double* args,
		double*
	) {
		const double x = args[0];
		const double y = args[1];
		const double z = args[2];
		if (
			   (x < _hist->GetXaxis()->GetBinLowEdge(1) or x > _hist->GetXaxis()->GetBinUpEdge(_hist->GetNbinsX()))
			or (y < _hist->GetYaxis()->GetBinLowEdge(1) or y > _hist->GetYaxis()->GetBinUpEdge(_hist->GetNbinsY()))
			or (z < _hist->GetZaxis()->GetBinLowEdge(1) or z > _hist->GetZaxis()->GetBinUpEdge(_hist->GetNbinsZ()))
		) {
			return 0;
		}
		return _hist->GetBinContent(_hist->FindBin(x, y, z));
	}

	protected:
	TH3* _hist;
};
"""


if __name__ == "__main__":
  ROOT.gInterpreter.Declare(FCN_CPP)
  fooHist = ROOT.TH3D("fooHist", "fooHist", 4, 0, 1, 4, 0, 2, 4, 0, 3)
  for i in range(1, fooHist.GetNbinsX() + 1):
    for j in range(1, fooHist.GetNbinsY() + 1):
      for k in range(1, fooHist.GetNbinsZ() + 1):
        fooHist.SetBinContent(i, j, k, float(i + j + k))
  fcnFunctor = ROOT.histogram3DFunctor(fooHist)
  fcn = ROOT.TF3("histogramFcn", fcnFunctor, 0, 1, 0, 2, 0, 3, 1)
  ROOT.gStyle.SetOptStat(False)
  # ROOT.gStyle.SetImageScaling(3)
  drawTF3(
    fcn         = fcn,
    binnings    = (
      HistAxisBinning(30, 0, 1),
      HistAxisBinning(30, 0, 2),
      HistAxisBinning(30, 0, 3),
    ),
    outFileName = f"testFunc.pdf",
    # outFileName = f"testFunc.png",
  )
