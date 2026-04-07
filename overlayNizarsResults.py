#!/usr/bin/env python3


from __future__ import annotations

import ctypes
import functools
import os

import ROOT

from MomentCalculator import MomentIndices
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  timer.start("Total execution time")
  ROOT.gROOT.SetBatch(True)
  ROOT.EnableImplicitMT()
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("./rootlogon.C") == 0, "Error loading './rootlogon.C'"

  tBinLabels = (
    "t010020",
    "t020032",
    "t032050",
    "t050075",
    # "t075100",  #TODO missing in Nizar's directory
  )

  for tBinLabel in tBinLabels:
    momentPlotFileNizar = f"dataPhotoProdEtaPi0/polarized/merged/{tBinLabel}/Nizar/UnnormalizedMoments_vs_Mass_{tBinLabel}.root"
    plotDirLinAlg = f"./plotsPhotoProdEtaPi0/merged/{tBinLabel}/All.maxL_4"
    with ROOT.TFile.Open(momentPlotFileNizar) as fileNizar:
      chi2Total = 0.0
      nmbPointsTotal = 0
      for momentIndex in MomentIndices(maxL = 4, polarized = True).qnIndices:
        # get objects to be overlaid
        pointIndex = momentIndex.momentIndex
        L = momentIndex.L
        M = momentIndex.M
        momentPlotNameLinAlg = f"unnorm_phys_mass_compare_H{pointIndex}_{L}_{M}_{'Im' if pointIndex == 2 else 'Re'}"
        momentPlotFileLinAlg = f"{plotDirLinAlg}/{momentPlotNameLinAlg}.root"
        with ROOT.TFile.Open(momentPlotFileLinAlg) as file:
          canvLinAlg = file.Get("c1_n2" if pointIndex == 2 else "c1")
          momentHistStack = canvLinAlg.GetPrimitive(f"{plotDirLinAlg}/{momentPlotNameLinAlg}")
          graphNizar = fileNizar.Get(f"gr_unnormalized_moments_{pointIndex}_{L}_{M}_vs_mass")
          # overlay the two plots
          momentHist = canvLinAlg.GetPrimitive(f"Moment {'Imag Part' if pointIndex == 2 else 'Real Part'}")
          momentHist.SetLineColor(ROOT.kBlue + 1)
          momentHist.SetMarkerColor(ROOT.kBlue + 1)
          momentHist.SetMarkerStyle(ROOT.kOpenCircle)
          momentHist.SetMarkerSize(1.0)
          canvLinAlg.Draw()
          graphNizar.SetLineColor(ROOT.kRed + 1)
          graphNizar.SetMarkerColor(ROOT.kRed + 1)
          graphNizar.Draw("P SAME")
          graphNizar.SetMarkerSize(0.8)
          # extend y axis range if necessary to accommodate Nizar's results
          xMin = ctypes.c_double(0.0)
          xMax = ctypes.c_double(0.0)
          yMin = ctypes.c_double(0.0)
          yMax = ctypes.c_double(0.0)
          graphNizar.ComputeRange(xMin, yMin, xMax, yMax)
          yLimits = (min(momentHistStack.GetMinimum(), yMin.value, -0.1 * momentHistStack.GetMaximum()), max(momentHistStack.GetMaximum(), yMax.value))
          yRange = yLimits[1] - yLimits[0]
          momentHistStack.SetMinimum(yLimits[0] - 0.05 * yRange)
          momentHistStack.SetMaximum(yLimits[1] + 0.05 * yRange)
          # compute chi^2 between the two results
          chi2ForMoment = 0.0
          # loop over points in Nizar's graph and find corresponding points in LinAlg result to compute chi^2
          for pointIndex in range(graphNizar.GetN()):
            mass               = graphNizar.GetPointX(pointIndex)
            momentValNizar     = graphNizar.GetPointY(pointIndex)
            # momentValErrNizar  = graphNizar.GetErrorY(pointIndex)
            massBinIndex       = momentHist.GetXaxis().FindBin(mass)
            momentValLinAlg    = momentHist.GetBinContent(massBinIndex)
            momentValErrLinAlg = momentHist.GetBinError  (massBinIndex)
            chi2ForMoment += (momentValNizar - momentValLinAlg) ** 2 / momentValErrLinAlg ** 2 if momentValErrLinAlg != 0 else 0.0
          chi2PerPoint = chi2ForMoment / graphNizar.GetN() if graphNizar.GetN() > 0 else float("inf")
          chi2Total += chi2ForMoment
          nmbPointsTotal += graphNizar.GetN()
          # adjust legend
          legend = canvLinAlg.GetPrimitive("TPave")
          firstEntry = legend.GetListOfPrimitives().First()
          firstEntry.SetLabel("Lin. Alg. Method, GJ")
          legend.AddEntry(graphNizar, "MCMC Method, GJ", "LP")
          legend.AddEntry(ROOT.nullptr, f"#chi^{{2}}/point = {chi2PerPoint:.2f}", "")
          canvLinAlg.SaveAs(f"{plotDirLinAlg}/Nizar_{momentPlotNameLinAlg}.pdf")
      chi2TotalPerPoint = chi2Total / nmbPointsTotal
      print(f"Total chi^2 / point for all moments in {tBinLabel}: {chi2TotalPerPoint:.2f}")
