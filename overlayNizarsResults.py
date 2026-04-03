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
      for momentIndex in MomentIndices(maxL = 4, polarized = True).qnIndices:
        # get objects to be overlaid
        i = momentIndex.momentIndex
        L = momentIndex.L
        M = momentIndex.M
        momentPlotNameLinAlg = f"unnorm_phys_mass_compare_H{i}_{L}_{M}_Re"
        momentPlotFileLinAlg = f"{plotDirLinAlg}/{momentPlotNameLinAlg}.root"
        with ROOT.TFile.Open(momentPlotFileLinAlg) as file:
          canvLinAlg = file.Get("c1")
          momentHistStack = canvLinAlg.GetPrimitive(f"{plotDirLinAlg}/{momentPlotNameLinAlg}")
          graphNizar = fileNizar.Get(f"gr_unnormalized_moments_{i}_{L}_{M}_vs_mass")
          # overlay the two plots
          momentHist = canvLinAlg.GetPrimitive(f"Moment Real Part")
          momentHist.SetLineColor(ROOT.kBlue + 1)
          momentHist.SetMarkerColor(ROOT.kBlue + 1)
        momentHist.SetMarkerStyle(ROOT.kOpenCircle)
        momentHist.SetMarkerSize(1.0)
        legend = canvLinAlg.GetPrimitive("TPave")
        firstEntry = legend.GetListOfPrimitives().First()
        firstEntry.SetLabel("Lin. Alg. Method, HF")
        canvLinAlg.Draw()
        graphNizar.SetLineColor(ROOT.kRed + 1)
        graphNizar.SetMarkerColor(ROOT.kRed + 1)
        graphNizar.Draw("P SAME")
        graphNizar.SetMarkerSize(0.8)
        legend.AddEntry(graphNizar, "MCMC Method, GJ", "LP")
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
        canvLinAlg.SaveAs(f"{plotDirLinAlg}/Nizar_{momentPlotNameLinAlg}.pdf")
