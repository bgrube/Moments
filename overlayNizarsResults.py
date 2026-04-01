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

  momentPlotFileNizar = "/work/halld2/home/nseptian/EtaPi0_4Gamma_data/moment_gr/UnnormalizedMoments_vs_Mass_t010020.root"
  with ROOT.TFile.Open(momentPlotFileNizar) as fileNizar:
    for momentIndex in MomentIndices(maxL = 4, polarized = True).qnIndices:
      # get objects to be overlaid
      i = momentIndex.momentIndex
      L = momentIndex.L
      M = momentIndex.M
      momentPlotFileLinAlg = f"./plotsPhotoProdEtaPi0/merged/t010020/All.maxL_4/unnorm_phys_mass_compare_H{i}_{L}_{M}_Re.root"
      canvLinAlg = None
      with ROOT.TFile.Open(momentPlotFileLinAlg) as file:
        canvLinAlg = file.Get("c1")
      momentHistStack = canvLinAlg.GetPrimitive(f"./plotsPhotoProdEtaPi0/merged/t010020/All.maxL_4/unnorm_phys_mass_compare_H{i}_{L}_{M}_Re")
      graphNizar = fileNizar.Get(f"gr_unnormalized_moments_{i}_{L}_{M}_vs_mass")
      # overlay the two plots
      momentHist = canvLinAlg.GetPrimitive(f"Moment Real Part")
      momentHist.SetLineColor(ROOT.kBlue + 1)
      momentHist.SetMarkerColor(ROOT.kBlue + 1)
      momentHist.SetMarkerStyle(ROOT.kOpenCircle)
      momentHist.SetMarkerSize(1.0)
      legend = canvLinAlg.GetPrimitive("TPave")
      firstEntry = legend.GetListOfPrimitives().First()
      firstEntry.SetLabel("Linear Algebra Method")
      canvLinAlg.Draw()
      graphNizar.SetLineColor(ROOT.kRed + 1)
      graphNizar.SetMarkerColor(ROOT.kRed + 1)
      graphNizar.Draw("P SAME")
      graphNizar.SetMarkerSize(0.8)
      legend.AddEntry(graphNizar, "MCMC", "LP")
      # extend y axis range if necessary to accommodate Nizar's results
      xMin = ctypes.c_double(0.0)
      xMax = ctypes.c_double(0.0)
      yMin = ctypes.c_double(0.0)
      yMax = ctypes.c_double(0.0)
      graphNizar.ComputeRange(xMin, yMin, xMax, yMax)
      print(f"graphNizar: x range = [{xMin.value}, {xMax.value}], y range = [{yMin.value}, {yMax.value}]")
      print(f"{momentHistStack.GetMinimum()} <= y <= {momentHistStack.GetMaximum()} before overlaying Nizar's results")
      momentHistStack.SetMinimum(min(momentHistStack.GetMinimum(), yMin.value))
      momentHistStack.SetMaximum(max(momentHistStack.GetMaximum(), yMax.value))
      print(f"{momentHistStack.GetMinimum()} <= y <= {momentHistStack.GetMaximum()} after overlaying Nizar's results")
      canvLinAlg.SaveAs(f"H{i}_{L}_{M}_Re_Nizar.pdf")
