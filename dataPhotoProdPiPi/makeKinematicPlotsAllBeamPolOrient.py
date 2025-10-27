#!/usr/bin/env python3


from __future__ import annotations

import os
import subprocess

import ROOT


def plotHistogram(
  hist:    ROOT.TH1,
  plotDir: str,
  logZ:    bool = False,
) -> None:
  """Plots a histogram to a PDF file in the specified directory"""
  histName = hist.GetName()
  print(f"Plotting histogram '{histName}'")
  ROOT.gStyle.SetOptStat("i")
  # ROOT.gStyle.SetOptStat(1111111)
  ROOT.TH1.SetDefaultSumw2(True)  # use sqrt(sum of squares of weights) as uncertainty
  canv = ROOT.TCanvas()
  hist.SetMinimum(0)
  if "TH3" in hist.ClassName():
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
  else:
    hist.Draw("COLZ")
  if logZ and "TH2" in hist.ClassName():
    canv.SetLogz()
  canv.SaveAs(f"{plotDir}/{histName}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  dataDirName = "./polarized"
  # data sets to process
  dataPeriods = (
    "2017_01",
    # "2018_08",
  )
  tBinLabels  = (
    "tbin_0.1_0.2",
    "tbin_0.2_0.3",
    # "tbin_0.3_0.4",
    # "tbin_0.4_0.5",
  )
  subsystems  = (
    "PiPi",
    "PipP",
    "PimP",
  )
  # beam-polarization orientations to combine
  beamPolLabels = (
    "PARA_0",
    "PARA_135",
    "PERP_45",
    "PERP_90",
  )

  for dataPeriod in dataPeriods:
    for tBinLabel in tBinLabels:
      for subsystem in subsystems:
        # generate real-data plots for all beam-polarization orientations combined
        plotDir = f"{dataDirName}/{dataPeriod}/{tBinLabel}/{subsystem}/plots_realData"
        # `hadd` ROOT files with plots for real data for all beam-polarization orientations
        sourcePlotFileNames = tuple(f"{plotDir}/{beamPolLabel}/plots.root" for beamPolLabel in beamPolLabels)
        mergedPlotFileName  = f"{plotDir}/plots.root"
        print(f"`hadd`ing files for data period '{dataPeriod}', t bin '{tBinLabel}', subsystem '{subsystem}': {sourcePlotFileNames}")
        subprocess.run(f"hadd -f {mergedPlotFileName} {' '.join(sourcePlotFileNames)}", shell = True, check = True)
        # plot all histograms in `hadd`ed ROOT file
        print(f"Reading histograms from '{mergedPlotFileName}'")
        mergedPlotFile = ROOT.TFile.Open(mergedPlotFileName, "READ")
        for key in mergedPlotFile.GetListOfKeys():
          obj = key.ReadObj()
          if obj.InheritsFrom("TH1"):
            plotHistogram(obj, plotDir)
            # plot log version of selected 2D histograms
            histName = obj.GetName()
            if histName.startswith("massPiPiVs"):
              obj.SetName(f"{histName}_log")
              plotHistogram(obj, plotDir, logZ = True)

        # overlay real-data mass distribution for all beam-polarization orientations combined
        # with mass distribution from accepted phase-space MC
        histMassName = f"mass{subsystem}"
        histMassData = mergedPlotFile.Get(histMassName)
        #!NOTE! we use the same phase-space MC for all beam-polarization orientations, hence we
        #       overlay the MC distribution for only one orientation
        mcPlotFile = ROOT.TFile.Open(f"{dataDirName}/{dataPeriod}/{tBinLabel}/{subsystem}/plots_mcReco/PARA_0/plots.root", "READ")
        histMassMc = mcPlotFile.Get(histMassName)
        canv = ROOT.TCanvas()
        histStack = ROOT.THStack(f"mass{subsystem}DataAndMc", f";{histMassMc.GetXaxis().GetTitle()};{histMassMc.GetYaxis().GetTitle()}")
        histStack.Add(histMassMc)
        histStack.Add(histMassData)
        histMassMc.SetLineColor    (ROOT.kBlue + 1)
        histMassMc.SetMarkerColor  (ROOT.kBlue + 1)
        histMassData.SetLineColor  (ROOT.kRed  + 1)
        histMassData.SetMarkerColor(ROOT.kRed  + 1)
        histStack.Draw("NOSTACK")
        histMassData.SetTitle("RF-subtracted Data")
        histMassMc.SetTitle("Accepted Phase-Space MC")
        canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
        canv.SaveAs(f"{plotDir}/{histStack.GetName()}.pdf")
        mergedPlotFile.Close()
        mcPlotFile.Close()
