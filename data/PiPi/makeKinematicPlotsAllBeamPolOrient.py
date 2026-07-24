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
  if hist.GetDimension() == 3:
    hist.GetXaxis().SetTitleOffset(1.5)
    hist.GetYaxis().SetTitleOffset(2)
    hist.GetZaxis().SetTitleOffset(1.5)
    hist.Draw("BOX2Z")
  else:
    hist.Draw("COLZ")
  if logZ and hist.GetDimension() == 2:
    canv.SetLogz()
  canv.SaveAs(f"{plotDir}/{histName}.pdf")


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"

  dataDirPath = "./polarized"
  # data sets to process
  dataPeriods = (
    # "2017_01",
    "2017_01_ver05",
    # "2018_08",
  )
  tBinLabels  = (
    "tbin_0.100_0.114",  # lowest |t| bin of SDME analysis
    # "tbin_0.1_0.2",
    # "tbin_0.2_0.3",
    # "tbin_0.3_0.4",
    # "tbin_0.4_0.5",
  )
  subsystems  = (
    "PiPi",
    # "PipP",
    # "PimP",
  )
  # beam-polarization orientations to combine
  beamPolLabels = (
    "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
  )

  for dataPeriod in dataPeriods:
    for tBinLabel in tBinLabels:
      for subsystem in subsystems:
        # generate real-data plots for all beam-polarization orientations combined
        plotBaseDir = f"{dataDirPath}/{dataPeriod}/{tBinLabel}/{subsystem}"
        # # `hadd` ROOT files with plots for real data for all beam-polarization orientations
        # sourcePlotFileNames = tuple(f"{plotBaseDir}/plots_REAL_DATA/{beamPolLabel}/plots.root" for beamPolLabel in beamPolLabels)
        # mergedPlotFileName  = f"{plotBaseDir}/plots_REAL_DATA/plots.root"
        # mergedPlotFileName = f"{plotBaseDir}/plots_REAL_DATA/{beamPolLabels[0]}/plots.root"
        # print(f"`hadd`ing files for data period '{dataPeriod}', t bin '{tBinLabel}', subsystem '{subsystem}': {sourcePlotFileNames}")
        # subprocess.run(f"hadd -f {mergedPlotFileName} {' '.join(sourcePlotFileNames)}", shell = True, check = True)
        # # plot all histograms in `hadd`ed ROOT file
        # print(f"Reading histograms from '{mergedPlotFileName}'")
        # mergedPlotFile = ROOT.TFile.Open(mergedPlotFileName, "READ")
        # for key in mergedPlotFile.GetListOfKeys():
        #   obj = key.ReadObj()
        #   if obj.InheritsFrom("TH1"):
        #     plotHistogram(obj, plotBaseDir)
        #     # plot log version of selected 2D histograms
        #     histName = obj.GetName()
        #     if histName.startswith("massPiPiVs"):
        #       obj.SetName(f"{histName}_log")
        #       plotHistogram(obj, plotBaseDir, logZ = True)

        # overlay real-data mass distribution for all beam-polarization orientations combined
        # with mass distribution from accepted phase-space MC
        histMassName = f"mass{subsystem}"
        # histMassRd = mergedPlotFile.Get(histMassName)
        plotFileRd = ROOT.TFile.Open(f"{plotBaseDir}/plots_REAL_DATA/PARA_0/plots.root", "READ")
        histMassRd = plotFileRd.Get(histMassName)
        #!NOTE! we use the same phase-space MC for all beam-polarization orientations, hence we
        #       overlay the MC distribution for only one orientation
        plotFileMc = ROOT.TFile.Open(f"{plotBaseDir}/plots_ACCEPTED_PHASE_SPACE/PARA_0/plots.root", "READ")
        histMassMc = plotFileMc.Get(histMassName)
        # dfMc = ROOT.RDataFrame("PiPi", f"{plotBaseDir}/weightedMc.maxL_4/PARA_0.fineBins/phaseSpace_acc_weighted_flat_allTerms_reweighted.root")
        # histMassMc = dfMc.Histo1D((histMassName, ";m_{#pi#pi} [GeV];Counts", 400, 0.28, 2.28), "mass").GetValue()
        histMassRd.SetTitle("RF-subtracted Data")
        histMassMc.SetTitle("Accepted Phase-Space MC")
        # histMassMc.SetTitle("Reweighted MC")
        print(f"!!! {histMassRd.GetEntries()=} vs. {histMassMc.GetEntries()=}")
        print(f"!!! {histMassRd.Integral()=} vs. {histMassMc.Integral()=}")
        canv = ROOT.TCanvas()
        histStack = ROOT.THStack(f"mass{subsystem}DataAndMc", f";{histMassMc.GetXaxis().GetTitle()};{histMassMc.GetYaxis().GetTitle()}")
        histStack.Add(histMassMc)
        histStack.Add(histMassRd)
        histMassMc.SetLineColor  (ROOT.kBlue + 1)
        histMassMc.SetMarkerColor(ROOT.kBlue + 1)
        histMassRd.SetLineColor  (ROOT.kRed  + 1)
        histMassRd.SetMarkerColor(ROOT.kRed  + 1)
        # histStack.SetMaximum(3000)
        histStack.Draw("NOSTACK")
        canv.BuildLegend(0.7, 0.8, 0.99, 0.99)
        canv.SaveAs(f"./{histStack.GetName()}.pdf")
        # canv.SaveAs(f"{plotBaseDir}/{histStack.GetName()}.pdf")
        # mergedPlotFile.Close()
        # plotFileMc.Close()
