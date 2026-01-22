#!/usr/bin/env python3


from __future__ import annotations

import itertools
import os

import ROOT

from makeKinematicPlots import decomposeHistEvenOdd


def copyTH2(histSrc: ROOT.TH2) -> ROOT.TH2:
  """Copies content of a TH2 into a new TH2 of the same name"""
  histName = histSrc.GetName()
  histSrc.SetName(f"{histName}_old")
  xAxis = histSrc.GetXaxis()
  yAxis = histSrc.GetYaxis()
  histCopy = ROOT.TH2D(
    histName, histSrc.GetTitle(),
    histSrc.GetNbinsX(), xAxis.GetXmin(), xAxis.GetXmax(),
    histSrc.GetNbinsY(), yAxis.GetXmin(), yAxis.GetXmax(),
  )
  histCopy.SetXTitle(xAxis.GetTitle())
  histCopy.SetYTitle(yAxis.GetTitle())
  for xBin in range(1, histSrc.GetNbinsX() + 1):
    for yBin in range(1, histSrc.GetNbinsY() + 1):
      histCopy.SetBinContent(xBin, yBin, histSrc.GetBinContent(xBin, yBin))
      histCopy.SetBinError  (xBin, yBin, histSrc.GetBinError  (xBin, yBin))
  return histCopy


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.EnableImplicitMT()
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetOptStat(False)

  if True:
  # if False:
    plotsFile = ROOT.TFile.Open("./polarized/2018_08/tbin_0.1_0.2/PiPi/plots_ACCEPTED_PHASE_SPACE/PARA_0/plots.root", "READ")
    hist = plotsFile.Get("anglesHFPiPi_0.72_0.76")
    canv = ROOT.TCanvas()
    hist.Draw("COLZ")
    # box = ROOT.TBox()
    # box.SetFillStyle(0)
    # box.SetLineWidth(2)
    # box.SetLineColor(ROOT.kRed + 1)
    # box.DrawBox(-0.9,  -30, -0.2,  +30)
    # box.DrawBox(+0.2, -180, +0.9, -150)
    # box.DrawBox(+0.2, +150, +0.9, +180)
    # box.DrawBox(+0.3, -180, +0.7, -120)
    ellipse = ROOT.TEllipse()
    ellipse.SetFillStyle(0)
    ellipse.SetLineWidth(2)
    ellipse.SetLineColor(ROOT.kRed + 1)
    ellipse.DrawEllipse(-0.55,    0, 0.35, 30,   0, 360, 0)
    ellipse.DrawEllipse( 0.55, -180, 0.35, 30,   0, 180, 0)
    ellipse.DrawEllipse( 0.55, +180, 0.35, 30, 180, 360, 0)
    canv.RedrawAxis()
    canv.SaveAs(f"./{hist.GetName()}.pdf")

    accPsData = ROOT.RDataFrame("PiPi", "./polarized/2018_08/tbin_0.1_0.2/PiPi/phaseSpace_acc_flat_PARA_0.root")
    CUT_CPP_CODE = """
      bool
      isInHoles(
        const double cosTheta,
        const double phi
      ) {
        static std::vector<TEllipse*> holes = {
          new TEllipse(-0.55,    0, 0.35, 30),
          new TEllipse( 0.55, -180, 0.35, 30),
          new TEllipse( 0.55, +180, 0.35, 30)
        };
        for (const auto& hole : holes) {
          if (hole->IsInside(cosTheta, phi)) {
            return true;
          }
        }
        return false;
      }
    """
    ROOT.gInterpreter.Declare(CUT_CPP_CODE)

    hist = (
      accPsData.Filter("(0.72 < mass and mass < 0.76)")
              .Filter("not isInHoles(cosTheta, phiDeg)")
              #  .Filter("not ((cosTheta > 0.8) and (-100 < phiDeg and phiDeg < +100))")
              #  .Filter("not ((0.3 < cosTheta and cosTheta < 0.7) and (-180 < phiDeg and phiDeg < -120))")
              #  .Filter("not (-90 < phiDeg and phiDeg < +90)")
              #  .Filter("(phiDeg > 0)")
              .Histo2D(
                ("anglesHFPiPi_0.72_0.76_cut", ";cos#theta_{HF};#phi_{HF} [deg]", 100, -1,+1, 72, -180, +180),
                "cosTheta", "phiDeg"
              ).GetValue()
    )
    canv = ROOT.TCanvas()
    hist.Draw("COLZ")
    canv.RedrawAxis()
    canv.SaveAs(f"./{hist.GetName()}.pdf")

  accPsFileName = "./polarized/2018_08/tbin_0.1_0.2/PiPi/phaseSpace_acc_flat_PARA_0.root"
  accPsTreeName = "PiPi"
  df = ROOT.RDataFrame(accPsTreeName, accPsFileName).Filter("(0.72 < mass and mass < 0.76)")
  histAccPs2DComp = None
  if True:
  # if False:
    # decompose acceptance histogram into phi-odd and phi-even parts
    histAccPs = df.Histo3D(
      ("accPs3D", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 100, -1, +1, 72, -180, +180, 72, -180, +180),
      "cosTheta", "phiDeg", "PhiDeg",
    ).GetValue()
    histAccPsOdd, histAccPsEven, histAccPsSum = decomposeHistEvenOdd(histAccPs)
    # plot slices in Phi
    PhiSliceWidth = 4  # number of Phi bins to combine
    for hist in (histAccPsEven, histAccPsOdd, histAccPsSum):
      for PhiBinIndex in itertools.chain([0], range(1, histAccPs.GetNbinsZ() + 1, PhiSliceWidth)):
        # index 0 means to project over all Phi bins
        if PhiBinIndex > 0:
          hist.GetZaxis().SetRange(PhiBinIndex,PhiBinIndex + PhiSliceWidth - 1)
        hist2D = hist.Project3D("yx")
        hist2D.SetName(f"{str(hist.GetName()).replace('3D', '2D')}")
        if PhiBinIndex > 0:
          hist2D.SetName(f"{hist2D.GetName()}_{hist.GetZaxis().GetBinLowEdge(PhiBinIndex):.0f}_{hist.GetZaxis().GetBinUpEdge(PhiBinIndex + PhiSliceWidth - 1):.0f}")
        hist2D.SetTitle("")
        canv = ROOT.TCanvas()
        hist2D.Rebin2D(4, 3)  # reduce number of bins for better visibility
        if hist is histAccPsOdd:
          # for some unknown reason this hist2D is not drawn
          # workaround is to copy content into a new histogram
          hist2D = copyTH2(hist2D)
          valRange = max(abs(hist2D.GetMaximum()), abs(hist2D.GetMinimum()))
          hist2D.SetMaximum(+valRange)
          hist2D.SetMinimum(-valRange)
          ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # use pos/neg color palette and symmetric z axis
        hist2D.Draw("COLZ")
        canv.SaveAs(f"./{hist2D.GetName()}.pdf")
        if hist is histAccPsOdd:
          ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette
        if PhiBinIndex == 0 and hist is histAccPsSum:
          histAccPs2DComp = hist2D
      hist.GetZaxis().SetRange(0, 0)  # reset z-axis range
    # plot 3D distributions
    for hist in (histAccPsEven, histAccPsOdd, histAccPsSum, histAccPs):
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      with ROOT.TFile.Open(f"./{hist.GetName()}.root", "RECREATE") as outFile:
        hist.Write()
      hist.Rebin3D(4, 3, 3)  # reduce number of bins for better visibility
      canv = ROOT.TCanvas()
      # if hist is histAccPsOdd:
      #   valRange = max(abs(hist.GetMaximum()), abs(hist.GetMinimum()))
      #   hist.SetMaximum(+valRange)
      #   hist.SetMinimum(-valRange)
      #   ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # use pos/neg color palette and symmetric z axis
      hist.Draw("COLZ")
      canv.SaveAs(f"./{hist.GetName()}.pdf")
      # if hist is histAccPsOdd:
      #   ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette

  if True:
  # if False:
    hists = [
      df.Histo2D(
          (f"accPs2D", ";cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180),
          "cosTheta", "phiDeg",
        ),
    ]
    PhiDegNmbBins = 18
    PhiDegBinWidth = 360 / PhiDegNmbBins
    for binIndex in range(0, PhiDegNmbBins):
      PhiDegBinMin    = -180 + binIndex * PhiDegBinWidth
      PhiDegBinMax    = PhiDegBinMin + PhiDegBinWidth
      PhiDegBinFilter = f"({PhiDegBinMin} < PhiDeg) and (PhiDeg < {PhiDegBinMax})"
      histNameSuffix  = f"_{PhiDegBinMin:.0f}_{PhiDegBinMax:.0f}"
      hists += [
        df.Filter(PhiDegBinFilter) \
          .Histo2D(
            (f"accPs2D{histNameSuffix}", ";cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180),
            "cosTheta", "phiDeg",
          ),
      ]
    for hist in hists:
      canv = ROOT.TCanvas()
      hist.Draw("COLZ")
      canv.SaveAs(f"./{hist.GetName()}.pdf")

  if True:
  # if False:
    # plot regenerated accepted phase space
    accPsFileName = "../plotsTestPhotoProd.momentsRd.accEven.phys.detuneAccFull/acceptedPhaseSpace.root"
    accPsTreeName = "data"
    df = ROOT.RDataFrame(accPsTreeName, accPsFileName)
    if True:
    # if False:
      hist = df.Histo3D(
        ("accPs3D", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 100, -1, +1, 72, -180, +180, 72, -180, +180),
        "cosTheta", "phiDeg", "PhiDeg",
      ).GetValue()
      hist2D = hist.Project3D("yx")
      hist2D.SetName(f"{str(hist.GetName()).replace('3D', '2D')}")
      hist2D.SetTitle("")
      canv = ROOT.TCanvas()
      hist2D.Rebin2D(4, 3)  # reduce number of bins for better visibility
      hist2D.Draw("COLZ")
      canv.SaveAs(f"./{hist2D.GetName()}_regenerated.pdf")
      if histAccPs2DComp is not None:
        canv = ROOT.TCanvas()
        histAccPs2DComp.Divide(hist2D)
        histAccPs2DComp.SetMinimum(0.2)
        histAccPs2DComp.SetMaximum(0.4)
        histAccPs2DComp.Draw("COLZ")
        canv.SaveAs(f"./{hist2D.GetName()}_regenerated_ratio.pdf")

  if True:
  # if False:
    # plot difference between Phi < 0 and Phi > 0
    inTreeName = "PiPi"
    for inFileName, label in [
      ("./polarized/2018_08/tbin_0.1_0.2/PiPi/phaseSpace_acc_flat_PARA_0.root", "accPs"),
      ("./polarized/2018_08/tbin_0.1_0.2/PiPi/data_flat_PARA_0.root",           "realData"),
    ]:
      df = ROOT.RDataFrame(inTreeName, inFileName).Filter("(0.72 < mass and mass < 0.76)")
      treeHasWeightColumn = "eventWeight" in df.GetColumnNames()
      print(f"Processing file '{inFileName}' with label '{label}' and {treeHasWeightColumn=}")
      if True:
      # if False:
        if treeHasWeightColumn:
          histPhiPos = df.Filter("(PhiDeg > 0)").Histo2D(
            (f"anglesHFPhiPos_{label}", "#Phi > 0;cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "cosTheta", "phiDeg", "eventWeight").GetValue()
          histPhiNeg = df.Filter("(PhiDeg < 0)").Histo2D(
            (f"anglesHFPhiNeg_{label}", "#Phi < 0;cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "cosTheta", "phiDeg", "eventWeight").GetValue()
        else:
          histPhiPos = df.Filter("(PhiDeg > 0)").Histo2D(
            (f"anglesHFPhiPos_{label}", "#Phi > 0;cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "cosTheta", "phiDeg").GetValue()
          histPhiNeg = df.Filter("(PhiDeg < 0)").Histo2D(
            (f"anglesHFPhiNeg_{label}", "#Phi < 0;cos#theta_{HF};#phi_{HF} [deg]", 100, -1, +1, 72, -180, +180), "cosTheta", "phiDeg").GetValue()
        histPhiDiff = histPhiPos.Clone(f"anglesHFPhiDiff_{label}")
        histPhiDiff.SetTitle("(#Phi > 0)#minus (#Phi < 0)")
        histPhiDiff.Add(histPhiNeg, -1)
        for hist in (histPhiPos, histPhiNeg, histPhiDiff):
          canv = ROOT.TCanvas()
          hist.Rebin2D(4, 3)  # reduce number of bins for better visibility
          if hist is histPhiDiff:
            ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # use pos/neg color palette and symmetric z axis
            histPhiDiffRange = max(abs(histPhiDiff.GetMaximum()), abs(histPhiDiff.GetMinimum()))
            hist.SetMaximum(+histPhiDiffRange)
            hist.SetMinimum(-histPhiDiffRange)
          hist.Draw("COLZ")
          legend = ROOT.TLegend(0.15, 0.75, 0.35, 0.85)
          legend.SetHeader("FOO FOO BAR")
          legend.AddEntry(hist, "Line 1", "f")
          legend.AddEntry(hist, "Line 2", "f")
          legend.AddEntry(hist, "Line 3", "f")
          legend.Draw()
          canv.SaveAs(f"./{hist.GetName()}.pdf")
          ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette
