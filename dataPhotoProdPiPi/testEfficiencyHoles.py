#!/usr/bin/env python3


from __future__ import annotations

import os

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  assert ROOT.gROOT.LoadMacro("../rootlogon.C") == 0, "Error loading '../rootlogon.C'"
  ROOT.gStyle.SetOptStat(False)

  if True:
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

  df = ROOT.RDataFrame("PiPi", "./polarized/2018_08/tbin_0.1_0.2/PiPi/phaseSpace_acc_flat_PARA_0.root")
  if True:
    # decompose acceptance histogram into phi-odd and phi-even parts
    histAccPs = df.Histo3D(
      ("accPs3D", ";cos#theta_{HF};#phi_{HF} [deg];#Phi [deg]", 100, -1, +1, 72, -180, +180, 72, -180, +180),
      "cosTheta", "phiDeg", "PhiDeg",
    ).GetValue()
    histAccPsOdd  = histAccPs.Clone(f"{histAccPs.GetName()}_odd")
    histAccPsEven = histAccPs.Clone(f"{histAccPs.GetName()}_even")
    assert histAccPs.GetNbinsY() % 2 == 0, "Number of phi bins must be even!"
    for cosThetaBin in range(1, histAccPs.GetNbinsX() + 1):
      for PhiBin in range(1, histAccPs.GetNbinsZ() + 1):
        for phiBinNeg in range(1, histAccPs.GetNbinsY() // 2 + 1):
          phiBinPos = histAccPs.GetYaxis().FindBin(-histAccPs.GetYaxis().GetBinCenter(phiBinNeg))
          valPhiPos = histAccPs.GetBinContent(cosThetaBin, phiBinPos, PhiBin)
          valPhiNeg = histAccPs.GetBinContent(cosThetaBin, phiBinNeg, PhiBin)
          valPhiOdd  = (valPhiPos - valPhiNeg) / 2
          valPhiEven = (valPhiPos + valPhiNeg) / 2
          histAccPsOdd.SetBinContent (cosThetaBin, phiBinPos, PhiBin, +valPhiOdd)
          histAccPsOdd.SetBinContent (cosThetaBin, phiBinNeg, PhiBin, -valPhiOdd)
          histAccPsEven.SetBinContent(cosThetaBin, phiBinPos, PhiBin, valPhiEven)
          histAccPsEven.SetBinContent(cosThetaBin, phiBinNeg, PhiBin, valPhiEven)
    histAccPsSum = histAccPs.Clone(f"{histAccPs.GetName()}_sum")
    histAccPsSum.Add(histAccPsOdd, histAccPsEven)
    for hist in (histAccPsEven, histAccPsOdd, histAccPsSum):
      hist.GetXaxis().SetTitleOffset(1.5)
      hist.GetYaxis().SetTitleOffset(2)
      hist.GetZaxis().SetTitleOffset(1.5)
      with ROOT.TFile.Open(f"./{hist.GetName()}.root", "RECREATE") as outFile:
        hist.Write()
      hist.Rebin3D(2, 2, 2)
    for hist in (histAccPsEven, histAccPsSum):
      canv = ROOT.TCanvas()
      hist.Draw("COLZ")
      canv.SaveAs(f"./{hist.GetName()}.pdf")
    print(f"{histAccPsOdd.GetName()}: {histAccPsOdd.GetMinimum()=}, {histAccPsOdd.GetMaximum()=}")
    valRange = max(abs(histAccPsOdd.GetMaximum()), abs(histAccPsOdd.GetMinimum()))
    histAccPsOdd.SetMaximum(+valRange)
    histAccPsOdd.SetMinimum(-valRange)
    canv = ROOT.TCanvas()
    ROOT.gStyle.SetPalette(ROOT.kLightTemperature)  # use pos/neg color palette and symmetric z axis
    histAccPsOdd.Draw("COLZ")
    canv.SaveAs(f"./{histAccPsOdd.GetName()}.pdf")
    ROOT.gStyle.SetPalette(ROOT.kBird)  # restore default color palette

  if True:
    PhiDegNmbBins = 24
    PhiDegBinWidth = 360 / PhiDegNmbBins
    hists = []
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
