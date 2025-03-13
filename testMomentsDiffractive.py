#!/usr/bin/env python3


from __future__ import annotations

import functools
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import threadpoolctl
from typing import (
  Any,
  Collection,
)

import py3nj
from uncertainties import UFloat, ufloat

import ROOT

from PlottingUtilities import (
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def plotComparison(
  measVals:           tuple[tuple[float, float, tuple[int, int, int]], ...],  # tuple(tuple(value, uncertainty, indices), ...)
  trueVals:           tuple[float, ...],
  realPart:           bool,
  useMomentSubscript: bool,
  dataLabel:          str = "",
  pdfFileNamePrefix:  str = "",
) -> None:
  momentIndex = measVals[0][2][0]
  if realPart:
    fileNameSuffix    = "Re"
    legendEntrySuffix = "Real Part"
  else:
    fileNameSuffix    = "Im"
    legendEntrySuffix = "Imag Part"

  # overlay measured and input values
  hStack = ROOT.THStack(f"h{dataLabel}_Compare_H{momentIndex if useMomentSubscript else ''}_{fileNameSuffix}", "")
  nmbBins = len(measVals)
  # create histogram with measured values
  labelSubscript = f"_{{{momentIndex}}}" if useMomentSubscript else ""
  hMeas = ROOT.TH1D(f"Measured #it{{H}}{labelSubscript} {legendEntrySuffix}", ";;Value", nmbBins, 0, nmbBins)
  for index, measVal in enumerate(measVals):
    hMeas.SetBinContent(index + 1, measVal[0])
    hMeas.SetBinError  (index + 1, 1e-100 if measVal[1] < 1e-100 else measVal[1])  # ensure that also points with zero uncertainty are drawn
    hMeas.GetXaxis().SetBinLabel(index + 1, f"#it{{H}}{labelSubscript}({measVal[2][1]}, {measVal[2][2]})")
  hMeas.SetLineColor(ROOT.kRed)
  hMeas.SetMarkerColor(ROOT.kRed)
  hMeas.SetMarkerStyle(ROOT.kFullCircle)
  hMeas.SetMarkerSize(0.75)
  hStack.Add(hMeas, "PEX0")
  # create histogram with input values
  hInput = ROOT.TH1D("Input Values", ";;Value", nmbBins, 0, nmbBins)
  for index, trueVal in enumerate(trueVals):
    hInput.SetBinContent(index + 1, trueVal)
    hInput.SetBinError  (index + 1, 1e-100)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
  hInput.SetMarkerColor(ROOT.kBlue)
  hInput.SetLineColor(ROOT.kBlue)
  hInput.SetLineWidth(2)
  hStack.Add(hInput, "PE")
  canv = ROOT.TCanvas()
  hStack.Draw("NOSTACK")
  # adjust y-range
  ROOT.gPad.Update()
  actualYRange = ROOT.gPad.GetUymax() - ROOT.gPad.GetUymin()
  yRangeFraction = 0.1 * actualYRange
  hStack.SetMaximum(ROOT.gPad.GetUymax() + yRangeFraction)
  hStack.SetMinimum(ROOT.gPad.GetUymin() - yRangeFraction)
  # adjust style of automatic zero line
  hStack.GetHistogram().SetLineColor(ROOT.kBlack)
  hStack.GetHistogram().SetLineStyle(ROOT.kDashed)
  # hStack.GetHistogram().SetLineWidth(0)  # remove zero line; see https://root-forum.cern.ch/t/continuing-the-discussion-from-an-unwanted-horizontal-line-is-drawn-at-y-0/50877/1
  canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
  canv.SaveAs(f"{pdfFileNamePrefix}{hStack.GetName()}.pdf")

  # plot residuals
  residuals = tuple((measVal[0] - trueVals[index]) / measVal[1] if measVal[1] > 0 else 0 for index, measVal in enumerate(measVals))
  hResidual = ROOT.TH1D(f"h{dataLabel}_Residuals_H{momentIndex if useMomentSubscript else ''}_{fileNameSuffix}",
    f"Residuals #it{{H}}{labelSubscript} {legendEntrySuffix};;(measured - input) / #sigma_{{measured}}", nmbBins, 0, nmbBins)
  chi2 = sum(tuple(residual**2 for residual in residuals[1 if momentIndex == 0 else 0:]))  # exclude Re and Im of H_0(0, 0) from chi^2
  ndf  = len(residuals[1 if momentIndex == 0 else 0:])
  for index, residual in enumerate(residuals):
    hResidual.SetBinContent(index + 1, residual)
    hResidual.SetBinError  (index + 1, 1e-100)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
    hResidual.GetXaxis().SetBinLabel(index + 1, hMeas.GetXaxis().GetBinLabel(index + 1))
  hResidual.SetMarkerColor(ROOT.kBlue)
  hResidual.SetLineColor(ROOT.kBlue)
  hResidual.SetLineWidth(2)
  hResidual.SetMinimum(-3)
  hResidual.SetMaximum(+3)
  canv = ROOT.TCanvas()
  hResidual.Draw("PE")
  # draw zero line
  xAxis = hResidual.GetXaxis()
  line = ROOT.TLine()
  line.SetLineStyle(ROOT.kDashed)
  line.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 0, xAxis.GetBinUpEdge(xAxis.GetLast()), 0)
  # shade 1 sigma region
  box = ROOT.TBox()
  box.SetFillColorAlpha(ROOT.kBlack, 0.15)
  box.DrawBox(xAxis.GetBinLowEdge(xAxis.GetFirst()), -1, xAxis.GetBinUpEdge(xAxis.GetLast()), +1)
  # draw chi^2 info
  label = ROOT.TLatex()
  label.SetNDC()
  label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignBottom)
  label.DrawLatex(0.12, 0.9075, f"#it{{#chi}}^{{2}}/n.d.f. = {chi2:.2f}/{ndf}, prob = {stats.distributions.chi2.sf(chi2, ndf) * 100:.0f}%")
  canv.SaveAs(f"{pdfFileNamePrefix}{hResidual.GetName()}.pdf")


def printAndPlotMoments(
  physMoments:       list[tuple[tuple[int, int, int], complex]],  # list(tuple(indices, value))
  physMomentsCov:    dict[tuple[int, int, int, int, int, int], tuple[float, float, float]],  # dict(tuple((indicesA, indicesB), (cov[ReRe], cov[ImIm], cov[ReIm])))
  trueMoments:       list[tuple[tuple[int, int, int], complex]] | None,  # list(tuple(indices, value)); if None true values are 1 for H_0(0, 0) and 0 for all other moments
  dataLabel:         str = "",
  pdfFileNamePrefix: str = "",
) -> None:
  # print moments
  for physMoment in physMoments:
    print(f"Re[H^phys_{physMoment[0][0]}(L = {physMoment[0][1]}, M = {physMoment[0][2]})] = {physMoment[1].real} +- {np.sqrt(physMomentsCov[(*physMoment[0], *physMoment[0])][0])}")  # diagonal element for ReRe
    print(f"Im[H^phys_{physMoment[0][0]}(L = {physMoment[0][1]}, M = {physMoment[0][2]})] = {physMoment[1].imag} +- {np.sqrt(physMomentsCov[(*physMoment[0], *physMoment[0])][1])}")  # diagonal element for ImIm
  # compare with true values
  for momentIndex in range(3):
    # Re[H_i]
    measVals = tuple((physMoment[1].real, np.sqrt(physMomentsCov[(*physMoment[0], *physMoment[0])][0]), physMoment[0]) for physMoment in physMoments if physMoment[0][0] == momentIndex)
    if trueMoments:
      trueVals = tuple(trueMoment[1].real for trueMoment in trueMoments if trueMoment[0][0] == momentIndex)
    else:
      trueVals = tuple(1 if physMoment[0] == (0, 0, 0) else 0 for physMoment in physMoments if physMoment[0][0] == momentIndex)
    plotComparison(measVals, trueVals, realPart = True, useMomentSubscript = True, dataLabel = dataLabel, pdfFileNamePrefix = pdfFileNamePrefix)
    # Im[H_i]
    measVals = tuple((physMoment[1].imag, np.sqrt(physMomentsCov[(*physMoment[0], *physMoment[0])][1]), physMoment[0]) for physMoment in physMoments if physMoment[0][0] == momentIndex)
    if trueMoments:
      trueVals = tuple(trueMoment[1].imag for trueMoment in trueMoments if trueMoment[0][0] == momentIndex)
    else:
      trueVals = tuple(0 for physMoment in physMoments if physMoment[0][0] == momentIndex)
    plotComparison(measVals, trueVals, realPart = False, useMomentSubscript = True, dataLabel = dataLabel, pdfFileNamePrefix = pdfFileNamePrefix)


# see e.g. LHCb, PRD 92 (2015) 112009
def generateDataLegPolLC(
  nmbEvents:         int,
  maxDegree:         int,
  parameters:        Collection[float],
  outFileNamePrefix: str = "",
) -> Any:
  """Generates data according to linear combination of Legendre polynomials"""
  assert len(parameters) >= maxDegree + 1, f"Need {maxDegree + 1} parameters; only {len(parameters)} were given: {parameters}"
  # linear combination of legendre polynomials up to given degree
  terms = tuple(f"[{degree}] * ROOT::Math::legendre({degree}, x)" for degree in range(maxDegree + 1))
  print("linear combination =", " + ".join(terms))
  legendrePolLC = ROOT.TF1("legendrePolLC", " + ".join(terms), -1, +1)
  legendrePolLC.SetNpx(1000)  # used in numeric integration performed by GetRandom()
  for index, parameter in enumerate(parameters):
    legendrePolLC.SetParameter(index, parameter)
  legendrePolLC.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()
  legendrePolLC.Draw()
  canv.SaveAs(f"{outFileNamePrefix}hLegendrePolLC.pdf")

  # generate random data that follow linear combination of legendre polynomials
  treeName = "data"
  fileName = f"{outFileNamePrefix}{legendrePolLC.GetName()}.root"
  RootUtilities.declareInCpp(legendrePolLC = legendrePolLC)
  df = ROOT.RDataFrame(nmbEvents) \
           .Define("cosTheta", "PyVars::legendrePolLC.GetRandom()") \
           .Define("theta",    "std::acos(cosTheta)") \
           .Filter('if (rdfentry_ == 0) { cout << "Running event loop in generateDataLegPolLC()" << endl; } return true;') \
           .Snapshot(treeName, fileName)  # snapshot is needed or else the `cosTheta` column would be regenerated for every triggered loop
                                          # noop filter before snapshot logs when event loop is running
  return df


def calculateLegMoments(
  dataFrame: Any,
  maxDegree: int,
) -> dict[tuple[int, ...], UFloat]:
  """Calculates moments of Legendre polynomials"""
  nmbEvents = dataFrame.Count().GetValue()
  moments: dict[tuple[int], UFloat] = {}
  for degree in range(maxDegree + 5):
    # unnormalized moments
    dfMoment = dataFrame.Define("legendrePol", f"ROOT::Math::legendre({degree}, cosTheta)")
    momentVal = dfMoment.Sum("legendrePol").GetValue()
    momentErr = math.sqrt(nmbEvents) * dfMoment.StdDev("legendrePol").GetValue()  # iid events: Var[sum_i^N f(x_i)] = sum_i^N Var[f] = N * Var[f]; see https://www.wikiwand.com/en/Monte_Carlo_integration
    # normalize moments with respect to H(0)
    legendrePolIntegral = 1 / (2 * degree + 1)  # = 1/2 * int_-1^+1; factor 1/2 takes into account integral for H(0)
    norm = 1 / (nmbEvents * legendrePolIntegral)
    moment = norm * ufloat(momentVal, momentErr)
    print(f"H(L = {degree}) = {moment}")
    moments[(degree, )] = moment
  print(moments)
  return moments


# follows Chung, PRD56 (1997) 7299
# see also
#     Suh-Urk's note Techniques of Amplitude Analysis for Two-pseudoscalar Systems (twobody0.pdf)
#     E852, PRD 60 (1999) 092001
#     https://en.wikipedia.org/wiki/Spherical_harmonics#Spherical_harmonics_expansion
def generateDataSphHarmLC(
  nmbEvents:         int,
  maxEll:            int,  # maximum spin of decaying object
  parameters:        Collection[float],  # make sure that resulting linear combination is positive definite
  outFileNamePrefix: str = "",
) -> Any:
  """Generates data according to linear combination of spherical harmonics"""
  nmbTerms = 6 * maxEll  # Eq. (17)
  assert len(parameters) >= nmbTerms, f"Need {nmbTerms} parameters; only {len(parameters)} were given: {parameters}"
  # linear combination of spherical harmonics up to given maximum orbital angular momentum
  # using Eq. (12) in Eq. (6): I = sum_L (2 L + 1 ) / (4pi)  H(L 0) (D_00^L)^* + sum_{M = 1}^L H(L M) 2 Re[D_M0^L]
  # and (D_00^L)^* = D_00^L = sqrt(4 pi / (2 L + 1)) (Y_L^0)^* = Y_L^0
  # and Re[D_M0^L] = d_M0^L cos(M phi) = Re[sqrt(4 pi / (2 L + 1)) (Y_L^M)^*] = Re[sqrt(4 pi / (2 L + 1)) Y_L^M]
  # i.e. Eq. (13) becomes: I = sum_L sqrt((2 L + 1 ) / (4pi)) sum_{M = 0}^L tau(M) H(L M) Re[Y_L^M]
  terms = []
  termIndex = 0
  for L in range(2 * maxEll + 1):
    termsM = []
    for M in range(min(L, 2) + 1):
      termsM.append(f"[{termIndex}] * {1 if M == 0 else 2} * ROOT::Math::sph_legendre({L}, {M}, std::acos(x)) * std::cos({M} * TMath::DegToRad() * y)")  # ROOT defines this as function of theta (not cos(theta)!); sigh
      termIndex += 1
    terms.append(f"std::sqrt((2 * {L} + 1 ) / (4 * TMath::Pi())) * ({' + '.join(termsM)})")
  print("linear combination =", " + ".join(terms))
  sphericalHarmLC = ROOT.TF2("sphericalHarmlLC", " + ".join(terms), -1, +1, -180, +180)
  sphericalHarmLC.SetNpx(500)  # used in numeric integration performed by GetRandom()
  sphericalHarmLC.SetNpy(500)
  sphericalHarmLC.SetContour(100)
  for index, parameter in enumerate(parameters):
    sphericalHarmLC.SetParameter(index, parameter)
  sphericalHarmLC.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()
  sphericalHarmLC.Draw("COLZ")
  canv.SaveAs(f"{outFileNamePrefix}hSphericalHarmlLC.pdf")

  # generate random data that follow linear combination of of spherical harmonics
  treeName = "data"
  fileName = f"{outFileNamePrefix}{sphericalHarmLC.GetName()}.root"
  RootUtilities.declareInCpp(sphericalHarmLC = sphericalHarmLC)
  df = ROOT.RDataFrame(nmbEvents) \
           .Define("point",    "double cosTheta, phiDeg; PyVars::sphericalHarmLC.GetRandom2(cosTheta, phiDeg); std::vector<double> point = {cosTheta, phiDeg}; return point;") \
           .Define("cosTheta", "point[0]") \
           .Define("theta",    "std::acos(cosTheta)") \
           .Define("phiDeg",   "point[1]") \
           .Define("phi",      "TMath::DegToRad() * phiDeg") \
           .Filter('if (rdfentry_ == 0) { cout << "Running event loop in generateDataSphHarmLC()" << endl; } return true;') \
           .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                          # noop filter before snapshot logs when event loop is running
  return df


# C++ implementation of RDataFrame custom action that calculates covariance between two columns
ROOT.gROOT.LoadMacro("./Covariance.C+")

def calculateSphHarmMoments(
  dataFrame:         Any,
  maxEll:            int,  # maximum spin of decaying object
  integralMatrix:    dict[tuple[int, ...], complex] | None = None,  # acceptance integral matrix
  pdfFileNamePrefix: str = "",
) -> tuple[list[tuple[tuple[int, int], complex]], dict[tuple[int, ...], tuple[float, ...]]]:  # moment values and covariances
  """Calculates moments of spherical harmonics"""
  # define moments
  dfMoment = dataFrame
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      # unnormalized moments
      dfMoment = dfMoment.Define(f"Re_f_{L}_{M}", f"std::sqrt((4 * TMath::Pi()) / (2 * {L} + 1)) * ReYlm({L}, {M}, theta, phi)")
      dfMoment = dfMoment.Define(f"Im_f_{L}_{M}", f"std::sqrt((4 * TMath::Pi()) / (2 * {L} + 1)) * ImYlm({L}, {M}, theta, phi)")
  # calculate moments and their covariance matrix
  # since we have iid events, Var[sum_i^N f(x_i)] = sum_i^N Var[f] = N * Var[f]; see https://www.wikiwand.com/en/Monte_Carlo_integration
  # similarly for the covariances
  nmbEvents = dataFrame.Count().GetValue()
  nmbMoments = (2 * maxEll + 2) * (2 * maxEll + 3) // 2
  H_meas = np.zeros((nmbMoments), dtype = np.complex128)
  # V_meas_ReRe = np.zeros((nmbMoments, nmbMoments), dtype = np.float64)
  # V_meas_ImIm = np.zeros((nmbMoments, nmbMoments), dtype = np.float64)
  # V_meas_ReIm = np.zeros((nmbMoments, nmbMoments), dtype = np.float64)
  Re_f = np.zeros((nmbMoments, nmbEvents), dtype = np.float64)
  Im_f = np.zeros((nmbMoments, nmbEvents), dtype = np.float64)
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      iMoment = L * (L + 1) // 2 + M
      # calculate value
      momentValRe = dfMoment.Sum(f"Re_f_{L}_{M}").GetValue()
      momentValIm = dfMoment.Sum(f"Im_f_{L}_{M}").GetValue()
      momentVal = momentValRe - 1j * momentValIm  # moment is defined by (Y_L^M)^*
      H_meas[iMoment] = momentVal
      # get values of spherical harmonics as Numpy arrays
      Re_f[iMoment, :] = dfMoment.AsNumpy(columns = [f"Re_f_{L}_{M}"])[f"Re_f_{L}_{M}"]
      Im_f[iMoment, :] = dfMoment.AsNumpy(columns = [f"Im_f_{L}_{M}"])[f"Im_f_{L}_{M}"]
      # # calculate value covariances
      # #TODO optimize by exploiting symmetry
      # for L_p in range(2 * maxEll + 2):
      #   for M_p in range(L_p + 1):
      #     iMoment_p = L_p * (L_p + 1) // 2 + M_p
      #     V_meas_ReRe[iMoment, iMoment_p] = nmbEvents * dfMoment.Book(ROOT.std.move(ROOT.Covariance["double"]()), [f"Re_f_{L}_{M}", f"Re_f_{L_p}_{M_p}"]).GetValue()
      #     V_meas_ImIm[iMoment, iMoment_p] = nmbEvents * dfMoment.Book(ROOT.std.move(ROOT.Covariance["double"]()), [f"Im_f_{L}_{M}", f"Im_f_{L_p}_{M_p}"]).GetValue()
      #     V_meas_ReIm[iMoment, iMoment_p] = nmbEvents * dfMoment.Book(ROOT.std.move(ROOT.Covariance["double"]()), [f"Re_f_{L}_{M}", f"Im_f_{L_p}_{M_p}"]).GetValue()
  # V_meas_ReRe_np = nmbEvents * np.cov(Re_f)
  # for L in range(2 * maxEll + 2):
  #   for M in range(L + 1):
  #     iMoment = L * (L + 1) // 2 + M
  #     momentErr = math.sqrt(nmbEvents) * dfMoment.StdDev(f"Re_f_{L}_{M}").GetValue()
  #     print(f"!!! {momentErr} vs. {math.sqrt(V_meas_ReRe_np[iMoment, iMoment])}; Delta = {momentErr - math.sqrt(V_meas_ReRe_np[iMoment, iMoment])}")
  # print(f"V_meas_ReRe =\n{V_meas_ReRe}")
  # print(f"vs.\n{V_meas_ReRe_np}")
  # print(f"ratio\n{np.real_if_close(V_meas_ReRe / V_meas_ReRe_np)}")
  # V_meas_ImIm_np = nmbEvents * np.cov(Im_f)
  # print(f"V_meas_ImIm =\n{V_meas_ImIm}")
  # print(f"vs.\n{V_meas_ImIm_np}")
  # print(f"ratio\n{np.real_if_close(V_meas_ImIm / V_meas_ImIm_np)}")
  # V_meas_ReIm_np = nmbEvents * np.cov(Re_f, Im_f)[:nmbMoments, nmbMoments:]  #!NOTE! numpy.cov(x, y) returns the covariance matrix for the stacked vector (x^T, y^T)^T
  # print(f"V_meas_ReIm =\n{V_meas_ReIm}")
  # print(f"vs.\n{V_meas_ReIm_np}")
  # print(f"ratio\n{np.real_if_close(V_meas_ReIm / V_meas_ReIm_np)}")
  # V_meas_Hermit = V_meas_ReRe + V_meas_ImIm + 1j * (V_meas_ReIm.T - V_meas_ReIm)  # Hermitian covariance matrix
  # V_meas_pseudo = V_meas_ReRe - V_meas_ImIm + 1j * (V_meas_ReIm.T + V_meas_ReIm)  # pseudo-covariance matrix
  # V_meas_aug = np.block([
  #   [V_meas_Hermit,               V_meas_pseudo],
  #   [np.conjugate(V_meas_pseudo), np.conjugate(V_meas_Hermit)],
  # ])  # augmented covariance matrix
  f = Re_f + 1j * Im_f
  V_meas_aug = nmbEvents * np.cov(f, np.conjugate(f))
  # V_meas_aug_np = nmbEvents * np.cov(f, np.conjugate(f))
  # print(f"V_meas_aug =\n{V_meas_aug}")
  # print(f"vs.\n{V_meas_aug_np}")
  # print(f"ratio\n{np.real_if_close(V_meas_aug / V_meas_aug_np)}")
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      iMoment = L * (L + 1) // 2 + M
      print(f"H_meas(L = {L}, M = {M}) = {H_meas[iMoment]}")
  # correct for detection efficiency
  H_phys = np.zeros((nmbMoments), dtype = np.complex128)
  V_phys_aug = np.zeros((2 * nmbMoments, 2 * nmbMoments), dtype = np.complex128)
  if integralMatrix is None:
    H_phys     = H_meas
    V_phys_aug = V_meas_aug
  else:
    I_acc = np.zeros((nmbMoments, nmbMoments), dtype = np.complex128)
    for L in range(2 * maxEll + 2):
      for M in range(L + 1):
        for Lp in range(2 * maxEll + 2):
          for Mp in range(Lp + 1):
            I_acc[L * (L + 1) // 2 + M, Lp * (Lp + 1) // 2 + Mp] = integralMatrix[(L, M, Lp, Mp)]
    print(f"I_acc = \n{np.array2string(I_acc, precision = 3, suppress_small = True, max_line_width = 150)}")
    eigenVals, eigenVecs = np.linalg.eig(I_acc)
    print(f"I_acc eigenvalues = {eigenVals}")
    # print(f"I_acc eigenvectors = {eigenVecs}")
    # print(f"I_acc determinant = {np.linalg.det(I_acc)}")
    # plot integral matrix and its inverse
    I_inv = np.linalg.inv(I_acc)
    # eigenVals, eigenVecs = np.linalg.eig(I_inv)
    # print(f"I^-1 eigenvalues = {eigenVals}")
    print(f"I^-1 = \n{np.array2string(I_inv, precision = 3, suppress_small = True, max_line_width = 150)}")
    matricesToPlot = [
      (I_acc.real,         f"{pdfFileNamePrefix}I_acc_real.pdf"),
      (I_acc.imag,         f"{pdfFileNamePrefix}I_acc_imag.pdf"),
      (np.absolute(I_acc), f"{pdfFileNamePrefix}I_acc_abs.pdf"),
      (np.angle(I_acc),    f"{pdfFileNamePrefix}I_acc_arg.pdf"),
      (I_inv.real,         f"{pdfFileNamePrefix}I_inv_real.pdf"),
      (I_inv.imag,         f"{pdfFileNamePrefix}I_inv_imag.pdf"),
      (np.absolute(I_inv), f"{pdfFileNamePrefix}I_inv_abs.pdf"),
      (np.angle(I_inv),    f"{pdfFileNamePrefix}I_inv_arg.pdf"),
    ]
    for matrix, pdfFileName in matricesToPlot:
      fig, ax = plt.subplots()
      fig.colorbar(ax.matshow(matrix), ax = ax)
      plt.savefig(pdfFileName)
    plt.close("all")
    H_phys = I_inv @ H_meas
    # linear uncertainty propagation
    J = I_inv  # Jacobian of efficiency correction
    J_conj = np.zeros((nmbMoments, nmbMoments), dtype = np.complex128)  # conjugate Jacobian
    J_aug = np.block([
      [J,                    J_conj],
      [np.conjugate(J_conj), np.conjugate(J)],
    ])  # augmented Jacobian
    V_phys_aug = J_aug @ (V_meas_aug @ np.asmatrix(J_aug).H)  #!NOTE! @ is left-associative
  # normalize such that H_0(0, 0) = 1
  norm = H_phys[0]
  H_phys /= norm
  V_phys_aug /= norm**2
  # calculate covariances of real and imaginary parts
  V_phys_Hermit = V_phys_aug[:nmbMoments, :nmbMoments]  # Hermitian covariance matrix
  V_phys_pseudo = V_phys_aug[:nmbMoments, nmbMoments:]  # pseudo-covariance matrix
  V_phys_ReRe = (np.real(V_phys_Hermit) + np.real(V_phys_pseudo)) / 2
  V_phys_ImIm = (np.real(V_phys_Hermit) - np.real(V_phys_pseudo)) / 2
  V_phys_ReIm = (np.imag(V_phys_pseudo) - np.imag(V_phys_Hermit)) / 2
  # reformat output
  momentsPhys:    list[tuple[tuple[int, int], complex]]    = []
  momentsPhysCov: dict[tuple[int, ...], tuple[float, ...]] = {}  # cov[(L, M, L', M')] = (ReRe, ImIm, ReIm)
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      iMoment = L * (L + 1) // 2 + M
      print(f"H_phys(L = {L}, M = {M}) = {H_phys[iMoment]}")
      momentsPhys.append(((L, M), H_phys[iMoment]))
      for L_p in range(2 * maxEll + 2):
        for M_p in range(L_p + 1):
          iMoment_p = L_p * (L_p + 1) // 2 + M_p
          momentsPhysCov[(L, M, L_p, M_p)] = (V_phys_ReRe[iMoment, iMoment_p], V_phys_ImIm[iMoment, iMoment_p], V_phys_ReIm[iMoment, iMoment_p])
  # print(momentsPhys)
  #TODO encapsulate moment values and covariances in object that takes care of the index mapping
  return momentsPhys, momentsPhysCov


WAVE_SET: dict[int, list[tuple[int, int]]] = {
  # negative-reflectivity waves
  -1 : [  # J, M, refl; see Eq. (41)
    (0, 0),  # S_0
    (1, 0),  # P_0
    (1, 1),  # P_-
    (2, 0),  # D_0
    (2, 1),  # D_-
  ],
  # positive-reflectivity waves
  +1 : [  # J, M, refl; see Eq. (42)
    (1, 1),  # P_+
    (2, 1),  # D_+
  ],
}

# follows Chung, PRD56 (1997) 7299
# see also
#     Suh-Urk's note Techniques of Amplitude Analysis for Two-pseudoscalar Systems (twobody0.pdf)
#     E852, PRD 60 (1999) 092001
#     https://en.wikipedia.org/wiki/Spherical_harmonics#Spherical_harmonics_expansion
def generateDataPwd(
  nmbEvents:         int,
  prodAmps:          dict[int, tuple[complex, ...]],
  efficiencyFormula: str | None = None,
  outFileNamePrefix: str = "",
) -> Any:
  """Generates data according to partial-wave decomposition for fixed set of 7 lowest waves up to \\ell = 2 and |m| = 1"""
  # construct TF2 for intensity in Eq. (28) with rank = 1 and using wave set in Eqs. (41) and (42)
  assert len(prodAmps) == len(WAVE_SET), f"Need {len(WAVE_SET)} parameters; only {len(prodAmps)} were given: {prodAmps}"
  incoherentTerms = []
  for refl in (-1, +1):
    coherentTerms = []
    for waveIndex, wave in enumerate(WAVE_SET[refl]):
      ell:    int = wave[0]
      m:      int = wave[1]
      parity: int = (-1)**ell
      # see Eqs. (26) and (27) for rank = 1
      V = f"complexT({prodAmps[refl][waveIndex].real}, {prodAmps[refl][waveIndex].imag})"  # complexT is a typedef for std::complex<double> in `basisFunctions.C`
      A = f"std::sqrt((2 * {ell} + 1) / (4 * TMath::Pi())) * wignerDReflConj({2 * ell}, {2 * m}, 0, {parity}, {refl}, TMath::DegToRad() * y, std::acos(x))"
      coherentTerms.append(f"{V} * {A}")
    incoherentTerms.append(f"std::norm({' + '.join(coherentTerms)})")
  # see Eq. (28) for rank = 1
  intensityFormula = f"({' + '.join(incoherentTerms)})" + ("" if efficiencyFormula is None else f" * ({efficiencyFormula})")
  print(f"intensity = {intensityFormula}")
  intensityFcn = ROOT.TF2("intensity", intensityFormula, -1, +1, -180, +180)
  intensityFcn.SetTitle(";cos#theta;#phi [deg]")
  intensityFcn.SetNpx(500)  # used in numeric integration performed by GetRandom()
  intensityFcn.SetNpy(500)
  intensityFcn.SetContour(100)
  intensityFcn.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()
  intensityFcn.Draw("COLZ")
  canv.SaveAs(f"{outFileNamePrefix}hIntensity.pdf")

  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = f"{outFileNamePrefix}{intensityFcn.GetName()}.root"
  RootUtilities.declareInCpp(intensityFcn = intensityFcn)
  df = ROOT.RDataFrame(nmbEvents) \
           .Define("point",    "double cosTheta, phiDeg; PyVars::intensityFcn.GetRandom2(cosTheta, phiDeg); std::vector<double> point = {cosTheta, phiDeg}; return point;") \
           .Define("cosTheta", "point[0]") \
           .Define("theta",    "std::acos(cosTheta)") \
           .Define("phiDeg",   "point[1]") \
           .Define("phi",      "TMath::DegToRad() * phiDeg") \
           .Filter('if (rdfentry_ == 0) { cout << "Running event loop in generateDataPwd()" << endl; } return true;') \
           .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                          # noop filter before snapshot logs when event loop is running
  return df


def normFactor(m: int) -> float:
  """Calculates normalization factor in reflectivity basis"""
  # see Eq. (19)
  if m > 0:
    return 1 / math.sqrt(2)
  elif m == 0:
    return 1 / 2
  else:
    return 0


def calculateInputPwdMoment(
  prodAmps: dict[int, tuple[complex, ...]],
  L: int,
  M: int,
) -> complex:
  """Calculates value of moment with L and M for given production amplitudes"""
  # Eq. (29) for rank = 1
  sum = 0 + 0j
  for refl in (-1, +1):
    for waveIndex_1, wave_1 in enumerate(WAVE_SET[refl]):
      ell_1: int = wave_1[0]
      m_1:   int = wave_1[1]
      for waveIndex_2, wave_2 in enumerate(WAVE_SET[refl]):
        ell_2: int = wave_2[0]
        m_2:   int = wave_2[1]
        b = normFactor(m_2) * normFactor(m_1) * (
                               py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1,  2 * m_2,  2 * M,  2 * m_1, ignore_invalid = True)  # (ell_2  m_2  L  M | ell_1  m_1)
          + (-1)**M *          py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1,  2 * m_2, -2 * M,  2 * m_1, ignore_invalid = True)  # (ell_2  m_2  L -M | ell_1  m_1)
          - refl * (-1)**m_2 * py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1, -2 * m_2,  2 * M,  2 * m_1, ignore_invalid = True)  # (ell_2 -m_2  L  M | ell_1  m_1)
          - refl * (-1)**m_1 * py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1,  2 * m_2,  2 * M, -2 * m_1, ignore_invalid = True)  # (ell_2  m_2  L  M | ell_1 -m_1)
        )
        sum += math.sqrt((2 * ell_2 + 1) / (2 * ell_1 + 1)) * prodAmps[refl][waveIndex_1] * prodAmps[refl][waveIndex_2].conjugate() * b \
               * py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1, 0, 0, 0, ignore_invalid = True)  # (ell_2 0  L 0 | ell_1 0)
  return sum


def calculateInputPwdMoments(
  prodAmps: dict[int, tuple[complex, ...]],
  maxEll:   int,  # maximum spin of decaying object
) -> list[float]:
  """Calculates moments for given production amplitudes"""
  moments: list[float] = []
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      moment: complex = calculateInputPwdMoment(prodAmps, L, M)
      if (abs(moment.imag) > 1e-15):
        print(f"Warning: non vanishing imaginary part for moment H({L} {M}) = {moment}")
      moments.append(moment.real)
  # normalize to first moment
  moments = [moment / moments[0] for moment in moments]
  index = 0
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      print(f"H_input(L = {L}, M = {M}) = {moments[index]}")
      index += 1
  # print(moments)
  return moments


def calculateWignerDMoment(
  dataFrame: Any,
  L:         int,
  M:         int,
) -> tuple[UFloat, UFloat]:  # real and imag part with uncertainty
  """Calculates unnormalized moment of Wigner-D function D^L_{M 0}"""
  # unnormalized moment
  dfMoment = dataFrame.Define("WignerD",  f"wignerD({2 * L}, {2 * M}, 0, phi, theta)") \
                      .Define("WignerDRe", "real(WignerD)") \
                      .Define("WignerDIm", "imag(WignerD)")
  momentVal   = dfMoment.Sum[ROOT.std.complex["double"]]("WignerD").GetValue()
  # iid events: Var[sum_i^N f(x_i)] = sum_i^N Var[f] = N * Var[f]; see https://www.wikiwand.com/en/Monte_Carlo_integration
  momentErrRe = math.sqrt(nmbEvents) * dfMoment.StdDev("WignerDRe").GetValue()
  momentErrIm = math.sqrt(nmbEvents) * dfMoment.StdDev("WignerDIm").GetValue()
  return ufloat(momentVal.real, momentErrRe), ufloat(momentVal.imag, momentErrIm)


def calculateWignerDMoments(
  dataFrame: Any,
  maxEll:    int,  # maximum spin of decaying object
) -> None:
  """Calculates moments of Wigner-D function D^L_{M 0}"""
  nmbEvents = dataFrame.Count().GetValue()
  # moments: list[tuple[tuple[int, int], UFloat]] = []
  for L in range(2 * maxEll + 2):
    for M in range(-L, +L + 1):
      # unnormalized moments
      momentRe, momentIm = calculateWignerDMoment(dataFrame, L, M)
      # normalize moments with respect to H(0 0)
      norm = 1 / nmbEvents
      momentRe *= norm
      momentIm *= norm
      print(f"H(L = {L}, M = {M}) = {(momentRe, momentIm)}")
  #     moments.append(((L, M), moment))
  # print(moments)


def generateData2BodyPS(
  nmbEvents:         int,  # number of events to generate
  efficiencyFormula: str | None = None,
  outFileNamePrefix: str = "",
) -> Any:
  """Generates RDataFrame with two-body phase-space distribution weighted by given detection efficiency"""
  # construct efficiency function
  efficiencyFcn = ROOT.TF2("efficiency", "1" if efficiencyFormula is None else efficiencyFormula, -1, +1, -180, +180)
  efficiencyFcn.SetTitle(";cos#theta;#phi [deg]")
  efficiencyFcn.SetNpx(500)  # used in numeric integration performed by GetRandom()
  efficiencyFcn.SetNpy(500)
  efficiencyFcn.SetContour(100)
  efficiencyFcn.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()
  efficiencyFcn.Draw("COLZ")
  canv.SaveAs(f"{outFileNamePrefix}hEfficiency.pdf")

  # generate isotropic distributions in cos theta and phi and weight with efficiency function
  treeName = "data"
  fileName = f"{outFileNamePrefix}{efficiencyFcn.GetName()}.root"
  RootUtilities.declareInCpp(efficiencyFcn = efficiencyFcn)
  df = ROOT.RDataFrame(nmbEvents) \
           .Define("point", "double cosTheta, phiDeg; PyVars::efficiencyFcn.GetRandom2(cosTheta, phiDeg); std::vector<double> point = {cosTheta, phiDeg}; return point;") \
           .Define("cosTheta", "point[0]") \
           .Define("theta",    "std::acos(cosTheta)") \
           .Define("phiDeg",   "point[1]") \
           .Define("phi",      "TMath::DegToRad() * phiDeg") \
           .Filter('if (rdfentry_ == 0) { cout << "Running event loop in generateData2BodyPS()" << endl; } return true;') \
           .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                          # noop filter before snapshot logs when event loop is running
  return ROOT.RDataFrame(treeName, fileName)


def calcIntegralMatrix(
  phaseSpaceDataFrame: Any,
  maxEll:              int,  # maximum orbital angular momentum
  nmbEvents:           int,  # number of events in RDataFrame
) -> dict[tuple[int, ...], complex]:
  """Calculates integral matrix of spherical harmonics from provided phase-space data"""
  # define spherical harmonics
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      phaseSpaceDataFrame = phaseSpaceDataFrame.Define(   f"Y_{L}_{M}",   f"Ylm({L}, {M}, theta, phi)")
      phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"Re_Y_{L}_{M}", f"ReYlm({L}, {M}, theta, phi)")
      phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"Im_Y_{L}_{M}", f"ImYlm({L}, {M}, theta, phi)")
  # define integral matrix
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      for Lp in range(2 * maxEll + 2):
        for Mp in range(Lp + 1):
          # phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"I_{L}_{M}_{Lp}_{Mp}", f"(4 * TMath::Pi() / {nmbEvents}) * std::sqrt((double)(2 * {Lp} + 1) / (2 * {L} + 1)) * Y_{Lp}_{Mp} * std::conj(Y_{L}_{M})")
          phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"I_{L}_{M}_{Lp}_{Mp}", f"(4 * TMath::Pi() / {nmbEvents}) * std::sqrt((double)(2 * {Lp} + 1) / (2 * {L} + 1)) * (2 - ({Mp} == 0)) * Re_Y_{Lp}_{Mp} * std::conj(Y_{L}_{M})")
          # phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"I_{L}_{M}_{Lp}_{Mp}", f"(4 * TMath::Pi() / {nmbEvents}) * std::sqrt((double)(2 * {Lp} + 1) / (2 * {L} + 1)) * (2 - ({Mp} == 0)) * Im_Y_{Lp}_{Mp} * std::conj(Y_{L}_{M})")
  # calculate integral matrix
  I: dict[tuple[int, ...], complex] = {}
  for L in range(2 * maxEll + 2):
    for M in range(L + 1):
      for Lp in range(2 * maxEll + 2):
        for Mp in range(Lp + 1):
          I[(L, M, Lp, Mp)] = phaseSpaceDataFrame.Sum[ROOT.std.complex["double"]](f"I_{L}_{M}_{Lp}_{Mp}").GetValue()
          # print(f"I_{L}_{M}_{Lp}_{Mp} = {I[(L, M, Lp, Mp)]}")
  # phaseSpaceDataFrame.Snapshot("foo", "foo.root", ["I_0_0_1_0", "I_1_0_0_0", "Re_Y_0_0", "Re_Y_1_0", "Y_0_0", "Y_1_0"])
  return I


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  ROOT.gROOT.SetBatch(True)
  ROOT.gRandom.SetSeed(1234567890)
  # ROOT.EnableImplicitMT(10)
  setupPlotStyle()
  threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
  print(f"Initial state of ThreadpoolController before setting number of threads\n{threadController.info()}")
  with threadController.limit(limits = 3):
    print(f"State of ThreadpoolController after setting number of threads\n{threadController.info()}")
    timer.start("Total execution time")

    # set parameters of test case
    outFileDirName    = Utilities.makeDirPath("./plotsTestDiffractive")
    nmbEvents         = 1000
    nmbMcEvents       = 1000000
    # formulas for detection efficiency: x = cos(theta), y = phi in [-180, +180] deg
    # efficiencyFormula = "1"  # acc_perfect
    # efficiencyFormula = "2 - x * x"  # acc_1
    # efficiencyFormula = "1 - x * x"  # acc_2
    # efficiencyFormula = "180 * 180 - y * y"  # acc_3
    efficiencyFormula = "0.25 + (1 - x * x) * (1 - y * y / (180 * 180))"  # acc_4
    # efficiencyFormula = "(-4 * ((x + 1) / 2 - 1) * ((x + 1) / 2) * ((x + 1) / 2) * ((x + 1) / 2))"  # Mathematica: PiecewiseExpand[BernsteinBasis[4,3,x]]
    # efficiencyFormula += " * ((((y + 180) / 360) / 3) * (2 * ((y + 180) / 360) * (5 * ((y + 180) / 360) - 9) + 9))"  # PiecewiseExpand[BernsteinBasis[3,1,x]+BernsteinBasis[3,3,x]/3]; acc_5

    # # Legendre polynomials
    # chose parameters such that resulting linear combinations are positive definite
    # maxOrder = 5
    # # parameters = (1, 1, 0.5, -0.5, -0.25, 0.25)
    # parameters = (0.5, 0.5, 0.25, -0.25, -0.125, 0.125)
    # dataModel = generateDataLegPolLC(nmbEvents,  maxDegree = maxOrder, parameters = parameters)

    # # spherical harmonics
    # maxOrder = 2
    # # parameters = (1, 0.025, 0.02, 0.015, 0.01, -0.02, 0.025, -0.03, -0.035, 0.04, 0.045, 0.05)
    # parameters = (2, 0.05, 0.04, 0.03, 0.02, -0.04, 0.05, -0.06, -0.07, 0.08, 0.09, 0.10)
    # dataModel = generateDataSphHarmLC(nmbEvents, maxEll = maxOrder, parameters = parameters)

    # normalize parameters 0th moment and pad with 0
    # inputMoments = [par / parameters[0] for par in parameters]
    # if len(inputMoments) < len(moments):
    #   inputMoments += [0] * (len(moments) - len(inputMoments))

    # partial-wave decomposition
    maxEll = 2
    prodAmps: dict[int, tuple[complex, ...]] = {
      # negative-reflectivity waves
      -1 : (
         1   + 0j,    # S_0
         0.3 - 0.8j,  # P_0
        -0.4 + 0.1j,  # P_-
        -0.1 - 0.2j,  # D_0
         0.2 - 0.5j,  # D_-
      ),
      # positive-reflectivity waves
      +1 : (
         0.5 + 0j,    # P_+
        -0.1 + 0.3j,  # D_+
      ),
    }
    inputMoments: list[float] = calculateInputPwdMoments(prodAmps, maxEll = maxEll)
    dataPwaModel = generateDataPwd(nmbEvents, prodAmps, efficiencyFormula, outFileNamePrefix = f"{outFileDirName}/")
    # print("!!!", dataModel.AsNumpy())

    # plot data
    canv = ROOT.TCanvas()
    if "phi" in dataPwaModel.GetColumnNames():
      hist = dataPwaModel.Histo2D(ROOT.RDF.TH2DModel("hData", ";cos#theta;#phi [deg]", 25, -1, +1, 25, -180, +180), "cosTheta", "phiDeg")
      hist.SetMinimum(0)
      hist.Draw("COLZ")
    else:
      hist = dataPwaModel.Histo1D(ROOT.RDF.TH1DModel("hData", ";cos#theta", 100, -1, +1), "cosTheta")
      hist.SetMinimum(0)
      hist.Draw()
    canv.SaveAs(f"{outFileDirName}/{hist.GetName()}.pdf")

    # calculate moments
    dataAcceptedPS = generateData2BodyPS(nmbMcEvents, efficiencyFormula, outFileNamePrefix = f"{outFileDirName}/")
    integralMatrix = calcIntegralMatrix(dataAcceptedPS, maxEll = maxEll, nmbEvents = nmbMcEvents)
    print("Moments of accepted phase-space data")
    calculateSphHarmMoments(dataAcceptedPS, maxEll = maxEll, integralMatrix = integralMatrix, pdfFileNamePrefix = f"{outFileDirName}/")
    # calculateLegMoments(dataModel, maxDegree = maxOrder)
    print("Moments of data generated according to model")
    moments:    list[tuple[tuple[int, int], complex]]
    momentsCov: dict[tuple[int, ...], tuple[float, ...]]
    moments, momentsCov = calculateSphHarmMoments(dataPwaModel, maxEll = maxEll, integralMatrix = integralMatrix, pdfFileNamePrefix = f"{outFileDirName}/")
    # calculateWignerDMoments(dataModel, maxEll = maxOrder)
    #TODO check whether using Eq. (6) instead of Eq. (13) yields moments that fulfill Eqs. (11) and (12)

    # Re[H_i]
    measVals  = tuple((moment[1].real, math.sqrt(momentsCov[(*moment[0], *moment[0])][0]), (0, *moment[0])) for moment in moments)
    inputVals = tuple(inputMoment.real for inputMoment in inputMoments)
    plotComparison(measVals, inputVals, realPart = True, useMomentSubscript = False, pdfFileNamePrefix = f"{outFileDirName}/")
    # Im[H_i]
    measVals  = tuple((moment[1].imag, math.sqrt(momentsCov[(*moment[0], *moment[0])][1]), (0, *moment[0])) for moment in moments)
    inputVals = tuple(inputMoment.imag for inputMoment in inputMoments)
    plotComparison(measVals, inputVals, realPart = False, useMomentSubscript = False, pdfFileNamePrefix = f"{outFileDirName}/")

    timer.stop("Total execution time")
    print(timer.summary)
