#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3

# always flush print() to reduce garbling of log files due to buffering
import functools
print = functools.partial(print, flush = True)


import ctypes
import numpy as np
import nptyping as npt
from typing import (
  Any,
  Dict,
  List,
  Optional,
  Sequence,
  Tuple,
  TypedDict,
)

import py3nj
import ROOT

import MomentCalculator
import OpenMp
import PlottingUtilities
import RootUtilities


# set of all possible waves up to ell = 2
# PROD_AMPS: Dict[int, Dict[Tuple[int, int,], complex]] = {
#   # negative-reflectivity waves
#   -1 : {
#     #J   M    amplitude
#     (0,  0) :  1   + 0j,    # S_0^-
#     (1, -1) : -0.4 + 0.1j,  # P_-1^-
#     (1,  0) :  0.3 - 0.8j,  # P_0^-
#     (1, +1) : -0.8 + 0.7j,  # P_+1^-
#     (2, -2) :  0.1 - 0.4j,  # D_-2^-
#     (2, -1) :  0.5 + 0.2j,  # D_-1^-
#     (2,  0) : -0.1 - 0.2j,  # D_ 0^-
#     (2, +1) :  0.2 - 0.1j,  # D_+1^-
#     (2, +2) : -0.2 + 0.3j,  # D_+2^-
#   },
#   # positive-reflectivity waves
#   +1 : {
#     #J   M    amplitude
#     (0,  0) :  0.5 + 0j,    # S_0^+
#     (1, -1) :  0.5 - 0.1j,  # P_-1^+
#     (1,  0) : -0.8 - 0.3j,  # P_0^+
#     (1, +1) :  0.6 + 0.3j,  # P_+1^+
#     (2, -2) :  0.2 + 0.1j,  # D_-2^+
#     (2, -1) :  0.2 - 0.3j,  # D_-1^+
#     (2,  0) :  0.1 - 0.2j,  # D_ 0^+
#     (2, +1) :  0.2 + 0.5j,  # D_+1^+
#     (2, +2) : -0.3 - 0.1j,  # D_+2^+
#   },
# }
PROD_AMPS2 = [
  # negative-reflectivity waves
  #                                                           refl J   M    amplitude
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 0,  0),  1.0 + 0.0j),  # S_0^-
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 1, -1), -0.4 + 0.1j),  # P_-1^-
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 1,  0),  0.3 - 0.8j),  # P_0^-
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 1, +1), -0.8 + 0.7j),  # P_+1^-
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2, -2),  0.1 - 0.4j),  # D_-2^-
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2, -1),  0.5 + 0.2j),  # D_-1^-
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2,  0), -0.1 - 0.2j),  # D_ 0^-
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2, +1),  0.2 - 0.1j),  # D_+1^-
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(-1, 2, +2), -0.2 + 0.3j),  # D_+2^-
  # positive-reflectivity waves
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 0,  0),  0.5 + 0.0j),  # S_0^+
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 1, -1),  0.5 - 0.1j),  # P_-1^+
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 1,  0), -0.8 - 0.3j),  # P_0^+
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 1, +1),  0.6 + 0.3j),  # P_+1^+
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2, -2),  0.2 + 0.1j),  # D_-2^+
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2, -1),  0.2 - 0.3j),  # D_-1^+
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2,  0),  0.1 - 0.2j),  # D_ 0^+
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2, +1),  0.2 + 0.5j),  # D_+1^+
  MomentCalculator.AmplitudeValue(MomentCalculator.QnWaveIndex(+1, 2, +2), -0.3 - 0.1j),  # D_+2^+
]



# define maximum L quantum number of moments
MAX_L = 5


# default TH3 plotting options
TH3_NMB_BINS = 25
TH3_BINNINGS = (
  PlottingUtilities.HistAxisBinning(TH3_NMB_BINS,   -1,   +1),
  PlottingUtilities.HistAxisBinning(TH3_NMB_BINS, -180, +180),
  PlottingUtilities.HistAxisBinning(TH3_NMB_BINS, -180, +180),
)
TH3_TITLE = ";cos#theta;#phi [deg];#Phi [deg]"
class Th3PlotKwargsType(TypedDict):
  binnings:  Tuple[PlottingUtilities.HistAxisBinning, PlottingUtilities.HistAxisBinning, PlottingUtilities.HistAxisBinning]
  histTitle: str
TH3_PLOT_KWARGS: Th3PlotKwargsType = {"histTitle" : TH3_TITLE, "binnings" : TH3_BINNINGS}


# def calcSpinDensElemSetFromWaves(
#   refl:         int,      # reflectivity
#   m1:           int,      # m
#   m2:           int,      # m'
#   prodAmp1:     complex,  # [ell]_m^refl
#   prodAmp1NegM: complex,  # [ell]_{-m}^refl
#   prodAmp2:     complex,  # [ell']_m'^refl
#   prodAmp2NegM: complex,  # [ell']_{-m'}^refl
# ) -> Tuple[complex, complex, complex]:
#   """Calculates element of spin-density matrix components from given partial-wave amplitudes assuming rank 1"""
#   rhos: List[complex] = 3 * [0 + 0j]
#   rhos[0] +=                    (           prodAmp1     * prodAmp2.conjugate() + (-1)**(m1 - m2) * prodAmp1NegM * prodAmp2NegM.conjugate())  # Eq. (150)
#   rhos[1] +=            -refl * ((-1)**m1 * prodAmp1NegM * prodAmp2.conjugate() + (-1)**m2        * prodAmp1     * prodAmp2NegM.conjugate())  # Eq. (151)
#   rhos[2] += -(0 + 1j) * refl * ((-1)**m1 * prodAmp1NegM * prodAmp2.conjugate() - (-1)**m2        * prodAmp1     * prodAmp2NegM.conjugate())  # Eq. (152)
#   return (rhos[0], rhos[1], rhos[2])


# def calcMomentSetFromWaves(
#   prodAmps: Dict[int, Dict[Tuple[int, int,], complex]],
#   L:        int,
#   M:        int,
# ) -> Tuple[complex, complex, complex]:
#   """Calculates values of (H_0, H_1, H_2) with L and M from given production amplitudes assuming rank 1"""
#   # Eqs. (154) to (156) assuming that rank is 1
#   moments: List[complex] = 3 * [0 + 0j]
#   for refl in (-1, +1):
#     for wave1 in prodAmps[refl]:
#       ell1:         int     = wave1[0]
#       m1:           int     = wave1[1]
#       prodAmp1:     complex = prodAmps[refl][wave1]
#       prodAmp1NegM: complex = prodAmps[refl][(ell1, -m1)]
#       for wave2 in prodAmps[refl]:
#         ell2:         int     = wave2[0]
#         m2:           int     = wave2[1]
#         prodAmp2:     complex = prodAmps[refl][wave2]
#         prodAmp2NegM: complex = prodAmps[refl][(ell2, -m2)]
#         term = np.sqrt((2 * ell2 + 1) / (2 * ell1 + 1)) * (
#             py3nj.clebsch_gordan(2 * ell2, 2 * L, 2 * ell1, 0,      0,     0,      ignore_invalid = True)  # (ell_2 0,    L 0 | ell_1 0  )
#           * py3nj.clebsch_gordan(2 * ell2, 2 * L, 2 * ell1, 2 * m2, 2 * M, 2 * m1, ignore_invalid = True)  # (ell_2 m_2,  L M | ell_1 m_1)
#         )
#         rhos: Tuple[complex, complex, complex] = calcSpinDensElemSetFromWaves(refl, m1, m2, prodAmp1, prodAmp1NegM, prodAmp2, prodAmp2NegM)
#         # print(f"!!! {refl}; ({ell1}, {m1}); ({ell2}, {m2}) = {rhos}")
#         if term == 0:  # invalid Clebsch-Gordan
#           continue
#         moments[0] +=  term * rhos[0]  # H_0; Eq. (124)
#         moments[1] += -term * rhos[1]  # H_1; Eq. (125)
#         moments[2] += -term * rhos[2]  # H_2; Eq. (125)
#   return (moments[0], moments[1], moments[2])


# def calcAllMomentsFromWaves(
#   prodAmps: Dict[int, Dict[Tuple[int, int], complex]],  # Dict[reflectivity, Dict[(L, M), value]]
#   maxL:     int,  # maximum L quantum number of moments
# ) -> MomentCalculator.MomentResult:
#   """Calculates moments for given production amplitudes assuming rank 1; the H_2(L, 0) are omitted"""
#   momentIndices = MomentCalculator.MomentIndices(maxL)
#   momentsFlatIndex = np.zeros((len(momentIndices), ), dtype = npt.Complex128)
#   norm = 1.0
#   for L in range(maxL + 1):
#     for M in range(L + 1):
#       # get all moments for given (L, M)
#       moments: List[complex] = list(calcMomentSetFromWaves(prodAmps, L, M))
#       tolerance = 1e-15
#       assert (abs(moments[0].imag) < tolerance) and (abs(moments[1].imag) < tolerance) and (abs(moments[2].real) < tolerance), (
#         f"expect (Im[H_0({L} {M})], Im[H_1({L} {M})], and Re[H_2({L} {M})]) < {tolerance} but found ({moments[0].imag}, {moments[1].imag}, {moments[2].real})")
#       # set respective real and imaginary parts exactly to zero.
#       moments[0] = moments[0].real + 0j
#       moments[1] = moments[1].real + 0j
#       moments[2] = 0 + moments[2].imag * 1j
#       assert M != 0 or (M == 0 and moments[2] == 0), f"expect H_2({L} {M}) == 0 but found {moments[2].imag}"
#       # normalize to H_0(0, 0)
#       if L == M == 0:
#         assert moments[0].imag == 0, f"Expect H_0(0, 0) to be real-valued but got Im[H_0(0, 0)] = {moments[0].imag}."
#         norm = moments[0].real  # H_0(0, 0)
#       for momentIndex, moment in enumerate(moments[:2 if M == 0 else 3]):
#         qnIndex   = MomentCalculator.QnMomentIndex(momentIndex, L, M)
#         flatIndex = momentIndices.indexMap.flatIndex_for[qnIndex]
#         momentsFlatIndex[flatIndex] = moment / norm
#   HTrue = MomentCalculator.MomentResult(momentIndices, label = "true")
#   HTrue._valsFlatIndex = momentsFlatIndex
#   return HTrue


def genDataFromWaves(
  nmbEvents:         int,                            # number of events to generate
  polarization:      float,                          # photon-beam polarization
  # prodAmps:          Dict[int, Dict[Tuple[int, int,], complex]],  # partial-wave amplitudes
  amplitudeSet:      MomentCalculator.AmplitudeSet,  # partial-wave amplitudes
  efficiencyFormula: Optional[str] = None,           # detection efficiency used to generate data
) -> ROOT.RDataFrame:
  """Generates data according to set of partial-wave amplitudes (assuming rank 1) and given detection efficiency"""
  # construct and draw efficiency function
  efficiencyFcn = ROOT.TF3("efficiencyGen", efficiencyFormula if efficiencyFormula else "1", -1, +1, -180, +180, -180, +180)
  PlottingUtilities.drawTF3(efficiencyFcn, **TH3_PLOT_KWARGS, pdfFileName = "./hEfficiencyGen.pdf", nmbPoints = 100, maxVal = 1.0)

  # construct TF3 for intensity distribution in Eq. (153)
  # x = cos(theta) in [-1, +1], y = phi in [-180, +180] deg, z = Phi in [-180, +180] deg
  intensityComponentTerms: List[Tuple[str, str, str]] = []  # terms in sum of each intensity component
  for refl in (-1, +1):
    for amp1 in amplitudeSet.amplitudes(onlyRefl = refl):
      l1 = amp1.qn.l
      m1 = amp1.qn.m
      decayAmp1 = f"Ylm({l1}, {m1}, std::acos(x), TMath::DegToRad() * y)"
      for amp2 in amplitudeSet.amplitudes(onlyRefl = refl):
        l2 = amp2.qn.l
        m2 = amp2.qn.m
        decayAmp2 = f"Ylm({l2}, {m2}, std::acos(x), TMath::DegToRad() * y)"
        rhos: Tuple[complex, complex, complex] = amplitudeSet.spinDensElementSet(refl, l1, l2, m1, m2)
        terms = tuple(f"{decayAmp1} * complexT({rho.real}, {rho.imag}) * std::conj({decayAmp2})" for rho in rhos)  # Eq. (153)
        intensityComponentTerms.append((terms[0], terms[1], terms[2]))
  # for refl in (-1, +1):
  #   for wave1 in prodAmps[refl]:
  #     ell1:         int     = wave1[0]
  #     m1:           int     = wave1[1]
  #     prodAmp1:     complex = prodAmps[refl][wave1]
  #     prodAmp1NegM: complex = prodAmps[refl][(ell1, -m1)]
  #     decayAmp1 = f"Ylm({ell1}, {m1}, std::acos(x), TMath::DegToRad() * y)"
  #     for wave2 in prodAmps[refl]:
  #       ell2:         int     = wave2[0]
  #       m2:           int     = wave2[1]
  #       prodAmp2:     complex = prodAmps[refl][wave2]
  #       prodAmp2NegM: complex = prodAmps[refl][(ell2, -m2)]
  #       decayAmp2 = f"Ylm({ell2}, {m2}, std::acos(x), TMath::DegToRad() * y)"
  #       rhos: Tuple[complex, complex, complex] = calcSpinDensElemSetFromWaves(refl, m1, m2, prodAmp1, prodAmp1NegM, prodAmp2, prodAmp2NegM)
  #       terms = tuple(f"{decayAmp1} * complexT({rho.real}, {rho.imag}) * std::conj({decayAmp2})" for rho in rhos)  # Eq. (153)
  #       intensityComponentTerms.append((terms[0], terms[1], terms[2]))
  # sum terms for each intensity component
  intensityComponentsFormula = []
  for iComponent in range(3):
    intensityComponentsFormula.append(f"({' + '.join([term[iComponent] for term in intensityComponentTerms])})")
  # sum intensity components
  intensityFormula = (
    f"std::real({intensityComponentsFormula[0]} "
    f"- {intensityComponentsFormula[1]} * {polarization} * std::cos(2 * TMath::DegToRad() * z) "
    f"- {intensityComponentsFormula[2]} * {polarization} * std::sin(2 * TMath::DegToRad() * z))"
    + (f" * ({efficiencyFormula})" if efficiencyFormula else ""))  # Eq. (163)
  print(f"intensity = {intensityFormula}")
  intensityFcn = ROOT.TF3("intensity", intensityFormula, -1, +1, -180, +180, -180, +180)
  intensityFcn.SetTitle(";cos#theta;#phi [deg];#Phi [deg]")
  intensityFcn.SetNpx(100)  # used in numeric integration performed by GetRandom()
  intensityFcn.SetNpy(100)
  intensityFcn.SetNpz(100)
  intensityFcn.SetMinimum(0)
  PlottingUtilities.drawTF3(intensityFcn, **TH3_PLOT_KWARGS, pdfFileName = "./hIntensity.pdf")

  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = f"{intensityFcn.GetName()}.photoProd.root"
  #TODO switch that allows loading from file
  df = ROOT.RDataFrame(nmbEvents)
  RootUtilities.declareInCpp(intensityFcn = intensityFcn)  # use Python object in C++
  df.Define("point",    "double cosTheta, phiDeg, PhiDeg; PyVars::intensityFcn.GetRandom3(cosTheta, phiDeg, PhiDeg); std::vector<double> point = {cosTheta, phiDeg, PhiDeg}; return point;") \
    .Define("cosTheta", "point[0]") \
    .Define("theta",    "std::acos(cosTheta)") \
    .Define("phiDeg",   "point[1]") \
    .Define("phi",      "TMath::DegToRad() * phiDeg") \
    .Define("PhiDeg",   "point[2]") \
    .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genDataFromWaves()" << endl; } return true;') \
    .Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](["cosTheta", "theta", "phiDeg", "phi", "PhiDeg", "Phi"]))
    # snapshot is needed or else the `point` column would be regenerated for every triggered loop
    # noop filter before snapshot logs when event loop is running
    # !Note! for some reason, this is very slow
  return ROOT.RDataFrame(treeName, fileName)


def genAccepted2BodyPsPhotoProd(
  nmbEvents:         int,                   # number of events to generate
  efficiencyFormula: Optional[str] = None,  # detection efficiency used for acceptance correction
) -> ROOT.RDataFrame:
  """Generates RDataFrame with two-body phase-space distribution weighted by given detection efficiency"""
  # construct and draw efficiency function
  efficiencyFcn = ROOT.TF3("efficiencyReco", efficiencyFormula if efficiencyFormula else "1", -1, +1, -180, +180, -180, +180)
  PlottingUtilities.drawTF3(efficiencyFcn, **TH3_PLOT_KWARGS, pdfFileName = "./hEfficiencyReco.pdf", nmbPoints = 100, maxVal = 1.0)

  # generate isotropic distributions in cos theta, phi, and Phi and weight with efficiency function
  treeName = "data"
  fileName = f"{efficiencyFcn.GetName()}.photoProd.root"
  #TODO switch that allows loading from file
  df = ROOT.RDataFrame(nmbEvents)
  RootUtilities.declareInCpp(efficiencyFcn = efficiencyFcn)
  df.Define("point",    "double cosTheta, phiDeg, PhiDeg; PyVars::efficiencyFcn.GetRandom3(cosTheta, phiDeg, PhiDeg); std::vector<double> point = {cosTheta, phiDeg, PhiDeg}; return point;") \
    .Define("cosTheta", "point[0]") \
    .Define("theta",    "std::acos(cosTheta)") \
    .Define("phiDeg",   "point[1]") \
    .Define("phi",      "TMath::DegToRad() * phiDeg") \
    .Define("PhiDeg",   "point[2]") \
    .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genData2BodyPSPhotoProd()" << endl; } return true;') \
    .Snapshot(treeName, fileName, ROOT.std.vector[ROOT.std.string](["theta", "phi", "Phi"]))
    # snapshot is needed or else the `point` column would be regenerated for every triggered loop
    # noop filter before snapshot logs when event loop is running
  return ROOT.RDataFrame(treeName, fileName)


if __name__ == "__main__":
  OpenMp.setNmbOpenMpThreads(5)
  ROOT.gROOT.SetBatch(True)
  ROOT.gRandom.SetSeed(1234567890)
  # ROOT.EnableImplicitMT(10)
  PlottingUtilities.setupPlotStyle()
  ROOT.gBenchmark.Start("Total execution time")

  # get data
  nmbEvents = 1000
  nmbMcEvents = 10000000
  polarization = 1.0
  # formulas for detection efficiency
  # x = cos(theta) in [-1, +1], y = phi in [-180, +180] deg, z = Phi in [-180, +180] deg
  # efficiencyFormulaGen = "1"  # acc_perfect
  # efficiencyFormulaGen = "(1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / 1.5**3"  # acc_1; even in all variables
  # efficiencyFormulaGen = "(0.75 + 0.25 * x) * (0.75 + 0.25 * (y / 180)) * (0.75 + 0.25 * (z / 180))"  # acc_2; odd in all variables
  #TODO fix '-' in y term
  efficiencyFormulaGen = "(0.6 + 0.4 * x) * (0.6 - 0.4 * (y / 180)) * (0.6 + 0.4 * (z / 180))"  # acc_3; odd in all variables
  # detune efficiency used to correct acceptance w.r.t. the one used to generate the data
  efficiencyFormulaDetune = ""
  # efficiencyFormulaDetune = "(0.35 + 0.15 * x) * (0.35 - 0.15 * (y / 180)) * (0.35 + 0.15 * (z / 180))"  # detune_odd; detune by odd terms
  # efficiencyFormulaDetune = "0.1 * (1.5 - y * y / (180 * 180)) / 1.5"  # detune_even; detune by even terms in phi only
  # efficiencyFormulaDetune = "0.1 * (1.5 - x * x) * (1.5 - z * z / (180 * 180)) / (1.5**2)"  # detune_even; detune by even terms in cos(theta) and Phi
  # efficiencyFormulaDetune = "0.1 * (1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - z * z / (180 * 180)) / (1.5**3)"  # detune_even; detune by even terms in all variables
  if efficiencyFormulaDetune:
    efficiencyFcnDetune = ROOT.TF3("efficiencyDetune", efficiencyFormulaDetune, -1, +1, -180, +180, -180, +180)
    PlottingUtilities.drawTF3(efficiencyFcnDetune, **TH3_PLOT_KWARGS, pdfFileName = "./hEfficiencyDetune.pdf", nmbPoints = 100, maxVal = 1.0)
    efficiencyFormulaReco = f"{efficiencyFormulaGen} + {efficiencyFormulaDetune}"
  else:
    efficiencyFormulaReco = efficiencyFormulaGen

  # input from partial-wave amplitudes
  ROOT.gBenchmark.Start("Time to generate MC data from partial waves")
  amplitudeSet = MomentCalculator.AmplitudeSet(PROD_AMPS2)
  # HTrue: MomentCalculator.MomentResult = calcAllMomentsFromWaves(PROD_AMPS, maxL = MAX_L)
  HTrue: MomentCalculator.MomentResult = amplitudeSet.allMoments(maxL = MAX_L)
  print("True moment values:")
  for L in range(MAX_L + 1):
    for M in range(L + 1):
      moments = []
      for momentIndex in range(2 if M == 0 else 3):
        qnIndex = MomentCalculator.QnMomentIndex(momentIndex, L, M)
        moments.append(HTrue[qnIndex].val)
      print(f"(H_0({L} {M}), H_1({L} {M})" + ("" if M == 0 else f", H_2({L} {M}))") + f" = {tuple(moments)}")
  # dataPwaModel = genDataFromWaves(nmbEvents, polarization, PROD_AMPS, efficiencyFormulaGen)
  dataPwaModel = genDataFromWaves(nmbEvents, polarization, amplitudeSet, efficiencyFormulaGen)
  ROOT.gBenchmark.Stop("Time to generate MC data from partial waves")

  # plot data
  canv = ROOT.TCanvas()
  nmbBins = 25
  hist = dataPwaModel.Histo3D(
    ROOT.RDF.TH3DModel("hData", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
    "cosTheta", "phiDeg", "PhiDeg")
  hist.SetMinimum(0)
  hist.GetXaxis().SetTitleOffset(1.5)
  hist.GetYaxis().SetTitleOffset(2)
  hist.GetZaxis().SetTitleOffset(1.5)
  hist.Draw("BOX2")
  canv.SaveAs(f"{hist.GetName()}.pdf")

  # generate accepted phase-space data
  ROOT.gBenchmark.Start("Time to generate phase-space MC data")
  dataAcceptedPs = genAccepted2BodyPsPhotoProd(nmbMcEvents, efficiencyFormulaReco)
  ROOT.gBenchmark.Stop("Time to generate phase-space MC data")
  # calculate integral matrix
  nmbOpenMpThreads = ROOT.getNmbOpenMpThreads()
  momentIndices    = MomentCalculator.MomentIndices(MAX_L)
  dataSet          = MomentCalculator.DataSet(polarization, dataPwaModel, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbMcEvents)
  ROOT.gBenchmark.Start(f"Time to calculate integral matrix using {nmbOpenMpThreads} OpenMP threads")
  integralMatrix = MomentCalculator.AcceptanceIntegralMatrix(momentIndices, dataSet)
  integralMatrix.calculate()
  # integralMatrix.loadOrCalculate()
  integralMatrix.save()
  ROOT.gBenchmark.Stop(f"Time to calculate integral matrix using {nmbOpenMpThreads} OpenMP threads")

  # calculate and print moments of accepted phase-space data
  ROOT.gBenchmark.Start(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")
  # momentsPs = MomentCalculator.MomentCalculator(momentIndices,
  #   MomentCalculator.DataSet(polarization, dataAcceptedPs, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbMcEvents), integralMatrix)
  # moments of acceptance function
  momentsPs = MomentCalculator.MomentCalculator(momentIndices,
    MomentCalculator.DataSet(polarization, dataAcceptedPs, phaseSpaceData = dataAcceptedPs, nmbGenEvents = nmbMcEvents), integralMatrix = None)
  momentsPs.calculate()
  assert momentsPs.HPhys is not None, "momentsPs.HPhys is None"
  print("Measured moments of accepted phase-space data")
  momentsPs.HMeas.print()
  print("Integral matrix")
  print(f"Acceptance integral matrix = \n{integralMatrix}")
  eigenVals, eigenVecs = np.linalg.eig(integralMatrix._IFlatIndex)
  print(f"I_acc eigenvalues = {eigenVals}")
  print(f"Physical moments of accepted phase-space data\n{momentsPs.HPhys}")
  HTruePs = MomentCalculator.MomentResult(momentIndices, label = "true")    # set all true moment values to 0
  HTruePs._valsFlatIndex[momentIndices.indexMap.flatIndex_for[MomentCalculator.QnMomentIndex(momentIndex = 0, L = 0, M = 0)]] = 1  # set true H_0(0, 0) to 1
  PlottingUtilities.plotMomentsInBin(HData = momentsPs.HPhys, HTrue = HTruePs, pdfFileNamePrefix = "hPs_")
  ROOT.gBenchmark.Stop(f"Time to calculate moments of phase-space MC data using {nmbOpenMpThreads} OpenMP threads")

  # calculate moments
  ROOT.gBenchmark.Start(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")
  moments = MomentCalculator.MomentCalculator(momentIndices, dataSet, integralMatrix)
  moments.calculate()
  print("Measured moments of data generated according to PWA model")
  moments.HMeas.print()
  print("Integral matrix")
  print(f"Acceptance integral matrix = \n{integralMatrix}")
  eigenVals, eigenVecs = np.linalg.eig(integralMatrix._IFlatIndex)
  print(f"I_acc eigenvalues = {eigenVals}")
  print(f"Physical moments of data generated according to PWA model\n{moments.HPhys}")
  assert moments.HPhys is not None, "moments.HPhys is None"
  PlottingUtilities.plotMomentsInBin(HData = moments.HPhys, HTrue = HTrue, pdfFileNamePrefix = "h_")
  ROOT.gBenchmark.Stop(f"Time to calculate moments using {nmbOpenMpThreads} OpenMP threads")

  ROOT.gBenchmark.Stop("Total execution time")
  _ = ctypes.c_float(0.0)  # dummy argument required by ROOT; sigh
  ROOT.gBenchmark.Summary(_, _)
  print("!Note! the 'TOTAL' time above is wrong; ignore")

  OpenMp.restoreNmbOpenMpThreads()
