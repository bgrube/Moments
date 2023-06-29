#!/usr/bin/env python3


import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Any, Collection, Dict, List, Optional, Tuple

import py3nj
from uncertainties import UFloat, ufloat

import ROOT


# see https://root-forum.cern.ch/t/tf1-eval-as-a-function-in-rdataframe/50699/3
def declareInCpp(**kwargs: Any) -> None:
  '''Creates C++ variables (names defined by keys) for PyROOT objects (given by values) in PyVars:: namespace'''
  for key, value in kwargs.items():
    ROOT.gInterpreter.Declare(  # type: ignore
f'''
namespace PyVars
{{
  auto& {key} = *reinterpret_cast<{type(value).__cpp_name__}*>({ROOT.addressof(value)});
}}
''')


# see e.g. LHCb, PRD 92 (2015) 112009
def generateDataLegPolLC(
  nmbEvents:  int,
  maxDegree:  int,
  parameters: Collection[float],
) -> Any:
  '''Generates data according to linear combination of Legendre polynomials'''
  assert len(parameters) >= maxDegree + 1, f"Need {maxDegree + 1} parameters; only {len(parameters)} were given: {parameters}"
  # linear combination of legendre polynomials up to given degree
  terms = tuple(f"[{degree}] * ROOT::Math::legendre({degree}, x)" for degree in range(maxDegree + 1))
  print("linear combination =", " + ".join(terms))
  legendrePolLC = ROOT.TF1("legendrePolLC", " + ".join(terms), -1, +1)  # type: ignore
  legendrePolLC.SetNpx(1000)  # used in numeric integration performed by GetRandom()
  for index, parameter in enumerate(parameters):
    legendrePolLC.SetParameter(index, parameter)
  legendrePolLC.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()  # type: ignore
  legendrePolLC.Draw()
  canv.SaveAs(f"{legendrePolLC.GetName()}.pdf")

  # generate random data that follow linear combination of legendre polynomials
  treeName = "data"
  fileName = f"{legendrePolLC.GetName()}.root"
  df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  declareInCpp(legendrePolLC = legendrePolLC)
  df.Define("CosTheta", "PyVars::legendrePolLC.GetRandom()") \
    .Define("Theta",    "std::acos(CosTheta)") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `CosTheta` column would be regenerated for every triggered loop
                                   # noop filter before snapshot logs when event loop is running
  return ROOT.RDataFrame(treeName, fileName)  # type: ignore


def calculateLegMoments(
  dataFrame: Any,
  maxDegree: int,
) -> Dict[Tuple[int, ...], UFloat]:
  '''Calculates moments of Legendre polynomials'''
  nmbEvents = dataFrame.Count().GetValue()
  moments: Dict[Tuple[int], UFloat] = {}
  for degree in range(maxDegree + 5):
    # unnormalized moments
    dfMoment = dataFrame.Define("legendrePol", f"ROOT::Math::legendre({degree}, CosTheta)")
    momentVal = dfMoment.Sum("legendrePol").GetValue()
    momentErr = math.sqrt(nmbEvents) * dfMoment.StdDev("legendrePol").GetValue()  # iid events: Var[sum_i^N f(x_i)] = sum_i^N Var[f] = N * Var[f]; see https://www.wikiwand.com/en/Monte_Carlo_integration
    # normalize moments with respect to H(0)
    legendrePolIntegral = 1 / (2 * degree + 1)  # = 1/2 * int_-1^+1; factor 1/2 takes into account integral for H(0)
    norm = 1 / (nmbEvents * legendrePolIntegral)
    moment = norm * ufloat(momentVal, momentErr)  # type: ignore
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
  nmbEvents:  int,
  maxL:       int,  # maximum spin of decaying object
  parameters: Collection[float],  # make sure that resulting linear combination is positive definite
) -> Any:
  '''Generates data according to linear combination of spherical harmonics'''
  nmbTerms = 6 * maxL  # Eq. (17)
  assert len(parameters) >= nmbTerms, f"Need {nmbTerms} parameters; only {len(parameters)} were given: {parameters}"
  # linear combination of spherical harmonics up to given maximum orbital angular momentum
  # using Eq. (12) in Eq. (6): I = sum_L (2 L + 1 ) / (4pi)  H(L 0) (D_00^L)^* + sum_{M = 1}^L H(L M) 2 Re[D_M0^L]
  # and (D_00^L)^* = D_00^L = sqrt(4 pi / (2 L + 1)) (Y_L^0)^* = Y_L^0
  # and Re[D_M0^L] = d_M0^L cos(M phi) = Re[sqrt(4 pi / (2 L + 1)) (Y_L^M)^*] = Re[sqrt(4 pi / (2 L + 1)) Y_L^M]
  # i.e. Eq. (13) becomes: I = sum_L sqrt((2 L + 1 ) / (4pi)) sum_{M = 0}^L tau(M) H(L M) Re[Y_L^M]
  terms = []
  termIndex = 0
  for L in range(2 * maxL + 1):
    termsM = []
    for M in range(min(L, 2) + 1):
      termsM.append(f"[{termIndex}] * {1 if M == 0 else 2} * ROOT::Math::sph_legendre({L}, {M}, std::acos(x)) * std::cos({M} * TMath::DegToRad() * y)")  # ROOT defines this as function of theta (not cos(theta)!); sigh
      termIndex += 1
    terms.append(f"std::sqrt((2 * {L} + 1 ) / (4 * TMath::Pi())) * ({' + '.join(termsM)})")
  print("linear combination =", " + ".join(terms))
  sphericalHarmLC = ROOT.TF2("sphericalHarmlLC", " + ".join(terms), -1, +1, -180, +180)  # type: ignore
  sphericalHarmLC.SetNpx(500)  # used in numeric integration performed by GetRandom()
  sphericalHarmLC.SetNpy(500)
  sphericalHarmLC.SetContour(100)
  for index, parameter in enumerate(parameters):
    sphericalHarmLC.SetParameter(index, parameter)
  sphericalHarmLC.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()  # type: ignore
  sphericalHarmLC.Draw("COLZ")
  canv.SaveAs(f"{sphericalHarmLC.GetName()}.pdf")

  # generate random data that follow linear combination of of spherical harmonics
  treeName = "data"
  fileName = f"{sphericalHarmLC.GetName()}.root"
  df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  declareInCpp(sphericalHarmLC = sphericalHarmLC)
  df.Define("point",    "double CosTheta, PhiDeg; PyVars::sphericalHarmLC.GetRandom2(CosTheta, PhiDeg); std::vector<double> point = {CosTheta, PhiDeg}; return point;") \
    .Define("CosTheta", "point[0]") \
    .Define("Theta",    "std::acos(CosTheta)") \
    .Define("PhiDeg",   "point[1]") \
    .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                   # noop filter before snapshot logs when event loop is running
  return ROOT.RDataFrame(treeName, fileName)  # type: ignore


# C++ implementation of RDataFrame custom action that calculates covariance between two columns
ROOT.gROOT.LoadMacro("./Covariance.C++")  # type: ignore

def calculateSphHarmMoments(
  dataFrame:      Any,
  maxL:           int,  # maximum spin of decaying object
  integralMatrix: Optional[Dict[Tuple[int, ...], complex]] = None,  # acceptance integral matrix
) -> Tuple[List[Tuple[Tuple[int, int], complex]], Dict[Tuple[int, ...], Tuple[float, ...]]]:  # moment values and covariances
  '''Calculates moments of spherical harmonics'''
  # define moments
  dfMoment = dataFrame
  for L in range(2 * maxL + 2):
    for M in range(L + 1):
      # unnormalized moments
      dfMoment = dfMoment.Define(f"Re_f_{L}_{M}", f"std::sqrt((4 * TMath::Pi()) / (2 * {L} + 1)) * ReYlm({L}, {M}, Theta, Phi)")
      dfMoment = dfMoment.Define(f"Im_f_{L}_{M}", f"std::sqrt((4 * TMath::Pi()) / (2 * {L} + 1)) * ImYlm({L}, {M}, Theta, Phi)")
  # calculate moments
  nmbEvents = dataFrame.Count().GetValue()
  nmbMoments = (2 * maxL + 2) * (2 * maxL + 3) // 2
  H_meas = np.zeros((nmbMoments), dtype = np.complex128)
  V_meas_ReRe = np.zeros((nmbMoments, nmbMoments), dtype = np.float64)
  V_meas_ImIm = np.zeros((nmbMoments, nmbMoments), dtype = np.float64)
  V_meas_ReIm = np.zeros((nmbMoments, nmbMoments), dtype = np.float64)
  Re_f = np.zeros((nmbMoments, nmbEvents), dtype = np.float64)
  Im_f = np.zeros((nmbMoments, nmbEvents), dtype = np.float64)
  for L in range(2 * maxL + 2):
    for M in range(L + 1):
      iMoment = L * (L + 1) // 2 + M
      # calculate value
      momentValRe = dfMoment.Sum(f"Re_f_{L}_{M}").GetValue() / nmbEvents
      momentValIm = dfMoment.Sum(f"Im_f_{L}_{M}").GetValue() / nmbEvents
      momentVal = momentValRe - 1j * momentValIm  # moment is defined by (Y_L^M)^*
      print(f"H_meas(L = {L}, M = {M}) = {momentVal}")
      H_meas[iMoment] = momentVal
      # get values of spherical harmonics as Numpy arrays
      Re_f[iMoment, :] = dfMoment.AsNumpy(columns = [f"Re_f_{L}_{M}"])[f"Re_f_{L}_{M}"]
      Im_f[iMoment, :] = dfMoment.AsNumpy(columns = [f"Im_f_{L}_{M}"])[f"Im_f_{L}_{M}"]
      # calculate value covariances
      #TODO optimize by exploiting symmetry
      for L_p in range(2 * maxL + 2):
        for M_p in range(L_p + 1):
          iMoment_p = L_p * (L_p + 1) // 2 + M_p
          V_meas_ReRe[iMoment, iMoment_p] = dfMoment.Book(ROOT.std.move(ROOT.Covariance["double"]()), [f"Re_f_{L}_{M}", f"Re_f_{L_p}_{M_p}"]).GetValue()  # type: ignore
          V_meas_ImIm[iMoment, iMoment_p] = dfMoment.Book(ROOT.std.move(ROOT.Covariance["double"]()), [f"Im_f_{L}_{M}", f"Im_f_{L_p}_{M_p}"]).GetValue()  # type: ignore
          V_meas_ReIm[iMoment, iMoment_p] = dfMoment.Book(ROOT.std.move(ROOT.Covariance["double"]()), [f"Re_f_{L}_{M}", f"Im_f_{L_p}_{M_p}"]).GetValue()  # type: ignore
  V_meas_ReRe_np = np.cov(Re_f)
  print(f"V_meas_ReRe =\n{V_meas_ReRe}")
  print(f"vs.\n{V_meas_ReRe_np}")
  print(f"ratio\n{np.real_if_close(V_meas_ReRe / V_meas_ReRe_np)}")
  V_meas_ImIm_np = np.cov(Im_f)
  print(f"V_meas_ImIm =\n{V_meas_ImIm}")
  print(f"vs.\n{V_meas_ImIm_np}")
  print(f"ratio\n{np.real_if_close(V_meas_ImIm / V_meas_ImIm_np)}")
  V_meas_ReIm_np = np.cov(Re_f, Im_f)[:nmbMoments, nmbMoments:]  # !Note! numpy.cov(x, y) returns the covariance matrix for the stacked vector (x^T, y^T)^T
  print(f"V_meas_ReIm =\n{V_meas_ReIm}")
  print(f"vs.\n{V_meas_ReIm_np}")
  print(f"ratio\n{np.real_if_close(V_meas_ReIm / V_meas_ReIm_np)}")
  # raise ValueError
  V_meas_Hermit = V_meas_ReRe + V_meas_ImIm + 1j * (V_meas_ReIm.T - V_meas_ReIm)  # Hermitian covariance matrix
  V_meas_pseudo = V_meas_ReRe - V_meas_ImIm + 1j * (V_meas_ReIm.T + V_meas_ReIm)  # pseudo-covariance matrix
  V_meas_aug = np.block([
    [V_meas_Hermit,               V_meas_pseudo],
    [np.conjugate(V_meas_pseudo), np.conjugate(V_meas_Hermit)],
  ])  # augmented covariance matrix
  f = Re_f + 1j * Im_f
  V_meas_aug_np = np.cov(f, np.conjugate(f))
  print(f"V_meas_aug =\n{V_meas_aug}")
  print(f"vs.\n{V_meas_aug_np}")
  print(f"ratio\n{np.real_if_close(V_meas_aug / V_meas_aug_np)}")
  # correct for acceptance
  H_true = np.zeros((nmbMoments), dtype = np.complex128)
  V_true_aug = np.zeros((2 * nmbMoments, 2 * nmbMoments), dtype = np.complex128)
  if integralMatrix is None:
    H_true     = H_meas
    V_true_aug = V_meas_aug
  else:
    I = np.zeros((nmbMoments, nmbMoments), dtype = np.complex128)
    for L in range(2 * maxL + 2):
      for M in range(L + 1):
        for Lp in range(2 * maxL + 2):
          for Mp in range(Lp + 1):
            I[L * (L + 1) // 2 + M][Lp * (Lp + 1) // 2 + Mp] = integralMatrix[(L, M, Lp, Mp)]
    eigenVals, eigenVecs = np.linalg.eig(I)
    print(f"I eigenvalues = {eigenVals}")
    # print(f"I eigenvectors = {eigenVecs}")
    # print(f"I determinant = {np.linalg.det(I)}")
    print(f"I = \n{np.array2string(I, precision = 3, suppress_small = True, max_line_width = 150)}")
    Iinv = np.linalg.inv(I)
    # eigenVals, eigenVecs = np.linalg.eig(Iinv)
    # print(f"I^-1 eigenvalues = {eigenVals}")
    print(f"I^-1 = \n{np.array2string(Iinv, precision = 3, suppress_small = True, max_line_width = 150)}")
    plt.figure().colorbar(plt.matshow(Iinv.real))
    plt.savefig("Iinv_real.pdf")
    plt.figure().colorbar(plt.matshow(Iinv.imag))
    plt.savefig("Iinv_imag.pdf")
    plt.figure().colorbar(plt.matshow(np.absolute(Iinv)))
    plt.savefig("Iinv_abs.pdf")
    plt.figure().colorbar(plt.matshow(np.angle(Iinv)))
    plt.savefig("Iinv_arg.pdf")
    H_true = Iinv @ H_meas
    # linear uncertainty propagation
    J = Iinv  # Jacobian of acceptance correction
    J_conj = np.zeros((nmbMoments, nmbMoments), dtype = np.complex128)  # conjugate Jacobian
    J_aug = np.block([
      [J,                    J_conj],
      [np.conjugate(J_conj), np.conjugate(J)],
    ])  # augmented Jacobian
    V_true_aug = J_aug @ (V_meas_aug @ np.asmatrix(J_aug).H)  #!Note! @ is left-associative
  V_true_Hermit = V_true_aug[:nmbMoments, :nmbMoments]  # Hermitian covariance matrix
  V_true_pseudo = V_true_aug[:nmbMoments, nmbMoments:]  # pseudo-covariance matrix
  # conariances of real and imaginary parts
  V_true_ReRe = (np.real(V_true_Hermit) + np.real(V_true_pseudo)) / 2
  V_true_ImIm = (np.real(V_true_Hermit) - np.real(V_true_pseudo)) / 2
  V_true_ReIm = (np.imag(V_true_pseudo) - np.imag(V_true_Hermit)) / 2
  # reformat output
  momentsTrue:    List[Tuple[Tuple[int, int], complex]]    = []
  momentsTrueCov: Dict[Tuple[int, ...], Tuple[float, ...]] = {}  # cov[(L, M, L', M')] = (ReRe, ImIm, ReIm)
  for L in range(2 * maxL + 2):
    for M in range(L + 1):
      iMoment = L * (L + 1) // 2 + M
      print(f"H_true(L = {L}, M = {M}) = {H_true[iMoment]}")
      momentsTrue.append(((L, M), H_true[iMoment]))
      for L_p in range(2 * maxL + 2):
        for M_p in range(L_p + 1):
          iMoment_p = L_p * (L_p + 1) // 2 + M_p
          momentsTrueCov[(L, M, L_p, M_p)] = (V_true_ReRe[iMoment, iMoment_p], V_true_ImIm[iMoment, iMoment_p], V_true_ReIm[iMoment, iMoment_p])
  # print(momentsTrue)
  #TODO encapsulate moment values and covariances in object that takes care of the index mapping
  return momentsTrue, momentsTrueCov


# C++ implementation of (complex conjugated) Wigner D function
# also provides complexT typedef for std::complex<double>
ROOT.gROOT.LoadMacro("./wignerD.C++")  # type: ignore

WAVE_SET: Dict[int, List[Tuple[int, int]]] = {
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
  prodAmps:          Dict[int, Tuple[complex, ...]],
  acceptanceFormula: Optional[str] = None,
) -> Any:
  '''Generates data according to partial-wave decomposition for fixed set of 7 lowest waves up to \ell = 2 and |m| = 1'''
  # generate data according to Eq. (28) with rank = 1 and using wave set in Eqs. (41) and (42)
  assert len(prodAmps) == len(WAVE_SET), f"Need {len(WAVE_SET)} parameters; only {len(prodAmps)} were given: {prodAmps}"
  incoherentTerms = []
  for refl in (-1, +1):
    coherentTerms = []
    for waveIndex, wave in enumerate(WAVE_SET[refl]):
      ell:    int = wave[0]
      m:      int = wave[1]
      parity: int = (-1)**ell
      # see Eqs. (26) and (27) for rank = 1
      V = f"complexT({prodAmps[refl][waveIndex].real}, {prodAmps[refl][waveIndex].imag})"  # complexT is a typedef for std::complex<double> in wignerD.C
      A = f"std::sqrt((2 * {ell} + 1) / (4 * TMath::Pi())) * wignerDReflConj({2 * ell}, {2 * m}, 0, {parity}, {refl}, TMath::DegToRad() * y, std::acos(x))"
      coherentTerms.append(f"{V} * {A}")
    incoherentTerms.append(f"std::norm({' + '.join(coherentTerms)})")
  # see Eq. (28) for rank = 1
  intensityFormula = f"({' + '.join(incoherentTerms)})" + ("" if acceptanceFormula is None else f" * ({acceptanceFormula})")
  print(f"intensity = {intensityFormula}")
  intensityFcn = ROOT.TF2("intensity", intensityFormula, -1, +1, -180, +180)  # type: ignore
  intensityFcn.SetTitle(";cos#theta;#phi [deg]")
  intensityFcn.SetNpx(500)  # used in numeric integration performed by GetRandom()
  intensityFcn.SetNpy(500)
  intensityFcn.SetContour(100)
  intensityFcn.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()  # type: ignore
  intensityFcn.Draw("COLZ")
  canv.SaveAs(f"{intensityFcn.GetName()}.pdf")

  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = f"{intensityFcn.GetName()}.root"
  df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  declareInCpp(intensityFcn = intensityFcn)
  df.Define("point",    "double CosTheta, PhiDeg; PyVars::intensityFcn.GetRandom2(CosTheta, PhiDeg); std::vector<double> point = {CosTheta, PhiDeg}; return point;") \
    .Define("CosTheta", "point[0]") \
    .Define("Theta",    "std::acos(CosTheta)") \
    .Define("PhiDeg",   "point[1]") \
    .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                   # noop filter before snapshot logs when event loop is running
  return ROOT.RDataFrame(treeName, fileName)  # type: ignore


def theta(m: int) -> float:
  '''Calculates normalization factor in reflectivity basis'''
  # see Eq. (19)
  if m > 0:
    return 1 / math.sqrt(2)
  elif m == 0:
    return 1 / 2
  else:
    return 0


def calculateInputPwdMoment(
  prodAmps: Dict[int, Tuple[complex, ...]],
  L: int,
  M: int,
) -> complex:
  '''Calculates value of moment with L and M for given production amplitudes'''
  # Eq. (29) for rank = 1
  sum = 0 + 0j
  for refl in (-1, +1):
    for waveIndex_1, wave_1 in enumerate(WAVE_SET[refl]):
      ell_1: int = wave_1[0]
      m_1:   int = wave_1[1]
      for waveIndex_2, wave_2 in enumerate(WAVE_SET[refl]):
        ell_2: int = wave_2[0]
        m_2:   int = wave_2[1]
        b = theta(m_2) * theta(m_1) * (
                               py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1,  2 * m_2,  2 * M,  2 * m_1, ignore_invalid = True)  # (ell_2  m_2  L  M | ell_1  m_1)
          + (-1)**M *          py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1,  2 * m_2, -2 * M,  2 * m_1, ignore_invalid = True)  # (ell_2  m_2  L -M | ell_1  m_1)
          - refl * (-1)**m_2 * py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1, -2 * m_2,  2 * M,  2 * m_1, ignore_invalid = True)  # (ell_2 -m_2  L  M | ell_1  m_1)
          - refl * (-1)**m_1 * py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1,  2 * m_2,  2 * M, -2 * m_1, ignore_invalid = True)  # (ell_2  m_2  L  M | ell_1 -m_1)
        )
        sum += math.sqrt((2 * ell_2 + 1) / (2 * ell_1 + 1)) * prodAmps[refl][waveIndex_1] * prodAmps[refl][waveIndex_2].conjugate() * b \
               * py3nj.clebsch_gordan(2 * ell_2, 2 * L, 2 * ell_1, 0, 0, 0, ignore_invalid = True)  # (ell_2 0  L 0 | ell_1 0)
  return sum


def calculateInputPwdMoments(
  prodAmps: Dict[int, Tuple[complex, ...]],
  maxL:     int,  # maximum spin of decaying object
) -> List[float]:
  '''Calculates moments for given production amplitudes'''
  moments: List[float] = []
  for L in range(2 * maxL + 2):
    for M in range(L + 1):
      moment: complex = calculateInputPwdMoment(prodAmps, L, M)
      if (abs(moment.imag) > 1e-15):
        print(f"Warning: non vanishing imaginary part for moment H({L} {M}) = {moment}")
      moments.append(moment.real)
  # normalize to first moment
  moments = [moment / moments[0] for moment in moments]
  index = 0
  for L in range(2 * maxL + 2):
    for M in range(L + 1):
      print(f"H_input(L = {L}, M = {M}) = {moments[index]}")
      index += 1
  print(moments)
  return moments


def calculateWignerDMoment(
  dataFrame: Any,
  L:         int,
  M:         int,
) -> Tuple[UFloat, UFloat]:  # real and imag part with uncertainty
  '''Calculates unnormalized moment of Wigner-D function D^L_{M 0}'''
  # unnormalized moment
  dfMoment = dataFrame.Define("WignerD",  f"wignerD({2 * L}, {2 * M}, 0, Phi, theta)") \
                      .Define("WignerDRe", "real(WignerD)") \
                      .Define("WignerDIm", "imag(WignerD)")
  momentVal   = dfMoment.Sum[ROOT.std.complex["double"]]("WignerD").GetValue()  # type: ignore
  # iid events: Var[sum_i^N f(x_i)] = sum_i^N Var[f] = N * Var[f]; see https://www.wikiwand.com/en/Monte_Carlo_integration
  momentErrRe = math.sqrt(nmbEvents) * dfMoment.StdDev("WignerDRe").GetValue()
  momentErrIm = math.sqrt(nmbEvents) * dfMoment.StdDev("WignerDIm").GetValue()
  return ufloat(momentVal.real, momentErrRe), ufloat(momentVal.imag, momentErrIm)


def calculateWignerDMoments(
  dataFrame: Any,
  maxL:      int,  # maximum spin of decaying object
) -> None:
  '''Calculates moments of Wigner-D function D^L_{M 0}'''
  nmbEvents = dataFrame.Count().GetValue()
  # moments: List[Tuple[Tuple[int, int], UFloat]] = []
  for L in range(2 * maxL + 2):
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
  acceptanceFormula: Optional[str] = None,
) -> Any:
  '''Generates RDataFrame with two-body phase-space distribution weighted by given acceptance'''
  # construct acceptance function
  acceptanceFcn = ROOT.TF2("acceptance", "1" if acceptanceFormula is None else acceptanceFormula, -1, +1, -180, +180)  # type: ignore
  acceptanceFcn.SetTitle(";cos#theta;#phi [deg]")
  acceptanceFcn.SetNpx(500)  # used in numeric integration performed by GetRandom()
  acceptanceFcn.SetNpy(500)
  acceptanceFcn.SetContour(100)
  acceptanceFcn.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()  # type: ignore
  acceptanceFcn.Draw("COLZ")
  canv.SaveAs(f"{acceptanceFcn.GetName()}.pdf")

  # generate isotropic distributions in cos theta and phi and weight with acceptance function
  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = "phaseSpace.root"
  df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  declareInCpp(acceptanceFcn = acceptanceFcn)
  df.Define("point", "double CosTheta, PhiDeg; PyVars::acceptanceFcn.GetRandom2(CosTheta, PhiDeg); std::vector<double> point = {CosTheta, PhiDeg}; return point;") \
    .Define("CosTheta", "point[0]") \
    .Define("Theta",    "std::acos(CosTheta)") \
    .Define("PhiDeg",   "point[1]") \
    .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                   # noop filter before snapshot logs when event loop is running
  return ROOT.RDataFrame(treeName, fileName)  # type: ignore


def calcIntegralMatrix(
  phaseSpaceDataFrame: Any,
  maxL:                int,  # maximum orbital angular momentum
  nmbEvents:           int,  # number of events in RDataFrame
) -> Dict[Tuple[int, ...], complex]:
  '''Calculates integral matrix of spherical harmonics from provided phase-space data'''
  # define spherical harmonics
  for L in range(2 * maxL + 2):
    for M in range(L + 1):
      phaseSpaceDataFrame = phaseSpaceDataFrame.Define(   f"Y_{L}_{M}",   f"Ylm({L}, {M}, Theta, Phi)")
      phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"Re_Y_{L}_{M}", f"ReYlm({L}, {M}, Theta, Phi)")
      phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"Im_Y_{L}_{M}", f"ImYlm({L}, {M}, Theta, Phi)")
  # define integral matrix
  for L in range(2 * maxL + 2):
    for M in range(L + 1):
      for Lp in range(2 * maxL + 2):
        for Mp in range(Lp + 1):
          # phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"I_{L}_{M}_{Lp}_{Mp}", f"(4 * TMath::Pi() / {nmbEvents}) * std::sqrt((double)(2 * {Lp} + 1) / (2 * {L} + 1)) * Y_{Lp}_{Mp} * std::conj(Y_{L}_{M})")
          phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"I_{L}_{M}_{Lp}_{Mp}", f"(4 * TMath::Pi() / {nmbEvents}) * std::sqrt((double)(2 * {Lp} + 1) / (2 * {L} + 1)) * (2 - ({Mp} == 0)) * Re_Y_{Lp}_{Mp} * std::conj(Y_{L}_{M})")
          # phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"I_{L}_{M}_{Lp}_{Mp}", f"(4 * TMath::Pi() / {nmbEvents}) * std::sqrt((double)(2 * {Lp} + 1) / (2 * {L} + 1)) * (2 - ({Mp} == 0)) * Im_Y_{Lp}_{Mp} * std::conj(Y_{L}_{M})")
  # calculate integral matrix
  I: Dict[Tuple[int, ...], complex] = {}
  for L in range(2 * maxL + 2):
    for M in range(L + 1):
      for Lp in range(2 * maxL + 2):
        for Mp in range(Lp + 1):
          I[(L, M, Lp, Mp)] = phaseSpaceDataFrame.Sum[ROOT.std.complex["double"]](f"I_{L}_{M}_{Lp}_{Mp}").GetValue()  # type: ignore
          # print(f"I_{L}_{M}_{Lp}_{Mp} = {I[(L, M, Lp, Mp)]}")
  # phaseSpaceDataFrame.Snapshot("foo", "foo.root", ["I_0_0_1_0", "I_1_0_0_0", "Re_Y_0_0", "Re_Y_1_0", "Y_0_0", "Y_1_0"])
  # raise ValueError
  return I


def setupPlotStyle():
  #TODO remove dependency from external file or add file to repo
  ROOT.gROOT.LoadMacro("~/rootlogon.C")  # type: ignore
  ROOT.gROOT.ForceStyle()  # type: ignore
  ROOT.gStyle.SetCanvasDefW(600)  # type: ignore
  ROOT.gStyle.SetCanvasDefH(600)  # type: ignore
  ROOT.gStyle.SetPalette(ROOT.kBird)  # type: ignore
  # ROOT.gStyle.SetPalette(ROOT.kViridis)  # type: ignore
  ROOT.gStyle.SetLegendFillColor(ROOT.kWhite)  # type: ignore
  ROOT.gStyle.SetLegendBorderSize(1)  # type: ignore
  # ROOT.gStyle.SetOptStat("ni")  # type: ignore  # show only name and integral
  # ROOT.gStyle.SetOptStat("i")  # type: ignore  # show only integral
  ROOT.gStyle.SetOptStat("")  # type: ignore
  ROOT.gStyle.SetStatFormat("8.8g")  # type: ignore
  ROOT.gStyle.SetTitleColor(1, "X")  # type: ignore  # fix that for some mysterious reason x-axis titles of 2D plots and graphs are white
  ROOT.gStyle.SetTitleOffset(1.35, "Y")  # type: ignore


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore
  ROOT.gRandom.SetSeed(123456789)  # type: ignore
  # ROOT.EnableImplicitMT(10)  # type: ignore
  setupPlotStyle()

  # get data
  nmbEvents = 1000
  nmbMcEvents = 10000
  acceptanceFormula = "1"
  # acceptanceFormula = "1 - x * x"
  # acceptanceFormula = "2 - x * x"

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
  # dataModel = generateDataSphHarmLC(nmbEvents, maxL = maxOrder, parameters = parameters)

  # normalize parameters 0th moment and pad with 0
  # inputMoments = [par / parameters[0] for par in parameters]
  # if len(inputMoments) < len(moments):
  #   inputMoments += [0] * (len(moments) - len(inputMoments))

  # partial-wave decomposition
  maxOrder = 2
  prodAmps: Dict[int, Tuple[complex, ...]] = {
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
  inputMoments: List[float] = calculateInputPwdMoments(prodAmps, maxL = maxOrder)
  dataModel = generateDataPwd(nmbEvents, prodAmps, acceptanceFormula)
  # print("!!!", dataModel.AsNumpy())

  # plot data
  canv = ROOT.TCanvas()  # type: ignore
  if "Phi" in dataModel.GetColumnNames():
    hist = dataModel.Histo2D(ROOT.RDF.TH2DModel("hData", ";cos#theta;#phi [deg]", 25, -1, +1, 25, -180, +180), "CosTheta", "PhiDeg")  # type: ignore
    hist.SetMinimum(0)
    hist.Draw("COLZ")
  else:
    hist = dataModel.Histo1D(ROOT.RDF.TH1DModel("hData", ";cos#theta", 100, -1, +1), "CosTheta")  # type: ignore
    hist.SetMinimum(0)
    hist.Draw()
  canv.SaveAs(f"{hist.GetName()}.pdf")

  # calculate moments
  dataAcceptedPS = generateData2BodyPS(nmbMcEvents, acceptanceFormula)
  integralMatrix = calcIntegralMatrix(dataAcceptedPS, maxL = maxOrder, nmbEvents = nmbMcEvents)
  print("Moments of accepted phase-space data")
  calculateSphHarmMoments(dataAcceptedPS, maxL = maxOrder, integralMatrix = integralMatrix)
  # calculateLegMoments(dataModel, maxDegree = maxOrder)
  print("Moments of data generated according to model")
  moments:    List[Tuple[Tuple[int, int], complex]]
  momentsCov: Dict[Tuple[int, ...], Tuple[float, ...]]
  moments, momentsCov = calculateSphHarmMoments(dataModel, maxL = maxOrder, integralMatrix = integralMatrix)
  # calculateWignerDMoments(dataModel, maxL = maxOrder)
  #TODO check whether using Eq. (6) instead of Eq. (13) yields moments that fulfill Eqs. (11) and (12)

  hStackRe = ROOT.THStack("hCompareRe", "")  # type: ignore
  nmbBins = len(moments)
  # create histogram with measured values
  histMeasRe = ROOT.TH1D("Measured Moments (Real Part)", ";;value", nmbBins, 0, nmbBins)  # type: ignore
  for index, moment in enumerate(moments):
    histMeasRe.SetBinContent(index + 1, moment[1].real)
    histMeasRe.SetBinError  (index + 1, momentsCov[(*moment[0], *moment[0])][0])  # diagonal element for ReRe
    histMeasRe.GetXaxis().SetBinLabel(index + 1, f"H({' '.join(tuple(str(n) for n in moment[0]))})")
  histMeasRe.SetLineColor(ROOT.kRed)  # type: ignore
  histMeasRe.SetMarkerColor(ROOT.kRed)  # type: ignore
  histMeasRe.SetMarkerStyle(ROOT.kFullCircle)  # type: ignore
  histMeasRe.SetMarkerSize(0.75)
  hStackRe.Add(histMeasRe, "PEX0")
  # create histogram with input values
  histInput = ROOT.TH1D("Input values", ";;value", nmbBins, 0, nmbBins)  # type: ignore
  for index, inputMoment in enumerate(inputMoments):
    histInput.SetBinContent(index + 1, inputMoment)
    histInput.SetBinError  (index + 1, 1e-16)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
  histInput.SetMarkerColor(ROOT.kBlue)  # type: ignore
  histInput.SetLineColor(ROOT.kBlue)  # type: ignore
  hStackRe.Add(histInput, "PE")
  canv = ROOT.TCanvas()  # type: ignore
  hStackRe.Draw("NOSTACK")
  hStackRe.GetHistogram().SetLineColor(ROOT.kBlack)  # type: ignore  # make automatic zero line dashed
  hStackRe.GetHistogram().SetLineStyle(ROOT.kDashed)  # type: ignore  # make automatic zero line dashed
  # hStack.GetHistogram().SetLineWidth(0)  # remove zero line; see https://root-forum.cern.ch/t/continuing-the-discussion-from-an-unwanted-horizontal-line-is-drawn-at-y-0/50877/1
  canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
  canv.SaveAs(f"{hStackRe.GetName()}.pdf")

  hStackIm = ROOT.THStack("hCompareIm", "")  # type: ignore
  nmbBins = len(moments)
  # create histogram with measured values
  histMeasIm = ROOT.TH1D("Measured Moments (Imaginary Part)", ";;value", nmbBins, 0, nmbBins)  # type: ignore
  for index, moment in enumerate(moments):
    histMeasIm.SetBinContent(index + 1, moment[1].imag)
    histMeasIm.SetBinError  (index + 1, momentsCov[(*moment[0], *moment[0])][1])  # diagonal element for ImIm
    histMeasIm.GetXaxis().SetBinLabel(index + 1, f"H({' '.join(tuple(str(n) for n in moment[0]))})")
  histMeasIm.SetLineColor(ROOT.kRed)  # type: ignore
  histMeasIm.SetMarkerColor(ROOT.kRed)  # type: ignore
  histMeasIm.SetMarkerStyle(ROOT.kFullCircle)  # type: ignore
  histMeasIm.SetMarkerSize(0.75)
  hStackIm.Add(histMeasIm, "PEX0")
  # create histogram with input values
  histInput = ROOT.TH1D("Input values", ";;value", nmbBins, 0, nmbBins)  # type: ignore
  for index, inputMoment in enumerate(inputMoments):
    histInput.SetBinContent(index + 1, 0)
    histInput.SetBinError  (index + 1, 1e-16)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
  histInput.SetMarkerColor(ROOT.kBlue)  # type: ignore
  histInput.SetLineColor(ROOT.kBlue)  # type: ignore
  hStackIm.Add(histInput, "PE")
  canv = ROOT.TCanvas()  # type: ignore
  hStackIm.Draw("NOSTACK")
  hStackIm.GetHistogram().SetLineColor(ROOT.kBlack)  # type: ignore  # make automatic zero line dashed
  hStackIm.GetHistogram().SetLineStyle(ROOT.kDashed)  # type: ignore  # make automatic zero line dashed
  
  canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
  canv.SaveAs(f"{hStackIm.GetName()}.pdf")

  # draw residuals
  #TODO calculate and print chi^2 / ndf
  residuals = tuple(moment[1].real - inputMoments[index] for index, moment in enumerate(moments))
  # residuals = tuple((moment[1].nominal_value - inputMoments[index]) / moment[1].std_dev if moment[1].std_dev > 0 else 0 for index, moment in enumerate(moments))
  histRes = ROOT.TH1D("hResiduals", ";;(measured - input) / #sigma_{measured}", nmbBins, 0, nmbBins)  # type: ignore
  chi2    = sum(tuple(residual**2 for residual in residuals))
  ndf     = len(residuals) - 1  # H(0, 0) has by definition a vanishing residual
  for index, residual in enumerate(residuals):
    histRes.SetBinContent(index + 1, residual)
    histRes.SetBinError  (index + 1, 1e-16)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
    histRes.GetXaxis().SetBinLabel(index + 1, histMeasRe.GetXaxis().GetBinLabel(index + 1))
  histRes.SetMarkerColor(ROOT.kBlue)  # type: ignore
  histRes.SetLineColor(ROOT.kBlue)  # type: ignore
  histRes.SetMinimum(-3)
  histRes.SetMaximum(+3)
  canv = ROOT.TCanvas()  # type: ignore
  histRes.Draw("PE")
  # draw zero line
  xAxis = histRes.GetXaxis()
  line = ROOT.TLine()  # type: ignore
  line.SetLineStyle(ROOT.kDashed)  # type: ignore
  line.DrawLine(xAxis.GetBinLowEdge(xAxis.GetFirst()), 0, xAxis.GetBinUpEdge(xAxis.GetLast()), 0)
  # shade 1 sigma region
  box = ROOT.TBox()  # type: ignore
  box.SetFillColorAlpha(ROOT.kBlack, 0.15)  # type: ignore
  box.DrawBox(xAxis.GetBinLowEdge(xAxis.GetFirst()), -1, xAxis.GetBinUpEdge(xAxis.GetLast()), +1)
  # draw chi^2 info
  label = ROOT.TLatex()  # type: ignore
  label.SetNDC()
  label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignBottom)  # type: ignore
  label.DrawLatex(0.12, 0.9075, f"#chi^{{2}}/n.d.f. = {chi2:.2f}/{ndf}, prob = {stats.distributions.chi2.sf(chi2, ndf) * 100:.0f}%")
  canv.SaveAs(f"{histRes.GetName()}.pdf")
