#!/usr/bin/env python3

# equation numbers refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=2

import functools
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Any, Collection, Dict, List, Optional, Tuple

import py3nj
from uncertainties import UFloat, ufloat

import ROOT


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


# # C++ implementation of RDataFrame custom action that calculates covariance between two columns
# ROOT.gROOT.LoadMacro("./Covariance.C++")  # type: ignore
# C++ implementation of (complex conjugated) Wigner D function and spherical harmonics
# also provides complexT typedef for std::complex<double>
ROOT.gROOT.LoadMacro("./wignerD.C++")  # type: ignore


# see https://root-forum.cern.ch/t/tf1-eval-as-a-function-in-rdataframe/50699/3
def declareInCpp(**kwargs: Any) -> None:
  '''Creates C++ variables (names = keys of kwargs) for PyROOT objects (values of kwargs) in PyVars:: namespace'''
  for key, value in kwargs.items():
    ROOT.gInterpreter.Declare(  # type: ignore
f'''
namespace PyVars
{{
  auto& {key} = *reinterpret_cast<{type(value).__cpp_name__}*>({ROOT.addressof(value)});
}}
''')


# set of all possible waves up to ell = 2
PROD_AMPS: Dict[int, Dict[Tuple[int, int,], complex]] = {
  # negative-reflectivity waves
  -1 : {
    #J   M    amplitude
    (0,  0) :  1   + 0j,    # S_0^-
    (1, -1) : -0.4 + 0.1j,  # P_-1^-
    (1,  0) :  0.3 - 0.8j,  # P_0^-
    (1, +1) : -0.8 + 0.7j,  # P_+1^-
    (2, -2) :  0.1 - 0.4j,  # D_-2^-
    (2, -1) :  0.5 + 0.2j,  # D_-1^-
    (2,  0) : -0.1 - 0.2j,  # D_ 0^-
    (2, +1) :  0.2 - 0.1j,  # D_+1^-
    (2, +2) : -0.2 + 0.3j,  # D_+2^-
  },
  # positive-reflectivity waves
  +1 : {
    #J   M    amplitude
    (0,  0) :  0.5 + 0j,    # S_0^+
    (1, -1) :  0.5 - 0.1j,  # P_-1^+
    (1,  0) : -0.8 - 0.3j,  # P_0^+
    (1, +1) :  0.6 + 0.3j,  # P_+1^+
    (2, -2) :  0.2 + 0.1j,  # D_-2^+
    (2, -1) :  0.2 - 0.3j,  # D_-1^+
    (2,  0) :  0.1 - 0.2j,  # D_ 0^+
    (2, +1) :  0.2 + 0.5j,  # D_+1^+
    (2, +2) : -0.3 - 0.1j,  # D_+2^+
  },
}


def drawIntensityTF3(
  fcn:      ROOT.TF3,  # type: ignore
  histName: str,
  nmbBins:  int = 25,
) -> None:
  '''Draws given TF3 for intensity distribution'''
  canv = ROOT.TCanvas()  # type: ignore
  fcnHist = ROOT.TH3F(histName, ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, 0, 180)  # type: ignore
  xAxis = fcnHist.GetXaxis()
  yAxis = fcnHist.GetYaxis()
  zAxis = fcnHist.GetZaxis()
  for xBin in range(1, nmbBins + 1):
    for yBin in range(1, nmbBins + 1):
      for zBin in range(1, nmbBins + 1):
        x = xAxis.GetBinCenter(xBin)
        y = yAxis.GetBinCenter(yBin)
        z = zAxis.GetBinCenter(zBin)
        fcnHist.SetBinContent(xBin, yBin, zBin, fcn.Eval(x, y, z))
  fcnHist.SetMinimum(0)
  fcnHist.GetXaxis().SetTitleOffset(1.5)
  fcnHist.GetYaxis().SetTitleOffset(2)
  fcnHist.GetZaxis().SetTitleOffset(1.5)
  fcnHist.Draw("BOX2")
  canv.SaveAs(f"{fcnHist.GetName()}.pdf")


def calcSpinDensElemSetFromWaves(
  refl:         int,      # reflectivity
  m1:           int,      # m
  m2:           int,      # m'
  prodAmp1:     complex,  # [ell]_m^refl
  prodAmp1NegM: complex,  # [ell]_{-m}^refl
  prodAmp2:     complex,  # [ell']_m'^refl
  prodAmp2NegM: complex,  # [ell']_{-m'}^refl
) -> Tuple[complex, complex, complex]:
  '''Calculates element of spin-density matrix components from given partial-wave amplitudes assuming rank 1'''
  # Eqs. (149) to (151)
  rhos: List[complex] = 3 * [0 + 0j]
  rhos[0] +=                    (           prodAmp1     * prodAmp2.conjugate() + (-1)**(m1 - m2) * prodAmp1NegM * prodAmp2NegM.conjugate())  # Eq. (149)
  rhos[1] +=            -refl * ((-1)**m1 * prodAmp1NegM * prodAmp2.conjugate() + (-1)**m2        * prodAmp1     * prodAmp2NegM.conjugate())  # Eq. (150)
  rhos[2] += -(0 + 1j) * refl * ((-1)**m1 * prodAmp1NegM * prodAmp2.conjugate() - (-1)**m2        * prodAmp1     * prodAmp2NegM.conjugate())  # Eq. (151)
  return tuple(rhos)


def calcMomentSetFromWaves(
  prodAmps: Dict[int, Dict[Tuple[int, int,], complex]],
  L:        int,
  M:        int,
) -> Tuple[complex, complex, complex]:
  '''Calculates values of (H_0, H_1, H_2) with L and M from given production amplitudes assuming rank 1'''
  # Eqs. (153) to (155) assuming that rank is 1
  moments: List[complex] = 3 * [0 + 0j]
  for refl in (-1, +1):
    for wave1 in prodAmps[refl]:
      ell1:         int     = wave1[0]
      m1:           int     = wave1[1]
      prodAmp1:     complex = prodAmps[refl][wave1]
      prodAmp1NegM: complex = prodAmps[refl][(ell1, -m1)]
      for wave2 in prodAmps[refl]:
        ell2:         int     = wave2[0]
        m2:           int     = wave2[1]
        prodAmp2:     complex = prodAmps[refl][wave2]
        prodAmp2NegM: complex = prodAmps[refl][(ell2, -m2)]
        term = math.sqrt((2 * ell2 + 1) / (2 * ell1 + 1)) * (
            py3nj.clebsch_gordan(2 * ell2, 2 * L, 2 * ell1, 0,      0,     0,      ignore_invalid = True)  # (ell_2 0,    L 0 | ell_1 0  )
          * py3nj.clebsch_gordan(2 * ell2, 2 * L, 2 * ell1, 2 * m2, 2 * M, 2 * m1, ignore_invalid = True)  # (ell_2 m_2,  L M | ell_1 m_1)
        )
        if term == 0:  # invalid Clebsch-Gordan
          continue
        rhos: Tuple[complex, complex, complex] = calcSpinDensElemSetFromWaves(refl, m1, m2, prodAmp1, prodAmp1NegM, prodAmp2, prodAmp2NegM)
        moments[0] +=  term * rhos[0]  # H_0; Eq. (123)
        moments[1] += -term * rhos[1]  # H_1; Eq. (124)
        moments[2] += -term * rhos[2]  # H_2; Eq. (124)
  return tuple(moments)


def getMaxSpin(prodAmps: Dict[int, Dict[Tuple[int, int,], complex]]) -> int:
  '''Gets maximum spin from set of production amplitudes'''
  maxSpin = 0
  for refl in (-1, +1):
    for wave in prodAmps[refl]:
      ell = wave[0]
      maxSpin = ell if ell > maxSpin else maxSpin
  return maxSpin


def calcAllMomentsFromWaves(prodAmps: Dict[int, Dict[Tuple[int, int,], complex]]) -> List[Tuple[complex, complex, complex]]:
  '''Calculates moments for given production amplitudes assuming rank 1'''
  moments: List[Tuple[complex, complex, complex]] = []
  maxSpin = getMaxSpin(prodAmps)
  norm = 1
  index = 0
  for L in range(2 * maxSpin + 2):  # calculate moments 2 units above maximum L (should be zero)
    for M in range(L + 1):          # calculate moments 1 unit above maximum M (should be zero)
      momentSet: List[complex] = list(calcMomentSetFromWaves(prodAmps, L, M))
      if ((abs(momentSet[0].imag) > 1e-15) or (abs(momentSet[1].imag) > 1e-15) or (abs(momentSet[2].real) > 1e-15)):
        print(f"Warning: expect (Im[H_0({L} {M})], Im[H_1({L} {M})], Re[H_2({L} {M})]) = (0, 0, 0)  found ({momentSet[0].imag}, {momentSet[1].imag}, {momentSet[2].imag})")
      else:
        # set respective real and imaginary parts exactly to zero.
        momentSet[0] = momentSet[0].real + 0j
        momentSet[1] = momentSet[1].real + 0j
        momentSet[2] = 0 + momentSet[2].imag * 1j
      if (M == 0 and momentSet[2] != 0):
        print(f"Warning: expect H_2({L} {M}) = 0  found {momentSet[2].imag}")
      # normalize to H_0(0, 0)
      norm = momentSet[0] if index == 0 else norm
      moments.append(tuple(moment / norm for moment in momentSet))
      index += 1
  index = 0
  for L in range(2 * maxSpin + 2):
    for M in range(L + 1):
      print(f"(H_0({L} {M}), H_1({L} {M}), H_2({L} {M})) = {moments[index]}")
      index += 1
  return moments


def genDataFromWaves(
  nmbEvents:         int,                                         # number of events to generate
  polarization:      float,                                       # photon-beam polarization
  prodAmps:          Dict[int, Dict[Tuple[int, int,], complex]],  # partial-wave amplitudes
  efficiencyFormula: Optional[str] = None,                        # detection efficiency
) -> ROOT.RDataFrame:  # type: ignore
  '''Generates data according to set of partial-wave amplitudes assuming rank 1'''
  # construct TF3 for intensity distribution in Eq. (152)
  #   x = cos(theta)
  #   y = phi [deg]
  #   z = Phi [deg]
  intensityComponentTerms: List[Tuple[str, str, str]] = []  # terms in sum of each intensity component
  for refl in (-1, +1):
    for wave1 in prodAmps[refl]:
      ell1:         int     = wave1[0]
      m1:           int     = wave1[1]
      prodAmp1:     complex = prodAmps[refl][wave1]
      prodAmp1NegM: complex = prodAmps[refl][(ell1, -m1)]
      decayAmp1 = f"Ylm({ell1}, {m1}, std::acos(x), TMath::DegToRad() * y)"
      for wave2 in prodAmps[refl]:
        ell2:         int     = wave2[0]
        m2:           int     = wave2[1]
        prodAmp2:     complex = prodAmps[refl][wave2]
        prodAmp2NegM: complex = prodAmps[refl][(ell2, -m2)]
        decayAmp2 = f"Ylm({ell2}, {m2}, std::acos(x), TMath::DegToRad() * y)"
        rhos: Tuple[complex, complex, complex] = calcSpinDensElemSetFromWaves(refl, m1, m2, prodAmp1, prodAmp1NegM, prodAmp2, prodAmp2NegM)
        terms = tuple(f"{decayAmp1} * complexT({rho.real}, {rho.imag}) * std::conj({decayAmp2})" for rho in rhos)  # Eq. (152)
        intensityComponentTerms.append(terms)
  # sum terms for each intensity component
  intensityComponentsFormula = []
  for iComponent in range(3):
    intensityComponentsFormula.append(f"({' + '.join([term[iComponent] for term in intensityComponentTerms])})")
  # sum intensity components
  intensityFormula = (
    f"std::real({intensityComponentsFormula[0]} "
    f"- {intensityComponentsFormula[1]} * {polarization} * std::cos(TMath::DegToRad() * z) "
    f"- {intensityComponentsFormula[2]} * {polarization} * std::sin(TMath::DegToRad() * z))"
    + ("" if efficiencyFormula is None else f" * ({efficiencyFormula})"))  # Eq. (112)
  print(f"intensity = {intensityFormula}")
  intensityFcn = ROOT.TF3("intensity", intensityFormula, -1, +1, -180, +180, 0, 180)  # type: ignore
  intensityFcn.SetTitle(";cos#theta;#phi [deg];#Phi [deg]")
  intensityFcn.SetNpx(100)  # used in numeric integration performed by GetRandom()
  intensityFcn.SetNpy(100)
  intensityFcn.SetNpz(100)
  intensityFcn.SetMinimum(0)

  # draw intensity function
  #   intensityFcn.Draw("BOX2") does not work
  #   draw function "by hand" instead
  drawIntensityTF3(intensityFcn, histName = "hIntensity")

  # generate random data that follow intensity given by partial-wave amplitudes
  treeName = "data"
  fileName = f"{intensityFcn.GetName()}.root"
  df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  declareInCpp(intensityFcn = intensityFcn)
  df.Define("point",    "double cosTheta, phiDeg, PhiDeg; PyVars::intensityFcn.GetRandom3(cosTheta, phiDeg, PhiDeg); std::vector<double> point = {cosTheta, phiDeg, PhiDeg}; return point;") \
    .Define("cosTheta", "point[0]") \
    .Define("theta",    "std::acos(cosTheta)") \
    .Define("phiDeg",   "point[1]") \
    .Define("phi",      "TMath::DegToRad() * phiDeg") \
    .Define("PhiDeg",   "point[2]") \
    .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genDataFromWaves" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                   # noop filter before snapshot logs when event loop is running
  return ROOT.RDataFrame(treeName, fileName)  # type: ignore


def genAccepted2BodyPsPhotoProd(
  nmbEvents:         int,                   # number of events to generate
  efficiencyFormula: Optional[str] = None,  # detection efficiency
) -> ROOT.RDataFrame:  # type: ignore
  '''Generates RDataFrame with two-body phase-space distribution weighted by given detection efficiency'''
  # construct efficiency function
  efficiencyFcn = ROOT.TF3("efficiency", "1" if efficiencyFormula is None else efficiencyFormula, -1, +1, -180, +180, 0, 180)  # type: ignore
  efficiencyFcn.SetTitle(";cos#theta;#phi [deg];#Phi [deg]")
  efficiencyFcn.SetNpx(100)  # used in numeric integration performed by GetRandom()
  efficiencyFcn.SetNpy(100)
  efficiencyFcn.SetNpz(100)
  efficiencyFcn.SetMinimum(0)

  # draw efficiency function
  drawIntensityTF3(efficiencyFcn, histName = "hEfficiency")

  # generate isotropic distributions in cos theta, phi, and Phi and weight with efficiency function
  treeName = "data"
  fileName = f"{efficiencyFcn.GetName()}.root"
  df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  declareInCpp(efficiencyFcn = efficiencyFcn)
  df.Define("point", "double cosTheta, phiDeg, PhiDeg; PyVars::efficiencyFcn.GetRandom3(cosTheta, phiDeg, PhiDeg); std::vector<double> point = {cosTheta, phiDeg, PhiDeg}; return point;") \
    .Define("cosTheta", "point[0]") \
    .Define("theta",    "std::acos(cosTheta)") \
    .Define("phiDeg",   "point[1]") \
    .Define("phi",      "TMath::DegToRad() * phiDeg") \
    .Define("PhiDeg",   "point[2]") \
    .Define("Phi",      "TMath::DegToRad() * PhiDeg") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop in genData2BodyPSPhotoProd" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                   # noop filter before snapshot logs when event loop is running
  return ROOT.RDataFrame(treeName, fileName)  # type: ignore


def calcIntegralMatrix(
  phaseSpaceDataFrame: ROOT.RDataFrame,  # (accepted) phase space data  # type: ignore
  nmbEvents:           int,              # number of events in RDataFrame
  polarization:        float,            # photon-beam polarization
  maxL:                int,              # maximum orbital angular momentum
) -> Dict[Tuple[int, ...], complex]:
  '''Calculates integral matrix of spherical harmonics for from provided phase-space data'''
  # define basis functions for physical moments; Eq. (174)
  for momentIndex in range(3):
    for L in range(2 * maxL + 2):
      for M in range(L + 1):
        phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"f_meas_{momentIndex}_{L}_{M}",
          f"f_meas({momentIndex}, {L}, {M}, theta, phi, Phi, {polarization})")
        phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"f_phys_{momentIndex}_{L}_{M}",
          f"f_phys({momentIndex}, {L}, {M}, theta, phi, Phi, {polarization})")
  # define integral-matrix elements; Eq. (177)
  for momentIndex_meas in range(3):
    for L_meas in range(2 * maxL + 2):
      for M_meas in range(L_meas + 1):
        for momentIndex_phys in range(3):
          for L_phys in range(2 * maxL + 2):
            for M_phys in range(L_phys + 1):
              phaseSpaceDataFrame = phaseSpaceDataFrame.Define(f"I_{momentIndex_meas}_{L_meas}_{M_meas}_{momentIndex_phys}_{L_phys}_{M_phys}",
              f"(8 * TMath::Pi() * TMath::Pi() / {nmbEvents})"
              f" * f_meas_{momentIndex_meas}_{L_meas}_{M_meas} * f_phys_{momentIndex_phys}_{L_phys}_{M_phys}")
  # calculate integral matrix
  I_acc: Dict[Tuple[int, ...], complex] = {}
  for momentIndex_meas in range(3):
    for L_meas in range(2 * maxL + 2):
      for M_meas in range(L_meas + 1):
        for momentIndex_phys in range(3):
          for L_phys in range(2 * maxL + 2):
            for M_phys in range(L_phys + 1):
              I_acc[(momentIndex_meas, L_meas, M_meas, momentIndex_phys, L_phys, M_phys)] = (
                phaseSpaceDataFrame.Sum[ROOT.std.complex["double"]](  # type: ignore
                f"I_{momentIndex_meas}_{L_meas}_{M_meas}_{momentIndex_phys}_{L_phys}_{M_phys}").GetValue())
  return I_acc


def calculatePhotoProdMoments(
  inData:         ROOT.RDataFrame,                                  # input data with angular distribution  # type: ignore
  polarization:   float,                                            # photon-beam polarization
  maxL:           int,                                              # maximum spin of decaying object
  integralMatrix: Optional[Dict[Tuple[int, ...], complex]] = None,  # acceptance integral matrix
) -> Tuple[List[Tuple[Tuple[int, int, int], complex]], Dict[Tuple[int, ...], Tuple[float, ...]]]:  # moment values and covariances
  '''Calculates photoproduction moments'''
  # define measured moments; Eq. (178)
  dfMoment = inData
  for momentIndex in range(3):
    for L in range(2 * maxL + 2):
      for M in range(L + 1):
        dfMoment = dfMoment.Define(f"Re_f_meas_{momentIndex}_{L}_{M}",
          f"std::real(f_meas({momentIndex}, {L}, {M}, theta, phi, Phi, {polarization}))")
        dfMoment = dfMoment.Define(f"Im_f_meas_{momentIndex}_{L}_{M}",
          f"std::imag(f_meas({momentIndex}, {L}, {M}, theta, phi, Phi, {polarization}))")
  # calculate moments and their covariance matrix; Eq. (179) and (180)
  nmbEvents = inData.Count().GetValue()
  nmbMoments = 3 * (2 * maxL + 2) * (2 * maxL + 3) // 2
  H_meas    = np.zeros((nmbMoments),            dtype = np.complex128)
  Re_f_meas = np.zeros((nmbMoments, nmbEvents), dtype = np.float64)
  Im_f_meas = np.zeros((nmbMoments, nmbEvents), dtype = np.float64)
  for momentIndex in range(3):
    for L in range(2 * maxL + 2):
      for M in range(L + 1):
        iMoment = momentIndex * (2 * maxL + 2) * (2 * maxL + 3) // 2 + L * (L + 1) // 2 + M
        # calculate value
        momentValRe = dfMoment.Sum(f"Re_f_meas_{momentIndex}_{L}_{M}").GetValue()
        momentValIm = dfMoment.Sum(f"Im_f_meas_{momentIndex}_{L}_{M}").GetValue()
        momentVal = momentValRe + 1j * momentValIm
        H_meas[iMoment] = momentVal
        # get values of basis functions as Numpy arrays
        Re_f_meas[iMoment, :] = dfMoment.AsNumpy(columns = [f"Re_f_meas_{momentIndex}_{L}_{M}"])[f"Re_f_meas_{momentIndex}_{L}_{M}"]
        Im_f_meas[iMoment, :] = dfMoment.AsNumpy(columns = [f"Im_f_meas_{momentIndex}_{L}_{M}"])[f"Im_f_meas_{momentIndex}_{L}_{M}"]
  f_meas = Re_f_meas + 1j * Im_f_meas
  V_meas_aug = nmbEvents * np.cov(f_meas, np.conjugate(f_meas))  # augmented covariance matrix
  # normalize to H_0(0, 0)
  H_meas /= nmbEvents
  V_meas_aug /= nmbEvents**2
  for momentIndex in range(3):
    for L in range(2 * maxL + 2):
      for M in range(L + 1):
        iMoment = momentIndex * (2 * maxL + 2) * (2 * maxL + 3) // 2 + L * (L + 1) // 2 + M
        print(f"H^meas_{momentIndex}(L = {L}, M = {M}) = {H_meas[iMoment]}")
  # correct for detection efficiency
  H_phys = np.zeros((nmbMoments), dtype = np.complex128)
  V_phys_aug = np.zeros((2 * nmbMoments, 2 * nmbMoments), dtype = np.complex128)
  if integralMatrix is None:
    H_phys     = H_meas
    V_phys_aug = V_meas_aug
  else:
    I_acc = np.zeros((nmbMoments, nmbMoments), dtype = np.complex128)
    for momentIndex_meas in range(3):
      for L_meas in range(2 * maxL + 2):
        for M_meas in range(L_meas + 1):
          iMoment_meas = momentIndex_meas * (2 * maxL + 2) * (2 * maxL + 3) // 2 + L_meas * (L_meas + 1) // 2 + M_meas
          for momentIndex_phys in range(3):
            for L_phys in range(2 * maxL + 2):
              for M_phys in range(L_phys + 1):
                iMoment_phys = momentIndex_phys * (2 * maxL + 2) * (2 * maxL + 3) // 2 + L_phys * (L_phys + 1) // 2 + M_phys
                I_acc[iMoment_meas, iMoment_phys] = integralMatrix[(momentIndex_meas, L_meas, M_meas, momentIndex_phys, L_phys, M_phys)]
    # eigenVals, eigenVecs = np.linalg.eig(I_acc)
    # print(f"I eigenvalues = {eigenVals}")
    # # print(f"I eigenvectors = {eigenVecs}")
    # # print(f"I determinant = {np.linalg.det(I)}")
    # print(f"I = \n{np.array2string(I_acc, precision = 3, suppress_small = True, max_line_width = 150)}")
    I_inv = np.linalg.inv(I_acc)
    # # eigenVals, eigenVecs = np.linalg.eig(Iinv)
    # # print(f"I^-1 eigenvalues = {eigenVals}")
    # print(f"I^-1 = \n{np.array2string(Iinv, precision = 3, suppress_small = True, max_line_width = 150)}")
    # plt.figure().colorbar(plt.matshow(Iinv.real))
    # plt.savefig("Iinv_real.pdf")
    # plt.figure().colorbar(plt.matshow(Iinv.imag))
    # plt.savefig("Iinv_imag.pdf")
    # plt.figure().colorbar(plt.matshow(np.absolute(Iinv)))
    # plt.savefig("Iinv_abs.pdf")
    # plt.figure().colorbar(plt.matshow(np.angle(Iinv)))
    # plt.savefig("Iinv_arg.pdf")
    H_phys = I_inv @ H_meas
    # linear uncertainty propagation
    J = I_inv  # Jacobian of efficiency correction
    J_conj = np.zeros((nmbMoments, nmbMoments), dtype = np.complex128)  # conjugate Jacobian
    J_aug = np.block([
      [J,                    J_conj],
      [np.conjugate(J_conj), np.conjugate(J)],
    ])  # augmented Jacobian
    V_phys_aug = J_aug @ (V_meas_aug @ np.asmatrix(J_aug).H)  #!Note! @ is left-associative
  V_phys_Hermit = V_phys_aug[:nmbMoments, :nmbMoments]  # Hermitian covariance matrix
  V_phys_pseudo = V_phys_aug[:nmbMoments, nmbMoments:]  # pseudo-covariance matrix
  # covariances of real and imaginary parts
  V_phys_ReRe = (np.real(V_phys_Hermit) + np.real(V_phys_pseudo)) / 2
  V_phys_ImIm = (np.real(V_phys_Hermit) - np.real(V_phys_pseudo)) / 2
  V_phys_ReIm = (np.imag(V_phys_pseudo) - np.imag(V_phys_Hermit)) / 2
  # reformat output
  momentsPhys:    List[Tuple[Tuple[int, int, int], complex]] = []
  momentsPhysCov: Dict[Tuple[int, ...], Tuple[float, ...]]   = {}  # cov[(L, M, L', M')] = (ReRe, ImIm, ReIm)
  for momentIndex_phys in range(3):
    for L_phys in range(2 * maxL + 2):
      for M_phys in range(L_phys + 1):
        iMoment_phys = momentIndex_phys * (2 * maxL + 2) * (2 * maxL + 3) // 2 + L_phys * (L_phys + 1) // 2 + M_phys
        print(f"H^meas_{momentIndex_phys}(L = {L_phys}, M = {M_phys}) = {H_phys[iMoment_phys]}")
        momentsPhys.append(((momentIndex_phys, L_phys, M_phys), H_phys[iMoment_phys]))
        for momentIndex_meas in range(3):
          for L_meas in range(2 * maxL + 2):
            for M_meas in range(L_meas + 1):
              iMoment_meas = momentIndex_meas * (2 * maxL + 2) * (2 * maxL + 3) // 2 + L_meas * (L_meas + 1) // 2 + M_meas
              momentsPhysCov[(momentIndex_meas, L_meas, M_meas, L_phys, M_phys)] = (
                (V_phys_ReRe[iMoment_meas, iMoment_phys], V_phys_ImIm[iMoment_meas, iMoment_phys], V_phys_ReIm[iMoment_meas, iMoment_phys]))
  # print(momentsPhys)
  #TODO encapsulate moment values and covariances in object that takes care of the index mapping
  return momentsPhys, momentsPhysCov


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
  ROOT.gRandom.SetSeed(1234567890)  # type: ignore
  # ROOT.EnableImplicitMT(10)  # type: ignore
  setupPlotStyle()
  ROOT.gBenchmark.Start("Total execution time")  # type: ignore

  # get data
  nmbEvents = 1000000
  nmbMcEvents = 100000
  polarization = 1.0
  # formulas for detection efficiency: x = cos(theta), y = phi in [-180, +180] deg
  efficiencyFormula = "1"  # acc_perfect
  # efficiencyFormula = "(1.5 - x * x) * (1.5 - y * y / (180 * 180)) * (1.5 - (z - 90) * (z - 90) / (90 * 90))"  # acc_1

  # input from partial-wave amplitudes
  inputMoments: List[Tuple[complex, complex, complex]] = calcAllMomentsFromWaves(PROD_AMPS)
  dataPwaModel = genDataFromWaves(nmbEvents, polarization, PROD_AMPS, efficiencyFormula)

  # plot data
  canv = ROOT.TCanvas()  # type: ignore
  nmbBins = 25
  hist = dataPwaModel.Histo3D(
    ROOT.RDF.TH3DModel("hData", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, 0, 180),
    "cosTheta", "phiDeg", "PhiDeg")
  hist.SetMinimum(0)
  hist.GetXaxis().SetTitleOffset(1.5)
  hist.GetYaxis().SetTitleOffset(2)
  hist.GetZaxis().SetTitleOffset(1.5)
  hist.Draw("BOX2")
  canv.SaveAs(f"{hist.GetName()}.pdf")

  # calculate moments
  dataAcceptedPs = genAccepted2BodyPsPhotoProd(nmbMcEvents, efficiencyFormula)
  integralMatrix = calcIntegralMatrix(dataAcceptedPs, nmbEvents = nmbMcEvents, polarization = polarization, maxL = getMaxSpin(PROD_AMPS))
  print("Moments of accepted phase-space data")
  calculatePhotoProdMoments(dataAcceptedPs, polarization = polarization, maxL = getMaxSpin(PROD_AMPS), integralMatrix = integralMatrix)

  ROOT.gBenchmark.Show("Total execution time")  # type: ignore
