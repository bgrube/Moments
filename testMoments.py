#!/usr/bin/env python3


import math
from typing import Any, Collection, Dict, List, Tuple

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
) -> Tuple[str, str]:
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
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `CosTheta` column would be regenerated for every triggered loop
                                   # use noop filter to log when event loop is running
  return treeName, fileName


def calculateLegMoments(
  dataFrame: Any,
  maxDegree: int,
) -> Dict[Tuple[int, ...], UFloat]:
  '''Calculates moments of Legendre polynomials'''
  nmbEvents = dataFrame.Count().GetValue()
  moments: Dict[Tuple[int, ...], UFloat] = {}
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
) -> Tuple[str, str]:
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
  df.Define("point",    "double CosTheta, Phi; PyVars::sphericalHarmLC.GetRandom2(CosTheta, Phi); std::vector<double> point = {CosTheta, Phi}; return point;") \
    .Define("CosTheta", "point[0]") \
    .Define("Phi",      "point[1]") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                   # use noop filter to log when event loop is running
  return treeName, fileName


def calculateSphHarmMoments(
  dataFrame: Any,
  maxL:      int,  # maximum spin of decaying object
) -> List[Tuple[Tuple[int, ...], UFloat]]:
  '''Calculates moments of spherical harmonics'''
  nmbEvents = dataFrame.Count().GetValue()
  moments: List[Tuple[Tuple[int, ...], UFloat]] = []
  for L in range(2 * maxL + 2):
    for M in range(min(L, 2) + 1):
      # unnormalized moments
      dfMoment = dataFrame.Define("sphericalHarm", f"ROOT::Math::sph_legendre({L}, {M}, std::acos(CosTheta)) * std::cos({M} * TMath::DegToRad() * Phi)")
      momentVal = dfMoment.Sum("sphericalHarm").GetValue()
      momentErr = math.sqrt(nmbEvents) * dfMoment.StdDev("sphericalHarm").GetValue()  # iid events: Var[sum_i^N f(x_i)] = sum_i^N Var[f] = N * Var[f]; see https://www.wikiwand.com/en/Monte_Carlo_integration
      # normalize moments with respect to H(0 0)
      #     Integrate[Re[SphericalHarmonicY[L, M, x, y]] * Sin[x], {y, -Pi, Pi}, {x, 0, Pi}]
      norm = 1 / (nmbEvents * math.sqrt((2 * L + 1) / (4 * math.pi)))
      moment = norm * ufloat(momentVal, momentErr)  # type: ignore
      print(f"H(L = {L}, M = {M}) = {moment}")
      moments.append(((L, M), moment))
  print(moments)
  return moments


# C++ implementation of (complex conjugated) Wigner D function
ROOT.gROOT.LoadMacro("./wignerD.C++")

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
  nmbEvents: int,
  prodAmps:  Dict[int, Tuple[complex, ...]],
) -> Tuple[str, str]:
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
      V = f"complexT({prodAmps[refl][waveIndex].real}, {prodAmps[refl][waveIndex].imag})"  # complexT is a typedef in wignerD.C
      A = f"std::sqrt((2 * {ell} + 1) / (4 * TMath::Pi())) * wignerDReflConj({2 * ell}, {2 * m}, 0, {parity}, {refl}, TMath::DegToRad() * y, std::acos(x))"
      coherentTerms.append(f"{V} * {A}")
    incoherentTerms.append(f"std::norm({' + '.join(coherentTerms)})")
  # see Eqs. (28) for rank = 1
  print("intensity =", " + ".join(incoherentTerms))
  intensity = ROOT.TF2("intensity", " + ".join(incoherentTerms), -1, +1, -180, +180)  # type: ignore
  intensity.SetNpx(500)  # used in numeric integration performed by GetRandom()
  intensity.SetNpy(500)
  intensity.SetContour(100)
  intensity.SetMinimum(0)

  # draw function
  canv = ROOT.TCanvas()  # type: ignore
  intensity.Draw("COLZ")
  canv.SaveAs(f"{intensity.GetName()}.pdf")

  # generate random data that follow linear combination of of spherical harmonics
  treeName = "data"
  fileName = f"{intensity.GetName()}.root"
  df = ROOT.RDataFrame(nmbEvents)  # type: ignore
  declareInCpp(intensity = intensity)
  df.Define("point",    "double CosTheta, Phi; PyVars::intensity.GetRandom2(CosTheta, Phi); std::vector<double> point = {CosTheta, Phi}; return point;") \
    .Define("CosTheta", "point[0]") \
    .Define("Phi",      "point[1]") \
    .Filter('if (rdfentry_ == 0) { cout << "Running event loop" << endl; } return true;') \
    .Snapshot(treeName, fileName)  # snapshot is needed or else the `point` column would be regenerated for every triggered loop
                                   # use noop filter to log when event loop is running
  return treeName, fileName


def theta(m: int) -> float:
  '''Calculates normalization factor in reflectivity basis'''
  # see Eq. (19)
  if m > 0:
    return 1 / math.sqrt(2)
  elif m == 0:
    return 1 / 2
  else:
    return 0


def calculateTruePwdMoment(
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


def calculateTruePwdMoments(
  prodAmps: Dict[int, Tuple[complex, ...]],
  maxL:     int,  # maximum spin of decaying object
) -> List[float]:
  '''Calculates true moments for given production amplitudes'''
  moments: List[float] = []
  for L in range(2 * maxL + 2):
    for M in range(min(L, 2) + 1):
      moment: complex = calculateTruePwdMoment(prodAmps, L, M)
      if (abs(moment.imag) > 1e-15):
        print(f"Warning: non vanishing imaginary part for moment H({L} {M}) = {moment}")
      moments.append(moment.real)
  # normalize to first moment
  moments = [moment / moments[0] for moment in moments]
  index = 0
  for L in range(2 * maxL + 2):
    for M in range(min(L, 2) + 1):
      print(f"H_true(L = {L}, M = {M}) = {moments[index]}")
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
  dfMoment = dataFrame.Define("WignerD",  f"wignerD({2 * L}, {2 * M}, 0, TMath::DegToRad() * Phi, std::acos(CosTheta))") \
                      .Define("WignerDRe", "real(WignerD)") \
                      .Define("WignerDIm", "imag(WignerD)")
  momentVal   = dfMoment.Sum[ROOT.std.complex["double"]]("WignerD").GetValue()
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
  # moments: List[Tuple[Tuple[int, ...], UFloat]] = []
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


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)  # type: ignore
  ROOT.gRandom.SetSeed(1234567890)  # type: ignore

  # get data
  nmbEvents = 1000
  # # Legendre polynomials
  # chose parameters such that resulting linear combinations are positive definite
  # maxOrder = 5
  # # parameters = (1, 1, 0.5, -0.5, -0.25, 0.25)
  # parameters = (0.5, 0.5, 0.25, -0.25, -0.125, 0.125)
  # treeName, fileName = generateDataLegPolLC(nmbEvents,  maxDegree = maxOrder, parameters = parameters)

  # # spherical harmonics
  # maxOrder = 2
  # # parameters = (1, 0.025, 0.02, 0.015, 0.01, -0.02, 0.025, -0.03, -0.035, 0.04, 0.045, 0.05)
  # parameters = (2, 0.05, 0.04, 0.03, 0.02, -0.04, 0.05, -0.06, -0.07, 0.08, 0.09, 0.10)
  # treeName, fileName = generateDataSphHarmLC(nmbEvents, maxL = maxOrder, parameters = parameters)

  # normalize parameters 0th moment and pad with 0
  # trueMoments = [par / parameters[0] for par in parameters]
  # if len(trueMoments) < len(moments):
  #   trueMoments += [0] * (len(moments) - len(trueMoments))

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
  trueMoments: List[float] = calculateTruePwdMoments(prodAmps, maxL = maxOrder)
  treeName, fileName = generateDataPwd(nmbEvents, prodAmps)
  ROOT.EnableImplicitMT(10)  # type: ignore
  dataFrame = ROOT.RDataFrame(treeName, fileName)  # type: ignore
  # print("!!!", dataFrame.AsNumpy())

  # plot data
  canv = ROOT.TCanvas()  # type: ignore
  if "Phi" in dataFrame.GetColumnNames():
    hist = dataFrame.Histo2D(ROOT.RDF.TH2DModel("hData", "", 25, -1, +1, 25, -180, +180), "CosTheta", "Phi")  # type: ignore
    hist.SetMinimum(0)
    hist.Draw("COLZ")
  else:
    hist = dataFrame.Histo1D(ROOT.RDF.TH1DModel("hData", "", 100, -1, +1), "CosTheta")  # type: ignore
    hist.SetMinimum(0)
    hist.Draw()
  canv.SaveAs(f"{hist.GetName()}.pdf")

  # calculate moments
  # calculateLegMoments(dataFrame, maxDegree = maxOrder)
  moments: List[Tuple[Tuple[int, ...], UFloat]] = calculateSphHarmMoments(dataFrame, maxL = maxOrder)
  calculateWignerDMoments(dataFrame, maxL = maxOrder)
  #TODO check whether using Eq. (6) instead of Eq. (13) yields moments that fulfill Eqs. (11) and (12)

  hStack = ROOT.THStack("hCompare", "")  # type: ignore
  nmbBins = len(moments)
  # create histogram with measured values
  histMeas = ROOT.TH1D("Measured values", ";;value", nmbBins, 0, nmbBins)  # type: ignore
  for index, moment in enumerate(moments):
    histMeas.SetBinContent(index + 1, moment[1].nominal_value)
    histMeas.SetBinError  (index + 1, moment[1].std_dev)
    histMeas.GetXaxis().SetBinLabel(index + 1, f"H({' '.join(tuple(str(n) for n in moment[0]))})")
  histMeas.SetLineColor(ROOT.kRed)  # type: ignore
  histMeas.SetMarkerColor(ROOT.kRed)  # type: ignore
  histMeas.SetMarkerStyle(ROOT.kFullCircle)  # type: ignore
  histMeas.SetMarkerSize(0.75)
  hStack.Add(histMeas, "PEX0")
  # create histogram with true values
  histTrue = ROOT.TH1D("True values", ";;value", nmbBins, 0, nmbBins)  # type: ignore
  for index, trueMoment in enumerate(trueMoments):
    histTrue.SetBinContent(index + 1, trueMoment)
    histTrue.SetBinError  (index + 1, 1e-16)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
  histTrue.SetMarkerColor(ROOT.kBlue)  # type: ignore
  histTrue.SetLineColor(ROOT.kBlue)  # type: ignore
  hStack.Add(histTrue, "PE")
  canv = ROOT.TCanvas()  # type: ignore
  hStack.Draw("NOSTACK")
  hStack.GetHistogram().SetLineColor(ROOT.kBlack)  # type: ignore  # make automatic zero line dashed
  hStack.GetHistogram().SetLineStyle(ROOT.kDashed)  # type: ignore  # make automatic zero line dashed
  # hStack.GetHistogram().SetLineWidth(0)  # remove zero line; see https://root-forum.cern.ch/t/continuing-the-discussion-from-an-unwanted-horizontal-line-is-drawn-at-y-0/50877/1
  canv.BuildLegend(0.7, 0.75, 0.99, 0.99)
  canv.SaveAs(f"{hStack.GetName()}.pdf")

  # draw residuals
  #TODO calculate and print chi^2 / ndf
  residuals = tuple((moment[1].nominal_value - trueMoments[index]) / moment[1].std_dev if moment[1].std_dev > 0 else 0 for index, moment in enumerate(moments))
  histRes = ROOT.TH1D("hResiduals", ";;(measured - true) / #sigma_{measured}", nmbBins, 0, nmbBins)  # type: ignore
  chi2Ndf = sum(tuple(residual**2 for residual in residuals)) / (len(residuals) - 1)  # H(0, 0) has by definition a vanishing residual
  for index, residual in enumerate(residuals):
    histRes.SetBinContent(index + 1, residual)
    histRes.SetBinError  (index + 1, 1e-16)  # must not be zero, otherwise ROOT does not draw x error bars; sigh
    histRes.GetXaxis().SetBinLabel(index + 1, histMeas.GetXaxis().GetBinLabel(index + 1))
  histRes.SetMarkerColor(ROOT.kBlue)  # type: ignore
  histRes.SetLineColor(ROOT.kBlue)  # type: ignore
  histRes.SetMinimum(-3)
  histRes.SetMaximum(+3)
  canv = ROOT.TCanvas()  # type: ignore
  histRes.Draw("PE")
  label = ROOT.TLatex()  # type: ignore
  label.SetNDC()
  label.SetTextAlign(ROOT.kHAlignLeft + ROOT.kVAlignBottom)  # type: ignore
  label.DrawLatex(0.12, 0.9075, f"#chi^{{2}}/n.d.f. = {chi2Ndf:.2f}")
  canv.SaveAs(f"{histRes.GetName()}.pdf")
