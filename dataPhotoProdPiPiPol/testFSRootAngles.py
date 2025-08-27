#!/usr/bin/env python3


from __future__ import annotations

import os
import numpy as np

import ROOT


# Alex' code to calculate helicity angles
CPP_CODE_ALEX = """
TVector3
helPiPlusVector(
	const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
	const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
	const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
	const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
	// boost all 4-vectors into the resonance rest frame
	TLorentzVector locPiPlusP4_Resonance (PxPip,    PyPip,    PzPip,    EPip);
	TLorentzVector locPiMinusP4_Resonance(PxPim,    PyPim,    PzPim,    EPim);
	TLorentzVector locRecoilP4_Resonance (PxRecoil, PyRecoil, PzRecoil, ERecoil);
	TLorentzVector locBeamP4_Resonance   (PxBeam,   PyBeam,   PzBeam,   EBeam);
	const TLorentzVector resonanceP4 = locPiPlusP4_Resonance + locPiMinusP4_Resonance;
	const TVector3 boostP3 = -resonanceP4.BoostVector();
	locPiPlusP4_Resonance.Boost (boostP3);
	locPiMinusP4_Resonance.Boost(boostP3);
	locRecoilP4_Resonance.Boost (boostP3);
	locBeamP4_Resonance.Boost   (boostP3);

	// COORDINATE SYSTEM:
	// Normal to the production plane
	const TVector3 y = (locBeamP4_Resonance.Vect().Unit().Cross(-locRecoilP4_Resonance.Vect().Unit())).Unit();
	// Helicity: z-axis opposite recoil proton in rho rest frame
	const TVector3 z = -locRecoilP4_Resonance.Vect().Unit();
	const TVector3 x = y.Cross(z).Unit();
	const TVector3 v(locPiPlusP4_Resonance.Vect() * x, locPiPlusP4_Resonance.Vect() * y, locPiPlusP4_Resonance.Vect() * z);
	return v;
}

double
helcostheta_Alex(
	const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
	const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
	const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
	const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
	const TVector3 v = helPiPlusVector(
		PxPip,    PyPip,    PzPip,    EPip,
		PxPim,    PyPim,    PzPim,    EPim,
		PxRecoil, PyRecoil, PzRecoil, ERecoil,
		PxBeam,   PyBeam,   PzBeam,   EBeam
	);
	return v.CosTheta();
}

double
helphideg_Alex(
	const double PxPip,    const double PyPip,    const double PzPip,    const double EPip,
	const double PxPim,    const double PyPim,    const double PzPim,    const double EPim,
	const double PxRecoil, const double PyRecoil, const double PzRecoil, const double ERecoil,
	const double PxBeam,   const double PyBeam,   const double PzBeam,   const double EBeam
) {
	const TVector3 v = helPiPlusVector(
		PxPip,    PyPip,    PzPip,    EPip,
		PxPim,    PyPim,    PzPim,    EPim,
		PxRecoil, PyRecoil, PzRecoil, ERecoil,
		PxBeam,   PyBeam,   PzBeam,   EBeam
	);
	return v.Phi() * TMath::RadToDeg();
}
"""


def lambdaKaellen(
  alpha: float,
  beta:  float,
  gamma: float,
) -> float:
  """Calculates the Källén function"""
  return alpha**2 + beta**2 + gamma**2 - 2 * (alpha * beta + alpha * gamma + beta * gamma)

def breakupMomentum(
  M:  float,
  m1: float,
  m2: float,
) -> float:
  """Calculates the breakup momentum for M -> m1 + m2"""
  return np.sqrt(lambdaKaellen(M**2, m1**2, m2**2)) / (2 * M)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.C"
  ROOT.gInterpreter.Declare(CPP_CODE_ALEX)

  E_beam_lab   = 9.0  # [GeV]
  m_X          = 1.0  # [GeV]
  minusT       = 0.5  # [GeV^2]
  pip_theta_Hf = np.radians(25)  # [rad]
  pip_phi_Hf   = np.radians(50)  # [rad]
  m_pi         = 0.13957018  # [GeV]
  m_p          = 0.938272  # [GeV]

  # construct event gamma + p -> pi+ + pi- + p in lab frame
  beam_lab   = ROOT.Math.PxPyPzMVector(0, 0, E_beam_lab, 0)  # beam photon along z axis of lab frame
  target_lab = ROOT.Math.PxPyPzMVector(0, 0, 0, m_p)  # proton at rest
  # (i) construct X and recoil 4-momenta in (x, z) plane in (gamma, p) center-of-momentum frame; i.e. production plane normal is along y-axis
  M = (beam_lab + target_lab).M()
  p_beam_Cms = breakupMomentum(M, 0, m_p)
  E_beam_Cms = p_beam_Cms
  p_X_Cms    = breakupMomentum(M, m_X, m_p)
  E_X_Cms    = np.sqrt(p_beam_Cms**2 + m_X**2)
  # invert t = (p_beam - p_X)^2 = m_beam^2 + m_X^2 - 2 * (E_beam * E_X - p_beam * p_X * cos(theta))
  cosTheta_X_Cms = (-minusT - m_X**2 + 2 * E_beam_Cms * E_X_Cms) / (2 * p_beam_Cms * p_X_Cms)
  X_Cms = ROOT.Math.PxPyPzMVector(
    p_X_Cms * np.sqrt(1 - cosTheta_X_Cms**2),
    0,
    p_X_Cms * cosTheta_X_Cms,
    m_X,
  )
  recoil_Cms = ROOT.Math.PxPyPzMVector(
    -X_Cms.Px(),
    -X_Cms.Py(),
    -X_Cms.Pz(),
    m_p,
  )
  # (ii) boost X and recoil from center-of-momentum to lab frame, i.e. along z axis
  boostToLab = ROOT.Math.Boost(-((beam_lab + target_lab).BoostToCM()))
  # print(f"!!! boost to lab frame = ({boostToLab.BoostVector().X()}, {boostToLab.BoostVector().Y()}, {boostToLab.BoostVector().Z()})")
  X_lab      = boostToLab(X_Cms)
  recoil_lab = boostToLab(recoil_Cms)
  # print(f"!!! must be zero: {str(recoil_lab - (beam_lab + target_lab - X_lab))=}")
  # (iii) construct pi+ 4-momentum in helicity frame
  p_pip_Hf = breakupMomentum(m_X, m_pi, m_pi)
  pip_Hf = ROOT.Math.PxPyPzMVector(
    p_pip_Hf * np.sin(pip_theta_Hf) * np.cos(pip_phi_Hf),
    p_pip_Hf * np.sin(pip_theta_Hf) * np.sin(pip_phi_Hf),
    p_pip_Hf * np.cos(pip_theta_Hf),
    m_pi,
  )
  # (iv) rotate such that z_Hf-axis is opposite to recoil direction in helicity frame and construct pi- 4-momentum
  boostToHf = ROOT.Math.Boost(X_lab.BoostToCM())
  recoil_Hf = boostToHf(recoil_lab)
  theta_Hf_axis = np.sign(-recoil_Hf.X()) * (-recoil_Hf).Theta()
  # print(f"!!!!! {np.degrees(np.sign(pip_Hf.X()) * pip_Hf.Theta())=}, {str(pip_Hf)=}")
  # print(f"!!!!! {np.degrees(theta_Hf_axis)=}, {str(recoil_Hf)=}")
  # print(f"!!!!! before {np.degrees(np.arccos(pip_Hf.Vect().Unit().Dot(-recoil_Hf.Vect().Unit())))=}")
  rotateY = ROOT.Math.RotationY(theta_Hf_axis)  # active rotation
  pip_Hf = rotateY(pip_Hf)
  pim_Hf = ROOT.Math.PxPyPzMVector(
    -pip_Hf.Px(),
    -pip_Hf.Py(),
    -pip_Hf.Pz(),
    m_pi,
  )
  # print(f"!!!!! after {np.degrees(np.arccos(pip_Hf.Vect().Unit().Dot(-recoil_Hf.Vect().Unit())))=}")
  # recoil_Hf = rotateY.Inverse()(recoil_Hf)
  # print(f"!!!!! must be along -z: {str(recoil_Hf)=}")
  if True:
    # (v) boost pions from helicity to lab frame
    boostToLab = ROOT.Math.Boost(-(X_lab.BoostToCM()))
    pip_lab    = boostToLab(pip_Hf)
    pim_lab    = boostToLab(pim_Hf)
    # print(f"!!! must be zero: {str(X_lab - (pip_lab + pim_lab))=}")
    # print(f"!!! must be zero: {str(beam_lab + target_lab - (pip_lab + pim_lab + recoil_lab))=}")
  else:
    #!NOTE! this does not work; the non-collinear boosts induce a Wigner rotation that shifts the angles
    # (v) boost pions from helicity to (gamma, p) center-of-momentum frame
    boostToCms = ROOT.Math.Boost(-(X_Cms.BoostToCM()))
    pip_Cms    = boostToCms(pip_Hf)
    pim_Cms    = boostToCms(pim_Hf)
    # (vi) boost pions from (gamma, p) center-of-momentum frame to lab frame
    pip_lab    = boostToLab(pip_Cms)
    pim_lab    = boostToLab(pim_Cms)

  print(f"Comparing to to true values: {np.degrees(pip_theta_Hf)=} deg, {np.degrees(pip_phi_Hf)=} deg")
  # use Alex' functions to recover input values
  pip_theta_Hf_Alex = np.degrees(np.arccos(ROOT.helcostheta_Alex(
    pip_lab.X(),    pip_lab.Y(),    pip_lab.Z(),    pip_lab.E(),
    pim_lab.X(),    pim_lab.Y(),    pim_lab.Z(),    pim_lab.E(),
    recoil_lab.X(), recoil_lab.Y(), recoil_lab.Z(), recoil_lab.E(),
    beam_lab.X(),   beam_lab.Y(),   beam_lab.Z(),   beam_lab.E(),
  )))
  pip_phi_Hf_Alex = ROOT.helphideg_Alex(
    pip_lab.X(),    pip_lab.Y(),    pip_lab.Z(),    pip_lab.E(),
    pim_lab.X(),    pim_lab.Y(),    pim_lab.Z(),    pim_lab.E(),
    recoil_lab.X(), recoil_lab.Y(), recoil_lab.Z(), recoil_lab.E(),
    beam_lab.X(),   beam_lab.Y(),   beam_lab.Z(),   beam_lab.E(),
  )
  print(f"Alex' functions: {pip_theta_Hf_Alex=} deg, {pip_phi_Hf_Alex=} deg")
  # use FSRoot functions to recover input values
  pip_theta_Hf_FSRoot = np.degrees(np.arccos(ROOT.FSMath.helcostheta(
    pip_lab.X(),    pip_lab.Y(),    pip_lab.Z(),    pip_lab.E(),
    pim_lab.X(),    pim_lab.Y(),    pim_lab.Z(),    pim_lab.E(),
    recoil_lab.X(), recoil_lab.Y(), recoil_lab.Z(), recoil_lab.E(),
  )))
  pip_phi_Hf_FSRoot = np.degrees(ROOT.FSMath.helphi(
    pip_lab.X(),    pip_lab.Y(),    pip_lab.Z(),    pip_lab.E(),
    pim_lab.X(),    pim_lab.Y(),    pim_lab.Z(),    pim_lab.E(),
    recoil_lab.X(), recoil_lab.Y(), recoil_lab.Z(), recoil_lab.E(),
    beam_lab.X(),   beam_lab.Y(),   beam_lab.Z(),   beam_lab.E(),
  ))
  pip_phi_Hf_FSRoot_Flipped = np.degrees(ROOT.FSMath.helphi(
    pim_lab.X(),    pim_lab.Y(),    pim_lab.Z(),    pim_lab.E(),
    pip_lab.X(),    pip_lab.Y(),    pip_lab.Z(),    pip_lab.E(),
    recoil_lab.X(), recoil_lab.Y(), recoil_lab.Z(), recoil_lab.E(),
    beam_lab.X(),   beam_lab.Y(),   beam_lab.Z(),   beam_lab.E(),
  ))
  print(f"FSRoot functions: {pip_theta_Hf_FSRoot=} deg, {pip_phi_Hf_FSRoot=} deg, {pip_phi_Hf_FSRoot_Flipped=} deg")
