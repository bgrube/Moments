#!/usr/bin/env python3


from __future__ import annotations

import os
import numpy as np

import ROOT

from plotDataTree import CPP_CODE_ALEX


def lambdaKaellen(
  alpha: float,
  beta:  float,
  gamma: float,
) -> float:
  """Calculate the Källén function"""
  return alpha**2 + beta**2 + gamma**2 - 2 * (alpha * beta + alpha * gamma + beta * gamma)


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gROOT.ProcessLine(f".x {os.environ['FSROOT']}/rootlogon.FSROOT.C")
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
  M2 = (beam_lab + target_lab).M2()
  p_beam_Cms = np.sqrt(lambdaKaellen(M2, 0,      m_p**2) / M2) / 2
  E_beam_Cms = p_beam_Cms
  p_X_Cms    = np.sqrt(lambdaKaellen(M2, m_X**2, m_p**2) / M2) / 2
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
  print(f"!!! must be zero: {str(recoil_lab - (beam_lab + target_lab - X_lab))=}")
  # (iii) construct pi+ 4-momentum in helicity frame
  p_pip_Hf = np.sqrt(lambdaKaellen(m_X**2, m_pi**2, m_pi**2)) / (2 * m_X)
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
  print(f"!!!!! {np.degrees(np.sign(pip_Hf.X()) * pip_Hf.Theta())=}, {str(pip_Hf)=}")
  print(f"!!!!! {np.degrees(theta_Hf_axis)=}, {str(recoil_Hf)=}")
  print(f"!!!!! before {np.degrees(np.arccos(pip_Hf.Vect().Unit().Dot(-recoil_Hf.Vect().Unit())))=}")
  rotateY = ROOT.Math.RotationY(theta_Hf_axis)  # active rotation
  pip_Hf = rotateY(pip_Hf)
  pim_Hf = ROOT.Math.PxPyPzMVector(
    -pip_Hf.Px(),
    -pip_Hf.Py(),
    -pip_Hf.Pz(),
    m_pi,
  )
  print(f"!!!!! after {np.degrees(np.arccos(pip_Hf.Vect().Unit().Dot(-recoil_Hf.Vect().Unit())))=}")
  recoil_Hf = rotateY.Inverse()(recoil_Hf)
  print(f"!!!!! must be along -z: {str(recoil_Hf)=}")
  # # (iv) boost pions from helicity to (gamma, p) center-of-momentum frame
  # boostToCms = -(X_Cms.BoostToCM())
  # pip_Cms    = ROOT.Math.VectorUtil.boost(pip_Hf, boostToCms)
  # pim_Cms    = ROOT.Math.VectorUtil.boost(pim_Hf, boostToCms)
  # # (v) boost pions from (gamma, p) center-of-momentum frame to lab frame
  # pip_lab    = ROOT.Math.VectorUtil.boost(pip_Cms,    boostToLab)
  # pim_lab    = ROOT.Math.VectorUtil.boost(pim_Cms,    boostToLab)
  # (v) boost pions from helicity to lab frame
  boostToLab = ROOT.Math.Boost(-(X_lab.BoostToCM()))
  pip_lab    = boostToLab(pip_Hf)
  pim_lab    = boostToLab(pim_Hf)
  print(f"!!! must be zero: {str(X_lab - (pip_lab + pim_lab))=}")
  print(f"!!! must be zero: {str(beam_lab + target_lab - (pip_lab + pim_lab + recoil_lab))=}")

  # recover input values
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
  print(f"??? {pip_theta_Hf_Alex=}, {pip_phi_Hf_Alex=}")
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
  print(f"??? {pip_theta_Hf_FSRoot=}, {pip_phi_Hf_FSRoot=}")

  # boost all 4-vectors into the resonance rest frame
  locPiPlusP4_Resonance  = ROOT.TLorentzVector(pip_lab.X(),    pip_lab.Y(),    pip_lab.Z(),    pip_lab.E()   )
  locPiMinusP4_Resonance = ROOT.TLorentzVector(pim_lab.X(),    pim_lab.Y(),    pim_lab.Z(),    pim_lab.E()   )
  locRecoilP4_Resonance  = ROOT.TLorentzVector(recoil_lab.X(), recoil_lab.Y(), recoil_lab.Z(), recoil_lab.E())
  locBeamP4_Resonance    = ROOT.TLorentzVector(beam_lab.X(),   beam_lab.Y(),   beam_lab.Z(),   beam_lab.E()  )
  resonanceP4 = locPiPlusP4_Resonance + locPiMinusP4_Resonance
  boostP3 = -resonanceP4.BoostVector()
  locPiPlusP4_Resonance.Boost (boostP3)
  locPiMinusP4_Resonance.Boost(boostP3)
  locRecoilP4_Resonance.Boost (boostP3)
  locBeamP4_Resonance.Boost   (boostP3)
  locPiPlusP4_Resonance.Print()
  print(f"!!! vs. {str(pip_Hf)=}, {pip_Hf.E()=}")
  locPiMinusP4_Resonance.Print()
  print(f"!!! vs. {str(pim_Hf)=}, {pim_Hf.E()=}")
  locRecoilP4_Resonance.Print()
  print(f"!!! vs. {str(recoil_Hf)=}, {recoil_Hf.E()=}")
  y = (locBeamP4_Resonance.Vect().Unit().Cross(-locRecoilP4_Resonance.Vect().Unit())).Unit()
  z = -locRecoilP4_Resonance.Vect().Unit()
  x = y.Cross(z).Unit()
  x.Print()
  y.Print()
  z.Print()
