#!/usr/bin/env python3


from __future__ import annotations

import os

import ROOT


# C++ function to calculate invariant mass of a pair of particles
CPP_CODE_MASSPAIR = """
double
massPair(
	const double Px1, const double Py1, const double Pz1, const double E1,  // 4-momentum of particle 1 [GeV]
	const double Px2, const double Py2, const double Pz2, const double E2   // 4-momentum of particle 2 [GeV]
)	{
	const TLorentzVector p1(Px1, Py1, Pz1, E1);
	const TLorentzVector p2(Px2, Py2, Pz2, E2);
	return (p1 + p2).M();
}
"""

# C++ function to calculate mandelstam t = (p1 - p2)^2
CPP_CODE_MANDELSTAM_T = """
double
mandelstamT(
  const double Px1, const double Py1, const double Pz1, const double E1,  // 4-momentum of particle 1 [GeV]
  const double Px2, const double Py2, const double Pz2, const double E2   // 4-momentum of particle 2 [GeV]
) {
  const TLorentzVector p1(Px1, Py1, Pz1, E1);
  const TLorentzVector p2(Px2, Py2, Pz2, E2);
  return (p1 - p2).M2();
}
"""


def lorentzVectors(realData: bool = True) -> tuple[str, str, str, str, str]:
  """Returns Lorentz-vectors for beam photon, target proton, recoil proton, pi+, and pi-"""
  targetProton = "0, 0, 0, 0.938271999359130859375"  # proton mass value from phase-space generator
  if realData:
    return (
      "beam_p4_kin.Px(), beam_p4_kin.Py(), beam_p4_kin.Pz(), beam_p4_kin.Energy()",  # beam photon
      targetProton,
      "p_p4_kin.Px(),    p_p4_kin.Py(),    p_p4_kin.Pz(),    p_p4_kin.Energy()",     # recoil proton
      "pip_p4_kin.Px(),  pip_p4_kin.Py(),  pip_p4_kin.Pz(),  pip_p4_kin.Energy()",   # pi+
      "pim_p4_kin.Px(),  pim_p4_kin.Py(),  pim_p4_kin.Pz(),  pim_p4_kin.Energy()",   # pi-
    )
  else:
    return(
      "Px_Beam,          Py_Beam,          Pz_Beam,          E_Beam",           # beam photon
      targetProton,
      "Px_FinalState[0], Py_FinalState[0], Pz_FinalState[0], E_FinalState[0]",  # recoil proton
      "Px_FinalState[1], Py_FinalState[1], Pz_FinalState[1], E_FinalState[1]",  # pi+
      "Px_FinalState[2], Py_FinalState[2], Pz_FinalState[2], E_FinalState[2]",  # pi-
    )


CPP_CODE_FLIPYAXIS = """
double
flipYAxis(
	double     phi,
	const bool flip = false
) {
	if (not flip) {
		return phi;
	}
	phi += TMath::Pi();
	// ensure [-pi, +pi] range
	while (phi > TMath::Pi()) {
		phi -= TMath::TwoPi();
	}
	while (phi < -TMath::Pi()) {
		phi += TMath::TwoPi();
	}
	return phi;
}
"""
ROOT.gInterpreter.Declare(CPP_CODE_FLIPYAXIS)


def defineAngleFormulas(
  df:          ROOT.RDataFrame,
  lvBeam:      str,  # argument list with Lorentz-vector components
  lvRecoil:    str,  # argument list with Lorentz-vector components
  lvA:         str,  # argument list with Lorentz-vector components
  lvB:         str,  # argument list with Lorentz-vector components
  frame:       str            = "Hf",   # can be either "Hf" for helicity or "Gj" for Gottfried-Jackson frame
  flipYAxis:   bool           = False,  # if set y-axis is inverted
  columnNames: dict[str, str] = {       # names of columns to define: key: column, value: name
    "cosThetaCol" : "cosTheta",
    "thetaCol"    : "theta",
    "phiCol"      : "phi",
  },
) -> ROOT.RDataFrame:
  """Defines formulas for (A, B) pair mass, and angles (cos(theta), phi) of particle A in X rest frame for reaction beam + target -> X + recoil with X -> A + B using the given Lorentz-vector components"""
  assert frame == "Hf" or frame == "Gj", f"Unknown frame '{frame}'"
  cosThetaCol = columnNames["cosThetaCol"]
  thetaCol    = columnNames["thetaCol"   ]
  phiCol      = columnNames["phiCol"     ]
  phiDegCol   = columnNames["phiCol"     ] + "Deg"
  print(f"Defining columns {cosThetaCol}, {thetaCol}, {phiCol}, and {phiDegCol}")
  return (
    df.Define(cosThetaCol, "(Double32_t)" + (f"FSMath::helcostheta({lvA}, {lvB}, {lvRecoil})" if frame == "Hf" else f"FSMath::gjcostheta({lvA}, {lvB}, {lvBeam})") )  #!NOTE! frames have different signatures (see FSBasic/FSMath.h)
      .Define(thetaCol,    f"(Double32_t)std::acos({cosThetaCol})")
      .Define(
        phiCol,
        # use A as analyzer
        # y_HF/GJ = p_beam x p_recoil if flipYAxis is False else -yHF
        "(Double32_t)" +
        (
          f"flipYAxis(FSMath::helphi({lvA}, {lvB}, {lvRecoil}, {lvBeam}), {'true' if flipYAxis else 'false'})" if frame == "Hf"  # use z_HF = -p_recoil
          else f"flipYAxis(FSMath::gjphi ({lvA}, {lvB}, {lvRecoil}, {lvBeam}), {'true' if flipYAxis else 'false'})"                   # use z_GJ = p_beam
        )
      )
      .Define(phiDegCol,   f"(Double32_t){phiCol} * TMath::RadToDeg()")
  )


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat("i")
  ROOT.gSystem.AddDynamicPath("$FSROOT/lib")
  ROOT.gROOT.SetMacroPath("$FSROOT:" + ROOT.gROOT.GetMacroPath())
  assert ROOT.gROOT.LoadMacro(f"{os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C") == 0, f"Error loading {os.environ['FSROOT']}/rootlogon.FSROOT.sharedLib.C"
  ROOT.gInterpreter.Declare(CPP_CODE_MASSPAIR)
  ROOT.gInterpreter.Declare(CPP_CODE_MANDELSTAM_T)

  dataSigRegionFileName = "./amptools_tree_data_tbin1_ebin4.root"
  dataBkgRegionFileName = "./amptools_tree_bkgnd_tbin1_ebin4.root"
  phaseSpaceAccFileName = "./amptools_tree_accepted_tbin1_ebin4*.root"
  phaseSpaceGenFileName = "./amptools_tree_thrown_tbin1_ebin4*.root"
  treeName              = "kin"
  columnsToWrite        = ["mass", "cosTheta", "theta", "phi", "phiDeg"]

  # convert real data
  # create friend trees with correct weights
  for dataFileName, weightFormula in [(dataSigRegionFileName, "Weight"), (dataBkgRegionFileName, "-Weight")]:
    friendFileName = f"{dataFileName}.weights"
    if os.path.exists(friendFileName):
      print(f"File '{friendFileName}' already exists, skipping creation of friend tree")
      continue
    print(f"Creating file '{friendFileName}' that contains friend tree with weights for file '{dataFileName}'")
    (
      ROOT.RDataFrame(treeName, dataFileName)
          .Define("eventWeight", weightFormula)
          .Snapshot(treeName, friendFileName, ["eventWeight"])
    )
  # attach friend trees to data tree
  dataTChain   = ROOT.TChain(treeName)
  weightTChain = ROOT.TChain(treeName)
  for dataFileName in [dataSigRegionFileName, dataBkgRegionFileName]:
    print(f"Reading real data from '{dataFileName}'")
    dataTChain.Add(dataFileName)
    weightTChain.Add(f"{dataFileName}.weights")
  dataTChain.AddFriend(weightTChain)
  realData = ROOT.RDataFrame(dataTChain)
  lvBeamPhoton, lvTargetProton, lvRecoilProton, lvPip, lvPim = lorentzVectors(realData = True)
  for pairLabel, pairLvs, lvRecoil, flipYAxis in (
    ("PiPi", (lvPip, lvPim         ), lvRecoilProton, True ),
    ("PipP", (lvPip, lvRecoilProton), lvPim,          False),
    ("PimP", (lvPim, lvRecoilProton), lvPip,          False),
  ):  # loop over two-body subsystems of pi+ pi- p final state
    df = defineAngleFormulas(
      realData,
      lvBeamPhoton, lvRecoil, pairLvs[0], pairLvs[1],
      frame     = "Hf",
      flipYAxis = flipYAxis,
    )
    outFileName = f"data_flat.{pairLabel}.root"
    outTreeName = pairLabel
    print(f"Writing real data to tree '{outTreeName}' in '{outFileName}'")
    df.Define("mass",   f"(Double32_t)massPair({pairLvs[0]}, {pairLvs[1]})") \
      .Define("minusT", f"(Double32_t)-mandelstamT({lvTargetProton}, {lvRecoilProton})") \
      .Snapshot(outTreeName, outFileName, columnsToWrite + ["eventWeight"])

  # convert MC data
  lvBeamPhoton, lvTargetProton, lvRecoilProton, lvPip, lvPim = lorentzVectors(realData = False)
  for inFileName, outFileBaseName in [(phaseSpaceAccFileName, "phaseSpace_acc_flat"), (phaseSpaceGenFileName, "phaseSpace_gen_flat")]:
    print(f"Reading MC data from '{inFileName}'")
    mcData = ROOT.RDataFrame(treeName, inFileName)
    for pairLabel, pairLvs, lvRecoil, flipYAxis in (
      ("PiPi", (lvPip, lvPim         ), lvRecoilProton, True ),
      ("PipP", (lvPip, lvRecoilProton), lvPim,          False),
      ("PimP", (lvPim, lvRecoilProton), lvPip,          False),
    ):  # loop over two-body subsystems of pi+ pi- p final state
      df = defineAngleFormulas(
        mcData,
        lvBeamPhoton, lvRecoil, pairLvs[0], pairLvs[1],
        frame     = "Hf",
        flipYAxis = flipYAxis,
      )
      outFileName = f"{outFileBaseName}.{pairLabel}.root"
      outTreeName = pairLabel
      print(f"Writing MC data to tree '{outTreeName}' in '{outFileName}'")
      df.Define("mass",   f"(Double32_t)massPair({pairLvs[0]}, {pairLvs[1]})") \
        .Define("minusT", f"(Double32_t)-mandelstamT({lvTargetProton}, {lvRecoilProton})") \
        .Snapshot(outTreeName, outFileName, columnsToWrite)
