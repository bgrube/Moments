#!/usr/bin/env python3


from __future__ import annotations

import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)

  # data for lowest t bin [0.1, 0.2] GeV^2
  dataFileName          = "./polALL_t010020_m104172_Phase1_DTOT_selected_nominal_wPhotonSyst_acc_flat.root"
  phaseSpaceAccFileName = "./polALL_t010020_m104172_Phase1_FTOT_selected_nominal_wPhotonSyst_acc_flat.root"
  inTreeName            = "kin"
  outTreeName           = "etaPi0"

  # convert data
  for inFileName, outFileName in ((dataFileName, "data_flat.root"), (phaseSpaceAccFileName, "phaseSpace_acc_flat.root")):
    print(f"Reading file '{inFileName}' and writing data to file '{outFileName}'")
    df = (
      ROOT.RDataFrame(inTreeName, inFileName)
          .Define("beamPol",     "(Double32_t)Pol")
          .Define("mass",        "(Double32_t)Mpi0eta")
          .Define("cosTheta",    "(Double32_t)cosTheta_eta_gj")
          .Define("theta",       "(Double32_t)std::acos(cosTheta)")
          .Define("phi",         "(Double32_t)phi_eta_gj")
          .Define("phiDeg",      "(Double32_t)phi * TMath::RadToDeg()")
          .Define("PhiDeg",      "(Double32_t)Phi * TMath::RadToDeg()")
          .Define("eventWeight", "(Double32_t)Weight")
    )
    df.Snapshot(outTreeName, outFileName, ("beamPol", "mass", "cosTheta", "theta", "phi", "phiDeg", "Phi", "PhiDeg", "eventWeight"))
