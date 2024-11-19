#!/usr/bin/env python3


import ROOT


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)

  # data for lowest t bin [0.1, 0.2] GeV^2
  fractionOfEventsToKeep = 0.1
  inFileName             = "data_flat.root"
  outFileName            = f"data_flat_downsampled_{fractionOfEventsToKeep}.root"
  treeName               = "PiPi"

  nmbEventsBefore = ROOT.RDataFrame(treeName, inFileName).Count().GetValue()
  nmbEventsAfter  = (
    ROOT.RDataFrame(treeName, inFileName)
      .Define("acceptEvent", f"(bool)(gRandom->Rndm() < {fractionOfEventsToKeep})")
      .Filter("acceptEvent == true")
      .Snapshot(treeName, outFileName)
      .Count().GetValue()
  )
  print(f"Randomly selected {nmbEventsAfter} out of {nmbEventsBefore} events (= {100 * nmbEventsAfter / nmbEventsBefore}%) and wrote them to '{outFileName}'")
