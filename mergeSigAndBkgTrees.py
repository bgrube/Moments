#!/usr/bin/env python3
"""
This module converts input data into the format expected by `MomentCalculator`.

Usage: Run this module as a script to convert input data files.
"""


from __future__ import annotations

import functools

import ROOT

from makeMomentsInputTree import getDataFrameWithCorrectEventWeights
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  Utilities.printGitInfo()
  timer = Utilities.Timer()
  timer.start("Total execution time")
  ROOT.gROOT.SetBatch(True)
  ROOT.EnableImplicitMT()

  dataRootDir      = "./dataPhotoProdEtapEta/unpolarized/2018_08/ALLT/Will"
  # sigFilePath      = f"{dataRootDir}/tree_signal_Unpol.root"
  # bkgFilePath      = f"{dataRootDir}/tree_bkgnd_Unpol.root"
  # outFileName      = f"{dataRootDir}/tree_data_Unpol.root"
  sigFilePath      = f"{dataRootDir}/tree_accepted_signal_Unpol.root"
  bkgFilePath      = f"{dataRootDir}/tree_accepted_bkgnd_Unpol.root"
  outFileName      = f"{dataRootDir}/tree_accepted_Unpol.root"
  inTreeName       = "nt"  # name of tree in input files
  sigWeightFormula = "1.0"   # formula for calculating event weight for signal events
  bkgWeightFormula = "-1.0"  # formula for calculating event weight for background events
  outTreeName      = "kin"  # name of the output tree
  outWeightColName = "eventWeight"  # name of column in output tree that contains the event weight

  mergedDf: ROOT.RDataFrame = getDataFrameWithCorrectEventWeights(
    dataSigRegionFileNames  = (sigFilePath, ),
    dataBkgRegionFileNames  = (bkgFilePath, ),
    treeName                = inTreeName,
    sigRegionWeightFormula  = sigWeightFormula,
    bkgRegionWeightFormula  = bkgWeightFormula,
  )
  print(f"Writing merged tree '{outTreeName}' to file '{outFileName}'")
  mergedDf = mergedDf.Snapshot(outTreeName, outFileName)

  timer.stop("Total execution time")
  print(timer.summary)
