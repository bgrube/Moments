#!/usr/bin/env python3
"""
This script merges signal and background trees with event weights into a single output tree.
"""


from __future__ import annotations

import argparse
import functools

import ROOT

import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  timer = Utilities.Timer()
  timer.start("Total execution time")
  ROOT.gROOT.SetBatch(True)

  parser = argparse.ArgumentParser(description = "Merges signal and background trees with event weights into a single output tree.")
  parser.add_argument("--sigFilePath",         type = str,                          help = "Path to ROOT file with signal events")
  parser.add_argument("--bkgFilePath",         type = str,                          help = "Path to ROOT file with background events")
  parser.add_argument("--inTreeName",          type = str,                          help = "Name of tree in input files")
  parser.add_argument("--outFileName",         type = str,                          help = "Path of output ROOT file")
  parser.add_argument("--outTreeName",         type = str, default = None,          help = "Name of the output tree (default: name of input tree)")
  parser.add_argument("--sigWeightFormula",    type = str, default = "1.0",         help = "Formula for calculating the event weight for signal events (default: '%(default)s')")
  parser.add_argument("--bkgWeightFormula",    type = str, default = "-1.0",        help = "Formula for calculating the event weight for background events (default: '%(default)s')")
  parser.add_argument("--weightColNameOutput", type = str, default = "eventWeight", help = "Name of column in output tree that contains the event weight (default: '%(default)s')")
  args = parser.parse_args()
  if args.outTreeName is None:
    args.outTreeName = args.inTreeName
  Utilities.print_command_line_arguments(args)

  mergedDf: ROOT.RDataFrame = Utilities.getDataFrameWithCorrectEventWeights(
    dataSigRegionFileNames  = (args.sigFilePath, ),
    dataBkgRegionFileNames  = (args.bkgFilePath, ),
    treeName                = args.inTreeName,
    sigRegionWeightFormula  = args.sigWeightFormula,
    bkgRegionWeightFormula  = args.bkgWeightFormula,
    weightColNameOutput     = args.weightColNameOutput,
  )
  print(f"Writing merged tree '{args.outTreeName}' to file '{args.outFileName}'")
  mergedDf = mergedDf.Snapshot(args.outTreeName, args.outFileName)

  timer.stop("Total execution time")
  print(timer.summary)
