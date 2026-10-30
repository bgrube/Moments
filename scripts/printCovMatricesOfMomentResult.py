#!/usr/bin/env python3
"""Converts moment results from a pickle file to a JSON file"""

from __future__ import annotations

import argparse

from moments.MomentCalculator import MomentResultsKinematicBinning


def printCovMatrix(
  matrix,
  nmbElements = 10,
  label = "covariance matrix"
) -> None:
  print(f"Printing first {nmbElements}x{nmbElements} elements of {label}")
  for i in range(nmbElements):
    row = ""
    for j in range(nmbElements):
      row += "         " if j < i else f"{matrix[i][j]:8.1f} "
    print(row)


def main(args: argparse.Namespace) -> None:
  momentResults = MomentResultsKinematicBinning.loadPickle(args.pickle_file_path)
  printCovMatrix(momentResults[0]._V_ReReFlatIndex, args.nmb_elements, "_V_ReReFlatIndex")
  printCovMatrix(momentResults[0]._V_ImImFlatIndex, args.nmb_elements, "_V_ImImFlatIndex")
  printCovMatrix(momentResults[0]._V_ReImFlatIndex, args.nmb_elements, "_V_ReImFlatIndex")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description = "Prints the first N x N elements of the covariance matrices of a moment result from a pickle file."
  )
  parser.add_argument("--pickle-file-path", type = str,               help = "Path to the input pickle file")
  parser.add_argument("--nmb-elements",     type = int, default = 10, help = "Number of elements N to print from the covariance matrices; default: '%(default)s'")
  args = parser.parse_args()
  main(args)
