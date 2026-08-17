#!/usr/bin/env python3
"""Converts moment results from a pickle file to a JSON file"""

from __future__ import annotations

import argparse

from moments.MomentCalculator import MomentResultsKinematicBinning
from workflow.JsonUtilities import toJsonStr


def main(args: argparse.Namespace) -> None:
  print(f"Converting file '{args.pickle_file_path}' to JSON file '{args.json_file_path}'")
  momentResults = MomentResultsKinematicBinning.loadPickle(args.pickle_file_path)
  with open(file = args.json_file_path, mode = "w", encoding = "utf-8") as jsonFile:
    jsonFile.write(toJsonStr(momentResults, indent = 2) + "\n")  #TODO extend this to write full information in `MomentResultsKinematicBinning`


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description = "Converts moment results from a pickle file to a JSON file."
  )
  parser.add_argument("--pickle-file-path", type = str,                             help = "Path to the input pickle file")
  parser.add_argument("--json-file-path",   type = str, default = "./moments.json", help = "Path to the output JSON file; default: '%(default)s'")
  args = parser.parse_args()
  main(args)
