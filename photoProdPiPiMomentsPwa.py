#!/usr/bin/env python3
"""
This module calculates the moment values from partial-wave amplitudes
obtained from unpolarized and polarized pi+ pi- photoproduction data.
The calculated moments are written to a file to be read by the
plotting script `photoProdPiPiPlotMoments.py`.

Usage: Run this module as a script to perform the moment calculations
and to generate the output files.
"""


from __future__ import annotations

from copy import deepcopy
import functools
import os
import pandas as pd
import threadpoolctl

import ROOT
from wurlitzer import pipes, STDOUT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  KinematicBinningVariable,
  MomentResult,
  MomentResultsKinematicBinning,
  QnWaveIndex,
)
from photoProdPiPiCalcMoments import (
  CFG_POLARIZED_PIPI,
  CFG_UNPOLARIZED_PIPI,
)
from PlottingUtilities import setupPlotStyle
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


def readMomentResultsPwa(
  dataFileName: str,
  maxL:         int,  # maximum L quantum number of moments
  waves:        list[tuple[str, QnWaveIndex]],  # wave labels and quantum numbers
  binVarMass:   KinematicBinningVariable,       # binning variable for mass bins
  normalize:    bool = False,
) -> MomentResultsKinematicBinning:
  """Reads the partial-amplitude values from the PWA fit and calculates the corresponding moments"""
  print(f"Reading partial-wave amplitude values from file '{dataFileName}'")
  waveLabels = [wave[0] for wave in waves]
  amplitudesDf = pd.read_csv(
    dataFileName,
    sep   = r"\s+",  # values are whitespace separated
    names = ["mass", ] + waveLabels,
  )
  # converts columns to correct types
  def strToComplex(s: str) -> complex:
    """Converts string of form '(float,float)' to complex"""
    real, imag = s.strip("()").split(",")
    return complex(float(real), float(imag))
  for wave in waves:
    amplitudesDf[wave[0]] = amplitudesDf[wave[0]].apply(strToComplex)
  # convert dataframe to MomentResultsKinematicBinning
  momentResults: list[MomentResult] = []
  for amplitudesRow in amplitudesDf.to_dict(orient = "records"):  # iterate over list of dictionaries
    massBinCenter = amplitudesRow["mass"]
    # unfortunately, pd.read_csv does not check whether number of columns in CSV file matches given column list
    # protect against case where file has more amplitude columns than given by `waves`
    assert isinstance(massBinCenter, float), f"Something is wrong: Expect float number for bin center but got {type(massBinCenter)}: {massBinCenter}"
    print(f"Reading partial-wave amplitudes for mass bin at {massBinCenter} GeV")
    amplitudeSet = AmplitudeSet(
      amps      = [AmplitudeValue(qn = wave[1], val = amplitudesRow[wave[0]]) for wave in waves],
      tolerance = 1e-8,
    )
    momentResults.append(
      amplitudeSet.photoProdMomentSet(
        maxL                = maxL,
        normalize           = normalize,
        printMomentFormulas = False,
        binCenters          = {binVarMass: massBinCenter},
      )
    )
  return MomentResultsKinematicBinning(momentResults)


if __name__ == "__main__":
  cfg = deepcopy(CFG_UNPOLARIZED_PIPI)  # perform unpolarized analysis
  # cfg = deepcopy(CFG_POLARIZED_PIPI)    # perform polarized analysis

  # for maxL in (2, 4, 5, 8, 10, 12, 20):
  for maxL in (8, ):
    print(f"Calculating moment values with L_max = {maxL} from partial-wave amplitudes")
    cfg.maxL = maxL
    thisSourceFileName = os.path.basename(__file__)
    logFileName = f"{cfg.outFileDirName}/{os.path.splitext(thisSourceFileName)[0]}_{cfg.outFileNamePrefix}.log"
    print(f"Writing output to log file '{logFileName}'")
    with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
      Utilities.printGitInfo()
      timer = Utilities.Timer()
      ROOT.gROOT.SetBatch(True)
      setupPlotStyle()
      threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
      print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
      with threadController.limit(limits = 4):
        print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")

        timer.start("Total execution time")

        pwaAmplitudesFileName = None
        waves: list[tuple[str, QnWaveIndex]] = []

        #TODO add this info to AnalysisConfig?
        if cfg.polarization is None:
          # unpolarized data
          pwaAmplitudesFileName = "./dataPhotoProdPiPiUnpol/PWA_S_P_D/amplitudes_range_tbin.txt"
          # pwaAmplitudesFileName = "./dataPhotoProdPiPiUnpol/PWA_S_P_D_F/amplitudes_new_SPDF.txt"
          waves = [  # order must match columns in file with partial-wave amplitudes
            ("S_0",  QnWaveIndex(refl = None, l = 0, m =  0)),
            # P-waves
            ("P_0",  QnWaveIndex(refl = None, l = 1, m =  0)),
            ("P_+1", QnWaveIndex(refl = None, l = 1, m = +1)),
            ("P_-1", QnWaveIndex(refl = None, l = 1, m = -1)),
            # D-waves
            ("D_0",  QnWaveIndex(refl = None, l = 2, m =  0)),
            ("D_+1", QnWaveIndex(refl = None, l = 2, m = +1)),
            ("D_+2", QnWaveIndex(refl = None, l = 2, m = +2)),
            ("D_-1", QnWaveIndex(refl = None, l = 2, m = -1)),
            ("D_-2", QnWaveIndex(refl = None, l = 2, m = -2)),
            # # F-waves
            # ("F_0",  QnWaveIndex(refl = None, l = 3, m =  0)),
            # ("F_+1", QnWaveIndex(refl = None, l = 3, m = +1)),
            # ("F_+2", QnWaveIndex(refl = None, l = 3, m = +2)),
            # ("F_+3", QnWaveIndex(refl = None, l = 3, m = +3)),
            # ("F_-1", QnWaveIndex(refl = None, l = 3, m = -1)),
            # ("F_-2", QnWaveIndex(refl = None, l = 3, m = -2)),
            # ("F_-3", QnWaveIndex(refl = None, l = 3, m = -3)),
          ]
        else:
          # polarized data
          pwaAmplitudesFileName = "./dataPhotoProdPiPiPol/PWA_S_P_D/amplitudes_SPD.txt"
          waves = [  # order must match columns in file with partial-wave amplitudes
            # S-waves
            ("S_0+",  QnWaveIndex(refl = +1, l = 0, m =  0)),
            ("S_0-",  QnWaveIndex(refl = -1, l = 0, m =  0)),
            # P-waves
            ("P_0+",  QnWaveIndex(refl = +1, l = 1, m =  0)),
            ("P_+1+", QnWaveIndex(refl = +1, l = 1, m = +1)),
            ("P_-1+", QnWaveIndex(refl = +1, l = 1, m = -1)),
            ("P_0-",  QnWaveIndex(refl = -1, l = 1, m =  0)),
            ("P_+1-", QnWaveIndex(refl = -1, l = 1, m = +1)),
            ("P_-1-", QnWaveIndex(refl = -1, l = 1, m = -1)),
            # D-waves
            ("D_0+",  QnWaveIndex(refl = +1, l = 2, m =  0)),
            ("D_+1+", QnWaveIndex(refl = +1, l = 2, m = +1)),
            ("D_+2+", QnWaveIndex(refl = +1, l = 2, m = +2)),
            ("D_-1+", QnWaveIndex(refl = +1, l = 2, m = -1)),
            ("D_-2+", QnWaveIndex(refl = +1, l = 2, m = -2)),
            ("D_0-",  QnWaveIndex(refl = -1, l = 2, m =  0)),
            ("D_+1-", QnWaveIndex(refl = -1, l = 2, m = +1)),
            ("D_+2-", QnWaveIndex(refl = -1, l = 2, m = +2)),
            ("D_-1-", QnWaveIndex(refl = -1, l = 2, m = -1)),
            ("D_-2-", QnWaveIndex(refl = -1, l = 2, m = -2)),
          ]

        print(f"Calculating PWA moments")
        momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_pwa_SPD.pkl"
        # momentResultsFileName = f"{cfg.outFileDirName}/{cfg.outFileNamePrefix}_moments_pwa_SPDF.pkl"
        momentResultsPwa: MomentResultsKinematicBinning = readMomentResultsPwa(
          dataFileName = pwaAmplitudesFileName,
          maxL         = cfg.maxL,
          waves        = waves,
          binVarMass   = cfg.binVarMass,
          normalize    = cfg.normalizeMoments
        )
        print(f"Writing PWA moments to file '{momentResultsFileName}'")
        momentResultsPwa.save(momentResultsFileName)

        timer.stop("Total execution time")
        print(timer.summary)
