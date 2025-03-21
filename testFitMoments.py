#!/usr/bin/env python3
"""
Test script that performs closure test for moment analysis that uses
extended maximum likelihood-fitting of the data.
"""


from __future__ import annotations

import functools
import numpy as np
import nptyping as npt
import os
import threadpoolctl


import ROOT
import iminuit as im
from wurlitzer import pipes, STDOUT

from MomentCalculator import (
  AmplitudeSet,
  AmplitudeValue,
  MomentResult,
  QnWaveIndex,
)
from PlottingUtilities import (
  drawTF3,
  HistAxisBinning,
  setupPlotStyle,
)
import RootUtilities  # importing initializes OpenMP and loads `basisFunctions.C`
from testMomentsPhotoProd import (
  genDataFromWaves,
  TH3_ANG_PLOT_KWARGS,
)
import Utilities


# always flush print() to reduce garbling of log files due to buffering
print = functools.partial(print, flush = True)


if __name__ == "__main__":
  # set parameters of test case
  nmbPwaMcEvents   = 1000000 # number of "data" events to generate from partial-wave amplitudes
  nmbPsMcEvents    = 100000  # number of phase-space events to generate
  beamPolarization = 1.0     # polarization of photon beam
  maxL             = 4       # maximum L quantum number of moments
  outputDirName    = Utilities.makeDirPath("./plotsTestFitMoments")

  thisSourceFileName = os.path.basename(__file__)
  # logFileName = f"{outputDirName}/{os.path.splitext(thisSourceFileName)[0]}.log"
  # print(f"Writing output to log file '{logFileName}'")
  # with open(logFileName, "w") as logFile, pipes(stdout = logFile, stderr = STDOUT):  # redirect all output into log file
  if True:
    print(f"Using iminuit version {im.__version__}")
    Utilities.printGitInfo()
    timer = Utilities.Timer()
    ROOT.gROOT.SetBatch(True)
    ROOT.gRandom.SetSeed(1234567890)
    setupPlotStyle()
    threadController = threadpoolctl.ThreadpoolController()  # at this point all multi-threading libraries must be loaded
    print(f"Initial state of ThreadpoolController before setting number of threads:\n{threadController.info()}")
    with threadController.limit(limits = 4):
      print(f"State of ThreadpoolController after setting number of threads:\n{threadController.info()}")
      timer.start("Total execution time")

      # define angular distribution of signal
      partialWaveAmplitudesSig: tuple[AmplitudeValue, ...] = (  # set of all possible partial waves up to ell = 2
        # negative-reflectivity waves
        AmplitudeValue(QnWaveIndex(refl = -1, l = 0, m =  0), val =  1.0 + 0.0j),  # S_0^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = -1), val = -0.4 + 0.1j),  # P_-1^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m =  0), val =  0.3 - 0.8j),  # P_0^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 1, m = +1), val = -0.8 + 0.7j),  # P_+1^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -2), val =  0.1 - 0.4j),  # D_-2^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = -1), val =  0.5 + 0.2j),  # D_-1^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m =  0), val = -0.1 - 0.2j),  # D_ 0^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +1), val =  0.2 - 0.1j),  # D_+1^-
        AmplitudeValue(QnWaveIndex(refl = -1, l = 2, m = +2), val = -0.2 + 0.3j),  # D_+2^-
        # positive-reflectivity waves
        AmplitudeValue(QnWaveIndex(refl = +1, l = 0, m =  0), val =  0.5 + 0.0j),  # S_0^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = -1), val =  0.5 - 0.1j),  # P_-1^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m =  0), val = -0.8 - 0.3j),  # P_0^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 1, m = +1), val =  0.6 + 0.3j),  # P_+1^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -2), val =  0.2 + 0.1j),  # D_-2^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = -1), val =  0.2 - 0.3j),  # D_-1^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m =  0), val =  0.1 - 0.2j),  # D_ 0^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +1), val =  0.2 + 0.5j),  # D_+1^+
        AmplitudeValue(QnWaveIndex(refl = +1, l = 2, m = +2), val = -0.3 - 0.1j),  # D_+2^+
      )
      amplitudeSetSig = AmplitudeSet(partialWaveAmplitudesSig)

      # calculate true moment values and generate data from partial-wave amplitudes
      HTruth: MomentResult = amplitudeSetSig.photoProdMomentSet(maxL)
      print(f"True moment values\n{HTruth}")
      with timer.timeThis("Time to generate MC data from partial waves"):
        dataPwaModel = genDataFromWaves(
          nmbEvents         = nmbPwaMcEvents,
          polarization      = beamPolarization,
          amplitudeSet      = amplitudeSetSig,
          efficiencyFormula = None,
          regenerateData    = True,
          outFileNamePrefix = f"{outputDirName}/",
        )
        # plot data generated from partial-wave amplitudes
        canv = ROOT.TCanvas()
        nmbBins = 25
        hist = dataPwaModel.Histo3D(
          ROOT.RDF.TH3DModel("hData", ";cos#theta;#phi [deg];#Phi [deg]", nmbBins, -1, +1, nmbBins, -180, +180, nmbBins, -180, +180),
          "cosTheta", "phiDeg", "PhiDeg")
        hist.SetMinimum(0)
        hist.GetXaxis().SetTitleOffset(1.5)
        hist.GetYaxis().SetTitleOffset(2)
        hist.GetZaxis().SetTitleOffset(1.5)
        hist.Draw("BOX2Z")
        canv.SaveAs(f"{outputDirName}/{hist.GetName()}.pdf")

      # construct and draw intensity function with moments as parameters from formula
      # formula uses variables: x = cos(theta) in [-1, +1]; y = phi in [-180, +180] deg; z = Phi in [-180, +180] deg
      intensityFormula = HTruth.intensityFormula(
        polarization     = beamPolarization,
        thetaFormula     = "std::acos(x)",
        phiFormula       = "TMath::DegToRad() * y",
        PhiFormula       = "TMath::DegToRad() * z",
        printFormula     = True,
        useMomentSymbols = True,
      )
      intensityFcn = ROOT.TF3("intensityMoments", intensityFormula, -1, +1, -180, +180, -180, +180)
      for qnIndex in HTruth.indices.qnIndices:
        Hval = HTruth[qnIndex].val
        print(f"{qnIndex.label} = {Hval}")
        intensityFcn.SetParameter(qnIndex.label, Hval.imag if qnIndex.momentIndex == 2 else Hval.real)
      intensityFcn.SetNpx(100)
      intensityFcn.SetNpy(100)
      intensityFcn.SetNpz(100)
      intensityFcn.SetMinimum(0)
      drawTF3(intensityFcn, **TH3_ANG_PLOT_KWARGS, pdfFileName = f"{outputDirName}/{intensityFcn.GetName()}.pdf")

      # construct vectorized intensity function using TFormula and OpenMP
      # formula uses variables: x = theta in [0, pi] rad; y = phi in [-pi, +pi] rad; z = Phi in [-pi, +pi] rad
      intensityFormula = HTruth.intensityFormula(
        polarization     = beamPolarization,
        thetaFormula     = "x",
        phiFormula       = "y",
        PhiFormula       = "z",
        printFormula     = True,
        useMomentSymbols = True,
      )
      ROOT.gInterpreter.ProcessLine(f'TFormula intensityFormula = TFormula("intensity", "{intensityFormula}");')
      ROOT.gInterpreter.ProcessLine(
        """
        std::vector<double>
        intensityFcnVectorized(
          const std::vector<double>& thetas,  // polar angles of analyzer [rad]
          const std::vector<double>& phis,    // azimuthal angles of analyzer [rad]
          const std::vector<double>& Phis,    // azimuthal angle of beam-polarization vector w.r.t. production plane [rad]
          const std::vector<double>& moments
        ) {
          const size_t nmbEvents = thetas.size();
          assert(phis.size() == nmbEvents);
          assert(Phis.size() == nmbEvents);
          auto intensities = std::vector<double>(nmbEvents);
          // multi-threaded loop over events using OpenMP
          #pragma omp parallel for
          for (size_t i = 0; i < nmbEvents; ++i) {
            const double angles[3] = {thetas[i], phis[i], Phis[i]};
            intensities[i] = intensityFormula.EvalPar(angles, moments.data());
          }
          return intensities;
        }
        """
      )
      def intensityFcnVectorized(
        dataPoints: tuple[
          npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
          npt.NDArray[npt.Shape["nmbEvents"], npt.Float64],
          npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]
        ],
        moments: npt.NDArray[npt.Shape["nmbMoments"], npt.Float64],
      ) -> tuple[float, npt.NDArray[npt.Shape["nmbEvents"], npt.Float64]]:
        """Wrapper function that calls the vectorized intensity function defined in C++"""
        thetas, phis, Phis = dataPoints
        intensities = np.array(ROOT.intensityFcnVectorized(
          np.ascontiguousarray(thetas),
          np.ascontiguousarray(phis),
          np.ascontiguousarray(Phis),
          moments,
        ))
        return (0.0, intensities)
      momentValues = np.array([HTruth[qnIndex].val.imag if qnIndex.momentIndex == 2 else HTruth[qnIndex].val.real for qnIndex in HTruth.indices.qnIndices])
      momentLabels = tuple(qnIndex.label for qnIndex in HTruth.indices.qnIndices)
      thetas = np.array([0,    1,    2],    np.double)
      phis   = np.array([0.5,  1.5,  2.5],  np.double)
      Phis   = np.array([0.75, 1.75, 2.75], np.double)
      print(f"{intensityFcnVectorized(dataPoints = (thetas, phis, Phis), moments = momentValues)=}")
      # for parIndex in range(len(momentValues)):
      #   intensityFcn.SetParameter(momentLabels[parIndex], momentValues[parIndex])
      for theta, phi, Phi in zip(thetas, phis, Phis):
        print(f"{intensityFcn.Eval(np.cos(theta), np.rad2deg(phi), np.rad2deg(Phi))=}")

      timer.stop("Total execution time")
      print(timer.summary)
