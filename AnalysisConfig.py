"""
This module defines the input parameters for the various moment analyses
"""

from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
)

from MomentCalculator import KinematicBinningVariable
from PlottingUtilities import HistAxisBinning
import Utilities


@dataclass
class AnalysisConfig:
  """Stores configuration parameters for the moment analysis; defaults are for unpolarized production"""
  # defaults are for unpolarized pipi analysis
  treeName:                 str                      = "PiPi"  # name of tree to read from data and MC files
  dataFileName:             str                      = "./dataPhotoProdPiPiUnpol/data_flat.PiPi.root"  # file with real data to analyze
  psAccFileName:            str                      = "./dataPhotoProdPiPiUnpol/phaseSpace_acc_flat.PiPi.root"  # file with accepted phase-space MC
  psGenFileName:            str | None               = "./dataPhotoProdPiPiUnpol/phaseSpace_gen_flat.PiPi.root"  # file with generated phase-space MC
  polarization:             float | str | None       = None  # photon-beam polarization; None = unpolarized photoproduction; polarized photoproduction: either polarization value or name of polarization
  _maxL:                    int                      = 8  # maximum L of physical and measured moments
  _outFileDirBaseName:      str                      = "./plotsPhotoProdPiPiUnpol"  # base name of directory into which all output will be written
  outFileDirName:           str                      = field(init = False)  # directory into which all output will be written
  outFileNamePrefix:        str                      = field(init = False)  # name prefix prepended to output file names
  # _normalizeMoments:        bool                     = True
  _normalizeMoments:        bool                     = False
  nmbBootstrapSamples:      int                      = 0
  # nmbBootstrapSamples:      int                      = 10000
  # plotAngularDistributions: bool                     = True
  plotAngularDistributions: bool                     = False
  # plotAccIntegralMatrices:  bool                     = True
  plotAccIntegralMatrices:  bool                     = False
  # calcAccPsMoments:         bool                     = True
  calcAccPsMoments:         bool                     = False
  # plotAccPsMoments:         bool                     = True
  plotAccPsMoments:         bool                     = False
  plotMomentsInBins:        bool                     = True
  # plotMomentsInBins:        bool                     = False
  # plotMeasuredMoments:      bool                     = True
  plotMeasuredMoments:      bool                     = False
  # plotCovarianceMatrices:   bool                     = True
  plotCovarianceMatrices:   bool                     = False
  limitNmbPsAccEvents:      int                      = 0
  # limitNmbPsAccEvents:      int                      = 100000
  binVarMass:               KinematicBinningVariable = field(default_factory=lambda: KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ))
  massBinning:              HistAxisBinning          = field(default_factory=lambda: HistAxisBinning(nmbBins = 100, minVal = 0.4, maxVal = 1.4))  # same binning as used by CLAS

  def __post_init__(self) -> None:
    """Creates output directory and initializes member variables"""
    #TODO move creation of directory to separate function
    self.outFileDirName    = Utilities.makeDirPath(f"{self._outFileDirBaseName}.maxL_{self.maxL}")
    self.outFileNamePrefix = "norm" if self.normalizeMoments else "unnorm"
    self.massBinning.var   = self.binVarMass

  @property
  def maxL(self) -> int:
    return self._maxL

  @maxL.setter
  def maxL(
    self,
    value: int
  ) -> None:
    assert value > 0, f"maxL must be > 0, but is {value}"
    self._maxL = value
    self.__post_init__()

  @property
  def outFileDirBaseName(self) -> str:
    return self._outFileDirBaseName

  @outFileDirBaseName.setter
  def outFileDirBaseName(
    self,
    value: str
  ) -> None:
    self._outFileDirBaseName = value
    self.__post_init__()

  @property
  def normalizeMoments(self) -> bool:
    return self._normalizeMoments

  @normalizeMoments.setter
  def normalizeMoments(
    self,
    value: bool
  ) -> None:
    self._normalizeMoments = value
    self.__post_init__()


# configurations for unpolarized pi+ pi- data
CFG_UNPOLARIZED_PIPI_CLAS = AnalysisConfig()
CFG_UNPOLARIZED_PIPI_PWA  = AnalysisConfig(
  _outFileDirBaseName = "./plotsPhotoProdPiPiUnpolPwa",
  massBinning         = HistAxisBinning(nmbBins = 56, minVal = 0.28, maxVal = 1.40),  # binning used in PWA of unpolarized data
)
# configuration for polarized pi+ pi- data
CFG_POLARIZED_PIPI = AnalysisConfig(
  dataFileName        = "./dataPhotoProdPiPiPol/data_flat_0.0.root",
  psAccFileName       = "./dataPhotoProdPiPiPol/phaseSpace_acc_flat.root",
  psGenFileName       = "./dataPhotoProdPiPiPol/phaseSpace_gen_flat.root",
  polarization        = "beamPol", # read polarization from tree column
  _maxL               = 4,
  _outFileDirBaseName = "./plotsPhotoProdPiPiPol",
  massBinning         = HistAxisBinning(nmbBins = 50, minVal = 0.28, maxVal = 2.28),  # binning used in PWA of polarized data
)
# configuration for unpolarized pi+ p data
CFG_UNPOLARIZED_PIPP = AnalysisConfig(
  dataFileName        = "./dataPhotoProdPiPiUnpol/data_flat.PipP.root",
  psAccFileName       = "./dataPhotoProdPiPiUnpol/phaseSpace_acc_flat.PipP.root",
  psGenFileName       = "./dataPhotoProdPiPiUnpol/phaseSpace_gen_flat.PipP.root",
  _maxL               = 4,
  _outFileDirBaseName = "./plotsPhotoProdPipPUnpol",
  binVarMass          = KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{#pi}^{#plus}#it{p}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ),
  massBinning         = HistAxisBinning(nmbBins = 75, minVal = 1.1, maxVal = 2.6),
)
# configuration for Nizar's polarized eta pi0 data
CFG_NIZAR = AnalysisConfig(
  treeName            = "etaPi0",
  dataFileName        = "./dataTestNizar/data_flat.root",
  psAccFileName       = "./dataTestNizar/phaseSpace_acc_flat.root",
  psGenFileName       = None,
  polarization        = "beamPol", # read polarization from tree column
  _maxL               = 4,
  _outFileDirBaseName = "./plotsTestNizar",
  # _normalizeMoments   = True,
  binVarMass          = KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{#eta}#it{#pi}^{0}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ),
  massBinning         = HistAxisBinning(nmbBins = 17, minVal = 1.04, maxVal = 1.72),
)
# configuration for Kevin's K- K_S Delta++ data
CFG_KEVIN = AnalysisConfig(
  treeName                 = "ntFSGlueX_100_11100_angles",
  dataFileName             = "./dataPhotoProdKmKS/data/pipkmks_100_11100_B4_M16_*_SKIM_A2.root.angles",
  psAccFileName            = "./dataPhotoProdKmKS/phaseSpace/pipkmks_100_11100_B4_M16_SIGNAL_SKIM_A2.root.angles",
  psGenFileName            = "./dataPhotoProdKmKS/phaseSpace/pipkmks_100_11100_B4_M16_MCGEN_GENERAL_SKIM_A2.root.angles",
  polarization             = "beamPol", # read polarization from tree column
  _maxL                    = 4,
  _outFileDirBaseName      = "./plotsTestKevin",
  # _normalizeMoments        = True,
  # plotAngularDistributions = True,
  # plotAccIntegralMatrices  = True,
  binVarMass               = KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{K}^{#minus}#it{K}_{S}^{0}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ),
  massBinning              = HistAxisBinning(nmbBins = 20, minVal = 1.0, maxVal = 1.8),  # original binning: 200 bins in [0.6, 2.6] GeV
)
