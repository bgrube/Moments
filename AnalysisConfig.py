"""
This module defines the input parameters for the various moment analyses
"""

from __future__ import annotations

from dataclasses import (
  dataclass,
  field,
)
from enum import Enum, auto

import ROOT

from MomentCalculator import KinematicBinningVariable
from PlottingUtilities import HistAxisBinning
import Utilities


@dataclass
class AnalysisConfig:
  """Stores configuration parameters for the moment analysis; defaults are for unpolarized production"""

  class MethodType(Enum):
    """Enumerates methods used for moment analysis"""
    LIN_ALG_BG_SUBTR_NEG_WEIGHTS = auto()  # linear-algebra method for acceptance correction + background subtraction by negative event weights
    LIN_ALG_BG_SUBTR_MOMENTS     = auto()  # linear-algebra method for acceptance correction + background subtraction by subtracting moments
    MAX_LIKELIHOOD_FIT           = auto()  # maximum-likelihood fit for acceptance correction + background subtraction by subtracting moments

  # defaults are for unpolarized pipi analysis
  method:                   AnalysisConfig.MethodType = MethodType.LIN_ALG_BG_SUBTR_NEG_WEIGHTS  # method used to estimate moments from data
  treeName:                 str                       = "PiPi"  # name of tree to read from data and MC files
  dataFileName:             str                       = "./dataPhotoProdPiPiUnpol/data_flat.PiPi.root"  # file with real data to analyze
  psAccFileName:            str | None                = "./dataPhotoProdPiPiUnpol/phaseSpace_acc_flat.PiPi.root"  # file with accepted phase-space MC
  psGenFileName:            str | None                = "./dataPhotoProdPiPiUnpol/phaseSpace_gen_flat.PiPi.root"  # file with generated phase-space MC
  polarization:             float | str | None        = None  # photon-beam polarization; None = unpolarized photoproduction; polarized photoproduction: either polarization value or name of polarization
  maxL:                     int                       = 8  # maximum L of physical and measured moments
  outFileDirBaseName:       str                       = "./plotsPhotoProdPiPiUnpolCLAS"  # base name of directory into which all output will be written
  # normalizeMoments:         bool                      = True
  normalizeMoments:         bool                      = False
  nmbBootstrapSamples:      int                       = 0
  # nmbBootstrapSamples:      int                       = 10000
  # plotAngularDistributions: bool                      = True
  plotAngularDistributions: bool                      = False
  # plotAccIntegralMatrices:  bool                      = True
  plotAccIntegralMatrices:  bool                      = False
  # calcAccPsMoments:         bool                      = True
  calcAccPsMoments:         bool                      = False
  # plotAccPsMoments:         bool                      = True
  plotAccPsMoments:         bool                      = False
  plotMomentsInBins:        bool                      = True
  # plotMomentsInBins:        bool                      = False
  # plotMeasuredMoments:      bool                      = True
  plotMeasuredMoments:      bool                      = False
  # plotCovarianceMatrices:   bool                      = True
  plotCovarianceMatrices:   bool                      = False
  limitNmbPsAccEvents:      int                       = 0
  # limitNmbPsAccEvents:      int                       = 100000
  binVarMass:               KinematicBinningVariable  = field(
    default_factory = lambda: KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    )
  )
  massBinning:              HistAxisBinning           = field(
    default_factory = lambda: HistAxisBinning(  # same binning as used by CLAS
      nmbBins = 100,
      minVal  = 0.4,   # [GeV]
      maxVal  = 1.4,   # [GeV]
      _var    = None,  # set by init()
    )
  )

  @property
  def outFileDirName(self) -> str:
    """Returns name of directory into which all output will be written"""
    return f"{self.outFileDirBaseName}.maxL_{self.maxL}"

  @property
  def outFileNamePrefix(self) -> str:
    """Returns name prefix prepended to output file names"""
    return "norm" if self.normalizeMoments else "unnorm"

  def init(
    self,
    createOutFileDir: bool = False,
  ) -> None:
    """Creates output directory and initializes member variables; needs to be called before passing the config to any consumer"""
    self.massBinning.var = self.binVarMass
    if createOutFileDir:
      Utilities.makeDirPath(self.outFileDirName)

  class DataType(Enum):
    """Enumerates used input data types"""
    REAL_DATA             = auto()
    REAL_DATA_SIGNAL      = auto()  # real data in signal region
    REAL_DATA_SIDEBAND    = auto()  # real data in sideband region(s)
    GENERATED_PHASE_SPACE = auto()
    ACCEPTED_PHASE_SPACE  = auto()

  def loadData(
    self,
    dataType: AnalysisConfig.DataType,
  ) -> ROOT.RDataFrame | None:
    """Returns a ROOT RDataFrame with given data type"""
    if dataType == AnalysisConfig.DataType.REAL_DATA:
      print(f"Loading real data from tree '{self.treeName}' in file '{self.dataFileName}'")
      return ROOT.RDataFrame(self.treeName, self.dataFileName)
    elif dataType == AnalysisConfig.DataType.REAL_DATA_SIGNAL:
      print(f"Loading real-data signal events with weight = 1 from tree '{self.treeName}' in file '{self.dataFileName}'")
      return ROOT.RDataFrame(self.treeName, self.dataFileName).Filter("eventWeight == 1")
    elif dataType == AnalysisConfig.DataType.REAL_DATA_SIDEBAND:
      df = ROOT.RDataFrame(self.treeName, self.dataFileName).Filter("eventWeight < 0")
      if df.Count().GetValue() < 1:
        return None
      else:
        print(f"Loading real-data sideband events with weight < 0 from tree '{self.treeName}' in file '{self.dataFileName}'")
        return df
      # # RDataFrame::Redefine() introduced only for ROOT V6.26+
      # # quick hack to ensure that data in background region have positive weight by dropping the eventWeight column and re-adding it with opposite value
      # data.Define("eventWeight2", "-eventWeight")
      # data.Snapshot(self.treeName, f"{self.outFileDirName}/foo.root", ("beamPol", "beamPolPhi", "cosTheta", "theta", "phi", "phiDeg", "Phi", "PhiDeg", "mass", "minusT", "eventWeight2"))
      # data = ROOT.RDataFrame(self.treeName, f"{self.outFileDirName}/foo.root")
      # data = dataPwaModelBkgRegion.Define("eventWeight", "eventWeight2")
    elif dataType == AnalysisConfig.DataType.GENERATED_PHASE_SPACE:
      if self.psGenFileName is None:
        print("??? Warning: File name for generated phase-space data was not provided. Acceptance may not be calculated correctly.")
        return None
      else:
        print(f"Loading generated phase-space data from tree '{self.treeName}' in file '{self.psGenFileName}'")
        return ROOT.RDataFrame(self.treeName, self.psGenFileName)
    elif dataType == AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE:
      if self.psAccFileName is None:
        print("??? Warning: File name for accepted phase-space data was not provided. Assuming perfect acceptance.")
        return None
      else:
        print(f"Loading accepted phase-space data from tree '{self.treeName}' in file '{self.psAccFileName}'")
        return ROOT.RDataFrame(self.treeName, self.psAccFileName)
    else:
      raise ValueError(f"Unknown data type: {dataType}")


# configurations for unpolarized pi+ pi- data in CLAS kinematic range
CFG_UNPOLARIZED_PIPI_CLAS = AnalysisConfig()
CFG_UNPOLARIZED_PIPI_PWA  = AnalysisConfig(
  outFileDirBaseName = "./plotsPhotoProdPiPiUnpolPwa",
  massBinning        = HistAxisBinning(nmbBins = 56, minVal = 0.28, maxVal = 1.40),  # binning used in PWA of unpolarized data
)
CFG_UNPOLARIZED_PIPI_JPAC = AnalysisConfig(
  dataFileName       = "./dataPhotoProdPiPiUnpolJPAC/mc_full/tbin_0.4_0.5/data_flat.PiPi.root",  # file with real data to analyze
  psAccFileName      = None,  # no file with accepted phase-space MC
  psGenFileName      = None,  # no file with generated phase-space MC
  outFileDirBaseName = "./plotsPhotoProdPiPiUnpolJPAC",
  # massBinning        = HistAxisBinning(nmbBins = 25, minVal = 0.4, maxVal = 1.40),
  # massBinning        = HistAxisBinning(nmbBins = 2, minVal = 0.4, maxVal = 0.42),
)
# configuration for polarized pi+ pi- data
CFG_POLARIZED_PIPI = AnalysisConfig(
  dataFileName       = "./dataPhotoProdPiPiPol/data_flat_0.0.root",
  psAccFileName      = "./dataPhotoProdPiPiPol/phaseSpace_acc_flat.root",
  psGenFileName      = "./dataPhotoProdPiPiPol/phaseSpace_gen_flat.root",
  polarization       = "beamPol", # read polarization from tree column
  maxL               = 4,
  outFileDirBaseName = "./plotsPhotoProdPiPiPol",
  massBinning        = HistAxisBinning(nmbBins = 50, minVal = 0.28, maxVal = 2.28),  # binning used in PWA of polarized data
)
# configuration for unpolarized pi+ p data
CFG_UNPOLARIZED_PIPP = AnalysisConfig(
  dataFileName       = "./dataPhotoProdPiPiUnpol/data_flat.PipP.root",
  psAccFileName      = "./dataPhotoProdPiPiUnpol/phaseSpace_acc_flat.PipP.root",
  psGenFileName      = "./dataPhotoProdPiPiUnpol/phaseSpace_gen_flat.PipP.root",
  maxL               = 4,
  outFileDirBaseName = "./plotsPhotoProdPipPUnpol",
  binVarMass         = KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{#pi}^{#plus}#it{p}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ),
  massBinning        = HistAxisBinning(nmbBins = 75, minVal = 1.1, maxVal = 2.6),
)
# configuration for Nizar's polarized eta pi0 data
CFG_NIZAR = AnalysisConfig(
  treeName           = "etaPi0",
  dataFileName       = "./dataTestNizar/data_flat.root",
  psAccFileName      = "./dataTestNizar/phaseSpace_acc_flat.root",
  psGenFileName      = None,
  polarization       = "beamPol", # read polarization from tree column
  maxL               = 4,
  outFileDirBaseName = "./plotsTestNizar",
  # normalizeMoments   = True,
  binVarMass         = KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{#eta}#it{#pi}^{0}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ),
  massBinning        = HistAxisBinning(nmbBins = 17, minVal = 1.04, maxVal = 1.72),
)
# configuration for Kevin's K- K_S Delta++ data
CFG_KEVIN = AnalysisConfig(
  treeName                 = "ntFSGlueX_100_11100_angles",
  dataFileName             = "./dataPhotoProdKmKS/data/pipkmks_100_11100_B4_M16_*_SKIM_A2.root.angles",
  psAccFileName            = "./dataPhotoProdKmKS/phaseSpace/pipkmks_100_11100_B4_M16_SIGNAL_SKIM_A2.root.angles",
  psGenFileName            = "./dataPhotoProdKmKS/phaseSpace/pipkmks_100_11100_B4_M16_MCGEN_GENERAL_SKIM_A2.root.angles",
  polarization             = "beamPol", # read polarization from tree column
  maxL                     = 4,
  outFileDirBaseName       = "./plotsTestKevin",
  # normalizeMoments         = True,
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
