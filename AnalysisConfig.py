"""
This module defines the input parameters for the various moment analyses
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import (
  dataclass,
  field,
)
from enum import Enum, auto

import ROOT

from MomentCalculator import (
  KinematicBinningVariable,
  MomentIndices,
)
from PlottingUtilities import HistAxisBinning
import Utilities


def defineOverwriteRDataFrame(
  df:         ROOT.RDataFrame,
  colName:    str,
  newFormula: str,
) -> ROOT.RDataFrame:
  """Defines column `colName` using formula `newFormula`; overwriting column definition if it already exists"""
  if not df.HasColumn(colName):
    # print(f"Defining column '{colName}' = '{newFormula}'")
    df = df.Define(colName, newFormula)
  else:
    print(f"Redefining column '{colName}' to '{newFormula}'")
    df = df.Redefine(colName, newFormula)
  return df


@dataclass
class BeamPolInfo:
  """Stores information about beam polarization for a specific orientation"""
  pol:    float | str  # photon-beam polarization magnitude value or column name
  PhiLab: float | str  # azimuthal angle of photon beam polarization in lab frame [deg] value or column name

  def __str__(self) -> str:
    result = "pol = "
    if isinstance(self.pol, float):
      result += f"{self.pol:.4f}"
    else:
      result += f"'{self.pol}'"
    result += ", PhiLab = "
    if isinstance(self.PhiLab, float):
      result += f"{self.PhiLab:.1f} deg"
    else:
      result += f"'{self.PhiLab}'"
    return result


BEAM_POL_INFOS: dict[str, dict[str, BeamPolInfo | None]] = {  # data period : {<beam-polarization orientation> : BeamPolInfo(...)}; `None`` means unpolarized
  "merged" : {  # several merged data periods with different polarization values
    "All" : BeamPolInfo(  # read polarization values from the given column names
      pol    = "beamPol",
      PhiLab = "beamPolPhiLabDeg",
    ),
    # approximate polarization values across all data-taking periods; don't use for serious analyses
    "PARA_0" : BeamPolInfo(
      pol    = 0.35,
      PhiLab = 0,
    ),
    "PERP_45" : BeamPolInfo(
      pol    = 0.35,
      PhiLab = 45,
    ),
    "PERP_90" : BeamPolInfo(
      pol    = 0.35,
      PhiLab = 90,
    ),
    "PARA_135" : BeamPolInfo(
      pol    = 0.35,
      PhiLab = -45,
    ),
    "AMO"   : None,
    "Unpol" : None,
  },
  # polarization values from Version 9 of `makePolVals` tool from https://halldweb.jlab.org/wiki-private/index.php/TPOL_Polarization
  # beam polarization angles in lab frame taken from `Lab Phi` column of tables 2 to 5 in GlueX-doc-3977
  "2017_01" : {  # polarization magnitudes obtained by running `.x makePolVals.C(17, 1, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      pol    = 0.3537,
      PhiLab = 1.8,
    ),
    "PERP_45" : BeamPolInfo(
      pol    = 0.3484,
      PhiLab = 47.9,
    ),
    "PERP_90" : BeamPolInfo(
      pol    = 0.3472,
      PhiLab = 94.5,
    ),
    "PARA_135" : BeamPolInfo(
      pol    = 0.3512,
      PhiLab = -41.6,
    ),
    "AMO"   : None,
    "Unpol" : None,
  },
  "2018_01" : {  # polarization magnitudes obtained by running `.x makePolVals.C(18, 1, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      pol    = 0.3420,
      PhiLab = 4.1,
    ),
    "PERP_45" : BeamPolInfo(
      pol    = 0.3474,
      PhiLab = 48.5,
    ),
    "PERP_90" : BeamPolInfo(
      pol    = 0.3478,
      PhiLab = 94.2,
    ),
    "PARA_135" : BeamPolInfo(
      pol    = 0.3517,
      PhiLab = -42.4,
    ),
    "AMO"   : None,
    "Unpol" : None,
  },
  "2018_08" : {  # polarization magnitudes obtained by running `.x makePolVals.C(18, 2, 0, 75)` in ROOT shell
    "PARA_0" : BeamPolInfo(
      pol    = 0.3563,
      PhiLab = 3.3,
    ),
    "PERP_45" : BeamPolInfo(
      pol    = 0.3403,
      PhiLab = 48.3,
    ),
    "PERP_90" : BeamPolInfo(
      pol    = 0.3430,
      PhiLab = 92.9,
    ),
    "PARA_135" : BeamPolInfo(
      pol    = 0.3523,
      PhiLab = -42.1,
    ),
    "AMO"   : None,
    "Unpol" : None,
  },
  "2019_11" : {  # Spring 2020 polarization magnitudes obtained by running `root makePolVals2020.C'(0)'`  #TODO did Alex already produce PhiLab values?
    "PARA_0" : BeamPolInfo(
      pol    = 0.3525,
      PhiLab = 0,
    ),
    "PERP_45" : BeamPolInfo(
      pol    = 0.3535,
      PhiLab = 45,
    ),
    "PERP_90" : BeamPolInfo(
      pol    = 0.3536,
      PhiLab = 90,
    ),
    "PARA_135" : BeamPolInfo(
      pol    = 0.3721,
      PhiLab = -45,
    ),
    "AMO"   : None,
    "Unpol" : None,
  },
}


@dataclass
class AnalysisConfig:
  """Stores configuration parameters for the moment analysis; defaults are for unpolarized pi+pi- analysis"""

  class MethodType(Enum):
    """Enumerates methods used for moment analysis"""
    LIN_ALG_BG_SUBTR_NEG_WEIGHTS = 0  # linear-algebra method for acceptance correction + background subtraction by negative event weights
    LIN_ALG_BG_SUBTR_MOMENTS     = 1  # linear-algebra method for acceptance correction + background subtraction by subtracting moments
    MAX_LIKELIHOOD_FIT           = 2  # maximum-likelihood fit for acceptance correction + background subtraction by subtracting moments

  # defaults are for unpolarized pi+pi- analysis
  method:                   AnalysisConfig.MethodType = MethodType.LIN_ALG_BG_SUBTR_NEG_WEIGHTS  # method used to estimate moments from data
  treeName:                 str                       = "PiPi"  # name of tree to read from data and MC files
  # Spring 2017 low-energy data
  dataFileName:             str                       = "./dataPhotoProdPiPi/unpolarized/2017_01/tbin_0.4_0.5/PiPi/data_flat.root"  # file with real data to analyze
  psAccFileName:            str | None                = "./dataPhotoProdPiPi/unpolarized/2017_01/tbin_0.4_0.5/PiPi/phaseSpace_acc_flat.root"  # file with accepted phase-space MC
  psGenFileName:            str | None                = "./dataPhotoProdPiPi/unpolarized/2017_01/tbin_0.4_0.5/PiPi/phaseSpace_gen_flat.root"  # file with generated phase-space MC
  #TODO make it easier to switch data
  # # Fall 2018 low-energy data
  # dataFileName:             str                       = "./dataPhotoProdPiPi/unpolarized/2018_08/tbin_0.4_0.5/PiPi/data_flat.root"  # file with real data to analyze
  # psAccFileName:            str | None                = "./dataPhotoProdPiPi/unpolarized/2018_08/tbin_0.4_0.5/PiPi/phaseSpace_acc_flat.root"  # file with accepted phase-space MC
  # psGenFileName:            str | None                = "./dataPhotoProdPiPi/unpolarized/2018_08/tbin_0.4_0.5/PiPi/phaseSpace_gen_flat.root"  # file with generated phase-space MC
  polarization:             float | str | None        = None  # photon-beam polarization; None = unpolarized photoproduction; polarized photoproduction: either polarization value or name of polarization
  maxL:                     int | tuple[int, int]     = 8  # if int: maximum L of physical and measured moments; if tuple: (max L of physical moments, max L of measured moments)
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
  # plotMomentsInBins:        bool                      = True
  plotMomentsInBins:        bool                      = False
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
      _var    = None,  # set by init()  #TODO why keep a separate binVarMass member instead of using binning.var?
    )
  )

  @property
  def maxLPhys(self) -> int:
    """Returns maximum L of physical moments"""
    return self.maxL if isinstance(self.maxL, int) else self.maxL[0]

  @property
  def maxLMeas(self) -> int:
    """Returns maximum L of measured moments"""
    return self.maxL if isinstance(self.maxL, int) else self.maxL[1]

  @property
  def momentIndicesPhys(self) -> MomentIndices:
    """Returns moment indices for physical moments"""
    return MomentIndices(maxL = self.maxLPhys, polarized = (self.polarization is not None))

  @property
  def momentIndicesMeas(self) -> MomentIndices:
    """Returns moment indices for measured moments"""
    return MomentIndices(maxL = self.maxLMeas, polarized = (self.polarization is not None))

  @property
  def momentIndices(self) -> tuple[MomentIndices, MomentIndices]:
    """Returns moment indices for physical and measured moments"""
    return (self.momentIndicesPhys, self.momentIndicesMeas)

  @property
  def outFileDirName(self) -> str:
    """Returns name of directory into which all output will be written"""
    return f"{self.outFileDirBaseName}.maxL_{self.maxL if isinstance(self.maxL, int) else f'{self.maxL[0]}_{self.maxL[1]}'}"

  @property
  def outFileNamePrefix(self) -> str:
    """Returns name prefix prepended to output file names"""
    return "norm" if self.normalizeMoments else "unnorm"

  def init(
    self,
    createOutFileDir: bool = False,
  ) -> None:
    """Creates output directory and initializes member variables; needs to be called before passing the config to any consumer"""
    if not isinstance(self.maxL, int):
      assert self.maxL[0] <= self.maxL[1], f"Maximum L for physical moments {self.maxL[0]=} must be smaller than or equal to maximum L for measured moments {self.maxL[1]=}"
    self.massBinning.var = self.binVarMass
    if createOutFileDir:
      Utilities.makeDirPath(self.outFileDirName)

  class DataType(Enum):
    """Enumerates used input data types"""
    REAL_DATA             = 0
    REAL_DATA_SIGNAL      = 1  # real data in signal region
    REAL_DATA_SIDEBAND    = 2  # real data in sideband region(s)
    GENERATED_PHASE_SPACE = 3
    ACCEPTED_PHASE_SPACE  = 4

  def loadData(
    self,
    dataType:             AnalysisConfig.DataType,
    additionalCuts:       Iterable[str] | None = None,  # optional additional cuts to be applied to loaded data
    additionalColumnDefs: dict[str, str]       = {},    # additional columns to define
  ) -> ROOT.RDataFrame | None:
    """Returns a ROOT RDataFrame with given data type, applies optional additional cuts, and defines optional additional columns"""
    df = None
    if dataType == AnalysisConfig.DataType.REAL_DATA:
      print(f"Loading real data from tree '{self.treeName}' in file '{self.dataFileName}'")
      df = ROOT.RDataFrame(self.treeName, self.dataFileName)
    elif dataType == AnalysisConfig.DataType.REAL_DATA_SIGNAL:
      print(f"Loading real-data signal events with weight = 1 from tree '{self.treeName}' in file '{self.dataFileName}'")
      df = ROOT.RDataFrame(self.treeName, self.dataFileName)
      if "eventWeight" in df.GetColumnNames():
        df = df.Filter("eventWeight == 1")
      # if there is no `eventWeight` column assume all events have weight 1
    elif dataType == AnalysisConfig.DataType.REAL_DATA_SIDEBAND:
      df = ROOT.RDataFrame(self.treeName, self.dataFileName)
      if "eventWeight" in df.GetColumnNames():
        df = df.Filter("eventWeight < 0")
        if df.Count().GetValue() > 0:
          print(f"Loading real-data sideband events with weight < 0 from tree '{self.treeName}' in file '{self.dataFileName}'")
        else:
          df = None  # no events with weight < 0
      else:
        # if there is no `eventWeight` column assume all events have weight 1 and hence there are no sideband events
        df = None
    elif dataType == AnalysisConfig.DataType.GENERATED_PHASE_SPACE:
      if self.psGenFileName is None:
        print("??? Warning: File name for generated phase-space data was not provided. Acceptance may not be calculated correctly.")
        df = None
      else:
        print(f"Loading generated phase-space data from tree '{self.treeName}' in file '{self.psGenFileName}'")
        df = ROOT.RDataFrame(self.treeName, self.psGenFileName)
    elif dataType == AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE:
      if self.psAccFileName is None:
        print("??? Warning: File name for accepted phase-space data was not provided. Assuming perfect acceptance.")
        df = None
      else:
        print(f"Loading accepted phase-space data from tree '{self.treeName}' in file '{self.psAccFileName}'")
        df = ROOT.RDataFrame(self.treeName, self.psAccFileName)
    else:
      raise ValueError(f"Unknown data type: {dataType}")
    # add additional columns if requested
    if df is not None and additionalColumnDefs:
      for columnName, columnFormula in additionalColumnDefs.items():
        print(f"Defining additional column '{columnName}' = '{columnFormula}'")
        df = defineOverwriteRDataFrame(df, columnName, columnFormula)
    # apply additional cuts
    if df is not None and additionalCuts is not None:
      for cut in additionalCuts:
        print(f"Applying additional cut: '{cut}'")
        df = df.Filter(cut)
    return df


# configurations for unpolarized pi+ pi- data in CLAS kinematic range
CFG_UNPOLARIZED_PIPI_CLAS = AnalysisConfig()
CFG_UNPOLARIZED_PIPI_PWA  = AnalysisConfig(
  outFileDirBaseName = "./plotsPhotoProdPiPiUnpolPwa",
  massBinning        = HistAxisBinning(nmbBins = 56, minVal = 0.28, maxVal = 1.40),  # binning used in PWA of unpolarized data
)
CFG_UNPOLARIZED_PIPI_JPAC = AnalysisConfig(
  # dataFileName       = "./dataPhotoProdPiPiUnpolJPAC/mc_full/tbin_0.4_0.5/data_flat.root"  # Lukasz's data
  # dataFileName       = "./dataPhotoProdPiPiUnpolJPAC/ideal/data_reweighted_flat.root"  # data generated from real parts of true moments up to L = 4
  # psAccFileName      = None,  # no file with accepted phase-space MC
  # psGenFileName      = None,  # no file with generated phase-space MC
  dataFileName       = "./dataPhotoProdPiPiUnpolJPAC/ideal_8GeV/data_flat.PiPi.root",  # data_reweighted_flat.root boosted to lab frame and passed through simulation, reconstruction, and selection
  psAccFileName      = "./dataPhotoProdPiPiUnpolJPAC/ideal_8GeV/phaseSpace_acc_flat.PiPi.root",
  psGenFileName      = "./dataPhotoProdPiPiUnpolJPAC/ideal_8GeV/phaseSpace_gen_flat.PiPi.root",
  outFileDirBaseName = "./plotsPhotoProdPiPiUnpolJPAC",
  # massBinning        = HistAxisBinning(nmbBins = 25, minVal = 0.4, maxVal = 1.40),
  # massBinning        = HistAxisBinning(nmbBins = 2, minVal = 0.4, maxVal = 0.42),
)
# configuration for polarized pi+ pi- data
CFG_POLARIZED_PIPI = AnalysisConfig(
  dataFileName       = "./dataPhotoProdPiPi/polarized/2017_01/tbin_0.1_0.2/PiPi/data_flat_0.0.root",
  psAccFileName      = "./dataPhotoProdPiPi/polarized/2017_01/tbin_0.1_0.2/PiPi/phaseSpace_acc_flat.root",
  psGenFileName      = "./dataPhotoProdPiPi/polarized/2017_01/tbin_0.1_0.2/PiPi/phaseSpace_gen_flat.root",
  polarization       = "beamPol",  # read polarization from tree column
  maxL               = 4,
  outFileDirBaseName = "./plotsPhotoProdPiPiPol",
  massBinning        = HistAxisBinning(nmbBins = 50, minVal = 0.28, maxVal = 2.28),  # binning used in PWA of polarized data
)
# configuration for unpolarized pi+ p data
CFG_UNPOLARIZED_PIPP = AnalysisConfig(
  dataFileName       = "./dataPhotoProdPiPi/unpolarized/2017_01/tbin_0.4_0.5/PipP/data_flat.root",
  psAccFileName      = "./dataPhotoProdPiPi/unpolarized/2017_01/tbin_0.4_0.5/PipP/phaseSpace_acc_flat.root",
  psGenFileName      = "./dataPhotoProdPiPi/unpolarized/2017_01/tbin_0.4_0.5/PipP/phaseSpace_gen_flat.root",
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
CFG_POLARIZED_ETAPI0 = AnalysisConfig(
  treeName           = "EtaPi0",
  dataFileName       = "./dataPhotoProdEtaPi0/polarized/merged/t010020/EtaPi0/data_flat_All.root",
  psAccFileName      = "./dataPhotoProdEtaPi0/polarized/merged/t010020/EtaPi0/phaseSpace_acc_flat_All.root",
  psGenFileName      = "./dataPhotoProdEtaPi0/polarized/merged/t010020/EtaPi0/phaseSpace_gen_flat_All.root",
  polarization       = "beamPol",  # read polarization from tree column
  maxL               = 4,
  outFileDirBaseName = "./plotsPhotoProdEtaPi0",
  binVarMass         = KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{#eta}#it{#pi}^{0}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ),
  massBinning        = HistAxisBinning(nmbBins = 17, minVal = 1.04, maxVal = 1.72),  # 40 MeV wide bins
)
# configuration for Zach's polarized eta' pi0 data
CFG_POLARIZED_ETAPPI0 = AnalysisConfig(
  treeName           = "kin",
  dataFileName       = "./dataPhotoProdEtapPi0/polarized/merged/tbin_0.1_0.5/EtaPi0/data_flat_PARA_0.root",
  psAccFileName      = "./dataPhotoProdEtapPi0/polarized/merged/tbin_0.1_0.5/EtaPi0/phaseSpace_acc_flat_PARA_0.root",
  psGenFileName      = "./dataPhotoProdEtapPi0/polarized/merged/tbin_0.1_0.5/EtaPi0/phaseSpace_gen_flat_PARA_0.root",
  polarization       = "beamPol",  # read polarization from tree column
  maxL               = 4,
  outFileDirBaseName = "./plotsPhotoProdEtapPi0",
  binVarMass         = KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{#eta}'#it{#pi}^{0}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ),
  massBinning        = HistAxisBinning(nmbBins = 20, minVal = 1.2, maxVal = 2.0),  # 40 MeV wide bins
)
# configuration for Will's unpolarized eta' eta data
CFG_UNPOLARIZED_ETAPETA = AnalysisConfig(
  treeName           = "EtapEta",
  dataFileName       = "./dataPhotoProdEtapEta/unpolarized/2018_08/ALLT/EtapEta/data_flat_Unpol.root",
  psAccFileName      = "./dataPhotoProdEtapEta/unpolarized/2018_08/ALLT/EtapEta/phaseSpace_acc_flat_Unpol.root",
  psGenFileName      = "./dataPhotoProdEtapEta/unpolarized/2018_08/ALLT/EtapEta/phaseSpace_gen_flat_Unpol.root",
  polarization       = None,  # unpolarized photoproduction
  maxL               = 4,
  outFileDirBaseName = "./plotsPhotoProdEtapEta",
  # plotMomentsInBins  = True,
  binVarMass         = KinematicBinningVariable(
    name      = "mass",
    label     = "#it{m}_{#it{#eta}'#it{#eta}}",
    unit      = "GeV/#it{c}^{2}",
    nmbDigits = 3,
  ),
  # massBinning        = HistAxisBinning(nmbBins = 8, minVal = 1.5, maxVal = 3.5),  # 250 MeV wide bins
  massBinning        = HistAxisBinning(nmbBins = 15, minVal = 1.5, maxVal = 3.0),  # 100 MeV wide bins
)
# configuration for Kevin's K- K_S Delta++ data
CFG_KEVIN = AnalysisConfig(
  treeName                 = "ntFSGlueX_100_11100_angles",
  dataFileName             = "./dataPhotoProdKmKS/data/pipkmks_100_11100_B4_M16_*_SKIM_A2.root.angles",
  psAccFileName            = "./dataPhotoProdKmKS/phaseSpace/pipkmks_100_11100_B4_M16_SIGNAL_SKIM_A2.root.angles",
  psGenFileName            = "./dataPhotoProdKmKS/phaseSpace/pipkmks_100_11100_B4_M16_MCGEN_GENERAL_SKIM_A2.root.angles",
  polarization             = "beamPol",  # read polarization from tree column
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
