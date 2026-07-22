"""
This module defines the parameters for the various moment analyses
"""

from __future__ import annotations

from collections.abc import (
  Callable,
  Iterable,
)
from dataclasses import (
  dataclass,
  field,
)
from enum import Enum

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


#TODO maybe using a function would be more flexible here
BEAM_POL_INFOS: dict[str, dict[str, BeamPolInfo | None]] = {  # <data-period label> : {<beam-polarization label> : BeamPolInfo(...)}; `None` means unpolarized
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
class SubsystemInfo:
  """Stores information about the (A, B) two-body subsystem, for which moments are calculated, for the reaction beam + target -> A + B + recoil"""
  lvALabel:          str  # label of Lorentz-vector of daughter A (analyzer)
  lvBLabel:          str  # label of Lorentz-vector of daughter B
  lvRecoilLabel:     str  # label of Lorentz-vector of recoil particle
  pairLabel:         str  # label for particle pair (e.g. "PiPi" for pi+ pi- pair)
  ATLatexLabel:      str = ""  # optional LaTeX label for particle A (analyzer)
  BTLatexLabel:      str = ""  # optional LaTeX label for particle B
  recoilTLatexLabel: str = ""  # optional LaTeX label for recoil particle
  pairTLatexLabel:   str = ""  # optional LaTeX label for particle pair (e.g. "#pi#pi" for pi+ pi- pair)


@dataclass
class DataConfig:
  """Stores parameters of data for moment calculation and provides methods to load the data"""
  dataFilePath:   str         # file path of real data to analyze
  psAccFilePath:  str | None  # file path of accepted phase-space MC
  psGenFilePath:  str | None  # file path of generated phase-space MC
  polarization:   float | str | None  # photon-beam polarization; None = unpolarized photoproduction; polarized photoproduction: either polarization value or name of polarization column  #TODO use BeamPolInfo
  maxL:           int | tuple[int, int]  # if int: maximum L of physical and measured moments; if tuple: (max L of physical moments, max L of measured moments)
  outFileDirPath: str  # base path of directory, into which all output of the moment calculation will be written

  def __post_init__(self) -> None:
    """Checks that maximum L tuple has valid values"""
    if isinstance(self.maxL, tuple):
      assert self.maxL[0] <= self.maxL[1], f"Maximum L for physical moments {self.maxL[0]=} must be smaller than or equal to maximum L for measured moments {self.maxL[1]=}"

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

  def createOutFileDir(self) -> None:
    """Creates output directory if it does not exist yet"""
    Utilities.makeDirPath(self.outFileDirPath)

  def loadData(
    self,
    dataType:             AnalysisConfig.DataType,
    treeName:             str,
    additionalCuts:       Iterable[str]  = (),  # optional additional cuts to be applied to loaded data
    additionalColumnDefs: dict[str, str] = {},  # additional columns to define
  ) -> ROOT.RDataFrame | None:
    """Returns a ROOT RDataFrame for given data type, applies optional additional cuts, and defines optional additional columns"""
    df = None
    if dataType == AnalysisConfig.DataType.REAL_DATA:
      print(f"Loading real data from tree '{treeName}' in file '{self.dataFilePath}'")
      df = ROOT.RDataFrame(treeName, self.dataFilePath)
    elif dataType == AnalysisConfig.DataType.REAL_DATA_SIGNAL:
      print(f"Loading real-data signal events with weight = 1 from tree '{treeName}' in file '{self.dataFilePath}'")
      df = ROOT.RDataFrame(treeName, self.dataFilePath)
      if "eventWeight" in df.GetColumnNames():
        df = df.Filter("eventWeight == 1")
      # if there is no `eventWeight` column assume all events have weight 1
    elif dataType == AnalysisConfig.DataType.REAL_DATA_SIDEBAND:
      df = ROOT.RDataFrame(treeName, self.dataFilePath)
      if "eventWeight" in df.GetColumnNames():
        df = df.Filter("eventWeight < 0")
        if df.Count().GetValue() > 0:
          print(f"Loading real-data sideband events with weight < 0 from tree '{treeName}' in file '{self.dataFilePath}'")
        else:
          df = None  # no events with weight < 0
      else:
        # if there is no `eventWeight` column assume all events have weight 1 and hence there are no sideband events
        df = None
    elif dataType == AnalysisConfig.DataType.GENERATED_PHASE_SPACE:
      if self.psGenFilePath is None:
        print("??? Warning: File name for generated phase-space data was not provided. Acceptance may not be calculated correctly.")
        df = None
      else:
        print(f"Loading generated phase-space data from tree '{treeName}' in file '{self.psGenFilePath}'")
        df = ROOT.RDataFrame(treeName, self.psGenFilePath)
    elif dataType == AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE:
      if self.psAccFilePath is None:
        print("??? Warning: File name for accepted phase-space data was not provided. Assuming perfect acceptance.")
        df = None
      else:
        print(f"Loading accepted phase-space data from tree '{treeName}' in file '{self.psAccFilePath}'")
        df = ROOT.RDataFrame(treeName, self.psAccFilePath)
    else:
      raise ValueError(f"Unknown data type: {dataType}")
    # add additional columns if requested
    if df is not None and additionalColumnDefs:
      for columnName, columnFormula in additionalColumnDefs.items():
        print(f"Defining additional column '{columnName}' = '{columnFormula}'")
        df = defineOverwriteRDataFrame(df, columnName, columnFormula)
    # apply additional cuts
    if df is not None:
      for cut in additionalCuts:
        print(f"Applying additional cut: '{cut}'")
        df = df.Filter(cut)
    return df


@dataclass
class AnalysisConfig:
  """Stores configuration parameters for the moment analysis; defaults are for unpolarized pi+pi- analysis"""

  class MethodType(Enum):
    """Enumerates methods used for moment analysis"""
    LIN_ALG_BG_SUBTR_NEG_WEIGHTS = 0  # linear-algebra method for acceptance correction + background subtraction by negative event weights
    LIN_ALG_BG_SUBTR_MOMENTS     = 1  # linear-algebra method for acceptance correction + background subtraction by subtracting moments
    MAX_LIKELIHOOD_FIT           = 2  # maximum-likelihood fit for acceptance correction + background subtraction by subtracting moments

  class CoordSysType(Enum):
    """Enumerates coordinate systems in which moments can be calculated"""
    HF = 0  # helicity frame
    GJ = 1  # Gottfried-Jackson frame

  class DataType(Enum):
    """Enumerates used data types"""
    REAL_DATA             = 0
    REAL_DATA_SIGNAL      = 1  # real data in signal region
    REAL_DATA_SIDEBAND    = 2  # real data in sideband region(s)
    GENERATED_PHASE_SPACE = 3
    ACCEPTED_PHASE_SPACE  = 4

  class DataFormat(Enum):
    """Enumerates input formats of data files"""
    ALEX            = 0  # Alex' data format  #TODO improve naming
    AMPTOOLS        = 1  # AmpTools format
    JPAC_MC         = 2  # MC truth data in JPAC text format
    TLORENTZVECTORS = 3  # TLorentzVector for each particle
    FSROOT          = 4  # FSROOT format

  # defaults are for unpolarized gamma p -> (pi+ pi-) p data
  method:                   AnalysisConfig.MethodType         = MethodType.LIN_ALG_BG_SUBTR_NEG_WEIGHTS  # method used to estimate moments from data
  frame:                    CoordSysType                      = CoordSysType.HF  # coordinate system, in which moments are calculated
  subsystem:                SubsystemInfo                     = field(default_factory = lambda:
    SubsystemInfo(  # pi+pi- subsystem to analyze; pi+ is the analyzer
      lvALabel          = "pip",     # label of pi+ Lorentz-vector
      lvBLabel          = "pim",     # label of pi- Lorentz-vector
      lvRecoilLabel     = "recoil",  # label of recoil-proton Lorentz-vector
      pairLabel         = "PiPi",
      ATLatexLabel      = "#it{#pi}^{#plus}",
      BTLatexLabel      = "#it{#pi}^{#minus}",
      recoilTLatexLabel = "#it{p}",
      pairTLatexLabel   = "#it{#pi}^{#plus}#it{#pi}^{#minus}",
    )
  )
  dataDirBasePath:          str                               = "./dataPhotoProdPiPi/unpolarized"  # base directory for data
  dataPeriods:              tuple[str, ...]                   = (  # labels of data periods to process
    "2017_01",
    "2018_08",
  )
  tBinLabels:               tuple[str, ...]                   = (  # labels of t bins to process
    "tbin_0.4_0.5",
  )
  beamPolLabels:            tuple[str, ...]                   = (  # labels of beam polarizations to process, e.g. "PARA_0"; use "AMO" or "Unpol" for unpolarized data  #TODO store list of BeamPolInfos and generate labels with a function
    "Unpol",
  )
  inputDataFormats:         dict[DataType, DataFormat]        = field(default_factory = lambda: {  # input data formats for each data type
    AnalysisConfig.DataType.REAL_DATA             : AnalysisConfig.DataFormat.ALEX,
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : AnalysisConfig.DataFormat.ALEX,
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : AnalysisConfig.DataFormat.AMPTOOLS,
  })
  inputTreeName:            str                               = "kin"   # name of tree for data in input format; must contain Lorentz-vectors of all particles
  convertedTreeName:        str                               = "PiPi"  # name of tree for data in converted format
  maxLs:                    tuple[int | tuple[int, int], ...] = (  # if int: maximum L of physical and measured moments; if tuple: (max L of physical moments, max L of measured moments)
    4,
    6,
    8,
  )
  outFileDirBasePath:       str                               = "./plotsPhotoProdPiPiUnpolCLAS"  # base name of directory into which all output of moment calculation will be written
  # normalizeMoments:         bool                              = True
  normalizeMoments:         bool                              = False
  nmbBootstrapSamples:      int                               = 0
  # nmbBootstrapSamples:      int                               = 10000
  # plotAngularDistributions: bool                              = True
  plotAngularDistributions: bool                              = False
  # plotAccIntegralMatrices:  bool                              = True
  plotAccIntegralMatrices:  bool                              = False
  # calcAccPsMoments:         bool                              = True
  calcAccPsMoments:         bool                              = False
  # plotAccPsMoments:         bool                              = True
  plotAccPsMoments:         bool                              = False
  # plotMomentsInBins:        bool                              = True
  plotMomentsInBins:        bool                              = False
  # plotMeasuredMoments:      bool                              = True
  plotMeasuredMoments:      bool                              = False
  # plotCovarianceMatrices:   bool                              = True
  plotCovarianceMatrices:   bool                              = False
  limitNmbPsAccEvents:      int                               = 0
  # limitNmbPsAccEvents:      int                               = 100000
  massBinning:              HistAxisBinning                   = field(default_factory = lambda:
    HistAxisBinning(  # same binning as used by CLAS
      nmbBins = 100,  # 10 MeV wide bins
      minVal  = 0.4,  # [GeV]
      maxVal  = 1.4,  # [GeV]
      _var    = KinematicBinningVariable(
        name      = "mass",
        label     = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}",
        unit      = "GeV/#it{c}^{2}",
        nmbDigits = 3,
      ),
    )
  )

  def inputDataDirBasePath(
    self,
    dataPeriod: str,
    tBinLabel:  str,
  ) -> str:
    """Generates base path of directory with data in input format"""
    return f"{self.dataDirBasePath}/{dataPeriod}/{tBinLabel}/input"

  @staticmethod
  def _default_inputFilePath(
    cfg:          AnalysisConfig,
    dataType:     AnalysisConfig.DataType,
    dataPeriod:   str,
    tBinLabel:    str,
    beamPolLabel: str
  ) -> str:
    """Default function that returns path of data file in input format based on data type, data period, t bin label, and beam polarization label"""
    # one input file for each data type
    if dataType == AnalysisConfig.DataType.REAL_DATA:
      return f"{cfg.inputDataDirBasePath(dataPeriod, tBinLabel)}/tree_data_{beamPolLabel}.root"
    elif dataType == AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE:
      return f"{cfg.inputDataDirBasePath(dataPeriod, tBinLabel)}/tree_accepted_{beamPolLabel}.root"
    elif dataType == AnalysisConfig.DataType.GENERATED_PHASE_SPACE:
      return f"{cfg.inputDataDirBasePath(dataPeriod, tBinLabel)}/tree_thrown_{beamPolLabel}.root"
    else:
      raise ValueError(f"Unknown data type: {dataType}")

  _inputFilePath: Callable[[AnalysisConfig, AnalysisConfig.DataType, str, str, str], str] = field(
        default_factory = lambda: AnalysisConfig._default_inputFilePath,
        repr            = False,
        compare         = False,
    )

  def inputFilePath(
    self,
    dataType:     AnalysisConfig.DataType,
    dataPeriod:   str,
    tBinLabel:    str,
    beamPolLabel: str,
  ) -> str:
    """Returns file path of data file in input format based on data type, data period, t bin label, and beam polarization label; can be overwritten by providing a different function to the `_inputFilePath` member"""
    return self._inputFilePath(self, dataType, dataPeriod, tBinLabel, beamPolLabel)

  def outFileDirPath(
    self,
    dataPeriod:   str,
    tBinLabel:    str,
    beamPolLabel: str,
    maxL:         int | tuple[int, int],
  ) -> str:
    """Generates path of directory, into which all output of the moment calculation will be written"""
    return f"{self.outFileDirBasePath}/{dataPeriod}/{tBinLabel}/{beamPolLabel}.maxL_{maxL if isinstance(maxL, int) else f'{maxL[0]}_{maxL[1]}'}"

  def getConfigInputData(
    self,
    dataPeriod:   str,
    tBinLabel:    str,
    beamPolLabel: str,
    maxL:         int | tuple[int, int],
  ) -> DataConfig:
    """Returns a `DataConfig` object for data files in input format for a given data period, t bin label, and beam polarization label"""
    return DataConfig(
      dataFilePath       = self.inputFilePath(AnalysisConfig.DataType.REAL_DATA,             dataPeriod, tBinLabel, beamPolLabel),
      psAccFilePath      = self.inputFilePath(AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE,  dataPeriod, tBinLabel, beamPolLabel),
      psGenFilePath      = self.inputFilePath(AnalysisConfig.DataType.GENERATED_PHASE_SPACE, dataPeriod, tBinLabel, beamPolLabel),
      polarization       = "beamPol" if BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel] is not None else None,  # if beam was polarized, converted data are expected to have `beamPol` column with polarization value
      maxL               = maxL,
      outFileDirPath     = self.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel, maxL),
    )

  def convertedDataDirBasePath(
    self,
    dataPeriod: str,
    tBinLabel:  str,
  ) -> str:
    """Generates base path of directory with data in converted format"""
    return f"{self.dataDirBasePath}/{dataPeriod}/{tBinLabel}/{self.subsystem.pairLabel}"

  def convertedFilePath(
    self,
    dataType:     AnalysisConfig.DataType,
    dataPeriod:   str,
    tBinLabel:    str,
    beamPolLabel: str,
  ) -> str:
    """Generates path of data file in converted format based on data type, data period, t bin label, and beam polarization label"""
    if dataType == AnalysisConfig.DataType.REAL_DATA:
      return f"{self.convertedDataDirBasePath(dataPeriod, tBinLabel)}/data_flat_{beamPolLabel}.root"
    elif dataType == AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE:
      return f"{self.convertedDataDirBasePath(dataPeriod, tBinLabel)}/phaseSpace_acc_flat_{beamPolLabel}.root"
    elif dataType == AnalysisConfig.DataType.GENERATED_PHASE_SPACE:
      return f"{self.convertedDataDirBasePath(dataPeriod, tBinLabel)}/phaseSpace_gen_flat_{beamPolLabel}.root"
    else:
      raise ValueError(f"Unknown data type: {dataType}")

  @property
  def outFileNamePrefix(self) -> str:
    """Returns name prefix prepended to moment output file names"""
    return "norm" if self.normalizeMoments else "unnorm"

  def getConfigConvertedData(
    self,
    dataPeriod:   str,
    tBinLabel:    str,
    beamPolLabel: str,
    maxL:         int | tuple[int, int],
  ) -> DataConfig:
    """Returns a `DataConfig` object for data in converted format for a given data period, t bin label, and beam polarization label"""
    return DataConfig(
      dataFilePath       = self.convertedFilePath(AnalysisConfig.DataType.REAL_DATA,             dataPeriod, tBinLabel, beamPolLabel),
      psAccFilePath      = self.convertedFilePath(AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE,  dataPeriod, tBinLabel, beamPolLabel),
      psGenFilePath      = self.convertedFilePath(AnalysisConfig.DataType.GENERATED_PHASE_SPACE, dataPeriod, tBinLabel, beamPolLabel),
      polarization       = "beamPol" if BEAM_POL_INFOS[dataPeriod[:7]][beamPolLabel] is not None else None,  # if beam was polarized, converted data are expected to have `beamPol` column with polarization value
      maxL               =  maxL,
      outFileDirPath     = self.outFileDirPath(dataPeriod, tBinLabel, beamPolLabel, maxL),
    )


# configurations for unpolarized gamma p -> (pi+ pi-) p data in CLAS kinematic range
CFG_UNPOLARIZED_PIPI_CLAS = AnalysisConfig()
CFG_UNPOLARIZED_PIPI_PWA  = AnalysisConfig(
  outFileDirBasePath = "./plotsPhotoProdPiPiUnpolPwa",
  massBinning        = HistAxisBinning(  # binning used in PWA of unpolarized data
    nmbBins = 56,    # 20 MeV wide bins
    minVal  = 0.28,  # [GeV]
    maxVal  = 1.40,  # [GeV]
    _var    = KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    ),
  ),
)
CFG_UNPOLARIZED_PIPI_JPAC = AnalysisConfig(
  # dataFileName       = "./dataPhotoProdPiPiUnpolJPAC/mc_full/tbin_0.4_0.5/data_flat.root"  # Lukasz's data
  # dataFileName       = "./dataPhotoProdPiPiUnpolJPAC/ideal/data_reweighted_flat.root"  # data generated from real parts of true moments up to L = 4
  # psAccFileName      = None,  # no file with accepted phase-space MC
  # psGenFileName      = None,  # no file with generated phase-space MC
  # dataFileName       = "./dataPhotoProdPiPiUnpolJPAC/ideal_8GeV/data_flat.PiPi.root",  # data_reweighted_flat.root boosted to lab frame and passed through simulation, reconstruction, and selection
  # psAccFileName      = "./dataPhotoProdPiPiUnpolJPAC/ideal_8GeV/phaseSpace_acc_flat.PiPi.root",
  # psGenFileName      = "./dataPhotoProdPiPiUnpolJPAC/ideal_8GeV/phaseSpace_gen_flat.PiPi.root",
  outFileDirBasePath = "./plotsPhotoProdPiPiUnpolJPAC",
  # massBinning        = HistAxisBinning(
  #   nmbBins = 25, minVal = 0.4, maxVal = 1.40,
  #   # nmbBins = 2, minVal = 0.4, maxVal = 0.42
  #   _var    = KinematicBinningVariable(
  #     name      = "mass",
  #     label     = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}",
  #     unit      = "GeV/#it{c}^{2}",
  #     nmbDigits = 3,
  #   ),
  # ),
)


# configuration for polarized gamma p -> (pi+ pi-) p data
def inputFilePathPiPiPol(
  cfg:          AnalysisConfig,
  dataType:     AnalysisConfig.DataType,
  dataPeriod:   str,
  tBinLabel:    str,
  beamPolLabel: str
) -> str:
  """Returns path of data file in input format based on data type, data period, t bin label, and beam polarization label for polarized pi+ pi- data"""
  if dataType == AnalysisConfig.DataType.REAL_DATA:
    return f"{cfg.inputDataDirBasePath(dataPeriod, tBinLabel)}/tree_data_{beamPolLabel}.root"
  elif dataType == AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE:
    return f"{cfg.inputDataDirBasePath(dataPeriod, tBinLabel)}/tree_accepted*.root"
  elif dataType == AnalysisConfig.DataType.GENERATED_PHASE_SPACE:
    return f"{cfg.inputDataDirBasePath(dataPeriod, tBinLabel)}/tree_thrown*.root"
  else:
    raise ValueError(f"Unknown data type: {dataType}")

CFG_POLARIZED_PIPI = AnalysisConfig(
  dataDirBasePath    = "./dataPhotoProdPiPi/polarized",
  dataPeriods        = (
    # "2017_01",
    "2017_01_ver05",  #!NOTE! SDME analysis: 0.60 < m_pipi < 0.88 GeV
    # "2018_08",
  ),
  tBinLabels         = (
    "tbin_0.100_0.114",  # lowest |t| bin of SDME analysis  #TODO actual upper limit seems to 0.11364635 GeV^2
    # "tbin_0.1_0.2",
    # "tbin_0.2_0.3",
    # "tbin_0.3_0.4",
    # "tbin_0.4_0.5",
  ),
  beamPolLabels      = (
    "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
    # "AMO",
  ),
  inputDataFormats   = {
    AnalysisConfig.DataType.REAL_DATA             : AnalysisConfig.DataFormat.AMPTOOLS,
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : AnalysisConfig.DataFormat.AMPTOOLS,
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : AnalysisConfig.DataFormat.AMPTOOLS,
  },
  _inputFilePath     = inputFilePathPiPiPol,  # use custom function to generate paths of data files in input format for polarized data
  maxLs              = (
    4,
    6,
    8,
  ),
  outFileDirBasePath = "./plotsPhotoProdPiPiPol",
  massBinning        = HistAxisBinning(
    # nmbBins = 50,    # 40 MeV wide bins used in PWA of polarized data
    # minVal  = 0.28,  # [GeV]
    # maxVal  = 2.28,  # [GeV]
    nmbBins = 7,     # 40 MeV wide bins in mass range of SDME analysis; for 2017_01_ver05 data
    minVal  = 0.60,  # [GeV]
    maxVal  = 0.88,  # [GeV]
    _var    = KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{#pi}^{#plus}#it{#pi}^{#minus}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    ),
  ),
)


# configuration for unpolarized gamma p -> (pi+ p) pi- data
CFG_UNPOLARIZED_PIPP = AnalysisConfig(
  subsystem          = SubsystemInfo(  # pi+ p subsystem to analyze; pi+ is the analyzer
    lvALabel          = "pip",    # label of pi+ Lorentz-vector
    lvBLabel          = "recoil", # label of proton Lorentz-vector
    lvRecoilLabel     = "pim",    # label of "recoil" pi- Lorentz-vector
    pairLabel         = "PipP",
    ATLatexLabel      = "#it{#pi}^{#plus}",
    BTLatexLabel      = "#it{p}",
    recoilTLatexLabel = "#it{#pi}^{#minus}",
    pairTLatexLabel   = "#it{#pi}^{#plus}#it{p}",
  ),
  maxLs              = (
    4,
    6,
    8,
  ),
  outFileDirBasePath = "./plotsPhotoProdPipPUnpol",
  massBinning        = HistAxisBinning(
    nmbBins = 75,   # 20 MeV wide bins
    minVal  = 1.1,  # [GeV]
    maxVal  = 2.6,  # [GeV]
    _var    = KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{#pi}^{#plus}#it{p}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    ),
  ),
)


# configuration for Nizar's polarized gamma p -> (eta pi0) p data, with eta -> gamma gamma
CFG_POLARIZED_ETAPI0 = AnalysisConfig(
  frame              = AnalysisConfig.CoordSysType.GJ,
  subsystem          = SubsystemInfo(  # eta pi0 subsystem to analyze; eta is the analyzer
    lvALabel          = "eta",     # label of eta Lorentz-vector
    lvBLabel          = "pi0",     # label of pi0 Lorentz-vector
    lvRecoilLabel     = "recoil",  # label of recoil-proton Lorentz-vector
    pairLabel         = "EtaPi0",
    ATLatexLabel      = "#it{#eta}",
    BTLatexLabel      = "#it{#pi}^{0}",
    recoilTLatexLabel = "#it{p}",
    pairTLatexLabel   = "#it{#eta}#it{#pi}^{0}",
  ),
  dataDirBasePath    = "./dataPhotoProdEtaPi0/polarized",
  dataPeriods        = ("merged", ),  # merged Phase-I + Spring 2020 data with different polarization values
  tBinLabels         = (
    "t010020",
    "t020032",
    "t032050",
    "t050075",
    "t075100",
  ),
  beamPolLabels      = ("All", ),  # input files contain all beam polarization orientations merged together
  inputDataFormats   = {
    AnalysisConfig.DataType.REAL_DATA             : AnalysisConfig.DataFormat.AMPTOOLS,
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : AnalysisConfig.DataFormat.AMPTOOLS,
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : AnalysisConfig.DataFormat.AMPTOOLS,
  },
  convertedTreeName  = "EtaPi0",
  maxLs              = (
    4,
    6,
    8,
  ),
  outFileDirBasePath = "./plotsPhotoProdEtaPi0",
  massBinning        = HistAxisBinning(
    nmbBins = 17,    # 40 MeV wide bins
    minVal  = 1.04,  # [GeV]
    maxVal  = 1.72,  # [GeV]
    _var    = KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{#eta}#it{#pi}^{0}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    ),
  ),
)


# configuration for Zach's polarized gamma p -> (eta' pi0) p data, with eta' -> pi+ + pi- + eta and eta -> gamma gamma
CFG_POLARIZED_ETAPPI0 = AnalysisConfig(
  subsystem          = SubsystemInfo(  # eta' pi0 subsystem to analyze; eta' is the analyzer
    lvALabel          = "etap",    # label of eta' Lorentz-vector
    lvBLabel          = "pi0",     # label of pi0 Lorentz-vector
    lvRecoilLabel     = "recoil",  # label of recoil-proton Lorentz-vector
    pairLabel         = "EtapPi0",
    ATLatexLabel      = "#it{#eta}'",
    BTLatexLabel      = "#it{#pi}^{0}",
    recoilTLatexLabel = "#it{p}",
    pairTLatexLabel   = "#it{#eta}'#it{#pi}^{0}",
  ),
  convertedTreeName  = "EtapPi0",
  maxLs              = (
    4,
    6,
    8,
  ),
  outFileDirBasePath = "./plotsPhotoProdEtapPi0",
  massBinning        = HistAxisBinning(
    nmbBins = 20,   # 40 MeV wide bins
    minVal  = 1.2,  # [GeV]
    maxVal  = 2.0,  # [GeV]
    _var    = KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{#eta}'#it{#pi}^{0}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    ),
  ),
)


# configuration for Will's unpolarized gamma p -> (eta' eta) p data
CFG_UNPOLARIZED_ETAPETA = AnalysisConfig(
  subsystem          = SubsystemInfo(  # eta' eta subsystem to analyze; eta' is the analyzer
    lvALabel          = "etap",    # label of eta' Lorentz-vector
    lvBLabel          = "eta",     # label of eta Lorentz-vector
    lvRecoilLabel     = "recoil",  # label of recoil-proton Lorentz-vector
    pairLabel         = "EtapEta",
    ATLatexLabel      = "#it{#eta}'",
    BTLatexLabel      = "#it{#eta}",
    recoilTLatexLabel = "#it{p}",
    pairTLatexLabel   = "#it{#eta}'#it{#eta}",
  ),
  dataDirBasePath    = f"./dataPhotoProdEtapEta/unpolarized",
  dataPeriods        = (
    "2017_01",
    "2018_01",
    "2018_08",
    "2019_11",
  ),
  tBinLabels         = (
    # "ALLT",
    # "LOWT",
    "XSCUTS",
  ),
  beamPolLabels      = ("Unpol", ),
  inputDataFormats   = {
    AnalysisConfig.DataType.REAL_DATA             : AnalysisConfig.DataFormat.FSROOT,
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : AnalysisConfig.DataFormat.FSROOT,
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : AnalysisConfig.DataFormat.FSROOT,
  },
  # inputTreeName      = "kin",  # for 2018_08 ALLT real data
  inputTreeName      = "nt",
  convertedTreeName  = "EtapEta",
  maxLs              = (
    4,
    6,
    8,
  ),
  outFileDirBasePath = "./plotsPhotoProdEtapEta",
  # plotMomentsInBins  = True,
  massBinning        = HistAxisBinning(
    # nmbBins = 8,    # 250 MeV wide bins
    # minVal  = 1.5,  # [GeV]
    # maxVal  = 3.5,  # [GeV]
    nmbBins = 15,   # 100 MeV wide bins
    minVal  = 1.5,  # [GeV]
    maxVal  = 3.0,  # [GeV]
    _var    = KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{#eta}'#it{#eta}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    ),
  ),
)


# configuration for Kevin's gamma p -> (K- K_S) Delta++ data
CFG_KEVIN = AnalysisConfig(
  subsystem                = SubsystemInfo(  # K- K_S subsystem to analyze; K- is the analyzer
    lvALabel          = "K-",      # label of K- Lorentz-vector
    lvBLabel          = "K_S",     # label of K_S Lorentz-vector
    lvRecoilLabel     = "recoil",  # label of recoil-proton Lorentz-vector
    pairLabel         = "KmKS",
    ATLatexLabel      = "#it{K}^{#minus}",
    BTLatexLabel      = "#it{K}_{S}^{0}",
    recoilTLatexLabel = "#it{p}",
    pairTLatexLabel   = "#it{K}^{#minus}#it{K}_{S}^{0}",
  ),
  inputTreeName            = "ntFSGlueX_100_11100_angles",
  convertedTreeName        = "KmKS",
  # dataFileName             = "./dataPhotoProdKmKS/data/pipkmks_100_11100_B4_M16_*_SKIM_A2.root.angles",
  # psAccFileName            = "./dataPhotoProdKmKS/phaseSpace/pipkmks_100_11100_B4_M16_SIGNAL_SKIM_A2.root.angles",
  # psGenFileName            = "./dataPhotoProdKmKS/phaseSpace/pipkmks_100_11100_B4_M16_MCGEN_GENERAL_SKIM_A2.root.angles",
  # polarization             = "beamPol",  # read polarization from tree column
  # maxL                     = 4,
  maxLs                    = (
    4,
    6,
    8,
  ),
  outFileDirBasePath       = "./plotsTestKevin",
  # normalizeMoments         = True,
  # plotAngularDistributions = True,
  # plotAccIntegralMatrices  = True,
  massBinning              = HistAxisBinning(
    nmbBins = 20,   # 40 MeV wide bins; original binning: 200 bins in [0.6, 2.6] GeV
    minVal  = 1.0,  # [GeV]
    maxVal  = 1.8,  # [GeV]
    _var    = KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{K}^{#minus}#it{K}_{S}^{0}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    ),
  ),
)


# configuration for Gabriel's gamma p -> K_S (K_L) p data
CFG_POLARIZED_KSKL = AnalysisConfig(
  subsystem          = SubsystemInfo(  # K_S K_L subsystem to analyze; K_L is the analyzer
    lvALabel          = "K_L",     # label of K_S Lorentz-vector
    lvBLabel          = "K_S",     # label of K_L Lorentz-vector
    lvRecoilLabel     = "recoil",  # label of recoil-proton Lorentz-vector
    pairLabel         = "KSKL",
    ATLatexLabel      = "#it{K}_{L}^{0}",
    BTLatexLabel      = "#it{K}_{S}^{0}",
    recoilTLatexLabel = "#it{p}",
    pairTLatexLabel   = "#it{K}_{S}^{0}#it{K}_{L}^{0}",
  ),
  dataDirBasePath    = "./dataPhotoProdKSKL/polarized",
  dataPeriods        = ("merged", ),  # merged Phase-I(?) data with different polarization values
  tBinLabels         = ("AllT", ),
  beamPolLabels      = (
    "PARA_0",
    # "PARA_135",
    # "PERP_45",
    # "PERP_90",
  ),
  inputDataFormats   = {
    AnalysisConfig.DataType.REAL_DATA             : AnalysisConfig.DataFormat.FSROOT,
    AnalysisConfig.DataType.ACCEPTED_PHASE_SPACE  : AnalysisConfig.DataFormat.FSROOT,
    AnalysisConfig.DataType.GENERATED_PHASE_SPACE : AnalysisConfig.DataFormat.FSROOT,
  },
  inputTreeName      = "flatTree",
  convertedTreeName  = "KSKL",
  maxLs              = (
    4,
    6,
    8,
  ),
  outFileDirBasePath = "./plotsPhotoProdKSKL",
  massBinning        = HistAxisBinning(
    nmbBins = 70,   # 20 MeV wide bins
    minVal  = 1.2,  # [GeV]
    maxVal  = 2.6,  # [GeV]
    _var    = KinematicBinningVariable(
      name      = "mass",
      label     = "#it{m}_{#it{K}_{S}^{0}#it{K}_{L}^{0}}",
      unit      = "GeV/#it{c}^{2}",
      nmbDigits = 3,
    ),
  ),
)
