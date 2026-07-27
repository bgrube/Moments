[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bgrube/Moments)

# Framework for moment analysis of two-(pseudo)scalar meson systems from data of the GlueX experiment

## Moment calculation

The `moments` package provides the code to perform the actual calculation of moments and their covariance matrix from data.

### Data format

The data are expected in the form of ROOT `RDataFrame` objects that must contain the following columns

- `theta`: polar angle (in rad) of one of the two mesons (the analyzer) in a suitable rest frame of the meson pair, e.g. helicity or Gottfried-Jackson frame
- `phi`: corresponding azimuthal angle (in rad)

For linearly polarized photoproduction, an additional column is required:

- `Phi`: azimuthal angle (in rad) of the polarization plane of the beam photon and the production plane, which is spanned by the beam and the meson pair

The degree of the beam polarization can be passed either in form of a constant value or as a string that defined the column name in the `RDataFrame`.

If the `RDataFrame` contains an `eventWeight` columns these weights are applied to the data. This can be used to implement background subtraction schemes.

To perform a moment analysis, the user has to provide the measured data and a sample of Monte Carlo (MC) data for acceptance correction. This MC sample has to be generated assuming a phase-space distribution of the two-meson subsystem and then passed through the detector simulation, event reconstruction, and event selection similar to the real-data sample. To ensure a correct absolute acceptance correction, in addition the number of generated phase-space events that is passed to the detector simulation is required. The `DataSet` class collects all the information required for the moment calculation.

## Workflow

The `workflow` package provides library code used by the scripts in the `scripts` directory to manage data, perform the moment calculations, and plot the results. The `AnalysisConfig` class collects all information needed for the moment analysis. A typical workflow is outlined below.

### Step 1: Conversion of data into format expected by `moments` package

The script `scripts/convertInputData.py` implements the conversion of common ROOT tree formats used in GlueX into the tree format expected by the `moments` package. The function `AnalysisConfig.inputFilePath()` defines the path of the input data for each data type (i.e. real data, generated phase-space MC, and accepted phase-space MC) for a given data-taking period (e.g. "2017_01"), a label indicating the analyzed t bin (e.g. "tbin_0.1_0.2"), and a label for the beam polarization (e.g. "PARA_135" or "Unpol"). The function `AnalysisConfig._inputFilePath()` that generates the file paths can be overridden when constructing the `AnalysisConfig` object. The `DataConversionUtilities.py` module provides functions that perform the data conversion. In particular, `AnalysisConfig.lorentzVectors()` maps columns in the input trees to the Lorentz-vectors of the initial- and final-state particles. Additional data formats and reactions channels have to be defined here. The particles are defined by the `AnalysisConfig.subsystem` member. `AnalysiConfig.inputDataFormats` defines the format of each input data. The paths of the converted files are defined by `AnalysisConfig.convertedFilePath()`, which is currently not overridable.

The data samples to process are defined the `dataPeriods`, `tBinLabels`, and `beamPolLabels` members of `AnalysisConfig`.

GlueX real data are often separated into two files, one containing the signal region and the other the sideband regions used for background subtraction. If the weights in the sideband file have the correct sign, the two files can simply be `hadd`ed or chained to obtain the real-data input file for `scripts/convertInputData.py`. However, trees prepared for AmpTools contain sideband weights with the opposite sign. The script `script/mergeSigAndBkgTrees.py` merges these trees and corrects the sign of the sideband wight accordingly.

#### Step 1A: Plot kinematic distributions (optional)

The script `scripts/plotKinematicDistributions.py` generated histograms for the basic kinematic distributions. Binning and plotting ranges of the histograms need to be adjusted to the analyzed channel.

### Step 2: Calculation of moments

The script `scripts/calculateMoments.py` calculates the moments from the converted data and writes them in the form of `.pkl` files into subdirectories below the path defined by `AnalysisConfig.outFileDirBasePath`. Currently, the moment calculation can be performed in two frames, helicity or Gottfried-Jackson, defined by `AnalysisConfig.frame`. The maximum $L$ quantum number $L_\text{max}$, that is used in the moment calculation, is defined by the `AnalysisConfig.maxLs` tuple. The script calculates the moments separately for each entry in the tuple. If the entry is a single integer, the $L_\text{max}$ value is used for both measured and physical moments, i.e. the acceptance integral matrix is a square matrix. If the entry is a tuple of two integers, the first element defines $L_\text{max}^\text{phys}$ for the physical moments, the second element, which must not be smaller than the first one, $L_\text{max}^\text{meas}$ for the measured moments, i.e. the acceptance integral matrix is a rectangular matrix. The moment calculation is performed independently for each mass bin defined by `AnalysisConfig.massBinning`.

#### Step 2A: Combine moments from independent data samples

The moment values (and their covariances) calculated from statistically independent data samples, e.g. the four beam-polarization orientations, can be combined using the `scripts/combineMoments.py` script.

### Step 3: Plot moments

The script `scripts/plotMoments.py` reads the moment values from the `.pkl` files created in the previous step and generates plots. Which plots are generated can be steered by a set of flags in `AnalysisConfig`. The files for the generated plots are written into the directory containing the `.pkl` file. The script also allows to load another `.pkl` file to overlay as a comparison. This could be, for example, the moment values calculated from partial-wave amplitudes. The script `scripts/calculateMomentsFromPwa.py` calculates moments from the result of a mass-independent partial-wave analysis and writes them into a `.pkl` file. To quantify the agreement between the two sets of moment values, `scripts/plotMoments.py` can calculate and plot the corresponding $\chi^2$ values.

The script `overlayMoments.py` can be used to overlay more than two sets of moments.

---

## Further reading

[1] B. Grube, "Moment analysis of two-(pseudo)scalar meson systems", <https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124>, <https://github.com/bgrube/moments-two-pseudoscalars>

[2] B. Grube, "Uncertainty propagation for functions of complex random variables", <https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6125>, <https://github.com/bgrube/uncertainty-propagation-complex>
