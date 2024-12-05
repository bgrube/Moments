#!/usr/bin/python3
#!NOTE! only on ifarm this shebang selects the correct Python3 version for ROOT

from datetime import datetime
import glob
import multiprocessing
from multiprocessing.pool import ThreadPool
import os.path
import subprocess
import zipfile

import runList


TIME_COMMAND = "/usr/bin/time --verbose "  # times process and prints other process information (requires GNU time; bash builtin does not provide this option); trailing space is required


def callProcess(command):
  with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, universal_newlines = True) as process:
    #!NOTE! 'universal_newlines = True' becomes 'text = True' from Python 3.7 on
    out, err = process.communicate()
    return (out, err)


#TODO implement mode where all files that belong to the same run are merged
def runFlatten(
  inputFileNamePattern,
  outputDirName,
  flattenExePath,
  flattenCuts,
  nmbParallelJobs,
  flattenMcModeString = None
):
  assert nmbParallelJobs > 0, f"Number of parallel jobs must be > 0; got {nmbParallelJobs}"
  startTimeFlatten = datetime.now()

  # get all files that match the given naming pattern
  inputFileNames = sorted(glob.glob(inputFileNamePattern))
  if not inputFileNames:
    print(f"Found no input files for '{inputFileNamePattern}'; nothing to do")
    return
  RunNmbs = set(runList.getRunNmbsFromFileNamesInt(inputFileNames))
  print(f"Flattening trees for {len(inputFileNames)} file(s), which match '{inputFileNamePattern}' and correspond to {len(RunNmbs)} run(s), and writing flattened file(s) to '{outputDirName}'")

  os.makedirs(outputDirName, exist_ok = True)

  # see https://stackoverflow.com/a/25120960
  print(f"Running {nmbParallelJobs} flatten jobs in parallel")
  processPool     = ThreadPool(nmbParallelJobs)
  processResults  = []
  outputFileNames = []  # memorize output file names for creation of ranking trees
  nmbInputFiles   = len(inputFileNames)
  for index, inputFileName in enumerate(inputFileNames):
    outputFileName = f"{outputDirName}/{os.path.basename(inputFileName)}"
    outputFileNames.append(outputFileName)
    logFileName = f"{outputFileName}.log"
    flattenCommand = str(
      f"{TIME_COMMAND}"
      f"{flattenExePath}"
       " -print 1"
       " -usePolarization 1"
      f" {flattenCuts}"
      f' -in "{inputFileName}"'
      f' -out "{outputFileName}"'
    )
    if flattenMcModeString:
      flattenCommand += f" -mctag '{flattenMcModeString}'"
    flattenCommand += f' &> "{logFileName}"'  # redirect stdout and stderr to log file
    print(f"Starting process [{index + 1} of {nmbInputFiles}]: {flattenCommand}")
    processResults.append(processPool.apply_async(callProcess, (flattenCommand, )))
  # close process pool and wait for each process to complete
  processPool.close()
  processPool.join()
  # for processResult in processResults:
  #   out, err = processResult.get()
  #   print(f"stdout: {out}\n" + f"stderr: {err}")  # stdout and stderr are empty because they were redirected into log file
  print(f"Creating flat trees in '{outputDirName}' took {datetime.now() - startTimeFlatten}\n")
  return outputFileNames


def runRanking(
  inputFileNames,
  rankExePath,
  rankTreeName,
  rankModeString,
  rankVariableName,
  rankVariable,
  rankCuts,
  nmbParallelJobs
):
  if not inputFileNames:
    print(f"No input files; nothing to do")
    return
  assert nmbParallelJobs > 0, f"Number of parallel jobs must be > 0; got {nmbParallelJobs}"
  flatTreesDir = os.path.dirname(inputFileNames[0])
  startTimeRank = datetime.now()
  print(f"Creating ranking trees for {len(inputFileNames)} file(s) in '{flatTreesDir}'")

  print(f"Running {nmbParallelJobs} ranking jobs in parallel")
  processPool    = ThreadPool(nmbParallelJobs)
  processResults = []
  nmbInputFiles  = len(inputFileNames)
  for index, inputFileName in enumerate(inputFileNames):
    logFileName = f"{inputFileName}.{rankVariableName}.log"
    rankCommand = str(
      f"{TIME_COMMAND}"
      f"{rankExePath}"
      f" -mode '{rankModeString}'"
      f" -nt '{rankTreeName}'"
      f" -rvname '{rankVariableName}'"
      f" -rv '{rankVariable}'"
      f" -cuts '{rankCuts}'"
      f' -i "{inputFileName}"'
      f' &> "{logFileName}"'
    )
    print(f"Starting process [{index + 1} of {nmbInputFiles}]: {rankCommand}")
    processResults.append(processPool.apply_async(callProcess, (rankCommand, )))
  # close process pool and wait for each process to complete
  processPool.close()
  processPool.join()
  print(f"Creating ranking trees in '{flatTreesDir}' took {datetime.now() - startTimeRank}\n\n")


# creates zip file of all files matching given pattern
#!NOTE! saves full path of input files
#       use `unzip -j <zipFileName>` to ignore any paths and extract all files into current directory
def zipAndRemoveFiles(zipFileName, fileNamePattern):
  fileNamesToZip = sorted(glob.glob(fileNamePattern))
  if not fileNamesToZip:
    print(f"Found no files found matching '{fileNamePattern}'; nothing to do")
    return
  print(f"Zipping {len(fileNamesToZip)} files matching '{fileNamePattern}' into '{zipFileName}'")
  if fileNamesToZip:
    with zipfile.ZipFile(zipFileName, "w", zipfile.ZIP_DEFLATED) as zf:  # use zlib compression
      for fileName in fileNamesToZip:
        zf.write(fileName)
    print(f"Removing {len(fileNamesToZip)} files matching '{fileNamePattern}'")
    countFiles = 0
    for fileName in fileNamesToZip:
      if os.path.isfile(fileName):
        os.remove(fileName)
        countFiles += 1
    print(f"Removed {countFiles} files")


if __name__ == "__main__":
  #TODO add command-line interface

  # dataSet = "signal"
  dataSet = "phaseSpace"
  dirs = [
    (f"./{dataSet}/tree_pippim__B4_gen_amp_030994_???.root",  # input files
     f"./{dataSet}_FSRoot"),
  ]
  flattenMcModeString = None
  createRankingTree   = True

  flattenExePath = "/home/bgrube/Analysis/halld_my/hd_utilities/FlattenForFSRoot/flatten"
  flattenCuts    = "-combos 1 -chi2 25"  # the first option removes duplicate combos

  rankExePath       = f"{os.environ['FSROOT']}/Executables/FSModeCreateRankingTree"
  rankModeString    = "100_110"
  rankTreeName      = f"ntFSGlueX_{rankModeString}"
  rankVariableName  = "Chi2Rank"           # branch name for ranking result
  rankVariable      = "1000 * Chi2"        # value used for ranking
  rankCuts          = "abs(RFDeltaT) < 2"  # [ns]

  nmbParallelJobs   = 10  # None = number of CPU cores

  startTime = datetime.now()
  if not nmbParallelJobs:
    nmbParallelJobs = multiprocessing.cpu_count()
  for restFileNamePattern, outputDirName in dirs:

    flatFileNames = runFlatten(
      restFileNamePattern,
      outputDirName,
      flattenExePath,
      flattenCuts,
      nmbParallelJobs,
      flattenMcModeString
    )
    if flatFileNames:
      if createRankingTree:
        runRanking(
          flatFileNames,
          rankExePath,
          rankTreeName,
          rankModeString,
          rankVariableName,
          rankVariable,
          rankCuts,
          nmbParallelJobs
        )
      # for i in [""] + ([f".{rankVariableName}"] if createRankingTree else []):
      #   zipFileName = f"{os.path.splitext(runList.removeRunNmbFromFileName(flatFileNames[0]))[0]}{i}.log.zip"
      #   zipAndRemoveFiles(f"{zipFileName}", f"{outputDirName}/*.root{i}.log")

  print(f"Total execution time was {datetime.now() - startTime} seconds")
