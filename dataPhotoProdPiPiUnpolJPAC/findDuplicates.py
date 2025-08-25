#!/usr/bin/env python3


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


if __name__ == "__main__":

  # read data
  inputFileName =  "./mc/mc_full_model/mc0.4-0.5_ful.dat"
  print(f"Reading file '{inputFileName}'")
  pandasDf = pd.read_csv(inputFileName, sep = r"\s+")

  # count event multiplicities
  array = pandasDf.to_numpy()
  columns = pandasDf.columns
  events, multiplicities = np.unique(array, axis = 0, return_counts = True)
  print(f"Out of {len(array)} events, {len(events[multiplicities > 1])} appear more than once contributing in total {np.sum(multiplicities[multiplicities > 1])} events to the sample")

  # plot distribution of duplicate event counts
  maxCount = np.max(multiplicities)
  bins = np.arange(0.5, maxCount + 1.5)
  plt.hist(multiplicities[multiplicities == 1], bins = bins, log = True, color = "0.6")  # events that are unique
  plt.hist(multiplicities[multiplicities >  1], bins = bins, log = True)                 # events with duplicates
  # major ticks every 5, minor ticks at each integer
  ax = plt.gca()
  ax.xaxis.set_major_locator(MultipleLocator(5))
  ax.xaxis.set_minor_locator(MultipleLocator(1))
  plt.xlim(0.5, np.max(multiplicities) + 0.5)
  plt.xlabel("Multiplicity of Event")
  plt.ylabel("Frequency")
  plt.savefig("frequency_of_event_multiplicity.pdf")
  plt.show()
