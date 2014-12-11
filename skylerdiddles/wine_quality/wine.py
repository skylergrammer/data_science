#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import scipy.stats

def read_data(filename):

  data = pd.read_csv(filename, sep=";")

  return data


def make_histograms(data):
   
  # Id number of columns
  col_num = len(data.columns)

  # Create number of rows and columns
  nrows = int(np.sqrt(col_num))
  ncols = int(col_num / nrows) 

  # Create the figure space
  fig, ax = plt.subplots(figsize=(11,8.5), nrows=nrows, ncols=ncols)
  fig.subplots_adjust(wspace=0.1, hspace=0.15, left=0.05, right=0.95)
  
  # Reshape the vector of columnn names to a matrix of (nrows,ncols)
  keys = np.array([x for x in data]).reshape(ax.shape)

  # Iterate over the axis and column-name matrices and plot histogram
  for i,keyi in enumerate(keys):
    for j,keyj in enumerate(keyi):
      ax[i,j].hist(data[keys[i,j]], bins=20)
      ax[i,j].set_title(keys[i,j])
      ax[i,j].tick_params(axis='both', which='both', bottom="off",left="off",
                          labelbottom="off", labelleft="off")



def correlation_matrix(data, target_name=None):

  # Matrix of predictor data
  predictors = [data[x] for x in data if x != target_name]

  # Compute correlation coefficients matrix
  cor_matrix = np.corrcoef(predictors)
  
  # Print out matrix; mask out values < 0.7
  for row in cor_matrix: 
    row = " ".join(["{:3.2}".format(x) if np.abs(x) > 0.75 else `0.00` for x in row ])
    print(row)

  return cor_matrix


def zscores(data, target="None"):

  for key in data:
    if key != target:
      std = np.nanstd(data[key])
      mean = np.nanmean(data[key])
      
      z = (data[key]-mean) / std
      data[key] = z

  return data


def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  args = parser.parse_args()

  target = "quality"

  data = read_data(args.input)

  cor_matrix = correlation_matrix(data)  

  zdata = zscores(data, target=target)


if __name__ == "__main__":
  main()

