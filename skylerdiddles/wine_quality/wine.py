#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import scipy.stats
import pybrain.datasets

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



def correlation_matrix(data, target_name=None, verbose=False):

  # Matrix of predictor data
  predictors = [data[x] for x in data if x != target_name]

  # Compute correlation coefficients matrix
  cor_matrix = np.corrcoef(predictors)
  
  if verbose:
    # Print out matrix; mask out values < 0.7
    for row in cor_matrix: 
      row = " ".join(["{:3.2}".format(x) for x in row ])
      print(row)

  return cor_matrix


def zscores(data, target="None"):

  # Create a copy of the data frame which will hold the z-scores
  zdata = pd.DataFrame.copy(data, deep=True)

  # Iterate over all columns and replace with z-scores (ignores target column)
  for key in zdata:

    if key != target:
      std = np.nanstd(zdata[key])
      mean = np.nanmean(zdata[key])     
      z = (zdata[key]-mean) / std
      
      zdata[key] = z

  return zdata


def construct_NN(data):
  alldata = pybrain.datasets.ClassificationDataSet(2, 1, nb_classes=3)


class NeuralNet:

  def __init__(self, data, target):
    self.target = data[target]  
    self.predictors = data.drop(target, axis=1)
    self.N_predictors = len(self.predictors.columns)
    self.N_target_vals = len(set(self.target))
    
    # Number of data points in target vector
    N = self.target.shape[0]

    # Validate the data
    assert N > 0
    assert self.target.shape == (N,)
    assert self.predictors.shape == (N, self.N_predictors)
    assert self.N_target_vals > 1
    assert self.N_predictors > 0


  def construct(self):
    # Contsuct dataset structure for NN
    alldata = pybrain.datasets.ClassificationDataSet(self.N_predictors, 1, nb_classes=self.N_target_vals)
    alldata.addSample(self.predictors, self.target)

    # Construct the training and testing data sets
    tstdata, trndata = alldata.splitWithProportion(0.25)
    
    # Initialize training and testing data sets
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()


def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  args = parser.parse_args()

  # Identify target column
  target = "quality"

  # Readx data into a pandas data frame
  data = read_data(args.input)

  # Compute correlation matrix
  cor_matrix = correlation_matrix(data, verbose=False)  

  # Transform data to z-scores (mean=0, std=1)
  zdata = zscores(data, target=target)


  NNdata = NeuralNet(zdata, target)

  NNdata.construct()

if __name__ == "__main__":
  main()

