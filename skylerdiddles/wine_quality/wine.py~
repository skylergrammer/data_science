#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import scipy.stats
import sklearn.cluster
import pybrain.datasets
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError
import random

def read_data(filename):
  '''
  Read data into a pandas dataframe.
  '''
  data = pd.read_csv(filename, sep=";")

  return data


def make_histograms(data):
  '''  
  Create histograms for the predictors and target.
  
  Requires a panda data frame as input.
  '''
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
  '''
  Compute the correlation matrix for all predictors (does not include target).

  Requires a pandas data frame, dictionary, or numpy rec array.
  '''
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
  '''
  Normalizes the predictors to a mean=0 and standard-deviation = 1. Excludes the
  target.  Creates a copy of the pandas data frame so that the raw values may be
  preserved.  

  Requires a pandas data frame.
  '''
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


def contingency_table(actual, predicted):

  ratio = np.array(predicted) / np.array(actual)

  success = np.where(ratio == 1.0)[0]

  return float(len(success))/float(len(ratio))


def recode_to_nbins(x, nbins=3, use_split_vals=False):

  # Make into array
  x = np.array(x)

  # Determine bin boundaries
  freq,bin_edges = np.histogram(x, bins=nbins)
  
  print("Range: Number: Value")

  # Iterate through bin edges and bin data
  for i,bin_index in enumerate(bin_edges[:-1]):
    if i < nbins-1:
      in_bin_i = np.where((x < bin_edges[i+1]) & (x >= bin_edges[i]))[0]
     
      inrange = " ".join([str(bin_edges[i]),"<= x  < "+str(bin_edges[i+1])])
      nvals = str(len(in_bin_i))
      print(" : ".join([inrange, nvals, str(i)]))
    
    else:
      in_bin_i = np.where((x <= bin_edges[i+1]) & (x >= bin_edges[i]))[0]

      inrange = " ".join([str(bin_edges[i]),"<= x <= "+str(bin_edges[i+1])])
      nvals = str(len(in_bin_i))
      print(" : ".join([inrange, nvals, str(i)]))

    x[in_bin_i] = i+1

  return x


class NeuralNet:
  '''
  Class that holds the data in a format that can be input into a Pybrain neural
  network.  
  '''
  def __init__(self, data, target, nclasses=10):
    
    # Target classes: converted 1-N to 0-N-1
    self.target = data[target]-1
    
    # Predictor values
    self.predictors = np.array([data[key] for key in data if key != target]).T

    # Number of predictors    
    self.N_predictors = len(data.columns)-1 
    
    # Total number of values the target may take  
    self.nclasses = nclasses
    # Number of data points in target vector
    N = self.target.shape[0]
    self.N = N

    # Validate the data
    assert N > 0
    assert self.target.shape == (N,)
    assert self.predictors.shape == (N, self.N_predictors)
    assert self.N_predictors > 0

  
  def constructNN(self, lr=0.1, wd=0.01, p=0.1, lrdecay=0.0):
    
    # Contsuct dataset structure for NN  
    ds = pybrain.datasets.ClassificationDataSet(self.N_predictors, 1, 
                                                nb_classes=self.nclasses)
    ds.setField("target", self.target.reshape(-1,1))
    ds.setField("input", self.predictors)
   
    # Construct the training and testing data sets
    ds._convertToOneOfMany()
    
    '''  
    print "\nNumber of training patterns: ", len(ds)
    print "Input and output dimensions: ", ds.indim, ds.outdim
    print "First sample (input, target, class):"
    print ds['input'][0], ds['target'][0], ds['class'][0]
    print "Learning rate = %s" % lr    
    print "Weight decay = %s" % wd
    print "Momentum = %s" % p
    print "Learning rate decay = %s\n" % lrdecay
    '''

    # Number of neurons per layer
    input_layer_nodes = ds.indim
    hidden_layer_nodes = (ds.indim+ds.outdim) / 1.5
    output_layer_nodes = ds.outdim

    # Construct the network object 
    network = buildNetwork(input_layer_nodes, hidden_layer_nodes, output_layer_nodes, 
                           bias=True, outclass=SoftmaxLayer)
    
    # Set training methodology for network    
    trainer = BackpropTrainer(network, ds, learningrate=lr, momentum=p,
                              weightdecay=wd, lrdecay=lrdecay)
    
    self.ds = ds
    self.network = network
    self.trainer = trainer
 

  def runNN(self, epochs=20):

    # Train network until error function converges
    x,y = self.trainer.trainUntilConvergence(verbose = False, validationProportion = 0.10, 
                                       maxEpochs = epochs, continueEpochs=10)

    # Network predicted values for data set
    prediction = self.trainer.testOnClassData(self.ds)

    return (prediction, x, y)
    



def lets_figure_shit_out(NNdata, niter=100):

  learning_rates_decay = np.linspace(0, 1.0, 11)
  learning_rates = np.logspace(0,5, 6)*0.00001
  momenta = np.linspace(0, 1, 11)
  weights_decay = np.linspace(0, 1, 11)

  output = []

  for each in range(niter):

    lrdecay = random.choice(learning_rates_decay)
    lr = random.choice(learning_rates)
    p = random.choice(momenta)
    wd = random.choice(weights_decay)
    epoch = random.choice(max_epochs)

    NNdata.constructNN(lr=lr, lrdecay=lrdecay, wd=wd, p=p)
    predicted_quality, x, y = NNdata.runNN(epochs=10)
    accuracy = contingency_table(NNdata.target, predicted_quality)
    
    params = (each, epoch, lrdecay, lr, p, wd, accuracy)

    print("learning rate decay = %s" % lrdecay)
    print("learning rate = %s" % lr)
    print("weights decay = %s" % wd)
    print("momentum = %s" % p)
    print("Max epochs = %s" % 10)
    print("Accuracy = %0.2f" % accuracy)
    exit()

def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  args = parser.parse_args()

  # Identify target column
  target = "quality"

  # Readx data into a pandas data frame
  data = read_data(args.input)

  #data["quality"] = recode_to_nbins(data["quality"], nbins=2)

  # Compute correlation matrix
  cor_matrix = correlation_matrix(data, verbose=False)  

  # Transform data to z-scores (mean=0, std=1)
  zdata = zscores(data, target=target)

  # Object to hold data readable by neural network
  NNdata = NeuralNet(zdata, target, nclasses=10)

  lets_figure_shit_out(NNdata, niter=10)
  
  '''
  # Construct the neural network
  NNdata.constructNN(lr=1.0, lrdecay=0.1, wd=0.5, p=0.8)

  # Train the neural network and output predicted values
  predicted_quality, x, y = NNdata.runNN(epochs=1)

  # Compute the accuracy neural network predictions
  accuracy = contingency_table(NNdata.target, predicted_quality)
  
  print("\nAccuracy = %s.\n" % accuracy)
  '''
if __name__ == "__main__":
  main()

