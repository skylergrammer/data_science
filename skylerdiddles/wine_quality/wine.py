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
import progressbar as pbar


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


def success_rate(actual, predicted, show=False):

  # Difference between actual and predicted quality scores
  diff = np.array(predicted) - np.array(actual)
 
  # Number of records correctly predicted
  success = np.where(diff == 0)[0]
  # Number of almost-correctly-predicted records
  almost = np.where(np.abs(diff) == 1)[0]

  accuracy = (float(len(success))/float(len(diff)), 
              float(len(almost)+len(success))/float(len(diff))) 

  # Show a histogram of difference between predicted and actual    
  if show:
    max_diff = np.max(diff)
    min_diff = np.min(diff)
    nbins = max_diff - min_diff

    plt.hist(diff, bins=int(nbins), normed=True, align="left")
    plt.title(["%0.2f" % x for x in accuracy])
    plt.show()


  return accuracy[0]


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

  
  def constructNN(self, lr=0.1, wd=0.001, p=0.1, lrdecay=0.0, verbose=False):
    
    # Contsuct dataset structure for NN  
    ds = pybrain.datasets.ClassificationDataSet(self.N_predictors, 1, 
                                                nb_classes=self.nclasses)
    ds.setField("target", self.target.reshape(-1,1))
    ds.setField("input", self.predictors)
   
    # Construct the training and testing data sets
    ds._convertToOneOfMany()
    
    if verbose:
      print "\nNumber of training patterns: ", len(ds)
      print "Input and output dimensions: ", ds.indim, ds.outdim
      print "First sample (input, target, class):"
      print ds['input'][0], ds['target'][0], ds['class'][0]
      print "Learning rate = %s" % lr    
      print "Weight decay = %s" % wd
      print "Momentum = %s" % p
      print "Learning rate decay = %s\n" % lrdecay
    

    # Number of neurons per layer
    input_layer_nodes = ds.indim
    hidden_layer_nodes = (ds.indim+ds.outdim)*3.0
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
    print("Runnning %s training epochs." % epochs)
    # Train network until error function converges
    x,y = self.trainer.trainUntilConvergence(verbose = False, validationProportion = 0.25, 
                                             maxEpochs = epochs, continueEpochs=10)
    #self.trainer.trainEpochs(epochs=epochs)
    #self.trainer.trainOnDataset(self.ds)
    # Network predicted values for data set
    prediction = self.trainer.testOnClassData(self.ds)

    return prediction
    

def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("--v", action="store_true")
  parser.add_argument("--search", action="store_true")
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

  # Construct the neural network
  NNdata.constructNN(lrdecay=1.0, lr=0.1, p=0.0, wd=0.0, verbose=args.v)

  # Train the neural network and output predicted values
  predicted_quality = NNdata.runNN(epochs=100)

  # Compute the accuracy neural network predictions
  accuracy = success_rate(NNdata.target, predicted_quality, show=True)
  
  print("\nAccuracy = %0.2f\n" % accuracy)


if __name__ == "__main__":
  main()

