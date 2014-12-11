import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd


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

  plt.show()


def correlation_matrix(data, target_name):

  # Matrix of predictor data
  predictors = [data[x] for x in data if x != target_name]

  # Compute correlation coefficients matrix
  cor_matrix = np.corrcoef(predictors)
  
  # Print out matrix; mask out values < 0.75
  for row in cor_matrix: 
    row = " ".join(["{:2}".format(x) if np.abs(x) > 0.75 else `0` for x in row ])
    print(row)

  return cor_matrix


def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  args = parser.parse_args()

  target = "quality score"

  data = read_data(args.input)

  cor_matrix = correlation_matrix(data, target)
  
  
  make_histograms(data)

if __name__ == "__main__":
  main()

