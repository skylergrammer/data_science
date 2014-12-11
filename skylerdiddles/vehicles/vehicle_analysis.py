import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import sklearn.linear_model

def readData(filename):

  f = open(filename,"r")
  header = f.readline().split(",")
  f.close()

  data = np.genfromtxt(filename, names=header, skip_header=1,
                       delimiter=',', skip_footer=0, autostrip=True,
                       usecols=range(0,len(header)-2), missing_values=("???"), 
                       filling_values=np.nan)
  random.shuffle(data)
  return data


def normalize_data(x):

  # Stats
  x_std = np.nanstd(x)
  x_mean = np.nanmean(x)

  # 3sigma clipping
  diff = lambda x, x0: np.abs(x-x0)
  x = np.array([i if diff(i, x_mean) < 3*x_std else np.nan for i in x])
  
  x_range = np.nanmax(x) - np.nanmin(x)
  min_x = np.nanmin(x)

  # Normalize data to 0:1 range
  if x_range == 0.0:
    transform_x = [] 
  else:      
    transform_x = (x - min_x)/x_range
    
  return transform_x


class Transformed():

  def __init__(self, data, predictors, target):
    
    # Target attribute to hold raw/transformed data
    if target[1] == "log":     
      print("\nNatural log transforming %s\n" % target[0])
      self.y = np.log(data[target[0]])
    else:
      self.y = data[target[0]]
    
    # Attribute to hold normalized target values
    self.ny = normalize_data(self.y)
    

    # Predictor attributes
    normarr = []
    rawarr = []
    for key in predictors:

      # Raw/transformed predictor attributes
      if key[1] == "log":
        print("\nNatural log transforming %s\n" % key[0])
        x_trans = np.log(data[key[0]])
        x_norm = normalize_data(x_trans)
      else:
        x_trans = data[key[0]]
        x_norm = normalize_data(data[key[0]])

      # Normalized predictor attributes
      if any(x_norm):
        rawarr.append(x_trans)
        normarr.append(x_norm)
      else:
        continue
 
   #Predictor attribute is a list
    self.nx = normarr
    self.x = rawarr


def get_n_bins(x0):

  # Remove nans
  x = x0[~np.isnan(x0)]
  
  # Stats
  x_std = np.std(x)
  npoints = len(x)  
  x_range = np.max(x) - np.min(x)
  iqr = lambda x: float(np.subtract(*np.percentile(x, [75, 25])))

  # Bin width
  h = 2 * iqr(x) * np.power(npoints, -0.333)

  # Number of bins
  nbins = x_range / h

  return int(nbins)
  

def ml_linear_regression(xi, y):

  regr = sklearn.linear_model.LinearRegression()
  
  regr.fit(xi, y)
  
  return regr


def remove_nans(xi, y):
  x1, x2 = xi
  foo = zip(x1, x2, y)

  no_nan_ind = [i for i,each in enumerate(foo) if not np.any(np.isnan(each))]
  
  return no_nan_ind  


class RegressionMatrices:

  def __init__(self, xi, y, train_frac):
    x1, x2 = xi

    # Remove rows that have NaNs on the predictors or target
    x1 = x1[remove_nans(xi, y)]
    x2 = x2[remove_nans(xi, y)]
    y = y[remove_nans(xi, y)]
    
    N = int(train_frac*len(y))
  
    # Partition out the training data
    x1_train = x1[:N]
    x2_train = x2[:N]
    y_train = y[:N]

    # Partition out the testing data
    x1_test = x1[N:]
    x2_test = x2[N:]
    y_test = y[N:]

    # Create the training matrices
    self.xmatrix_train = np.matrix([x1_train, x2_train]).reshape(-1,2)
    self.ymatrix_train = y_train.reshape(-1,1)
    self.x1matrix_train = x1_train.reshape(-1,1)
    self.x2matrix_train = x2_train.reshape(-1,1)

    # Create the testing matrices
    self.xmatrix_test = np.matrix([x1_test, x2_test]).reshape(-1,2)
    self.ymatrix_test = y_test.reshape(-1,1)  
    self.x1matrix_test = x1_test.reshape(-1,1)
    self.x2matrix_test = x2_test.reshape(-1,1)

def main():

  parser = argparse.ArgumentParser(description="Performs bivariate linear regression.")
  parser.add_argument("file", help="Data file.")
  args = parser.parse_args()

  # Read in data
  data = readData(args.file)
  
  # Set target and predictors as well as any transformation (log or linear)
  target = ("co2", "log")
  predictors = [("comb08", "log"), ("displ", "log")]
  
  # Transform and normalize the data
  data_normal = Transformed(data, predictors, target)

  reg_matrices = RegressionMatrices(data_normal.x, data_normal.y, train_frac=0.7)
  
  # Run the linear regression models on each predictor
  modelx1 = ml_linear_regression(reg_matrices.x1matrix_train, reg_matrices.ymatrix_train)  
  rsqrx1 = modelx1.score(reg_matrices.x1matrix_test, reg_matrices.ymatrix_test)

  modelx2 = ml_linear_regression(reg_matrices.x2matrix_train, reg_matrices.ymatrix_train)  
  rsqrx2 = modelx2.score(reg_matrices.x2matrix_test, reg_matrices.ymatrix_test)
  
  # Plot the results
  fig1 = plt.figure()
  plt.plot(reg_matrices.x1matrix_train, reg_matrices.ymatrix_train, 'bo',
           label="Training")
  plt.plot(reg_matrices.x1matrix_test, reg_matrices.ymatrix_test, 'go',
           label="Testing")
  plt.plot(reg_matrices.x1matrix_test, modelx1.predict(reg_matrices.x1matrix_test),
           'r', lw=3, label="Model") 
  plt.legend(frameon=False)
  plt.title("$R^{2}$ = %s" % rsqrx1)
  plt.xlabel("$ln(mpg)$")
  plt.ylabel("$ln(CO_{2} Output)$")

  fig2 = plt.figure()
  plt.plot(reg_matrices.x2matrix_train, reg_matrices.ymatrix_train, 'bo',
           label="Training")
  plt.plot(reg_matrices.x2matrix_test, reg_matrices.ymatrix_test, 'go',
           label="Testing")
  plt.plot(reg_matrices.x2matrix_test, modelx2.predict(reg_matrices.x2matrix_test),
           'r', lw=3, label="Model") 
  plt.legend(frameon=False)
  plt.title("$R^{2}$ = %s" % rsqrx2)
  plt.xlabel("$ln(Displacement)$")
  plt.ylabel("$ln(CO_{2} Output)$")

  plt.show()


  #print model.score(reg_matrices.xmatrix_test, reg_matrices.ymatrix_test)
  
if __name__ == "__main__":
  main()
