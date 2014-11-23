import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import mlpy

def readData(filename):

  f = open(filename,"r")
  header = f.readline().split(",")
  f.close()

  data = np.genfromtxt(filename, names=header, skip_header=1,
                       delimiter=',', skip_footer=0, autostrip=True,
                       usecols=range(0,len(header)-2), missing_values="???", filling_values=np.nan)

  return data

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("file")
  args = parser.parse_args()

  data = readData(args.file)

  random.shuffle(data)

  train_data = data[1:5000]
  test_data = data[-5000:-1]

  x = train_data["displ"].reshape(-1,1)
  xt = test_data["displ"].reshape(-1,1)
  y = train_data["comb08"]
  yt = test_data["comb08"]

  K = mlpy.kernel_gaussian(x, x, sigma=1) # training kernel matrix
  Kt = mlpy.kernel_gaussian(xt, x, sigma=1) # testing kernel matrix
  krr = mlpy.KernelRidge(lmb=1)
  krr.learn(K, y)
  ypredicted = krr.pred(Kt)

  fig = plt.figure(1)
  plot1 = plt.plot(x[:, 0], y, 'g.', label="training")
  plot2 = plt.plot(xt[:, 0], yt, 'b.', label="testing")
  plot2 = plt.plot(xt[:, 0], ypredicted, 'ro', label="predicted")
  plt.xlabel("Volume Displacement [Liters]")
  plt.ylabel("Gas Mileage [mpg]")
  plt.legend(frameon=False)
  plt.show()

if __name__ == "__main__":
  main()
