import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def make_data(npoints=1000):

  x = np.linspace(0,1, num=npoints)

  y0 = lambda x, m, b: m*x + b
  
  m = np.random.random()
  b = np.random.random()

  rndm_noise = np.random.normal(loc=0.0, scale=0.05, size=len(x))

  yobs = y0(x, m, b) + rndm_noise

  return (x, yobs, float(m), float(b))


def calc_error(m, b, x, y):

  totalError = 0.
  nPoints = float(len(x))
  for i,each in enumerate(x):
    totalError += np.power(y[i] - (m*x[i] + b), 2)

  return totalError / nPoints


def grad_descent(m, b, x, y, learningRate=0.005):

  b_grad = 0.
  m_grad = 0.
  nPoints = float(len(x))
 
  for i, each in enumerate(x):
    dy = y[i] - (m*x[i] + b)
    m_grad += (-2.0/nPoints) * x[i] * dy
    b_grad += (-2.0/nPoints) * dy
    
  m_new = m - (learningRate*m_grad)
  b_new = b - (learningRate*b_grad)

  return (m_new, b_new)


def my_fit(x, y):


  # Initialize random m and b
  m0 = 0.
  b0 = 0.
  
  # Lists to hold the errors and iteration number
  errors = []
  itNum = []
  
  # Perform gradient descent for 5000 iterations
  for iteration in range(5000):
  
    m_new, b_new = grad_descent(m0, b0, x, y)
    error = calc_error(m_new, b_new, x, y)

    m0 = m_new
    b0 = b_new

    errors.append(error)
    itNum.append(iteration)

  return (m0, b0)

def sklearn_fit(x, y):

  fitter = LinearRegression(fit_intercept=True)

  x = x.reshape(-1,1)
  y = y.reshape(-1,1)

  fitter.fit(x, y)

  yfit = fitter.predict(x)

  return yfit


def main():

  # Generate data
  x, y, m_act, b_act = make_data(npoints=500)

  # Fit with sklearn
  yfit = sklearn_fit(x, y)

  # Fit with my implementation
  m_fit, b_fit = my_fit(x, y)

  fig2 = plt.figure()
  plt.plot(x, y, "ro")
  plt.plot(x, m_fit*x+b_fit, ls="-", color="k", lw=4, label="Mine")
  plt.plot(x, yfit, ls="--", color="g", lw=4, label="Sklearn")
  plt.legend(loc="best")
  plt.show()

main()
