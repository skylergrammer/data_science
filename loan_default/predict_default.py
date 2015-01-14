#! /usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mlpy
from sklearn.cross_validation import train_test_split
import datetime
import random

def get_data(filename):

  # Read data into a pandas dataframe
  df = pd.read_csv(filename)

  # Drop duplicate entries based on repeating Social Security Number
  df.drop_duplicates("ssn", inplace=True)
  
  # Remove rows with NaNs in the target column
  df = df[pd.notnull(df['default'])]

  # Derive age from DOB
  age = np.array([datetime.date.today().year - datetime.datetime.strptime(x, "%Y-%m-%d").year
                  for x in df["dob"]])

  df["age"] = age

  # Remove columns not used for anything
  useless_labels = ["key", "fname", "mname", "lname", "generation", "dob", "dod", 
                    "addr1", "city", "country", "postcode", "phone", "email", 
                    "website", "ssn_last4", "drlic", "passport", "gov_id", 
                    "address", "pending", "ssn"]

  df.drop(useless_labels, axis=1, inplace=True)

  return df
  
def normalize_variables(df):
  '''
  Encode continuous variables to a new variable with mean = 0 and std = 1. 
  Categorical variables are encoded to integer variables with categories taking
  values 0 to N-1, where N is the number of unique string categories.
  '''

  df2 = df.copy(deep=True)

  for each in df2:

    variable = df2[each]

    if type(variable[0]) is str:
      set_values, encoded = np.unique(variable, return_inverse=True)

    elif type(variable[0]) is np.float64:
      standard_dev = np.std(variable)
      mean = np.mean(variable)
      encoded = (variable - mean) / standard_dev

    else:
      encoded = variable

    df2[each] = encoded

  return df2


def correlation_matrix(data, target=None):
  '''
  Compute the correlation matrix for all predictors (does not include target).

  Requires a pandas data frame, dictionary, or numpy rec array.
  '''
  # Matrix of predictor data
  predictors = [data[x] for x in data if x != target]

  # Compute correlation coefficients matrix
  cor_matrix = np.corrcoef(predictors)

  for row in cor_matrix: 
    row = " ".join(["{:3.2}".format(x) for x in row ])
    print(row)

  return cor_matrix


def see_distributions(df):

  for each in df:
    fig = plt.figure()
    df[each].plot(kind="hist", stacked=True, bins=20, label=each)
    plt.legend()
  plt.show()

class LogisticsModel:

  def __init__(self, data, target):


    train_msk = np.random.rand(len(data)) < 0.75
    training_data = data[train_msk]
    testing_data = data[~train_msk]

    self.training_y = training_data[target]
    self.training_x = training_data.drop(target, axis=1)
    self.test_y = testing_data[target]
    self.test_x = testing_data.drop(target, axis=1)

    self.solver = mlpy.LibLinear(solver_type='l1r_lr', C=0.01)
    
  def train(self):

    self.solver.learn(self.training_x, self.training_y)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("data")
  args = parser.parse_args()

  target = "default"

  df = get_data(args.data)
 
  df_normal = normalize_variables(df)

  cor_matrix = correlation_matrix(df_normal, target=target)

  model_env = LogisticsModel(df, target=target)
 
  model_env.train()


if __name__ == "__main__":
  main()
