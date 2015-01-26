#! /usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import random
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.learning_curve import learning_curve


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

  # Isolate default and convert to default = 1 and Repaid = 0
  default = df["default"]
  y = np.where(default == "Default", 1, 0)  

  # Remove columns not used for anything
  useless_labels = ["key", "fname", "mname", "lname", "generation", "dob", "dod", 
                    "addr1", "city", "country", "postcode", "phone", "email", 
                    "website", "ssn_last4", "drlic", "passport", "gov_id", 
                    "address", "pending", "ssn", "default"]

  df.drop(useless_labels, axis=1, inplace=True)

  # A list of features used (just for reference)
  features = ["creddebt", "debtinc", "othdebt", "age", "employ", "gender" ]

  # Convert gender to Male = 1 and Female = 0  
  df["gender"] = np.where(df["gender"] == "M", 1, 0)
  
  # Float matrix of features
  X = df.as_matrix().astype(np.float)

  # Rescale the features to a mean of 0 and range -1 to 1
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  Xr = np.tile(X,(100,1))

  yr = np.tile(y, 100)
  
  return Xr, yr


def confusion_matrices(y, yp):

  cm = confusion_matrix(y, yp)

  plt.matshow(cm)
  plt.title('Confusion matrix')
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


class LogisticsModel:

  def __init__(self, X, y):


    train_msk = np.random.rand(len(y)) < 0.8
    training_data = X[train_msk]
    testing_data = X[~train_msk]

    self.training_y = y[train_msk]
    self.training_x = X[train_msk,]
    self.testing_y = y[~train_msk]
    self.testing_x = X[~train_msk,]

    
  def train(self):

    classifier = LogisticRegression(penalty='l1', dual=False, tol=0.000001, 
                                      C=10.0, fit_intercept=True, class_weight=None,
                                      random_state=None)

    X_train_new = classifier.fit_transform(self.training_x, self.training_y)
    X_test_new = classifier.transform(self.testing_x)

    self.testing_x = X_test_new
    self.training_x = X_train_new

    classifier.fit(X_train_new, self.training_y)

    if self.training_x.shape != X_train_new.shape:
      print("Reduced the number of features from %s to %s.") \
            % (self.training_x.shape[1], X_train_new.shape[1])

    self.trainer = classifier    


  def score(self):

    accuracy_training = self.trainer.score(self.training_x, self.training_y)
    accuracy_testing = self.trainer.score(self.testing_x, self.testing_y)

    return (accuracy_training, accuracy_testing)


  def predict(self, x):

    predicted_y = self.trainer.predict(x)

    return predicted_y


  def make_learning_curves(self):

    sizes, t_scores, v_scores = learning_curve(self.trainer, self.training_x, 
                                               self.training_y)
    t_scores_mean = np.median(t_scores, axis=1)
    v_scores_mean = np.median(v_scores, axis=1)

    plt.plot(sizes, t_scores_mean, "go", ls="--", label="Training")
    plt.plot(sizes, v_scores_mean, "ro", ls="--", label="Validation")
    plt.legend()
    plt.show()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("data")
  parser.add_argument("--s", dest="show", action="store_true", default=False)
  args = parser.parse_args()

  # Read in the data to pandas data frame
  X, y = get_data(args.data)

  # Put data into the Logistic Model object which creates training and testing sets
  model_env = LogisticsModel(X, y)

  # Train the model on the training set
  model_env.train()

  # Score the model on the training and testing set
  accuracy_training, accuracy_testing = model_env.score()
  print("Training accuracy: %s; testing accuracy %s.") % (accuracy_training, accuracy_testing)

  # Get preidiction for training set  
  yp = model_env.predict(model_env.training_x) 

  # Produce learning curves for testing ste and training set
  model_env.make_learning_curves()

  # Produce graphical representation of the training confusion matrix
  confusion_matrices(model_env.training_y, yp)

  if args.show: plt.show()

if __name__ == "__main__":
  main()
