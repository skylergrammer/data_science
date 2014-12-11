import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_data(filename):

  f = open(filename, "r")
  header = f.readline().replace('\"',"").replace(" ","_").split(";")
  f.close()

  print header

  return

def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  args = parser.parse_args()

  data = read_data(args.input)

if __name__ == "__main__":
  main()

