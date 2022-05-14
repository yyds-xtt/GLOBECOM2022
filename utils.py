import numpy as np 

def mega(x):
  return x*1e6

def kilo(x): 
  return x*1e3

def giga(x): 
  return x*1e9

def mini(x): 
  return x*1e-3

def dBm(x): 
  return mini(10**(x/10))

def dB(x): 
  return 10**(x/10)

def todB(x): 
  return 10.*np.log10(x)

def find_mean_mode(arr1, num_2): 
  '''
  add num2 to the last arr1 of num1 and find mean of the array
  '''
  num_2 = np.array([num_2])
  d = np.vstack((arr1, num_2))
  mean = np.mean(d, axis=0)
  return mean

def find_mean_queue(arr1, num_1, num_2): 
  '''
  update queue to the last element of arr1 by num1, num2 
  '''
  num = np.maximum(0, arr1[-1, :] - num_1) + num_2

  return find_mean_mode(arr1, num)
