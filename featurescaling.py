import numpy as np

def square(x):
  return x*x

def sqrt(x):
  return x ** 1/2

# Finding min and max values of each column in the dataset


# Z SCORE Part

def columnStandardDeviation(dataset, colMeans):
  std_dev = list()
  for i in range(len(dataset[0])):
    stddevs = 0
    for j in range (len(dataset)):
        stddevs += square(  float ( dataset[j][i] ) - colMeans[i] )
    #print("finished")
    std_dev.append( sqrt( stddevs / (len(dataset) - 1)) )
  return std_dev



#standardization 
def zscore(dataset, means, stdevs ):
  for row in dataset:
    for i in range(len(row)):
      if row[i] == 'na':
        #row[i] = np.nan
        continue
      else:
        row[i] = (float(row[i]) - means[i] / stdevs[i])
  