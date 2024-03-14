import numpy as np
from csv import reader
import csv

class PCA:
    
    def __init__(self, num_components):
        self.num_components = num_components
        self.components     = None
        self.mean           = None
        self.variance_share = None
    
    
    def fit(self, X):
        # data centering
        self.mean = np.mean(X, axis = 0)
        X        -= self.mean
        
        # calculate eigenvalues & vectors
        cov_matrix      = np.cov(X.T)
        values, vectors = np.linalg.eig(cov_matrix)
        
        # sort eigenvalues & vectors 
        sort_idx = np.argsort(values)[::-1]
        values   = values[sort_idx]
        vectors  = vectors[:, sort_idx]
        
        # store principal components & variance
        self.components = vectors[:self.num_components]
        self.variance_share = np.sum(values[:self.num_components]) / np.sum(values)
    
    
    def transform(self, X):
        # data centering
        X -= self.mean
        
        # decomposition
        return np.dot(X, self.components.T)
    



def columnMeans(dataset):
  means = list()
  naCount = 0
  for i in range(len(dataset[0])):
    sum = 0
    for j in range (len(dataset)):
      if dataset[j][i] == 'na':
        naCount += 1
        continue
        #sum += 0   #not sure
      else:
        sum += float ( dataset[j][i] )
    means.append(sum / ( len(dataset) - naCount ) )
  return means

def square(x):
  return x*x

def sqrt(x):
  return x ** 1/2



def str_column_to_float(value, colMeans, j):

   # Convert a string to float
    if value == 'na':
        #return 0
        return colMeans[j]
    else:
        return float(value.strip())

def preprocess_data(dataset, colMeans):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            dataset[i][j] = str_column_to_float(dataset[i][j], colMeans, j)



def columnStandardDeviation(dataset, colMeans):
  std_dev = list()
  for i in range(len(dataset[0])):
    stddevs = 0
    for j in range (len(dataset)):
      if dataset[j][i] == 'na':
        stddevs += 0   #not sure
      else:
        stddevs += square(  float ( dataset[j][i] ) - colMeans[i] )
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
  

def readTraining():
    filename = 'data/aps_failure_training_set.csv'
    data = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # to skip the headers
        read_data = list(reader)

        for row in read_data:
            rows_with_skipped_columns = row[2:]
            data.append(rows_with_skipped_columns)

    return data


def readTest():
    filename = 'data/aps_failure_test_set.csv'
    data = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # to skip the headers
        read_data = list(reader)

        for row in read_data:
            rows_with_skipped_columns = row[2:]
            data.append(rows_with_skipped_columns)

    return data


def applyPCA(num_components):
    #"askdıdwıoqwjdıoasdıjoqwjdoqwı"
    data = readTraining()
    originalData = data
    colMeans = columnMeans(data)
    preprocess_data(data, colMeans)
    data_array = np.array(data, dtype=float)
    #print(data_array.shape)
    stddevs = columnStandardDeviation(data_array, colMeans)
    zscore(data_array, colMeans, stddevs)
    #print(data_array.shape)

    X = np.array(data_array, dtype=float)  # Convert data to numpy array
    pca = PCA(num_components)
    pca.fit(X)

    return pca.transform(X), originalData


def applyPCAtest(num_components):
    data = readTest()
    originalData = data
    colMeans = columnMeans(data)
    preprocess_data(data, colMeans)
    data_array = np.array(data, dtype=float)

    stddevs = columnStandardDeviation(data_array, colMeans)
    zscore(data_array, colMeans, stddevs)
    #print(data_array.shape)

    Xtest = np.array(data_array, dtype=float)  # Convert data to numpy array
    pcatest = PCA(num_components)
    pcatest.fit(Xtest)

    return pcatest.transform(Xtest), originalData




def findLabels():

    labels = []
    filename = 'data/aps_failure_training_set.csv'


    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # to skip the headers
        labels = [row[1] for row in reader]

    labels = np.array(labels)

    # Convert labels to binary 0 1 neg pos 
    label_mapping = {'neg': 0, 'pos': 1}
    return np.array([label_mapping[label] for label in labels])





