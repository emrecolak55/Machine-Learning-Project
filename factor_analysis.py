import numpy as np

import csv_operations as cop

def factor_analysis(dataset, num_of_factors):
    # correlation matrix, represents each column as a variable
    corr_mtx = np.ma.corrcoef(dataset, rowvar=False)
    
    reduced_factors = principal_axis_factoring(corr_mtx, num_of_factors)
    reduced_dataset = (dataset[:,:, np.newaxis]*reduced_factors).sum(axis=1)
    reduced_dataset /= np.linalg.norm(reduced_dataset, axis=0)
    return reduced_dataset
    
# m = number of factors to extract
def principal_axis_factoring(correlation_matrix, m, max_iterations=200, tolerance=1e-4):
    N = correlation_matrix.shape[0]
    
    factors = np.random.rand(N, m) # initialize the factors with random values
    
    # update factors with the correlation matrix until the difference converges to tolerance
    for it in range(max_iterations):
        # matrix multiplication
        factors_next = np.einsum('ij,jk->ik', correlation_matrix, factors) # equal to (a[:,:, np.newaxis]*b).sum(axis=1)
        
        # normalize factors
        factors_next /= np.linalg.norm(factors_next, axis=0)
        
        # check convergence
        difference = np.linalg.norm(factors_next - factors, ord='fro')
        if difference < tolerance:
            break
        
        factors = factors_next
        
    return factors_next

if __name__ == '__main__':
    
    training_data_filename = 'aps_failure_training_set.csv'
    
    dataset, clazz, headers = cop.read_csv(training_data_filename)
    
    #dataset = [[55, None, 68, 84, -84], [67, 33, None, 53, -53], [29, None, 51, 37, -37], [56, 45, 99, 85, -85]]
    
    # convert to masked array to mask unavailable data
    dataset = np.ma.array(dataset, dtype=float)
    
    # standardization
    means = np.nanmean(dataset, axis=0, dtype=float)
    stds = np.nanstd(dataset, axis=0, dtype=float, ddof=1)
    dataset = (dataset - means) / stds
    
    a = factor_analysis(dataset, 20)
    cop.write_csv('fct-out.csv', a)