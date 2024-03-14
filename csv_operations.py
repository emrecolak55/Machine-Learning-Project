import csv

# reads a csv file with the given file name to a list as an NxM matrix. N = number of examples, M = number of features + features_starting_index
def read_csv(filename):
    dataset = []
    clazz = []
    with open(filename, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        
        headers = next(reader, None) # to skip the headers
        
        # fix reading class column in test data
        if headers[1] == 'class':
            features_starting_index = 2
        else: 
            features_starting_index = 1
        
        
        for current in reader:
            clazz.append(current[1] if features_starting_index == 2 else None)
            row = [float(item) if item != 'na' else None for item in current[features_starting_index:]]
            
            dataset.append(row)
    
    return dataset, clazz, headers
        
# writes a dataset (NxM matrix) to a csv file with the given file name, optionally headers can be added
def write_csv(filename, dataset, headers=None):
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if headers:
            writer.writerow(headers)
        writer.writerows(dataset)

# writes a csv file according to the submission guidelines
# dataset is 1-D that contains the class
def create_submission_file(filename, dataset):
    headers = ['id', 'class']
    write_csv('submissions\\' + filename, list(enumerate(dataset, 1)), headers)
    