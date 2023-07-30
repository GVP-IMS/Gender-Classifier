### CV MODULE OVERVIEW ###
# Run this independent module first to prepare the data for the main program
# Creates the data partitions with a custom number of Tweets

# Import Space
import csv
import random

# Determine language
print('Choose the language (de, fr, it, nl):')
lan = str(input())

# Determine the size of each fold
print('Enter the number of instances in each fold (recommended: 10000):')
instance = int(input())

# Preprocess the IT csv file to avoid the error: "_csv.Error: line contains NUL"
if lan == "it":
    file_path = '../Data/Original/IT/tweets-it.csv'
    with open(file_path, 'rb') as fi:
        data = fi.read().replace(b'\x00', b'')
    with open(file_path, 'wb') as fo:
        fo.write(data)

# Read data from (German, for now) CSV file
with open(f'../Data/Original/{lan.upper()}/tweets-{lan}.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = list(reader)  # Each row from the CSV is a list entry: [[row1], [row2]...]


# Determine number and size of folds
num_folds = 10
fold_size = len(data) // num_folds

# Perform k-fold cross validation
for i in range(num_folds):
    
    # determine start and end indices for testing data
    test_start = i * fold_size
    test_end = (i + 1) * fold_size

    # extract testing data
    test_data = data[test_start:test_end]

    # extract training data
    train_data = data[:test_start] + data[test_end:]
    train_data = train_data[:instance]


    # save testing data to CSV file
    with open(f'../Data/Partitions/test_data_fold_{i+1}_{lan}.csv', 'w', newline='', encoding='utf-8') as test_file:
        writer = csv.writer(test_file)
        writer.writerows(test_data)

    # save training data to CSV file
    with open(f'../Data/Partitions/train_data_fold_{i+1}_{lan}.csv', 'w', newline='', encoding='utf-8') as train_file:
        writer = csv.writer(train_file)
        writer.writerows(train_data)
