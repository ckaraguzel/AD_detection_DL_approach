# Import necessary libraries
import pickle
import random
import numpy as np
from sklearn.utils import shuffle

# Define the base path for loading the data and path for saving the Train_Test data
Base_Path = "Deep Learning"
Data_Path = ''

# Load the pickle files containing the data
with open(Base_Path + "/pixels_CN_1.pkl", 'rb') as f:
    cn_1 = pickle.load(f)
with open(Base_Path + "/pixels_CN_2.pkl", 'rb') as f:
    cn_2 = pickle.load(f)
with open(Base_Path + "/pixels_MCI.pkl", 'rb') as f:
    mci = pickle.load(f)
    
# Extend the cn_1 list with cn_2
cn_1.extend(cn_2)

# Remove the 70th element from mci due to shape mismatch
mci.pop(70)
# Delete cn_2 as it's no longer needed
del cn_2

# Set the random seed for reproducibility
random.seed(42)

# Calculate the number of samples for training (80% of the data)
l_cn = len(cn_1)
l_mci = len(mci)
num_samples_1 = int(0.8 * l_cn)
num_samples_2 = int(0.8* l_mci)

# Randomly sample indices for the training data
train_indices_1 = random.sample(list(range(l_cn)), num_samples_1)
train_indices_2 = random.sample(list(range(l_mci)), num_samples_2)

# Get the remaining indices for the test data
test_indices_1 = list(set(list(range(l_cn))) - set(train_indices_1))
test_indices_2 = list(set(list(range(l_mci))) - set(train_indices_2))

# Define a function to scale the data
def scale(arr):
    return (arr - np.min(arr))/(np.max(arr) - np.min(arr))

# Sample 80% of the entries for the training data and scale them
X_train_1 = [np.array([scale(cn_1[i][j]) for j in range(65)]) for i in train_indices_1]
X_train_2 = [np.array([scale(mci[i][j]) for j in range(65)]) for i in train_indices_2]

# Sample the remaining 20% for the test data and scale them
X_test_1 = [np.array([scale(cn_1[i][j]) for j in range(65)]) for i in test_indices_1]
X_test_2 = [np.array([scale(mci[i][j]) for j in range(65)]) for i in test_indices_2]

# Create labels for the training data
y_train_1 = np.zeros(len(X_train_1))  # Class 0 for X_train_1
y_train_2 = np.ones(len(X_train_2))   # Class 1 for X_train_2

# Stack the training data and labels
X_train = np.vstack((X_train_1, X_train_2))
y_train = np.hstack((y_train_1, y_train_2))

# Shuffle the training data and labels
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Save the training data and labels
np.save(Data_Path + 'y_train.npy', y_train)
np.save(Data_Path + 'X_train.npy', X_train)
del X_train, y_train, mci, cn1

# Create labels for the test data
y_test_1 = np.zeros(len(X_test_1))  # Class 0 for X_train_1
y_test_2 = np.ones(len(X_test_2))   # Class 1 for X_train_2

# Stack the test data and labels
X_test = np.vstack((X_test_1, X_test_2))
y_test = np.hstack((y_test_1, y_test_2))

# Shuffle the test data and labels
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Save the test data and labels
np.save(Data_Path + 'X_test.npy', X_test)
np.save(Data_Path + 'y_test.npy', y_test)
del y_test, X_test
