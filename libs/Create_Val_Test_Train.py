# Import necessary libraries
import pickle
import random
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the base path for loading the data and path for saving the Train_Test data
Base_Path = ""
Data_Path = ''

# Load the pickle files containing the data
with open(Base_Path + "CN_dataset.pkl", 'rb') as f:
    cn = pickle.load(f)  # Load the CN dataset
with open(Base_Path + "MCI_dataset.pkl", 'rb') as f:
    mci = pickle.load(f)  # Load the MCI dataset

# Define the index of the bad data point
bad_index = 70 + len(cn)

# Select the relevant columns from the MCI and CN datasets and assign labels
mci = mci[['Sex','Age','pixels']]
mci['type'] = 1  # Assign label 1 to MCI data
cn = cn[['Sex','Age','pixels']]
cn['type'] = 0  # Assign label 0 to CN data

# Concatenate the MCI and CN datasets
df = pd.concat([cn,mci])
del cn,mci  # Delete the original datasets to save memory
df = df.reset_index(drop = True)  # Reset the index of the concatenated dataset
df = df.drop(bad_index).reset_index(drop = True)  # Drop the bad data point

# Initialize a MinMaxScaler
scaler = MinMaxScaler()

# Scale the pixel values to be between 0 and 1 and store them in a new 'images' column
df['images'] = [np.array([np.round(scaler.fit_transform(df['pixels'][i][j]),3) for j in range(65)], dtype = 'float32') for i in range(len(df))]
df = df.drop(columns = 'pixels')  # Drop the original 'pixels' column

# Shuffle the dataset
df = shuffle(df, random_state = 42)
l_df = len(df)

# Split the dataset into training, validation, and test sets
df_train, df_validation, df_test = df.iloc[:int(0.7*l_df)], df.iloc[int(0.7*l_df):int(0.85*l_df)], df.iloc[int(0.85*l_df):]

# Save the training, validation, and test sets as .npy files
np.save('df_train.npy',df_train)
np.save('df_validation.npy',df_validation)
np.save('df_test.npy',df_test)
