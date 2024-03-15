import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from libs.preprocessor import Preprocessor
from libs.utils import *

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = 'data/raw' # Input path
OUTPUT_PATH = 'data/ml_ready'

# create output folder if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists('reports'):
    os.makedirs('reports')

# Preprocess the training and testing data
preprocessor = Preprocessor()
preprocessor.process_directory(INPUT_PATH, INPUT_PATH.replace('raw', 'interim/resized'))

#Create DataFrames with metadata of preprocessed data (all combined)
df_preprocessed = get_metadata("data/interim/resized")

# print the df head
ic(df_preprocessed.head())

# Save the DataFrame to a CSV file
df_preprocessed.to_csv('reports/image_sizes_labels_and_data_preprocessed.csv', index=False)

# Get the image array for each image and add it to the data frame with metadata
df_preprocessed['Image_array'] = df_preprocessed['Path'].apply(lambda x: preprocessor.get_image_array(x))

