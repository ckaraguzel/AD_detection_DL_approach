import cv2
import os
import pandas as pd
import shutil

import sys
sys.path.append(os.getcwd())

import numpy as np
from icecream import ic
#import matplotlib.pyplot as plt
#from libs.preprocessor import Preprocessor
#from libs.utils import *

def remove_duplicates(df):
    # Convert "Acq Date" column to datetime format
        df['Acq Date'] = pd.to_datetime(df['Acq Date'], format='%m/%d/%Y')
    
    # Sort by "Subject" and then by "Acq Date" in descending order
        df_sorted = df.sort_values(by=['Subject', 'Acq Date'], ascending=[True, False])
    
    # Drop duplicates based on "Subject" column, keep the first occurrence (latest date)
        new_df = df_sorted.drop_duplicates(subset='Subject', keep='first')
        return new_df

def get_filtrated_data_paths(data_folder, list_im_id):
        data_paths = []

        # Walk through the directory tree
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.dcm'):  # Filter out files with .dcm extension
                    full_path = os.path.join(root, file)
                    data_paths.append(full_path)

        filtered_data_paths= [path for path in data_paths if any(path.endswith(suffix+".dcm") for suffix in list_im_id)]
        return filtered_data_paths

def save_images_to_new_folder(data_paths, new_folder_path):
    # Ensure the new folder exists, create it if not
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    
    # Iterate over each data path
    for data_path in data_paths:
        # Extract filename from the path
        filename = os.path.basename(data_path)
        
        # Copy the file to the new folder
        shutil.copy(data_path, os.path.join(new_folder_path, filename))
        
