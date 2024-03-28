import cv2
import os
import pandas as pd
import shutil
import numpy as np
import pydicom

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
        

def crop_black_frame(dicom_path):
    # Read DICOM file
    dicom_data = pydicom.dcmread(dicom_path)
    
    # Extract pixel data
    image_array = dicom_data.pixel_array
    
    # Convert to uint8 format for OpenCV compatibility
    image_uint8 = (image_array / np.max(image_array) * 255).astype(np.uint8)
    
    # Apply adaptive thresholding
    _, binary_image = cv2.threshold(image_uint8, 20, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour is found
    if len(contours) == 0:
        raise ValueError("No black frame found.")
    
    # Get the bounding box of the largest contour (assumed to be the black frame)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Crop the image using the bounding box
    cropped_image = image_array[y:y+h, x:x+w]

    return cropped_image


try:
    cropped_image = crop_black_frame(dicom_file)
    
    # Save cropped image as a new DICOM file
    dicom_data = pydicom.dcmread(dicom_file)
    dicom_data.PixelData = cropped_image.tobytes()
    dicom_data.Rows, dicom_data.Columns = cropped_image.shape
    dicom_data.save_as(output_file)
    
    print("Cropped image saved as:", output_file)
except ValueError as e:
    print("Error:", e)

# Example usage
#dicom_file = "/Users/cisilkaraguzel/Documents/GitHub/AD_detection_DL_approach/data-ADNI/raw/All_AD_3loc_T1_axial/ADNI/002_S_0619/3-plane_localizer/2008-08-13_15_18_48.0/I116116/ADNI_002_S_0619_MR_3-plane_localizer__br_raw_20080813225809528_1_S55371_I116116.dcm"  # Replace with your DICOM file path
#output_file = "cropped_image.dcm"  # Path to save the cropped DICOM image

