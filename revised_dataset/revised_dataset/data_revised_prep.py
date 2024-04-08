import cv2
import os
import pandas as pd
import shutil
import numpy as np
import pydicom
import matplotlib.pyplot as plt

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
        df_sorted = df.sort_values(by=['Subject', 'Acq Date'], ascending=[True, True])
    
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

def get_new_filtered_data_paths(data_folder, list_im_id):
    data_paths = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.dcm'):  # Filter out files with .dcm extension
                full_path = os.path.join(root, file)
                data_paths.append(full_path)

    filtered_data_paths = []
    for suffix in list_im_id:
        paths_with_suffix = [path for path in data_paths if path.endswith(suffix + ".dcm")]
        filtered_data_paths.append(paths_with_suffix)

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

# next function plots the images starting from i to i+num_images next to each other.
def plot_new_dicom_images(dicom_file_paths, num_images=5, i=0):
    num_images = min(num_images, len(dicom_file_paths))
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))

    for i, dicom_file_path in enumerate(dicom_file_paths[i:i+num_images]):
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_file_path)

        # Extract the pixel data
        pixel_data = ds.pixel_array

        # Plot the image
        axes[i].imshow(pixel_data, cmap=plt.cm.bone)
        axes[i].axis('off')  # Turn off axis
    plt.show()

def get_instance_number(dicom_file_path):
    # Read DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)

    # Check if 'InstanceNumber' tag exists
    if 'InstanceNumber' in dicom_data:
        return int(dicom_data.InstanceNumber)

    # If the instance number tag is not available, return None or raise an exception
    return None

#def sort_data_paths(data_folder_path, id_list):
     #takes data folder path and list of image id's and returns sorted paths according to MRI slices.
     #id_list is a list of image id's.
     #data_folder_path
#    sorted_data_paths= []
    
#    for id in id_list:
#        sorted_data_paths.append(sorted(get_filtrated_data_paths(data_folder_path, [id]),key = lambda path: get_instance_number(path)))

#    return sorted_data_paths

def get_pixel_array(data_path):
    path = pydicom.dcmread(data_path)
    # Extract pixel data
    image_array = path.pixel_array
    return image_array

def generate_pixels(sorted_paths):
    pixel = []
    for j in range(0, len(sorted_paths)):
        j_pixel = []  # Create an empty list for each j iteration
        for i in range(64, 129):
            ds = pydicom.dcmread(sorted_paths[j][i])
            pixel_arr = ds.pixel_array
            j_pixel.append(pixel_arr)  # Append each pixel array to j_pixel
        pixel.append(j_pixel)
    return pixel