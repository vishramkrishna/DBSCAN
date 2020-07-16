import os
import re
import pandas as pd
from datetime import datetime

def import_csv_data(path):
    """
    Imports .csv data files
    
    Args:
        path (str): The path to the directory containing a set of training (.csv) files. 
    Returns:
        data_frames (list): A list of Pandas DataFrame.
    """

    files = os.listdir(path)

    data_frames = []

    for f in files:
        if f[-4:] == '.csv':
            df = pd.read_csv(path + f, delimiter = ',')
            data_frames.append(df)

    return data_frames

def read_features_list(path_to_file):
    """
    Imports .csv data files
    
    Args:
        path_to_file (str): The path to the file containing the list of features. 
    Returns:
        features (list): A list of str representing features in the data set.
    """

    f = open(path_to_file, 'r')
    features = f.readlines()

    for i in range(0, len(features)):
        features[i] = re.sub('\\n', '', features[i])

    return features

def export_data_frame_to_csv(data_frame):
    """
    Imports .csv data files
    
    Args:
        data_frame (DataFrame: The data_frame to export to a .csv file. 
    Returns:
        None
    """

    now = datetime.now()
    s = now.strftime('%y_%m_%d_%H_%M_%S')
    data_frame.to_csv('Data/Result/results_' + s + '.csv', index = None, header=True)
        
    return

