import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

def add_data_frames(data_frames):
    """
    Concats data frames that are provided in a list. 
    
    Args:
        data_frames (DataFrame): A list of Pandas dataframe.
    Returns:
        df: A Pandas DataFrame.
    """
    
    df = pd.concat(data_frames, ignore_index=True)

    return df

def remove_rows_with_nan(data_frame, features):
    """
    Removes rows with NaN values from a Pandas dataframe.
    
    Args:
        data_frame (DataFrame): A Pandas dataframe.
        features (list): A list of str representing the headers of df. 
    Returns:
        df: A Pandas DataFrame.
    """
    
    for f in features:
        if data_frame[f].dtype != object:

            data_frame = data_frame[np.isfinite(data_frame[f])]
    
    return data_frame

def create_sliding_window_data_frame(data, window_size, feature_list):
    """
    Creates a new data frame, where the specified features are grouped by the size of the window
    Args:
        data (DataFrame): A Pandas DataFrame.
        window_size (int): The size of the window.
        feature_list (list): A list of str representing the headers of data.
    Returns: 
        df: A Pandas DataFrame
    """

    counter_a = 0
    table_list = []

    while counter_a + window_size < len(data):
        entry_list = []
    
        for counter_b in range(counter_a, counter_a + window_size):
            for counter_c in range(0, len(feature_list)):
                entry = data.loc[counter_b, feature_list[counter_c]]    
                entry_list.append(entry)
    
        table_list.append(entry_list)
        counter_a = counter_a + 1
    
    df = pd.DataFrame(table_list, columns = range(0, window_size * len(feature_list)))
    
    return df

def create_sliding_window_data_frame_from_predictions(predictions, data_frame, window_size):
    """
    Creates a sliding window data frame based on the predicted outliers. 
    
    Args:
        predictions (list): A list where -1 corresponds to an outlier and 1 does not correspond to an outlier. 
        data_frame (DataFrame): A Pandas dataframe (corresponding to the testing data frame).
        window_size (int): The size of the window.
    Returns:
        data_frame_outliers: A Pandas DataFrame.
    """

    data_frame_outliers = pd.DataFrame(columns=list(data_frame.columns))
    
    i = 0
    
    for j in range(0, len(predictions)):
        if predictions[j] == -1:
            for k in range(0, window_size):
                data_frame_outliers.loc[i] = data_frame.iloc[j + k]
                i = i + 1
            data_frame_outliers.loc[i] = [np.nan] * len(data_frame_outliers.columns)
            i = i + 1
    
    return data_frame_outliers


def calculate_the_number_of_outliers(predictions):
    """
    Calculates the number and percentage of outliers, and prints the results.
    
    Args:
        predictions (list): A list where -1 corresponds to an outlier and 1 does not correspond to an outlier. 
    Returns:
        None
    """
   
    labels = predictions.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    n_c0 = list(labels).count(0)
    n_c1 = list(labels).count(1)
    n_c2 = list(labels).count(2)
    n_c3 = list(labels).count(3)
    n_c4 = list(labels).count(4)
    n_c5 = list(labels).count(5)
    n_c6 = list(labels).count(6)
    uniquelabel_ = set(labels)
    print("unique labels",uniquelabel_)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Estimated number of C0: %d' % n_c0)
    print('Estimated number of  C1: %d' % n_c1)
    print('Estimated number of C2: %d' % n_c2)
    print('Estimated number of  C3: %d' % n_c3)
    print('Estimated number of C4: %d' % n_c4)
    print('Estimated number of  C5: %d' % n_c5)
    print('Estimated number of C6: %d' % n_c6)
    print("Percentage of outliers: ", n_noise_ /  labels.size* 100)

    #outliers = 0

    #for p in predictions:
    #    if p == -1:
    #        outliers = outliers + 1
            
    #print("Number of outliers: ", outliers)
    return

def apply_label_encoder(data_frame):
    """
    Applies a label encoder to columns in a data frame that are of type object.
    
    Args:
        data_frame (DataFrame): A Pandas dataframe.
    Returns:
        data: A Pandas DataFrame.
    """

    encs = dict()
    data = data_frame.copy()
    for c in data.columns:
        if data[c].dtype == "object":
            encs[c] = LabelEncoder()
            data[c] = encs[c].fit_transform(data[c])

    return data


def apply_onehot_encoder(data_frame,features):
    """
    Applies a one hot encoder to columns in a data frame that are of type object.
    
    Args:
        data_frame (DataFrame): A Pandas dataframe.
    Returns:
        data: A Pandas DataFrame.
    """

    encs = dict()
    data = features.copy()
    for c in data:
        if data[c].dtype == "object":
            transformed = feature_encoder.transform(data_frame[feature_to_encode])
            ohe_df = pd.DataFrame(transformed)
            df = pd.concat([data_frame, ohe_df], axis=1)

    return data

def apply_hot_encoder(data_frame,feature_to_encode):
    dummies = pd.get_dummies(data_frame[[feature_to_encode]])
    res = pd.concat([data_frame, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)


def one_hot_encode_column(data_frame, feature_to_encode):
    """
    Applies a binary encoder to columns in a data frame that are of type object.
    
    Args:
        data_frame (DataFrame): A Pandas dataframe.
    Returns:
        data: A Pandas DataFrame.
    """
    feature_encoder = LabelBinarizer()
    feature_encoder.fit(data_frame[feature_to_encode])
    transformed = feature_encoder.transform(data_frame[feature_to_encode])
    ohe_df = pd.DataFrame(transformed)
    df = pd.concat([data_frame, ohe_df], axis=1)
    return df

def aggregate_data(replaced_nan):
    """
    Aggregagates the data on local ip
    columns in a data frame that are of type object.
    
    Args: 
        data_frame (DataFrame): A Pandas dataframe.
    Returns:
        data: A Pandas DataFrame.
    """
    grouped_single = replaced_nan.groupby('protocol_name').agg({'total_bytes': ['mean']})
    grouped_single.columns = ['total_bytes_mean']
    grouped_single = grouped_single.reset_index()
    data_frame = pd.concat([replaced_nan, grouped_single], axis=1)
    return data_frame



def apply_encoder(data_frame_a,features):
     """
    Applies a one hot encoder,label encoder and aggragtes the dataframe.
    
    Args:
        data_frame_a (DataFrame): A Pandas dataframe (testing_data).
       
    Returns:
        updated_data_frame_a: A Pandas DataFrame (testing_data).
        
    """
     data_frames =  (data_frame_a)
     replaced_nan = replace_nan(data_frame_a)  
     agg_data = aggregate_data(replaced_nan)
     label_encoded = apply_label_encoder(replaced_nan)  
     apply_hot_encoder_vb = one_hot_encode_column(label_encoded,'application_name')         
     shape = data_frame_a.shape
     rows = shape[0]
     data_frames1 = label_encoded.iloc[:rows, :]
     data_frames2 = data_frames1.reset_index(drop=True)


     return data_frames2




def replace_nan(data_frame):
    """
    Replaces NaN values in a data frame
    
    Args:
        data_frame (DataFrame): A Pandas dataframe.
    Returns:
        df: A Pandas DataFrame.
    """

    df = data_frame.copy()

   
    df['other_ip_country'].replace(np.nan, 'unknown_ip_country', inplace = True)
    df['application_tag'].replace(np.nan, 'unknown_application_tag', inplace = True)
    df['application_name'].replace(np.nan, 'unknown_application_name', inplace = True)
    df['group_name'].replace(np.nan, 'unknown_group_name', inplace = True)
    df['category'].replace(np.nan, 'unknown_category', inplace = True) 
    df['agent'].replace(np.nan, 'unknown_agent', inplace = True)
    df['ssh_client'].replace(np.nan, 'unknown_ssh_client', inplace = True)
    df['ssh_server'].replace(np.nan, 'unknown_ssh_server', inplace = True)
    df['tls_version'].replace(np.nan, 'tls_version', inplace = True)
    df['cipher_suite'].replace(np.nan, 'unknown_cipher_suite', inplace = True)
    df['client_sni'].replace(np.nan, 'unknown_client_sni', inplace = True)
    df['client_ja3'].replace(np.nan, 'unknown_client_ja3', inplace = True)
    df['server_cn'].replace(np.nan, 'unknown_server_cn', inplace = True)
    df['server_ja3'].replace(np.nan, 'unknown_server_ja3', inplace = True)
    df['fingerprint'].replace(np.nan, 'unknown_fingerprint', inplace = True)
    df['hostname'].replace(np.nan, 'unknown_hostname', inplace = True)
    df['mdns_answer'].replace(np.nan, 'unknown_mdns_answer', inplace = True)
    df['class_ident'].replace(np.nan, 'unknown_class_ident', inplace = True)
    df['total_local_bytes'].replace(np.nan, -1, inplace = True)
    df['total_other_bytes'].replace(np.nan, -1, inplace = True)
    df['total_bytes'].replace(np.nan, -1, inplace = True)
    df['total_packets'].replace(np.nan, -1, inplace = True)


    return df

def create_data_frame_from_predictions(predictions, data_frame):
    """
    Creates a data frame based on the predicted outliers. 
    
    Args:
        predictions (list): A list where -1 corresponds to an outlier and 1 does not correspond to an outlier. 
        data_frame (DataFrame): A Pandas dataframe (corresponding to the testing data frame).
    Returns:
        data_frame_outliers: A Pandas DataFrame.
    """

    data_frame_outliers = pd.DataFrame(columns=list(data_frame.columns))
    labels = predictions.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)   
    n_noise_ = list(labels).count(-1)
    j = 0

    for i in labels:
        if labels[i] == -1:
            data_frame_outliers.loc[j] = data_frame.iloc[i]
            j = j + 1

    return data_frame_outliers
