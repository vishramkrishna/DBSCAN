from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from pylab import *

def run_DBCSAN(testing_data, features):
     """
    tests DBSCAN
    
    Args:
         testing_data (DataFrame): A Pandas dataframe.
        features (list): A list of str representing the headers of the training/testing data frames. 
    Returns:
        predictions (list): A list where -1 represents an anomaly and 1 represents not an anomaly.
    """


     shape = testing_data.shape
     rows = shape[0]
     X = StandardScaler().fit_transform(testing_data[features])
     predictions = DBSCAN(eps=0.5,min_samples=25,metric='euclidean', metric_params=None, algorithm='auto', p=None, n_jobs=None).fit(X)  
     core_samples_mask = np.zeros_like(predictions.labels_, dtype=bool)
     core_samples_mask[predictions.core_sample_indices_] = True
     labels = predictions.labels_      

     return predictions
     
     
     #core_samples_mask = np.zeros_like(predictions.labels_, dtype=bool)
     #core_samples_mask[predictions.core_sample_indices_] = True
     #labels = predictions.labels_
     #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
     #n_noise_ = list(labels).count(-1)
     #uniquelabel_ = set(labels)
     #print('Estimated number of clusters: %d' % n_clusters_)
     #print('Estimated number of noise points: %d' % n_noise_)
     #print("Percentage of outliers: ", n_noise_ /rows * 100)
     #print("unique labels",uniquelabel_)
    
     #unique_labels = set(labels)
     #colors = [plt.cm.Spectral(each)
     #        for each in np.linspace(0, 1, len(unique_labels))]


     #for k, col in zip(unique_labels, colors):
     #  if k == -1:
     ##     # Black used for noise.
     #   col = [0, 0, 0, 1]
     #class_member_mask = (labels == k)
     #xy = X[class_member_mask]

     #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
     #       markeredgecolor='k', markersize=6)

     #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
     #       markeredgecolor='k', markersize=6)
     #plt.show()
     #--new code starts-#
     ##X = testing_data[features].iloc[:rows, :]
     #xy = X[class_member_mask & core_samples_mask]
     
     #plt.plot(xy[:, 0], xy[:, 1], 'o',markerfacecolor=tuple(col),
     #        markeredgecolor='k', markersize=10)

     #xy = X[class_member_mask & ~core_samples_mask]
     #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k',
     #        markersize=6)

     #plt.title('Estimated number of clusters: %d' % n_clusters_)
     #pca = PCA(n_components=2).fit(testing_data[features])
     #pca_2d = pca.transform(testing_data[features])
     #for i in uniquelabel_:
     # if uniquelabel_[i] == 0:
     #  c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
     # elif uniquelabel_[i] == 1:
     #  c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g', marker='o')
     # elif uniquelabel_[i] == -1:
     #  c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',  marker='*')

     # #plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])

     # plt.title('DBSCAN finds 2 clusters and noise')
     # plt.show()
     
     



   

