from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, num_clusters):
    features, labels, true_centers = data

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=num_clusters, init='random', random_state=42)
    kmeans.fit(scaled_features)

    cluster_centers = kmeans.cluster_centers_

    distances = np.sqrt(np.sum((scaled_features - cluster_centers[kmeans.labels_]) ** 2, axis=1))

    sse = np.sum(distances ** 2)

    return sse




def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    

    data, label ,center = make_blobs(n_samples=20, centers=5, center_box=(-20,20), random_state=12,return_centers=True)

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    #dct = answers["2A: blob"] = [np.zeros(0)]
    
    dct = answers["2A: blob"] = [ data, label ,center]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    dataset = make_blobs(n_samples=20, centers=5, center_box=(-20, 20), random_state=12,return_centers=True)
    
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair   
    return_sse_values = []
    for k in range(1, 8 + 1):
        sse = fit_kmeans(dataset, k)
        return_sse_values.append((k, sse))

    return_sse_values_formatted = [[k, float(sse)] for k, sse in return_sse_values]

    dct = answers["2C: SSE plot"] = return_sse_values_formatted
        
    #dct = answers["2C: SSE plot"] = [[0.0, 100.0]]

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    
    
    data, label ,center = make_blobs(n_samples=20, centers=5, center_box=(-20,20), random_state=12,return_centers=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    sse_inertia_values = []
    for k in list(range(1, 9)):
        kmeans = KMeans(n_clusters=k, init='random', random_state=42)
        kmeans.fit(X_scaled)  
        sse_inertia_values.append((k,kmeans.inertia_))
    
    return_sse_values_formatted = [[k, float(sse)] for k, sse in sse_inertia_values]

    #answers["2D: SSE plot"] = return_sse_values_formatted

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = return_sse_values_formatted

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
