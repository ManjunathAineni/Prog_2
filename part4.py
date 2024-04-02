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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage as scipy_linkage

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, n_clusters, linkage='ward'):
    
    data_array, labels = dataset

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_array)

    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster.fit(data_scaled)

    return cluster.labels_


def fit_modified(dataset, distance_threshold, linkage_method):
   
    data_array, labels = dataset

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_array)

    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage=linkage_method)
    cluster.fit(data_scaled)

    return cluster.labels_


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    
    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}
    
    seed = 42
    nc = datasets.make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=seed)
    answers["4A: datasets"]["nc"] = list(nc)
    nm = datasets.make_moons(n_samples=100, noise=0.05, random_state=seed)
    answers["4A: datasets"]["nm"] = list(nm)
    bvv = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    answers["4A: datasets"]["bvv"] = list(bvv)
    X, y = datasets.make_blobs(n_samples=100, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)
    answers["4A: datasets"]["add"] = list(add)
    b = datasets.make_blobs(n_samples=100, random_state=seed)
    answers["4A: datasets"]["b"] = list(b)
    
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
  
    """
    
    def fit_cluster_linkage_types(dataset, n_clusters, linkage):
        data, labels = dataset
            
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
            
        cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster.fit(data_scaled)
            
        return cluster.labels_

    given_data = {
        "nc": nc,
        "nm": nm,
        "bvv": bvv,
        "add": add,
        "b": b
    }

    clusters = [2]
    data_keys = ['nc', 'nm', 'bvv', 'add', 'b'] 
    link_types = ['single', 'complete', 'ward', 'average']
    pdf_name = "4b_report.pdf"
    pdf_pages = []

    fig, axes = plt.subplots(len(link_types), len(data_keys), figsize=(20, 16))
    fig.suptitle('Scatter plots for different datasets and linkage types (2 clusters)', fontsize=16)
            
    for i, link_type in enumerate(link_types):
        for j, data_key in enumerate(data_keys):
            data, labels = given_data[data_key]

            for k in clusters:
                pred_labels = fit_cluster_linkage_types(given_data[data_key], n_clusters=k, linkage=link_type)

                ax = axes[i, j]
                ax.scatter(data[:, 0], data[:, 1], c=pred_labels, cmap='viridis')
                ax.set_title(f'{link_type.capitalize()} Linkage\n{data_key}, k={k}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_pages.append(fig)
    plt.close(fig)

    with PdfPages(pdf_name) as pdf:
        for page in pdf_pages:
            pdf.savefig(page)


    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
    
    def find_distance_threshold(data_sets, linkage_method):
        
        scaler = StandardScaler()
    
        scaled_data = scaler.fit_transform(data_sets)

        linkage_matrix = linkage(scaled_data, method=linkage_method)
    
        merge_distances = np.diff(linkage_matrix[:, 2])
    
        max_rate_change_index = np.argmax(merge_distances)
    
        distance_threshold = linkage_matrix[max_rate_change_index, 2]
    
        return distance_threshold
   
    given_data = {
        "nc": nc,
        "nm": nm,
        "bvv": bvv,
        "add": add,
        "b": b
    }
    
    data_keys = ['nc', 'nm', 'bvv', 'add', 'b']
    link_type = ['single', 'complete', 'ward', 'average']
    pdf_name = "4c_report.pdf"
    pdf_pages = []
    distance_thresholds = {}
    
    fig, axes = plt.subplots(len(link_type), len(data_keys), figsize=(20, 16))
    fig.suptitle('Scatter plots for different datasets and linkage types', fontsize=16)
    
    for i, link_type in enumerate(link_type):
        for j, data_key in enumerate(data_keys):
            data, labels = given_data[data_key]
            
            distance_threshold = find_distance_threshold(data, link_type)
            distance_thresholds[(data_key, link_type)] = distance_threshold
            
            pred_labels = fit_modified(given_data[data_key], distance_threshold, link_type)
    
            ax = axes[i, j]
            ax.scatter(data[:, 0], data[:, 1], c=pred_labels, cmap='viridis')
            ax.set_title(f'{link_type.capitalize()} Linkage\n{data_key}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_pages.append(fig)
    plt.close(fig)
    
    with PdfPages(pdf_name) as pdf:
        for page in pdf_pages:
            pdf.savefig(page)
    
    
    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
