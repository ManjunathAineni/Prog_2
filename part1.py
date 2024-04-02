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
from scipy.cluster.hierarchy import dendrogram, linkage  

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. 
# Do NOT move it. Change the arguments and return according to the question asked. 
def fit_kmeans(data, k):
    dataset, labels = data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dataset)
    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    kmeans.fit(data_scaled)
    pred_labels = kmeans.labels_ 
    return pred_labels

def fit_kmeans_1D(data, k):
    dataset, labels = data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dataset)
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(data_scaled)
    pred_labels = kmeans.labels_ 
    return pred_labels

def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: 
        - noisy_circles (nc)
        - noisy_moons (nm)
        - blobs with varied variances (bvv)
        - Anisotropicly distributed data (add)
        - blobs (b). 
        Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), 
        with any random state. (with random_state = 42). 
        Not setting the correct random_state will prevent me from checking your results.
    """

    dct = answers["1A: datasets"] = {}
    seed = 42
    nc = datasets.make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=seed)
    answers["1A: datasets"]["nc"] = list(nc)
    nm = datasets.make_moons(n_samples=100, noise=0.05, random_state=seed)
    answers["1A: datasets"]["nm"] = list(nm)
    bvv = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    answers["1A: datasets"]["bvv"] = list(bvv)
    X, y = datasets.make_blobs(n_samples=100, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)
    answers["1A: datasets"]["add"] = list(add)
    b = datasets.make_blobs(n_samples=100, random_state=seed)
    answers["1A: datasets"]["b"] = list(b)

    """
    B. Write a function called fit_kmeans that takes dataset (before any processing on it), 
    i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, 
    and returns the predicted labels from k-means clustering. 
    Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), 
    prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    dct = answers["1B: fit_kmeans"] = fit_kmeans

    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) 
    with each column corresponding to the datasets generated in part 1.A, 
    and each row being k=[2,3,5,10] different number of clusters. 
    For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) 
    and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    given_data = {
        "nc": nc,
        "nm": nm,
        "bvv": bvv,
        "add": add,
        "b": b
    }

    cluster_counts = [2, 3, 5, 10]
    data_keys = ['nc', 'nm', 'bvv', 'add', 'b']
    pdf_file = "1c_report.pdf"
    pdf_pages = []

    fig, axes = plt.subplots(len(cluster_counts), len(data_keys), figsize=(20, 16))

    for j, key in enumerate(data_keys):
        data, labels = given_data[key]

        for i, count in enumerate(cluster_counts):
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            kmeans = KMeans(n_clusters=count, init='random', random_state=42)
            kmeans.fit(scaled_data)
            predicted_labels = kmeans.labels_

            ax = axes[i, j]
            ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
            ax.set_title(f'{key}, k={count}')

    plt.tight_layout()

    with PdfPages(pdf_file) as pdf:
        pdf.savefig(fig)

    dct = answers["1C: cluster successes"] = {"bvv": [3], "add": [3],"b":[3]} 
    dct = answers["1C: cluster failures"] = ["nc","nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization 
    for the k=2,3 cases. 
    You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """
    cluster_counts = [2, 3]
    data_keys = ['nc', 'nm', 'bvv', 'add', 'b']

    for index in range(4):
        pdf_file = f"1d_report_{index}.pdf"  # Dynamic file name based on index
    
        fig, axes = plt.subplots(len(cluster_counts), len(data_keys), figsize=(20, 16))

        for j, key in enumerate(data_keys):
            data, labels = given_data[key]

            for i, count in enumerate(cluster_counts):
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)

                kmeans = KMeans(n_clusters=count, init='random')
                kmeans.fit(scaled_data)
                predicted_labels = kmeans.labels_

                ax = axes[i, j]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'{key}, k={count}')

        plt.tight_layout()

        with PdfPages(pdf_file) as pdf:
            pdf.savefig(fig)


    dct = answers["1D: datasets sensitive to initialization"] = ["nc","nm"]

    return answers


if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
