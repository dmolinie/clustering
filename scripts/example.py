""" Example of use of the `clustering` package

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: July 2024
Last revised: June 2026

License: GPLv3
"""

import numpy as np

from clustering import metrics as mts       # Data analysis tools
from clustering import ecd                  # KS & ECD Tests
from clustering import cluster as clt       # Data partitioning
from clustering import datasets as sets     # Example datasets


# Generate a simple dataset
database = sets.gaussian(250, 5, (0, 100), 1., 10, seed=21)


#--------------------------   Data Partitioning   ---------------------------#
# In the following examples, the full sets of parameters for any clustering
# functions are explicitly provided, but one may provide only some of them,
# or even no parameter; in such a case, default values will be used. For any
# clustering function, a dedicated function that checks the set of parameters
# is made available in the same module as the one implementing the clustering
# function, e.g. `kmeans_params` from the `kmeans` module, where the `KMeans`
# clustering algorithm is implemented as a class and a standalone function.

# K-Means
kms_params = {
   'nb_clusters': 2, 'cluster': True, 'margins': 0.01, 'tmax': 100,
   'seed': 0, 'verbose': True, 'distance': 'euclidean'}
clusters_kms = clt.cluster(
    database, method='kmeans', fuse=0., **kms_params)

# Kernel K-Means
kkms_params = {
   'nb_clusters': 2, 'cluster': True, 'margins': 0.01, 'tmax': 100,
   'seed': 0, 'verbose': True, 'distance': 'euclidean',
   'kernel': 'gaussian', 'ker_params': {'sigma': 0.1}}
clusters_kkms = clt.cluster(
    database, method='kernel_kmeans', fuse=0., **kkms_params)

# KSOM
ksom_params = {
    'grid_size': (3, 3), 'cluster': True, 'grid': False,
    'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
    'lrn_rate_0': 0.95, 'seed': 10, 'verbose': True,
    'distance': 'euclidean'}
clusters_ksom = clt.cluster(
    database, method='ksom', fuse=0., **ksom_params)

# BSOM
bsom_params = {
   'grid_size': (3, 3), 'nb_grids': 10, 'cluster': True,
   'grid': False, 'margins': 0.01, 'tmax': 1000,
   'nbh_rate_0': 1.00, 'lrn_rate_0': 0.95, 'seed': 10,
   'verbose': True, 'distance': 'euclidean'}
clusters_bsom = clt.cluster(
    database, method='bsom', fuse=0., **bsom_params)

# SPRADA
# Parameters for `density`
quant_params = {
   'span': 'sphere_span',
   'volume': 'hypersphere',
   'distance': 'euclidean'}
# Parameters for `quantile`
stats_params = {'q': 0.66}
# Check the parameters
sprada_params = {
   'fclt1': 'ksom', 'fclt1_params': ksom_params,
   'fclt2': 'kmeans', 'fclt2_params': kms_params,
   'fquant': 'density', 'fquant_params': quant_params,
   'fstats': 'quantile', 'fstats_params': stats_params,
   'comparison': 'greater'}
clusters_sprada = clt.cluster(
    database, method='sprada', fuse=0., **sprada_params)

# DBSCAN (Scikit-Learn)
dbscan_params = {
   'eps': 0.5, 'min_samples': 5,
   'metric': 'minkowski', 'p': 2, 'cluster': True}
clusters_dbscan = clt.cluster(
    database, method='dbscan', fuse=0., **dbscan_params)

# OPTICS (Scikit-Learn)
optics_params = {
   'min_samples': 5, 'max_eps': np.inf,
   'metric': 'minkowski', 'p': 2, 'cluster': True}
clusters_optics = clt.cluster(
    database, method='optics', fuse=0., **optics_params)
#----------------------------------------------------------------------------#

#-----------------------   Data Cluster Evaluation   ------------------------#
# Quantifiers & HyDAS
quant_params = {
    'span': 'sphere_span',
    'volume': 'hypersphere',
    'distance': 'euclidean'}
quants = mts.quantifiers(clusters_ksom, **quant_params)
score = mts.hydas(clusters_ksom, **quant_params)

# Indexes
j_idx = mts.jaccard_index(clusters_ksom[0], clusters_ksom[1])
s_idx = mts.sorensen_index(clusters_ksom[0], clusters_ksom[1])

# KS Test
ks_test = ecd.ks_test(clusters_ksom)
ks_test_dba = ecd.ks_test(clusters_ksom, database)
#----------------------------------------------------------------------------#

#--------------------------   Display Clustering   --------------------------#
if __name__ == '__main__':
    # Display the two first dimensions of the database (among 5)
    import matplotlib.pyplot as plt

    # Wrap the clusters into a list
    clusters_sets = [
        [clusters_kms, "K-Means"],
        [clusters_kkms, "Kernel K-Means"],
        [clusters_ksom, "KSOM"],
        [clusters_bsom, "BSOM"],
        [clusters_sprada, "SPRADA"],
        [clusters_dbscan, "DBSCAN"],
        [clusters_optics, "OPTICS"],
    ]

    # Instantiate a figure with 7 graphs, one per clustering method
    fig, axs = plt.subplots(
        1, len(clusters_sets), figsize=(19.20, 4.00), sharey=True)

    # For every method
    for axis, clusters in zip(axs, clusters_sets):
        # Plot the clusters
        for clust in clusters[0]:
            axis.plot(clust.value[:, 0], clust.value[:, 1], '+')
        # Labels & Titles
        axis.set_title(clusters[1], size=20)
        # Remove the x & y ticks
        axis.set_xticks([])
        axis.set_yticks([])

    plt.tight_layout()
    plt.show()
#----------------------------------------------------------------------------#
