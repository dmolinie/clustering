""" Illustration for the `clustering` package

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: July 2024
Last revised: June 2026

License: GPLv3
"""

import numpy as np

from clustering import metrics as mts       # Data analysis tools
from clustering import cluster as clt       # Data partitioning
from clustering import datasets as sets     # Example datasets


# Generate a simple 2D dataset
database = sets.gaussian(250, 2, (0, 100), 0.5, 12, seed=10)

#--------------------------   Data Partitioning   ---------------------------#
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
   'lrn_rate_0': 0.95, 'seed': 0, 'verbose': True,
   'distance': 'euclidean'}
clusters_ksom = clt.cluster(
   database, method='ksom', fuse=0., **ksom_params)

# BSOM
bsom_params = {
  'grid_size': (3, 3), 'nb_grids': 10, 'cluster': True,
  'grid': False, 'margins': 0.01, 'tmax': 1000,
  'nbh_rate_0': 1.00, 'lrn_rate_0': 0.95, 'seed': 0,
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
stats_params = {'q': 0.50}
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
#
# # OPTICS (Scikit-Learn)
# optics_params = {
#   'min_samples': 5, 'max_eps': np.inf,
#   'metric': 'minkowski', 'p': 2, 'cluster': True}
# clusters_optics = clt.cluster(
#    database, method='optics', fuse=0., **optics_params)
#----------------------------------------------------------------------------#

import clustering.tools as tls

# tls.save_ws('data.out',
#    ('database',
#     'clusters_kms',
#     'clusters_kkms',
#     'clusters_ksom',
#     'clusters_bsom',
#     'clusters_sprada',
#     'clusters_dbscan',
#     # 'clusters_optics'
#     ))

tls.load_ws('data.out')

##############################################################################
##                           Display the Results                            ##
##############################################################################

import matplotlib.pyplot as plt

def remove_spaces(fig, no_xspace=False, no_yspace=False):
    """ Remove space between the axes of a figure """
    if no_xspace:
        fig.subplots_adjust(wspace=0.)      # Remove horizontal space
    if no_yspace:
        fig.subplots_adjust(hspace=0.)      # Remove vertical space

def slope(pt1, pt2):
    """ Compute the linear's slope `a` """
    return (pt2[0]-pt1[0]) / (pt1[1]-pt2[1])

def yintercept(pt1, pt2):
    """ Compute the linear's y-intercept `b` """
    return np.sum(pt1**2 - pt2**2) / (2*(pt1[1]-pt2[1]))

def truncate_line(i, j, mat, pats):
    """ Take a set of 2D points, find the closest 2D pattern for any of
    them, and extract the points for which either the `i`-th or `j`-th
    pattern is the actual closest one """
    cats = np.argmin([mts.euclidean_nd(pat, mat, 1) for pat in pats], 0)
    return mat[np.where(np.logical_or(cats == i, cats == j))]


# Wrap the clusters into a list
clusters_sets = [
    [clusters_kms, "K-Means"],
    [clusters_kkms, "Kernel K-Means"],
    [clusters_ksom, "KSOM"],
    [clusters_bsom, "BSOM"],
    [clusters_sprada, "SPRADA"],
    [clusters_dbscan, "DBSCAN"],
#    [clusters_optics, "OPTICS"]
    ]


# Instantiate a figure with 7 graphs, one per clustering method
fig, axs = plt.subplots(1, len(clusters_sets),
    figsize=(19.20, 19.20/len(clusters_sets)), sharey=True)

# Get the extremal values of the clusters (add a small offset for display)
mini, maxi = database.min(0), database.max(0)
mini, maxi = mini-5., maxi+5.

# Instantiate the clusters' separators (lines)
pts = np.empty((1000, 2), float)
pts[:, 0] = np.linspace(mini[0], maxi[0], 1000)


# For every method
for axis, clusters in zip(axs, clusters_sets):

    # Plot the clusters
    for clust in clusters[0]:
        axis.plot(clust.value[:, 0], clust.value[:, 1], '+')

    # Extract the clusters' patterns
    patterns = [clust.pattern for clust in clusters[0]]

    # Draw the separators between the clusters
    for i, pat1 in enumerate(patterns):
        for j, pat2 in enumerate(patterns[i+1:], i+1):
            pts[:, 1] = slope(pat1, pat2)*pts[:, 0] + yintercept(pat1, pat2)
            line = truncate_line(i, j, pts, patterns)
            axis.plot(line[:, 0], line[:, 1], 'k--', linewidth=1.)

    axis.set_title(clusters[1], size=15)

    # Remove the x & y ticks
    axis.set_xticks([])
    axis.set_yticks([])

    # Truncate the lines
    axis.set_xlim((mini[0], maxi[0]))
    axis.set_ylim((mini[1], maxi[1]))

# Adjust the plots
plt.tight_layout()
# Remove the spaces between the subplots
remove_spaces(fig, True, True)

# Save the figure in a file
plt.savefig('illustration.pdf', bbox_inches='tight', dpi=300)
plt.close()

##############################################################################
