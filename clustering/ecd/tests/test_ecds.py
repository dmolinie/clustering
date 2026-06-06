import numpy as np
from clustering.formats._database import Database
from clustering.ecd._ecds import *


def test_cumulative_distributions():
    """ Empirical Cumulative Distribution Functions (ECDs) """
    # Generate a dummy list of clusters
    arr = np.arange(50., dtype=float).reshape(-1, 5)
    arrs = [arr+i for i in range(4)]
    # Compute the ECDs without the database
    vals, cdfs = cumulative_distributions(arrs, None, 100)
    print(np.shape(vals), np.shape(cdfs))
    # Compute the ECDs with the database
    vals, cdfs = cumulative_distributions(arrs, arrs[0], 100)
    print(np.shape(vals), np.shape(cdfs))

def test_cdfs_matrix():
    """ Build the matrix of the MHDs between the ECDs """
    from clustering.ksom import ksom_cluster
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Cluster the database
    clusters = ksom_cluster(database, grid_size=(3, 3))
    clusters = [clt for clt in clusters if len(clt) != 0]   # or `prune`
    print(len(clusters))
    # Compute the Empirical Cumulative Distributions (ECDs)
    vals, cdfs = cumulative_distributions(clusters, pts=100)
    # Build the matrix of MHDs
    matrix = cdfs_matrix(vals, cdfs)
    print(matrix.shape)

def test_cdfs_states():
    """ Build the closeness states matrix """
    from clustering.ksom import ksom_cluster
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Cluster the database
    clusters = ksom_cluster(database, grid_size=(3, 3))
    clusters = [clt for clt in clusters if len(clt) != 0]   # or `prune`
    print(len(clusters))
    # Compute the Empirical Cumulative Distributions (ECDs)
    vals, cdfs = cumulative_distributions(clusters, pts=100)
    # Build the matrix of MHDs
    matrix = cdfs_matrix(vals, cdfs)
    # Build the states matrix
    states, idx = cdfs_states(matrix)
    print(states)

def test_cdfs_groups():
    """ Build the closeness map """
    from clustering.ksom import ksom_cluster
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Cluster the database
    clusters = ksom_cluster(database, grid_size=(3, 3))
    clusters = [clt for clt in clusters if len(clt) != 0]   # or `prune`
    print(len(clusters))
    # Compute the Empirical Cumulative Distributions (ECDs)
    vals, cdfs = cumulative_distributions(clusters, pts=100)
    # Build the matrix of MHDs
    matrix = cdfs_matrix(vals, cdfs)
    # Build the states matrix
    states, idx = cdfs_states(matrix)
    # Build the groups of clusters
    groups = cdfs_groups(idx, len(clusters))
    print(groups)
    print("Estimated number of regions:", len(groups))

def test_ks_test():
    """ Kolmogorov-Smirnov (KS) Test """
    # Generate a dummy list of clusters
    arr = np.arange(50., dtype=float).reshape(-1, 5)
    arrs = [arr+i for i in range(4)]
    # Compute the ECDs without the database
    print(ks_test(arrs, None, 100))
    # Compute the ECDs with the database
    print(ks_test(arrs, arrs[0], 100))


# Launch test/example functions
test_cumulative_distributions()

test_cdfs_matrix()

test_cdfs_states()

test_cdfs_groups()

test_ks_test()

