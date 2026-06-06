import numpy as np
from clustering.formats._database import Database
from clustering.sprada._sprada import *


def test_SPRADA():
    """ Self-Parameterized Recursively Assessed Decomposition Algorithm """
    #--- Apply the SPRADA decomposition to a NumPy array
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    # Instantiate a `SPRADA` object with default parameters
    tree = SPRADA()
    # Build the clusters
    clusters = tree.fit(array)
    print(type(clusters[0]))
    print(len(clusters))
    # Build the matrix of the MHDs
    matrix = tree.build_matrix()
    # Regroup the clusters by distances
    groups = tree.build_groups()
    # Merge the clusters of the groups
    clusters = tree.merge_groups()
    print(type(clusters[0]))
    print(len(clusters))
    #--- Apply the SPRADA decomposition to a Database and build Clusters
    database = Database(array, np.arange(len(array)))
    # Instantiate a `SPRADA` object with default parameters
    tree = SPRADA()
    # Build the clusters
    clusters = tree.fit(database)
    print(type(clusters[0]))
    print(len(clusters))
    # Build the matrix of the MHDs
    matrix = tree.build_matrix()
    # Regroup the clusters by distances
    groups = tree.build_groups()
    # Merge the clusters of the groups
    clusters = tree.merge_groups()
    print(type(clusters[0]))
    print(len(clusters))
    #--- Apply the SPRADA decomposition with specific parameters
    #--- and leave the clusters as `np.ndarrays` (no `Cluster` objects)
    # Parameters for `KohonenSOM`
    ksom_params = {
        'grid_size': (3, 3), 'cluster': True, 'grid': False,
        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        'distance': 'euclidean'}
    # Parameters for `KMeans`
    kmeans_params = {
        'nb_clusters': 2, 'cluster': True,
        'margins': 0.01, 'tmax': 100, 'seed': None,
        'verbose': False, 'distance': 'euclidean'}
    # Initialize the tree with these functions & parameters
    tree = SPRADA(
        fclt1='ksom', fclt1_params=ksom_params,
        fclt2='kmeans', fclt2_params=kmeans_params)
    # Build the clusters (NumPy ndarrays returned as `cluster` is False
    # in both `ksom_params` and `kmeans_params`)
    clusters = tree.fit(database)
    print(type(clusters[0]))
    print(len(clusters))
    # Build the matrix of the MHDs
    matrix = tree.build_matrix(100)
    # Regroup the clusters by distances
    groups = tree.build_groups(0.1)
    # Merge the clusters of the groups
    clusters = tree.merge_groups()
    print(type(clusters[0]))
    print(len(clusters))

def test_SPRADA_1d():
    """ Self-Parameterized Recursively Assessed Decomposition Algorithm """
    #--- Apply the SPRADA decomposition to a NumPy array
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    #--- Test on 1D row vector
    array = array.ravel()
    database = Database(array, np.arange(len(array)))
    # Instantiate a `SPRADA` object with default parameters
    tree = SPRADA()
    # Build the clusters
    clusters = tree.fit(array)
    print(len(clusters))
    # Build the matrix of the MHDs
    matrix = tree.build_matrix()
    # Regroup the clusters by distances
    groups = tree.build_groups()
    # Merge the clusters of the groups
    clusters = tree.merge_groups()
    print(len(clusters))
    #--- Test on 1D column vector
    array = array.reshape(-1, 1)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `SPRADA` object with default parameters
    tree = SPRADA()
    # Build the clusters
    clusters = tree.fit(array)
    print(len(clusters))
    # Build the matrix of the MHDs
    matrix = tree.build_matrix()
    # Regroup the clusters by distances
    groups = tree.build_groups()
    # Merge the clusters of the groups
    clusters = tree.merge_groups()
    print(len(clusters))

def test_SPRADA_init():
    """ Instantiate a SPRADA object (constructor) """
    #--- Use the default parameters
    tree = SPRADA()
    #--- Use specific functions with their default parameters
    tree = SPRADA(
        fclt1='ksom', fclt2='ksom', fquant='std', fstats='mean')
    #--- Use specific functions & their parameters
    # Parameters for `density`
    den_params = {
        'span': 'sphere_span',
        'volume': 'hypersphere',
        'distance': 'euclidean'}
    # Parameters for `quantile`
    stats_params = {'q': 0.25}
    # Parameters for `KohonenSOM`
    ksom_params = {
        'grid_size': (3, 3), 'cluster': True, 'grid': False,
        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        'distance': 'euclidean'}
    # Parameters for `KMeans`
    kmeans_params = {
        'nb_clusters': 2, 'cluster': True,
        'margins': 0.01, 'tmax': 100, 'seed': None,
        'verbose': False, 'distance': 'euclidean'}
    # Initialize the tree with these functions & parameters
    tree = SPRADA(
        fclt1='ksom', fclt1_params=ksom_params,
        fclt2='kmeans', fclt2_params=kmeans_params,
        fquant='density', fquant_params=den_params,
        fstats='quantile', fstats_params=stats_params,
        comparison='greater')

def test_SPRADA_fit():
    """ Build the tree & pass the clusters to the ECD Test """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate a tree with `std` quantifier
    tree = SPRADA(fquant='std', comparison='lower')
    # Build the clusters (set `comparison` to 'lower',
    # since for `std`, the lower, the better
    clusters = tree.fit(database)
    print(len(clusters))
    print(clusters == tree.ecdtest.clusters)
    # Instantiate a `SPRADA` object with default parameters
    tree = SPRADA(fquant='density', comparison='greater')
    # Build the clusters (set `comparison` to 'greater',
    # since for `density`, the greater, the better
    clusters = tree.fit(database)
    print(len(clusters))
    print(clusters == tree.ecdtest.clusters)

def test_SPRADA_build_matrix():
    """ Compute the ECDs and build the matrix of the MHDs """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate a `SPRADA` object with default parameters
    tree = SPRADA()
    # Build the clusters
    clusters = tree.fit(array)
    # Build the matrix of the MHDs
    matrix = tree.build_matrix()
    print(matrix.shape)

def test_SPRADA_build_groups():
    """ Build the groups by Regions Growing """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate a `SPRADA` object with default parameters
    tree = SPRADA()
    # Build the clusters
    clusters = tree.fit(array)
    # Build the matrix of the MHDs
    matrix = tree.build_matrix()
    # Regroup the clusters by distances
    groups = tree.build_groups()
    print(groups)

def test_SPRADA_merge_groups():
    """ Merge the clusters of the groups identified by the ECD Test """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate a `SPRADA` object with default parameters
    tree = SPRADA()
    # Build the clusters
    clusters = tree.fit(array)
    print(len(clusters))
    # Build the matrix of the MHDs
    matrix = tree.build_matrix()
    # Regroup the clusters by distances
    groups = tree.build_groups()
    # Merge the clusters of the groups
    clusters = tree.merge_groups()
    print(len(clusters))

def test_sprada_params():
    """ Check the parameters for SPRADA clustering """
    #--- Get the default parameters
    print(sprada_params())
    #--- Change only the number of clusters and leave the other default params
    print(sprada_params(
        fstats='quantile', fstats_params={'q': 0.25}, eps=0.10))
    #--- Change all the parameters
    # Parameters for `density`
    den_params = {
        'span': 'sphere_span',
        'volume': 'hypersphere',
        'distance': 'euclidean'}
    # Parameters for `quantile`
    stats_params = {'q': 0.25}
    # Parameters for `KohonenSOM`
    ksom_params = {
        'grid_size': (3, 3), 'cluster': True, 'grid': False,
        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        'distance': 'euclidean'}
    # Parameters for `KMeans`
    kmeans_params = {
        'nb_clusters': 2, 'cluster': True,
        'margins': 0.01, 'tmax': 100, 'seed': None,
        'verbose': False, 'distance': 'euclidean'}
    print(sprada_params(
        fclt1='ksom', fclt1_params=ksom_params,
        fclt2='kmeans', fclt2_params=kmeans_params,
        fquant='density', fquant_params=den_params,
        fstats='quantile', fstats_params=stats_params,
        comparison='greater', pts=100, eps=0.10))

def test_sprada_cluster():
    """ Instantiate & train SPRADA on a dataset and cluster it """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Split the database with default parameters
    clusters = sprada_cluster(database)
    print(len(clusters))
    # Split the database with some specific parameters
    clusters = sprada_cluster(database,
        fclt1='ksom', fclt2='kmeans',
        fquant='density', fstats='mean',
        comparison='greater', pts=100, eps=0.10)
    print(len(clusters))



# Launch test/example functions
test_SPRADA()

test_SPRADA_1d()

test_SPRADA_init()

test_SPRADA_fit()

test_SPRADA_build_matrix()

test_SPRADA_build_groups()

test_SPRADA_merge_groups()

test_sprada_params()

test_sprada_cluster()

