import numpy as np
from clustering.formats._database import Database
from clustering.sprada._recursive import *


def test_Recursive():
    """ Recursive Decomposition """
    #--- Apply the Recursive decomposition to a NumPy array
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    # Instantiate a `Recursive` object with default parameters
    tree = Recursive()
    # Build the clusters
    clusters = tree.fit(array)
    print(type(clusters[0]))
    print(len(clusters))
    #--- Apply the Recursive decomposition to a Database and build Clusters
    database = Database(array, np.arange(len(array)))
    # Instantiate a `Recursive` object with default parameters
    tree = Recursive()
    # Build the clusters
    clusters = tree.fit(database)
    print(type(clusters[0]))
    print(len(clusters))
#    #--- Apply the Recursive decomposition with specific parameters
#    # Parameters for `KohonenSOM`
#    ksom_params = {
#        'grid_size': (3, 3), 'cluster': True, 'grid': False,
#        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
#        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
#        'distance': 'euclidean'}
#    # Parameters for `KMeans`
#    kmeans_params = {
#        'nb_clusters': 2, 'cluster': True,
#        'margins': 0.01, 'tmax': 100, 'seed': None,
#        'verbose': False, 'distance': 'euclidean'}
#    # Parameters for `density`
#    den_params = {
#        'span': 'sphere_span',
#        'volume': 'hypersphere',
#        'distance': 'euclidean'}
#    # Parameters for `quantile`
#    stats_params = {'q': 0.25}
#    # Initialize the tree with these functions & parameters
#    tree = Recursive(
#        fclt1='ksom', fclt1_params=ksom_params,
#        fclt2='kmeans', fclt2_params=kmeans_params,
#        fquant='density', fquant_params=den_params,
#        fstats='quantile', fstats_params=stats_params,
#        comparison='greater')
#    # Build the clusters (NumPy ndarrays returned as `cluster` is False
#    # in both `ksom_params` and `kmeans_params`)
#    clusters = tree.fit(database)
#    print(type(clusters[0]))
#    print(len(clusters))
    #--- Compare the comparison strategies
    # Std is 'the lower the better'
    # 1st quantile with 'greater': 25% considered compact (75% to split)
    tree = Recursive(
        fquant='std', comparison='lower',
        fstats='quantile', fstats_params={'q': 0.25})
    clusters = tree.fit(database)
    print(len(clusters))
    # Std is 'the lower the better'
    # 3rd quantile with 'greater': 75% considered compact (25% to split)
    tree = Recursive(
        fquant='std', comparison='lower',
        fstats='quantile', fstats_params={'q': 0.75})
    clusters = tree.fit(database)
    print(len(clusters))
    # Density is 'the greater the better'
    # 3rd quantile with 'greater': 25% considered compact (75% to split)
    tree = Recursive(
        fquant='density', comparison='greater',
        fstats='quantile', fstats_params={'q': 0.75})
    clusters = tree.fit(database)
    print(len(clusters))
    # Density is 'the greater the better'
    # 1st quantile with 'greater': 25% considered compact (75% to split)
    tree = Recursive(
        fquant='density', comparison='greater',
        fstats='quantile', fstats_params={'q': 0.25})
    clusters = tree.fit(database)
    print(len(clusters))

def test_Recursive_1d():
    """ Recursive Decomposition """
    #--- Apply the Recursive decomposition to a NumPy array
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    #--- Test on 1D row vector
    array = array.ravel()
    database = Database(array, np.arange(len(array)))
    tree = Recursive()
    # Build the clusters
    clusters = tree.fit(array)
    print(len(clusters))
    #--- Test on 1D column vector
    array = array.reshape(-1, 1)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `Recursive` object with default parameters
    tree = Recursive()
    # Build the clusters
    clusters = tree.fit(database)
    print(len(clusters))

def test_Recursive_init():
    """ Instantiate a Recursive object (constructor) """
    #--- Use the default parameters
    tree = Recursive()
    #--- Use specific functions with their default parameters
    tree = Recursive(
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
    tree = Recursive(
        fclt1='ksom', fclt1_params=ksom_params,
        fclt2='kmeans', fclt2_params=kmeans_params,
        fquant='density', fquant_params=den_params,
        fstats='quantile', fstats_params=stats_params,
        comparison='greater')

def test_Recursive_fit():
    """ Build the tree recursively """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate a tree with `std` quantifier
    tree = Recursive(fquant='std', comparison='lower')
    # Build the clusters (set `comparison` to 'lower',
    # since for `std`, the lower, the better
    clusters = tree.fit(database)
    print(len(clusters))
    # Instantiate a `Recursive` object with default parameters
    tree = Recursive(fquant='density', comparison='greater')
    # Build the clusters (set `comparison` to 'greater',
    # since for `density`, the greater, the better
    clusters = tree.fit(database)
    print(len(clusters))

def test_recursive_params():
    """ Check the parameters for Recursive clustering """
    #--- Get the default parameters
    print(recursive_params())
    #--- Change only the number of clusters and leave the other default params
    print(recursive_params(fstats='quantile', fstats_params={'q': 0.25}))
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
    print(recursive_params(
        fclt1='ksom', fclt1_params=ksom_params,
        fclt2='kmeans', fclt2_params=kmeans_params,
        fquant='density', fquant_params=den_params,
        fstats='quantile', fstats_params=stats_params,
        comparison='greater'))

def test_recursive_cluster():
    """ Instantiate & train a Recursive Tree on a dataset and cluster it """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Split the database with default parameters
    clusters = recursive_cluster(database)
    print(len(clusters))
    # Split the database with some specific parameters
    clusters = recursive_cluster(database,
        fclt1='ksom', fclt2='kmeans',
        fquant='density', fstats='mean',
        comparison='greater')
    print(len(clusters))



# Launch test/example functions
test_Recursive()

test_Recursive_1d()

test_Recursive_init()

test_Recursive_fit()

test_recursive_params()

test_recursive_cluster()

