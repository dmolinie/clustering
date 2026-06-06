import numpy as np
from clustering.formats._database import Database
from clustering.ecd._ecd_test import *


def test_ECDTest():
    """ Empirical Cumulative Distribution (ECD) Test """
    # Provide the parameters for KSOM clustering (optional)
    kparams = {
        'grid_size': (3, 3), 'cluster': True, 'grid': False,
        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        'distance': 'euclidean'}
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    #--- Use the ECD Test on a NumPy array
    # Instantiate an `ECDTest` object
    ecd = ECDTest()
    # Cluster the database using the specified method
    ecd.fit(method='ksom', database=array, **kparams)
    # Build the matrix of MHDs
    ecd.build_matrix(50)
    # Build the groups of clusters
    ecd.build_groups(0.15)
    # Retrieve the estimated number of regions in the space
    est_reg = ecd.nb_regions()
    #--- Use the ECD Test on a Database and build Clusters
    database = Database(array, np.arange(len(array)))
    # Instantiate an `ECDTest` object
    ecd = ECDTest()
    # Cluster the database using the specified method
    ecd.fit(method='ksom', database=database, **kparams)
    # Build the matrix of MHDs
    ecd.build_matrix(50)
    # Build the groups of clusters
    ecd.build_groups(0.15)
    # Retrieve the estimated number of regions in the space
    est_reg = ecd.nb_regions()

def test_ECDTest_1d():
    """ Empirical Cumulative Distribution (ECD) Test """
    # Provide the parameters for KSOM clustering (optional)
    kparams = {
        'grid_size': (3, 3), 'cluster': True, 'grid': False,
        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        'distance': 'euclidean'}
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    #--- Test on 1D row vector
    array = array.ravel()
    database = Database(array, np.arange(len(array)))
    # Instantiate an `ECDTest` object
    ecd = ECDTest()
    # Cluster the database using the specified method
    ecd.fit(method='ksom', database=database, **kparams)
    # Build the matrix of MHDs
    ecd.build_matrix(50)
    # Build the groups of clusters
    ecd.build_groups(0.15)
    # Retrieve the estimated number of regions in the space
    est_reg = ecd.nb_regions()
    #--- Test on 1D column vector
    array = array.reshape(-1, 1)
    database = Database(array, np.arange(len(array)))
    # Instantiate an `ECDTest` object
    ecd = ECDTest()
    # Cluster the database using the specified method
    ecd.fit(method='ksom', database=database, **kparams)
    # Build the matrix of MHDs
    ecd.build_matrix(50)
    # Build the groups of clusters
    ecd.build_groups(0.15)
    # Retrieve the estimated number of regions in the space
    est_reg = ecd.nb_regions()

def test_ECDTest_init():
    """ Instantiate an ECDTest object (constructor) """
    # Instantiate an `ECDTest` object
    ecd = ECDTest()

def test_ECDTest_fit():
    """ Set the clusters, or build them from a database """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate an `ECDTest` object
    ecd = ECDTest()
    # Directly pass the clusters to the ECD Test
    from clustering.ksom import ksom_cluster
    clusters = ksom_cluster(database, grid_size=(3, 3))
    clusters = [clt for clt in clusters if len(clt) != 0]   # or `prune`
    print(len(clusters))
    ecd.fit(clusters=clusters)
    print(len(ecd.clusters))
    # Or build the clusters using a dedicated method from the database
    # (empty clusters are automatically removed)
    ecd.fit(method='ksom', database=database, grid_size=(3, 3))
    print(len(ecd.clusters))

def test_ECDTest_build_matrix():
    """ Compute the ECDs and build the matrix of the MHDs """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate an `ECDTest` object
    ecd = ECDTest()
    # Cluster the database using the specified method
    ecd.fit(method='ksom', database=database, grid_size=(3, 3))
    # Build the matrix of MHDs
    ecd.build_matrix(50)
    print(ecd.matrix.shape)

def test_ECDTest_build_groups():
    """ Build the groups by Regions Growing """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate an `ECDTest` object
    ecd = ECDTest()
    # Cluster the database using the specified method
    ecd.fit(method='ksom', database=database, grid_size=(3, 3))
    # Build the matrix of MHDs
    ecd.build_matrix(50)
    # Build the groups of clusters
    ecd.build_groups(0.15)
    print(ecd.groups)

def test_ECDTest_nb_regions():
    """ Estimate the number of regions """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Instantiate an `ECDTest` objects
    ecd = ECDTest()
    # Cluster the database using the specified method
    ecd.fit(method='ksom', database=database, grid_size=(3, 3))
    # Build the matrix of MHDs
    ecd.build_matrix(50)
    # Build the groups of clusters
    ecd.build_groups(0.15)
    # Retrieve the estimated number of regions in the space
    est_reg = ecd.nb_regions()

def test_ecd_test():
    """ Estimate the number of regions in a feature space """
    # Provide the parameters for KSOM clustering (optional)
    kparams = {
        'grid_size': (3, 3), 'cluster': True, 'grid': False,
        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        'distance': 'euclidean'}
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    #--- Use the ECD Test on a NumPy array
    ecd_test('ksom', 100, 0.15, database=array, **kparams)
    #--- Use the ECD Test on a Database and build Clusters
    database = Database(array, np.arange(len(array)))
    ecd_test('ksom', 100, 0.15, database=database, **kparams)



# Launch test/example functions
test_ECDTest()

test_ECDTest_1d()

test_ECDTest_init()

test_ECDTest_fit()

test_ECDTest_build_matrix()

test_ECDTest_build_groups()

test_ECDTest_nb_regions()

test_ecd_test()

