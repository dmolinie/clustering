import numpy as np
from clustering.formats._database import Database
from clustering.region_growing._region_growing import *


def test_build():
    """ Empirical Cumulative Distribution (ECD) Test """
    from sklearn.cluster import DBSCAN
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Train DBSCAN
    dbscan = DBSCAN().fit(database)
    # Build the clusters
    clusters = build(database, dbscan.labels_, cluster=True)
    print(len(clusters))

def test_dbscan_params():
    """ Check the parameters for DBSCAN clustering """
    # Get the default parameters
    print(dbscan_params())
    # Change only the threshold and leave the other default params
    print(dbscan_params(eps=0.5))
    # Change all the parameters
    print(dbscan_params(
        eps=0.5, min_samples=5, cluster=True,
        metric='minkowski', p=2))

def test_dbscan_cluster():
    """ Instantiate & train DBSCAN on a dataset and cluster it """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Cluster the database with DBSCAN with default parameters
    clusters = dbscan_cluster(database)
    print(len(clusters))
    # Cluster the database with a specific threshold
    clusters = dbscan_cluster(database, eps=0.5)
    print(len(clusters))

def test_dbscan_cluster_1d():
    """ Instantiate & train DBSCAN on a dataset and cluster it """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    array = array.ravel()
    database = Database(array, np.arange(len(array)))
    # Cluster the database with DBSCAN with default parameters
    clusters = dbscan_cluster(database)
    print(len(clusters))
    # Cluster the database with a specific threshold
    clusters = dbscan_cluster(database, eps=0.5)
    print(len(clusters))

def test_optics_params():
    """ Check the parameters for OPTICS clustering """
    # Get the default parameters
    print(optics_params())
    # Change only the threshold and leave the other default params
    print(optics_params(max_eps=0.5))
    # Change all the parameters
    print(optics_params(
        min_samples=5, max_eps=10., cluster=True,
        metric='minkowski', p=2))

def test_optics_cluster():
    """ Instantiate & train OPTICS on a dataset and cluster it """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    database = Database(array, np.arange(len(array)))
    # Cluster the database with DBSCAN with default parameters
    clusters = optics_cluster(database)
    print(len(clusters))
    # Cluster the database with a specific threshold
    clusters = optics_cluster(database, max_eps=0.5)
    print(len(clusters))

def test_optics_cluster_1d():
    """ Instantiate & train OPTICS on a dataset and cluster it """
    # Generate a dummy database with 3 Gaussian distributions (regions)
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    array = array.ravel()
    database = Database(array, np.arange(len(array)))
    # Cluster the database with DBSCAN with default parameters
    clusters = optics_cluster(database)
    print(len(clusters))
    # Cluster the database with a specific threshold
    clusters = optics_cluster(database, max_eps=0.5)
    print(len(clusters))



# Launch test/example functions
test_build()

test_dbscan_params()

test_dbscan_cluster()

test_dbscan_cluster_1d()

test_optics_params()

test_optics_cluster()

test_optics_cluster_1d()

