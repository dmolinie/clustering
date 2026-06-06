import numpy as np
from clustering.formats._database import Database
from clustering.kmeans._kernel_kmeans import *


def test_KernelKMeans():
    """ Lloyd's Kernel KMeans clustering """
    #--- Use the Kernel K-Means on a NumPy array
    # Generate a dummy database
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3))))
    # Instantiate a `KernelKMeans` object
    kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)
    # Train the Kernel KMeans
    kkms.fit(array, verbose=True)
    # Build the clusters from the trained patterns
    clusters = kkms.build(array, cluster=False)     # NumPy arrays
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = kkms.winner(array[15])
    print(bmu)
    #--- Use the Kernel K-Means on a Database and build Clusters
    # Generate a dummy database
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KernelKMeans` object
    kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)
    # Train the Kernel KMeans
    kkms.fit(database, verbose=True)
    # Build the clusters from the trained patterns
    clusters = kkms.build(database, cluster=True)   # Cluster class objects
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = kkms.winner(database[15])
    print(bmu)

def test_KernelKMeans_1d():
    """ Lloyd's Kernel KMeans clustering """
    #--- Test on 1D row vector
    # Generate a dummy database
    rng = np.random.default_rng()
    array = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3))))
    array = array.ravel()
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KernelKMeans` object
    kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)
    # Train the Kernel KMeans
    kkms.fit(database, verbose=False)
    # Build the clusters from the trained patterns
    clusters = kkms.build(database, cluster=True)     # NumPy arrays
    print(type(clusters[0]))
    # Find the BMU of a single data
    bmu = kkms.winner(database[101])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = kkms.winner(database[95:105])
    print(bmu)
    #--- Test on 1D column vector
    # Generate a dummy database
    array = array.reshape(-1, 1)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KernelKMeans` object
    kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)
    # Train the Kernel KMeans
    kkms.fit(database, verbose=False)
    # Build the clusters from the trained patterns
    clusters = kkms.build(database, cluster=True)   # Cluster class objects
    print(type(clusters[0]))
    # Find the BMU of a single data
    bmu = kkms.winner(database[101])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = kkms.winner(database[95:105])
    print(bmu)

def test_KernelKMeans_init():
    """ Instantiate a KernelKMeans object (constructor) """
    kkms = KernelKMeans()
    kkms = KernelKMeans(2, 123)
    kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)

def test_KernelKMeans_fit():
    """ Train the Kernel KMeans on the data """
    # Generate a dummy database
    rng = np.random.default_rng()
    data = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3))))
    database = Database(data, np.arange(len(data)))
    # Instantiate a `KernelKMeans` object
    kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)
    # Train the Kernel KMeans
    kkms.fit(database, verbose=True)
    print(kkms.patterns.round(3))

def test_KernelKMeans_build():
    """ Build the clusters according to the current patterns """
    # Generate a dummy database
    rng = np.random.default_rng()
    data = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3))))
    database = Database(data, np.arange(len(data)))
    # Instantiate a `KernelKMeans` object
    kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)
    # Train the Kernel KMeans
    kkms.fit(database, verbose=True)
    # Build the clusters from the trained patterns
    kkms.build(database, cluster=True)

def test_KernelKMeans_winner():
    """ Find the Best Matching Unit for a data """
    # Generate a dummy database
    rng = np.random.default_rng()
    data = np.vstack(
        (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
         rng.normal(loc=5.0, scale=0.01, size=(100, 3))))
    database = Database(data, np.arange(len(data)))
    # Instantiate a `KernelKMeans` object
    kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)
    # Train the Kernel KMeans
    kkms.fit(database, verbose=True)
    # Build the clusters from the trained patterns
    kkms.build(database, cluster=True)
    # Find the BMU of a single data
    bmu = kkms.winner(database[101])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = kkms.winner(database[95:105])
    print(bmu)

def test_kkmeans_params():
    """ Check the parameters for the Kernel KMeans """
    # Get the default parameters
    print(kkmeans_params())
    # Change only the number of clusters and leave the other default params
    print(kkmeans_params(nb_clusters=4))
    # Change all the parameters
    print(kkmeans_params(
        nb_clusters=4, cluster=False, margins=0.1, tmax=50, seed=123,
        verbose=True, kernel='gaussian', ker_params={'sigma': 0.1}))

def test_kkmeans_cluster():
    """ Instantiate & train the Kernel K-Means on a dataset and cluster it """
    rng = np.random.default_rng()
    data = np.vstack(
        (rng.normal(loc=0.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=3.0, scale=0.01, size=(100, 3)),
         rng.normal(loc=9.0, scale=0.01, size=(100, 3))))
    database = Database(data, np.arange(len(data)))
    # Cluster the database with the KMeans with default parameters
    clusters = kkmeans_cluster(database)
    # Cluster the database with the KMeans with some specific parameters
    clusters = kkmeans_cluster(database, nb_clusters=5, verbose=True)



# Launch test/example functions
test_KernelKMeans()

test_KernelKMeans_1d()

test_KernelKMeans_init()

test_KernelKMeans_fit()

test_KernelKMeans_build()

test_KernelKMeans_winner()

test_kkmeans_params()

test_kkmeans_cluster()

