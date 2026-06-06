import numpy as np
from clustering.formats._database import Database
from clustering.kmeans._kmeans import *


def test_KMeans():
    """ Lloyd's KMeans clustering """
    #--- Use the K-Means on a NumPy array
    # Generate a dummy database
    array = np.arange(100.).reshape(-1, 5)
    # Instantiate a `KMeans` object
    kms = KMeans(2, distance='euclidean')
    # Train the KMeans
    kms.fit(array, verbose=True)
    # Build the clusters using the trained patterns
    clusters = kms.build(array, cluster=False)      # NumPy arrays
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = kms.winner(array[15])
    print(bmu)
    #--- Use the K-Means on a Database and build Clusters
    # Generate a dummy database
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KMeans` object
    kms = KMeans(2, distance='euclidean')
    # Train the KMeans
    kms.fit(database, verbose=True)
    # Build the clusters using the trained patterns
    clusters = kms.build(database, cluster=True)    # Cluster class objects
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = kms.winner(database[15])

def test_KMeans_1d():
    """ Lloyd's KMeans clustering """
    #--- Test on 1D row vector
    # Generate a dummy database
    array = np.arange(100.)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KMeans` object
    kms = KMeans(2, distance='euclidean')
    # Train the KMeans
    kms.fit(database, verbose=False)
    # Build the clusters using the trained patterns
    clusters = kms.build(database, cluster=True)    # Cluster class objects
    print(type(clusters[0]))
    # Find the BMU of a single data
    bmu = kms.winner(database[15])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = kms.winner(database[10:15])
    print(bmu)
    #--- Test on 1D column vector
    # Generate a dummy database
    array = np.arange(100.).reshape(-1, 1)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KMeans` object
    kms = KMeans(2, distance='euclidean')
    # Train the KMeans
    kms.fit(database, verbose=False)
    # Build the clusters using the trained patterns
    clusters = kms.build(database, cluster=True)    # Cluster class objects
    print(type(clusters[0]))
    # Find the BMU of a single data
    bmu = kms.winner(database[15])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = kms.winner(database[10:15])
    print(bmu)

def test_KMeans_init():
    """ Instantiate a KMeans object (constructor) """
    kms = KMeans()
    kms = KMeans(2, 123, 'euclidean')

def test_KMeans_fit():
    """ Train the KMeans on the data """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Instantiate a `KMeans` object
    kms = KMeans(2)
    # Train the KMeans
    kms.fit(database, verbose=True)
    print(kms.patterns.round(3))

def test_KMeans_build():
    """ Cluster input data according to the current patterns """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Instantiate a `KMeans` object
    kms = KMeans(2)
    # Train the KMeans
    kms.fit(database, verbose=True)
    # Build the clusters using the trained patterns
    clusters = kms.build(database, cluster=True)

def test_KMeans_winner():
    """ Find the Best Matching Unit for a data """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Instantiate a `KMeans` object
    kms = KMeans(2)
    # Train the KMeans
    kms.fit(database, verbose=True)
    # Build the clusters from the trained patterns
    kms.build(database, cluster=True)
    # Find the BMU of a single data
    bmu = kms.winner(database[15])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = kms.winner(database[10:15])
    print(bmu)

def test_kmeans_params():
    """ Check the parameters for the KMeans """
    # Get the default parameters
    print(kmeans_params())
    # Change only the number of clusters and leave the other default params
    print(kmeans_params(nb_clusters=4))
    # Change all the parameters
    print(kmeans_params(
        nb_clusters=4, cluster=False, margins=0.1, tmax=50,
        seed=123, verbose=True, distance='manhattan'))

def test_kmeans_cluster():
    """ Instantiate & train the K-Means on a dataset and cluster it """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Cluster the database with the KMeans with default parameters
    clusters = kmeans_cluster(database)
    # Cluster the database with the KMeans with some specific parameters
    clusters = kmeans_cluster(database, nb_clusters=5, verbose=True)



# Launch test/example functions
test_KMeans()

test_KMeans_1d()

test_KMeans_init()

test_KMeans_fit()

test_KMeans_build()

test_KMeans_winner()

test_kmeans_params()

test_kmeans_cluster()

