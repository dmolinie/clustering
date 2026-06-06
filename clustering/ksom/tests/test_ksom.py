import numpy as np
from clustering.formats._database import Database
from clustering.ksom._ksom import *


def test_KohonenSOM():
    """ Kohonen's Self-Organizing Map (KSOM) """
    #--- Use the K-SOM on a NumPy array
    # Generate a dummy database
    array = np.arange(100.).reshape(-1, 5)
    # Instantiate a `KohonenSOM` object
    ksom = KohonenSOM((3, 3), distance='euclidean')
    # Train the KSOM
    ksom.fit(array, verbose=True)
    # Build the clusters from the trained patterns
    clusters = ksom.build(array, cluster=False)     # NumPy arrays
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = ksom.winner(array[15])
    print(bmu)
    #--- Use the K-SOM on a Database and build Clusters
    # Generate a dummy database
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KohonenSOM` object
    ksom = KohonenSOM((3, 3), distance='euclidean')
    # Train the KSOM
    ksom.fit(database, verbose=True)
    # Build the clusters from the trained patterns
    clusters = ksom.build(database, cluster=True)   # Cluster class objects
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = ksom.winner(database[15])
    print(bmu)

def test_KohonenSOM_1d():
    """ Kohonen's Self-Organizing Map (KSOM) """
    #--- Test on 1D row vector
    # Generate a dummy database
    array = np.arange(100.)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KohonenSOM` object
    ksom = KohonenSOM((3, 3), distance='euclidean')
    # Train the KSOM
    ksom.fit(database, verbose=False)
    # Build the clusters from the trained patterns
    clusters = ksom.build(database, cluster=True)
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = ksom.winner(database[15])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = ksom.winner(database[10:15])
    print(bmu)
    #--- Test on 1D column vector
    # Generate a dummy database
    array = np.arange(100.).reshape(-1, 1)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `KohonenSOM` object
    ksom = KohonenSOM((3, 3), distance='euclidean')
    # Train the KSOM
    ksom.fit(database, verbose=False)
    # Build the clusters from the trained patterns
    clusters = ksom.build(database, cluster=True)
    # Find the BMU of a single data
    bmu = ksom.winner(database[15])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = ksom.winner(database[10:15])
    print(bmu)

def test_KohonenSOM_init():
    """ Instantiate a KohonenSOM object (constructor) """
    kms = KohonenSOM()
    kms = KohonenSOM((3, 3), 123, 'euclidean')

def test_KohonenSOM_fit():
    """ Reset the patterns and train the KSOM on the provided data """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Instantiate a `KohonenSOM` object
    ksom = KohonenSOM((2, 2))
    # Train the KSOM
    ksom.fit(database, verbose=True)
    print(ksom.patterns.round(3))
    # Train a set of KSOMs and average them as one
    ksom.fit(database, verbose=True)
    print(ksom.patterns.round(3))

def test_KohonenSOM_build():
    """ Build the KSOM's neural grid with the current patterns """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Instantiate a `KohonenSOM` object
    ksom = KohonenSOM((3, 3))
    # Train the KSOM
    ksom.fit(database, verbose=True)
    # Build the clusters from the trained patterns
    ksom.build(database, cluster=True)

def test_KohonenSOM_winner():
    """ Find the Best Matching Unit for a data """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Instantiate a `KohonenSOM` object
    ksom = KohonenSOM((3, 3))
    # Train the KSOM
    ksom.fit(database, verbose=True)
    # Build the clusters from the trained patterns
    ksom.build(database, cluster=True)
    # Find the BMU of a single data
    bmu = ksom.winner(database[15])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = ksom.winner(database[10:15])
    print(bmu)

def test_ksom_params():
    """ Check the parameters for the Kohonen's SOM """
    # Get the default parameters
    print(ksom_params())
    # Change only the number of clusters and leave the other default params
    print(ksom_params(grid_size=(2, 2)))
    # Change all the parameters
    print(ksom_params(
        grid_size=(3, 3), cluster=True, grid=False, margins=0.01,
        tmax=1000, nbh_rate_0=1.00, lrn_rate_0=0.95,
        seed=123, verbose=False, distance='euclidean'))

def test_ksom_cluster():
    """ Instantiate & train a KSOM on a dataset and cluster it """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Cluster the database with the KSOM with default parameters
    clusters = ksom_cluster(database)
    # Cluster the database with the KSOM with some specific parameters
    clusters = ksom_cluster(database, grid_size=(2, 2), verbose=True)



# Launch test/example functions
test_KohonenSOM()

test_KohonenSOM_1d()

test_KohonenSOM_init()

test_KohonenSOM_fit()

test_KohonenSOM_build()

test_KohonenSOM_winner()

test_ksom_params()

test_ksom_cluster()

