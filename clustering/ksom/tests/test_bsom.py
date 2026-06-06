import numpy as np
from clustering.formats._database import Database
from clustering.ksom._bsom import *


def test_BiLevelSOM():
    """ Bi-Level Self-Organizing Maps (BSOM) """
    #--- Use the B-SOM on a NumPy array
    # Generate a dummy database
    array = np.arange(100.).reshape(-1, 5)
    # Instantiate a `BiLevelSOM` object
    bsom = BiLevelSOM((3, 3), distance='euclidean')
    # Train the BSOM
    bsom.fit(array, nb_grids=3, verbose=True)
    # Build the clusters from the trained patterns
    clusters = bsom.build(array, cluster=False)     # NumPy arrays
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = bsom.winner(array[15])
    print(bmu)
    #--- Use the B-SOM on a Database and build Clusters
    # Generate a dummy database
    database = Database(array, np.arange(len(array)))
    # Instantiate a `BiLevelSOM` object
    bsom = BiLevelSOM((3, 3), distance='euclidean')
    # Train the BSOM
    bsom.fit(database, nb_grids=3, verbose=True)
    # Build the clusters from the trained patterns
    clusters = bsom.build(database, cluster=True)   # Cluster class objects
    print(type(clusters[0]))
    # Find the BMU of a data
    bmu = bsom.winner(database[15])
    print(bmu)

def test_BiLevelSOM_1d():
    """ Bi-Level Self-Organizing Maps (BSOM) """
    #--- Test on 1D rows vector
    # Generate a dummy database
    array = np.arange(100.)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `BiLevelSOM` object
    bsom = BiLevelSOM((3, 3), distance='euclidean')
    # Train the BSOM
    bsom.fit(database, nb_grids=3, verbose=False)
    # Build the clusters from the trained patterns
    clusters = bsom.build(array, cluster=True)
    print(type(clusters[0]))
    # Find the BMU of a single data
    bmu = bsom.winner(database[15])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = bsom.winner(database[10:15])
    print(bmu)
    #--- Test on 1D column vector
    # Generate a dummy database
    array = np.arange(100.).reshape(-1, 1)
    database = Database(array, np.arange(len(array)))
    # Instantiate a `BiLevelSOM` object
    bsom = BiLevelSOM((3, 3), distance='euclidean')
    # Train the BSOM
    bsom.fit(array, nb_grids=3, verbose=False)
    # Build the clusters from the trained patterns
    clusters = bsom.build(array, cluster=True)
    print(type(clusters[0]))
    # Find the BMU of a single data
    bmu = bsom.winner(database[15])
    print(bmu)
    # Find the BMUs of a set of data
    bmu = bsom.winner(database[10:15])
    print(bmu)

def test_BiLevelSOM_fit():
    """ Reset the patterns and train the BSOM on the provided data """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Instantiate a `BiLevelSOM` object
    bsom = BiLevelSOM((2, 2))
    # Train a set of KSOMs and average them as one BSOM
    bsom.fit(database, nb_grids=5, verbose=True)
    print(bsom.patterns.round(3))

def test_bsom_params():
    """ Check the parameters for the BSOM """
    # Get the default parameters
    print(bsom_params())
    # Change only the number of clusters and leave the other default params
    print(bsom_params(grid_size=(2, 2)))
    # Change all the parameters
    print(bsom_params(
        grid_size=(3, 3), nb_grids=5, cluster=True,
        grid=False, margins=0.01, tmax=100,
        nbh_rate_0=1.00, lrn_rate_0=0.95,
        seed=123, verbose=False, distance='euclidean'))

def test_bsom_cluster():
    """ Instantiate & train a BSOM on a dataset and cluster it  """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Cluster the database with the BSOM with default parameters
    clusters = bsom_cluster(database, nb_grids=3, verbose=True)
    # Cluster the database with the BSOM with some specific parameters
    clusters = bsom_cluster(database, grid_size=(2, 2), nb_grids=10)



# Launch test/example functions
test_BiLevelSOM()

test_BiLevelSOM_1d()

test_BiLevelSOM_fit()

test_bsom_params()

test_bsom_cluster()

