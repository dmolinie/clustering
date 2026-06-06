import numpy as np
from clustering.formats._database import Database
from clustering.ksom import ksom_cluster
from clustering.cluster._save_clt import *


def test_similarity():
    """ Compute the similarities of the features of two vectors """
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(2, 2))
    # Compute the similarity between the patterns of the clusters
    patterns = np.array([clt.pattern for clt in clusters])
    print(similarity(patterns, 0.5))

def test_save_pats():
    """ Save the patterns and the tags in a CSV file """
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(2, 2))
    # Save the patterns in a CSV file
    patterns = np.array([clt.pattern for clt in clusters])
    save_pats(patterns, ['C1', 'C2', 'C3', 'C4', 'C5'])

def test_save_sim():
    """ Compare the features of any couple of vectors and save them as CSV """
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(2, 2))
    # Compute & save the similarity between the patterns of the clusters
    patterns = np.array([clt.pattern for clt in clusters])
    save_sim(patterns, ['C1', 'C2', 'C3', 'C4', 'C5'])

def test_save_dist():
    """ Compute and save in a CSV file the euclidean distance between
    any couple of patterns """
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(2, 2))
    # Compute & save the similarity between the patterns of the clusters
    patterns = np.array([clt.pattern for clt in clusters])
    save_dist(patterns, 'manhattan')

def test_save_patterns():
    """ Save the patterns of the Node objects and some related stats """
    # Generate a dummy database & cluster it
    database = Database(
        np.arange(100.).reshape(-1, 2), np.arange(50), ['C1', 'C2'])
    clusters = ksom_cluster(database, grid_size=(2, 2))
    clusters = [clt for clt in clusters if len(clt) != 0]
    # Save the patterns and some stats about them in a file
    save_patterns(clusters)

def test_save_clusters():
    """ Save a list of clusters as unique CSV files """
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(2, 2))
    # Save the clusters in a file
    save_clusters(clusters)

def test_save_statistics():
    """ Write some statistics of the clusters in a CSV file """
    import clustering.metrics as mts
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(2, 2))
    # Compute some characterizing metrics about the clusters
    quants = mts.quantifiers(clusters)
    # Save the clusters in a file
    save_statistics(['Sils', 'HyDen', 'AvStd'], quants)



# Launch test/example functions
test_similarity()

test_save_pats()

test_save_sim()

test_save_dist()

test_save_patterns()

test_save_clusters()

test_save_statistics()

