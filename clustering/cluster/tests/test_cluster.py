import numpy as np
from clustering.formats._database import Database, Cluster
from clustering.ksom import ksom_cluster
from clustering.cluster._cluster import *


def test_prune():
    """ Remove the empty clusters from a list """
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(4, 4))
    print(len(clusters), '--', [len(clt) for clt in clusters])
    # Prune the empty clusters
    prune(clusters)
    print(len(clusters), '--', [len(clt) for clt in clusters])

def test_merge():
    """ Merge the closest clusters to refine splitting """
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(3, 3))
    print(len(clusters), '--', [len(clt) for clt in clusters])
    # Merge the closest clusters
    prune(clusters)
    merge(clusters, 0.01)
    print(len(clusters), '--', [len(clt) for clt in clusters])

def test_merge_1d():
    """ Merge the closest clusters to refine splitting """
    #--- Test 1D row array
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.), np.arange(100))
    clusters = ksom_cluster(database, grid_size=(3, 3))
    print(len(clusters), '--', [len(clt) for clt in clusters])
    # Merge the closest clusters
    prune(clusters)
    merge(clusters, 0.01)
    print(len(clusters), '--', [len(clt) for clt in clusters])
    #--- Test 1D column array
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 1), np.arange(100))
    clusters = ksom_cluster(database, grid_size=(3, 3))
    print(len(clusters), '--', [len(clt) for clt in clusters])
    # Merge the closest clusters
    prune(clusters)
    merge(clusters, 0.01)
    print(len(clusters), '--', [len(clt) for clt in clusters])

def test_sort():
    """ Sort the clusters of a list by size """
    # Generate a dummy database & cluster it
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    clusters = ksom_cluster(database, grid_size=(2, 2))
    print(len(clusters), '--', [len(clt) for clt in clusters])
    # Sort the clusters by size
    clusters = sort(clusters)
    print(len(clusters), '--', [len(clt) for clt in clusters])

def test_get_clust_func():
    """ Get the reference to a specified distance function """
    # KMeans clustering
    print(get_clust_func('kmeans'))
    # Kohonen SOMs
    print(get_clust_func('ksom'))
    # SPRADA method
    print(get_clust_func('sprada'))

def test_cluster():
    """ Split a a database using a given clustering methods """
    # Generate a dummy database
    database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    # Cluster the database with the KMeans
    kparams = {
        'nb_clusters': 2, 'cluster': True,
        'margins': 0.01, 'tmax': 100, 'seed': None,
        'verbose': True, 'distance': 'euclidean'}
    clusters = cluster(database, 'kmeans', **kparams)
    # Cluster the database with the KSOM
    kparams = {
        'grid_size': (3, 3), 'cluster': True, 'grid': False,
        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        'distance': 'euclidean'}
    clusters = cluster(database, 'ksom', **kparams)

def test_rebuild_idx():
    """ Rebuild the training and testing indexes """
    # Generate dummy data and wrap them into a set of Clusters
    data = np.linspace(0, 100, 10000).reshape(1000, 10)
    tstp = np.linspace(0, 10, 1000)
    # Wrap the data into a Database
    database = Database(data, tstp)
    # Create dummy clusters on the 80% of the database
    clusters = [Cluster(data[i*100:(i+1)*100], tstp[i*100:(i+1)*100])
                for i in range(8)]
    # Rebuild the indexes (with an offset of 5)
    offset = 5
    outrange = len(database) - offset
    ind_trn_lst, ind_trn, ind_gen = rebuild_idx(database, clusters, outrange)
    print(ind_trn.shape, ind_gen.shape)
    # Directly use the indexes as np.ndarrays
    ind_trn_lst, ind_trn, ind_gen =\
        rebuild_idx(database.index, [clt.index for clt in clusters], outrange)
    print(ind_trn.shape, ind_gen.shape)



# Launch test/example functions
test_prune()

test_merge()

test_merge_1d()

test_sort()

test_get_clust_func()

test_cluster()

test_rebuild_idx()

