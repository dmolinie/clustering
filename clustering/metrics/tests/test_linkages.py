import numpy as np
from clustering.metrics._linkages import *


def test_linkage_single():
    """ Single linkage aggregation metric """
    # Generate dummy data
    clt1 = np.arange(10, 40, dtype=float).reshape(-1, 5)
    clt2 = np.arange(20, 80, dtype=float).reshape(-1, 5)
    # Compute the linkage distance
    print(linkage_single(clt1, clt2, 'euclidean'))

def test_linkage_complete():
    """ Complete linkage aggregation metric """
    # Generate dummy data
    clt1 = np.arange(10, 40, dtype=float).reshape(-1, 5)
    clt2 = np.arange(20, 80, dtype=float).reshape(-1, 5)
    # Compute the linkage distance
    print(linkage_complete(clt1, clt2, 'euclidean'))

def test_linkage_mean():
    """ Mean linkage aggregation metric """
    # Generate dummy data
    clt1 = np.arange(10, 40, dtype=float).reshape(-1, 5)
    clt2 = np.arange(20, 80, dtype=float).reshape(-1, 5)
    # Compute the linkage distance
    print(linkage_mean(clt1, clt2, 'euclidean'))

def test_linkage_ward():
    """ Ward linkage aggregation metric """
    # Generate dummy data
    clt1 = np.arange(10, 40, dtype=float).reshape(-1, 5)
    clt2 = np.arange(20, 80, dtype=float).reshape(-1, 5)
    # Compute the linkage distance
    print(linkage_ward(clt1, clt2, 'euclidean'))

def test_get_link_func():
    """ Get the reference to a specified linkage function """
    # Get the linkage function
    flink = get_link_func('ward')
    # Compute the linkage between 1D arrays
    arr = np.arange(50., dtype=float)
    print(flink(arr, arr+1.))
    # Compute the linkage between ND arrays
    mat = arr.reshape(-1, 5)
    print(flink(mat, mat+1.))

def test_linkages():
    """ Linkage distance between any possible 2-cluster set """
    # Generate a dummy list of clusters
    arr = np.arange(50., dtype=float)
    arrs = [arr+i for i in range(4)]
    # Compute the linkages between every cluster of the list
    print(linkages(arrs, 'ward', 'euclidean'))     # 4*3/2=6 values

def test_linkages_1d():
    """ Check 1D data """
    #--- Test 1D row array
    clt1 = np.arange(10, 40, dtype=float)
    clt2 = np.arange(20, 80, dtype=float)
    print(linkage_single(clt1, clt2))
    print(linkage_complete(clt1, clt2))
    print(linkage_mean(clt1, clt2))
    print(linkage_ward(clt1, clt2))
    #--- Test 1D column array
    clt1 = np.arange(10, 40, dtype=float).reshape(-1, 1)
    clt2 = np.arange(20, 80, dtype=float).reshape(-1, 1)
    print(linkage_single(clt1, clt2))
    print(linkage_complete(clt1, clt2))
    print(linkage_mean(clt1, clt2))
    print(linkage_ward(clt1, clt2))



# Launch test/example functions
test_linkage_single()

test_linkage_complete()

test_linkage_mean()

test_linkage_ward()

test_get_link_func()

test_linkages()

test_linkages_1d()

