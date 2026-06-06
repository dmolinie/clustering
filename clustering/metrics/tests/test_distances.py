import numpy as np
from clustering.metrics._distances import *


def test_manhattan():
    """ Manhattan distance """
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances for 1D data
    print(manhattan_1d(arr, arr+1., 0))
    print(manhattan_1d(arr, arr+1., 1))
    # Compute the distances for ND data
    print(manhattan_nd(arr, arr+1., 0))
    print(manhattan_nd(arr, mat, 0))
    print(manhattan_nd(arr, mat, 1))

def test_euclidean():
    """ Euclidean distance """
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances for 1D data
    print(euclidean_1d(arr, arr+1., 0))
    print(euclidean_1d(arr, arr+1., 1))
    # Compute the distances for ND data
    print(euclidean_nd(arr, arr+1., 0))
    print(euclidean_nd(arr, mat, 0))
    print(euclidean_nd(arr, mat, 1))

def test_minkowski():
    """ Minkowski distance """
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances for 1D data
    print(minkowski_1d(arr, arr+1., 0))
    print(minkowski_1d(arr, arr+1., 1))
    # Compute the distances for ND data
    print(minkowski_nd(arr, arr+1., 0))
    print(minkowski_nd(arr, mat, 0))
    print(minkowski_nd(arr, mat, 1))

def test_canberra():
    """ Canberra distance """
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances for 1D data
    print(canberra_1d(arr, arr+1., 0))
    print(canberra_1d(arr, arr+1., 1))
    # Compute the distances for ND data
    print(canberra_nd(arr, arr+1., 0))
    print(canberra_nd(arr, mat, 0))
    print(canberra_nd(arr, mat, 1))

def test_test_cosine():
    """ Cosine distance """
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances for 1D data
    print(cosine_1d(arr, arr+1., 0))
    print(cosine_1d(arr, arr+1., 1))
    # Compute the distances for ND data
    print(cosine_nd(arr, arr+1., 0))
    print(cosine_nd(arr, mat, 0))
    print(cosine_nd(arr, mat, 1))

def test_tanimoto():
    """ Tanimoto distance """
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances for 1D data
    print(tanimoto_1d(arr, arr+1., 0))
    print(tanimoto_1d(arr, arr+1., 1))
    # Compute the distances for ND data
    print(tanimoto_nd(arr, arr+1., 0))
    print(tanimoto_nd(arr, mat, 0))
    print(tanimoto_nd(arr, mat, 1))

def test_czekanowski():
    """ Czekanowski distance """
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances for 1D data
    print(czekanowski_1d(arr, arr+1., 0))
    print(czekanowski_1d(arr, arr+1., 1))
    # Compute the distances for ND data
    print(czekanowski_nd(arr, arr+1., 0))
    print(czekanowski_nd(arr, mat, 0))
    print(czekanowski_nd(arr, mat, 1))

def test_get_dist_func():
    """ Get the reference to a specified distance function """
    # Generate dummy data
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Get the function distance for 1D data
    fdist = get_dist_func('euclidean', 1)
    # Compute the distances for 1D data
    print(fdist(arr, arr+1., 0))
    print(fdist(arr, arr+1., 1))
    # Get the function distance for ND data
    fdist = get_dist_func('euclidean', 2)
    # Compute the distances for ND data
    print(fdist(arr, arr+1., 0))
    print(fdist(arr, mat, 0))
    print(fdist(arr, mat, 1))

def test_hausdorff():
    """ Modified Hausdorff Distance between two curves (2D vectors) """
    crv1 = np.arange(10., 50., dtype=float).reshape(-1, 5)
    crv2 = np.arange(20., 80., dtype=float).reshape(-1, 5)
    print(hausdorff(crv1, crv2))



# Launch test/example functions
test_manhattan()

test_euclidean()

test_minkowski()

test_canberra()

test_test_cosine()

test_tanimoto()

test_czekanowski()

test_get_dist_func()

test_hausdorff()

