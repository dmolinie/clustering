import numpy as np
from clustering.metrics._kernels import *


def test_linear():
    """ Linear kernel """
    # Generate dummy data
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances
    print(linear_1d(arr, arr+1., 0))
    print(linear_1d(arr, arr+1., 1))
    # Compute the distances
    print(linear_nd(arr, arr+1., 0))
    print(linear_nd(arr, mat, 0))
    print(linear_nd(arr, mat, 1))

def test_circles():
    """ Circle kernel """
    # Generate dummy data
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances
    print(circles_1d(arr, arr+1., 0))
    print(circles_1d(arr, arr+1., 1))
    # Compute the distances
    print(circles_nd(arr, arr+1., 0))
    print(circles_nd(arr, mat, 0))
    print(circles_nd(arr, mat, 1))

def test_gaussian():
    """ Gaussian kernel """
    # Generate dummy data
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances
    print(gaussian_1d(arr, arr+1., 0))
    print(gaussian_1d(arr, arr+1., 1))
    # Compute the distances
    print(gaussian_nd(arr, arr+1., 0))
    print(gaussian_nd(arr, mat, 0))
    print(gaussian_nd(arr, mat, 1))

def test_polynomial():
    """ Polynomial kernel """
    # Generate dummy data
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the distances
    print(polynomial_1d(arr, arr+1., 0))
    print(polynomial_1d(arr, arr+1., 1))
    # Compute the distances
    print(polynomial_nd(arr, arr+1., 0))
    print(polynomial_nd(arr, mat, 0))
    print(polynomial_nd(arr, mat, 1))

def test_get_ker_func():
    """ Get the reference to a specified kernel function """
    # Generate dummy data
    arr = np.arange(5., dtype=float)
    mat = np.arange(15., dtype=float).reshape(-1, 5)
    # Get the function distance for 1D data
    fker = get_ker_func('linear', 1)
    # Compute the distances
    print(fker(arr, arr+1., 0))
    print(fker(arr, arr+1., 1))
    # Get the function distance for 2D data
    fker = get_ker_func('linear', 2)
    # Compute the distances
    print(fker(arr, arr+1., 0))
    print(fker(arr, mat, 0))
    print(fker(arr, mat, 1))



# Launch test/example functions
test_linear()

test_circles()

test_gaussian()

test_polynomial()

test_get_ker_func()

