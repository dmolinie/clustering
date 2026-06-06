import numpy as np
from clustering.metrics._scaling import *


def test_extrema():
    """ Minimal and maximal values of a dataset """
    print(extrema(np.arange(50., dtype=float).reshape(-1, 5)))

def test_check_scale():
    """ Check if a set of data is scaled between `bounds` """
    # Generate dummy data
    data = np.arange(100, dtype=float).reshape(-1, 5)
    #--- Scalar bounds
    # Check if the data are already scaled between 0 and 1
    scale, limits = check_scale(data, (0., 1.))
    print(scale)
    print(limits)
    # Rescale the data with the `rescale` function using the returned `limits`
    data, limits = rescale(data, bounds=(0., 1.), limits=limits)
    print(data.min(0), data.max(0))
    # Check again if the data are scaled between 0 and 1
    scale, limits = check_scale(data, (0., 1.))
    print(scale)
    #--- Vector bounds
    # Rescale the data between 0 and 1
    data = np.linspace(0., 1., 100).reshape(-1, 5)
    # Check if the data are already scaled between 0 and 1
    scale, limits = check_scale(data, (0., 1.))
    print(scale)     # False, as all the dimensions are not 0. and 1.
    print(limits)
    # Check if the data are scaled with vector bounds
    scale, limits = check_scale(data, limits)
    print(scale)
    print(limits)

def test_rescale():
    """ Rescale a dataset """
    # Generate dummy data
    data = np.arange(50., dtype=float).reshape(-1, 5)
    # Normalize the data
    data_norm, limits = rescale(data, bounds=(0., 1.))
    # Denormalize the data
    data_denorm, limits = rescale(data_norm, bounds=limits, limits=(0., 1.))
    print(extrema(data) == extrema(data_denorm))

def test_scale_1d():
    """ Check 1D data """
    #--- Test 1D row array
    arr = np.arange(10., dtype=float)
    print(extrema(arr))
    print(check_scale(arr, (0., 1.)))
    print(rescale(arr))
    #--- Test 1D column array
    arr = np.arange(10., dtype=float).reshape(-1, 1)
    print(extrema(arr))
    print(check_scale(arr, (0., 1.)))
    print(rescale(arr))



# Launch test/example functions
test_extrema()

test_check_scale()

test_rescale()

test_scale_1d()

