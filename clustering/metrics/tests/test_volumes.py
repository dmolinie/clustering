import numpy as np
from clustering.metrics._volumes import *


def test_test_hypercube():
    """ Hypercube's volume and surface """
    # Volume of a 3D cube of side 5
    print(hypercube(5.0, 3))
    # Surface of a 3D cube of side 5
    print(hypercube(5.0, 3, False))

def test_test_hypersphere():
    """ Hypersphere's volume and surface """
    # Volume of a 3D sphere of radius 5
    print(hypersphere(5.0, 3))
    # Surface of a 3D sphere of radius 5
    print(hypersphere(5.0, 3, False))

def test_get_vol_func():
    """ Get the reference to a specified hyper-volume function """
    fvol = get_vol_func('hypersphere')
    # Compute the distances
    print(fvol(2.5, 3, False))      # Surface of a 3D sphere
    print(fvol(2.5, 3, True))       # Volume of a 3D sphere



# Launch test/example functions
test_test_hypercube()

test_test_hypersphere()

test_get_vol_func()

