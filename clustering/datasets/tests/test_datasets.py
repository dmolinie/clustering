import numpy as np
from clustering.datasets._datasets import *


def test_nested_circles():
    """ Generate two 2D nested circles """
    data = nested_circles(500, 3, 0., 1., show=False)
    print(type(data))
    print(data.shape)

def test_strips():
    """ Generate three 2D vertical strips """
    data = strips(1500, 0.175, show=False)
    print(type(data))
    print(data.shape)

def test_gaussian():
    """ Generate a database (Data) with a Gaussian distribution """
    data = gaussian(250, 5, (0, 100), 0.5, 12)
    print(type(data))
    print(data.shape)

def test_gauss_3d():
    """ Generate a set of 3D Gaussian distributions """
    data = gauss_3d(250, 12, (0, 100), 0.5, show=False)
    print(type(data))
    print(data.shape)



# Launch test/example functions
test_nested_circles()

test_strips()

test_gaussian()

test_gauss_3d()

