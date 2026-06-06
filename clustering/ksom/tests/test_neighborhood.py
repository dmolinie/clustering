import numpy as np
from clustering.ksom._neighborhood import (
    _neighborhood_1st, _neighborhood_2nd, _neighborhood)


def test_neighborhood_1st():
    """ First order neighborhood """
    # Grid dimensions
    height, width, size = 3, 3, 9
    # Generate a grid for visualization
    grid = np.arange(size).reshape(height, width)
    print('\n', grid, '\n')
    # Find the 1st order neighbors for any node    
    for i in range(size):
        print(i, _neighborhood_1st(i, (height, width, size)))

def test_neighborhood_2nd():
    """ Second order neighborhood """
    # Grid dimensions
    height, width, size = 4, 5, 20
    # Generate a grid for visualization
    grid = np.arange(size).reshape(height, width)
    print('\n', grid, '\n')
    # Find the 2nd order neighbors for any node    
    for i in range(size):
        print(i, _neighborhood_2nd(i, (height, width, size)))

def test_neighborhood():
    """ Any order neighborhood """
    # Grid dimensions
    height, width, size = 5, 5, 25
    # Generate a grid for visualization
    grid = np.arange(size).reshape(height, width)
    print('\n', grid, '\n')
    # Find the 1st and 2nd order neighbors for any node    
    for i in range(size):
        a = _neighborhood_1st(i, (height, width, size))
        b = _neighborhood(i, (height, width, size), 1)
        print(a == b, end=' ')
        a = _neighborhood_2nd(i, (height, width, size))
        b = _neighborhood(i, (height, width, size), 2)
        print(a == b, end =' ')
    print()



# Launch test/example functions
test_neighborhood_1st()

test_neighborhood_2nd()

test_neighborhood()

