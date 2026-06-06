import numpy as np
from clustering.formats._database import Cluster
from clustering.formats._neuralgrid import *


def test_NeuralGrid():
    """ Neural Grid for the Kohonen Self-Organizing Maps """
    # Generate a dummy set of clusters
    value = np.arange(300).reshape(-1, 3)
    index = np.arange(100)
    nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
             for i in range(9)]
    # Wrap the nodes clusters into a NeuralGrid
    grid = NeuralGrid((3, 3), nodes, 'euclidean')
    # Search the BMU of a data
    data = nodes[5][8]
    print(grid.find_node(data))
    # Change the BMU for a new cluster
    grid.set_node((1, 2), value[90:100], index[90:100])
    print(grid.find_node(data))

def test_NeuralGrid_init():
    """ Instantiate a NeuralGrid object (constructor) """
    # Generate a dummy set of clusters
    value = np.arange(90).reshape(-1, 3)
    index = np.arange(30)
    nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
             for i in range(3)]
    # Wrap the nodes clusters into a NeuralGrid
    grid = NeuralGrid((3, 1), nodes)
    # Generate a dummy set of clusters
    value = np.arange(270).reshape(-1, 3)
    index = np.arange(90)
    nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
             for i in range(9)]
    # Wrap the nodes clusters into a NeuralGrid
    grid = NeuralGrid((3, 3), nodes)

def test_NeuralGrid_set_node():
    """ Set the grid's node at position `pos` """
    # Generate a dummy set of clusters
    value = np.arange(120).reshape(-1, 3)
    index = np.arange(40)
    nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
             for i in range(3)]
    # Wrap the nodes clusters into a NeuralGrid
    grid = NeuralGrid((3, 1), nodes)
    # Update the node (1, 0) of the grid
    clt = Cluster(value[30:40], index[30:40])
    # By instantiating a new cluster
    grid.set_node((1, 0), clt.value, clt.index, clt.pattern)
    # Or by directly setting a cluster to it
    grid[1, 0] = clt

def test_NeuralGrid_find_node():
    """ Identify the Best Matching Unit """
    # Generate a dummy set of clusters
    value = np.arange(90).reshape(-1, 3)
    index = np.arange(30)
    nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
             for i in range(3)]
    # Wrap the nodes clusters into a NeuralGrid
    grid = NeuralGrid((3, 1), nodes)
    # Search the BMU of a data
    data = nodes[1][5]
    print(grid.find_node(data))

def test_NeuralGrid_tolist():
    """ Flatten the nodes of the grid to a list of Clusters """
    # Generate a dummy set of clusters
    value = np.arange(90).reshape(-1, 3)
    index = np.arange(30)
    nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
             for i in range(3)]
    # Wrap the nodes clusters into a NeuralGrid
    grid = NeuralGrid((3, 1), nodes)
    # Search the BMU of a data
    nodes2 = grid.tolist()



# Launch test/example functions
test_NeuralGrid()

test_NeuralGrid_init()

test_NeuralGrid_set_node()

test_NeuralGrid_find_node()

test_NeuralGrid_tolist()

