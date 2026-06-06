""" Neural Grid class for the Kohonen Self-Organizing Maps

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: February 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['NeuralGrid']

from numbers import Number

import numpy as np

import clustering.metrics as mts
from clustering.formats._database import Cluster

_TYPES = "must be list, tuple or array"


##############################################################################
##                     Neural Grid (SOM's final output)                     ##
##############################################################################

class NeuralGrid():
    """ Neural Grid for the Kohonen Self-Organizing Maps

    Its attribute is a 2D matrix, each element being a Cluster object,
    referred to as a neuron or node, and consisting of a prototype of
    n components, with n the dimension of the feature space, and a list
    of the classified data in this space.

    Constructor
    -----------
    __init__(size, clusters, distance='euclidean')

    Magic Methods
    -------------
    __getitem__(pos)
        Get the node at `pos`.
    __setitem__(pos, cluster)
        Set the node at `pos`.
    __repr__
        Display the NeuralGrid.
    __str__
        Print the NeuralGrid.

    Attributes
    ----------
    grid : 2D list of np.ndarray or Clusters, getter & setter
        The SOM's grid.
    size : 2-tuple of ints, getter & setter
        The SOM's grid dimensions.
    distance : str, getter & setter
        The distance function in use.

    Methods
    -------
    set_node(pos, value, index, pattern=None,
             tags=None, classes=None, *, dtype=float)
        Set the grid's node (Cluster) at position `pos`.
    find_node(data)
        Identify the Best Matching Unit.
    tolist()
        Flatten the nodes of the grid to a list of Clusters.

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy set of clusters
    >>> value = np.arange(300).reshape(-1, 3)
    >>> index = np.arange(100)
    >>> nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
    ...          for i in range(9)]

    # Wrap the nodes clusters into a NeuralGrid
    >>> grid = NeuralGrid((3, 3), nodes)

    # Search the BMU of a data
    >>> data = nodes[5][8]
    >>> print(grid.find_node(data, 'euclidean'))
    [1 2]

    # Change the BMU for a new cluster
    >>> grid.set_node((1, 2), value[90:100], index[90:100])
    >>> print(grid.find_node(data, 'euclidean'))
    [2 0]
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, size, clusters, distance='euclidean'):
        """ Instantiate a NeuralGrid object (constructor)

        Parameters
        ----------
        size : 2-tuple of ints
            The size of the grid, organized as (rows, cols).
        clusters : tuple or list of Clusters
            The clusters to bind to the grid. The `i*cols+j`-th element
            is bounded to the `(i, j)` node of the grid.
        [OPT] distance : str
            The distance name; see the `get_dist_func` function from
            the `metrics` module for details.
                :Default: 'euclidean'

        Examples
        --------
        >>> import numpy as np

        # Simple 1-column grid
        >>> value = np.arange(90).reshape(-1, 3)
        >>> index = np.arange(30)
        >>> nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
        ...          for i in range(3)]

        # Wrap the nodes clusters into a NeuralGrid
        >>> grid = NeuralGrid((3, 1), nodes, 'euclidean')

        # 2D grid
        >>> value = np.arange(270).reshape(-1, 3)
        >>> index = np.arange(90)
        >>> nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
        ...          for i in range(9)]

        # Wrap the nodes clusters into a NeuralGrid
        >>> grid = NeuralGrid((3, 3), nodes)
        """

        # Get the distance function
        self._fdist = mts.get_dist_func(distance)
        self._dist = distance

        # Store the size of the grid
        self._size = size

        # Place the clusters into the grid
        self._grid = [[clusters[i*size[1]+j] for j in range(size[1])]
                      for i in range(size[0])]
    #------------------------------------------------------------------------#

    #--------------------------   Magic Methods   ---------------------------#
    def __getitem__(self, pos):
        """ Get the node at `pos` """
        if np.isscalar(pos):
            return self._grid[pos]
        return self._grid[pos[0]][pos[1]]

    def __setitem__(self, pos, cluster):
        """ Set `cluster` to the node located at `pos` in the grid """
        if np.isscalar(pos):
            if isinstance(cluster, Cluster):
                for i in range(self._size[1]):
                    self[pos, i] = cluster
            else:
                for i, clt in enumerate(cluster):
                    self[pos, i] = clt
        else:
            self._grid[pos[0]][pos[1]] = cluster

    def __repr__(self):
        """ Display the NeuralGrid """
        return self.__str__()

    def __str__(self):
        """ Print the NeuralGrid """
        msg = "\n[INFO ] Displaying the grid's content -- Beginning\n"
        msg += f"\t--> The grid has size {self._size[0]}x{self._size[1]} "
        msg += f"({self._size[0]*self._size[1]} nodes)\n"

        msg += "\n/*************************************/\n"
        for i in range(self._size[0]):
            for j in range(self._size[1]):
                msg += f"Cluster:\t {(i, j)}\n"
                msg += self[i][j].__str__()
                msg += "\n/*************************************/\n"
        msg +=  "\n[INFO ] Displaying the grid's content -- End\n"
        return msg
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def distance(self):
        """ Get the distance function in use """
        return self._dist

    @distance.setter
    def distance(self, distance):
        """ Set the distance function to use """
        self._fdist = mts.get_dist_func(distance)
        self._dist = distance

    @property
    def grid(self):
        """ Get the SOM's grid """
        return self._grid

    @grid.setter
    def grid(self, grid):
        """ Set the SOM's grid """
        self._grid = grid

    @property
    def size(self):
        """ Get the SOM's grid dimensions """
        return self._size

    @size.setter
    def size(self, size):
        """ Set the SOM's grid dimensions """
        self._size = size
    #------------------------------------------------------------------------#

    #------------------------   Set a Node in Grid   ------------------------#
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def set_node(self, pos, value, index, pattern=None,
                 tags=None, classes=None, *, dtype=float):
        """ Set the grid's node (Cluster) at `pos` location within the grid

        As the grid's nodes are `Cluster` objects, use the `set_cluster`
        method of the `Cluster` class to set the node; see this method
        for details.

        Parameters
        ----------
        pos : 2-tuple of ints
            The (i, j) location of the node to set.
        value : ND tuple or list, np.ndarray
            The data values.
        index : 1D tuple or list, np.ndarray
            The data (time)stamps; must have the same length as `value`.
        [OPT] pattern : 1D tuple or list, np.ndarray
            The cluster characteristic pattern (prototype, barycenter, etc.).
                :Default: None
        [OPT] tags : list of strings
            The data tags (column names); should have the same length as
            the number of columns (first dimension) of `value`. If string,
            wrap it into a list; if int, generate a list of that number
            of elements with name `Dim{i}`.
                :Default: None
        [OPT] classes : list, tuple or np.ndarray
            The a-priori classes to which belong any data; if provided,
            must have the same length as `value` and `index`.
                :Default: None

        Returns
        -------
        None : directly update the `grid` attribute.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy set of clusters
        >>> value = np.arange(120).reshape(-1, 3)
        >>> index = np.arange(40)
        >>> nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
        ...          for i in range(3)]

        # Wrap the nodes clusters into a NeuralGrid
        >>> grid = NeuralGrid((3, 1), nodes)

        # Update the node (1, 0) of the grid
        >>> clt = Cluster(value[30:40], index[30:40])

        # By instantiating a new cluster
        >>> grid.set_node((1, 0), clt.value, clt.index, clt.pattern)

        # Or by directly setting a cluster to it
        >>> grid[1, 0] = clt
        """
        self._grid[pos[0]][pos[1]].set_cluster(
            value, index, pattern, tags, classes, dtype=dtype)
    #------------------------------------------------------------------------#

    #----------------------   Best Macthing Unit BMU   ----------------------#
    def find_node(self, data):
        """ Identify the Best Matching Unit

        Identify the nearest prototype of an input data and return its
        position number. The distance between the data and any of the
        prototypes of the grid's nodes is computed and that with the
        lowest distance is identified as the Best Matching Unit (BMU);
        its position in the grid is a tuple of 2 integers, representing
        the line and the column numbers of the node within the grid.

        Parameters
        ----------
        data : Number, list, tuple or np.ndarray
            The data for which find the nearest node (pattern).

        Returns
        -------
        pos : 2-tuple of ints
            The (i, j) position of the BMU within the grid.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy set of clusters
        >>> value = np.arange(90).reshape(-1, 3)
        >>> index = np.arange(30)
        >>> nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
        ...          for i in range(3)]

        # Wrap the nodes clusters into a NeuralGrid
        >>> grid = NeuralGrid((3, 1), nodes)

        # Search the BMU of a data
        >>> data = nodes[1][5]
        >>> print(grid.find_node(data))
        [1 0]
        """

        # Check if the format of the input data is valid
        if not isinstance(data, (Number, list, tuple, np.ndarray)):
            raise TypeError(f"Wrong type `{type(data)}` for `data`: " + _TYPES)

        # Check the data type
        if np.shape(self._grid[0][0].pattern) != np.shape(data):
            raise AssertionError(
                "Data must have the same shape as the grid's ones")

        # Minimal distance initialization
        bmu = np.zeros(2, int)
        dist_min = self._fdist(self._grid[0][0].pattern, data)

        # Research of the Best Matching Unit BMU
        for i in range(self._size[0]):
            for j in range(self._size[1]):

                # Distance between the next node and the current data
                dist = self._fdist(self._grid[i][j].pattern, data)

                # Search for the minimal distance
                if dist < dist_min:
                    bmu[:] = [i, j]
                    dist_min = dist

        return bmu
    #------------------------------------------------------------------------#

    #-------------------------   Flatten to List   --------------------------#
    def tolist(self):
        """ Flatten the nodes of the grid to a list of Clusters

        Parameters
        ----------
        Nothing, use only `self`.

        Returns
        -------
        nodes : list of Cluster objects
            The flattened nodes; the node at position `(i, j)` in the
            grid is the `i*cols+j`-th element of the returned list, with
            `cols` the width of the grid.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy set of clusters
        >>> value = np.arange(270).reshape(-1, 3)
        >>> index = np.arange(90)
        >>> nodes = [Cluster(value[i*10:(i+1)*10], index[i*10:(i+1)*10])
        ...          for i in range(9)]

        # Wrap the nodes clusters into a NeuralGrid
        >>> grid = NeuralGrid((3, 3), nodes)

        # Search the BMU of a data
        >>> nodes2 = grid.tolist()
        """

        nodes = []
        for i in range(self._size[0]):
            for j in range(self._size[1]):
                if self[i, j].size != 0:
                    nodes.append(self[i, j])

        return nodes
    #------------------------------------------------------------------------#

##############################################################################
