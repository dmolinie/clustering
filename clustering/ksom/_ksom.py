""" Kohonen's Self-Organizing Maps (KSOMs) class

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: February 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['KohonenSOM', 'ksom_params', 'ksom_cluster']

import numpy as np

import clustering.tools as tls
import clustering.metrics as mts
import clustering.formats as dformat
from clustering._basecltcls import _BaseClustering, BOUNDS
from clustering.ksom._neighborhood import (
    _neighborhood_1st, _neighborhood_2nd)


##############################################################################
##                      Kohonen's Self-Organizing Maps                      ##
##############################################################################

class KohonenSOM(_BaseClustering):
    """ Kohonen's Self-Organizing Maps (KSOMs)

    Allow the declaration of an object representing a grid of neurons,
    the data-driven training of the neurons and the classification of
    the data by projection of them onto the grid.

    The Self-Organizing Maps are grid-like maps composed of connected,
    trainable neurons, that serve as prototypes for the classes they
    represent. The training procedure of the grid is described below;
    it stops when the prototypes' update is negligible from a round to
    another, or after a certain number of iterations.

    To update the grid, proceed as follows: randomly draw a data from
    the database and compute its distance to the prototypes of any nodes;
    the closest node is the Best Matching Unit (BMU). Update the BMU's
    prototype to better match the data, as well as those of its nearest
    neighbors; the learning is done by region rather than by neuron.

    The learning function (weights update) is

            Wi(t+1) = Wi(t) + eps(t) * h(t) * (data(t) - Wi(t))

    where
      - `t` is the current learning time (or an iteration number)
      - `Wi(t)` is the prototype of the node / neuron i at time t
      - `eps(t)` is the learning rate at time t
      - `h(t)` is the neighborhood function (impact of the BMU)
      - `data(t)` is the randomly drawn data instance at time t

    The learning rate `eps` aims to reduce the impact a new data has
    on the node's prototypes updates; the learning rate is high at
    the beginning and progressively decreases. This allows the grid
    to quickly approach a steady state, and then the decrease of the
    learning rate aims to avoid oscillations. This function should
    be decreasing over time (iteration) to allow a high learning
    rate at the beginning, and little updates later on to allow the
    convergence of the algorithm. A possible function is:

            eps(t) = eps0 * exp( -t / tmax )

    where `tmax` is the allowed time spawn (max nb of iterations)

    The neighborhood function `h` aims to propagate the BMU's update
    to its neighbors. This allows a whole region to learn rather than
    only a single node. This function should be decreasing when the
    distance to the BMU increases. A possible function is a Gaussian-
    like function whose mean is the BMU's prototype and the standard
    deviation is the maximal acceptable distance to the BMU:

            h(t) = exp{ -[d(BMU, node(j))]^2 / [2 * sigma(t)^2] }

    where `sigma(t)` is the neighborhood coefficient, similarly to the
    standard deviation of this Gaussian-like function; it is a repre-
    sentation of the impact of the BMU on its neighbors: the farthest
    the node, the lowest the impact. A possible `sigma` is:

            sigma(t) = sigma0 * exp( -t / tmax )

    Note: to preserve the data topology in the grid, the nodes are lin-
        ked in a topological manner. For any node, 3 neighborhood orders
        are considered, each being assigned a distance value:
          - 1st (d = 1): nodes in direct contact with the BMU,
          - 2nd (d = 2): those in contact with the 1st order nodes,
          - 3rd (d = 3): all the remaining nodes.
        The BMU has distance `d = 0`, so `h_{BMU}(t) = 1`; `h` decreases
        as the BMU-node distance increases.

    Constructor
    -----------
    __init__(grid_size=(3, 3), seed=0, distance='euclidean')

    Constants
    ---------
    MARGINS = 0.01
        The maximal authorized variation before convergence.
    TMAX = 1000
        The maximal authorized number of iterations for learning.
    NBH_RT_0 = 1.00
        The neighborhood rate (spatial-span of a region); should decay
        over time to ensure convergence.
    LRN_RT_0 = 0.95  
        The learning rate; should decay over time to ensure convergence.

    Attributes
    ----------
    grid_size : 2-tuple of ints, getter & setter
        The KSOM's grid size.
    distance : str, getter & setter
        The distance function in use.
    patterns : np.ndarray, getter only
        The patterns (~representatives) of the clusters.

    Methods
    -------
    fit(data, verbose=False)
        Reset the patterns and train the KSOM on the provided data.
    build(data, cluster=True, grid=False)
        Cluster input data with the trained KSOM.
    winner(data)
        Find the Best Matching Unit for a data.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    #--- Use the K-SOM on a NumPy array
    # Generate a dummy database
    >>> array = np.arange(100.).reshape(-1, 5)

    # Instantiate a `KohonenSOM` object
    >>> ksom = KohonenSOM((3, 3), distance='euclidean')

    # Train the KSOM
    >>> ksom.fit(array, verbose=True)
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
    [INFO] KSOM training -- End

    # Build the clusters from the trained patterns
    >>> clusters = ksom.build(array, cluster=False)     # NumPy arrays
    >>> print(type(clusters[0]))
    <class 'numpy.ndarray'>

    # Find the BMU of a data
    >>> bmu = ksom.winner(array[15])
    >>> print(bmu)
    3

    #--- Use the K-SOM on a Database and build Clusters
    # Generate a dummy database
    >>> database = Database(array, np.arange(len(array)))

    # Instantiate a `KohonenSOM` object
    >>> ksom = KohonenSOM((3, 3), distance='euclidean')

    # Train the KSOM
    >>> ksom.fit(database, verbose=True)
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
    [INFO] KSOM training -- End

    # Build the clusters from the trained patterns
    >>> clusters = ksom.build(database, cluster=True)   # Cluster class objects
    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>

    # Find the BMU of a data
    >>> bmu = ksom.winner(database[15])
    >>> print(bmu)
    3
    """

    # Constants
    MARGINS = 0.01                  # Max variation before convergence
    TMAX = 1000                     # Max number of iterations for learning
    NBH_RT_0 = 1.00                 # Initial neighborhood rate
    LRN_RT_0 = 0.95                 # Initial grid's learning rate

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, grid_size=(3, 3), seed=None, distance='euclidean'):
        """ Instantiate a KohonenSOM object (constructor)

        Note: member `patterns` is a list of float arrays (each element
            is an array), whose dimension equals that of the data inst-
            ances (same number of features). To be consistent with the
            output as a `NeuralGrid` object, the list is disposed such
            that the first line of the `patterns` list corresponds to
            first neuron of the Grid (top left), and the last line of
            the list is the last neuron (bottom right). Then, the nodes
            are run through from left to right and from top to bottom,
            whereas the list is only run through from top to bottom. As
            such, the node (m, n) of the grid corresponds to the line
            `m * length + n` of the list.

        Parameters
        ----------
        [OPT] grid_size : 2-tuple or 2-list
            Grid's dimensions organized as (height, width), with `height`
            the number of rows and `width` the number of columns.
                :Default: (3, 3)
        [OPT] seed : int
            The seed for the NumPy random number generator. See NumPy's
            `random.default_rng` function for details.
                :Default: None
        [OPT] distance : str
            The distance name; see the `get_dist_func` function from the
            `metrics` module for details.
                :Default: 'euclidean'

        Examples
        --------
        # Instantiate a `KohonenSOM` object
        >>> kms = KohonenSOM()
        >>> kms = KohonenSOM((3, 3), 123, 'euclidean')
        """

        if isinstance(grid_size, (tuple, list, np.ndarray)):
            if np.sum(grid_size) < 2:
                raise ValueError(f"Wrong value {grid_size} for `grid_size`: "
                    + "grid size must be a tuple of positive integers")
        else:
            raise TypeError(f"Wrong type {type(grid_size)} for `grid_size`: "
                  + "grid size must be a tuple or list")

        # Dimensions of the grid (Height, Width, Neurons=Hgt*Wdt)
        self._gsize = (grid_size[0], grid_size[1], grid_size[0]*grid_size[1])

        super().__init__(seed, distance)
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def grid_size(self):
        """ Get the KSOM's grid size """
        return self._gsize

    @grid_size.setter
    def grid_size(self, grid_size):
        """ Set the KSOM's grid size """
        self._gsize = grid_size
    #------------------------------------------------------------------------#

    #----------------------   KSOM's Nodes Learning   -----------------------#
    def _training(self, data, verbose):
        """ Core data-driven learning algorithm

        Parameters
        ----------
        data : np.ndarray or Database
            The data to train the KSOM.
        verbose : bool
            Display (True) or hide (False) display advancement messages.

        Returns
        -------
        None : directly update the `patterns` attribute.
        """
        # pylint: disable=too-many-locals

        tmax_inv = 1. / self.TMAX

        if verbose:
            inc = int(0.1*self.TMAX)
            pct = 0
            pos = inc

        # Initialize (reset) the grid's nodes' prototypes
        self._initialize(data, self._gsize[-1])

        if verbose:
            print("[INFO] KSOM training -- Start")

        # Repeat learning `tmax` times
        for k in range(self.TMAX):
            if verbose and k == pos:
                pos += inc
                pct += 10
                print(f"{pct}%... ", end='')

            # Get the state of the patterns to compare it after update
            patterns = self._pats.copy()

            # Randomly draw an input array
            temp = data[self._rng.integers(data.shape[0])].copy()

            # Best Matching Unit BMU
            bmu = np.argmin(self._fdist(temp, self._pats, 1))

            # 1st and 2nd orders neighbors (3rd = all remaining nodes)
            neighbors_1st = _neighborhood_1st(bmu, self._gsize)
            neighbors_2nd = _neighborhood_2nd(bmu, self._gsize)
            neighbors_3rd = [i for i in range(self._gsize[2])
                             if i not in neighbors_1st+neighbors_2nd+[bmu]]

            #---------  Prototypes' weights' update  ---------#
            # Decay learning and neighborhood rates
            exp_t = np.exp( -k * tmax_inv )
            lrn_rt = self.LRN_RT_0 * exp_t
            nbh_rt = 1. / (2. * (self.NBH_RT_0 * exp_t)**2)

            # Find the BMU
            wgt = self._pats[bmu]               # BMU --> d = 0 --> h = 1
            self._pats[bmu] = wgt + lrn_rt * (temp - wgt)

            # Update 1st order neighbors
            neigh = np.exp( -nbh_rt )           # 1st order --> d^2 = 1
            for i, wgt in zip(neighbors_1st, self._pats[neighbors_1st]):
                self._pats[i] = wgt + lrn_rt * neigh * (temp - wgt)

            # Update 2nd order neighbors
            neigh = np.exp( -4 * nbh_rt )       # 2nd order --> d^2 = 4
            for i, wgt in zip(neighbors_2nd, self._pats[neighbors_2nd]):
                self._pats[i] = wgt + lrn_rt * neigh * (temp - wgt)

            # Update 3rd order neighbors (all remaining nodes)
            neigh = np.exp( -9 * nbh_rt )       # 3rd order --> d^2 = 9
            for i, wgt in zip(neighbors_3rd, self._pats[neighbors_3rd]):
                self._pats[i] = wgt + lrn_rt * neigh * (temp - wgt)
            #-------------------------------------------------#

            # Exit the loop if no variation in the patterns update
            if np.isclose(self._pats, patterns, self.MARGINS).all():
                break

        if verbose:
            print("DONE")
            print("[INFO] KSOM training -- End")
    #------------------------------------------------------------------------#

    #--------------------------   Train the KSOM   --------------------------#
    def fit(self, data, verbose=False):
        """ Reset the patterns and train the KSOM on the provided data

        Train the KSOM on `data`. The nodes' patterns are initialized
        and are then updated by analyzing the data samples. Learning
        stops either when the patterns variate less than `MARGINS` from
        a round to another, or after `TMAX` iterations. Only train the
        patterns; to build the clusters, use the `build` method.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to train the KSOM.
        [OPT] verbose : bool
            Display (True) or hide (False) display advancement messages.
                :Default: False

        Other Parameters
        ----------------
        [CST] MARGINS : float, class constant
            The maximal authorized relative variation from a round to
            another; if the patterns (means) variate less than this,
            stop learning; else, continue.
        [CST] TMAX : int, class constant
            The maximal authorized number of iterations; learning auto-
            matically stops then.

        Returns
        -------
        None : directly update the `patterns` attribute.

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database
        >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

        # Instantiate a `KohonenSOM` object
        >>> ksom = KohonenSOM((2, 2))

        # Train the KSOM
        >>> ksom.fit(database, verbose=True)
        [INFO] KSOM training -- Start
        10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
        [INFO] KSOM training -- End

        >>> print(ksom.patterns.round(3))
        [[0.625 0.625 0.625 0.625 0.625]
         [0.185 0.185 0.185 0.185 0.185]
         [0.589 0.589 0.589 0.589 0.589]
         [0.895 0.895 0.895 0.895 0.895]]
        """

        # Normalize the data and set the `_norm` and `_limits` attributes
        data = self._normalize_self(data)

        # Train the KSOM's grid
        self._training(data, verbose)
    #------------------------------------------------------------------------#

    #--------------------------   Grid Building   ---------------------------#
    def build(self, data, cluster=True, grid=False):
        """ Cluster input data with the trained KSOM

        Build the clusters of the trained KSOM. Each instance of `data`
        is compared to the `patterns` attribute, and is associated with
        the closest one (BMU); it is then project onto the 2D grid.

        N.B.: some sets may be empty, since some of the actual patterns
            may be the closest to no instance from the input `data`.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to cluster.
        [OPT] cluster : bool
            Return the trained clusters as a list of np.ndarrays (False)
            or as a list of Cluster objects (True).
                :Default: True
        [OPT] grid : bool
            Project the clusters into a NeuralGrid class object (True)
            or leave them as a simple list.
                :Default: False

        Returns
        -------
        clusters : NeuralGrid or list of np.ndarrays or Clusters
            The clusters obtained after having trained the KSOM; the
            type of return depends on the method's arguments.

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database
        >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

        # Instantiate a `KohonenSOM` object
        >>> ksom = KohonenSOM((3, 3))

        # Train the KSOM
        >>> ksom.fit(database, verbose=True)
        [INFO] KSOM training -- Start
        10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
        [INFO] KSOM training -- End

        # Build the clusters from the trained patterns
        >>> ksom.build(database, cluster=True)
        """

        if self._pats is None:
            raise AssertionError(
                "Patterns not trained; please run `fit` method first")

        # Build the categories on the normalized data
        cats = self._build_cats(self._normalize(data))

        # List of np.ndarrays
        if not cluster:
            return [data[np.where(cats == i)] for i in range(self._gsize[-1])]

        # Get the index of the database
        index = dformat.get_index(data)

        # Build the clusters (denormalize the patterns)
        nodes = []
        for i, pat in enumerate(self._pats):
            idx = np.where(cats == i)[0]                    # `[0]` for Nx1 data
            nodes.append(dformat.Cluster(
                data[idx], index[idx],                      # Values & Index
                mts.rescale(pat, self._limits, BOUNDS)[0],  # Patterns
                dformat.get_tags(data),                     # Tags
                dformat.get_classes(data, idx)))            # Classes

        # List of Clusters
        if not grid:
            return nodes

        # NeuralGrid object with size `(height, width)`
        return dformat.NeuralGrid((self._gsize[:2]), nodes)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                         KSOM Shorthand Functions                         ##
##############################################################################

#-----------------------   KSOM Parameters Checking   -----------------------#
def ksom_params(**params):
    """ Check the parameters for the Kohonen's SOM

    Take up to 10 keyword arguments that define the required parameters
    for the instantiation and the training of the KSOM grid, check if
    all the expected keys are present and fulfill all the missing keys
    with default values. Finally return a fulfilled 10-key dictionary.
    If no argument is fed into the function, set the default values
    within a new 10-key dictionary and return it.

    Refer to the `KohonenSOM` class' documentation for additional infor-
    mation about the keys (inline keyword parameters).

    Parameters
    ----------
    [OPT] grid_size : 2-tuple of ints
        The KSOM's grid's dimensions organized as (height, width), with
        `height` the number of rows and `width` the number of columns.
            :Default: (3, 3)
    [OPT] cluster : bool
        Return the trained clusters as a list of `np.ndarray` (False) or
        `Cluster` objects (True).
            :Default: True
    [OPT] grid : bool
        Either project the clusters onto a `NeuralGrid` class object (True)
        or leave them as a simple list of `np.ndarray` or `Cluster` objects.
            :Default: False
    [OPT] margins : float
        Maximal accepted variation in the patterns from a round to an-
        other: if higher, continue training.
            :Default: 0.01
    [OPT] tmax : int
        The maximal number of iterations of the training algorithm.
            :Default: 1000
    [OPT] nbh_rate_0 : float
        The initial neighborhood rate; decayed over time.
            :Default: 1.00
    [OPT] lrn_rate_0 : float
        The initial learning rate, decayed over time.
            :Default: 0.95
    [OPT] seed : int
        The seed for the NumPy random number generator. See the NumPy's
        `random.default_rng` function for details.
            :Default: None
    [OPT] verbose : bool
        Display (True) or hide (False) display advancement messages.
            :Default: False
    [OPT] distance : str
        The distance name; see the `get_dist_func` function from the
        `metrics` module for details.
            :Default: 'euclidean'

    Returns
    -------
    params : 10-key dict
        The correctly fulfilled dictionary containing any of the keys.

    Examples
    --------
    # Get the default parameters
    >>> print(ksom_params())
    {'grid_size': (3, 3),
     'cluster': True,
     'grid': False,
     'margins': 0.01,
     'tmax': 1000,
     'nbh_rate_0': 1.0,
     'lrn_rate_0': 0.95,
     'seed': None,
     'verbose': False,
     'distance': 'euclidean'}

    # Change only the number of clusters and leave the other default params
    >>> print(ksom_params(grid_size=(2, 2)))
    {'grid_size': (2, 2),
     'cluster': True,
     'grid': False,
     'margins': 0.01,
     'tmax': 1000,
     'nbh_rate_0': 1.0,
     'lrn_rate_0': 0.95,
     'seed': None,
     'verbose': False,
     'distance': 'euclidean'}

    # Change all the parameters
    >>> kparams = {
    ...     'grid_size': (3, 3), 'cluster': True, 'grid': False,
    ...     'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
    ...     'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
    ...     'distance': 'euclidean'}
    {'grid_size': (2, 2),
     'cluster': False,
     'grid': False,
     'margins': 0.01,
     'tmax': 50,
     'nbh_rate_0': 1.0,
     'lrn_rate_0': 0.95,
     'seed': 123,
     'verbose': True,
     'distance': 'manhattan'}
    """
    # Default values
    kparams = {
        'grid_size': (3, 3), 'cluster': True, 'grid': False,
        'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        'distance': 'euclidean'}
    # Check parameter's keys
    return tls.check_keys(params, kparams)
#----------------------------------------------------------------------------#

#-----------------------   KSOM Training & Building   -----------------------#
def ksom_cluster(data, **params):
    """ Instantiate & train a KSOM on a dataset and cluster it

    Take a database and the parameters for the clustering, train the
    KSOM on it and build the so-built grid (clusters).

    Refer to the `KohonenSOM` class for details, as this function is
    a shorthand version of it.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to train the KSOM and to cluster afterwards.

    Other Parameters
    ----------------
    **params : inline keyword arguments, optional
        The parameters passed to the `ksom_params` function for checking.

    Returns
    -------
    clusters : NeuralGrid or list of np.ndarrays or Clusters
        The trained KSOM's grid with the database projected onto it,
        gathered within some sort of objects (depends on `params`).

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    # Generate a dummy database
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

    # Cluster the database with the KSOM with default parameters
    >>> clusters = ksom_cluster(database)

    # Cluster the database with the KSOM with some specific parameters
    >>> clusters = ksom_cluster(database, grid_size=(2, 2), verbose=True)
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
    [INFO] KSOM training -- End
    """
    # pylint: disable=invalid-name

    # Parameters to use
    kparams = ksom_params(**params)

    # Instantiate the Kohonen's SOM
    ksom = KohonenSOM(kparams['grid_size'], kparams['seed'], kparams['distance'])

    # Global constants
    ksom.MARGINS, ksom.TMAX = kparams['margins'], kparams['tmax']
    ksom.LRN_RT_0, ksom.NBH_RT_0 = kparams['lrn_rate_0'], kparams['nbh_rate_0']

    # Train the KSOM
    ksom.fit(data, kparams['verbose'])

    # Build the clusters (np.ndarrays, Clusters or NeuralGrid)
    return ksom.build(data, kparams['cluster'], kparams['grid'])
#----------------------------------------------------------------------------#

##############################################################################
