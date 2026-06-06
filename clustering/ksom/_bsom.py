""" Bi-Level Self-Organizing Maps (BSOMs) class

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: February 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['BiLevelSOM', 'bsom_params', 'bsom_cluster']

import numpy as np

import clustering.tools as tls
from clustering.ksom._ksom import KohonenSOM, ksom_params


##############################################################################
##                      Bi-Level Self-Organizing Maps                       ##
##############################################################################

class BiLevelSOM(KohonenSOM):
    """ Bi-Level Self-Organizing Maps (BSOMs)

    The Kohonen's Self-Organizing Maps (SOMs) are stochastic, thus two
    SOMs trained on the same dataset will likely lead to two different
    clustering. In the particular case of Data Mining, where no prior
    knowledge is accessible and no assumption can be made on the data,
    it is hardly possible to state which clustering is the most repre-
    sentative; in practice, several may even be equally meaningful.

    A possible solution to this problem could be to train several maps
    on the same dataset, and then average them all in some fashion to
    extract the most common regions of the space as unique clusters.
    In order to average the so-generated clustering, a solution could
    be to let a SOM do this, as they are suited for self-organization.

    The BSOMs are an improvement to the standard SOMs: several maps are
    trained on the same dataset, and their respective nodes' patterns
    (the clusters' representatives) are then stacked in a new dataset,
    that is itself used to train a new SOM, that is finally returned.
    The BSOMs operate in two steps (whence their name): first, train
    the several maps; second, average them all with a final SOM.

    Refer to the `KohonenSOM` class for additional information.

    Constructor
    -----------
    __init__(gsize=(3, 3), seed=None, distance='euclidean')

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
        The size of the grids of the SOMs used by the BSOM.
    distance : function reference, getter & setter
        The distance function in use. The setter takes a string (cf.
        constructor) and the getter returns a function reference.
    patterns : np.ndarray, getter only
        The patterns (~representatives) of the clusters.

    Methods
    -------
    fit(data, nb_grids=10, verbose=False)
        Reset the patterns and train the BSOM on the provided data.
    build(data, cluster=True, grid=False)
        Cluster input data with the trained BSOM.
    winner(data)
        Find the Best Matching Unit for a data.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    #--- Use the Bi-Level SOM on a NumPy array
    # Generate a dummy database
    >>> array = np.arange(100.).reshape(-1, 5)

    # Instantiate a `BiLevelSOM` object
    >>> bsom = BiLevelSOM((3, 3), distance='euclidean')

    # Train the BSOM
    >>> bsom.fit(array, nb_grids=3, verbose=True)
	    Training grid 1 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End
	    Training grid 2 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End
	    Training grid 3 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
    [INFO] KSOM training -- End
	    Training the statistical grid
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End

    # Build the clusters from the trained patterns
    >>> clusters = bsom.build(array, cluster=False)     # NumPy arrays
    >>> print(type(clusters[0]))
    <class 'numpy.ndarray'>

    # Find the BMU of a data
    >>> bmu = bsom.winner(array[15])
    >>> print(bmu)
    8

    #--- Use the Bi-Level SOM on a Database and build Clusters
    # Generate a dummy database
    >>> database = Database(array, np.arange(len(array)))

    # Instantiate a `BiLevelSOM` object
    >>> bsom = BiLevelSOM((3, 3), distance='euclidean')

    # Train the BSOM
    >>> bsom.fit(database, nb_grids=3, verbose=True)
	    Training grid 1 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End
	    Training grid 2 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End
	    Training grid 3 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
    [INFO] KSOM training -- End
	    Training the statistical grid
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End

    # Build the clusters from the trained patterns
    >>> clusters = bsom.build(database, cluster=True)   # Cluster class objects
    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>

    # Find the BMU of a data
    >>> bmu = bsom.winner(database[15])
    >>> print(bmu)
    6
    """

    #-----------------   Statistical BSOM's Grid Training   -----------------#
    # pylint: disable-next=arguments-renamed
    def fit(self, data, nb_grids=10, verbose=False):
        """ Reset the patterns and train the BSOM on the provided data

        Train the BSOM on `data`. The nodes' patterns are initialized
        and are then updated by analyzing the data samples. Learning
        stops either when the patterns variate less than `MARGINS` from
        a round to another, or after `TMAX` iterations. Only train the
        patterns; to build the clusters, use the `build` method.


     and
    their classification might change from a round to another.
    Train `nb_grids` grids, gather all their respective prototypes
    in a list, seen as a new database, and finally train another
    grid with it (the database is the nodes' prototypes). A more
    homogeneous and more deterministic grid is expected.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to train the BSOM.
        [OPT] nb_grids : int
            Number of distinct SOMs to train and use as a new dataset
            later on for the final statistical neural grid.
                :Default: 10
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

        # Instantiate a `BiLevelSOM` object
        >>> bsom = BiLevelSOM((2, 2))

        # Train the BSOM
        >>> bsom.fit(database, nb_grids=3, verbose=True)
        [INFO] BSOM training -- Start
        10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
        [INFO] BSOM training -- End

        >>> print(bsom.patterns.round(3))
        [[0.625 0.625 0.625 0.625 0.625]
         [0.185 0.185 0.185 0.185 0.185]
         [0.589 0.589 0.589 0.589 0.589]
         [0.895 0.895 0.895 0.895 0.895]]

        # Train a set of BSOMs and average them as one
        >>> bsom.fit(database, nb_grids=3, verbose=True)
	        Training grid 1 / 3
        [INFO] BSOM training -- Start
        10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
        [INFO] BSOM training -- End
	        Training grid 2 / 3
        [INFO] BSOM training -- Start
        10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
        [INFO] BSOM training -- End
	        Training grid 3 / 3
        [INFO] BSOM training -- Start
        10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
        [INFO] BSOM training -- End
	        Training the statistical grid
        [INFO] BSOM training -- Start
        10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
        [INFO] BSOM training -- End

        >>> print(bsom.patterns.round(3))
        [[0.469 0.469 0.469 0.469 0.469]
         [0.258 0.258 0.258 0.258 0.258]
         [0.567 0.567 0.567 0.567 0.567]
         [0.792 0.792 0.792 0.792 0.792]]
        """

        if nb_grids < 1:
            raise ValueError(f"Wrong value {nb_grids} for `nb_grids`: "
                + "the number of grids to train must be at least 1")

        # Normalize the data and set the `_norm` and `_limits` attributes
        data = self._normalize_self(data)

        if nb_grids == 1:
            # Train the KSOM's grid
            self._training(data, verbose)

        else:

            # Nodes' prototypes of the `nb_grids` grids
            dim = 1 if data.ndim == 1 else data.shape[-1]
            weights = np.zeros((nb_grids, self._gsize[-1], dim), float)

            # Train the statistical grid
            for i in range(nb_grids):
                if verbose:
                    print(f"\tTraining grid {i+1} / {nb_grids}")

                # Train the a new grid (weights reset done in `_training`)
                self._training(data, verbose)

                # Copying the prototypes of the new trained grid
                weights[i] = self._pats

            if verbose:
                print("\tTraining the statistical grid")

            # Nodes' prototypes of the `nb_grids` grids
            if dim == 1 or data.shape[1] == 1:  # 1D data
                self._training(weights.ravel(), verbose)
            else:                               # 2D data
                self._training(weights.reshape(-1, dim), verbose)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                         BSOM Shorthand Functions                         ##
##############################################################################

#-----------------------   BSOM Parameters Checking   -----------------------#
def bsom_params(**params):
    """ Check the parameters for the BSOM

    Take up to 11 keyword arguments that define the required parameters
    for the instantiation and the training of the BSOM grid, check if
    all the expected keys are present and fulfill all the missing keys
    with default values. Finally return a fulfilled 11-key dictionary.
    If no argument is fed into the function, set the default values
    within a new 11-key dictionary and return it.

    Refer to the `BiLevelSOM` class' documentation for additional infor-
    mation about the keys (inline keyword parameters).

    Parameters
    ----------
    [OPT] grid_size : 2-tuple of ints
        The BSOM's grid's dimensions organized as (height, width), with
        `height` the number of rows and `width` the number of columns.
            :Default: (3, 3)
    [OPT] nb_grids : int
        The number of grids to train and average.
            :Default: 10
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
    params : 11-key dict
        The correctly fulfilled dictionary containing any of the keys.

    Examples
    --------
    # Get the default parameters
    >>> print(bsom_params())
    {'grid_size': (3, 3),
     'nb_grids': 10,
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
    >>> print(bsom_params(grid_size=(2, 2)))
    {'grid_size': (2, 2),
     'nb_grids': 10,
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
    >>> print(bsom_params(
    ...     'grid_size': (3, 3), nb_grids=5, 'cluster': True,
    ...     'grid': False, 'margins': 0.01, 'tmax': 1000,
    ...     'nbh_rate_0': 1.00, 'lrn_rate_0': 0.95,
    ...     'seed': 123, 'verbose': False, 'distance': 'euclidean'}))
    {'grid_size': (2, 2),
     'nb_grids': 5,
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
    # Check the parameters for the standard KSOMs
    bparams = ksom_params(**params)
    # Check and add the number of grids to the dictionary
    bparams.update(tls.check_keys(params, {'nb_grids': 10}))
    return bparams
#----------------------------------------------------------------------------#

#-----------------------   BSOM Training & Building   -----------------------#
def bsom_cluster(data, **params):
    """ Instantiate & train a BSOM on a dataset and cluster it

    Take a database and the parameters for the clustering, train the
    BSOM on it and build the so-built grid (clusters).

    Refer to the `BiLevelSOM` class for details, as this function is
    a shorthand version of it.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to train the BSOM and to cluster afterwards.

    Other Parameters
    ----------------
    **params : inline keyword arguments, optional
        The parameters passed to the `bsom_params` function for checking.

    Returns
    -------
    clusters : NeuralGrid or list of np.ndarrays or Clusters
        The trained BSOM's grid with the database projected onto it,
        gathered within some sort of objects (depends on `params`).

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    # Generate a dummy database
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

    # Cluster the database with the BSOM with default parameters
    >>> clusters = bsom_cluster(database, nb_grids=3, verbose=True)
	    Training grid 1 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End
	    Training grid 2 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End
	    Training grid 3 / 3
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
    [INFO] KSOM training -- End
	    Training the statistical grid
    [INFO] KSOM training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... DONE
    [INFO] KSOM training -- End

    # Cluster the database with the BSOM with some specific parameters
    >>> clusters = bsom_cluster(database, grid_size=(2, 2), nb_grids=10)
    """

    # Parameters to use
    bparams = bsom_params(**params)

    # Instantiate the Kohonen's SOM
    bsom = BiLevelSOM(bparams['grid_size'], bparams['seed'], bparams['distance'])

    # Global constants
    bsom.MARGINS, bsom.TMAX = bparams['margins'], bparams['tmax']
    bsom.LRN_RT_0, bsom.NBH_RT_0 = bparams['lrn_rate_0'], bparams['nbh_rate_0']

    # Train the BSOM
    bsom.fit(data, bparams['nb_grids'], bparams['verbose'])

    # Build the clusters (np.ndarrays, Clusters or NeuralGrid)
    return bsom.build(data, bparams['cluster'], bparams['grid'])
#----------------------------------------------------------------------------#

##############################################################################
