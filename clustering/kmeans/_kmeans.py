""" Lloyd's KMeans

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: June 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['KMeans', 'kmeans_params', 'kmeans_cluster']

import numpy as np

import clustering.tools as tls
import clustering.metrics as mts
import clustering.formats as dformat
from clustering._basecltcls import _BaseClustering, BOUNDS


##############################################################################
##                              K-Means class                               ##
##############################################################################

class KMeans(_BaseClustering):
    """ Lloyd's KMeans clustering

    Allow to declare an object representing the clusters, to train them
    on the data with the KMeans algorithm, and to build the clusters.
    The patterns are first initialized, then any data is sent into the
    cluster whose pattern is the nearest to the data. Once all the data
    are clustered, the patterns are updated as the group's mean. These
    steps are repeated until some criterion is satisfied, for instance
    a maximal tolerated variation in the centers from a round to another.

    The steps of the algorithm are as follows:
     1. Draw a set of K data (called "seeds"), denoted as `S = {s_k}_k`
        from the dataset, that will serve as initial patterns (called
        "prototypes") to the groups, denoted as `P = {p_k}_k`. The K
        data are randomly drawn from the training dataset.
     2. Compute the distances between any data and any prototypes `p`,
        and build the groups, denoted as `C = {C_k}_k`; a data is lin-
        ked to prototype `p_k`, and thus to cluster `C_k`, if this pro-
        totype is the nearest one among all the prototypes of `P`.
     3. Update the prototypes as the algebraic mean of the data inside
        the groups: the new `p_k` is the mean of the data of `C_k`.
     4. If the convergence criterion is not satisfied (maximal number
        of iterations, maximal variation between two updates, etc.),
        go back to step 2; else, ends the update procedure and return
        the so-built groups (the clusters).

    N.B.: with the K-Means, the update procedure is such that a cluster
        can never get empty, except at initialization (for instance, if
        two clusters have close prototypes, or if K is greater than the
        number of data). In this implementation, a mechanism is added to
        discard the groups that are empty at initialization. This means
        that clustering may output less than K groups; an informative
        message is displayed in such a case.

    Constructor
    -----------
    __init__(nb_clusters=2, seed=None, distance='euclidean')

    Constants
    ---------
    MARGINS = 0.01
        The maximal authorized variation before convergence.
    TMAX = 100
        The maximal authorized number of iterations for learning.

    Attributes
    ----------
    nb_clusters : int, getter & setter
        The number (K) of clusters to build.
    distance : str, getter & setter
        The distance function in use.
    patterns : np.ndarray, getter only
        The patterns (~representatives) of the clusters.

    Methods
    -------
    fit(data, verbose=False)
        Reset the patterns and train the KMeans on the provided data.
    build(data, cluster=True)
        Cluster input data with the trained K-Means.
    winner(data)
        Find the Best Matching Unit for a data.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    #--- Use the K-Means on a NumPy array
    # Generate a dummy database
    >>> array = np.arange(100.).reshape(-1, 5)

    # Instantiate a `KMeans` object
    >>> kms = KMeans(2, distance='euclidean')

    # Train the KMeans
    >>> kms.fit(array, verbose=True)
    [INFO] K-Means training -- Start
    0 1 2 3 DONE
    [INFO] K-Means training -- End

    # Build the clusters using the trained patterns
    >>> clusters = kms.build(array, cluster=False)      # NumPy arrays
    >>> print(type(clusters[0]))
    <class 'numpy.ndarray'>

    # Find the BMU of a data
    >>> bmu = kms.winner(array[15])
    >>> print(bmu)
    0

    #--- Use the K-Means on a Database and build Clusters
    # Generate a dummy database
    >>> database = Database(array, np.arange(len(array)))

    # Instantiate a `KMeans` object
    >>> kms = KMeans(2, distance='euclidean')

    # Train the KMeans
    >>> kms.fit(database, verbose=True)
    [INFO] K-Means training -- Start
    0 1 2 3 DONE
    [INFO] K-Means training -- End

    # Build the clusters using the trained patterns
    >>> clusters = kms.build(database, cluster=True)    # Cluster class objects
    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>

    # Find the BMU of a data
    >>> bmu = kms.winner(database[15])
    >>> print(bmu)
    1
    """

    # Constants
    MARGINS = 0.01                  # Max variation before convergence
    TMAX = 100                      # Max number of iterations for learning

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, nb_clusters=2, seed=None, distance='euclidean'):
        """ Instantiate a KMeans object (constructor)

        Parameters
        ----------
        [OPT] nb_clusters : int
            Number of patterns / clusters (K).
                :Default: 2
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
        # Instantiate a `KMeans` object
        >>> kms = KMeans()
        >>> kms = KMeans(2, 123, 'euclidean')
        """
        # pylint: disable=duplicate-code

        if nb_clusters < 1:
            raise ValueError(f"Wrong value {nb_clusters} for `nb_clusters`: "
                + "the number of clusters should be at least 1")

        # Store the number of clusters
        self._nb_clusters = nb_clusters

        super().__init__(seed, distance)
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def nb_clusters(self):
        """ Get the number of clusters to build """
        return self._nb_clusters

    @nb_clusters.setter
    def nb_clusters(self, nbclusters):
        """ Set the number of clusters to build """
        self._nb_clusters = nbclusters
    #------------------------------------------------------------------------#

    #---------------------   Training of the K-Means   ----------------------#
    def fit(self, data, verbose=False):
        """ Reset the patterns and train the KMeans on the provided data

        Train the KMeans on `data`. The patterns are first initialized
        and are then updated by analyzing the data samples. Learning
        stops either when the means variate less than `MARGINS` from a
        round to another, or after `TMAX` iterations. Only train the
        patterns; to build the clusters, use the `build` method.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to train the K-Means.
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

        # Instantiate a `KMeans` object
        >>> kms = KMeans(2)

        # Train the KMeans
        >>> kms.fit(database, verbose=True)
        [INFO] K-Means training -- Start
        0 1 2 DONE
        [INFO] K-Means training -- End

        >>> print(kms.patterns.round(3))
        [[0.763 0.763 0.763 0.763 0.763]
         [0.237 0.237 0.237 0.237 0.237]]
        """

        # Normalize the data and set the `_norm` and `_limits` attributes
        data = self._normalize_self(data)

        # Initialize the clusters' patterns
        self._initialize(data, self.nb_clusters)

        means = np.zeros(self._pats.shape, float)       # Clusters' means
        cards = np.zeros((self._nb_clusters, 1), int)   # Clusters' cardinals

        if verbose:
            print("[INFO] K-Means training -- Start")

        # Repeat learning `tmax` times
        for k in range(self.TMAX):
            if verbose:
                print(k, end=' ')

            # Reset
            means[:] = 0.
            cards[:] = 0

            # Find the nearest centroid for each data
            # TODO Iterate patterns rather than data?
            for val in data[:]:
                bmu = np.argmin(self._fdist(val, self._pats, 1)) # Centroids
                means[bmu] += val   # Accumulate the values for every centroid
                cards[bmu] += 1     # Increment the number of elements

            # Search for empty clusters (should be very rare, but possible)
            if any(cards == 0):
                pos = np.where(cards == 0)[0]   # Empty cluster indexes
                means = np.delete(means, pos, 0)
                cards = np.delete(cards, pos, 0)
                self._pats = np.delete(self._pats, pos, 0)
                self._nb_clusters -= 1
                print("Patterns initialization led to invalid empty clusters:",
                      f"discarding them ({len(self._pats)} remaining)")

            # Update the centroids' patterns
            else:
                # Get the state of the patterns to compare it after update
                patterns = self._pats.copy()
                # Update the patterns
                self._pats[:] = means / cards
                # Exit the loop if no variation in the patterns update
                if np.isclose(self._pats, patterns, self.MARGINS).all():
                    break

        if verbose:
            print("DONE")
            print("[INFO] K-Means training -- End")
    #------------------------------------------------------------------------#

    #------------------------   Cluster Input Data   ------------------------#
    def build(self, data, cluster=True):
        """ Cluster input data with the trained K-Means

        Build the clusters of the trained KMeans. Each data instance of
        `data` is compared to the `patterns` attribute, is associated
        with the closest one (BMU), and is added to its cluster.

        N.B.: some sets may be empty, since some of the actual patterns
            may be the closest to no instance from the input `data`.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to cluster.
        [OPT] cluster : bool
            Return the trained clusters as a list of np.ndarrays (False)
            or as a list of Cluster class objects (True).
                :Default: True

        Returns
        -------
        clusters : list of np.ndarrays or Clusters
            The clusters obtained after training the KMeans; the type
            of return depends on the method's arguments.

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database
        >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

        # Instantiate a `KMeans` object
        >>> kms = KMeans(2)

        # Train the KMeans
        >>> kms.fit(database, verbose=True)
        [INFO] K-Means training -- Start
        0 1 DONE
        [INFO] K-Means training -- End

        # Build the clusters using the trained patterns
        >>> clusters = kms.build(database, cluster=True)
        """

        if self._pats is None:
            raise AssertionError(
                "Patterns not trained; please run `fit` method prior")

        # Build the categories on the normalized data
        cats = self._build_cats(self._normalize(data))

        # List of np.ndarrays
        if not cluster:
            return [data[np.where(cats == i)] for i in range(len(self._pats))]

        # Get the index of the database
        index = dformat.get_index(data)

        # Build the clusters (denormalize the patterns)
        clusters = []
        for i, pat in enumerate(self._pats):
            idx = np.where(cats == i)
            clusters.append(dformat.Cluster(
                data[idx], index[idx],                      # Values & Indexes
                mts.rescale(pat, self._limits, BOUNDS)[0],  # Means
                dformat.get_tags(data),                     # Tags
                dformat.get_classes(data, idx)))            # Classes

        # List of Clusters
        return clusters
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                       K-Means Shorthand Functions                        ##
##############################################################################

#----------------------   KMeans Parameters Checking   ----------------------#
def kmeans_params(**params):
    """ Check the parameters for the KMeans

    Take up to 7 keyword arguments that define the required parameters
    for the instantiation and training of the KMeans, check if all the
    expected keys are present and fulfill all the missing ones with
    default values. Finally return a fulfilled 7-key dictionary. If no
    argument is fed into the function, set the default values within a
    new 7-key dictionary and return it.

    Refer to the `KMeans` class' documentation for additional informa-
    tion about the keys (inline keyword parameters).

    Parameters
    ----------
    [OPT] nb_clusters : int
        The number of clusters (K).
            :Default: 2
    [OPT] cluster : bool
        Return the trained clusters as a list of `np.ndarray` (False) or
        `Cluster` class objects (True).
            :Default: True
    [OPT] margins : float
        Maximal accepted variation in the means from a round to another:
        if higher, continue training.
            :Default: 0.01
    [OPT] tmax : int
        The maximal number of iterations of the training algorithm.
            :Default: 100
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
    kparams : 7-key dict
        The correctly fulfilled dictionary containing any of the keys.

    Examples
    --------
    # Get the default parameters
    >>> kmeans_params()
    {'nb_clusters': 2,
     'cluster': True,
     'margins': 0.01,
     'tmax': 100,
     'seed': None,
     'verbose': False,
     'distance': 'euclidean'}

    # Change only the number of clusters and leave the other default params
    >>> kmeans_params(nb_clusters=4)
    {'nb_clusters': 4,
     'cluster': True,
     'margins': 0.01,
     'tmax': 100,
     'seed': None,
     'verbose': False,
     'distance': 'euclidean'}

    # Change all the parameters
    >>> kmeans_params(
    ...     nb_clusters=4, cluster=False, margins=0.1, tmax=50,
    ...     seed=123, verbose=True, distance='manhattan')
    {'nb_clusters': 4,
     'cluster': False,
     'margins': 0.1,
     'tmax': 50,
     'seed': 123,
     'verbose': True,
     'distance': 'manhattan'}
    """
    # Default values
    kparams = {
        'nb_clusters': 2, 'cluster': True, 'margins': 0.01, 'tmax': 100,
        'seed': None, 'verbose': False, 'distance': 'euclidean'}
    # Check parameter's keys
    return tls.check_keys(params, kparams)
#----------------------------------------------------------------------------#

#----------------------   KMeans Training & Building   ----------------------#
def kmeans_cluster(data, **params):
    """ Instantiate & train the K-Means on a dataset and cluster it

    Take a database and the parameters for the clustering, train the
    KMeans on it and build the so-obtained clusters.

    Refer to the `KMeans` class for details, as this function is a
    shorthand version of it.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to train the K-Means and to cluster afterwards.

    Other Parameters
    ----------------
    **params : inline keyword arguments, optional
        The parameters passed to the `kmeans_params` function for checking.

    Returns
    -------
    clusters : list of np.ndarrays or of Clusters
        The clusters obtained after having trained the KMeans, represented
        by some sort of objects (depends on `params`).

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    # Generate a dummy database
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

    # Cluster the database with the KMeans with default parameters
    >>> clusters = kmeans_cluster(database)

    # Cluster the database with the KMeans with some specific parameters
    >>> clusters = kmeans_cluster(database, nb_clusters=5, verbose=True)
    [INFO] K-Means training -- Start
    0 Patterns initialization led to invalid empty clusters:
        discarding them (3 remaining)
    1 2 3 4 DONE
    [INFO] K-Means training -- End
    """
    # pylint: disable=invalid-name

    # Parameters to use
    kparams = kmeans_params(**params)

    # Initial means instantiation
    means = KMeans(kparams['nb_clusters'], kparams['seed'], kparams['distance'])

    # Global constants
    means.MARGINS, means.TMAX = kparams['margins'], kparams['tmax']

    # Train the KMeans
    means.fit(data, kparams['verbose'])

    # Build the clusters
    return means.build(data, kparams['cluster'])
#----------------------------------------------------------------------------#

##############################################################################
