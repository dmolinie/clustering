""" Lloyd's Kernel KMeans

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: June 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['KernelKMeans', 'kkmeans_params', 'kkmeans_cluster']

import numpy as np

import clustering.tools as tls
import clustering.metrics as mts
import clustering.formats as dformat
from clustering._basecltcls import _BaseClustering
from clustering.kmeans._kmeans import kmeans_params


##############################################################################
##                           Kernel K-Means class                           ##
##############################################################################

class KernelKMeans(_BaseClustering):
    """ Lloyd's Kernel K-Means clustering

    Apply the regular K-Means to a dataset in a kernel space. The update
    procedure is the same as the K-Means, except that distances are com-
    puted in the kernel space; refer to the documentation of the `KMeans`
    class for details.

    Since the function to pass from the Euclidean space to this kernel
    space is generally unknown (or too complex), the "kernel trick" is
    used to compute a dot product in this nonlinear space, as defined by:

        d(x, y) = ||phi(x) - phi(y)||^2
                = ||phi(x)||^2 + ||phi(y)||^2 - 2*phi(x)*phi(y)

    where `d` is a distance from `x` to `y`, `phi` is the transition
    matrix from the Euclidean space to the kernel space, and `||x||`
    is the norm of `x`, such as `||x||^2 = <x>` (dot product of x).

    In general, `phi` is unknown (or even barely computable) while the
    dot product is, through the Gram Matrix, in which each coefficient
    is a dot product between two vectors of the basis.

    Constructor
    -----------
    __init__(nb_clusters=2, seed=None, kernel='gaussian', **ker_params)

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
    kernel : str, getter & setter
        The kernel function in use.
    clusters : list of np.ndarrays or Clusters, getter only
        The built clusters.
    patterns : np.ndarray, getter only
        The patterns (~representatives) of the clusters.

    Methods
    -------
    fit(data, verbose=False)
        Reset the patterns and train the kernel KMeans on the data.
    build(data, cluster=True)
        Cluster input data with the trained Kernel K-Means.
    winner(data)
        Find the Best Matching Unit for a data.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    #--- Use the Kernel K-Means on a NumPy array
    # Generate a dummy database
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3))))

    # Instantiate a `KernelKMeans` object
    >>> kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)

    # Train the Kernel KMeans
    >>> kkms.fit(array, verbose=True)
    [INFO] Kernel K-Means training -- Start
    0 DONE
    [INFO] Kernel K-Means training -- End

    # Build the clusters from the trained patterns
    >>> clusters = kkms.build(array, cluster=False)     # NumPy arrays
    >>> print(type(clusters[0]))
    <class 'numpy.ndarray'>

    # Find the BMU of a data
    >>> bmu = kkms.winner(array[15])
    >>> print(bmu)
    0

    #--- Use the Kernel K-Means on a Database and build Clusters
    # Generate a dummy database
    >>> database = Database(array, np.arange(len(array)))

    # Instantiate a `KernelKMeans` object
    >>> kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)

    # Train the Kernel KMeans
    >>> kkms.fit(database, verbose=True)
    [INFO] Kernel K-Means training -- Start
    0 DONE
    [INFO] Kernel K-Means training -- End

    # Build the clusters from the trained patterns
    >>> clusters = kkms.build(database, cluster=True)   # Cluster class objects
    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>

    # Find the BMU of a data
    >>> bmu = kkms.winner(database[15])
    >>> print(bmu)
    0
    """
    # pylint: disable=too-many-instance-attributes

    # Constants
    MARGINS = 0.01                  # Max variation before convergence
    TMAX = 100                      # Max number of iterations for learning

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, nb_clusters=2, seed=None, kernel='gaussian', **ker_params):
        """ Instantiate a KernelKMeans object (constructor)

        Parameters
        ----------
        [OPT] nb_clusters : int
            Number of patterns / clusters (K).
                :Default: 2
        [OPT] seed : int
            The seed for the NumPy random number generator. See NumPy's
            `random.default_rng` function for details.
                :Default: None
        [OPT] kernel : str
            The kernel name; see the `get_ker_func` function from the
            `metrics` module for details.
                :Default: 'gaussian'

        Other Parameters
        ----------------
        **ker_params : inline keyword arguments, optional
            The arguments for the kernel function; options depend on the
            kernel: `mean` (`circles`), `sigma` (`gaussian`), etc.

        Examples
        --------
        # Instantiate a `KernelKMeans` object
        >>> kkms = KernelKMeans()
        >>> kkms = KernelKMeans(2, 123)
        >>> kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)
        """
        # pylint: disable=duplicate-code

        if nb_clusters < 1:
            raise ValueError(f"Wrong value {nb_clusters} for `nb_clusters`: "
                + "the number of clusters should be at least 1")

        # Store the number of clusters
        self._nb_clusters = nb_clusters

        # Kernel to use (set in the `fit` method)
        self._ker = kernel
        self._fker = None
        self._fker_params = ker_params

        # Cluster's patterns and categories of the data
        # (Number of clusters required for instantiation)
        self._clusters = None

        super().__init__(seed, None)
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def nb_clusters(self):
        """ Get the number of clusters to build """
        return self._nb_clusters

    @nb_clusters.setter
    def nb_clusters(self, nb_clusters):
        """ Set the number of clusters to build """
        self._nb_clusters = nb_clusters

    @property
    def kernel(self):
        """ Get the kernel function in use """
        return self._ker

    @kernel.setter
    def kernel(self, kernel):
        """ Set the kernel function to use """
        self._fker = mts.get_ker_func(kernel)
        self._ker = kernel

    @property
    def clusters(self):
        """ Get the built clusters """
        return self._clusters
    #------------------------------------------------------------------------#

    #--------------------   Prototypes' Initialization   --------------------#
    def _initialize(self, data, nb_patterns):
        """ Random initialization of the nodes

        Randomly draw `nb_patterns` samples from `data`, and assign each
        one of these samples to a unique cluster's prototype (pattern).

        Parameters
        ----------
        data : np.ndarray or Database
            The data to train the Kernel K-Means.
        nb_patterns : int
            The number of patterns to initialize.

        Returns
        -------
        None : directly set the `_clusters` and `_pats` attributes, and
            possibly update the `_nb_clusters` attribute.
        """

        # Instantiate the `_pats` attribute
        super()._initialize(data, nb_patterns)

        # Flatten patterns for 1D data (easier handling thereafter)
        if data.ndim != self._pats.ndim:
            self._pats = self._pats.squeeze()

        # Get the kernel function from its name (passed in the constructor)
        # The function depends on the data dimension
        self._fker = mts.get_ker_func(self._ker, data.ndim)

        # Kernel patterns
        kpats = np.array([self._fker(pat, pat, 0, **self._fker_params)
                          for pat in self._pats], float)

        # Get the categories (classes) in the kernel space
        cats = [[] for k in range(self._nb_clusters)]
        for i, val in enumerate(data):
            cats[np.argmin(np.sqrt(
                self._fker(val, val, 0, **self._fker_params) + kpats
                - 2.*self._fker(val, self._pats, 1, **self._fker_params)))].append(i)

        # Get the indexes of the non-empty clusters
        valid = [i for i, cat in enumerate(cats) if cat]

        # Extract all the non-empty clusters
        if len(valid) != len(cats):
            cats = [cats[i] for i in valid]
            print("Patterns initialization led to invalid empty clusters:",
                  f"discarding them ({len(cats)} remaining)")

        # Initial clustering
        self._clusters = [data[cat] for cat in cats]

        # Update the clusters' means (in the euclidean space)
        self._pats[:] = np.array([clt.mean(0) for clt in self._clusters], float)

        # Update the number of clusters (for empty clusters)
        self._nb_clusters = len(self._clusters)
    #------------------------------------------------------------------------#

    #-------------------------   Categorize Data   --------------------------#
    def _build_cats(self, data):
        """ Build the categories (classes) of the input data

        Take a set of data and compare them to those of the sets of the
        `clusters` attribute. Associate any instance of `data` with the
        nearest cluster; this distance is computed in the kernel space
        using the kernel trick. The winner cluster is that at the lowest
        distance. Regroup the data indexes with the same category in the
        same set, and return the set of these sets, one per pattern.

        Note: the kernel is that specified in the `kernel` attribute.

        Parameters
        ----------
        data : np.ndarray
            The data to regroup by category.

        Returns
        -------
        cats : 2D list of ints
            The set of sets of indexes of the data belonging to the same
            category (i.e. with the same closest pattern).
        """

        # Gram's coefficients of the clusters (~ means)
        pats = np.zeros(len(self._clusters), float)
        for k, clt in enumerate(self._clusters):
            for val in clt:
                pats[k] += np.sum(self._fker(val, clt, 1, **self._fker_params))
            pats[k] /= len(clt)**2

        # Gram's coefficients of the data
        dist = np.zeros(len(self._clusters), float)
        cats = [[] for k in range(len(self._clusters))]
        for i, val in enumerate(data):
            for k, clt in enumerate(self._clusters):
                dist[k] =\
                    np.sum(self._fker(val, clt, 1, **self._fker_params)) / len(clt)
            trick = self._fker(val, val, 0, **self._fker_params) + pats - 2.*dist
            cats[np.argmin(np.sqrt(trick))].append(i)

        return cats
    #------------------------------------------------------------------------#

    #---------------------   Training of the KK-Means   ---------------------#
    def fit(self, data, verbose=False):
        """ Reset the patterns and train the Kernel KMeans on the data

        Train the Kernel KMeans on `data`. The patterns are initialized
        and are then updated by analyzing the samples. Learning stops
        either when the means variate less than `MARGINS` from a round
        to another, or after `TMAX` iterations. Only train the patterns;
        to build the clusters, use the `build` method.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to train the Kernel K-Means.
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
        None : directly update the `categories` attribute.

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database
        >>> rng = np.random.default_rng()
        >>> data = np.vstack(
        ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3))))
        >>> database = Database(data, np.arange(len(data)))

        # Instantiate a `KernelKMeans` object
        >>> kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)

        # Train the Kernel KMeans
        >>> kkms.fit(database, verbose=True)
        [INFO] Kernel K-Means training -- Start
        0 DONE
        [INFO] Kernel K-Means training -- End

        >>> print(kkms.patterns.round(3))
        [[9.949 9.922 9.931]
         [0.072 0.055 0.048]]
        """

        # Normalize the data and set the `_norm` and `_limits` attributes
        data = self._normalize_self(data)

        # Initialize the clusters and set the `clusters` attribute
        self._initialize(data, self.nb_clusters)

        if verbose:
            print("[INFO] Kernel K-Means training -- Start")

        # Repeat learning `tmax` times
        for k in range(self.TMAX):
            if verbose:
                print(k, end=' ')

            # Get the state of the patterns to compare it after update
            patterns = self._pats.copy()

            # Cluster the data
            cats = self._build_cats(data[:])

            # Update the clusters
            self._clusters = [data[cat] for cat in cats]

            # Update the clusters' means (in the Euclidean space)
            self._pats[:] = np.array([clt.mean(0) for clt in self._clusters], float)

            # Exit the loop if no variation in the means update
            if np.isclose(self._pats, patterns, self.MARGINS).all():
                break

        if verbose:
            print("DONE")
            print("[INFO] Kernel K-Means training -- End")

        # Reestablish 2D patterns, is necessary
        if self._pats.ndim == 1:
            self._pats = np.expand_dims(self._pats, 1)
    #------------------------------------------------------------------------#

    #------------------------   Build the Clusters   ------------------------#
    def build(self, data, cluster=True):
        """ Cluster input data with the trained Kernel K-Means

        Build the clusters of the trained Kernel KMeans. Every instance
        of `data` is compared to the samples of any set of the `clusters`
        attribute (in the kernel space, using the "kernel trick"), and
        the data whose closest cluster is the same are regrouped together
        into the same set. Return the list of such sets.

        N.B.: some sets may be empty, since some of the actual clusters
            may be the closest to no instance from the input `data`.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to cluster.
        [OPT] cluster : bool
            Return the trained clusters as a list of np.ndarrays
            (False) or as a list of Cluster class objects (True).
                :Default: True

        Returns
        -------
        clusters : list of np.ndarrays or Clusters
            The clusters obtained after training the Kernel KMeans;
            the type of return depends on the method's arguments.

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database
        >>> rng = np.random.default_rng()
        >>> data = np.vstack(
        ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3))))
        >>> database = Database(data, np.arange(len(data)))

        # Instantiate a `KernelKMeans` object
        >>> kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)

        # Train the Kernel KMeans
        >>> kkms.fit(database, verbose=True)
        [INFO] Kernel K-Means training -- Start
        0 DONE
        [INFO] Kernel K-Means training -- End

        # Build the clusters from the trained patterns
        >>> kkms.build(database, cluster=True)
        """

        if self._clusters is None:
            raise AssertionError(
                "Patterns not trained; please run `fit` method first")

        # Normalize the data
        data_norm = self._normalize(data)

        # Get the categories (classes)
        cats = self._build_cats(data_norm)

        # List of np.ndarrays
        if not cluster:
            return [data[cat] for cat in cats]

        # Get the index of the database
        index = dformat.get_index(data)

        # Build the clusters (denormalize the patterns)
        clusters = []
        for cat in cats:
            clusters.append(dformat.Cluster(
                data[cat], index[cat],                  # Values & Indexes
                data[cat].mean(0),                      # Means (Patterns)
                dformat.get_tags(data),                 # Tags
                dformat.get_classes(data, cat)))        # Classes

        return clusters
    #------------------------------------------------------------------------#

    #----------------------   Best Matching Unit BMU   ----------------------#
    def winner(self, data):
        """ Find the Best Matching Unit for a data

        Search for the nearest prototype of a data and return its posi-
        tion number. This research is processed in the kernel space by
        using the kernel trick. As such, the distance between the data
        and any cluster's patterns is computed in the kernel space.
        The winner cluster is that at the lowest distance from the data.

        Note: the kernel is that specified in the `kernel` attribute.

        Parameters
        ----------
        data : np.ndarray
            The data sample to categorize.

        Returns
        -------
        bmus : (array of) int(s)
            Cluster index to which belong `data`; if single array, return
            its BMU as an int; if set of arrays, return their BMUs as an
            array of ints, whose i-th value is the BMU of the i-th data.

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database
        >>> rng = np.random.default_rng()
        >>> data = np.vstack(
        ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3))))
        >>> database = Database(data, np.arange(len(data)))

        # Instantiate a `KernelKMeans` object
        >>> kkms = KernelKMeans(2, 0, 'gaussian', sigma=0.1)

        # Train the Kernel KMeans
        >>> kkms.fit(database, verbose=True)
        [INFO] Kernel K-Means training -- Start
        0 DONE
        [INFO] Kernel K-Means training -- End

        # Build the clusters from the trained patterns
        >>> kkms.build(database, cluster=True)

        # Find the BMU of a single data
        >>> bmu = kkms.winner(database[101])
        >>> print(bmu)
        0

        # Find the BMUs of a set of data
        >>> bmu = kkms.winner(database[95:105])
        >>> print(bmu)
        [1 1 1 1 1 0 0 0 0 0]
        """

        # Wrap any scalar input into np.ndarray
        if np.ndim(data) == 0:
            data = np.array([data])

        # Normalize the data
        data = self._normalize(data)

        # Build the categories (cluster index) to which belong every array
        cats = self._build_cats(data)

        # Rebuild the array of the BMUs
        size = sum(len(cat) for cat in cats)
        if size == 1:
            return tls.flat_list(cats)[0]

        # Fulfill the `bmus` array with the cluster indexes
        bmus = np.empty(size, int)
        for i, cat in enumerate(cats):
            for idx in cat:
                bmus[idx] = i

        return bmus
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                    Kernel K-Means Shorthand Functions                    ##
##############################################################################

#-------------------   Kernel KMeans Parameters Checking   ------------------#
def kkmeans_params(**params):
    """ Check the parameters for the Kernel KMeans

    Take up to 8 keyword arguments that define the required parameters
    for the instantiation and training of the Kernel KMeans, check if
    all the expected keys are present and fulfill all the missing ones
    with default values. Finally return a fulfilled 8-key dictionary.
    If no argument is fed into the function, set the default values
    within a new 8-key dictionary and return it.

    Refer to the `KernelKMeans` class' documentation for additional in-
    formation about the keys (inline keyword parameters).

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
    [OPT] kernel : str
        The kernel name; see the `get_ker_func` function from the
        `metrics` module for details.
            :Default: 'gaussian'
    [OPT] ker_params : dict
        The options passed to the kernel function; see the related func-
        tions for details; must be passed as a dictionary to distinguish
        them from the other inline keyword arguments.
            :Default: None (empty dict)

    Returns
    -------
    kparams : 8-key dict
        The correctly fulfilled dictionary containing any of the keys.

    Examples
    --------
    >>> kkmeans_params()
    {'nb_clusters': 2,
     'cluster': True,
     'margins': 0.01,
     'tmax': 100,
     'seed': None,
     'verbose': False,
     'kernel': 'gaussian',
     'ker_params': {'sigma': 0.1}}

    # Change only the number of clusters and leave the other default params
    >>> kkmeans_params(nb_clusters=4)
    {'nb_clusters': 4,
     'cluster': True,
     'margins': 0.01,
     'tmax': 100,
     'seed': None,
     'verbose': False,
     'kernel': 'gaussian',
     'ker_params': {'sigma': 0.1}}

    # Change all the parameters
    >>> kkmeans_params(
    ...     nb_clusters=4, cluster=False, margins=0.1, tmax=50, seed=123,
    ...     verbose=True, kernel='gaussian', ker_params={'sigma': 0.1})
    {'nb_clusters': 4,
     'cluster': False,
     'margins': 0.1,
     'tmax': 50,
     'seed': 123,
     'verbose': True,
     'kernel': 'gaussian',
     'ker_params': {'sigma': 0.1}}
    """

    # Linear KMeans parameters
    kparams = kmeans_params(**params)
    kparams.pop('distance') # Remove the distance (replaced with a kernel)

    # Default kernel
    kernel = {'kernel': 'gaussian', 'ker_params': {}}

    # Check kernel key
    kparams.update(tls.check_keys(params, kernel))

    return kparams
#----------------------------------------------------------------------------#

#------------------   Kernel KMeans Training & Building   -------------------#
def kkmeans_cluster(data, **params):
    """ Instantiate & train the Kernel K-Means on a dataset and cluster it

    Take a database and the parameters for the clustering, train the
    Kernel KMeans on it and build the so-built clusters.

    Refer to the `KernelKMeans` class for details, as this function
    is a shorthand version of it.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to train the Kernel K-Means and to cluster afterwards.

    Other Parameters
    ----------------
    **params : inline keyword arguments, optional
        The parameters passed to the `kkmeans_params` function for checking.

    Returns
    -------
    clusters : list of np.ndarrays or Clusters
        The clusters obtained after having trained the Kernel KMeans,
        represented by some sort of objects (depends on `params`).

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    # Generate a dummy database
    >>> rng = np.random.default_rng()
    >>> data = np.vstack(
    ...     (rng.normal(loc=0.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=3.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.0, scale=0.01, size=(100, 3))))
    >>> database = Database(data, np.arange(len(data)))

    # Cluster the database with the KMeans with default parameters
    >>> clusters = kkmeans_cluster(database)

    # Cluster the database with the KMeans with some specific parameters
    >>> clusters = kkmeans_cluster(database, nb_clusters=5, verbose=True)
    Patterns initialization led to invalid empty clusters:
        discarding them (3 remaining)
    [INFO] Kernel K-Means training -- Start
    0 DONE
    [INFO] Kernel K-Means training -- End
    """
    # pylint: disable=invalid-name

    # Parameters to use
    kparams = kkmeans_params(**params)

    # Initial means instantiation
    means = KernelKMeans(
        kparams['nb_clusters'], kparams['seed'],
        kparams['kernel'], **kparams['ker_params'])

    # Global constants
    means.MARGINS, means.TMAX = kparams['margins'], kparams['tmax']

    # Train the Kernel KMeans
    means.fit(data, kparams['verbose'])

    # Build the clusters
    return means.build(data, kparams['cluster'])
#----------------------------------------------------------------------------#

##############################################################################
