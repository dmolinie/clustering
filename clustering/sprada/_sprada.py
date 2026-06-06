""" Self-Parameterized Recursively Assessed Decomposition Algorithm (SPRADA)

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2021
Last revised: May 2026

License: GPLv3
"""
# TODO Add a `build` and `winner` methods?

__all__ = ['SPRADA', 'sprada_params', 'sprada_cluster']

import numpy as np

import clustering.tools as tls
import clustering.formats as dformat
from clustering import ecd
from clustering.sprada._recursive import Recursive, recursive_params


##############################################################################
##                            SPRADA Clustering                             ##
##############################################################################

class SPRADA(Recursive):
    """ Self-Parameterized Recursively Assessed Decomposition Algorithm

    Extension of the Recursive class of this module, by adding a stage
    of cluster merging with the ECD Test, defined as a dedicated class
    in the `ecd` module. Substantially, the `SPRADA` class uses the
    `Recursive` class for building the clusters, and uses the `ECDTest`
    class for refining them in a split-and-merge fashion.

    See the `Recursive` and `ECDTest` classes for algorithmic details.

    Constructor
    -----------
    __init__(
        fclt1='ksom', fclt2='kmeans', fquant='std',
        fstats='quantile', comparison='lower', **params)

    Attributes
    ----------
    ecdtest : ECDTest object, getter only
        The variable of the ECD Test.
    groups : 2D list of ints, getter only
        The groups of the indices of clusters.
    + Those of the `Recursive` class.

    Methods
    -------
    fit(data)
        Build the tree & pass the clusters to the ECD Test.
    build_matrix(pts=100)
        Compute the ECDs and build the matrix of the MHDs.
    build_groups(eps=0.15)
        Build the groups by Regions Growing.
    merge_groups()
        Merge the clusters of the groups identified by the ECD Test.
    + Those of the `Recursive` class.

    Examples
    --------
    ### See `Recursive` class Examples section for more examples

    >>> import numpy as np
    >>> from clustering.formats import Database


    #--- Apply the SPRADA decomposition to a NumPy array
    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))

    # Instantiate a `SPRADA` object with default parameters
    >>> tree = SPRADA()

    # Build the clusters
    >>> clusters = tree.fit(array)

    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>
    >>> print(len(clusters))
    19

    # Build the matrix of the MHDs
    >>> matrix = tree.build_matrix()

    # Regroup the clusters by distances
    >>> groups = tree.build_groups()

    # Merge the clusters of the groups
    >>> clusters = tree.merge_groups()

    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>
    >>> print(len(clusters))
    3


    #--- Apply the SPRADA decomposition to a Database and build Clusters
    >>> database = Database(array, np.arange(len(array)))

    # Instantiate a `SPRADA` object with default parameters
    >>> tree = SPRADA()

    # Build the clusters
    >>> clusters = tree.fit(database)

    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>
    >>> print(len(clusters))
    18

    # Build the matrix of the MHDs
    >>> matrix = tree.build_matrix()

    # Regroup the clusters by distances
    >>> groups = tree.build_groups()

    # Merge the clusters of the groups
    >>> clusters = tree.merge_groups()

    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>
    >>> print(len(clusters))
    3


    #--- Apply the SPRADA decomposition with specific parameters
    #--- and leave the clusters as `np.ndarrays` (no `Cluster` objects)

    # Parameters for `KohonenSOM`
    >>> ksom_params = {
    ...     'grid_size': (3, 3), 'cluster': True, 'grid': False,
    ...     'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
    ...     'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
    ...     'distance': 'euclidean'}

    # Parameters for `KMeans`
    >>> kmeans_params = {
    ...     'nb_clusters': 2, 'cluster': True,
    ...     'margins': 0.01, 'tmax': 100, 'seed': None,
    ...     'verbose': False, 'distance': 'euclidean'}

    # Initialize the tree with these functions & parameters
    >>> tree = SPRADA(
    ...     fclt1='ksom', fclt1_params=ksom_params,
    ...     fclt2='kmeans', fclt2_params=kmeans_params)

    # Build the clusters (NumPy ndarrays returned as `cluster` is False
    # in both `ksom_params` and `kmeans_params`)
    >>> clusters = tree.fit(database)

    >>> print(type(clusters[0]))
    <class 'numpy.ndarray'>
    >>> print(len(clusters))
    21

    # Build the matrix of the MHDs
    >>> matrix = tree.build_matrix(100)

    # Regroup the clusters by distances
    >>> groups = tree.build_groups(0.1)

    # Merge the clusters of the groups
    >>> clusters = tree.merge_groups()

    >>> print(type(clusters[0]))
    <class 'numpy.ndarray'>
    >>> print(len(clusters))
    3
    """

    #---------------------------   Constructor   ----------------------------#
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(self,
        fclt1='ksom', fclt2='kmeans', fquant='std',
        fstats='quantile', comparison='lower', **params):
        """ Instantiate a SPRADA object (constructor)

        Instantiate a Recursive class object, and add the `groups` att-
        ribute, which should contain the different groups rebuilt by
        the ECD Test. See `Recursive` class' constructor for details.

        Parameters
        ----------
        [OPT] fclt1 : str
            The clustering function for the initial split; can differ
            from `fclt2`. See `get_clust_func` from `cluster`.
                :Default: 'ksom'
        [OPT] fclt2 : str
            The clustering function for cluster refinement; can differ
            from `fclt1`. See `get_clust_func` from `cluster`.
                :Default: 'kmeans' 
        [OPT] fquant : str
            The quantifying function to estimate cluster's homogeneity.
            See the `get_quant_func` function from the `metrics` module.
                :Default: 'std'
        [OPT] fstats : str
            The statistical function to compute the dynamic threshold
            as a statistic of the quantifiers of the first split. See
            `get_stat_func` function from `metrics` module. Function
            'quantile' is recommended as its value (quantile) `q` can
            serve as a dynamic threshold (see `np.quantile`).
                :Default: 'quantile'
        [OPT] comparison : str
            If the quantifier of a cluster must be lower or greater than
            a certain threshold to be split or not. For instance, for a
            density, the greater, the better; but for the standard devia-
            tion, the lower, the better. Options are {'lower', 'greater'}.
                :Default: 'lower'

        Other Parameters
        ----------------
        [OPT] fclt1_params : dict
            The parameters for the clustering function for initial split.
            See functions returned by `get_clust_func` for details.
                :Default: {} (default parameters of `KohonenSOM`)
        [OPT] fclt2_params : dict
            The parameters for the clustering function for refinement.
            See functions returned by `get_clust_func` for details.
                :Default: {} (default parameters of `KMeans`)
        [OPT] fquant_params : dict
            The parameters for the quantifying function. See functions
            returned by `get_quant_func` for details.
                :Default: {}
        [OPT] fstats_params: dict
            The parameters for the statistical function. See functions
            returned by `get_stat_func` for details. Default to the para-
            meters for the `quantile` function.
                :Default: { {'q': 0.75}  if `comparison` is 'lower'
                          { {'q': 0.25}  if `comparison` is 'greater'

        Examples
        --------
        #--- Use the default parameters
        >>> tree = SPRADA()

        #--- Use specific functions with their default parameters
        >>> tree = SPRADA(
        ...     fquant='std', fstats='mean', fclt1='ksom', fclt2='ksom')

        #--- Use specific functions & their parameters
        # Parameters for `density`
        >>> den_params = {
        ...     'span': 'sphere_span',
        ...     'volume': 'hypersphere',
        ...     'distance': 'euclidean'}

        # Parameters for `quantile`
        # (0.25 with 'greater' comparison to split the 25% first clusters)
        >>> stats_params = {'q': 0.25}

        # Parameters for `KohonenSOM`
        >>> ksom_params = {
        ...     'grid_size': (3, 3), 'cluster': True, 'grid': False,
        ...     'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
        ...     'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
        ...     'distance': 'euclidean'}

        # Parameters for `KMeans`
        >>> kmeans_params = {
        ...     'nb_clusters': 2, 'cluster': True,
        ...     'margins': 0.01, 'tmax': 100, 'seed': None,
        ...     'verbose': False, 'distance': 'euclidean'}

        # Initialize the tree with these functions & parameters
        >>> tree = SPRADA(
        ...     fclt1='ksom', fclt1_params=ksom_params,
        ...     fclt2='kmeans', fclt2_params=kmeans_params,
        ...     fquant='density', fquant_params=den_params,
        ...     fstats='quantile', fstats_params=stats_params,
        ...     comparison='greater')
        """
        # Instantiate a `Recursive` tree
        super().__init__(fclt1, fclt2, fquant, fstats, comparison, **params)
        # Instantiate an `ECDTest` object
        self._ecdtest = ecd.ECDTest()
    #------------------------------------------------------------------------#

    #----------------------------   Properties   ----------------------------#
    @property
    def ecdtest(self):
        """ Get the `ECDTest` object """
        return self._ecdtest

    @property
    def groups(self):
        """ Get the `groups` attribute """
        return self._ecdtest.groups
    #------------------------------------------------------------------------#

    #----------------   Set Clusters & Instantiate ECDTest   ----------------#
    def fit(self, dataset):
        """ Build the tree & pass the clusters to the ECD Test

        Call the `fit` method of the parent class Recursive (see this
        method for details), instantiate an ECDTest class object (set
        as protected attribute) and fit it with the clusters issued by
        the `Recursive`'s `fit` method.

        Parameters
        ----------
        dataset : np.ndarray, Database or Cluster
            The dataset to hierarchically split.

        Returns
        -------
        clusters : list of np.ndarrays or Clusters
            The list of homogeneous clusters (the `clusters` attribute).
            The type of the clusters is managed by the `fclt1` & `fclt2`
            functions (specified in their dictionary of parameters).

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database with 3 Gaussian distributions (regions)
        >>> rng = np.random.default_rng()
        >>> array = np.vstack(
        ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
        >>> database = Database(array, np.arange(len(array)))

        #--- Instantiate a tree with `std` quantifier
        >>> tree = SPRADA(fquant='std', comparison='lower')

        # Build the clusters (set `comparison` to 'lower',
        # since for `std`, the lower, the better
        >>> clusters = tree.fit(database)

        >>> print(len(clusters))
        19
        >>> print(clusters == tree.ecdtest.clusters)
        True

        #--- Instantiate a `SPRADA` object with `density` quantifier
        >>> tree = SPRADA(fquant='density', comparison='greater')

        # Build the clusters (set `comparison` to 'greater',
        # since for `density`, the greater, the better
        >>> clusters = tree.fit(database)

        >>> print(len(clusters))
        6
        >>> print(clusters == tree.ecdtest.clusters)
        True
        """

        # Build the clusters using the `fit` method from the parent class
        super().fit(dataset)

        print(f"[INFO] SPRADA: {len(self._clusters)} clusters built -- "
            + "Starting the ECD merging procedure")

        # Pass the clusters issued by the `fit` method to the `ECDTest`
        self._ecdtest.fit(clusters=self._clusters)

        return self._clusters
    #------------------------------------------------------------------------#

    #----------------------   Build the MHDs Matrix   -----------------------#
    def build_matrix(self, pts=100):
        """ Compute the ECDs and build the matrix of the MHDs

        Call the `build_matrix` method of the `ecdtest` protected att-
        ribute to compute the clusters' ECDs and build the MHDs matrix.
        See method `build_matrix` of ECDTest class for details.

        Parameters
        ----------
        [OPT] pts : int
            The number of subintervals to cut the dataset/clusters data
            range up into.
                :Default: 100

        Returns
        -------
        matrix : np.ndarray
            The KxK MHDs matrix (the ECDTest class' `matrix` attribute).

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database with 3 Gaussian distributions (regions)
        >>> rng = np.random.default_rng()
        >>> array = np.vstack(
        ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
        >>> database = Database(array, np.arange(len(array)))

        # Instantiate a `SPRADA` object with default parameters
        >>> tree = SPRADA()

        # Build the clusters
        >>> clusters = tree.fit(array)

        # Build the matrix of the MHDs
        >>> matrix = tree.build_matrix()
        >>> print(matrix.shape)
        (3, 16, 16)
        """

        # Compute the ECDs and build the matrix of the MHDs
        self._ecdtest.build_matrix(pts)

        # Return the matrix of the MHDs
        return self._ecdtest.matrix
    #------------------------------------------------------------------------#

    #-------------------------   Build the Groups   -------------------------#
    def build_groups(self, eps=0.15):
        """ Build the groups by Regions Growing

        Call the `build_groups` method of the `ecdtest` protected att-
        ribute to form the groups of clusters wrt the MHDs between them.
        See method `build_groups` of ECDTest class for details.

        Parameters
        ----------
        [OPT] eps : float
            The percentage of the maximal value of the MHDs matrix, pro-
            per to each dimension; this value serves as a threshold: if
            a MHD is lower than this threshold, the involved ECDs (thus
            clusters) are considered close.
                :Default: 0.15

        Returns
        -------
        groups : list of sets of ints
            The list of groups (the ECDTest class' `groups` attribute).

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database with 3 Gaussian distributions (regions)
        >>> rng = np.random.default_rng()
        >>> array = np.vstack(
        ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
        >>> database = Database(array, np.arange(len(array)))

        # Instantiate a `SPRADA` object with default parameters
        >>> tree = SPRADA()

        # Build the clusters
        >>> clusters = tree.fit(array)

        # Build the matrix of the MHDs
        >>> matrix = tree.build_matrix()

        # Regroup the clusters by distances
        >>> groups = tree.build_groups()
        >>> print(groups)
        [{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, {14}, {15}]
        """

        # Build the groups of clusters
        self._ecdtest.build_groups(eps)

        # Return the groups
        return self._ecdtest.groups
    #------------------------------------------------------------------------#

    #--------------------------   Fuse Clusters   ---------------------------#
    def _merge_groups_array(self):
        """ Clusters merging for `np.ndarrays` """
        # Build the merged clusters from groups
        clusters = []
        for grp in self._ecdtest.groups:
            # If a group contains only one cluster, process next group (set)
            if len(grp) == 1:
                clusters.append(self._clusters[next(iter(grp))])
            # Merge the clusters as one
            else:
                clusters.append(np.concatenate([self._clusters[i] for i in grp], 0))
        return clusters

    def _merge_groups_cluster(self):
        """ Clusters merging for `Clusters` """
        tags = self._clusters[0].tags                # Cluster's tags
        if self._clusters[0].ndim == 1:
            shape = [None]
        else:
            shape = [None, self._clusters[0].shape[1]]
        # Get the total length of the futurely fused clusters
        sizes = [sum(len(self._clusters[k]) for k in grp)
                 for grp in self._ecdtest.groups]
        # Build the merged clusters from groups
        clusters = []
        for grp, size in zip(self._ecdtest.groups, sizes):
            # If a group contains only one cluster, process next group
            if len(grp) == 1:
                clusters.append(self._clusters[next(iter(grp))])
                continue
            # Data of the futurely merged groups
            shape[0] = size
            value = np.empty(shape, float)
            index = np.empty(size, int)
            patts = np.zeros_like(self._clusters[0].pattern)
            # Merge the clusters (stack their respective data)
            beg, end = 0, 0
            for i in grp:
                end += len(self._clusters[i])
                value[beg:end] = self._clusters[i].value
                index[beg:end] = self._clusters[i].index
                patts += self._clusters[i].pattern
                beg = end
            # Rebuild the a-priori classes, if any
            # (may be empty arrays, hence this special treatment)
            clss = np.hstack([self._clusters[i].classes for i in grp])
            # Merge the clusters as one
            clusters.append(
                dformat.Cluster(value, index, patts/len(grp), tags, clss))
        return clusters

    def merge_groups(self):
        """ Merge the clusters of the groups identified by the ECD Test

        For every set of indexes contained in the `groups` attribute,
        merge the corresponding clusters as one, and create the list of
        the so-merged clusters (one per group). Should be called only
        after the groups have been fully built (`build_groups` method).

        Parameters
        ----------
        Nothing, directly use `self`.

        Returns
        -------
        clusters : list of np.ndarrays or Clusters
            The list of the clusters obtained after completing the merg-
            ing process (driven by the ECD Test groups). The type is the
            the clusters is left unchanged (same as `fit` method).

        Examples
        --------
        >>> import numpy as np
        >>> from clustering.formats import Database

        # Generate a dummy database with 3 Gaussian distributions (regions)
        >>> rng = np.random.default_rng()
        >>> array = np.vstack(
        ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
        ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
        >>> database = Database(array, np.arange(len(array)))

        # Instantiate a `SPRADA` object with default parameters
        >>> tree = SPRADA()

        # Build the clusters
        >>> clusters = tree.fit(array)
        >>> print(len(clusters))
        12

        # Build the matrix of the MHDs
        >>> matrix = tree.build_matrix()

        # Regroup the clusters by distances
        >>> groups = tree.build_groups()

        # Merge the clusters of the groups
        >>> clusters = tree.merge_groups()
        >>> print(len(clusters))
        3
        """

        if self._ecdtest.groups is None:
            raise AssertionError(
                "No groups built, please run `build_groups` first")

        # Return a list of `np.ndarrays`
        if isinstance(self._clusters[0], np.ndarray):
            return self._merge_groups_array()
        # or a list of `Clusters`
        return self._merge_groups_cluster()
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Shorthand Functions                            ##
##############################################################################

#-------------------------   Parameters Checking   --------------------------#
def sprada_params(**params):
    """ Check the parameters for SPRADA clustering

    Take up to 11 keyword arguments that define the required parameters
    for the instantiation and training of the Recursive tree, check if
    all the expected keys are present and fulfill all the missing keys
    with default values. Finally return a fulfilled 11-key dictionary.
    If no argument is fed into the function, set the default values
    within a new 11-key dictionary and return it.

    Refer to the `SPRADA` class' documentation for additional informa-
    tion about the keys (inline keyword parameters).

    Parameters
    ----------
    [OPT] fclt1 : str
        The clustering function for the initial split; can differ
        from `fclt2`. See `get_clust_func` from `cluster`.
            :Default: 'ksom'
    [OPT] fclt1_params : dict
        The parameters for the clustering function for initial split.
        See the functions returned by `get_clust_func` for details.
            :Default: {} (default parameters of `KohonenSOM`)
    [OPT] fclt2 : str
        The clustering function for cluster refinement; can differ
        from `fclt1`. See `get_clust_func` from `cluster`.
            :Default: 'kmeans' 
    [OPT] fclt2_params : dict
        The parameters for the clustering function for refinement.
        See the functions returned by `get_clust_func` for details.
            :Default: {} (default parameters of `KMeans`)
    [OPT] fquant : str
        The quantifying function to estimate cluster's homogeneity. See
        the `get_quant_func` function from the `metrics` module.
            :Default: 'std'
    [OPT] fquant_params : dict
        The parameters for the quantifying function. See the functions
        returned by `get_quant_func` for details.
            :Default: {}
    [OPT] fstats : str
        The statistical function to compute the dynamic threshold as a
        statistic of the quantifiers of the first split (`fclt1`). See
        the `get_stat_func` function from the `metrics` module. Function
        'quantile' is recommended as its value (quantile) `q` can serve
        as a dynamic threshold (see `np.quantile`).
            :Default: 'quantile'
    [OPT] fstats_params: dict
        The parameters for the statistical function. See the functions
        returned by `get_stat_func` for details. Default to the para-
        meters for the `quantile` function.
            :Default: {'q': 0.75} (3rd quantile)
    [OPT] comparison : str
        If the quantifier of a cluster must be lower or greater than
        a certain threshold to be split or not. For instance, for a
        density, the greater, the better; but for the standard devia-
        tion, the lower, the better. Options are {'lower', 'greater'}.
            :Default: 'lower'
    [OPT] pts : int
        The number of points for the ECDs.
            :Default: 100
    [OPT] eps : float
        The threshold for the ECD Test.
            :Default: 0.10

    Returns
    -------
    kparams : 11-key dict
        A dict containing the 11 required keys and their parameters.
        Replace a valid key by the input argument or leave its default
        value if the key is not valid. Return the fulfilled dictionary.

    Examples
    --------
    #--- Get the default parameters
    >>> print(sprada_params())
    {'fclt1': 'ksom',
     'fclt1_params': {},
     'fclt2': 'ksom',
     'fclt2_params': {},
     'fquant': 'std',
     'fquant_params': {},
     'fstats': 'quantile',
     'fstats_params': {'q': 0.75},
     'comparison': 'lower',
     'pts': 100,
     'eps': 0.1}

    #--- Change only the number of clusters and leave the other default params
    >>> print(sprada_params(
    ...     fstats='quantile', fstats_params={'q': 0.25}, eps=0.10))
    {'fclt1': 'ksom',
     'fclt1_params': {},
     'fclt2': 'ksom',
     'fclt2_params': {},
     'fquant': 'std',
     'fquant_params': {},
     'fstats': 'quantile',
     'fstats_params': {'q': 0.25},
     'comparison': 'lower',
     'eps': 0.1,
     'pts': 100}

    #--- Change all the parameters
    # Parameters for `density`
    >>> den_params = {
    ...     'span': 'sphere_span',
    ...     'volume': 'hypersphere',
    ...     'distance': 'euclidean'}

    # Parameters for `quantile`
    >>> stats_params = {'q': 0.25}

    # Parameters for `KohonenSOM`
    >>> ksom_params = {
    ...     'grid_size': (3, 3), 'cluster': True, 'grid': False,
    ...     'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
    ...     'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
    ...     'distance': 'euclidean'}

    # Parameters for `KMeans`
    >>> kmeans_params = {
    ...     'nb_clusters': 2, 'cluster': True,
    ...     'margins': 0.01, 'tmax': 100, 'seed': None,
    ...     'verbose': False, 'distance': 'euclidean'}

    >>> print(sprada_params(
    ...     fclt1='ksom', fclt1_params=ksom_params,
    ...     fclt2='kmeans', fclt2_params=kmeans_params,
    ...     fquant='density', fquant_params=den_params,
    ...     fstats='quantile', fstats_params=stats_params,
    ...     comparison='greater', pts=100, eps=0.10))
    {'fclt1': 'ksom',
     'fclt1_params': {'grid_size': (3, 3),
      'cluster': True,
      'grid': False,
      'margins': 0.01,
      'tmax': 100,
      'nbh_rate_0': 1.0,
      'lrn_rate_0': 0.95,
      'seed': None,
      'verbose': False,
      'distance': 'euclidean'},
     'fclt2': 'kmeans',
     'fclt2_params': {'nb_clusters': 2,
      'cluster': True,
      'margins': 0.01,
      'tmax': 100,
      'seed': None,
      'verbose': False,
      'distance': 'euclidean'},
     'fquant': 'density',
     'fquant_params': {'span': 'sphere_span',
      'volume': 'hypersphere',
      'distance': 'euclidean'},
     'fstats': 'quantile',
     'fstats_params': {'q': 0.25},
     'comparison': 'greater',
     'pts': 100,
     'eps': 0.1}
    """

    # Check `fclt1`, `fclt2`, `fquant` and `fstats` and their parameters
    kparams = recursive_params(**params)

    # Default epsilon
    kparams.update(tls.check_keys(params, {'pts': 100, 'eps': 0.10}))

    return kparams
#----------------------------------------------------------------------------#

#-----------------------   Recursive Tree Building   ------------------------#
def sprada_cluster(data, **params):
    """ Instantiate & train SPRADA on a dataset and cluster it

    Split the dataset into homogeneous clusters. Operate a first decom-
    position, compute the split-or-not limit criterion based on it, and
    refine the obtained clusters by splitting the heterogeneous ones.

    Parameters
    ----------
    data : np.ndarray, Database or Cluster
        The dataset to hierarchically split.

    Other Parameters
    ----------------
    **params : inline keyword arguments, optional
        The parameters passed to the `sprada_params` function for checking.

    Returns
    -------
    clusters : list of np.ndarrays or Clusters
        The so-built tree, as a list of homogeneous clusters.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    # Generate a dummy database
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    >>> database = Database(array, np.arange(len(array)))

    # Split the database with default parameters
    >>> clusters = sprada_cluster(database)
    >>> print(len(clusters))
    3

    # Split the database with some specific parameters
    >>> clusters = sprada_cluster(database,
    ...     fclt1='ksom', fclt2='kmeans',
    ...     fquant='density', fstats='mean',
    ...     comparison='greater', pts=100, eps=0.10)
    >>> print(len(clusters))
    3
    """

    # Check the parameters
    kparams = sprada_params(**params)

    # Initialize the Recursive decomposition
    tree = SPRADA(
        fclt1=kparams['fclt1'], fclt1_params=kparams['fclt1_params'],
        fclt2=kparams['fclt2'], fclt2_params=kparams['fclt2_params'],
        fquant=kparams['fquant'], fquant_params=kparams['fquant_params'],
        fstats=kparams['fstats'], fstats_params=kparams['fstats_params'],
        comparison=kparams['comparison'])

    # Build the tree
    tree.fit(data)

    # Build the matrix of the MHDs between the ECDs
    tree.build_matrix(kparams['pts'])

    # Build the regions of the feature space
    tree.build_groups(kparams['eps'])

    # Merge the clusters (tree refinement)
    return tree.merge_groups()
#----------------------------------------------------------------------------#

##############################################################################
