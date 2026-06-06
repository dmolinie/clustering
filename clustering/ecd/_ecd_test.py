""" Empirical Cumulative Distribution (ECD) Test

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: August 2022
Last revised: May 2026

License: GPLv3
"""

__all__ = ['ECDTest', 'ecd_test']

from clustering.cluster._cluster import get_clust_func
from clustering.ecd._ecds import (
    cumulative_distributions, cdfs_matrix, cdfs_states, cdfs_groups)


##############################################################################
##                          ECD Test (Nb regions)                           ##
##############################################################################

class ECDTest():
    """ Empirical Cumulative Distribution (ECD) Test

    The ECD Test is a fairly intuitive method to estimate the number of
    regions in a feature space, that can be used as objective number of
    clusters to build for parametric clustering algorithm, such as the
    K-Means, Kohonen's SOMs, etc. It uses a split-and-merge approach:
    a dataset is first split into many pieces by any clustering method,
    and the clusters are then regrouped by proximity; the amount of so-
    formed groups is the estimate for the number of regions in the space.
    The ECD Test is greatly inspired from the Kolmogorov-Smirnov Test.

    The procedure is as follows:
     1. Split a dataset into (many) pieces using any clustering method;
        let refer to the clusters as `C = {C_k}_k`.
     2. Compute the Empirical Cumulative Distributions (ECDs) of any
        clusters `C_k`, denoted as `ECD_k{ecd_k}_k`, and compute the
        Modified Hausdorff Distances (MHDs) between any couple of ECDs,
        denoted as `MHD = {mhd_{ij}}_{i,j}`. Find the maximal value in
        the `MHD` set, and uses it as a threshold `tau`.
     3. Regroup any clusters whose MHDs between their respective ECDs
        is lower than `eps x tau`, where `eps` is a dynamic parameter.
        This regrouping is somehow a Region Growing clustering applied
        to the clusters' MHDs.
     4. Return the number of so-formed groups as the estimate for the
        optimal number of regions in the feature space.

    N.B.: sensu stricto, the ECD Test is not a clustering method, it is
        essentially a fashion to estimate the optimal number of clusters
        to build for parametric methods, similarly to the Elbow method.
        For a true clustering method using a split-and-mege approach,
        and that uses the ECD Test, see the `SPRADA` clustering class.

    Constructor
    -----------
    __init__()

    Attributes
    ----------
    clusters : list of np.ndarrays or Clusters, getter only
        The clusters on which operate the ECD Test.
    matrix : np.ndarray, getter only
        The matrix of the MHDs between the ECDs.
    groups : list of sets, getter only
        The groups isolated by the ECD Test.
    regions : int, getter only
        The estimate for the number of regions.

    Methods
    -------
    fit(method='ksom', *, database=None, clusters=None, **clt_params)
        Set the clusters, or build them from a database.
    build_matrix(pts=100)
        Compute the ECDs and build the matrix of the MHDs.
    build_groups(eps=0.15)
        Build the groups by Regions Growing.
    nb_regions()
        Estimate the number of regions.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    # Provide the parameters for KSOM clustering (optional)
    >>> kparams = {
    ...     'grid_size': (3, 3), 'cluster': True, 'grid': False,
    ...     'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
    ...     'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
    ...     'distance': 'euclidean'}

    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))

    #--- Use the ECD Test on a NumPy array
    # Instantiate an `ECDTest` object
    >>> ecd = ECDTest()

    # Cluster the database using the specified method
    >>> ecd.fit(method='ksom', database=array, **kparams)

    # Build the matrix of MHDs
    >>> ecd.build_matrix(50)

    # Build the groups of clusters
    >>> ecd.build_groups(0.15)

    # Retrieve the estimated number of regions in the space
    >>> est_reg = ecd.nb_regions()
    >>> print(est_reg)
    Estimated number of regions: 3

    #--- Use the ECD Test on a Database and build Clusters
    >>> database = Database(array, np.arange(len(array)))

    # Instantiate an `ECDTest` object
    >>> ecd = ECDTest()

    # Cluster the database using the specified method
    >>> ecd.fit(method='ksom', database=database, **kparams)

    # Build the matrix of MHDs
    >>> ecd.build_matrix(50)

    # Build the groups of clusters
    >>> ecd.build_groups(0.15)

    # Retrieve the estimated number of regions in the space
    >>> est_reg = ecd.nb_regions()
    Estimated number of regions: 3
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self):
        """ Instantiate an ECDTest object (constructor)

        Parameters
        ----------
        Nothing, just instantiate the attributes.

        Examples
        --------
        >>> ecd = ECDTest()
        """
        self._clusters = None
        self._matrix = None
        self._groups = None
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def clusters(self):
        """ Get the Clusters on which operate the ECD Test """
        return self._clusters

    @property
    def matrix(self):
        """ Get the matrix of the MHDs between the ECDs """
        return self._matrix

    @property
    def groups(self):
        """ Get the groups isolated by the ECD Test """
        return self._groups

    @property
    def regions(self):
        """ Get the estimated number of regions """
        return self.nb_regions()
    #------------------------------------------------------------------------#

    #----------------------   Set/Build the Clusters   ----------------------#
    def fit(self, method='ksom', *, database=None, clusters=None, **clt_params):
        """ Set the clusters, or build them from a database

        If the clusters are provided, affect them to the `clusters` att-
        ribute; else, if a database is provided, cut it up into pieces
        using the clustering function passed as the `fcluster` argument,
        if any, or use the default SOM, with default parameters.
        If both clusters and database are provided, ignore the latter,
        and use the former directly. If none of them is provided, print
        a warning message and return None (`clusters` attribute not set).

        N.B.: either `clusters` or `database` must be provided.

        IMP: the `clusters`, `database` and `method` are keyword arguments,
            thus they must be explicitly specified.

        Parameters
        ----------
        [OPT] method : str
            The clustering method to use; see the `get_clust_func` function.
                :Default: 'ksom'
        [OPT] database : np.ndarray or Database
            The database to split if `clusters` argument is left empty.
            The issued clusters are affected to the `clusters` attribute.
                :Default: None
        [OPT] clusters : list of np.ndarrays or Clusters
            The clusters to be set to the `clusters` attribute.
                :Default: None

        Other Parameters
        ----------------
        **clt_params : inline keyword arguments, optional
            The parameters for the clustering method, passed to it as
            inline arguments. See the `*_cluster` functions from the
            modules implementing them for details (e.g. `ksom_cluster`
            from `ksom`, `kmeans_cluster` from `kmeans`, etc.).

        Returns
        -------
        None : directly update the  `clusters` attribute.

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

        # Instantiate an `ECDTest` object
        >>> ecd = ECDTest()

        #--- Directly pass the clusters to the ECD Test
        >>> from clustering.ksom import ksom_cluster
        >>> clusters = ksom_cluster(database, grid_size=(3, 3))
        >>> clusters = [clt for clt in clusters if len(clt) != 0]   # or `prune`
        >>> print(len(clusters))
        4

        >>> ecd.fit(clusters=clusters)
        >>> print(len(ecd.clusters))
        4

        #--- Or build the clusters using a dedicated method from the database
        # (empty clusters are automatically removed)
        >>> ecd.fit(method='ksom', database=database, grid_size=(3, 3))
        >>> print(len(ecd.clusters))
        3
        """

        # If provided, use the clusters first
        if clusters is not None:
            self._clusters = clusters

        # Else, split the database using fcluster, if any
        else:

            # If a database if provided (return None else)
            if database is not None:

                # Get the clustering function
                fclt = get_clust_func(method)

                # Operate clustering (remove empty clusters)
                self._clusters = [
                    clust for clust in fclt(database, **clt_params)
                    if clust.size != 0]

            # Neither the clusters nor the database are provided
            else:
                raise AssertionError(
                    "Neither `clusters` nor `database` provided")
    #------------------------------------------------------------------------#

    #---------------------   Matrix of the Distances   ----------------------#
    def build_matrix(self, pts=100):
        """ Compute the ECDs and build the matrix of the MHDs

        Assuming the `clusters` attribute is set (using `fit` method),
        compute the Empirical Cumulative Distributions (ECDs) for each
        of them, for every value in the data range, divided in `pts`,
        evenly spaced subintervals. Once the ECDs are computed, compute
        the Modified Hausdorff Distance (MHD) between any pair of ECDs,
        and gather them all into the `matrix` attribute.
        See functions `cumulative_distributions` from module `metrics`
        and `cdfs_matrix` from the present module for details.

        Parameters
        ----------
        [OPT] pts : int
            The number of subintervals to cut the database/clusters
            data range up into.
                :Default: 100

        Returns
        -------
        None : directly set the KxK `matrix` attribute, with K the number
            of clusters in the `clusters` attribute.

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

        # Instantiate an `ECDTest` object
        >>> ecd = ECDTest()

        # Cluster the database using the specified method
        >>> ecd.fit(method='ksom', database=database, grid_size=(3, 3))

        # Build the matrix of MHDs
        >>> ecd.build_matrix(50)
        >>> print(ecd.matrix.shape)
        (3, 4, 4)
        """

        if self._clusters is None:
            raise AssertionError("No cluster fitted; please run `fit` first")

        # Empirical Cumulative Distributions of every cluster
        vals, cdfs = cumulative_distributions(self._clusters, pts=pts)

        # Matrix of the MHDs between any pair of ECDs
        self._matrix = cdfs_matrix(vals, cdfs, False)
    #------------------------------------------------------------------------#

    #-------------------------   Build the Groups   -------------------------#
    def build_groups(self, eps=0.15):
        """ Build the groups by Regions Growing

        Assuming the matrix of the MHDs is built (using `build_matrix`
        method), gather the ECDs (and thus clusters) by Region Growing
        clustering, using the `eps` argument as neighborhood distance.
        See functions `cdfs_states` and `cdfs_groups` from the present
        module for details.

        Parameters
        ----------
        [OPT] eps : float
            The percentage of the maximal value of the MHDs matrix,
            proper to each dimension; this value serves a threshold.
            If a MHD is lower than this threshold, the involved ECDs
            (and thus clusters) are considered as close.
                :Default: 0.15

        Returns
        -------
        None : directly set the `groups` attribute.

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

        # Instantiate an `ECDTest` object
        >>> ecd = ECDTest()

        # Cluster the database using the specified method
        >>> ecd.fit(method='ksom', database=database, grid_size=(3, 3))

        # Build the matrix of MHDs
        >>> ecd.build_matrix(50)

        # Build the groups of clusters
        >>> ecd.build_groups(0.15)
        >>> print(ecd.groups)
        [{0}, {1, 3, 4}, {2}]
        """

        if self._matrix is None:
            raise AssertionError(
                "No ECD Matrix; please run `build_matrix` first")

        # States of the closeness of every pair of ECDs
        states = cdfs_states(self._matrix, eps)[1]

        # Groups of clusters (~regions)
        self._groups = cdfs_groups(states, len(self._clusters))
    #------------------------------------------------------------------------#

    #------------------------   Number of Regions   -------------------------#
    def nb_regions(self):
        """ Estimate the number of regions 

        Function alias of the `regions` attribute (called by the corres-
        ponding class property).

        Parameters
        ----------
        Nothing, directly use `self`.

        Returns
        -------
        est : int
            The estimated number of regions in the space.

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

        # Instantiate an `ECDTest` object
        >>> ecd = ECDTest()

        # Cluster the database using the specified method
        >>> ecd.fit(method='ksom', database=database, grid_size=(3, 3))

        # Build the matrix of MHDs
        >>> ecd.build_matrix(50)

        # Build the groups of clusters
        >>> ecd.build_groups(0.15)

        # Retrieve the estimated number of regions in the space
        >>> est_reg = ecd.nb_regions()
        >>> print(est_reg)
        Estimated number of regions: 3
        """

        if self._groups is None:
            raise AssertionError(
                "No groups built; please run `build_groups` first")

        print("Estimated number of regions:", len(self._groups))

        return len(self._groups)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                       ECD Test (as lone function)                        ##
##############################################################################

def ecd_test(method='ksom', pts=100, eps=0.15, *,
    database=None, clusters=None, **clt_params):
    """ Estimate the number of regions in a feature space

    Perform the ECD Test either on a set of clusters passed as argument,
    or issued by cutting the database up using the clustering function
    passed as arguments. To do so, instantiate an ECDTest class object,
    and use its methods:
     - `fit` to set the clusters or build them from the database;
     - `compute_ecd` to compute the ECDs and build the MHDs matrix;
     - `build_groups` to fuse the ECDs (thus clusters) by Region Growing;
     - `nb_regions` to get the number of regions obtained.
    This function is a shortcut, standalone version of the ECDTest class.

    Refer to the `ECDTest` class' for details, as this function is a
    shorthand version of it.

    N.B.: either `clusters` or `database` must be provided.

    IMP: the `clusters`, `database` and `method` are keyword arguments,
        thus they must be explicitly specified.

    Parameters
    ----------
    [OPT] method : str
        The clustering method to use; see the `get_clust_func` function.
            :Default: 'ksom'
    [OPT] pts : int
        The number of subintervals to cut the database/clusters data
        range up into.
            :Default: 100
    [OPT] eps : float
        The percentage of the maximal value of the MHDs matrix, proper
        to each dimension; this value serves a threshold. If a MHD is
        lower than this threshold, the involved ECDs (and thus clusters)
        are considered as close.
            :Default: 0.15
    [OPT] database : np.ndarray or Database
        The database to split if `clusters` argument left empty. The
        issued clusters are used by to operate the ECD Test.
            :Default: None
    [OPT] clusters : list of np.ndarrays or Clusters
        The clusters on which operate the ECD Test.
            :Default: None

    Other Parameters
    ----------------
    **clt_params : inline keyword arguments, optional
        The parameters for the clustering method, passed to it as inline
        arguments. See the `*_cluster` functions from the modules imple-
        menting them for details (e.g. `ksom_cluster` from `ksom`,
        `kmeans_cluster` from `kmeans`, etc.).

    Returns
    -------
    est : int
        The number of regions of the feature space (candidate for the
        number of clusters to be set as meta-parameter).

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    # Provide the parameters for KSOM clustering (optional)
    >>> kparams = {
    ...     'grid_size': (3, 3), 'cluster': True, 'grid': False,
    ...     'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
    ...     'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
    ...     'distance': 'euclidean'}

    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))

    #--- Use the ECD Test on a NumPy array
    >>> ecd_test('ksom', 100, 0.15, database=array, **kparams)
    Estimated number of regions: 3

    #--- Use the ECD Test on a Database and build Clusters
    >>> database = Database(array, np.arange(len(array)))
    >>> ecd_test('ksom', 100, 0.15, database=database, **kparams)
    Estimated number of regions: 3
    """

    # Instantiate an ECDTest object
    ecdtest = ECDTest()

    # Set the clusters if provided, or operate clustering else
    ecdtest.fit(method=method, clusters=clusters, database=database, **clt_params)

    # Compute the clusters' ECDs, and build the matrix of their MHDs
    ecdtest.build_matrix(pts=pts)

    # Compute the state matrix, and build the groups by Region Growing
    ecdtest.build_groups(eps=eps)

    # Return the number of groups detected (final estimate)
    return ecdtest.nb_regions()

##############################################################################
