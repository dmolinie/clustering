""" Recursive decomposition of a dataset

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['Recursive', 'recursive_params', 'recursive_cluster']

import numpy as np

import clustering.tools as tls
import clustering.formats as dformat
from clustering.metrics import get_quant_func, get_stat_func
from clustering.cluster import get_clust_func


##############################################################################
##                         Recursive Decomposition                          ##
##############################################################################

class Recursive():
    """ Recursive Decomposition

    Allow to recursively split a dataset. The dataset is first split
    using a regular unsupervised clustering method (K-Means, KSOM, etc.)
    and the obtained clusters are then refined. This refinement is driven
    by a homogeneity criterion: if the data in a cluster do not satisfy
    that criterion, the cluster is split one more time; this procedure
    is then applied to the sub-clusters. The decomposition stops when
    any of the clusters satisfies the split-or-not criterion (they are
    homogeneous), or when a cluster contains only one data.

    Let refer to `dataset` as a dataset to cluster, `fclt1` and `fclt2`
    two clustering functions, `fquant` any quantifying function that
    characterizes a group of data and `fstats` any statistical function.

    The split procedure is a follows:
     1. Cluster `dataset` using `fclt1`; let refer to the so-issued
        clusters as `C = {C_k}_k`.
     2. Compute the quantifier for any group in set `C` using `fquant`;
        let refer to the set of these quantifiers as `Q = {q_k}_k`.
     3. Compute a statistic on set `Q` using `fstats`; this statistic
        serves as a threshold to estimate the compactness or homogeneity
        of the clusters: let refer to this threshold as `tau`.
     4. For any cluster in `C`, if its quantifier returned by `fquant`
        is lower (or greater, depending on the metric) than threshold
        `tau`, split this group with `fclt2`, and append the so-built
        sub-clusters to set `C`.
     5. If the quantifiers of any cluster in `C` is lower (or greater)
        than `tau`, or if they contain only data, stop the splitting;
        else, go back to step 4.

    The threshold `tau` of step 3 is computed on the initial clusters
    only (those issued by `fclt1`), it is never updated. Conceptually,
    it is motivated by the following assumption: in a clustering, in
    particular when they are many groups, some are likely homogeneous
    or compact, thus their quantifiers can serve as a reference, i.e.
    as a threshold. Therefore, the statistical function `fstats` aims
    to get a dynamic threshold, whose value depends on the dataset.

    In particular, if using the `quantile` function, its parameter `q`
    can be set so as to dynamically select how many clusters from the
    initial split as considered compact, and how many must be split.
    For instance, if `q` is set to 0.75, it means that 75% of the clus-
    ters are considered compact and not to be split, whilst the 25%
    remaining ones will be split again.

    N.B.: Quantifiers may differ; for some, the lower the better (e.g.
        standard deviation), whilst for others, the greater the better
        (e.g. hyper-density). To handle this, the `comparison` argument
        allows to select the comparison strategy. For instance, in the
        above example with the quantile, if `q` is set to 0.75 with the
        `lower` strategy, 75% of the clusters will be considered compact,
        whereas they will be considered not compact and to be split with
        the `greater` strategy.

    Constructor
    -----------
    __init__(
        fclt1='ksom', fclt2='kmeans', fquant='std',
        fstats='quantile', comparison='lower', **params)

    Attributes
    ----------
	fclt1 : str, getter & setter
    	The clustering function for initial splitting.
	fclt1_params : dict, getter & setter
    	The set of parameters for the clustering function `fclt1`.
	fclt2 : str, getter & setter
    	The clustering function for recursive tree refinement.
	fclt2_params : dict, getter & setter
    	The set of parameters for the clustering function `fclt2`.
	fquant : str, getter & setter
    	The quantification function to characterize the clusters.
	fquant_params : dict, getter & setter
    	The set of parameters for the quantification function `fquant`.
	fstats : str, getter & setter
    	The statistical function to build the dynamic threshold.
	fstats_params : dict, getter & setter
    	The set of parameters for the statistical function `fstats`.
    comparison : str, getter & setter
        The comparison strategy for the quantifiers.
	limit : float, getter & setter
    	The max value to consider a cluster well-built.
	clusters : list of Clusters, getter only
    	The clusters built.

    Methods
    -------
    fit(dataset)
        Build the tree recursively.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database


    #--- Apply the Recursive decomposition to a NumPy array
    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))

    # Instantiate a `Recursive` object with default parameters
    >>> tree = Recursive()

    # Build the clusters
    >>> clusters = tree.fit(array)

    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>
    >>> print(len(clusters))
    25


    #--- Apply the Recursive decomposition to a Database and build Clusters
    >>> database = Database(array, np.arange(len(array)))

    # Instantiate a `Recursive` object with default parameters
    >>> tree = Recursive()

    # Build the clusters
    >>> clusters = tree.fit(database)

    >>> print(type(clusters[0]))
    <class 'clustering.formats._database.Cluster'>
    >>> print(len(clusters))
    24


    #--- Compare the comparison strategies
    # Std is 'the lower the better'
    # 1st quantile with 'greater': 25% considered compact (75% to split)
    >>> tree = Recursive(
    ...     fquant='std', comparison='lower',
    ...     fstats='quantile', fstats_params={'q': 0.25})
    >>> clusters = tree.fit(database)
    >>> print(len(clusters))
    35

    # Std is 'the lower the better'
    # 3rd quantile with 'greater': 75% considered compact (25% to split)
    >>> tree = Recursive(
    ...     fquant='std', comparison='lower',
    ...     fstats='quantile', fstats_params={'q': 0.75})
    >>> clusters = tree.fit(database)
    >>> print(len(clusters))
    17

    # Density is 'the greater the better'
    # 3rd quantile with 'greater': 25% considered compact (75% to split)
    >>> tree = Recursive(
    ...     fquant='density', comparison='greater',
    ...     fstats='quantile', fstats_params={'q': 0.75})
    >>> clusters = tree.fit(database)
    >>> print(len(clusters))
    43

    # Density is 'the greater the better'
    # 1st quantile with 'greater': 25% considered compact (75% to split)
    >>> tree = Recursive(
    ...     fquant='density', comparison='greater',
    ...     fstats='quantile', fstats_params={'q': 0.25})
    >>> clusters = tree.fit(database)
    >>> print(len(clusters))
    13
    """

    #---------------------------   Constructor   ----------------------------#
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(self,
        fclt1='ksom', fclt2='kmeans', fquant='std',
        fstats='quantile', comparison='lower', **params):
        """ Instantiate a Recursive object (constructor)

        Take the clustering function `fclt1` to split the whole dataset,
        the function `fclt2` to split the subsequent clusters, the quan-
        tifying function `fquant` to characterize the groups of data and
        the statistical function `fstats` to compute the threshold from
        the initial clustering, and instantiate a `Recursive` object.

        The sets of parameters of any of the above functions can be pas-
        sed as inline keyword arguments. In addition, the `comparison`
        argument specifies the kind of quantifier: either the lower the
        better, or the greater the better.

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
        >>> tree = Recursive()

        #--- Use specific functions with their default parameters
        >>> tree = Recursive(
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
        >>> tree = Recursive(
        ...     fclt1='ksom', fclt1_params=ksom_params,
        ...     fclt2='kmeans', fclt2_params=kmeans_params,
        ...     fquant='density', fquant_params=den_params,
        ...     fstats='quantile', fstats_params=stats_params,
        ...     comparison='greater')
        """

        # Store the function names (for the getters)
        self._fnames = {'fclt1': fclt1, 'fclt2': fclt2,
                        'fquant': fquant, 'fstats': fstats}

        # Clustering function for initial split
        self._fclt1 = [
            get_clust_func(fclt1),
            params['fclt1_params'] if 'fclt1_params' in params else {}]

        # Clustering function for building the tree
        self._fclt2 = [
            get_clust_func(fclt2),
            params['fclt2_params'] if 'fclt2_params' in params else {}]

        # Quantifying function
        self._fquant = [
            get_quant_func(fquant),
            params['fquant_params'] if 'fquant_params' in params else {}]

        # Statistical function to compute the `limit` threshold
        self._fstats = [
            get_stat_func(fstats),
            params['fstats_params'] if 'fstats_params' in params else
                ({'q': 0.75} if comparison.lower() == 'lower' else {'q': 0.25})]

        # Quantifier threshold and comparison strategy
        self._params = {'limit': None, 'comparison': comparison.lower()}

        # List of final clusters
        self._clusters = []
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def fclt1(self):
        """ Get the first clustering function """
        return self._fnames['fclt1']

    @fclt1.setter
    def fclt1(self, fclt1):
        """ Set the first clustering function """
        self._fclt1[0] = get_clust_func(fclt1)
        self._fnames['fclt1'] = fclt1

    @property
    def fclt1_params(self):
        """ Get the parameter dict for the first clustering function """
        return self._fclt1[1]

    @fclt1_params.setter
    def fclt1_params(self, params):
        """ Set the parameter dict for the first clustering function """
        self._fclt1[1] = params

    @property
    def fclt2(self):
        """ Get the second clustering function """
        return self._fnames['fclt2']

    @fclt2.setter
    def fclt2(self, fclt2):
        """ Set the second clustering function """
        self._fclt2[0] = get_clust_func(fclt2)
        self._fnames['fclt2'] = fclt2

    @property
    def fclt2_params(self):
        """ Get the parameter dict for the second clustering function """
        return self._fclt2[1]

    @fclt2_params.setter
    def fclt2_params(self, params):
        """ Set the parameter dict for the second clustering function """
        self._fclt2[1] = params

    @property
    def fquant(self):
        """ Get the quantification function """
        return self._fnames['fquant']

    @fquant.setter
    def fquant(self, fquant):
        """ Set the quantification function """
        self._fquant[0] = get_quant_func(fquant)
        self._fnames['fquant'] = fquant

    @property
    def fquant_params(self):
        """ Get the parameter dict for the quantification function """
        return self._fquant[1]

    @fquant_params.setter
    def fquant_params(self, params):
        """ Set the parameter dict for the quantification function """
        self._fquant[1] = params

    @property
    def fstats(self):
        """ Get the statistical function """
        return self._fnames['fstats']

    @fstats.setter
    def fstats(self, fstats):
        """ Set the statistical function """
        self._fstats[0] = get_stat_func(fstats)
        self._fnames['fstats'] = fstats

    @property
    def fstats_params(self):
        """ Get the parameter dict for the statistical function """
        return self._fstats[1]

    @fstats_params.setter
    def fstats_params(self, params):
        """ Set the parameter dict for the statistical function """
        self._fstats[1] = params

    @property
    def comparison(self):
        """ Get the comparison strategy """
        return self._params['comparison']

    @comparison.setter
    def comparison(self, comparison):
        """ Set the comparison strategy """
        self._params['comparison'] = comparison

    @property
    def limit(self):
        """ Get the max value to consider a cluster well-built """
        return self._params['limit']

    @limit.setter
    def limit(self, limit):
        """ Set the max value to consider a cluster well-built """
        self._params['limit'] = limit

    @property
    def clusters(self):
        """ Get the built clusters """
        return self._clusters
    #------------------------------------------------------------------------#

    #-------------   Cluster's Homogeneity Quantifier's Limit   -------------#
    def _split_limit(self, clusters):
        """ Compute the split-or-not criterion's limit

        Take a list of clusters and compute their homogeneity; once these
        measures are computed, a statistical value of them is used as the
        split-or-not criterion's limit, used for building the Recursive
        tree. The homogeneity and statistical functions are specified in
        the `params` attribute.

        Parameters
        ----------
        clusters : np.ndarray, Database or Cluster
            List of the databases to analyze.

        Returns
        -------
        None : directly set the `limit` attribute.
        """

        # Compute the quantifier for each cluster
        quantifiers = []
        for clust in clusters:
            if len(clust) > 1:
                quantifiers.append(
                    np.mean(self._fquant[0](clust[:], **self._fquant[1])))

        # Compute the clusters' quantifiers' statistics
        self._params['limit'] = self._fstats[0](quantifiers, **self._fstats[1])
    #------------------------------------------------------------------------#

    #-------------------   Cluster's Splitting Decision   -------------------#
    def _split(self, cluster):
        """ Cluster's split-or-not decision

        Take a cluster to analyze and decide if it should be split. Its
        homogeneity is computed according to a homogeneity measure, and
        is compared to the `limit` attribute (see `split_limit` method).

        Parameters
        ----------
        cluster : np.ndarray, Database or Cluster
            Cluster to analyze and to decide if it should be split.

        Returns
        -------
        split : bool
            The decision to split or not the cluster. If it is considered
            compact, the return value is False (no split); on the contrary,
            it is True (the cluster should be split).
        """

        # Skip empty clusters
        if len(cluster) > 1:

            # Compare the homogeneity to the acceptable limit
            quant = np.mean(self._fquant[0](cluster[:], **self._fquant[1]))

            # The lower, the better (no split if quant < limit)
            if self._params['comparison'] == 'lower':
                return self._params['limit'] < quant
            # The greater, the better (no split if limit < quant)
            return quant < self._params['limit']

        # No split by default
        return False
    #------------------------------------------------------------------------#

    #-------------------   Recursive Dataset Splitting   --------------------#
    def _split_clusters(self, clusters):
        """ Recursively split the clusters

        Take a list of clusters and split each of them again and again until
        satisfying any criterion. For each cluster, the decision to split it
        is provided by the `_split` method and is used as criterion. If
        the decision to split is taken, the corresponding cluster is split
        by the `fcluster` param function and the returned cut up clusters
        (as a list) are recursively fed into the current method; on the
        contrary, it is added to the `clusters` list attribute. The split
        process stops when every sub-cluster satisfies the split criterion.

        Parameters
        ----------
        clusters : list of Cluster objects
            List of the Clusters to split hierarchically.

        Returns
        -------
        None : directly set the `clusters`.
        """

        # Run through all the input clusters
        for clust in clusters:

            if len(clust) == 1:
                self._clusters.append(clust)
                continue

            # Skip empty clusters
            if len(clust) != 0:

                # Is the cluster homogeneous?
                if not self._split(clust):
                    self._clusters.append(clust)

                # If not homogeneous, split the cluster
                else:
                    subclusts = self._fclt2[0](clust, **self._fclt2[1])
                    # Check that the subclusters have changed, and stop if not
                    stop = all(len(sclt) == len(clt)
                               for sclt, clt in zip(subclusts, clusters))
                    if not stop:
                        self._split_clusters(subclusts)
                    else:
                        self._clusters.append(clust)
    #------------------------------------------------------------------------#

    #--------------------------   Tree Building   ---------------------------#
    def fit(self, dataset):
        """ Build the tree recursively

        Split the dataset into homogeneous clusters. Operate a first
        decomposition, compute the split-or-not limit criterion based
        on it, and eventually refine the obtained clusters by splitting
        the heterogeneous ones.

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
        >>> tree = Recursive(fquant='std', comparison='lower')

        # Build the clusters (set `comparison` to 'lower',
        # since for `std`, the lower, the better
        >>> clusters = tree.fit(database)
        >>> print(len(clusters))
        17

        #--- Instantiate a `Recursive` object with `density` quantifier
        >>> tree = Recursive(fquant='density', comparison='greater')

        # Build the clusters (set `comparison` to 'greater',
        # since for `density`, the greater, the better
        >>> clusters = tree.fit(database)
        >>> print(len(clusters))
        7
        """

        # Check data type and format
        dataset = dformat.check_data(dataset)

        # Initial dataset decomposition
        clusters = self._fclt1[0](dataset, **self._fclt1[1])

        # Split-or-not criterion
        self._split_limit(clusters)

        # Recursive decomposition of the clusters
        self._split_clusters(clusters)

        # Return the list of the homogeneous clusters
        return self._clusters
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Shorthand functions                            ##
##############################################################################

#-------------------------   Parameters Checking   --------------------------#
def recursive_params(**params):
    """ Check the parameters for Recursive clustering

    Take up to 9 keyword arguments that define the required parameters
    for the instantiation and training of the Recursive tree, check if
    all the expected keys are present and fulfill all the missing keys
    with default values. Finally return a fulfilled 9-key dictionary.
    If no argument is fed into the function, set the default values
    within a new 9-key dictionary and return it.

    Refer to the `Recursive` class' documentation for additional infor-
    mation about the keys (inline keyword parameters).

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

    Returns
    -------
    kparams : 9-key dict
        A dict containing the 9 required functions and their parameters.
        Replace a valid key by the input argument or leave its default
        value if the key is not valid. Return the fulfilled dictionary.

    Examples
    --------
    #--- Get the default parameters
    >>> print(recursive_params())
    {'fclt1': 'ksom',
     'fclt1_params': {},
     'fclt2': 'kmeans',
     'fclt2_params': {},
     'fquant': 'std',
     'fquant_params': {},
     'fstats': 'quantile',
     'fstats_params': {'q': 0.75},
     'comparison': 'lower'}

    #--- Change only the number of clusters and leave the other default params
    >>> print(recursive_params(fstats='mean'))
    {'fclt1': 'ksom',
     'fclt1_params': {},
     'fclt2': 'kmeans',
     'fclt2_params': {},
     'fquant': 'std',
     'fquant_params': {},
     'fstats': 'mean',
     'fstats_params': {},
     'comparison': 'lower'}

    #--- Change all the parameters
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

    # Parameters for `density`
    >>> den_params = {
    ...     'span': 'sphere_span',
    ...     'volume': 'hypersphere',
    ...     'distance': 'euclidean'}

    # Parameters for `quantile`
    >>> stats_params = {'q': 0.75}

    # Check the parameters
    >>> print(recursive_params(
    ...     fclt1='ksom', fclt1_params=ksom_params,
    ...     fclt2='kmeans', fclt2_params=kmeans_params,
    ...     fquant='density', fquant_params=den_params,
    ...     fstats='quantile', fstats_params=stats_params,
    ...     comparison='greater'))
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
     'fstats_params': {'q': 0.75},
     'comparison': 'greater'}
    """
    # Default parameters
    tparams = {
        'fclt1': 'ksom', 'fclt1_params': {},
        'fclt2': 'ksom', 'fclt2_params': {},
        'fquant': 'std', 'fquant_params': {},
        'fstats': 'quantile', 'fstats_params': {'q': 0.75},
        'comparison': 'lower'}
    # Reset `fstats_params` if it was provided
    if ('fstats_params' in params
        or ('fstats' in params and params['fstats'] != 'quantile')):
        tparams['fstats_params'] = {}
    # Check the parameter's keys
    return tls.check_keys(params, tparams)
#----------------------------------------------------------------------------#

#-----------------------   Recursive Tree Building   ------------------------#
def recursive_cluster(data, **params):
    """ Instantiate & train a Recursive Tree on a dataset and cluster it

    Split the dataset into homogeneous clusters. Operate a first decom-
    position, compute the split-or-not limit criterion based on it, and
    refine the obtained clusters by splitting the heterogeneous ones.

    Refer to the `Recursive` class' for details, as this function is a
    shorthand version of it.

    Parameters
    ----------
    data : np.ndarray, Database or Cluster
        The dataset to hierarchically split.

    Other Parameters
    ----------------
    **params : inline keyword arguments, optional
        The parameters passed to the `recursive_params` function for checking.

    Returns
    -------
    clusters : list of np.ndarrays or Clusters
        The so-built tree, as a list of homogeneous clusters.

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

    # Split the database with default parameters
    >>> clusters = recursive_cluster(database)
    >>> print(len(clusters))
    103

    # Split the database with some specific parameters
    >>> clusters = recursive_cluster(database,
    ...     fclt1='ksom', fclt2='kmeans',
    ...     fquant='density', fstats='mean',
    ...     comparison='greater')
    >>> print(len(clusters))
    31
    """

    # Check the parameters
    tparams = recursive_params(**params)

    # Initialize the Recursive decomposition
    tree = Recursive(
        fclt1=tparams['fclt1'], fclt1_params=tparams['fclt1_params'],
        fclt2=tparams['fclt2'], fclt2_params=tparams['fclt2_params'],
        fquant=tparams['fquant'], fquant_params=tparams['fquant_params'],
        fstats=tparams['fstats'], fstats_params=tparams['fstats_params'],
        comparison=tparams['comparison'])

    # Build the tree
    return tree.fit(data)
#----------------------------------------------------------------------------#

##############################################################################
