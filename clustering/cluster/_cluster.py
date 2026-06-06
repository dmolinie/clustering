""" Shorthand function to cluster data

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'prune', 'merge', 'sort', 'get_clust_func', 'cluster', 'rebuild_idx']

import numpy as np

import clustering.metrics as mts


##############################################################################
##                            Clusters' cleaning                            ##
##############################################################################

#------------------------   Empty Clusters Pruning   ------------------------#
def prune(clusters):
    """ Remove the empty clusters from a list

    Parameters
    ----------
    clusters : list of Cluster objects
        The list of clusters for which to remove the empty ones.

    Returns
    -------
    clusters_pruned : list of Cluster objects
        The list of all the non-empty clusters of `clusters`.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(4, 4))
    >>> print(len(clusters), '--', [len(clt) for clt in clusters])
    16 -- [3, 2, 2, 1, 0, 1, 0, 1, 1, 2, 0, 2, 2, 1, 1, 1]

    # Prune the empty clusters
    >>> prune(clusters)
    >>> print(len(clusters), '--', [len(clt) for clt in clusters])
    13 -- [3, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1]
    """
    clusters[:] = [clust for clust in clusters if clust.size != 0]
#----------------------------------------------------------------------------#

#---------------------------   Clusters Merging   ---------------------------#
def merge(clusters, pct=0.01, linkage='ward', distance='euclidean'):
    """ Merge the closest clusters to refine splitting

    Compute the linkage distance between any pair of clusters in a list,
    find the maximal value, multiply it by `pct` arg. to get a merge-or-
    not criterion, and merge any 2-cluster set whose linkage distance is
    higher than this limit. Redo this until no more 2-cluster set left
    with a distance higher than this threshold.

    Parameters
    ----------
    clusters : list of Cluster objects
        The list of clusters to analyze (~ data clusters).
    [OPT] pct : float
        Percentage of the max linkage distance within the dataset.
        Compute the maximal distance, multiply it by the pct arg.,
        and use this value as the merge-or-not criterion's limit.
            :Default: 0.01 (1%)
    [OPT] linkage : str
        The linkage name; see the `get_link_func` function from the
        `metrics` module for details.
            :Default: `ward`
    [OPT] distance : str
        The distance name to use for linkage; see the `get_dist_func`
        function from the `metrics` module for details.
            :Default: `euclidean`

    Returns
    -------
    None : the clusters are directly merged with each other.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(3, 3))
    >>> print(len(clusters), '--', [len(clt) for clt in clusters])
    9 -- [3, 3, 1, 2, 1, 2, 2, 4, 2]

    # Merge the closest clusters
    >>> prune(clusters)
    >>> merge(clusters, 0.01)
    >>> print(len(clusters), '--', [len(clt) for clt in clusters])
    4 -- [3, 3, 4, 10]
    """

    if len(clusters) < 2:
        raise AssertionError("One cluster only in `clusters`, nothing to merge")

    # Compute the linkage distances
    links = mts.linkages(clusters, linkage, distance)

    # Merge-or-not criterion
    lim = pct * max(links)

    # Merge the clusters
    while len(clusters) > 1 and not all(links > lim):

        # Clusters to be removed once merged
        pos1 = 0
        rmv = [[], []]

        for i, clti in enumerate(clusters):
            pos = pos1
            for j, cltj in enumerate(clusters[i+1:]):

                classes = [clti.classes, cltj.classes]
                if classes[0] is not None and classes[1] is not None:
                    classes = classes[0] + classes[1]
                else:
                    classes = None

                j += i+1
                # Clusters merging
                if links[pos] < lim and j not in rmv[1]:
                    clusters[j].set_cluster(
                        np.concatenate((clti.value, cltj.value), 0),
                        np.hstack((clti.index, cltj.index)),
                        (clti.pattern + cltj.pattern) / 2.,
                        clti.tags,
                        classes)

                    rmv[0].append(i)        # Remove clt i (added to clt j)
                    rmv[1].append(j)        # Stop considering clt j
                    break                   # Stop comp. clusters to clt i

                pos += 1                    # Next cluster to compare
            pos1 += len(clusters) - (i+1)   # Next clt i (never seen again)

        # Remove the added clusters (those added to some other ones)
        for rem in reversed(rmv[0]):
            del clusters[rem]

        # Update the linkage distances with the new clusters
        links = mts.linkages(clusters, linkage, distance)
#----------------------------------------------------------------------------#

#----------------------------   Sort Clusters   -----------------------------#
def sort(clusters):
    """ Sort the clusters of a list by size

    Parameters
    ----------
    clusters : list of Cluster objects
        The list of the clusters to sort.

    Returns
    -------
    clusters_sorted : list of Cluster objects
        The items of `clusters`, sorted by size in decreasing order.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(2, 2))
    >>> print(len(clusters), '--', [len(clt) for clt in clusters])
    4 -- [3, 7, 6, 4]

    # Sort the clusters by size
    >>> clusters = sort(clusters)
    >>> print(len(clusters), '--', [len(clt) for clt in clusters])
    4 -- [7, 6, 4, 3]
    """

    size = np.empty(len(clusters), int)
    for i, clust in enumerate(clusters):
        size[i] = len(clust)
    ords = size.argsort()

    clusts = []
    for i in ords[::-1]:
        clusts.append(clusters[i])

    return clusts
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                       Clustering Method Selection                        ##
##############################################################################

#-----------------------   Get Clustering Function   ------------------------#
def get_clust_func(method='ksom'):
    """ Get the reference to a specified clustering function

    Take the name of a clustering method and return its corresponding
    function; return the `ksom_cluster` function for the `kohonen_som`
    as default. Every function has signature:
        fclt(data, **params)
    where `data` is the dataset to cluster and `params` are the para-
    meters of the function / clustering method.

    Parameters
    ----------
    [OPT] method : str
        The name of the clustering method to use; options are:
          - {'kmeans', 'kms', 'km'}: `kmeans_cluster` from `kmeans`;
          - {'kernel_kmeans', 'kkmeans', 'kkms', 'kkm'}:
                `kkmeans_cluster` from `kmeans`;
          - {'kohonen_som', 'ksom', 'som'}: `ksom_cluster` from `ksom`;
          - {'bilevel_som', 'bisom', 'bsom'}: `bsom_cluster` from `ksom`;
          - 'dbscan': `dbscan_cluster` from `region_growing`;
          - 'optics': `optics_cluster` from `region_growing`;
          - 'recursive': `recursive_cluster` from `sprada`.
          - 'sprada': `sprada_cluster` from `sprada`.
        :Default: 'ksom'

    Returns
    -------
    fclt : reference to a function
        The clustering function; has signature: 
            `fclt(data, *args)`

    Examples
    --------
    # KMeans clustering
    >>> print(get_clust_func('kmeans'))
    <function kmeans_cluster at 0x7e8b82b2af00>

    # Kohonen SOMs
    >>> print(get_clust_func('ksom'))
    <function ksom_cluster at 0x7e8b82b29a60>

    # SPRADA method
    >>> print(get_clust_func('sprada'))
    <function sprada_cluster at 0x7e8b82b699b0>
    """
    # pylint: disable=too-many-return-statements, import-outside-toplevel

    # Get the linkage function
    clt = method.lower()
    if clt in ('kmeans', 'kms', 'km'):
        from clustering.kmeans import kmeans_cluster
        return kmeans_cluster
    if clt in ('kernel_kmeans', 'kkmeans', 'kkms', 'kkm'):
        from clustering.kmeans import kkmeans_cluster
        return kkmeans_cluster
    if clt in ('kohonen_som', 'ksom', 'som'):
        from clustering.ksom import ksom_cluster
        return ksom_cluster
    if clt in ('bilevel_som', 'bisom', 'bsom'):
        from clustering.ksom import bsom_cluster
        return bsom_cluster
    if clt == 'dbscan':
        from clustering.region_growing import dbscan_cluster
        return dbscan_cluster
    if clt == 'optics':
        from clustering.region_growing import optics_cluster
        return optics_cluster
    if clt == 'recursive':
        # pylint: disable-next=cyclic-import # Not a true cyclic import
        from clustering.sprada import recursive_cluster
        return recursive_cluster
    if clt == 'sprada':
        # pylint: disable-next=cyclic-import # Not a true cyclic import
        from clustering.sprada import sprada_cluster
        return sprada_cluster

    # Raise an error if the linkage is invalid
    raise ValueError(f"Wrong value `{method}` for `method`; options are:\n"
        + "\t{'ksom', 'bsom', 'kmeans', 'kernel_kmeans',"
        + " 'dbscan', 'optics', 'recursive', 'sprada'}")
#----------------------------------------------------------------------------#

#-------------------------   Clustering Shorthand   -------------------------#
def cluster(data, method='ksom', fuse=0., linkage='ward', **clt_params):
    """ Split a a dataset using a given clustering method

    Take the dataset to cluster and the method to use for the splitting
    and its parameters as inline arguments. Then, apply the clustering
    function to the dataset and return the so-built clusters. Refine
    the clusters by removing the empty ones from the list, merging the
    closest ones together, and sort them by size in decreasing order.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to cluster (np matrix + list of str).
    [OPT] method : str
        The clustering method to use; see the `get_clust_func` function
        from the `processing` module.
            :Default: 'ksom'
    [OPT] fuse : float
        If not zero, fuse the clusters using the `merge` function from
        the `processing` module; pass this value as the `pct` argument
        for this function, which is the percentage of the max linkage
        distance within the dataset for the merging of the clusters.
        If set to 0., skip the merging step.
            :Default: 0. (no clustering merging)
    [OPT] linkage : str
        The linkage name; see the `get_link_func` function from the
        `metrics` module for details. Use only if `fuse` is not zero.
            :Default: `ward`

    Other Parameters
    ----------------
    **clt_params : inline keyword arguments, optional
        The parameters for the clustering method, passed to it as inline
        arguments. See the `*_cluster` functions from the modules imple-
        menting them for details (e.g. `ksom_cluster` from `ksom`,
        `kmeans_cluster` from `kmeans`, etc.).

    Returns
    -------
    clusters : list of np.ndarrays or Clusters
        The dataset split into several clusters, organized as a list.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database

    # Generate a dummy database
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

    # Cluster the database with the KMeans
    >>> kparams = {
    ...     'nb_clusters': 2, 'cluster': True,
    ...     'margins': 0.01, 'tmax': 100, 'seed': None,
    ...     'verbose': True, 'distance': 'euclidean'}
    >>> clusters = cluster(database, 'kmeans', **kparams)
    [INFO] K-Means training -- Start
    0 1 2 DONE
    [INFO] K-Means training -- End

    # Cluster the database with the KSOM
    >>> kparams = {
    ...     'grid_size': (3, 3), 'cluster': True, 'grid': False,
    ...     'margins': 0.01, 'tmax': 1000, 'nbh_rate_0': 1.00,
    ...     'lrn_rate_0': 0.95, 'seed': None, 'verbose': False,
    ...     'distance': 'euclidean'}
    >>> clusters = cluster(database, 'ksom', **kparams)
    [INFO] Grid training -- Start
    10%... 20%... 30%... 40%... 50%... 60%... 70%... 80%... 90%... DONE
    [INFO] Grid training -- End
    """

    # Get the clustering function
    fclt = get_clust_func(method)

    # List of clusters
    clusters = fclt(data, **clt_params)

    # Merge the clusters
    prune(clusters)
    if fuse != 0.:
        merge(clusters, fuse, linkage)

    # Return the clusters
    return sort(clusters)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                       Training & Testing Datasets                        ##
##############################################################################

def _has_index(obj):
    """ Check if np.ndarray or Database/Cluster """
    try:
        obj.index
    except AttributeError:
        has_idx = False
    else:
        has_idx = True
    return has_idx

def rebuild_idx(database, clusters, max_idx=None):
    """ Rebuild the training and testing indexes

    Take a database and the clusters obtained when partitioning it when
    using its only training data, and rebuild the training and testing
    indexes. Ignore the out-of-range data. To retrieve the training data
    indexes, compare the clusters' data indexes to that of the database's
    data, and retrieve their positional indexes in the database; all the
    remaining ones are the testing data indexes (when not out-of-range).

    If `database` is a `Database` object, get its `index` attribute; if
    is a `np.ndarray`, use it directly. This also applies to `clusters`,
    but with the `Cluster` class.

    Parameters
    ----------
    database : np.ndarray or Database
        The reference database on which clustering has been performed
        and on which to retrieve the training & testing indexes.
    clusters : list of np.ndarrays or Clusters
        The clusters on which to extract the training & testing indexes;
        Assumed to be built on the database's training data only.
    [OPT] max_idx : int
        The maximum possible index (for data offset): any index greater
        than `max_idx` either in `database` or `clusters` is ignored;
        use None to skip this option.
            :Defaut: None

    Returns
    -------
    ind_trn_lst : list of 1D np.ndarrays
        The list of the indexes of the data in every cluster (one array
        per cluster; all these data are the database's training data).
    ind_trn : 1D np.ndarray
        The 1D concatenation of all the arrays of `ind_trn_lst`.
    ind_gen : 1D np.ndarray
        The 1D array of the remaining indexes of `ind_trn_lst` that are
        not in `ind_trn`; these indexes serve for testing.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database, Cluster

    # Generate dummy data and wrap them into a set of Clusters
    >>> data = np.linspace(0, 100, 10000).reshape(1000, 10)
    >>> tstp = np.linspace(0, 10, 1000)

    # Wrap the data into a Database
    >>> database = Database(data, tstp)

    # Create dummy clusters on the 80% of the database
    >>> clusters = [Cluster(data[i*100:(i+1)*100], tstp[i*100:(i+1)*100])
    ...             for i in range(8)]

    # Rebuild the indexes
    >>> offset = 5
    >>> max_idx = len(database) - offset
    >>> ind_trn_lst, ind_trn, ind_gen = rebuild_idx(database, clusters, max_idx)
    >>> print(ind_trn.shape, ind_gen.shape)
    (800,) (195,)

    # Directly use the indexes as np.ndarrays
    >>> ind_trn_lst, ind_trn, ind_gen =\
    ...     rebuild_idx(database.index, [clt.index for clt in clusters], max_idx)
    >>> print(ind_trn.shape, ind_gen.shape)
    (800,) (195,)
    """
    # pylint: disable=multiple-statements   # Inline functions

    def _idx_range(idx, rg):
        """ Check that the indexes are below a given range """
        return idx[idx < rg]

    # Check if np.ndarray or Database/Cluster
    isdba = _has_index(database)
    isclt = _has_index(clusters[0])

    # Get the right function
    if isdba and isclt:
        # Database and set of Clusters
        def _fidx(dba, clt): return np.nonzero(np.isin(dba.index, clt.index))[0]
    elif isdba and not isclt:
        # Database and set of np.ndarrays
        def _fidx(dba, clt): return np.nonzero(np.isin(dba.index, clt))[0]
    elif not isdba and isclt:
        # np.ndarray and set of Clusters
        def _fidx(dba, clt): return np.nonzero(np.isin(dba, clt.index))[0]
    else:
        # np.ndarray and set of np.ndarrays
        def _fidx(dba, clt): return np.nonzero(np.isin(dba, clt))[0]

    # Rebuild the local training indexes (one per cluster)n
    if max_idx is None:
        ind_trn_lst = [_fidx(database, clust) for clust in clusters]
    else:
        ind_trn_lst = [
            _idx_range(_fidx(database, clust), max_idx) for clust in clusters]

    # Rebuild the training indexes
    ind_trn = np.hstack(ind_trn_lst)

    # Rebuild the testing indexes
    ind_gen = np.nonzero(
        np.logical_not(np.isin(np.arange(len(database)), ind_trn)))[0]

    if max_idx is not None:
        ind_gen = _idx_range(ind_gen, max_idx)

    return ind_trn_lst, ind_trn, ind_gen

##############################################################################
