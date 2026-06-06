""" Empirical Cumulative Distributions & Kolmogorov-Smirnov Test

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: August 2022
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'cumulative_distributions', 'cdfs_matrix',
    'cdfs_states', 'cdfs_groups', 'ks_test']

import numpy as np

import clustering.metrics as mts


##############################################################################
##                    Empirical Cumulative Distributions                    ##
##############################################################################

def cumulative_distributions(clusters, database=None, pts=100):
    """ Empirical Cumulative Distribution Functions (ECDs)

    Compute the ECDs of every cluster and of the database (if provided),
    for each value in the data range, divided in `pts` subintervals.

    Parameters
    ----------
    clusters : list of np.ndarrays or Clusters
        The clusters whose ECDs must be computed.
    [OPT] database : np.ndarray or Database
        The corresponding database of the clusters. If provided, its
        ECDs are added at the end of the ECDs matrix (last row).
            :Default: None
    [OPT] pts : int
        The number of subintervals to cut the database/clusters data
        range up into.
            :Default: 100

    Returns
    -------
    vals : 1D or 2D np.ndarray
        The set of the increasing reference values (has length `pts`);
        if the clusters contain 1D data, `vals` is a 1D array; else, it
        has the same last dimension as that of any cluster.
    cdfs : 2D or 3D np.ndarray
        The ECDs matrix, with shape `(K, pts, dim)`, where `K` is the
        length of `clusters` (+1 if `database` is provided) and `dim`
        is the last dimension of any cluster (i.e. the dimension of the
        feature space). If `dim` is 1 (for 1D data), this dimension is
        discarded, and `cdfs` has shape `(K, pts)` only.

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy list of clusters
    >>> arr = np.arange(50., dtype=float).reshape(-1, 5)
    >>> arrs = [arr+i for i in range(4)]

    # Compute the ECDs without the database
    >>> vals, cdfs = cumulative_distributions(arrs, None, 100)
    >>> print(np.shape(vals), np.shape(cdfs))
    (100,) (5, 100, 50)

    # Compute the ECDs with the database
    >>> vals, cdfs = cumulative_distributions(arrs, arrs[0], 100)
    >>> print(np.shape(vals), np.shape(cdfs))
    (100,) (6, 100, 50)
    """

    if database is not None:

        # Get the extremal values of the database
        limits = (np.min(database, 0), np.max(database, 0))

        # Get the dimension (nb of cols) of the database
        dim = 1 if np.ndim(database) == 1 else np.shape(database)[1]

    else:

        # Get the extremal values of every cluster
        # No instantiation list for dim requires a nonempty cluster
        lims = []
        for clust in clusters:                  # Check every cluster
            if len(clust) != 0:                 # Skip empty clusters
                lims.append((np.min(clust, 0), np.max(clust, 0)))
        lims = np.array(lims, float).round(6)   # 1e-16 --> 0.

        # Dimension of the last nonempty cluster
        # Get the dimension (nb of cols) of the database
        # pylint: disable-next=W0631
        dim = 1 if np.ndim(clust) == 1 else np.shape(clust)[-1]

        # Get the extremal values among all the clusters
        limits = (np.min(lims[:, 0], 0), np.max(lims[:, 1], 0))

    # Reference values and empirical cumulative distributions
    vals = np.linspace(limits[0], limits[1], pts)
    cdfs = np.zeros((len(clusters) if database is None else len(clusters)+1,
                     len(vals), dim), float)

    # ECDs of the clusters
    for k, clust in enumerate(clusters):
        if len(clust) != 0:             # Check if cluster is nonempty
            for i, val in enumerate(vals):
                cdfs[k, i] = np.mean(clust[:] <= val, 0)

    # ECDs of the database (if provided)
    if database is not None:
        for i, val in enumerate(vals):
            cdfs[-1, i] = np.mean(database[:] <= val, 0)

    # Remove last dimension of `cdfs` if 1D data (but keep it if Nx1 dims)
    if dim == 1 and clust.shape[-1] != 1:
        cdfs = cdfs.squeeze()

    return vals, cdfs

##############################################################################



##############################################################################
##                              ECDs Analysis                               ##
##############################################################################

#--------------------------   Matrix of the MHDs   --------------------------#
def _cdfs_mat_core_1d(vals, cdfs):
    """ Matrix of the MHDs for 1D data """
    matrix = np.zeros((len(cdfs), len(cdfs)), float)
    crvi = np.empty((cdfs.shape[1], 2), float)
    crvj = np.empty((cdfs.shape[1], 2), float)
    # Get the CDFs threshold values
    crvi[:, 0], crvj[:, 0] = vals[:], vals[:]
    # Compute the MHD for any pair of ECDs
    for i, crvi[:, 1] in enumerate(cdfs):
        for j, crvj[:, 1] in enumerate(cdfs[i+1:], i+1):
            matrix[i, j] = np.around(mts.hausdorff(crvi, crvj), 2)
    return matrix

def _cdfs_mat_core_2d(vals, cdfs, dim):
    """ Matrix of the MHDs for 2D data """
    matrix = np.zeros((dim, len(cdfs), len(cdfs)), float)
    crvi = np.empty((cdfs.shape[1], 2), float)
    crvj = np.empty((cdfs.shape[1], 2), float)
    # Set the dimension (one matrix per dimension)
    for k in range(dim):
        # Get the matrix of the CDFs for the k-th dimension
        mat = cdfs[:, :, k]                 # Dim k ECDs
        # Get the CDFs threshold values for the k-th dimension
        crvi[:, 0], crvj[:, 0] = vals[:, k], vals[:, k]
        # Compute the MHD for any pair of ECDs
        for i, crvi[:, 1] in enumerate(mat):
            for j, crvj[:, 1] in enumerate(mat[i+1:], i+1):
                matrix[k, i, j] = np.around(mts.hausdorff(crvi, crvj), 2)
    return matrix

def cdfs_matrix(vals, cdfs, database=False):
    """ Build the matrix of the MHDs between the ECDs

    Provided a set of Empirical Cumulative Distributions, compute the
    Modified Hausdorff Distance between any pair of ECDs. The ECDs are
    assumed gathered into a 3D array, whose first dimension is dedicated
    to the clusters, the second is the ECDs sensu stricto, corresponding
    to a given database's dimension as of the third array dimension.
    If the cdfs 3D array contains the ECDs of the database (assumed to
    be at the very bottom, in the last `row`), the flag `database` must
    be set to True; in such a case, the database's ECDs are ignored.
    Ex: if cdfs.shape is (5, 100, 3), this means there are 5 clusters,
        whose data have 3 dimensions, and the ECDs comprise 100 values.

    Parameters
    ----------
    vals : np.ndarray
        The reference values used for the computation of the ECDs
        (somehow the thresholds).
    cdfs : 2D or 3D np.ndarray
        The cluster EDCs; first dimension respective to the clusters,
        second to the ECDs themselves, third to the data dimension.
        If the data are 1D, `cdfs` is expected to be 2D only.
    [OPT] database : bool
        Specify if `cdfs` contains the database's ECDs.
            :Default: False

    Returns
    -------
    matrix : 2D or 3D np.ndarray
        The matrix of the MHDs, computed for every pair of ECDs. First
        dimension is for the data dimension (third dim of `cdfs`), and,
        for each, the 2D matrix is the table of the MHDs, whose rows and
        cols are the clusters' indexes. If the input data are 1D, first
        dimension of `matrix` is discarded.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    >>> database = Database(array, np.arange(len(array)))

    # Cluster the database
    >>> clusters = ksom_cluster(database, grid_size=(3, 3))
    >>> clusters = [clt for clt in clusters if len(clt) != 0]   # or `prune`
    >>> print(len(clusters))
    4

    # Compute the Empirical Cumulative Distributions (ECDs)
    >>> vals, cdfs = cumulative_distributions(clusters, pts=100)

    # Build the matrix of MHDs
    >>> matrix = cdfs_matrix(vals, cdfs)
    >>> print(matrix.shape)
    (3, 4, 4)
    """

    # Remove the database's CDFs
    if database:
        cdfs = cdfs[:-1]

    # Compute the MHD for any pair of ECDs
    if cdfs.ndim == 2:
        return _cdfs_mat_core_1d(vals, cdfs)
    return _cdfs_mat_core_2d(vals, cdfs, cdfs.shape[2])
#----------------------------------------------------------------------------#

#-----------------------------   Valid Pairs   ------------------------------#
def cdfs_states(matrix, eps=0.15):
    """ Build the closeness states matrix

    Provided the matrix of all the MHDs between any pair of ECDs, decide
    whether two ECDs (clusters) are close. To do so, compare each MHD to
    a threshold; do that for each dimension (first dim of `matrix`),
    and do the logical `and` between all, which represents the Boolean
    states matrix (`True` for a close pair, and `False` else). Identify
    also the corresponding clusters' indices for pairs set to `True`.
    For a given dimension, the threshold is defined as a fraction of the
    maximal value of the corresponding matrix.

    Parameters
    ----------
    matrix : 2D or 3D np.ndarray
        The matrix of the MHDs; must only contain MHDs between clusters,
        without those of the database. If 3D matrix, iterate the 1st dim-
        ension, and use every subsequent 2D matrix, one per dimension of
        the data/space; if 2D matrix, assume there is only one dimension.
    [OPT] eps : float
        The percentage of the maximal value of the MHDs matrix, proper
        to each dimension; this value serves a threshold. If a MHD is
        lower than this threshold, the involved ECDs (and thus clusters)
        are considered as close.
            :Default: 0.15

    Returns
    -------
    states : 2D np.ndarray
        The final states matrix.
    idx : list of tuples
        The list of the indices of the clusters whose corresponding
        state is set to True (i.e. they are close).

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    >>> database = Database(array, np.arange(len(array)))

    # Cluster the database
    >>> clusters = ksom_cluster(database, grid_size=(3, 3))
    >>> clusters = [clt for clt in clusters if len(clt) != 0]   # or `prune`
    >>> print(len(clusters))
    4

    # Compute the Empirical Cumulative Distributions (ECDs)
    >>> vals, cdfs = cumulative_distributions(clusters, pts=100)

    # Build the matrix of MHDs
    >>> matrix = cdfs_matrix(vals, cdfs)

    # Build the states matrix
    >>> states, idx = cdfs_states(matrix)
    >>> print(states)
    [[ True  True False False]
     [ True  True False False]
     [ True  True  True False]
     [ True  True  True  True]]
    """

    # ECDs closeness states (True or False) over all dimensions
    if matrix.ndim == 2:
        states = matrix <= np.max(matrix)*eps
    else:
        states = matrix[0] <= np.max(matrix[0])*eps
        for mat in matrix[1:]:
            states = np.logical_and(states, mat <= np.max(mat)*eps)

    # Indices of the close clusters (`True` state)
    idx = []
    for i, vali in enumerate(states):
        for j, valj in enumerate(vali[i+1:], i+1):
            if valj:
                idx.append((i, j))

    return states, idx
#----------------------------------------------------------------------------#

#---------------------------   Build the Groups   ---------------------------#
def cdfs_groups(idx, card=None):
    """ Build the closeness map

    Provided the closeness states matrix and the corresponding indices,
    link the clusters with each other into linkage lists (one per isol-
    ated groups of clusters, within which they are close together).
    Return the list of these sub-lists: its length is the estimated
    number of regions of the feature space (a candidate for clustering).

    Parameters
    ----------
    idx : list of tuples
        The list of the pairs of indices of close clusters.
    [OPT] card : int
        The total number of clusters. If not provided, set it to the
        highest value among the indices of `idx`.
            :Default: None

    Returns
    -------
    groups : 2D list of ints
        The list of the regions of the feature space, each represented
        by a sub-list of indices of close clusters. The clusters within
        a sub-list are connected to each other (close), whilst the sub-
        lists are disconnected regions (distant).

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    >>> database = Database(array, np.arange(len(array)))

    # Cluster the database
    >>> clusters = ksom_cluster(database, grid_size=(3, 3))
    >>> clusters = [clt for clt in clusters if len(clt) != 0]   # or `prune`
    >>> print(len(clusters))
    4

    # Compute the Empirical Cumulative Distributions (ECDs)
    >>> vals, cdfs = cumulative_distributions(clusters, pts=100)

    # Build the matrix of MHDs
    >>> matrix = cdfs_matrix(vals, cdfs)

    # Build the states matrix
    >>> states, idx = cdfs_states(matrix)

    # Build the groups of clusters
    >>> groups = cdfs_groups(idx, len(clusters))

    >>> print(groups)
    [{0}, {1, 3}, {2}]

    >>> print("Estimated number of regions:", len(groups))
    Estimated number of regions: 3
    """

    # Get the maximal cluster's index (starts from 0 --> +1)
    if card is None:
        card = np.max(np.ravel(idx)) + 1

    # Initial list: each cluster is considered as fully disconnected
    groups = [[i] for i in range(card)]

    # First mapping round
    for i in idx:
        groups[i[0]].append(i[1])

    # Sets are easier to deal with
    groups = [set(grp) for grp in groups]

    # Matrix of the intersections (True <==> nonempty intersection)
    mat = np.full((len(groups), len(groups)), False, dtype=bool)
    for i, grpi in enumerate(groups):
        for j, grpj in enumerate(groups[i+1:], i+1):
            mat[i, j] = len(grpi.intersection(grpj))

    # While there is still an nonempty intersection, merge the groups
    while np.any(mat):
        pos = np.where(mat)                         # Matrix' idx where True
        k = 0                                       # Local counter
        pop = []                                    # Groups to be removed

        # Set a group and fuse to it all the intersected groups
        while (k < len(pos[0])) and (pos[0][k] == pos[0][0]):
            i, j = pos[0][k], pos[1][k]             # Intersected groups
            groups[i] = groups[i].union(groups[j])  # Fuse 2nd group in 1st
            pop.append(j)                           # Remove the second group
            k += 1                                  # Nb of fuses done

        # Remove the fused groups
        for j in reversed(pop):
            groups.pop(j)

        # Recompute the intersections matrix with the refined groups
        mat = np.full((len(groups), len(groups)), False, dtype=bool)
        for i, grpi in enumerate(groups):
            for j, grpj in enumerate(groups[i+1:], i+1):
                mat[i, j] = len(grpi.intersection(grpj))

    return groups
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                         Kolmogorov-Smirnov Test                          ##
##############################################################################

def ks_test(clusters, database=None, pts=100):
    """ Kolmogorov-Smirnov Test

    Statistically compare two data distributions, and provide a closeness
    score (percentage) between both (the closer to 0, the more similar).

    Parameters
    ----------
    clusters : list of np.ndarrays or Clusters
        The clusters of which the ECDs must be computed.
    [OPT] database : np.ndarray or Database
        The corresponding database of the clusters. If provided,
        its ECDs are added at the end of the ECDs matrix (last row).
            :Default: None
    [OPT] pts : int
        The number of subintervals to cut the database/clusters data
        range up into.
            :Default: 100

    Returns
    -------
    kss : 2D np.ndarray
        The KS Test matrix (closeness percentage for every ECDs pair).

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy list of clusters
    >>> arr = np.arange(50., dtype=float).reshape(-1, 5)
    >>> arrs = [arr+i for i in range(4)]

    # Compute the ECDs without the database
    >>> ks_test(arrs, None, 100)
    array([[ 0., 10., 10., 10.],
           [10.,  0., 10., 10.],
           [10., 10.,  0., 10.],
           [10., 10., 10.,  0.]])

    # Compute the ECDs with the database
    >>> ks_test(arrs, arrs[0], 100)
    array([[ 0., 10., 10., 10.,  0.],
           [10.,  0., 10., 10., 10.],
           [10., 10.,  0., 10., 10.],
           [10., 10., 10.,  0., 10.],
           [ 0., 10., 10., 10.,  0.]])
    """

    cdfs = cumulative_distributions(clusters, database, pts)[1]

    # Means of the CDFs
    means = np.empty((cdfs.shape[0:2]), float)
    for i, cdf in enumerate(cdfs):
        means[i] = cdf.mean(1)

    # KS Test (matrix's upper right part)
    kss = np.zeros((cdfs.shape[0], cdfs.shape[0]), float)
    for i, meani in enumerate(means):
        for j, meanj in enumerate(means[i+1:], i+1):
            kss[i, j] = np.around(np.max(np.abs(meani - meanj)), 3)
    kss += kss.T    # Matrix's lower left part

    # Probas to percentages
    return 100. * kss

##############################################################################
