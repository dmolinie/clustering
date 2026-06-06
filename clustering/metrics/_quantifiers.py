""" Quantifiers to evaluate a database or cluster homogeneity

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'avstd', 'density', 'silhouettes',
    'quantifiers', 'hydas', 'get_quant_func']

import numpy as np

from clustering.metrics._distances import get_dist_func
from clustering.metrics._volumes import get_vol_func
from clustering.metrics._statistics import get_span_func


##############################################################################
##                        Average Standard Deviation                        ##
##############################################################################

def avstd(database, minmax=False):
    """ Average Standard Deviation

    Parameters
    ----------
    database : np.ndarray, Database or Cluster
        The data of AvStd to compute.
    [OPT] minmax : bool
        If False, return the AvStd (mean of the stds) only; else, return
        the min, mean and max std (for every dimension).
            :Default: False

    Returns
    -------
    mini, mean, maxi : floats or np.ndarrays
        The min, mean and max AvStd of the database (along the axes).

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> dba = np.arange(50., dtype=float).reshape(-1, 5)

    # Compute the AvStd
    >>> mini, mean, maxi = avstd(dba, True)
    >>> print(mini, mean, maxi)
    14.361406616345073
    """

    if not minmax:
        return np.mean(database.std(0))

    avstd_ = database.std(0)
    return np.min(avstd_), np.mean(avstd_), np.max(avstd_)

##############################################################################



##############################################################################
##                              Hyper-Density                               ##
##############################################################################

def density(cluster,
    span='sphere_span', volume='hypersphere', distance='euclidean'):
    """ Compute the density of a cluster

    Compute the span of the cluster and use this value to compute the
    volume of the corresponding hyper-volume. Divide the number of data
    within the cluster by this volume: it is the density of the cluster.
    Basically, the highest the density, the more homogeneous the data.

    Parameters
    ----------
    cluster : np.ndarray, Database or Cluster
        The data for which to compute the density.
    [OPT] span : str
        The span estimation method; see `get_span_func` for details.
            :Default: 'sphere_span'
    [OPT] volume : str
        The hyper-volume name; see `get_vol_func` for details.
            :Default: 'hypersphere'
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'

    Returns
    -------
    dens : float
        The cluster's density (nb of data / vol. of the hyper-volume).

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> dba = np.arange(50., dtype=float).reshape(-1, 5)

    # Compute the density of the dataset
    >>> density(dba)
    5.89337294836781e-09
    """

    # Check if the database is empty
    if len(cluster) < 2:
        raise AssertionError(f"Not enough data ({len(cluster)}), no density")

    # Check the span, volume and distance functions
    fspan = get_span_func(span)
    fvol = get_vol_func(volume)
#    fdist = get_dist_func(distance)    # Distance checked in `fspan`

    # Compute the hypervolume's span
    span = fspan(cluster, distance, True)

    if volume == 'hypersphere':
        span /= 2.

    # Density (True for the volume and False for the surface)
    dim = 1
    if cluster.ndim > 1:
        dim = cluster.shape[-1]

    return cluster.shape[0] / fvol(span, dim, True)

##############################################################################



##############################################################################
##                          Silhouette Coefficient                          ##
##############################################################################

def _dist_intra(val, cluster, fdist):
    """ Intra-cluster mean distance from a point """
    return np.sum(fdist(val, cluster[:], 1)) / (len(cluster)-1)

def _dist_inter(val, clusters, fdist):
    """ Inter-cluster minimal mean distance from a point """
    return min(np.sum(fdist(val, cluster[:], 1)) / len(cluster)
               for cluster in clusters)

def silhouettes(clusters, data=(None, None), distance='euclidean'):
    """ Silhouette Coefficient of a database composed of clusters

    Silhouettes measure how compact a cluster is, and how a data belongs
    to it and not to another one. It is based on two indicators: 1- The
    average distance between a point and all its neighbors within the
    same cluster; 2- The minimal mean distance between this point and
    all the data of every other cluster. The former captures the compact-
    ness of the cluster, whilst the latter attests how distant a data is
    from the other class. The Silhouette Coefficient is obtained by sub-
    tracting the former to the latter, and by dividing it by the maximum
    between both; it takes a value between -1 (worst) and +1 (best).
    There is a SC for every data; some treatment should be done (average,
    min, max, etc.) on every data of a cluster to get a global indicator.

    Parameters
    ----------
    clusters : list of np.ndarrays, Databases or Clusters
        The clusters for which the SCs must be computed. Must contain at
        least clusters.
    [OPT] data : 2-tuple
        The data itself and the cluster index to which it belongs. If
        not provided, all the SCs are computed; if only the data is set,
        compute its SC for every cluster; if the index is also set, com-
        pute its SC for this only cluster.
            :Default: (None, None)
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'

    Returns
    -------
    sils : float, 1D np.ndarray or 2D np.ndarray
        (float)    The SC of a data only if data != (None, None)
        (1D array) The list of the SCs of a data if data != (None, X)
        (2D array) The matrix of all the SCs (for each data) else

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> dba = np.arange(50., dtype=float).reshape(-1, 5)
    >>> clts = [dba + i for i in range(3)]

    # Compute the SCs for every data of every cluster
    >>> sils = silhouettes(clts)
    >>> _ = [print(sil.mean()) for sil in sils]
    -0.10406723626852657
    -0.11954966579482709
    -0.10406723626852658

    # Compute the SC of the data `clts[1][7]` from `clts[1]`
    >>> sil = silhouettes(clts, (clts[1][7], 1))
    >>> print(sil)
    -0.12322580645161291
    """

    # Get the distance
    fdist = get_dist_func(distance, clusters[0].ndim)

    # Silhouette Coefficient of only one data
    if data[0] is not None:

        # If data's class known, compute its SC for this cluster only
        if data[1] is not None:
            clust = clusters[data[1]]
            avg = _dist_intra(data[0], clust, fdist)
            dis = _dist_inter(
                data[0],
                (clt for clt in clusters  if id(clt) != id(clust)),
                fdist)
            return (dis - avg) / max(avg, dis)        # Silhouette Coefficient

        # Try every cluster for the data's SC
        sils = np.empty(len(clusters), float)
        for k, clust in enumerate(clusters):
            avg = _dist_intra(data[0], clust, fdist)
            dis = _dist_inter(
                data[0],
                (clt for clt in clusters if id(clt) != id(clust)),
                fdist)
            sils[k] = (dis - avg) / max(avg, dis)     # Silhouette Coefficient
        return sils

    # Silhouette Coefficients of all data
    sils = []
    # Run data after data through any cluster
    for k, clust in enumerate(clusters):
        clts = [clt for clt in clusters if id(clt) != id(clust)]
        sil = np.empty(len(clust), float)
        for i, val in enumerate(clust):
            avg = _dist_intra(val, clust, fdist)
            dis = _dist_inter(val, clts, fdist)

            # Silhouette Coefficient
            sil[i] = (dis - avg) / max(avg, dis)
        sils.append(sil)
    return sils

##############################################################################



##############################################################################
##                   Hybridized Density-AvStd-Silhouettes                   ##
##############################################################################

def quantifiers(clusters,
    span='sphere_span', volume='hypersphere', distance='euclidean', database=None):
    """ Compute the HyDensity, AvStd and Silhouettes quantifiers

    Take a list of clusters, and compute the Hyper-Density, the Average
    Standard Deviation and the Silhouette Coefficients for each of them.
    Return the quantifiers as a Kx3 matrix, ordered as the Silhouettes,
    AvStd and HyDensity. If the database is provided, its quantifiers
    are added in the first row of the matrix (with null Silhouettes).

    Parameters
    ----------
    clusters : list of np.ndarrays, Databases or Clusters
        The clusters on which compute the HyDAS scores.
    [OPT] span : str
        The span estimation method; see `get_span_func` for details.
            :Default: 'sphere_span'
    [OPT] volume : str
        The hyper-volume name; see `get_vol_func` for details.
            :Default: 'hypersphere'
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'
    [OPT] database : np.ndarray, Database or Cluster
        The corresponding database of the clusters. If provided,
        its quantifiers are computed and set at the very top row
        of the returned matrix (with Silhouettes all set to 0).
            :Default: None

    Returns
    -------
    quants : Kx3 np.ndarray, with K=len(clusters) [+1 if database]
        The matrix of the quantifiers: the Silhouettes, then AvStd and
        finally HyDensity. If the database is provided, its quantifiers
        correspond to the very first row of the matrix.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> dba = np.arange(50., dtype=float).reshape(-1, 5)
    >>> clts = [dba + i for i in range(3)]

    # Compute the quantifiers of the clusters (SCs, AvStd, HyDensity)
    >>> quants = quantifiers(clts)
    >>> print(quants)
    [[-1.04067236e-01  1.43614066e+01  5.89337295e-09]
     [-1.19549666e-01  1.43614066e+01  5.89337295e-09]
     [-1.04067236e-01  1.43614066e+01  5.89337295e-09]]
    """
    # TODO: AvStd or 1/AvStd

    # Assert that `clusters` is a list
    if not isinstance(clusters, (tuple, list)):
        clusters = [clusters]

    # Define the stats matrix
    if database is not None:
        start = 1
        stats = np.empty((len(clusters)+1, 3), float)
        stats[0] = (0.0,
                    avstd(database),
#                   1./avstd(database),
                    density(database, span, volume, distance))
    else:
        start = 0
        stats = np.empty((len(clusters), 3), float)

    # Fulfill the matrix with the quantifiers
    sils = silhouettes(clusters, distance=distance)
    for i, (clust, sil) in enumerate(zip(clusters, sils), start):
        stats[i] = (sil.mean(),
                    avstd(clust),
                    density(clust, span, volume, distance))

    return stats

def hydas(clusters, eps=0.50, database=None, stats=None, **quants):
    """ Hybridized Density-AvStd-Silhouettes-based metric HyDAS

    For a set of clusters, compute its AvStd, its HyDensity and its
    Silhouettes, and provide an averaged score based on that of the
    three prior metrics.

    Parameters
    ----------
    clusters : list of np.ndarrays, Databases or Clusters
        The clusters on which compute the HyDAS scores.
    [OPT] eps : float
        The threshold to consider a quantifier as acceptable. If it is
        a scalar, the same value is used for the 3 quantifiers; it can
        also be a vector of 3 components, each representing a threshold
        for a unique quantifier: first, the Silhouettes, second, the
        HyDensity, and third, the reverse AvStd.
            :Default: 0.50
    [OPT] database : np.ndarray, Database or Cluster
        The corresponding database of the clusters. If provided, its
        quantifiers are computed and set at the very top row of the
        returned matrix (with Silhouettes all set to 0).
            :Default: None
    [OPT] stats : np.ndarray
        The matrix of the quantifiers, to avoid to recompute it; see
        the `quantifiers` function for details.
            :Default: None (recompute the quantifiers)

    Other Parameters
    ----------------
    **quants : inline keyword arguments, optional
        The additional parameters passed to the `quantifiers` function;
        used only if `stats` is None to compute the quantifiers:
          - `span` (str): the span estimation method.
          - `volume` (str): the hyper-volume name.
          - `distance` (str): the distance name.

    Returns
    -------
    stats : Kx4 np.ndarray, with K=len(clusters) [+1 if database]
        The matrix of the quantifiers. The three first columns are the
        quantifiers scores, and the last one contains the HyDAS scores.
        If the database is provided, its quantifiers correspond to the
        very first row of the matrix.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> dba = np.arange(50., dtype=float).reshape(-1, 5)
    >>> clts = [dba + i for i in range(3)]

    # Compute the quantifiers of the clusters (SCs, AvStd, HyDensity)
    >>> quants = quantifiers(clts)

    # Build the HyDAS scores
    >>> stats = hydas(clts, stats=quants)
    >>> print(stats.round(3))
    [[-7.e-03 -8.e-03 -7.e-03  1.e+00]
     [ 1.e+00  1.e+00  1.e+00  8.e+00]
     [ 0.e+00  0.e+00  0.e+00  1.e+00]]
    """

    # Assert that `clusters` is a list
    if not isinstance(clusters, (tuple, list)):
        clusters = [clusters]

    # Compute the quantifiers if they are not provided
    if stats is None:
        stats = quantifiers(clusters, **quants, database=database)

    # Resize stats if it contains only the quantifiers (not HyDAS)
    if stats.shape[1] == 3:
        stats = np.resize(stats, (4, len(stats))).T

    # Normalize the quantifiers
    stats[:, :3] /= stats[:, :3].max(0)

    # Compare every quantifier to a threshold
    states = stats[:, :3] >= eps

    # Correspondences to attribute the HyDAS scores
    lut = np.array([[False, False, False], [False, False, True],
                    [False, True, False], [False, True, True],
                    [True, False, False], [True, False, True],
                    [True, True, False], [True, True, True]], bool)

    # Set the HyDAS scores (last column of stats)
    for i, state in enumerate(states):
        stats[i, 3] = np.argwhere((state[:] == lut).all(1))[0, 0]
    stats[:, 3] += 1  # Offset (start from 0)

    return stats

##############################################################################



##############################################################################
##                         Get Quantifier Function                          ##
##############################################################################

def get_quant_func(quantifier='density'):
    """ Get the reference to a specified quantifier function

    Take a quantifier's name and return its corresponding function; return
    the `density` quantifier as default. Every function has signature:
        fquant(data, *args)
    where `data` is an ND array and `args` are the quantifier's parameters
    (see the concerned function for details).

    Parameters
    ----------
    [OPT] quantifier : str
        The quantifier name among: {'std', 'avstd', 'density'}.
            :Default: 'density'

    Returns
    -------
    fquant : reference to a function
        The quantifier function; has signature: `fquant(data, *args)`.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> data = np.arange(15., dtype=float).reshape(-1, 5)

    # Compute the density of the dataset
    >>> fquant = get_quant_func('density')
    >>> den = fquant(data,
    ...     span='sphere_span', volume='hypersphere', distance='euclidean')
    >>> print(den)
    3.2624791802641023e-06

    # Compute the Silhouettes between two datasets
    >>> fquant = get_quant_func('silhouettes')
    >>> sils = fquant([data, data+1.])
    >>> print(sils)
    [array([-0.2       , -0.26666667, -0.37777778]),
     array([-0.37777778, -0.26666667, -0.2       ])]
    """

    # Get the quantifier function
    quant = quantifier.lower()
    if quant == 'std':
        return np.std
    if quant == 'avstd':
        return avstd
    if quant == 'density':
        return density
    if quant == 'silhouettes':
        return silhouettes
    if quant == 'hydas':
        return hydas

    # Raise an error if the quantifier is invalid
    raise ValueError(f"Wrong value `{quantifier}` for `quantifier`; options are:\n"
        + "\t{'std', 'avstd', 'density'}")

##############################################################################
