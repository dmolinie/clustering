""" Compute some statistics on a dataset

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'mean', 'std', 'mean_std', 'max_to_mean', 'max_span', 'sphere_span',
    'intra_distances', 'inter_distances', 'jaccard_index', 'sorensen_index',
    'dunn_index', 'bouldin_davies_index', 'get_span_func', 'get_stat_func']

import numpy as np

from clustering.metrics._distances import get_dist_func


##############################################################################
##                    Dataset Mean & Standard Deviation                     ##
##############################################################################

#---------------------------------   Mean   ---------------------------------#
def mean(data, axis=None, **kwargs):
    """ Mean of a dataset

    Parameters
    ----------
    data : np.ndarray, Database or Cluster
        The data of mean to compute.
    [OPT] axis : int
        The axis along which to compute the mean.
            :Default: None (all elements' mean)

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments, optional
        The positional parameters for the NumPy's `mean` function.

    Returns
    -------
    mean : float or np.ndarray
        The mean of the data.

    Examples
    --------
    >>> import numpy as np
    >>> mean(np.arange(50., dtype=float).reshape(-1, 5))
    24.5
    """
    return np.mean(data, axis, **kwargs)
#----------------------------------------------------------------------------#

#--------------------------   Standard Deviation   --------------------------#
def std(data, axis=None, **kwargs):
    """ Standard Deviation of a dataset

    Parameters
    ----------
    data : np.ndarray, Database or Cluster
        The data of std to compute.
    [OPT] axis : int
        The axis along which to compute the std.
            :Default: None (all elements' std)

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments, optional
        The positional parameters for the NumPy's `std` function.

    Returns
    -------
    std : float or np.ndarray
        The std of the data.

    Examples
    --------
    >>> import numpy as np
    >>> std(np.arange(50., dtype=float).reshape(-1, 5))
    14.430869689661812
    """
    return np.std(data, axis, **kwargs)
#----------------------------------------------------------------------------#

#----------------------   Mean & Standard Deviation   -----------------------#
def mean_std(data, axis=None, **kwargs):
    """ Mean and Standard Deviation of a dataset

    Parameters
    ----------
    data : np.ndarray, Database or Cluster
        The data of mean & std to compute.
    [OPT] axis : int
        The axis along which to compute the mean and the std.
            :Default: None (all elements' mean and std)

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments, optional
        The positional parameters for the NumPy's `mean` and `std` func-
        tions. These arguments are passed to both functions.

    Returns
    -------
    mean : float or np.ndarray
        The mean of the data.
    std : float or np.ndarray
        The standard deviation of the data.

    Examples
    --------
    >>> import numpy as np
    >>> mean_std(np.arange(50., dtype=float).reshape(-1, 5))
    (24.5, 14.430869689661812)
    """
    return np.mean(data, axis, **kwargs), np.std(data, axis, **kwargs)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                 Cluster's homogeneity & density measures                 ##
##############################################################################

#----------------   Maximal Distance to the Cluster's Mean   ----------------#
def max_to_mean(cluster, distance='euclidean'):
    """ Maximal distance to the mean of a cluster

    Compute the mean of the data and compare it to any of the data;
    return the maximal distance.

    Parameters
    ----------
    cluster : np.ndarray, Database or Cluster
        The data for which one wants the max distance from the mean.
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'

    Returns
    -------
    dist : float
        The maximal distance between the mean and any of the data.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> data = np.arange(50., dtype=float).reshape(-1, 5)

    # Compute the dataset's span
    >>> max_to_mean(data, 'euclidean')
    50.31152949374527
    """

    # Check if the data is empty
    if len(cluster) < 2:
        raise AssertionError(f"Not enough data ({cluster.size}) in `cluster` "
            + "to compute intra-cluster distances")

    # Get the distance to use
    fdist = get_dist_func(distance, cluster.ndim)

    # Maximal distance between the mean and any of the cluster's data
    return np.max(fdist(cluster.mean(0), cluster[:], 1))
#----------------------------------------------------------------------------#

#-------------------------   Maximal Cluster Span   -------------------------#
def max_span(cluster, distance='euclidean'):
    """ Maximal span of a cluster
        <=> Maximal distance between two data of a cluster

    Compute the distance between the two furthest points of a cluster.
    For each data, its distance to any of the other data is computed;
    the cluster's span is the maximal of all these distances.

    Note: assuming the number of data is N, there are N*(N-1) distances
          to compute (each of the data is compared to the (N-1) others);
          the complexity is thus O(N^2), what can be heavy. To reduce
          the number of operations, prefer the func. `max_to_mean`.

    Parameters
    ----------
    cluster : np.ndarray, Database or Cluster
        The data for which one wants the max distance from one another.
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'

    Returns
    -------
    span : float
        The maximal distance between two data within the given cluster.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> data = np.arange(50., dtype=float).reshape(-1, 5)

    # Compute the data span
    >>> max_span(data, 'euclidean')
    100.62305898749054
    """

    # Check if the dataset is empty
    if len(cluster) < 2:
        raise AssertionError(f"Not enough data ({cluster.size}) in `cluster` "
            + "to compute intra-cluster distances")

    # Get the distance to use
    fdist = get_dist_func(distance, cluster.ndim)

    # Find the two furthest points from one another
    dist_max = fdist(cluster[-2], cluster[-1], 0)
    for i, data in enumerate(cluster[:-2], 1):
        dist_max = np.maximum(dist_max, np.max(fdist(data, cluster[i:], 1)))

    # Maximal 2-data distance
    return dist_max
#----------------------------------------------------------------------------#

#-------------   Minimal Spheroid Covering the Cluster's Data   -------------#
def sphere_span(cluster, distance='euclidean', double=True):
    """ Minimal spheroid covering the data of a cluster

    Smallest spheroid whose center is the mean of the cluster's data and
    containing all the data instances within the cluster. It is an upper
    interpolation of the max cluster's span, but easier and faster to
    compute. Only one point (data's mean) is compared to any of the
    other data; the complexity is linear (O(N)), in comparison to the
    computation of all the 2-data distances, whose complexity is O(N^2).

    We define the two following notations:
      - Cluster's data's mean:
            m = mean_{i}(d_i)
      - Maximal distance between m and any data within the cluster:
            dmax = max_{i}(d_i, m)

    Parameters
    ----------
    cluster : np.ndarray, Database or Cluster
        Data for which one wants the max distance from the mean.
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'
    [OPT] double : bool
        Either double dmax dist, or average two highest ones:
          - True: only the largest spheroid is determined; the cluster's
            span is its diameter (2 times its radius);
          - False: the two largest spheroids are computed, and the
            cluster's span is the mean of their respective diameter
            (summation of their respective radius dmax1 + dmax2).
        :Default: True (double radius)

    Returns
    -------
    span : float
        Either the double of the max dist between two points, or the
        mean of the two highest ones; triggered by the 'double' arg.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy noisy data
    >>> data = np.arange(50., dtype=float) + 10.*np.random.random(50)
    >>> data = data.reshape(-1, 5)

    # Compute the spheroid span
    >>> sphere_span(data, 'euclidean', False)    # Fast estimate
    99.86225604982157
    >>> sphere_span(data, 'euclidean', True)     # True span
    104.26629079320999
    """

    # Check if the dataset is empty
    if len(cluster) < 2:
        raise AssertionError(f"Not enough data ({cluster.size}) in `cluster` "
            + "to compute intra-cluster distances")

    # Simple double of the maximal distance to the mean
    if double:
        return 2. * max_to_mean(cluster, distance)

    # Get the distance to use
    fdist = get_dist_func(distance, cluster.ndim)

    # Compute the distance of each point to the data's mean
    dist = fdist(cluster.mean(0), cluster[:], 1)

    # Find the index of the maximum
    idx = np.argmax(dist)

    # Mean of the two spheroids' diameters
    return dist[idx] + np.max(np.delete(dist, idx))
#----------------------------------------------------------------------------#

#--------------------------   Get Span Function   ---------------------------#
def get_span_func(span='sphere_span'):
    """ Get the reference to a specified span estimation function

    Take a dataset spanning type and return its corresponding function;
    return the `sphere_span` as default. Every function has signature:
        fspan(data, distance='euclidean', *args):
    where `data` is the dataset and `distance` is the distance to use
    (default to `euclidean`).

    Parameters
    ----------
    [OPT] span : str
        The span estimation method among:
        {'max_to_mean', 'max_span', 'sphere_span'}
            :Default: 'sphere_span'

    Returns
    -------
    fspan : reference to a function
        The span estimation function.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> data = np.arange(50., dtype=float).reshape(-1, 5)

    # Get the spanning function
    >>> fspan = get_span_func('sphere_span')

    # Compute the distances
    >>> fspan(data)
    100.62305898749054
    """

    # Get the span function
    spn = span.lower()
    if spn == 'sphere_span':
        return sphere_span
    if spn == 'max_to_mean':
        return max_to_mean
    if spn == 'max_span':
        return max_span

    # Raise an error if the span estimation function is invalid
    raise ValueError(f"Wrong value `{span}` for `span`; options are:\n"
        + "\t{'max_span', 'max_to_mean', 'sphere_span'}")
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                      Intra/Inter-Cluster Distances                       ##
##############################################################################

#-----------------------   Intra-Cluster Distances   ------------------------#
def intra_distances(cluster, symmetric=True, distance='euclidean'):
    """ Compute the intra-cluster distances

    Take a set of data, compute the distances between every pair of them,
    and wrap all these distances into an `NxN` matrix, with `N` the number
    of samples in `cluster`. They are called the "intra-cluster distances".

    Parameters
    ----------
    cluster : np.ndarray, Database or Cluster
        The cluster for which to compute the intra-cluster distances.
    [OPT] symmetric : bool
        If the distance is symmetric (should be), fill the diagonal with
        zeros, compute only the distances in the upper triangular matrix
        and mirror it in its lower part. Else, compute any distances.
            :Default: True
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'

    Returns
    -------
    distances : np.ndarray
        The `NxN` intra-cluster distances: value at index `(i, j)` is
        the distance between data `i` and data `j`. If the distance is
        symmetric, the values at indexes `(i, j)` and `(j, i)` are
        equal, and the diagonal is full of zeros. Use the `symmetric`
        argument to save computation if the distance indeed is.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy cluster of data
    >>> cluster = np.arange(15., dtype=float).reshape(-1, 3)

    # Compute the intra-cluster distances
    >>> dists = intra_distances(cluster, symmetric=True, distance='euclidean')
    >>> print(dists.round(2))
    [[ 0.    5.2  10.39 15.59 20.78]
     [ 5.2   0.    5.2  10.39 15.59]
     [10.39  5.2   0.    5.2  10.39]
     [15.59 10.39  5.2   0.    5.2 ]
     [20.78 15.59 10.39  5.2   0.  ]]
    """

    # Check if the cluster is empty
    if len(cluster) < 2:
        raise AssertionError(f"Not enough data ({cluster.size}) in `cluster` "
            + "to compute intra-cluster distances")

    # Get the distance to use
    fdist = get_dist_func(distance, cluster.ndim)

    # NxN distances, with N the number of data
    distances = np.zeros((len(cluster), len(cluster)), dtype=float)

    # If symmetric distance, compute only the distances between the unique pairs
    if symmetric:
        for i, data in enumerate(cluster, 0):
            dists = fdist(data, cluster[i+1:], 1)
            distances[i, i+1:] = dists  # Upper triangular
            distances[i+1:, i] = dists  # Lower triangular (mirror)
    # Else, compute the distances between any pair in both senses
    else:
        for i, data in enumerate(cluster):
            distances[i] = fdist(data, cluster[:], 1)

    return distances
#----------------------------------------------------------------------------#

#-----------------------   Inter-Cluster Distances   ------------------------#
def inter_distances(clusters, symmetric=True, distance='euclidean'):
    """ Compute the inter-cluster distances

    Take a set of clusters of data and compute the distances between any
    possible pair of data; to this end, iterate the clusters and compare
    any data of cluster `i` to any of cluster `j`; ignore the case when
    `i` and `j` are equal. Wrap any matrix of distances into a 2D list
    so that the matrix at index `(i, j)` is the set of distances between
    cluster `i` and cluster `j`. If the distance is symmetric, mirror
    the distances: matrix at index `(i, j)` is the transposed matrix of
    that at index `(j, i)`. If the distance is not symmetric, compute
    both matrices.

    Parameters
    ----------
    clusters : list of np.ndarrays, Databases or Clusters
        The clusters for which to compute the inter-cluster distances.
    [OPT] symmetric : bool
        If the distance is symmetric, compute the inter-cluster distances
        in one direction only, and mirror them for the other direction;
        else, compute the distances for both directions. In both cases,
        the diagonal index (`[i][i]`) are left empty.
            :Default: True
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'

    Returns
    -------
    distances : 2D list of np.ndarrays
        The `KxK` matrices of the inter-cluster distances: the matrix at
        index `(i, j)` is the matrix of the distances between any data
        of cluster `i` and any data of cluster `j`. For instance, the
        value at index `(k, l)` of the matrix at index `(i, j)` of the
        list is the distance between the `k`-th data of cluster `i` and
        the `l`-th data of cluster `j`. If the distance is symmetric,
        only the upper triangular matrix is computed, and the lower part
        of it is the mirrored values (with transposed matrices):
            `distances[i][j] = transpose(distances[j][i])`
        If the distance is not symmetric, the inter-cluster distances
        between any possible pair are computed. In both cases, the dia-
        gonal cells are ignored (`distances[i][i]` is left empty).

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy set of clusters
    >>> clusters = [
    ...     np.linspace(0., 1., 15).reshape(-1, 3),
    ...     np.linspace(0., 2., 18).reshape(-1, 3),
    ...     np.linspace(2., 5., 21).reshape(-1, 3)]

    # Compute the intra-cluster distances
    >>> dists = inter_distances(clusters, symmetric=True, distance='euclidean')

    # Compute the maximal inter-cluster distances
    >>> max_dists = np.array(
    ...     [np.max(d) for dist in dists for d in dist if len(d) > 0])
    >>> print(max_dists.round(2))
    [3.14 8.28 3.14 8.2  8.28 8.2 ]
    """

    # Check if the set of clusters is empty
    if len(clusters) < 2:
        raise AssertionError(f"Not enough clusters ({len(clusters)}) in `clusters` "
            + "to compute intra-cluster distances")

    # Get the distance to use
    fdist = get_dist_func(distance, clusters[0].ndim)

    # KxK matrices of distances, with K the number of clusters
    distances = [[[] for _ in range(len(clusters))] for _ in range(len(clusters))]

    # If symmetric distance, compute the matrices of distances between any
    # unique pair of clusters ((i, j) and (j, i) are the same)
    if symmetric:
        for i, clti in enumerate(clusters):
            for j, cltj in enumerate(clusters[i+1:], i+1):
                dists = np.zeros((len(clti), len(cltj)), dtype=float)
                # Compare any data of `clti` to all the data of `cltj`
                for k, data in enumerate(clti):
                    dists[k] = fdist(data, cltj, 1)
                distances[i][j] = dists     # Upper triangular
                distances[j][i] = dists     # Lower triangular (mirror)
    # Else, compute the matrices for any pair of clusters in both senses
    else:
        for i, clti in enumerate(clusters):
            for j, cltj in enumerate(clusters):
                if i == j:
                    continue
                dists = np.zeros((len(clti), len(cltj)), dtype=float)
                # Compare any data of `clti` to all the data of `cltj`
                for k, data in enumerate(clti):
                    dists[k] = fdist(data, cltj, 1)
                distances[i][j] = dists

    return distances
#----------------------------------------------------------------------------#

#--------------   Min/Mean/Max Intra/Inter-Cluster Distance   ---------------#
def _min_intra(data, fdist):
    """ Minimal intra-cluster distance """
    return min(np.min(fdist(val, data[i:], 1))
        for i, val in enumerate(data[:-1], 1))

def _mean_intra(data, fdist):
    """ Mean intra-cluster distance """
    card = len(data) * (len(data)-1) // 2
    return sum(np.sum(fdist(val, data[i:], 1))
        for i, val in enumerate(data[:-1], 1)) / card

def _max_intra(data, fdist):
    """ Maximal intra-cluster distance """
    return max(np.max(fdist(val, data[i:], 1))
        for i, val in enumerate(data[:-1], 1))

def _min_inter(sets, fdist):
    """ Minimal inter-cluster distance """
    return min(np.min(fdist(data, setj, 1)) for i, seti in enumerate(sets[:-1], 1)
        for setj in sets[i:] for data in seti)

def _mean_inter(sets, fdist):
    """ Mean inter-cluster distance (means are computed at the set's level
    (mean of the means), they are not weighted by the sizes of the sets) """
    return np.mean(
        [sum(np.sum(fdist(data, setj, 1)) for data in seti) / (len(seti)*len(setj))
         for i, seti in enumerate(sets[:-1], 1) for setj in sets[i:]])

def _max_inter(sets, fdist):
    """ Maximal inter-cluster distance """
    return max(np.max(fdist(data, setj, 1)) for i, seti in enumerate(sets[:-1], 1)
        for setj in sets[i:] for data in seti)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                             Database Indexes                             ##
##############################################################################

#-------------------------------   Jaccard   --------------------------------#
def jaccard_index(set1, set2):
    """ Jaccard Index between two datasets

    This Jaccard index J between two sets A and B is defined as:
        J(A, B) = |A ∩ B| / |A ∪ B|
    where `|.|` is the cardinal, `∩` is the intersection and `∪` is the
    union operators. This index ranges from 0 to 1: 0 means the datasets
    are fully disjoint, and 1 means they fully overlap.

    Parameters
    ----------
    set1, set2 : 1D np.ndarrays, Databases or Clusters
        The 1D arrays of data for which to compute the index.

    Returns
    -------
    index : float
        The Jaccard Index.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> set1 = np.arange(10., 50., dtype=float)
    >>> set2 = np.arange(20., 90., dtype=float)

    # Compute the Jaccard Index
    >>> jaccard_index(set1, set2)
    0.375
    """
    if isinstance(set1, np.ndarray):
        set1, set2 = set(set2.squeeze()), set(set2.squeeze())
    else:
        set1, set2 = set(set1.index), set(set2.index)
    return len(set1.intersection(set2)) / len(set1.union(set2))
#----------------------------------------------------------------------------#

#----------------------------   Sorensen-Dice   -----------------------------#
def sorensen_index(set1, set2):
    """ Sorensen-Dice Index (Czekanowski) between two datasets

    This Sorensen-Dice index SD between two sets A and B is defined as:
        SD(A, B) = 2 * |A ∩ B| / (|A| + |B|)
    where `|.|` is the cardinal and `∩` is the intersection operator. It
    ranges from 0 to 1: 0 means the datasets are fully disjoint, and 1
    means they fully overlap.

    Parameters
    ----------
    set1, set2 : 1D np.ndarrays, Databases or Clusters
        The 1D arrays of data for which to compute the index.

    Returns
    -------
    index : float
        The Sorensen-Dice Index.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> set1 = np.arange(10., 50., dtype=float)
    >>> set2 = np.arange(15., 90., dtype=float)

    # Compute the Sorensen-Dice Index
    >>> sorensen_index(set1, set2)
    0.5454545454545454
    """
    if isinstance(set1, np.ndarray):
        set1, set2 = set(set2.squeeze()), set(set2.squeeze())
    else:
        set1, set2 = set(set1.index), set(set2.index)
    return 2 * len(set1.intersection(set2)) / (len(set1) + len(set2))
#----------------------------------------------------------------------------#

#-------------------------   Dunn Index   -------------------------#
def dunn_index(sets, distance='euclidean'):
    """ Dunn Index of a set of datasets

    Take a set of datasets and compute their Dunn Index, that is the
    ration between the minimal inter-cluster distance and the maximal
    intra-cluster distance among all the datasets. This score has no
    maximal value, but the closer to zero, the worse.

    Parameters
    ----------
    sets : list of np.ndarrays, Databases or Clusters
        The datasets for which to compute the index.
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'

    Returns
    -------
    index : float
        The Dunn Index.

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy set of clusters
    >>> clusters = [
    ...     np.linspace(0., 1., 15).reshape(-1, 3),
    ...     np.linspace(0., 2., 18).reshape(-1, 3),
    ...     np.linspace(2., 5., 21).reshape(-1, 3)]

    # Compute the Dunn Index
    >>> print(dunn_index(clusters))
    0.01772039719113884
    """
    # Check if the set of datasets is empty
    if len(sets) < 2:
        raise AssertionError(f"Not enough datasets ({len(sets)}) in `sets` "
            + "to compute the Dunn Index")

    # Get the distance to use
    fdist = get_dist_func(distance, sets[0].ndim)

    # Minimal inter-cluster distance among all the sets
    num = _min_inter(sets, fdist)
    # Maximal intra-cluster distance among all the sets
    den = np.max([_max_intra(dset, fdist) for dset in sets])

    # Compute the Dunn Index
    return num / den
#----------------------------------------------------------------------------#

#-------------------------   Bouldin-Davies Index   -------------------------#
def bouldin_davies_index(sets, stat='min', distance='euclidean'):
    """ Bouldin-Davies Index of a set of datasets

    Take a set of datasets and compute their Bouldin-Davies Index, that
    is defined as the average of the maximal ratios between the intra-
    cluster distances and the inter-cluster distances.

    To compare sets with different sizes, statistical functions are app-
    lied to the sets of the intra- and inter-cluster distances, that can
    be specified using the `stat` argument, which defines the statistics
    computed on the intra-cluster distances; the one applied to the inter-
    cluster distances is its "reverse", e.g. 'max' if `stats` is 'min'.

    With the 'min' and 'mean' strategies, the closer to zero, the better;
    it is the other way around with 'max'.

    Parameters
    ----------
    sets : list of np.ndarrays, Databases or Clusters
        The datasets for which to compute the index.
    [OPT] stat : str
        The statistical function to use for the intra- and inter-cluster
        distances, among {'min', 'mean', 'max'}; this option applies to
        the intra-cluster distance, and the inter-cluster distances uses
        the reverse strategy:
          - 'min': minimal intra- and maximal inter-cluster distances
          - 'max': maximal intra- and minimal inter-cluster distances
          - 'mean': both mean intra- and inter-cluster distances
            :Default: 'min'
    [OPT] distance : str
        The distance name; see `get_dist_func` function for details.
            :Default: 'euclidean'

    Returns
    -------
    index : float
        The Bouldin-Davies Index.

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy set of clusters
    >>> clusters = [
    ...     np.linspace(0., 1., 15).reshape(-1, 3),
    ...     np.linspace(0., 2., 18).reshape(-1, 3),
    ...     np.linspace(2., 5., 21).reshape(-1, 3)]

    # Bouldin-Davies Index with 'min' statistics
    >>> print(bouldin_davies_index(clusters, 'min'))
    0.1207062576855671

    # Bouldin-Davies Index with 'mean' statistics
    >>> print(bouldin_davies_index(clusters, 'mean'))
    0.6566757095977932

    # Bouldin-Davies Index with 'max' statistics
    >>> print(bouldin_davies_index(clusters, 'max'))
    17.849844984736475
    """
    # Check if the set of datasets is empty
    if len(sets) < 2:
        raise AssertionError(f"Not enough datasets ({len(sets)}) in `sets` "
            + "to compute the Bouldin-Davies Index")

    # Get the distance to use
    fdist = get_dist_func(distance, sets[0].ndim)

    # Get the intra/inter-cluster distances function
    if stat == 'max':
        fintra = _max_intra
        finter = _min_inter
    elif stat == 'mean':
        fintra = _mean_intra
        finter = _mean_inter
    else:
        fintra = _min_intra
        finter = _max_inter

    # Compute the Bouldin-Davies Index
    somme = 0.
    for i, clti in enumerate(sets[:-1], 1):
        intrai = fintra(clti, fdist)
        # Sum the max ratio of the intra distances over the inter distances
        somme += max((fintra(cltj, fdist) + intrai) / finter([clti, cltj], fdist)
            for j, cltj in enumerate(sets[i:]))

    # Average the maximal ratios
    return 2. * somme / (len(sets) * len(sets)-1)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Get Statistic Function                          ##
##############################################################################

def get_stat_func(statistic='quantile'):
    """ Get the reference to a specified statistic function

    Take a statistic's name and return its corresponding function; return
    the `quantile` statistic as default. Every function has signature:
        fstat(data, *args)
    where `data` is an ND array and `args` are the statistic's parameters
    (see the concerned function for details).

    Parameters
    ----------
    [OPT] statistic : str
        The statistic name among:
	        {'min', 'quantile', 'max', 'mean', 'std', 'max_to_mean',
	         'max_span', 'sphere_span', 'jaccard_index', 'sorensen_index',
	         'dunn_index', 'bouldin_davies_index'}
            :Default: 'quantile'

    Returns
    -------
    fstat : reference to a function
        The statistic function; has signature: `fstat(data, *args)`.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> data = np.arange(15., dtype=float).reshape(-1, 5)

    # Compute the 1st quantile (25%) of the dataset
    >>> fstat = get_stat_func('quantile')
    >>> quant = fstat(data, 0.25)
    >>> print(quant)
    3.5

    # Compute the Sorensen index between two datasets (must be 1D)
    >>> data = data.ravel()
    >>> fstat = get_stat_func('sorensen_index')
    >>> index = fstat(data, data+3.)
    >>> print(index)
    0.8
    """
    # pylint: disable=too-many-return-statements

    # Get the statistic function
    stat = statistic.lower()
    if stat == 'min':
        return np.min
    if stat == 'quantile':
        return np.quantile
    if stat == 'max':
        return np.max
    if stat == 'mean':
        return np.mean
    if stat == 'std':
        return np.std
    if stat == 'max_to_mean':
        return max_to_mean
    if stat == 'max_span':
        return max_span
    if stat == 'sphere_span':
        return sphere_span
    if stat == 'jaccard_index':
        return jaccard_index
    if stat == 'sorensen_index':
        return sorensen_index
    if stat == 'dunn_index':
        return dunn_index
    if stat == 'bouldin_davies_index':
        return bouldin_davies_index

    # Raise an error if the statistic is invalid
    raise ValueError(f"Wrong value `{statistic}` for `statistic`; options are:\n"
        + "\t{'min', 'quantile', 'max', 'mean', 'std', 'max_to_mean',\n"
        + "\t 'max_span', 'sphere_span', 'jaccard_index', 'sorensen_index'\n"
        + "\t 'dunn_index', 'bouldin_davies_index'}")

##############################################################################
