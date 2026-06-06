""" Linkage functions

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: March 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'linkage_single', 'linkage_complete', 'linkage_mean',
    'linkage_ward', 'get_link_func', 'linkages']

import numpy as np

from clustering.metrics._distances import get_dist_func


##############################################################################
##                            Linkage Functions                             ##
##############################################################################

#----------------------------   Single Linkage   ----------------------------#
def linkage_single(cluster_a, cluster_b, distance='euclidean'):
    """ Single linkage aggregation metric

    Minimal distance between any point within the first cluster, and any
    point within the second cluster.

    Parameters
    ----------
    cluster_a, cluster_b : np.ndarray, Database or Cluster
        The two clusters whose single linkage coefficient to compute.
    [OPT] distance : str
        The name of the distance function to use; see the `get_dist_func`
        function for details.
            :Default: 'euclidean'

    Returns
    -------
    link : float
        The single linkage aggregation value.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> clt1 = np.arange(10, 40, dtype=float).reshape(-1, 5)
    >>> clt2 = np.arange(20, 80, dtype=float).reshape(-1, 5)

    # Compute the linkage distance
    >>> linkage_single(clt1, clt2, 'euclidean')
    0.0
    """

    # Get the distance
    fdist = get_dist_func(distance, cluster_a.ndim)

    # Analyse the two clusters' data
    return min(min(fdist(data, cluster_b[:], 1)) for data in cluster_a)
#----------------------------------------------------------------------------#

#---------------------------   Complete Linkage   ---------------------------#
def linkage_complete(cluster_a, cluster_b, distance='euclidean'):
    """ Complete linkage aggregation metric

    Maximal distance between any point within the first cluster, and any
    point within the second cluster.

    Parameters
    ----------
    cluster_a, cluster_b : np.ndarray, Database or Cluster
        The two clusters whose complete linkage coefficient to compute.
    [OPT] distance : str
        The name of the distance function to use; see the `get_dist_func`
        function for details.
            :Default: 'euclidean'

    Returns
    -------
    link : float
        The complete linkage aggregation value.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> clt1 = np.arange(10, 40, dtype=float).reshape(-1, 5)
    >>> clt2 = np.arange(20, 80, dtype=float).reshape(-1, 5)

    # Compute the linkage distance
    >>> linkage_complete(clt1, clt2, 'euclidean')
    145.34441853748632
    """

    # Get the distance
    fdist = get_dist_func(distance, cluster_a.ndim)

    # Analyse the two clusters' data
    return max(max(fdist(data, cluster_b[:], 1)) for data in cluster_a)
#----------------------------------------------------------------------------#

#-----------------------------   Mean Linkage   -----------------------------#
def linkage_mean(cluster_a, cluster_b, distance='euclidean'):
    """ Mean linkage aggregation metric

    Mean distance between any point within the first cluster, and any
    point within the second cluster.

    Parameters
    ----------
    cluster_a, cluster_b : np.ndarray, Database or Cluster
        The two clusters whose mean linkage coefficient to compute.
    [OPT] distance : str
        The name of the distance function to use; see the `get_dist_func`
        function for details.
            :Default: 'euclidean'

    Returns
    -------
    link : float
        The mean linkage aggregation value.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> clt1 = np.arange(10, 40, dtype=float).reshape(-1, 5)
    >>> clt2 = np.arange(20, 80, dtype=float).reshape(-1, 5)

    # Compute the linkage distance
    >>> linkage_mean(clt1, clt2, 'euclidean')
    59.00734940624446
    """

    # Get the distance
    fdist = get_dist_func(distance, cluster_a.ndim)

    # Cumulative distances (array)
    mean = sum(fdist(data, cluster_b[:], 1) for data in cluster_a)

    return sum(mean) / (cluster_a.shape[0]*cluster_b.shape[0])
#----------------------------------------------------------------------------#

#-----------------------------   Ward Linkage   -----------------------------#
def linkage_ward(cluster_a, cluster_b, distance='euclidean'):
    """ Ward linkage aggregation metric

    Ward distance between any point within the first cluster, and any
    point within the second cluster.

    Parameters
    ----------
    cluster_a, cluster_b : np.ndarray, Database or Cluster
        The two clusters whose Ward linkage coefficient to compute.
    [OPT] distance : str
        The name of the distance function to use; see the `get_dist_func`
        function for details.
            :Default: 'euclidean'

    Returns
    -------
    link : float
        The Ward linkage aggregation value.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> clt1 = np.arange(10, 40, dtype=float).reshape(-1, 5)
    >>> clt2 = np.arange(20, 80, dtype=float).reshape(-1, 5)

    # Compute the linkage distance
    >>> linkage_ward(clt1, clt2, 'euclidean')
    12500.0
    """

    # Get the distance
    fdist = get_dist_func(distance, cluster_a.ndim)

    card_a = cluster_a.shape[0]
    card_b = cluster_b.shape[0]

    # Means of the two clusters
    mean_a = cluster_a.mean(0)
    mean_b = cluster_b.mean(0)

    return (fdist(mean_a, mean_b, 0)**2 * (card_a*card_b) / (card_a+card_b))
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Get Linkage Function                           ##
##############################################################################

def get_link_func(linkage='ward'):
    """ Get the reference to a specified linkage function

    Take a linkage's name and return its corresponding function; return
    the `war` linkage as default. Every function has signature:
        flink(clt1, clt2, distance)
    where `clt1` and `clt2` are ND arrays.

    Parameters
    ----------
    [OPT] linkage : str
        The linkage name among: {'single', 'complete', 'mean', 'ward'}.
            :Default: 'ward'

    Returns
    -------
    flink : reference to a function
        The linkage function; has signature:
            `flink(clt1, clt2, distance)`

    Examples
    --------
    >>> import numpy as np

    # Get the linkage function
    >>> flink = get_link_func('ward')

    # Compute the linkage between 1D arrays
    >>> arr = np.arange(50., dtype=float)
    >>> flink(arr, arr+1.)
    2.5

    # Compute the linkage between ND arrays
    >>> mat = arr.reshape(-1, 5)
    >>> flink(mat, mat+1.)
    25.000000000000007
    """

    # Get the linkage function
    link = linkage.lower()
    if link == 'ward':
        return linkage_ward
    if link == 'single':
        return linkage_single
    if link == 'complete':
        return linkage_complete
    if link == 'mean':
        return linkage_mean

    # Raise an error if the linkage is invalid
    raise ValueError(f"Wrong value `{linkage}` for `linkage`; options are:\n"
        + "\t{'single', 'complete', 'mean', 'ward'}")

##############################################################################



##############################################################################
##                      Linkage Distances Computation                       ##
##############################################################################

def linkages(clusters, linkage='ward', distance='euclidean'):
    """ Linkage distance between any possible 2-cluster set

    Compute all the linkage indices of a list of clusters: for each one of
    them, compute its distance to any other cluster of the list and gather
    all these distances within a single list, and return it.

    Parameters
    ----------
    clusters : list of np.ndarrays, Databases or Clusters
        The list of clusters to analyze (~ data clusters).
    [OPT] linkage : str
        The linkage name among: {'single', 'complete', 'mean', 'ward'}.
            :Default: 'ward'
    [OPT] distance : str
        The distance name among: {'manhattan', 'euclidean', 'canberra',
        'cosine', 'tanimoto', 'czekanowski'}.
            :Default: 'euclidean'

    Returns
    -------
    links : list of floats
        The list of any linkage distance of the list of clusters; has
        N*(N-1)/2 values, where N is the length of `clusters`.

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy list of clusters
    >>> arr = np.arange(50., dtype=float)
    >>> arrs = [arr+i for i in range(4)]

    # Compute the linkages between every cluster of the list
    >>> linkages(arrs, 'ward', 'euclidean')     # 4*3/2=6 values
    array([ 25., 100., 225.,  25., 100.,  25.])
    """

    # Get the linkage function
    flink = get_link_func(linkage)

    # Linkage distance between any possible 2-cluster set
    pos = 0
    links = np.zeros(int(len(clusters)*(len(clusters)-1)/2), float)
    for i, clusti in enumerate(clusters, 1):
        for clustj in clusters[i:]:
            # Linkage index from cluster i to cluster j
            if clusti.size == 0 and clustj.size == 0:
                links[pos] = np.nan#-1.
            else:
                links[pos] = flink(clusti, clustj, distance)
            pos += 1
    #links[links == -1.] = max(links)
    return links

##############################################################################
