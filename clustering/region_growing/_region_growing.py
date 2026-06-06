""" Scikit-Learn DBSCAN & OPTICS shorthand functions

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: November 2023
Last revised: May 2026

License: GPLv3
"""

__all__ = ['build',
    'dbscan_params', 'dbscan_cluster', 'optics_params', 'optics_cluster']

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS

import clustering.tools as tls
import clustering.formats as dformat


##############################################################################
##                     Data checking & Cluster building                     ##
##############################################################################

#--------------------------   Build the Clusters   --------------------------#
def build(data, labels, cluster=True):
    """ Build the clusters from the list of the classes

    Take an ND database and the corresponding labels for each data, and
    build the corresponding groups (clusters), as either a simple list
    of datasets, or as a list of Cluster objects.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to be distributed to the different clusters.
    labels : list, tuple or np.ndarray
        The category of every data: value k at index m means that the
        data at index m in `data` belongs to cluster k.
    [OPT] cluster : bool
        Return the trained clusters as a list of np.ndarrays (False) or
        as a list of Cluster objects (True).
            :Default: True

    Returns
    -------
    clusters : list of Cluster objects
        The final clusters.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from sklearn.cluster import DBSCAN

    # Generate a dummy database with 3 Gaussian distributions (regions)
    >>> rng = np.random.default_rng()
    >>> array = np.vstack(
    ...     (rng.normal(loc=0.5, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=5.0, scale=0.01, size=(100, 3)),
    ...      rng.normal(loc=9.5, scale=0.01, size=(100, 3))))
    >>> database = Database(array, np.arange(len(array)))

    # Train DBSCAN
    >>> dbscan = DBSCAN().fit(database)

    # Build the clusters
    >>> clusters = build(database, dbscan.labels_, cluster=True)
    >>> print(len(clusters))
    3
    """

    # List of np.ndarrays
    if not cluster:
        return [(data[np.where(labels == k)]) for k in set(labels)]

    # List of Clusters
    nodes = []
    for k in set(labels):
        mask = np.where(labels == k)
        nodes.append(
            dformat.Cluster(
                data[mask],
                data.index[mask],
                pattern=data[mask].mean(0),
                tags=dformat.get_tags(data),
                classes=dformat.get_classes(data, mask)))
    return nodes
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Scikit Learn DBSCAN                            ##
##############################################################################

#----------------------   DBSCAN Parameters Checking   ----------------------#
def dbscan_params(**params):
    """ Check the parameters for DBSCAN clustering

    Take up to 6 keyword arguments that define the required parameters
    for the instantiation and training of DBSCAN, check if the expected
    keys are present and fulfill the missing keys with default values.
    Finally return a fulfilled 5-key dictionary. If no argument is fed
    into the function, set the default values within a new 5-key dict-
    ionary and return it.

    Refer to the Scikit-Learn's `DBSCAN` class' documentation for addi-
    tional information about the keys (inline keyword parameters).

    Parameters
    ----------
    [OPT] eps : float
        The maximum distance between two samples for one to be consi-
        dered in the neighborhood of the other.
            :Default: 0.5
    [OPT] min_samples : int
        The number of samples in a neighborhood for a point to be con-
        sidered a core point.
            :Default: 5
    [OPT] metric : str
        The metric (distance) to use when calculating distance between
        instances in a feature array.
            :Default: 'minkowski'
    [OPT] p : float
        The power of the Minkowski metric to be used to calculate dist-
        ance between points.
            :Default: 2 (thus Euclidean)
    [OPT] cluster : bool
        Return the trained clusters as a list of `np.ndarray` (False) or
        `Cluster` class objects (True).
            :Default: True
    + Any of the `DBSCAN` class from the `sklearn.cluster` module.

    Returns
    -------
    kparams : 5-key dict
        The correctly fulfilled dictionary containing any of the keys.

    Examples
    --------
    # Get the default parameters
    >>> print(dbscan_params())
    {'eps': 0.5,
     'min_samples': 5,
     'metric': 'minkowski',
     'p': 2,
     'cluster': True}

    # Change only the threshold and leave the other default params
    >>> print(dbscan_params(eps=0.5))
    {'eps': 0.5,
     'min_samples': 5,
     'metric': 'minkowski',
     'p': 2,
     'cluster': True}

    # Change all the parameters
    >>> print(dbscan_params(
    ...     eps=0.5, min_samples=5, cluster=True,
    ...     metric='minkowski', p=2))
    {'eps': 0.5,
     'min_samples': 5,
     'metric': 'minkowski',
     'p': 2,
     'cluster': True}
    """
    # Default values
    kparams = {
        'eps': 0.5, 'min_samples': 5,
        'metric': 'minkowski', 'p': 2, 'cluster': True}
    # Check the parameters
    return tls.check_keys(params, kparams)
#----------------------------------------------------------------------------#

#----------------------   DBSCAN Training & Building   ----------------------#
def dbscan_cluster(data, **params):
    """ Instantiate & train DBSCAN on a dataset and cluster it

    Take a dataset and the parameters for the clustering, train DBSCAN
    on it and build the clusters.

    Refer to the Scikit-Learn's `DBSCAN` class for details, as this
    function is a shorthand version of it.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to train DBSCAN and to cluster afterwards.

    Other Parameters
    ----------------
    **params : inline keyword arguments, optional
        The parameters passed to the `dbscan_params` function for checking.

    Returns
    -------
    clusters : list of np.ndarrays or Clusters
        The clusters obtained after having trained DBSCAN, represented
        by some sort of objects (depend on the parameters).

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

    # Cluster the database with DBSCAN with default parameters
    >>> clusters = dbscan_cluster(database)
    >>> print(len(clusters))
    3

    # Cluster the database with a specific threshold
    >>> clusters = dbscan_cluster(database, eps=0.5)
    >>> print(len(clusters))
    3
    """

    # Check data type and format
    data = dformat.check_data(data)

    # Ensure the data are 2D (required by Scikit-Learn)
    if data.ndim == 1:
        if isinstance(data, (dformat.Database, dformat.Cluster)):
            data.value = data.value.reshape(-1, 1)
        else:
            data = data.reshape(-1, 1)

    # Parameters to use
    kparams = dbscan_params(**params)
    cluster = kparams.pop('cluster')

    # Train DBSCAN
    dbscan = DBSCAN(**kparams).fit(data)

    # Build the clusters
    return build(data, dbscan.labels_, cluster)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Scikit Learn OPTICS                            ##
##############################################################################

#----------------------   OPTICS Parameters Checking   ----------------------#
def optics_params(**params):
    """ Check the parameters for OPTICS clustering

    Take up to 6 keyword arguments that define the required parameters
    the instantiation and training of OPTICS, check if the expected
    keys are present and fulfill the missing keys with default values.
    Finally return a fulfilled 5-key dictionary. If no argument is fed
    into the function, set the default values within a new 5-key dict-
    ionary and return it.

    Refer to the Scikit-Learn's `OPTICS` class' documentation for addi-
    tional information about the keys (inline keyword parameters).

    Parameters
    ----------
    [OPT] min_samples : int
        The number of samples in a neighborhood for a point to be con-
        sidered a core point.
            :Default: 5
    [OPT] max_eps : float
        The maximum distance between two samples for one to be consi-
        dered in the neighborhood of the other.
            :Default: np.inf
    [OPT] metric : str
        The metric (distance) to use when calculating distance between
        instances in a feature array.
            :Default: 'minkowski'
    [OPT] p : float
        The power of the Minkowski metric to be used to calculate dist-
        ance between points.
            :Default: 2 (thus Euclidean)
    [OPT] cluster : bool
        Return the trained clusters as a list of `np.ndarray` (False) or
        `Cluster` class objects (True).
            :Default: True
    + Any of the `OPTICS` class from the `sklearn.cluster` module.

    Returns
    -------
    kparams : 5-key dict
        The correctly fulfilled dictionary containing any of the keys.

    Examples
    --------
    # Get the default parameters
    >>> print(optics_params())
    {'min_samples': 5,
     'max_eps': inf,
     'metric': 'minkowski',
     'p': 2,
     'cluster': True}

    # Change only the threshold and leave the other default params
    >>> print(optics_params(max_eps=10.))
    {'min_samples': 5,
     'max_eps': inf,
     'metric': 'minkowski',
     'p': 2,
     'cluster': True}

    # Change all the parameters
    >>> print(optics_params(
    ...     min_samples=5, max_eps=10., cluster=True,
    ...     metric='minkowski', p=2))
    {'min_samples': 5,
     'max_eps': 10.0,
     'metric': 'minkowski',
     'p': 2,
     'cluster': True}
    """
    # Default values
    kparams = {
        'min_samples': 5, 'max_eps': np.inf,
        'metric': 'minkowski', 'p': 2, 'cluster': True}
    # Check the parameters
    return tls.check_keys(params, kparams)
#----------------------------------------------------------------------------#

#----------------------   OPTICS Training & Building   ----------------------#
def optics_cluster(data, **params):
    """ Instantiate & train OPTICS on a dataset and cluster it

    Take a dataset and the parameters for the clustering, train OPTICS
    on it and build the clusters.

    Refer to the Scikit-Learn's `OPTICS` class for details, as this
    function is a shorthand version of it.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to train OPTICS and to cluster afterwards.

    Other Parameters
    ----------------
    **params : inline keyword arguments, optional
        The parameters passed to the `optics_params` function for checking.

    Returns
    -------
    clusters : list of np.ndarrays or Clusters
        The clusters obtained after having trained OPTICS, represented
        by some sort of objects (depend on the parameters).

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

    # Cluster the database with DBSCAN with default parameters
    >>> clusters = optics_cluster(database)
    >>> print(len(clusters))
    8

    # Cluster the database with a specific threshold
    >>> clusters = optics_cluster(database, max_eps=0.5)
    >>> print(len(clusters))
    9
    """

    # Check data type and format
    data = dformat.check_data(data)

    # Ensure the data are 2D (required by Scikit-Learn)
    if data.ndim == 1:
        if isinstance(data, (dformat.Database, dformat.Cluster)):
            data.value = data.value.reshape(-1, 1)
        else:
            data = data.reshape(-1, 1)

    # Parameters to use
    kparams = optics_params(**params)
    cluster = kparams.pop('cluster')

    # Train OPTICS
    optics = OPTICS(**kparams).fit(data)

    # Build the clusters
    return build(data, optics.labels_, cluster)
#----------------------------------------------------------------------------#

##############################################################################
