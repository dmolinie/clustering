""" Distance functions

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: March 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'manhattan_1d', 'manhattan_nd', 'euclidean_1d', 'euclidean_nd',
    'minkowski_1d', 'minkowski_nd', 'canberra_1d', 'canberra_nd',
    'cosine_1d', 'cosine_nd', 'tanimoto_1d', 'tanimoto_nd',
    'czekanowski_1d', 'czekanowski_nd', 'hausdorff', 'get_dist_func']

import numpy as np


##############################################################################
##                            Distance Functions                            ##
##############################################################################

# Axis = 0 --> arr / arr
# Axis = 1 --> arr / mat

#------------------------------   Manhattan   -------------------------------#
def manhattan_1d(arr, mat, axis=0):
    """ Manhattan distance for 1D data """
    dist = np.abs(arr - mat)
    return dist if axis > 0 else np.sum(dist)

def manhattan_nd(arr, mat, axis=0):
    """ Manhattan distance for ND data """
    return np.sum(np.abs(arr - mat), axis)
#----------------------------------------------------------------------------#

#------------------------------   Euclidean   -------------------------------#
def euclidean_1d(arr, mat, axis=0):
    """ Euclidean distance for 1D data """
    dist = np.sqrt((arr - mat)**2)
    return dist if axis > 0 else np.sum(dist)

def euclidean_nd(arr, mat, axis=0):
    """ Euclidean distance for ND data """
    return np.sqrt(np.sum((arr - mat)**2, axis))
#----------------------------------------------------------------------------#

#------------------------------   Minkowski   -------------------------------#
def minkowski_1d(arr, mat, axis=0, power=1):
    """ Minkowski distance for 1D data """
    dist = np.pow(arr - mat, power)
    return dist if axis > 0 else np.sum(dist)

def minkowski_nd(arr, mat, axis=0, power=1):
    """ Minkowski distance for ND data """
    return (np.sum(np.abs(arr - mat)**power, axis))**(1/power)
#----------------------------------------------------------------------------#

#-------------------------------   Canberra   -------------------------------#
def canberra_1d(arr, mat, axis=0):
    """ Canberra distance for 1D data """
    dist = np.abs(arr - mat) / (np.abs(arr) + np.abs(mat))
    return dist if axis > 0 else np.sum(dist)

def canberra_nd(arr, mat, axis=0):
    """ Canberra distance for ND data """
    return np.sum(np.abs(arr - mat) / (np.abs(arr) + np.abs(mat)), axis)
#----------------------------------------------------------------------------#

#--------------------------------   Cosine   --------------------------------#
def cosine_1d(arr, mat, axis=0):
    """ Cosine distance for 1D data """
    dist = np.arccos(np.sign(arr * mat))
    return dist if axis > 0 else np.sum(dist)

def cosine_nd(arr, mat, axis=0):
    """ Cosine distance for ND data """
    return np.arccos(np.sum(arr * mat, axis)
                     / np.sqrt((np.sum(arr**2, 0) * np.sum(mat**2, axis))))
#----------------------------------------------------------------------------#

#-------------------------------   Tanimoto   -------------------------------#
def tanimoto_1d(arr, mat, axis=0):
    """ Tanimoto distance for 1D data """
    dot = (arr * mat)**2
    dist = dot / (arr**2 - dot + mat**2)
    return dist if axis > 0 else np.sum(dist)

def tanimoto_nd(arr, mat, axis=0):
    """ Tanimoto distance for ND data """
    dot = np.sum((arr * mat)**2, axis)
    return dot / (np.sum(arr**2, 0) - dot + np.sum(mat**2, axis))
#----------------------------------------------------------------------------#

#-------------------------   Czekanowski Distance   -------------------------#
def czekanowski_1d(arr, mat, axis=0):
    """ Czekanowski distance for 1D data """
    dist = 1. - 2*np.minimum(arr, mat) / (arr+mat)
    return dist if axis > 0 else np.sum(dist)

def czekanowski_nd(arr, mat, axis=0):
    """ Czekanowski distance for ND data """
    return 1. - 2*np.sum(np.minimum(arr, mat), axis) / np.sum(arr+mat, axis)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Get Distance Function                           ##
##############################################################################

def get_dist_func(distance='euclidean', ndim=None):
    """ Get the reference to a specified distance function

    Take a distance's name and return its corresponding function; return
    the `euclidean` distance as default. Every function has signature:
        fdist(arr, mat, axis=0)
    where `arr` is a 1D array and `mat` can either be a 1D or 2D array.
    The distance is the sum of the element-wise distances between `arr`
    and any array of `mat` (or `mat` itself if 1D array). Use the `axis`
    argument to specify the `axis` along which to sum the distances.

    If the `ndim` argument is set to 1, the returned function has shape:
        fdist(arr1, arr2, axis=0)
    where `arr1` and `arr2` are both 1D arrays. In such a case, the dis-
    tances are computed element-wise and summation is done only if `axis`
    is greater than 0.

    Parameters
    ----------
    [OPT] distance : str
        The distance name among: {'manhattan', 'euclidean', 'canberra',
        'cosine', 'tanimoto', 'czekanowski'}.
            :Default: 'euclidean'
    [OPT] ndim : int
        If `1`, the returned function computes and returns the element-
        wise distances (no summation); else, the function computes and
        returns the array-wise distances.
            :Default: None (ND data supposed)

    Returns
    -------
    fdist : reference to a function
        The distance function; has signature:
            {`fdist(arr1, arr2, axis0)`  if `ndim` is 1
            {`fdist(arr, mat, axis=0)`   else

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> arr = np.arange(5., dtype=float)
    >>> mat = np.arange(15., dtype=float).reshape(-1, 5)

    # Get the function distance for 1D data
    >>> fdist = get_dist_func('euclidean', 1)

    # Compute the distances for 1D data
    >>> fdist(arr, arr+1., 0)
    2.23606797749979
    >>> fdist(arr, arr+1., 1)
    array([1., 1., 1., 1., 1.])

    # Get the function distance for 2D data
    >>> fdist = get_dist_func('euclidean', 2)

    # Compute the distances for 2D data
    >>> fdist(arr, arr+1., 0)
    2.23606797749979
    >>> fdist(arr, mat, 0)
    array([11.18033989, 11.18033989, 11.18033989, 11.18033989, 11.18033989])
    >>> fdist(arr, mat, 1)
    array([ 0.        , 11.18033989, 22.36067977])
    """

    # Get the distance function
    dist = distance.lower()
    if dist == 'euclidean':
        return euclidean_1d if ndim == 1 else euclidean_nd
    if dist == 'manhattan':
        return manhattan_1d if ndim == 1 else manhattan_nd
    if dist == 'canberra':
        return canberra_1d if ndim == 1 else canberra_nd
    if dist == 'cosine':
        return cosine_1d if ndim == 1 else cosine_nd
    if dist == 'tanimoto':
        return tanimoto_1d if ndim == 1 else tanimoto_nd
    if dist == 'czekanowski':
        return czekanowski_1d if ndim == 1 else czekanowski_nd

    # Raise an error if the distance is invalid
    raise ValueError(f"Wrong value `{distance}` for `distance`; options are:\n"
        + "\t{'manhattan', 'euclidean', 'canberra', 'cosine', 'tanimoto', 'czekanowski'}")

##############################################################################



##############################################################################
##                              Miscellaneous                               ##
##############################################################################

#---------------------   Modified Hausdorff Distance   ----------------------#
def hausdorff(curve1, curve2, axis=1, distance='euclidean'):
    """ Modified Hausdorff Distance between two curves (2D vectors) """

    # Get the distance
    fdist = get_dist_func(distance, curve1.ndim)

    # Compute the Hausdorff distances
    dist = np.zeros((len(curve1), len(curve2)), dtype=float)
    for i, val in enumerate(curve1):
        dist[i] = fdist(val, curve2, axis)

    return (sum(np.min(dist, 0)) + sum(np.min(dist, 1))) / 2
#----------------------------------------------------------------------------#

##############################################################################
