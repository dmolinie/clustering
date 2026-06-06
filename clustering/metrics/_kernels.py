""" Kernel functions

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: March 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'linear_1d', 'linear_nd', 'circles_1d', 'circles_nd', 'gaussian_1d',
    'gaussian_nd', 'polynomial_1d', 'polynomial_nd', 'get_ker_func']

import numpy as np


##############################################################################
##                             Kernel Functions                             ##
##############################################################################

#----------------------------   Linear Kernel   -----------------------------#
def linear_1d(arr, mat, axis=0):
    """ Linear kernel for 1D data """
    ker = arr * mat
    return ker if axis > 0 else np.sum(ker)

def linear_nd(arr, mat, axis=0):
    """ Linear kernel for ND data """
    return np.sum(arr*mat, axis)
#----------------------------------------------------------------------------#

#----------------------------   Circle Kernel   -----------------------------#
def circles_1d(arr, mat, axis=0, mean=5.):
    """ Circle kernel for 1D data """
    ker = (arr-mean)*(mat-mean) + (arr-mean)**2 * (mat-mean)**2
    return ker if axis > 0 else np.sum(ker)

def circles_nd(arr, mat, axis=0, mean=5.):
    """ Circle kernel for ND data """
    return (np.sum((arr-mean)*(mat-mean), axis)
            + np.sum((arr-mean)**2) * np.sum((mat-mean)**2, axis))
#----------------------------------------------------------------------------#

#---------------------------   Gaussian Kernel   ----------------------------#
def gaussian_1d(arr, mat, axis=0, sigma=0.05):
    """ Gaussian kernel for 1D data  """
    ker = np.exp(-(arr-mat)**2 / (2.*sigma**2))
    return ker if axis > 0 else np.sum(ker)

def gaussian_nd(arr, mat, axis=0, sigma=0.05):
    """ Gaussian kernel for ND data  """
    return np.exp(-np.sum((arr-mat)**2, axis) / (2.*sigma**2))
#----------------------------------------------------------------------------#

#--------------------------   Polynomial Kernel   ---------------------------#
def polynomial_1d(arr, mat, axis=0, order=2):
    """ Polynomial kernel for 1D data  """
    ker = (1 + arr*mat)**order
    return ker if axis > 0 else np.sum(ker)

def polynomial_nd(arr, mat, axis=0, order=2):
    """ Polynomial kernel for ND data  """
    return (1 + np.sum(arr*mat, axis))**order
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Get Kernel Function                            ##
##############################################################################

def get_ker_func(kernel='linear', ndim=None):
    """ Get the reference to a specified kernel function

    Take a kernel's name and return its corresponding function; return
    the `linear` kernel as default. Every function has signature:
        fker(arr, mat, axis=0)
    where `arr` is a 1D array and `mat` can either be a 1D or 2D array.
    The distance is the sum of the element-wise distances between `arr`
    and any array of `mat` (or `mat` itself if 1D array). Use the `axis`
    argument to specify the `axis` along which to sum the distances.

    If the `ndim` argument is set to 1, the returned function has shape:
        fker(arr1, arr2)
    where `arr1` and `arr2` are both 1D arrays. In such a case, the dis-
    tances are computed element-wise and no summation is done.


    Parameters
    ----------
    [OPT] kernel : str
        The kernel name among:
            {'linear', 'circles', 'gaussian', 'polynomial'}.
            :Default: 'linear'
    [OPT] ndim : int
        If `1`, the returned function computes and returns the element-
        wise distances (no summation); else, the function computes and
        returns the array-wise distances.
            :Default: None (ND data supposed)

    Returns
    -------
    fker : reference to a function
        The kernel function; has signature:
            {`fker(arr1, arr2)`          if `ndim` is 1
            {`fker(arr, mat, axis=0)`    else

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> arr = np.arange(5., dtype=float)
    >>> mat = np.arange(15., dtype=float).reshape(-1, 5)

    # Get the function distance for 1D data
    >>> fker = get_ker_func('linear', 1)

    # Compute the distances for 1D data
    >>> fker(arr, arr+1., 0)
    40.0
    >>> fker(arr, arr+1., 1)
    array([ 0.,  2.,  6., 12., 20.])

    # Get the function distance for ND data
    >>> fker = get_ker_func('linear', 2)

    # Compute the distances for ND data
    >>> fker(arr, arr+1., 0)
    40.0
    >>> fker(arr, mat, 0)
    array([  0.,  18.,  42.,  72., 108.])
    >>> fker(arr, mat, 1)
    array([ 30.,  80., 130.])
    """

    # Get the kernel function
    kern = kernel.lower()
    if kern == 'linear':
        return linear_1d if ndim == 1 else linear_nd
    if kern == 'circles':
        return circles_1d if ndim == 1 else circles_nd
    if kern == 'gaussian':
        return gaussian_1d if ndim == 1 else gaussian_nd
    if kern == 'polynomial':
        return polynomial_1d if ndim == 1 else polynomial_nd

    # Raise an error if the kernel is invalid
    raise ValueError(f"Wrong value `{kernel}` for `kernel`; options are:\n"
        + "\t{'linear', 'circles', 'gaussian', 'polynomial'}")

##############################################################################
