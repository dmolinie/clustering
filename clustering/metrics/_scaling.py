""" Functions to rescale a dataset

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['extrema', 'check_scale', 'rescale']

import numpy as np


##############################################################################
##                           Rescaling Functions                            ##
##############################################################################

#-------------------------   Data Extremal Values   -------------------------#
def extrema(data, axis=None, **kwargs):
    """ Minimal and maximal values of a dataset

    Parameters
    ----------
    data : np.ndarray, Database or Cluster
        The data of extremal values to find.
    [OPT] axis : int
        The axis along which to compute the extremal values.
            :Default: None (all elements' min and max)

    Returns
    -------
    min : float or np.ndarray
        The minimal values in the data.
    max : float or np.ndarray
        The maximal values in the data.

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments, optional
        The positional parameters for the NumPy's `min` and `max` func-
        tions. These arguments are passed to both functions.

    Examples
    --------
    >>> import numpy as np
    >>> extrema(np.arange(50., dtype=float).reshape(-1, 5))
    (0.0, 49.0)
    """
    return (np.min(data, axis, **kwargs), np.max(data, axis, **kwargs))
#----------------------------------------------------------------------------#

#---------------------------   Check Data Scale   ---------------------------#
def check_scale(data, bounds, rtol=1e-2, axis=0, **kwargs):
    """ Check if a set of data is scaled between `bounds`

    Take a set of data and check if they are scaled between `bounds`;
    to this end, get the min and max values of `data` along `axis` and
    compare them to the values of `bounds` (scalars or vectors); if the
    relative difference is less than `rtol`, consider they are close and
    thus are scaled. Compare both the minimum and maximum of `data`.

    See the NumPy's `isclose` function for details.

    Parameters
    ----------
    data : np.ndarray or Database
        The data to train the KSOM.
    bounds : 2-tuple of floats or 1D np.ndarrays
        The range within which the data must be to consider them scaled,
        organized as (min, max). Can be single scalars, or arrays of the
        same length as the last dimension of `data`.
    [OPT] rtol : float
        The relative tolerance; see function `np.isclose` for details.
            :Default: 1e-2
    [OPT] axis : int
        The axis along which to compute the min and max values of `data`;
        see `extrema` function for details.
            :Default: 0 (column-wise)

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments, optional
        The positional parameters for the NumPy's `min` and `max` func-
        tions. These arguments are passed to both functions.

    Returns
    -------
    scale : bool
        If the data are already scaled, i.e. if their min and max values
        are close to `bounds[0]` and `bounds[1]`, respectively.
    limits : 2-tuple of np.ndarrays
        The min and max values of `data` along `axis`. Returned to avoid
        recomputing them afterwards in case of rescaling (they can be
        directly passed to the `rescale` function's `limits` argument).

    Examples
    --------
    >>> import numpy as np

    #--- Scalar bounds
    # Generate dummy data
    >>> data = np.arange(100, dtype=float).reshape(-1, 5)

    # Check if the data are already scaled between 0 and 1
    >>> scale, limits = check_scale(data, (0., 1.))
    >>> print(scale)
    False
    >>> print(limits)
    (array([0., 1., 2., 3., 4.]),
     array([95., 96., 97., 98., 99.]))

    # Rescale the data with the `rescale` function using the returned `limits`
    >>> data, limits = rescale(data, bounds=(0., 1.), limits=limits)
    >>> print(data.min(0), data.max(0))
    [0. 0. 0. 0. 0.] [1. 1. 1. 1. 1.]

    # Check again if the data are scaled between 0 and 1
    >>> scale, limits = check_scale(data, (0., 1.))
    >>> print(scale)
    True

    #--- Vector bounds
    # Generate dummy data between 0 and 1
    >>> data = np.linspace(0., 1., 100).reshape(-1, 5)

    # Check if the data are already scaled between 0 and 1
    >>> scale, limits = check_scale(data, (0., 1.))
    >>> print(scale)     # False, as all the dimensions are not 0. and 1.
    False
    >>> print(limits)
    (array([0.        , 0.01010101, 0.02020202, 0.03030303, 0.04040404]),
     array([0.95959596, 0.96969697, 0.97979798, 0.98989899, 1.        ]))

    # Check if the data are scaled with vector bounds
    >>> scale, limits = check_scale(data, limits)
    >>> print(scale)
    True
    >>> print(limits)
    (array([0.        , 0.01010101, 0.02020202, 0.03030303, 0.04040404]),
     array([0.95959596, 0.96969697, 0.97979798, 0.98989899, 1.        ]))
    """

    # Find the column-wise extremal values in the dataset
    limits = extrema(data, axis, **kwargs)

    # Check whether the dataset is already normalized or not
    t_min = np.isclose(limits[0], bounds[0], rtol).all()
    t_max = np.isclose(limits[1], bounds[1], rtol).all()

    # Set `_norm` to True if data already normalized, or False else
    scale = t_min and t_max

    return scale, limits
#----------------------------------------------------------------------------#

#------------------------   Database Normalization   ------------------------#
def rescale(data, bounds=(0., 1.), limits=None, axis=0):
    r""" Rescale a dataset

    Take the data to rescale (e.g. to normalize), its original `limits`
    (i.e. the lower and upper bounds of the interval within which the
    data in their current scale exist) and the `bounds` of the interval
    within which to rescale the data. This can be formalized as:
        data \in [limits[0]; limits[1]]
    which are linearly rescaled to exist inside [bounds[0]; bounds[1]].

    The rescaling is linear, and is expressed as:
        data_norm = (M2-m2) * (data-m1) / (M1-m1) + m2
    where (m1, M1) = `limits` and (m2, M2) = `bounds`, `m` referring to
    the lower bounds, and `M` to the upper bounds.

    If `limits` is not provided, it is the min and max along the speci-
    fied `axis` which are used as lower and upper limits; if `bounds` is
    not provided, the data are rescaled between 0 and 1.

    Parameters
    ----------
    data : np.ndarray, Database or Cluster
        The data to rescale.
    [OPT] bounds : 2-tuple
        The lower & upper bounds of the interval into which to rescale
        the data, i.e. the new min & max; if not provided, use 0 and 1.
            :Default: (0., 1.)
    [OPT] axis : int
        The axis along which to get the min & max of the data; used only
        if `limits` is not provided.
            :Default: 0
    [OPT] limits : 2-tuple
        The lower & upper bounds of the interval within which the data
        actually live; if not provided, use the min & max of the data.
        Essentially aims to save computation if already computed prior.
            :Default: None (get the data min and max)

    Returns
    -------
    data_resc : same type as the `data` argument
        The normalized data.
    limits : 2-tuple of floats
        The min & max values of the data, organized as (min, max).

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> data = np.arange(50., dtype=float).reshape(-1, 5)

    # Normalize the data
    >>> data_norm, limits = rescale(data, bounds=(0., 1.))

    # Denormalize the data
    >>> data_denorm, limits = rescale(
    ...     data_norm, bounds=limits, limits=(0., 1.))

    >>> print(extrema(data) == extrema(data_denorm))
    True
    """

    # Copy the input data (to preserve its type)
    data = data.copy()

    # New bounds for the rescaled data
    bounds = np.array(bounds, float)

    # If not provided, get the data extremal values
    if limits is None:
        limits = np.array(extrema(data, axis), float)
    else:
        limits = np.array(limits, float)

    # If the limits & bounds are the same, do nothing
    if np.isin(limits, bounds).all() and np.isin(bounds, limits).all():
        return data, limits

    # Operate the normalization between the upper and lower bounds
    coef = (bounds[1] - bounds[0]) / (limits[1] - limits[0])
#    data[:] = coef*(data[:]-limits[0]) + bounds[0]
    data = coef*(data-limits[0]) + bounds[0]
    return data, limits
#----------------------------------------------------------------------------#

##############################################################################
