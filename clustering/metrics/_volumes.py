""" Hyper-volumes and dataset's span

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['hypercube', 'hypersphere', 'get_vol_func']

import math

import numpy as np


##############################################################################
##                              Hyper-Volumes                               ##
##############################################################################

#----------------------   Hypercube's Volume/Surface   ----------------------#
def hypercube(side, dim, volume=True):
    """ Hypercube's volume and surface

    Compute and return the volume or surface of the n-cube of side S.
    A hypercube (or n-cube) is an n-dimensional analogue of a square
    (n = 2) and a cube (n = 3).

    The volume of an n-cube of side S is:
        volume = S^n

    The surface is:
        surface = Fn * S^2
    where Fn = n.(n-1).2^(n-3) is the number of 2-faces.

    Parameters
    ----------
    side : float
        Side length of the cube.
    dim : int
        Dimension of the hypercube.
    [OPT] volume : bool
        Either compute the volume (True) or the surface (False).
            :Default: True (volume)

    Returns
    -------
    cube : float
        Either the volume or the surface of the hypercube,
        depending on the state of the boolean `volume`.

    Examples
    --------
    # Volume of a 3D cube of side 5
    >>> hypercube(5.0, 3)
    125.0

    # Surface of a 3D cube of side 5
    >>> hypercube(5.0, 3, False)
    150.0
    """

    # Dimension validity
    if dim <= 0:
        raise ValueError(f"Wrong value `{dim}` for `dim`, cube dim. must be > 0")

    # Hypercube's volume
    if volume:
        return side ** dim

    # Hypercube's surface
    return dim * (dim-1) * 2.**(dim-3) * side**2.
#----------------------------------------------------------------------------#

#---------------------   Hypersphere's Volume/Surface   ---------------------#
def hypersphere(radius, dim, volume=True):
    """ Hypersphere's volume and surface

    Compute and return the volume or surface of the n-sphere of radius R.
    A hypersphere (or n-sphere) is an n-dimensional analogue of a circle
    (n = 2) and a sphere (n = 3).

    The volume of a d-sphere of radius R is:
        volume = (R^n * (pi)^(n/2)) / Gamma(n/2 + 1)
    And its surface is:
        surface = (2 * R^(n-1) * (pi)^(n/2)) / Gamma(n/2)
    where Gamma(x) is the gamma function:
        for z in C, Re{z} > 0,
        Gamma(z) = int_{0}^{+infty} t^(z-1) exp(-t) dt

    Note: be sure the fed argument is a radius, not the diameter
          (radius = diameter / 2).

    Parameters
    ----------
    radius : float
        Radius of the hypersphere.
    dim : int
        Dimension of the hypersphere.
    [OPT] volume : bool
        Either compute the volume (True) or the surface (False).
            :Default: True (volume)

    Returns
    -------
    sphere : float
        Either the volume or the surface of the hypersphere,
        depending on the state of the boolean `volume`.

    Examples
    --------
    # Volume of a 3D sphere of radius 5
    >>> hypersphere(5.0, 3)
    523.5987755982986

    # Surface of a 3D sphere of radius 5
    >>> hypersphere(5.0, 3, False)
    314.1592653589793
    """

    # Dimension validity
    if dim <= 0:
        raise ValueError(f"Wrong value `{dim}` for `dim`, sphere dim. must be > 0")

    # Hypersphere's volume
    if volume:
        return (radius * np.sqrt(np.pi))**dim / math.gamma(dim/2 + 1.)

    # Hypersphere's surface
    return 2. * np.pi**(dim/2) * (radius)**(dim-1) / math.gamma(dim/2)
#----------------------------------------------------------------------------#

#----------------------   Get Hyper-Volume Function   -----------------------#
def get_vol_func(volume='hypersphere'):
    """ Get the reference to a specified hyper-volume function

    Take a hyper-volume's name and return its corresponding function;
    return the `hypersphere` as default. Every function has signature:
        fvol(size, dim, volume=True)
    where `size` is the hyper-volume's attribute (side, radius, etc.),
    `dim` is its dimension and `volume` allows to return the surface
    (False) or volume (True).

    Parameters
    ----------
    [OPT] volume : str
        The hyper-volume name among: {'hypercube', 'hypersphere'}.
            :Default: 'hypersphere'

    Returns
    -------
    fvol : reference to a function
        The hyper-volume function.

    Examples
    --------
    >>> import numpy as np

    # Get the hyper-volume function
    >>> fvol = get_vol_func('hypersphere')

    # Compute the distances
    >>> fvol(2.5, 3, False)     # Surface of a 3D sphere
    78.53981633974483
    >>> fvol(2.5, 3, True)      # Volume of a 3D sphere
    65.44984694978733
    """

    # Get the hyper-volume function
    vol = volume.lower()
    if vol == 'hypercube':
        return hypercube
    if vol == 'hypersphere':
        return hypersphere

    # Raise an error if the span estimation function is invalid
    raise ValueError(f"Wrong value `{volume}` for `volume`; options are:\n"
        + "\t{'hypercube', 'hypersphere'}")
#----------------------------------------------------------------------------#

##############################################################################
