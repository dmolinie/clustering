""" Didactic datasets for testing

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: June 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['nested_circles', 'strips', 'gaussian', 'gauss_3d']

import numpy as np

import clustering.formats as dformat

RCPARAMS = {'font.size': 18}
FIGSIZE = (19.20, 10.80)
FIGEXT = 'pdf'
FPARAMS = {'rasterized': True}
FIGSAVE = {'dpi': 150, 'bbox_inches': 'tight'}


##############################################################################
##                            Academic datasets                             ##
##############################################################################

#--------------------------   Two Nested Circles   --------------------------#
def nested_circles(pts=500, gap=3, mean=0., ratio=1., show=False, **kwrng):
    """ Generate two 2D nested circles

    The two 2D circles are centered around (0, 0). The outermost circle
    has a radius `gap` times bigger than that of the inner circle.

    Parameters
    ----------
    [OPT] pts : int
        Number of points for the inner circle, and `gap*pts` for the
        outer circle.
            :Default: 500
    [OPT] gap : int
        Ratio of the outermost circle's radius over the innermost's radius.
            :Default: 3
    [OPT] show : bool
        Plot and show or not the figure.
            :Default: False
    [OPT] mean : float
        Mean of the 2D circle.
            :Default: 0.
    [OPT] ratio : float
        Multiplier of the `gap` argument.
            :Default: 1.

    Other Parameters
    ----------------
    **kwrng : inline keyword arguments, optional
        The parameters for the NumPy's `random.default_rng` function. In
        particular, `seed` may be used to generate reproducible datasets.

    Returns
    -------
    data : Database object
        The dataset formed by the two 2D circles. The data are the
        2-coordinate points and the index are the circle's number
        ('Inner' for the inner circle and 'Outer' for the outer circle).

    Examples
    --------
    >>> data = nested_circles(500, 3, 0., 1., show=False)

    >>> print(type(data))
    <class 'clustering.formats._database.Database'>

    >>> print(data.shape)
    (2000, 2)
    """
    # pylint: disable=too-many-locals

    rng = np.random.default_rng(**kwrng)

    noise = [-.25, 0.25]
    diff = noise[1] - noise[0]

    grps = ['Inner' for i in range(pts)] + ['Outer' for i in range(gap*pts)]
    data = np.empty(((gap+1)*pts, 2), float)

    # Inner circle
    tsp = np.linspace(0, 2.*np.pi, pts)
    cos = np.cos(tsp) + diff * rng.random(pts) + noise[0] + mean
    sin = np.sin(tsp) + diff * rng.random(pts) + noise[0] + mean
    data[:pts] = ratio*np.column_stack((cos, sin))

    # Outer circle
    points = int(gap * pts)
    gap = float(gap)
    tsp = np.linspace(0, 2.*np.pi, points)
    cos = gap * np.cos(tsp) + diff * rng.random(points) + noise[0] + mean
    sin = gap * np.sin(tsp) + diff * rng.random(points) + noise[0] + mean
    data[pts:] = ratio*np.column_stack((cos, sin))

    # Show the graphics
    if show:

        # pylint: disable-next=import-outside-toplevel
        import matplotlib.pyplot as plt
        plt.rcParams.update(RCPARAMS)

        # Plot the dataset
        plt.figure(figsize=FIGSIZE)
        plt.plot(data[:pts, 0], data[:pts, 1], '.', **FPARAMS)
        plt.plot(data[pts:, 0], data[pts:, 1], '.', **FPARAMS)

        # Plot and save the figure
        plt.tight_layout()
        plt.savefig('Circles.'+FIGEXT, **FIGSAVE)
        plt.show()

    return dformat.Database(data, np.arange(len(data)), ('x', 'y'), grps)
#----------------------------------------------------------------------------#

#------------------------   Three Vertical Strips   -------------------------#
def strips(pts=1500, sigma=0.175, show=False, **kwrng):
    """ Generate three 2D vertical strips

    The strips follow a uniform distribution along the vertical axis,
    and a Gaussian distribution along the horizontal axis. The strips
    are side by side, but can overlap, along the horizontal axis.

    Parameters
    ----------
    [OPT] pts : int
        Number of points for each strip.
            :Default: 1500
    [OPT] sigma : float
        Standard deviation of the Gaussian distributions.
            :Default: 0.175
    [OPT] show : bool
        Plot and show or not the figure.
            :Default: False

    Other Parameters
    ----------------
    **kwrng : inline keyword arguments, optional
        The parameters for the NumPy's `random.default_rng` function. In
        particular, `seed` may be used to generate reproducible datasets.

    Returns
    -------
    data : Database object
        The dataset formed by the three 2D strips. The data are the
        2-coordinate points, and the index are the color of each strip
        ('Blue', 'White' and 'Red').

    Examples
    --------
    >>> data = strips(1500, 0.175, show=False)

    >>> print(type(data))
    <class 'clustering.formats._database.Database'>

    >>> print(data.shape)
    (4500, 2)
    """

    rng = np.random.default_rng(**kwrng)

    data = np.empty((3*pts, 2), float)

    # Left strip (Blue)
    data[:pts, 0] = rng.normal(0.5, sigma, pts)
    data[:pts, 1] = rng.uniform(0, 2, pts)
    grps = ['Blue' for i in range(pts)]

    # Middle strip (White)
    data[pts:2*pts, 0] = rng.normal(1.5, sigma, pts)
    data[pts:2*pts, 1] = rng.uniform(0, 2, pts)
    grps += ['White' for i in range(pts)]

    # Right strip (Red)
    data[2*pts:, 0] = rng.normal(2.5, sigma, pts)
    data[2*pts:, 1] = rng.uniform(0, 2, pts)
    grps += ['Red' for i in range(pts)]

    # Show the graphics
    if show:

        # pylint: disable-next=import-outside-toplevel
        import matplotlib.pyplot as plt
        plt.rcParams.update(RCPARAMS)

        # Plot the data
        plt.figure(figsize=FIGSIZE)
        plt.plot(data[:pts, 0], data[:pts, 1], '.', color='b', **FPARAMS)
        plt.plot(
            data[pts:2*pts, 0], data[pts:2*pts, 1], '*', color='w', **FPARAMS)
        plt.plot(data[2*pts:, 0], data[2*pts:, 1], '+', color='r', **FPARAMS)

        # Plot settings
        plt.xlim(0, 3)
        plt.ylim(0, 2)
        axis = plt.gca()
        grey = 221/255
        axis.set_facecolor((grey, grey, grey))

        # Plot and save the figure
        plt.tight_layout()
        plt.savefig('Strips.'+FIGEXT, **FIGSAVE)
        plt.show()

    return dformat.Database(data, np.arange(len(data)), ('x', 'y'), grps)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                            Gaussian datasets                             ##
##############################################################################

#------------------   N-dimension Gaussian Distribution   -------------------#
def _ndgauss(pts=250, dim=5, limits=(0, 100), sigma=0.5, rng=None):
    """ Generate a N-dimensional Gaussian distribution

    Parameters
    ----------
    [OPT] pts : int
        Number of points of the distribution.
            :Default: 250
    [OPT] dim : int
        Dimension of the Gaussian distribution : nb of features).
            :Default: 5
    [OPT] limits : 2-tuple
        Range (min, max) in which the values are randomly drawn.
            :Default: (0, 100)
    [OPT] sigma : float
        Standard Deviation of the Gaussian distribution.
            :Default: 0.5

    Returns
    -------
    data : 2D np.ndarray
        The matrix of the Gaussian distrib. (`pts` rows and `dim` cols).
    """

    if rng is None:
        rng = np.random.default_rng()

    # Random mean for each feature within the limits range
    mean = rng.uniform(limits[0], limits[1], dim)

    # Generate the gaussian distribution
    return np.array([rng.normal(mean, sigma) for i in range(pts)])
#----------------------------------------------------------------------------#

#---------------------------   Gaussian Dataset   ---------------------------#
def gaussian(pts=250, dim=5, limits=(0, 100), sigma=0.5, models=12, **kwrng):
    """ Generate a dataset with a Gaussian distribution

    Call `models` times the function `ndgauss` and store these distri-
    butions within a Database class object.

    Parameters
    ----------
    [OPT] pts : int
        Number of points of each distribution.
            :Default: 250
    [OPT] dim : int
        Dimension of the Gaussian distributions (nb of features).
            :Default: 5
    [OPT] limits : 2-tuple
        Range (min, max) in which the values are randomly drawn.
            :Default: (0, 100)
    [OPT] sigma : float
        Standard Deviation of the Gaussian distributions.
            :Default: 0.5
    [OPT] models : int
        Number of distinct Gaussian distributions to generate.
            :Default: 12

    Other Parameters
    ----------------
    **kwrng : inline keyword arguments, optional
        The parameters for the NumPy's `random.default_rng` function. In
        particular, `seed` may be used to generate reproducible datasets.

    Returns
    -------
    data : Database object
        The Gaussian distributions gathered in a Database class object.

    Examples
    --------
    >>> data = gaussian(250, 5, (0, 100), 0.5, 12)

    >>> print(type(data))
    <class 'clustering.formats._database.Database'>

    >>> print(data.shape)
    (3000, 5)
    """

    rng = np.random.default_rng(**kwrng)

    # Generate gaussian distributions (values)
    values = np.zeros((pts*models, dim), float)
    for i in range(0, pts*models, pts):
        values[i:i+pts] = _ndgauss(pts, dim, limits, sigma, rng)

    # Build the dataset
    return dformat.Database(
        values, np.arange(len(values)),                     # Values & Indexes
        [str(i) for i in range(dim)],                       # Tags
        [str(i) for i in range(models) for j in range(pts)])    # Classes
#----------------------------------------------------------------------------#

#----------------------   3D Gaussian Distributions   -----------------------#
def gauss_3d(pts=250, models=12, limits=(0, 100), sigma=0.5, show=False, **kwrng):
    """ Generate a set of 3D Gaussian distributions

    Parameters
    ----------
    [OPT] pts : int
        Number of points of each distribution.
            :Default: 250
    [OPT] models : int
        Number of distinct Gaussian distributions to generate.
            :Default: 12
    [OPT] limits : 2-tuple
        Range (min, max) in which the values are randomly drawn.
            :Default: 5
    [OPT] sigma : float
        Standard Deviation of the Gaussian distributions.
            :Default: (0, 100)
    [OPT] show : bool
        Plot and show or not the figure.
            :Default: False

    Other Parameters
    ----------------
    **kwrng : inline keyword arguments, optional
        The parameters for the NumPy's `random.default_rng` function. In
        particular, `seed` may be used to generate reproducible datasets.

    Returns
    -------
    data : Database object
        The Gaussian distributions gathered in a Database class object.

    Examples
    --------
    >>> data = gauss_3d(250, 12, (0, 100), 0.5, show=False)

    >>> print(type(data))
    <class 'clustering.formats._database.Database'>

    >>> print(data.shape)
    (3000, 3)
    """

    # Generate the distributions
    dataset = gaussian(pts, 3, limits, sigma, models, **kwrng)

    # Show the graphics
    if show:

        # pylint: disable-next=import-outside-toplevel
        import matplotlib.pyplot as plt
        plt.rcParams.update(RCPARAMS)

        # Declare 3D figure
        fig = plt.figure(figsize=FIGSIZE)
        axis = fig.add_subplot(111, projection='3d')

        # Plot the 'models' distributions on the above figure
        for i in range(0, models*pts, pts):
            axis.plot(dataset[i:i+pts, 0], dataset[i:i+pts, 1],
                      dataset[i:i+pts, 2], '+', markersize=24, **FPARAMS)

        # Plot and save the figure
        plt.tight_layout()
        plt.savefig('Gauss3d.'+FIGEXT, **FIGSAVE)
        plt.show()

    return dataset
#----------------------------------------------------------------------------#

##############################################################################
