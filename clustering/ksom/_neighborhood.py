""" Find the neighbors of a given item within a grid

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: May 2026
Last revised: May 2026

License: GPLv3
"""

#__all__ = ['_neighborhood_1st', '_neighborhood_2nd', '_neighbors']


##############################################################################
##                           Find BMU's Neighbors                           ##
##############################################################################

#-----------------------   First Order Neighborhood   -----------------------#
def _neighborhood_1st(bmu, gsize):
    """ First order neighborhood

    Return a list of index of the BMU's neighboring nodes. A node is a
    1st order neighbor if it is in direct contact with the BMU. Let's
    consider a 4x4 grid:

            0   1   2   3
            4   5   6   7
            8   9   10  11
            12  13  14  15

    If the BMU is the node 5, its neighbors are {0-2, 4, 6, 8-10}. A
    problem can appear in case of the bordering nodes: the BMU 0 has
    2 neighbors, 1 and 3, thus a special attention is required here.
    More generally, all the border nodes (corners, first and last row
    and column) must be handled specifically.

    Parameters
    ----------
    bmu : int
        The Best Matching Unit whose neighboring nodes are seek.
    gsize : 3-tuple of ints
        The grid's dimensions, organized as (height, width, neurons).

    Returns
    -------
    neighbors : list of ints
        The list of the BMU's neighbors indexes
        (ie their respective position inside the neural grid).
    """

    # Retrieve the grid's width and number of neurons
    width, size = gsize[1], gsize[2]

    # Determine the number of the BMU's row
    row = (bmu // width) * width

    # Bound the neighbors to dismiss those out-of-range
    bounds = [[max(row-width, 0), row],             # Previous row
              [row, row+width],                     # BMU's row
              [row+width, min(row+2*width, size)]]  # Next row

    # Find the relative position of the BMU in the other rows
    bmus = [bmu-width, bmu, bmu+width]

    # Set the offset for each row
    offset = [[-1, 0, 1], [-1, 1], [-1, 0, 1]]

    # Get the neighbors
    return [b+i for b, off, bnds in zip(bmus, offset, bounds)
            for i in off if bnds[0] <= b+i < bnds[1]]
#----------------------------------------------------------------------------#

#----------------------   Second Order Neighborhood   -----------------------#
def _neighborhood_2nd(bmu, gsize):
    """ Second order neighborhood

    Return a list of index of the BMU's neighboring nodes. A node is a
    2nd order neighbor if it is in direct contact with a 1st order BMU's
    neighbor. Let's consider a 5x5 grid:

            0   1   2   3   4
            5   6   7   8   9
            10  11  12  13  14
            15  16  17  18  19
            20  21  22  23  24

    The 2nd order neighbors of the BMU 12 are {0-4, 5, 9, 10, 14, 15, 19,
    20-24}. The possible problems appear in the two outer (most external)
    lines & cols as some neighbors can be missing. A special attention is
    so required for grids smaller than 4x4.

    Parameters
    ----------
    bmu : int
        The Best Matching Unit whose neighboring nodes are seek.
    gsize : 3-tuple of ints
        The grid's dimensions, organized as (height, width, neurons).

    Returns
    -------
    neighbors : list of ints
        The list of the BMU's neighbors indexes
        (ie their respective position inside the neural grid).
    """

    # Retrieve the grid's width and number of neurons
    width, size = gsize[1], gsize[2]

    # Determine the number of the BMU's row
    row = (bmu // width) * width

    # Bound the neighbors to dismiss those out-of-range
    bounds = [[max(row-2*width, 0), max(row-width, 0)],         # Previous-1 row
              [max(row-width, 0), max(row, 0)],                 # Previous row
              [row, row+width],                                 # BMU's row
              [min(row+width, size), min(row+2*width, size)],   # Next row
              [min(row+2*width, size), min(row+3*width, size)]] # Next+1 row

    # Find the relative position of the BMU in the other rows
    bmus = [bmu-2*width, bmu-width, bmu, bmu+width, bmu+2*width]

    # Set the offset for each row
    offset = [[-2, -1, 0, 1, 2],
              [-2, 2], [-2, 2], [-2, 2],
              [-2, -1, 0, 1, 2]]

    # Get the neighbors
    return [b+i for b, off, bnds in zip(bmus, offset, bounds)
            for i in off if bnds[0] <= b+i < bnds[1]]
#----------------------------------------------------------------------------#

#------------------------   Any Order Neighborhood   ------------------------#
def _neighborhood(bmu, gsize, order=1):
    """ Any order neighborhood

    Search and return the neighborhood of any order. If `order` is set
    to 1, this function is equivalent to `_neighborhood_1st`; if it is
    set to 2, it is equivalent to `_neighborhood_2nd`.

    Parameters
    ----------
    bmu : int
        The Best Matching Unit whose neighboring nodes are seek.
    gsize : 3-tuple of ints
        The grid's dimensions, organized as (height, width, neurons).
    [OPT] order : int
        The order of the neighborhood.
            :Default: 1

    Returns
    -------
    neighbors : list of ints
        The list of the BMU's neighbors indexes
        (ie their respective position inside the neural grid).
    """

    # Retrieve the grid's width and number of neurons
    width, size = gsize[1], gsize[2]

    # Determine the number of the BMU's row
    row = (bmu // width) * width

    # Bound the neighbors to dismiss those out-of-range
    bounds = []
    # Previous rows
    for p in range(order, 0, -1):
        bounds += [[max(row-p*width, 0), max(row-(p-1)*width, 0)]]
    # BMU's row
    bounds += [[row, row+width]]
    # Next rows
    for p in range(1, order+1):
        bounds += [[min(row+p*width, size), min(row+(p+1)*width, size)]]

    # Find the relative position of the BMU in the other rows
    bmus = [bmu-p*width for p in range(order, 0, -1)] # Previous rows
    bmus.append(bmu)                                  # BMU's rows
    bmus += [bmu+p*width for p in range(1, order+1)]  # Next rows

    # Set the offset for each row
    offset = [range(-order, order+1)]       # First most distant row (full range)
    for _ in range(2*order-1):              # Intermediate rows (only borders)
        offset += [[-order, order]]
    offset.append(range(-order, order+1))   # Last most distant row (full range)

    # Get the neighbors
    return [b+i for b, off, bnds in zip(bmus, offset, bounds)
            for i in off if bnds[0] <= b+i < bnds[1]]
#----------------------------------------------------------------------------#

##############################################################################
