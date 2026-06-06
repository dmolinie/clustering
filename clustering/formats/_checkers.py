""" Functions to check the format of some attributes

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: February 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['check_data', 'get_index', 'get_tags', 'get_classes']

import numpy as np

_TYPE = "should be NumPy array, Database or Cluster"


##############################################################################
##                           Miscellaneous tools                            ##
##############################################################################

#--------------------------   Check Data Format   ---------------------------#
def check_data(data):
    """ Check data format

    Check if the provided data have a supported type; additionally, if
    the data are a 1D row vector, reshape them as a 2D column vector.

    Parameters
    ----------
    data : np.ndarray or Database
        Data to be checked and formatted.

    Returns
    -------
    data : same type as `data` argument
        The correctly formatted data.

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy database
    >>> dba = Database(np.arange(20).reshape(10, 2), np.arange(10))

    # Check the data format
    >>> data = check_data(np.arange(5)).shape
    >>> print(np.shape(data))
    (2,)
    >>> data = check_data(dba)
    >>> print(np.shape(data))
    (10, 2)
    """

    # Check if NumPy array, Database or Cluster
    try:
        data.shape
    except AttributeError as exc:
        raise TypeError(
            f"Wrong type `{type(data)}` for `data`: " + _TYPE) from exc

#    # Convert 1D row vector to 2D column vector
#    if isinstance(data, np.ndarray) and len(data) == data.size:
#        data = data.reshape(-1, 1)

    return data
#----------------------------------------------------------------------------#

#---------------------   Get the Index of a Database   ----------------------#
def get_index(database):
    """ Get the index of a database

    Parameters
    ----------
    database : np.ndarray or Database
        The database for which one wants to get the indexes.

    Returns
    -------
    index : list or None
        The list of indexes if found; None otherwise.

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy database
    >>> dba = Database(np.arange(20).reshape(10, 2), np.arange(10))

    # Retrieve some indexes
    >>> print(get_index(np.arange(5)))
    [0 1 2 3 4]
    >>> print(get_index(dba))
    [0 1 2 3 4 5 6 7 8 9]
    """

    # Numpy array --> Build a list of indexes
    if isinstance(database, np.ndarray):
        return np.arange(len(database))

    # Database or Cluster --> Get the index attribute
    try:
        index = database.index
    except AttributeError as exc:
        raise TypeError(
            f"Wrong type `{type(database)}` for `database`: " + _TYPE) from exc

    return index
#----------------------------------------------------------------------------#

#----------------------   Get the Tags of a Database   ----------------------#
def get_tags(database, idx=None):
    """ Get the tags of a database

    Parameters
    ----------
    database : np.ndarray or Database
        The database for which one wants to get the tags.
    [OPT] idx : int, list, tuple of np.ndarray of ints
        The indexes of the tags entries to return; if None, return the
        whole `tags` attribute.
            :Default: None

    Returns
    -------
    tags : None, str, or list
        The list of tags if found; None otherwise.

    Examples
    --------
    >>> import numpy as np

    >>> dba = Database(
    ...     np.arange(20).reshape(10, 2), np.arange(10),
    ...     tags=['C1', 'C2'])

    # Retrieve some tags
    >>> print(get_tags(np.arange(5)))
    None
    >>> print(get_tags(dba, idx=None))
    ['C1', 'C2']
    >>> print(get_tags(dba, idx=1))
    C2
    >>> print(get_tags(dba, idx=[0, 1]))
    ['C1', 'C2']
    """

    # Numpy array --> No tags
    if database is None or isinstance(database, np.ndarray):
        return None

    # Database or Cluster --> Get the index attribute
    try:
        tags = database.tags
    except AttributeError as exc:
        raise TypeError(
            f"Wrong type `{type(database)}` for `database`: " + _TYPE) from exc

    if len(tags) == 0 or idx is None:
        return tags
    return np.asarray(tags)[idx].tolist()
#----------------------------------------------------------------------------#

#--------------------   Get the Classes of a Database   ---------------------#
def get_classes(database, idx=None):
    """ Get the classes of a database

    Parameters
    ----------
    database : np.ndarray or Database
        The database for which one wants to get the classes, if any.
    [OPT] idx : int, list, tuple of np.ndarray of ints
        The indexes of the classes entries to return; if None, return
        the whole `classes` attribute.

    Returns
    -------
    classes : None, str, or list
        The list of classes if found; None otherwise.

    Examples
    --------
    >>> import numpy as np

    >>> dba = Database(
    ...     np.arange(20).reshape(10, 2), np.arange(10),
    ...     classes=[i for i in range(10)])

    # Retrieve some classes
    >>> print(get_classes(np.arange(5)))
    None
    >>> print(get_classes(dba, idx=None))
    [0 1 2 3 4 5 6 7 8 9]
    >>> print(get_classes(dba, idx=1))
    1
    >>> print(get_classes(dba, idx=[0, 1]))
    [0, 1]
    """

    # Numpy array --> No classes
    if database is None or isinstance(database, np.ndarray):
        return None

    # Database or Cluster --> Get the index attribute
    try:
        classes = database.classes
    except AttributeError as exc:
        raise TypeError(
            f"Wrong type `{type(database)}` for `database`: " + _TYPE) from exc

    if len(classes) == 0 or idx is None:
        return classes
    return classes[idx]
#----------------------------------------------------------------------------#

##############################################################################
