""" Database & Cluster container classes

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: February 2021
Last revised: May 2026

License: GPLv3
"""
# pylint: disable=C0302, R0913, R0917

__all__ = ['Database', 'Cluster']

from numbers import Number

import numpy as np

import clustering.metrics as mts
from clustering.formats._checkers import get_tags, get_classes

_EMPTY = "Empty Database, "
_TYPES = "must be list, tuple or array"


##############################################################################
##                          Class Data (protected)                          ##
##############################################################################

class _Data():
    """ Data class overloading numpy arrays

    Serve as mother class for `Database` and `Cluster` classes. Represent
    the data of a database or of a cluster as a numpy array and provide
    some useful tools to it (mean, std, limits, normalization, etc.).

    Since this class greatly relies on a numpy array (it is somehow an
    overloading of it), most of the tools available for that class are
    also available with this class (and for inherited classes as well).
    This is to ensure the maximum compatibility: both `_Data` and numpy
    `Array` should be exchangeable with each other.

    Constructor
    -----------
    __init__(vals, lims, dtype=float)

    Magic Methods
    -------------
    __getitem__(pos)
        Get the data at pos --> value = dba[pos].
    __setitem__(pos, value)
        Set the data at pos --> dba[pos] = value.
    __iter__()
        Iterate on the data --> for val in dba.
    __len__()
        Length of the data --> len(dba).
    __repr__()
        Display the data --> repr(dba).
    __str__()
        Print the data --> print(dba).
    + The standard scalar, bitwise and comparison operators.

    Attributes
    ----------
    value : ND np.ndarray, getter & setter
        The data values.
    shape : tuple of ints, getter only
        The shape of the data.
    size : int, getter only
        The size of the data (the number of items).
    ndim : int, getter only
        The imension of the data.
    dtype : datatype, getter only
        The type of the data.
    limits : 2-tuple of floats, getter & setter
        The data limits.

    Methods
    -------
    min(axis=None)
        Compute the data's minimal value along `axis`.
    max(axis=None)
        Compute the data's maximal value along `axis`.
    extrema(axis=0)
        Determine the data's min and max values along `axis`.
    mean(axis=None)
        Compute the mean of the data.
    std(axis=None)
        Compute the standard deviation of the data.
    mean_std(axis=None)
        Compute the mean and standard deviation of the data.
    density(span='sphere_span', volume='hypersphere', distance='euclidean')
        Compute the dataset's density.
    avstd(minmax=False)
        Compute the Average Standard Deviation.
    normalize(bounds=(0., 1.), axis=0)
        Normalize the data.
    denormalize(axis=0)
        Denormalize the data.

    Examples
    --------
    >>> import numpy as np

    # Generate a dummy database
    >>> dba = _Data(np.arange(20).reshape(10, 2), np.arange(10))

    # Compute some statistics
    >>> print(dba.mean())
    9.5
    >>> print(dba.std())
    5.766281297335398
    >>> print(dba.mean_std())
    (9.5, 5.766281297335398)

    # Compute some metrics
    >>> print(dba.density())
    0.019648758406406834
    >>> print(dba.avstd())
    5.744562646538029

    # Normalize & denormalize the database
    >>> print(dba.extrema())
    (0.0, 19.0)
    >>> dba.normalize()
    >>> print(dba.extrema())
    (-2.7755575615628914e-17, 1.0)
    >>> dba.denormalize()
    >>> print(dba.extrema())
    (0.0, 19.0)
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, vals, lims, dtype=float):
        """ Wrap the data into a _Data object

        Parameters
        ----------
        vals : ND tuple or list, np.ndarray
            The data values.
        lims : 2-tuple of floats
            The data min & max values.
        [OPT] dtype : data type
            The type of the data for the `value` attribute.
                :Default: float

        Examples
        --------
        >>> import numpy as np

        # Generate dummy data
        >>> array = np.arange(100).reshape(-1, 5)

        # Wrap the data into a `_Data` object
        >>> data = _Data(array, (array.min(0), array.max(0)))
        """
        if not isinstance(vals, (list, tuple, np.ndarray)):
            raise TypeError(f"Wrong type `{type(vals)}` for `vals`: " + _TYPES)
        self._value = np.array(vals, dtype)     # Data values
        self._limits = lims                     # Data limits
    #------------------------------------------------------------------------#

    #--------------------------   Magic Methods   ---------------------------#
    def __str__(self):
        """ Print the data (`value` attribute) """
        return repr(self._value)

    def __repr__(self):
        """ Display the data (`value` attribute) """
        return repr(self._value)

    def __len__(self):
        """ Length of the data (`value` attribute) """
        return len(self._value)

    def __getitem__(self, pos):
        """ Get the data (`value` attribute) at `pos` """
        return self._value[pos]

    def __setitem__(self, pos, value):
        """ Set the data (`value` attribute) at `pos` """
        self._value[pos] = value

    def __iter__(self):
        """ Iterate on the data (`value` attribute) """
        return iter(self._value)
    #------------------------------------------------------------------------#

    #-------------------   Standard Operator Overloads   --------------------#
    # Scalar operators
    def __add__(self, vals):
        r""" Addition operator `+` overload """
        return self._cp_update(self._value.__add__(vals))

    def __radd__(self, vals):
        r""" Addition operator `+` overload """
        return self._cp_update(self._value.__radd__(vals))

    def __sub__(self, vals):
        r""" Subtraction operator `-` overload """
        return self._cp_update(self._value.__sub__(vals))

    def __rsub__(self, vals):
        r""" Subtraction operator `-` overload """
        return self._cp_update(self._value.__rsub__(vals))

    def __mul__(self, vals):
        r""" Multiplication operator `*` overload """
        return self._cp_update(self._value.__mul__(vals))

    def __rmul__(self, vals):
        r""" Multiplication operator `*` overload """
        return self._cp_update(self._value.__rmul__(vals))

    def __pow__(self, p):
        r""" Exponentiation operator `**` overload """
        return self._cp_update(self._value.__pow__(p))

    def __rpow__(self, p):
        r""" Exponentiation operator `**` overload """
        return self._cp_update(self._value.__rpow__(p))

    def __truediv__(self, vals):
        r""" True division operator `/` overload """
        return self._cp_update(self._value.__truediv__(vals))

    def __rtruediv__(self, vals):
        r""" True division operator `/` overload """
        return self._cp_update(self._value.__rtruediv__(vals))

    def __floordiv__(self, vals):
        r""" Floor division operator `//` overload """
        return self._cp_update(self._value.__floordiv__(vals))

    def __rfloordiv__(self, vals):
        r""" Floor division operator `//` overload """
        return self._cp_update(self._value.__rfloordiv__(vals))

    def __mod__(self, vals):
        r""" Remainder (modulo) operator `%` overload """
        return self._cp_update(self._value.__mod__(vals))

    def __rmod__(self, vals):
        r""" Remainder (modulo) operator `%` overload """
        return self._cp_update(self._value.__rmod__(vals))

#    def __divmod__(self, vals):
#        # Error in copy
#        r""" Floor division & Remainder operator (`//`, `%`) overload """
#        return self._cp_update(self._value.__divmod__(vals))
#        return self._value.__divmod__(vals)

#    def __rdivmod__(self, vals):
#        # Error in copy
#        r""" Floor division & Remainder operator  (`//`, `%`) overload """
#        return self._value.__rdivmod__(vals)

    def __matmul__(self, vals):
        r""" Matrix Multiplication operator `@` overload """
        return self._cp_update(self._value.__matmul__(vals))

    def __rmatmul__(self, vals):
        r""" Matrix Multiplication operator `@` overload """
        return self._cp_update(self._value.__rmatmul__(vals))

    def __neg__(self):
        r""" Negation operator `-<obj>` overload """
        return self._cp_update(self._value.__neg__())

    def __pos__(self):
        r""" Positive operator `+<obj>` overload """
        return self._cp_update(self._value.__pos__())

    def __abs__(self):
        r""" Absolute operator `abs` overload """
        return self._cp_update(self._value.__abs__())

    # Bitwise operators
    def __lshift__(self, vals):
        r""" Bitwise Left Shift operator `<<` overload """
        return self._cp_update(self._value.__lshift__(vals))

    def __rlshift__(self, vals):
        r""" Bitwise Left Shift operator `<<` overload """
        return self._cp_update(self._value.__rlshift__(vals))

    def __rshift__(self, vals):
        r""" Bitwise Right Shift operator `>>` overload """
        return self._cp_update(self._value.__rshift__(vals))

    def __rrshift__(self, vals):
        r""" Bitwise Right Shift operator `>>` overload """
        return self._cp_update(self._value.__rrshift__(vals))

    def __and__(self, vals):
        r""" Bitwise AND operator `&` overload """
        return self._cp_update(self._value.__and__(vals))

    def __rand__(self, vals):
        r""" Bitwise AND operator `&` overload """
        return self._cp_update(self._value.__rand__(vals))

    def __or__(self, vals):
        r""" Bitwise OR operator `|` overload """
        return self._cp_update(self._value.__or__(vals))

    def __ror__(self, vals):
        r""" Bitwise OR operator `|` overload """
        return self._cp_update(self._value.__ror__(vals))

    def __xor__(self, vals):
        r""" Bitwise XOR operator `^` overload """
        return self._cp_update(self._value.__xor__(vals))

    def __rxor__(self, vals):
        r""" Bitwise XOR operator `^` overload """
        return self._cp_update(self._value.__rxor__(vals))

    def __invert__(self):
        r""" Bitwise NOT operator `~` overload """
        return self._cp_update(self._value.__invert__())

#    def __not__(self):
#        r""" Bitwise NOT operator `~` overload """
#        return self._cp_update(self._value.__not__())

    # Comparison operators
    def __lt__(self, vals):
        r""" Less than operator `<` overload """
        return self._cp_update(self._value.__lt__(vals))

    def __le__(self, vals):
        r""" Less than or equal to operator `<=` overload """
        return self._cp_update(self._value.__le__(vals))

    def __eq__(self, vals):
        r""" Equal to operator `==` overload """
        return self._cp_update(self._value.__eq__(vals))

    def __ne__(self, vals):
        r""" Not equal to operator `!=` overload """
        return self._cp_update(self._value.__ne__(vals))

    def __gt__(self, vals):
        r""" Greater than operator `>` overload """
        return self._cp_update(self._value.__gt__(vals))

    def __ge__(self, vals):
        r""" Greater than or equal to operator `>=` overload """
        return self._cp_update(self._value.__ge__(vals))
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def shape(self):
        """ Shape of the data (`value` attribute) """
        return self._value.shape

    @property
    def size(self):
        """ Size of the data (`value` attribute) """
        return self._value.size

    @property
    def ndim(self):
        """ Dimension of the data (`value` attribute) """
        return self._value.ndim

    @property
    def dtype(self):
        """ Type of the data (`value` attribute) """
        return self._value.dtype

    @property
    def value(self):
        """ Get the `value` attribute """
        return self._value

    @value.setter
    def value(self, vals):
        """ Set the `value` attribute """
        self._value = vals

    @property
    def limits(self):
        """ Get the `limits` attribute """
        return self._limits

    @limits.setter
    def limits(self, limits):
        """ Set the `limits` attribute """
        self._limits = limits
    #------------------------------------------------------------------------#

    #--------------------------   Copy & Select   ---------------------------#
    def _cp_update(self, values):
        """ Update the `value` attribute (for operator overloads) """
        return Database(values, self._limits.copy())

    def copy(self):
        """ Copy the current _Data object """
        return _Data(self._value.copy(), self._limits.copy())
    #------------------------------------------------------------------------#

    #--------------   Extremal Values and (De-)Normalization   --------------#
    def min(self, axis=None, **kwargs):
        """ Compute the data's minimal value along `axis` """
        if self.size == 0:
            raise AssertionError(_EMPTY + "no minimal value")
        return np.min(self._value[:], axis, **kwargs)

    def max(self, axis=None, **kwargs):
        """ Compute the data's maximal value along `axis` """
        if self.size == 0:
            raise AssertionError(_EMPTY + "no maximal value")
        return np.max(self._value, axis, **kwargs)

    def extrema(self, axis=None, **kwargs):
        """ Compute the data's min and max values along `axis` """
        if self.size == 0:
            raise AssertionError(_EMPTY + "no extrema")
        return mts.extrema(self._value, axis, **kwargs)

    def normalize(self, bounds=(0., 1.), axis=0):
        """ Normalize the data (see `metrics.rescale`) """
        if self.size == 0:
            raise AssertionError(_EMPTY + "nothing to normalize")
        self._value, self._limits = mts.rescale(
            self._value, bounds=bounds, axis=axis)

    def denormalize(self, axis=0):
        """ Denormalize the data (see `metrics.rescale`) """
        if self.size == 0:
            raise AssertionError(_EMPTY + "nothing to denormalize")
        self._value, self._limits = mts.rescale(
            self._value, bounds=self._limits, axis=axis)
    #------------------------------------------------------------------------#

    #-----------------   Data's Cluster Statistical Tools   -----------------#
    def mean(self, axis=None, **kwargs):
        """ Compute the mean of the data """
        if self.size == 0:
            raise AssertionError(_EMPTY + "no mean")
        return np.mean(self._value, axis, **kwargs)

    def std(self, axis=None, **kwargs):
        """ Compute the standard deviation of the data """
        if self.size == 0:
            raise AssertionError(_EMPTY + "no std")
        return np.std(self._value, axis, **kwargs)

    def mean_std(self, axis=None, **kwargs):
        """ Compute the mean and standard deviation of the data """
        if self.size == 0:
            raise AssertionError(_EMPTY + "no mean nor std")
        return (np.mean(self._value, axis, **kwargs),
                np.std(self._value, axis, **kwargs))

    def density(self,
        span='sphere_span', volume='hypersphere', distance='euclidean'):
        """ Compute the data's density
            (see `clustering.metrics.density`) """
        if self.size == 0:
            raise AssertionError(_EMPTY + "no density")
        return mts.density(self._value, span, volume, distance)

    def avstd(self, minmax=False):
        """ Compute the Average Standard Deviation
            (see `clustering.metrics.avstd`) """
        if self.size == 0:
            raise AssertionError(_EMPTY + "no AvStd")
        return mts.avstd(self._value, minmax)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                              Class Database                              ##
##############################################################################

class Database(_Data):
    """ Represent a database (values & indexes [& tags])

    Embed a set of indexes (e.g. names, numbers or timestamps) and a
    set of N-dimension values. An index should refer to a unique value.
    In a regular database, the data can been seen as a 2D matrix, whose
    first column contains the indexes, and the other columns are the
    data themselves, one per row. For data's handling simplicity, the
    index column is separated as a unique attribute (`index`), while
    the values are gathered in another column (`values`).

    Since the values are greatly more often accessed than the indexes,
    the class' magic methods and properties refer to the data's values.

    Constructor
    -----------
    __init__(value, index, tags=None, classes=None, *, dtype=float)

    Magic Methods
    -------------
    __getitem__(pos)
        Get the data at pos --> value = dba[pos].
    __setitem__(pos, value)
        Set the data at pos --> dba[pos] = value.
    __iter__()
        Iterate on the data --> for val in dba.
    __len__()
        Length of the data --> len(dba).
    __repr__()
        Display the data --> repr(dba).
    __str__()
        Print the data --> print(dba).
    + The standard scalar, bitwise and comparison operators.

    Attributes
    ----------
    value : ND np.ndarray, getter & setter
        The data values.
    shape : tuple of ints, getter only
        The shape of the data.
    size : int, getter only
        The size of the data (the number of items).
    ndim : int, getter only
        The dimension of the data.
    dtype : datatype, getter only
        The type of the data.
    limits : 2-tuple of floats, getter & setter
        The data limits.
    index : 1D np.ndarray, getter & setter
        The data's indexes.
    tags : list of strings, getter & setter
        The data's tags (one per dimension).
    classes : 1D np.ndarray, getter & setter
        The data's a-priori classes (one per data).

    Methods
    -------
    min(axis=None)
        Compute the data's minimal value along `axis`.
    max(axis=None)
        Compute the data's maximal value along `axis`.
    extrema(axis=0)
        Determine the data's min and max values along `axis`.
    mean(axis=None)
        Compute the mean of the data.
    std(axis=None)
        Compute the standard deviation of the data.
    mean_std(axis=None)
        Compute the mean and standard deviation of the data.
    density(span='sphere_span', volume='hypersphere', distance='euclidean')
        Compute the dataset's density.
    avstd(minmax=False)
        Compute the Average Standard Deviation.
    normalize(bounds=(0., 1.), axis=0)
        Normalize the data.
    denormalize(axis=0)
        Denormalize the data.
    copy()
        Copy the current Database object.
    select(rows=None, cols=None)
        Select part of the database.
    hstack(value, tags=None)
        Add data column(s) to the Database (value [& tags]).
    hstack_1d(value, index, classes=None)
        Append 1D data to a 1D Database (value & index [& classes]).
    vstack(value, index, classes=None)
        Add data row(s) to the Database (index & value [& classes]).
    remove(pos)
        Remove a (group of) data from the Database.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data, index, tags
    >>> value = np.arange(30).reshape(10, 3)
    >>> index = np.arange(10)
    >>> tags = [f'Dim{i}' for i in range(3)]
    >>> classes = [i for i in range(len(value))]

    # Wrap the data into a Database
    >>> dba = Database(value, index, tags, classes)

    # Compute some statistics
    >>> print(dba.mean(0))
    [13.5 14.5 15.5]
    >>> print(dba.std(0))
    [8.61684397 8.61684397 8.61684397]
    >>> print(dba.mean_std(0))
    (array([13.5, 14.5, 15.5]), array([8.61684397, 8.61684397, 8.61684397]))

    # Compute some metrics
    >>> print(dba.density())
    0.00018673606510585857
    >>> print(dba.avstd())
    8.616843969807043

    # Normalize & denormalize the database
    >>> print(dba.extrema())
    (0.0, 29.0)
    >>> dba.normalize()
    >>> print(dba.extrema())
    (0.0, 1.0)
    >>> dba.denormalize()
    >>> print(dba.extrema())
    (0.0, 29.0)

    # Manipulate the database
    >>> dba2 = dba.copy()

    # Append 3 new columns
    >>> dba2.hstack(value, ['Dim3', 'Dim4', 'Dim5'])
    >>> print(dba2.shape)
    (10, 6)

    # Append 10 new rows
    >>> dba2.vstack(dba2.value, dba2.index, dba2.classes)
    >>> print(dba2.shape)
    (20, 6)

    # Remove 3 rows
    >>> dba2.remove((1, 5, -1))
    >>> print(dba2.shape)
    (17, 6)
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, value, index, tags=None, classes=None, *, dtype=float):
        """ Instantiate a Database object (constructor)

        Parameters
        ----------
        value : ND tuple or list, np.ndarray
            The data values.
        index : int, list, tuple of np.ndarray of ints
            The data (time)stamps; must have the same length as `value`.
        [OPT] tags : list of strings
            The data tags (column names); should have the same length as
            the number of columns (first dimension) of `value`. If string,
            wrap it into a list; if int, generate a list of that number
            of elements with name `Dim{i}`.
                :Default: None
        [OPT] classes : list, tuple or np.ndarray
            The a-priori classes to which belong any data; if provided,
            must have the same length as `value` and `index`.
                :Default: None

        Other Parameters
        ----------------
        [OPT] dtype : data type, optional
            The type of the data for the `value` attribute.
                :Default: float

        Examples
        --------
        >>> import numpy as np

        # Generate dummy data, index, tags
        >>> value = np.arange(20.).reshape(10, 2)
        >>> index = np.arange(10.)
        >>> tags = [f'Dim{i}' for i in range(5)]
        >>> classes = [i for i in range(len(index))]

        # Wrap the data into a Database
        >>> dba = Database(value, index, 5)
        >>> dba = Database(value, index, tags)
        >>> dba = Database(value, index, classes)
        """
        # pylint: disable=too-many-branches

        # Check if the arguments have a supported type
        if not (isinstance(value, (Number, list, tuple, np.ndarray))
                and isinstance(index, (Number, list, tuple, np.ndarray))):
            raise TypeError(
                f"Wrong type for `value` (`{type(value)}`) and/or `index` "
                + f"(`{type(index)}`): both must be lists, tuples or np.ndarrays")

        val = np.atleast_1d(value)
        idx = np.atleast_1d(index)

        # Empty database
        if len(val) == 0 and len(idx) == 0:
            super().__init__(val, (None, None), dtype)
            self._index = np.array([], int)

        # Only one data in the database
        elif val.ndim == 1 and len(idx) == 1:
#            super().__init__(np.atleast_2d(val), (val.min(), val.max()), dtype)
            super().__init__(val, (val, val), dtype)
            self._index = idx

        # More than one data
        elif len(val) == len(idx):
            super().__init__(val, (val.min(0), val.max(0)), dtype)
            self._index = idx

        # Value and Index do not have the same length
        else:
            raise AssertionError(
                "Arguments `value` and `index` must have the same length")

        # Labels of the columns of the data
        if tags in (None, []):
            self._tags = []
        elif isinstance(tags, int):
            self._tags = [str(i) for i in range(tags)]
        elif isinstance(tags, str):
            self._tags = [tags]
        elif isinstance(tags, (list, tuple, np.ndarray)):
            self._tags = tags
        elif len(self._value) == 0:
            self._tags = tags
        else:
            raise TypeError(f"Invalid type `{type(tags)}` for `tags`: "
                + "must be int, string or list of strings")

        # Classes of the data
        if classes is None or len(classes) == 0:
            self._classes = np.empty(0)
        elif isinstance(classes, (list, tuple, np.ndarray)):
            self._classes = np.array(classes)
            if len(self._classes) != len(self.value):
                raise AssertionError(
                    "Arguments `value` and `classes` must have the same length")
        else:
            raise TypeError(f"Invalid type `{type(classes)}` for `classes`: "
                + "must be list of scalars")
    #------------------------------------------------------------------------#

    #-------------------   Magic Methods and Properties   -------------------#
    def __repr__(self):
        """ Display the `value` attribute """
        return repr(self._value)

    def __str__(self):
        """ Print the database (`index` & `value`) """
        msg = ""
        if len(self._tags) != 0:
            msg += f"Tags:\t {self._tags}\n"
        if self.size == 0:
            return msg[:-1]
        if self.ndim == 1:
            msg += f"{self._index[0]} {self._value}"
            return msg
        if len(self._classes) != 0:
            for cls, idx, val in zip(self._classes, self._index, self._value):
                msg += f"{cls} {idx} {val}\n"
        else:
            for idx, val in zip(self._index, self._value):
                msg += f"{idx} {val}\n"
        return msg[:-1] # Remove last '\n'

    @property
    def index(self):
        """ Get the data's indexes """
        return self._index

    @index.setter
    def index(self, idx):
        """ Set the data's indexes """
        if len(idx) == len(self.value):
            self._index = np.array(idx)
        else:
            raise AssertionError(
                "Argument `idx` must have the same length as `self.value`")

    @property
    def tags(self):
        """ Get the data's tags (one per dimension) """
        return self._tags

    @tags.setter
    def tags(self, tags):
        """ Set the data's tags (one per dimension) """
        self._tags = tags

    @property
    def classes(self):
        """ Get the data's classes (one per data) """
        return self._classes

    @classes.setter
    def classes(self, cls):
        """ Set the data's classes (one per data) """
        if len(cls) == len(self.value):
            self._classes = np.array(cls)
        else:
            raise AssertionError(
                "Argument `cls` must have the same length as `self.value`")
    #------------------------------------------------------------------------#

    #--------------------------   Copy Database   ---------------------------#
    def _cp_update(self, values):
        """ Update the `value` attribute (for operator overloads) """
        return Database(values, self._index.copy(),
                        self._tags.copy(), self._classes.copy())

    def copy(self):
        """ Copy the current Database object """
        return Database(self._value.copy(), self._index.copy(),
                        self._tags.copy(), self._classes.copy())
    #------------------------------------------------------------------------#

    #--------------------   Select Rows and/or Columns   --------------------#
    def _select_core(self, rows, cols):
        """ Core statements for the `select` method; separated for possible
        inheritance (e.g. `Cluster` class) """

        if rows is None:
            rows = slice(None, None)
        if cols is None:
            cols = slice(None, None)

        values = self._value[rows]

        if self._value.ndim < 2:
            values = self._value[rows]
        else:
            if values.ndim < 2:
                values = values[cols]
            else:
                values = values[:, cols]

        return (values, self._index[rows],
                get_tags(self, cols), get_classes(self, rows))

    def select(self, rows=None, cols=None):
        """ Select part of the database

        If 1D data, extract the data at `rows`, and ignore the `cols`
        argument. Dimension of the database may be changed if `cols`
        is such that `data[:, cols]` would result in a 1D array.

        Wrap the extracted sub-database into a new Database object and
        return it.

        Parameters
        ----------
        [OPT] rows : slice, list or np.ndarray
            The rows to select: if list of ints, return the specified
            rows; if slice, return the corresponding slice (use None to
            select the first and last elements, e.g., `slice(None, 10)`
            selects the 10 first items). If the `value` attribute is 1D
            data, extract the items at `rows`.
                :Default: None (any values returned)
        [OPT] cols : slice, list or np.ndarray
            The columns to select; do the exact same thing as rows, but
            alongside the dimensions either. Ignored with 1D data.
                :Default: None (any dimensions returned)

        Returns
        -------
        database : Database object
            The selected part of the database.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy database
        >>> dba = Database(
        ...     np.arange(20).reshape(10, 2),
        ...     np.arange(10),
        ...     [f'Dim{i}' for i in range(3)],
        ...     [i for i in range(10)])

        # Retrieve parts of the database
        >>> dba2 = dba.select(None, None)
        >>> dba2 = dba.select(slice(0, 5), None)
        >>> dba2 = dba.select(None, slice(0, 3))
        >>> dba2 = dba.select(slice(0, 5), slice(0, 3))
        """
        return Database(*self._select_core(rows, cols))
    #------------------------------------------------------------------------#

    #--------------------------   Data Addition   ---------------------------#
    def _add_data(self, value, index, classes=None):
        """ Add new data to the current Database; if 1D data, stack them
        horizontally; if 2D data, stack them vertically. In both cases,
        append the indexes to those of the current database. """

        if self.size == 0:
            self.__init__(value, index) # pylint: disable=C2801
            return

        # Check 'value' validity
        if not isinstance(value, (Number, list, tuple, np.ndarray)):
            raise TypeError(f"Wrong type `{type(value)}` for `value`: " + _TYPES)

        # Check 'index' validity
        if not isinstance(index, (Number, list, tuple, np.ndarray)):
            raise TypeError(f"Wrong type `{type(index)}` for `index`: " + _TYPES)

        index = np.atleast_1d(index)
        value = np.atleast_1d(value)

        # Check if there are as many indexes as data values
        if len(value) != len(index):
            raise AssertionError("Arguments `value` and `index` "
                + f"must have the same length ({len(value)} and {len(index)})")

        # Horizontally stack the 1D indexes
        self._index = np.hstack((self._index, index))
        # Horizontally (1D) or vertically (2D) stack the data
        self._value = np.concatenate((self._value, value), 0)
        # Horizontally stack the 1D classes, if any
        if classes is not None:
            if len(value) != len(classes):
                raise AssertionError("Arguments `value` and `classes` "
                    + f"must have the same length ({len(value)} and {len(classes)})")
            self._classes = np.hstack((self._classes, classes))

    def hstack(self, value, tags=None):
        """ Add data column(s) to the database (value [& tags])

        N.B.: only for 2D database; if 1D, use the `hstack_1d` method.

        Parameters
        ----------
        value : tuple or list, np.ndarray
            The data array to append as news columns (last position).
            Must have the same length as the data already present.
        [OPT] tags : (list of) string(s)
            The new column(s)'s name(s); appended to the `tags` attribute.
                :Default: None

        Returns
        -------
        None : append the columns directly to `self`.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy database
        >>> vals = np.arange(20).reshape(10, 2)
        >>> idx = np.arange(10)
        >>> tags = ['C1', 'C2']
        >>> dba = Database(vals, idx, tags)

        # Append 2 new columns
        >>> dba.hstack(vals, ['C3', 'C4'])
        >>> print(dba.shape)
        (10, 4)
        """

        if self._value.ndim == 1:
            raise AssertionError("This method is for 2D data, whilst the data "
                + "already present are 1D; use the `hstack_1d` method instead")

        # Check 'value' validity
        if not isinstance(value, (Number, list, tuple, np.ndarray)):
            raise TypeError(f"Wrong type `{type(value)}` for `value`: " + _TYPES)

        # Check the dimension of the `value` argument
        value = np.atleast_1d(value)

        # Check if the data to add correspond to a full column
        if len(value) != len(self._value):
            raise AssertionError(
                f"The length of `value` ({len(value)}) must be the same as "
                + f"that of `self.value` ({len(self.value)})")

        # Everything is in order
        self._value = np.hstack((self._value, value))

        if tags is not None:
            if len(self._tags) == 0:
                print("No tags in the original dataset, no tags added")
            elif isinstance(tags, (str, Number)):
                self._tags += [f'{tags}']
            else:
                self._tags += list(tags)

    def hstack_1d(self, value, index, classes=None):
        """ Append 1D data to a 1D Database (value & index [& classes])

        N.B.: only for 1D database; if 2D, use the `hstack` method.

        Parameters
        ----------
        value : 1D tuple or list, np.ndarray
            The data array to append as news rows (last position). Must
            have the same number of columns as the data already present.
        index : 1D tuple or list, np.ndarray
            The new data indexes; must have the same length as `value`.
        [OPT] classes : 1D list, tuple or np.ndarray
            The a-priori classes to which belong any data; if provided,
            must have the same length as `value` and `index`.
                :Default: None

        Returns
        -------
        None : append the rows directly to `self`.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy database
        >>> vals = np.arange(20)
        >>> idx = np.arange(20)
        >>> tags = ['C1']
        >>> cls = [i for i in range(len(vals))]
        >>> dba = Database(vals, idx, tags, cls)

        # Append a new set of data
        >>> dba.hstack_1d(vals, idx, cls)
        >>> print(dba.shape)
        (40,)
        """

        if self._value.ndim == 2 and self._value.shape[-1] != 1:
            raise AssertionError("This method is for 1D data, whilst the data "
                + "already present are 2D; use the `hstack` method instead")

        # Stack the data horizontally
        self._add_data(value, index, classes)

    def vstack(self, value, index, classes=None):
        """ Add data row(s) to the database (index & value [& classes]))
    
        N.B.: works only for 2D datasets, since "vstacking" a 1D dataset
            would mean to reshape it and turning it from 1D into 2D. To
            preserve the integrity of the database, "vstack" of 1D data
            is purposely made forbidden.

        Parameters
        ----------
        value : tuple or list, np.ndarray
            The data array to append as news rows (last position). Must
            have the same number of columns as the data already present.
        index : tuple or list, np.ndarray
            The new data indexes; must have the same length as `value`.
        [OPT] classes : list, tuple or np.ndarray
            The a-priori classes to which belong any data; if provided,
            must have the same length as `value` and `index`.
                :Default: None

        Returns
        -------
        None : append the rows directly to `self`.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy database
        >>> vals = np.arange(20).reshape(10, 2)
        >>> idx = np.arange(10)
        >>> cls = [i for i in range(len(vals))]
        >>> dba = Database(vals, idx, cls)

        # Append 10 new rows
        >>> dba.vstack(vals, idx, cls)
        >>> print(dba.shape)
        (20, 2)
        """

        if self._value.ndim == 1:
            raise AssertionError("Cannot vstack 1D data without reshaping "
                + "the database and changing its dimension; aborting")

        # Stack the data vertically (horizontally for `index` and `classes`)
        self._add_data(value, index, classes)
    #------------------------------------------------------------------------#

    #---------------------------   Remove Data   ----------------------------#
    def remove(self, pos):
        """ Remove a (group of) data from the database

        Parameters
        ----------
        pos : (list of) int(s)
            The indexes of the rows to remove from `value`, `index` and
            `classes` (if any). Can be a single int or a list of ints.

        Returns
        -------
        None : directly remove the data from `self`.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy database
        >>> vals = np.arange(20).reshape(10, 2)
        >>> idx = np.arange(10)
        >>> dba = Database(vals, idx)

        # Remove 1 row
        >>> dba.delete(0)
        >>> print(dba.shape)
        (9, 2)

        # Remove 3 rows
        >>> dba.delete((1, 5, -1))
        >>> print(dba.shape)
        (6, 2)
        """
        self._value = np.delete(self._value, pos, 0)
        self._index = np.delete(self._index, pos, 0)
        if len(self._classes) != 0:
            self._classes = np.delete(self._classes, pos, 0)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                              Class Cluster                               ##
##############################################################################

class Cluster(Database):
    """ Represent a cluster (values & indexes & pattern [& tags & classes])

    Inherit from the Database class to extend it with a prototype, which
    represents the barycenter of the data. Simply add it to the existing
    Database's attributes and update some methods to consider it.

    Constructor
    -----------
    __init__(value, index,
        pattern=None, tags=None, classes=None, *, dtype=float)

    Magic Methods
    -------------
    See `Database` class' documentation.

    Attributes
    ----------
    pattern : 1D np.array, getter & setter
        The data's pattern (~representative).
    + Those of the `Database` class.

    Methods
    -------
    set_cluster(value, index, pattern=None,
                tags=None, classes=None, *, dtype=float)
        (Re)set the cluster (constructor alias).
    + Those of the `Database` class.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data, index, tags
    >>> value = np.arange(30).reshape(10, 3)
    >>> index = np.arange(10)
    >>> pattern = value.mean(0)
    >>> tags = [f'Dim{i}' for i in range(3)]
    >>> classes = [i for i in range(len(value))]

    # Wrap the data into a cluster
    >>> clt = Cluster(value, index, pattern, tags, classes)

    # Compute some statistics
    >>> print(clt.mean(0))
    [13.5 14.5 15.5]
    >>> print(clt.std(0))
    [8.61684397 8.61684397 8.61684397]
    >>> print(clt.mean_std(0))
    (array([13.5, 14.5, 15.5]), array([8.61684397, 8.61684397, 8.61684397]))

    # Compute some metrics
    >>> print(clt.density())
    0.00018673606510585857
    >>> print(clt.avstd())
    8.616843969807043

    # Normalize & denormalize the database
    >>> print(clt.extrema())
    (0.0, 29.0)
    >>> clt.normalize()
    >>> print(clt.extrema())
    (0.0, 1.0)
    >>> clt.denormalize()
    >>> print(clt.extrema())
    (0.0, 29.0)

    # Manipulate the database
    >>> clt2 = clt.copy()

    # Append 3 new columns
    >>> clt2.hstack(value, ['Dim3', 'Dim4', 'Dim5'])
    >>> print(clt2.shape)
    (10, 6)

    # Append 10 new rows
    >>> clt2.vstack(clt2.value, clt2.index, clt2.classes)
    >>> print(clt2.shape)
    (20, 6)

    # Remove 3 rows
    >>> clt2.remove((1, 5, -1))
    >>> print(clt2.shape)
    (17, 6)

    # Rebuild the cluster
    >>> clt2.set_cluster(value, index, pattern, tags, classes)
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, value, index,
        pattern=None, tags=None, classes=None, *, dtype=float):
        """ Instantiate a Cluster object (constructor)

        Parameters
        ----------
        value : ND tuple or list, np.ndarray
            The data values.
        index : 1D tuple or list, np.ndarray
            The data (time)stamps; must have the same length as `value`.
        [OPT] pattern : 1D tuple or list, np.ndarray
            The cluster characteristic pattern (prototype, barycenter, etc.).
                :Default: None
        [OPT] tags : list of strings
            The data tags (column names); should have the same length as
            the number of columns (first dimension) of `value`. If string,
            wrap it into a list; if int, generate a list of that number
            of elements with name `Dim{i}`.
                :Default: None
        [OPT] classes : list, tuple or np.ndarray
            The a-priori classes to which belong any data; if provided,
            must have the same length as `value` and `index`.
                :Default: None

        Other Parameters
        ----------------
        [OPT] dtype : data type, optional
            The type of the data for the `value` attribute.
                :Default: float

        Examples
        --------
        >>> import numpy as np

        # Generate dummy data, index, patter, tags
        >>> value = np.arange(20.).reshape(10, 2)
        >>> index = np.arange(10.)
        >>> pattern = np.mean(value, 0)     # Mean as pattern
        >>> tags = [f'Dim{i}' for i in range(5)]
        >>> classes = [i for i in range(len(value))]

        # Wrap the data into a Cluster
        >>> clt = Cluster(value, index)
        >>> clt = Cluster(value, index, pattern)
        >>> clt = Cluster(value, index, pattern, 5)
        >>> clt = Cluster(value, index, pattern, tags)
        >>> clt = Cluster(value, index, pattern, tags, classes)
        """

        # Empty pattern (no Cluster possible)
        if pattern is None and np.size(value) == 0:
            raise AssertionError("Empty database and no pattern")

        # Build the Database with the data (values and indexes)
        super().__init__(value, index, tags, classes, dtype=dtype)

        # Add the pattern to the Database to form a Cluster
        if pattern is None:
            self._pattern = self.mean(0)
        elif np.ndim(pattern) == 0: # Scalar
            self._pattern = np.array(pattern)

        # Nonempty pattern
        elif isinstance(pattern, (Number, list, tuple, np.ndarray)):

            # Empty database or consistent dimensions if nonempty
            dim = 1 if self._value.ndim == 1 else self._value.shape[-1]
            if self.size == 0 or np.size(pattern) == dim:
                self._pattern = np.atleast_1d(np.array(pattern, float))

            # Inconsistent dimensions
            else:
                raise AssertionError("Last dimension of "
                    + f"`pattern` ({np.shape(pattern)[-1]}) and `self.value` "
                    + f"({self.value.shape[-1]}) should be the same")

        # Wrong type of argument
        else:
            raise TypeError(f"Wrong type `{type(pattern)}` for `pattern`: " + _TYPES)
    #------------------------------------------------------------------------#

    #-------------------   Magic Methods and Properties   -------------------#
    def __str__(self):
        """ Print the cluster (`pattern`, `index` & `value`) """
        msg = f"Pattern:\t {self._pattern}\n"
        msg += f"Data in:\t {len(self._index)} data instance(s)\n"
        msg += super().__str__()
        return msg

    @property
    def pattern(self):
        """ Get the data's pattern (~representative) """
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        """ Set the data's pattern (~representative) """
        self._pattern[:] = pattern[:]
    #------------------------------------------------------------------------#

    #---------------------------   Copy & Reset   ---------------------------#
    def _cp_update(self, values):
        """ Update the `value` attribute (for operator overloads) """
        return Cluster(values, self._index.copy(),
            self._pattern.copy(), self._tags.copy(), self._classes.copy())

    def copy(self):
        """ Copy the current Cluster object """
        return Cluster(self._value.copy(), self._index.copy(),
            self._pattern.copy(), self._tags.copy(), self._classes.copy())

    def set_cluster(self, value, index,
        pattern=None, tags=None, classes=None, *, dtype=float):
        """ (Re)set the cluster (constructor alias) """
        # pylint: disable-next=unnecessary-dunder-call
        self.__init__(value, index, pattern, tags, classes, dtype=dtype)
    #------------------------------------------------------------------------#

    #--------------------   Select Rows and/or Columns   --------------------#
    def select(self, rows=None, cols=None):
        """ Select part of the cluster

        If 1D data, extract the data at `rows`, and ignore the `cols`
        argument. Dimension of the cluster may be changed if `cols` is
        such that `data[:, cols]` would result in a 1D array.

        Wrap the extracted sub-cluster into a new Cluster object and
        return it.

        N.B.: for 1D data, the (1D) pattern is copied from the original
            cluster, and it is not updated; it may be unrepresentative
            of the so-extracted cluster.

        Parameters
        ----------
        [OPT] rows : slice, list or np.ndarray
            The rows to select: if list of ints, return the specified
            rows; if slice, return the corresponding slice (use None to
            select the first and last elements, e.g., `slice(None, 10)`
            selects the 10 first items). If the `value` attribute is 1D
            data, extract the items at `rows`.
                :Default: None (any values returned)
        [OPT] cols : slice, list or np.ndarray
            The columns to select; do the exact same thing as rows, but
            alongside the dimensions either. Ignored with 1D data.
                :Default: None (any dimensions returned)

        Returns
        -------
        cluster : Cluster object
            The selected part of the cluster.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy cluster
        >>> clt = Cluster(
        ...     np.arange(20).reshape(10, 2),
        ...     np.arange(10),
        ...     np.arange(2, dtype=float),
        ...     [f'Dim{i}' for i in range(3)],
        ...     [i for i in range(10)])

        # Retrieve parts of the cluster
        >>> clt2 = clt.select(None, None)
        >>> clt2 = clt.select(slice(0, 5), None)
        >>> clt2 = clt.select(None, slice(0, 3))
        >>> clt2 = clt.select(slice(0, 5), slice(0, 3))
        """

        # Extract the values, index, tags and notes
        values, index, tags, notes = self._select_core(rows, cols)

        # If 1D data, use the `pattern` attribute; else, extract the cols
        pattern = self._pattern if self._value.ndim == 1 else self._pattern[cols]

        # Wrap the extracted values into a new Cluster
        return Cluster(values, index, pattern, tags, notes)
    #------------------------------------------------------------------------#

    #--------------------------   Data Addition   ---------------------------#
    def hstack(self, value, tags=None):
        """ Add data column(s) to the cluster (value [& tags])

        Call the `hstack` method from the `Database` parent class, and
        append the mean of `value` as a new dimension for the `pattern`
        attribute. The other dimensions are left unchanged.

        N.B.: only for 2D cluster; if 1D, use the `hstack_1d` method;
            in such a case, the `pattern` attribute is left unchanged
            as it should be a scalar.

        Parameters
        ----------
        value : tuple or list, np.ndarray
            The data array to append as news columns (last position).
            Must have the same length as the data already present.
        [OPT] tags : (list of) string(s)
            The new column(s)'s name(s); appended to the `tags` attribute.
                :Default: None

        Returns
        -------
        None : append the columns directly to `self`.

        Examples
        --------
        >>> import numpy as np

        # Generate a dummy cluster
        >>> vals = np.arange(20).reshape(10, 2)
        >>> idx = np.arange(10)
        >>> pat = vals.mean(0)
        >>> tags = ['C1', 'C2']
        >>> clt = Cluster(vals, idx, pat, tags)

        >>> print(clt.shape)
        (10, 2)
        >>> print(clt.pattern.shape)
        (2,)

        # Append 2 new columns
        >>> clt.hstack(vals, ['C3', 'C4'])

        >>> print(clt.shape)
        (10, 4)
        >>> print(clt.pattern.shape)
        (4,)
        """

        # Append the input `value` to the `value` attribute
        super().hstack(value, tags)

        # Expand the `pattern` attribute with the column-wise means
        if self._value.ndim != 1:
            self._pattern = np.hstack((self._pattern, np.mean(value, 0)))
#------------------------------------------------------------------------#

##############################################################################
