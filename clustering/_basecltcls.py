""" Base class for clustering algorithms

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: May 2026
Last revised: May 2026

License: GPLv3
"""

__all__ = ['BOUNDS', '_BaseClustering']

import numpy as np

import clustering.metrics as mts
import clustering.formats as dformat

BOUNDS = (0., 1.)


##############################################################################
##                          Base Clustering Class                           ##
##############################################################################

class _BaseClustering():
    """ Base class for clustering methods """

    # Constants
    _MAX_SIZE = 10_000_000          # Max number of items in a lone matrix
    MARGINS = 0.01                  # Max variation before convergence

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, seed=None, distance='euclidean'):
        """ Check the distance function, instantiate the random number
        generator and declare the common attributes (set to None here)

        Parameters
        ----------
        [OPT] seed : int
            The seed for the NumPy random number generator. See NumPy's
            `random.default_rng` function for details.
                :Default: None
        [OPT] distance : str
            The distance name; see the `get_dist_func` function from the
            `metrics` module for details.
                :Default: 'euclidean'
        """

        # Random number generator
        self._rng = np.random.default_rng(seed)

        # Set the distance and declare the future distance function
        # (set later as the distance function depends on data dimension)
        self._dist = distance
        self._fdist = None

        # Patterns set in the `fit` & `_initialize` methods
        self._pats = None

        # Extremal values of the training data (for normalization)
        self._limits = None
        self._norm = None
    #------------------------------------------------------------------------#

    #----------------------   Properties/Attributes   -----------------------#
    @property
    def distance(self):
        """ Get the distance function in use """
        return self._dist

    @distance.setter
    def distance(self, distance):
        """ Set the distance function to use """
        self._fdist = mts.get_dist_func(distance)
        self._dist = distance

    @property
    def patterns(self):
        """ Get the clusters' patterns (~representatives) """
        # If training data were normalized, do not denormalize patterns
        if self._norm:
            return self._pats
        # Else, denormalize the patterns in the training data range
        return mts.rescale(self._pats, bounds=self._limits, limits=BOUNDS)[0]
    #------------------------------------------------------------------------#

    #---------------------   Patterns Initialization   ----------------------#
    def _initialize(self, data, nb_patterns):
        """ Randomly initialize the patterns

        Randomly draw `nb_patterns` samples from `data`, and assign each
        one of these samples to a unique cluster's prototype (pattern).

        Parameters
        ----------
        data : np.ndarray or Database
            The data to initialize the patterns (typically that also used
            for training).
        nb_patterns : int
            The number of patterns to initialize.

        Returns
        -------
        None : directly set the `patterns` attribute.
        """

        # Get the distance function from its name (passed in the constructor)
        # The function depends on the data dimension
        if self._dist is not None:
            self._fdist = mts.get_dist_func(self._dist, data.ndim)

        # Initialize the patterns as random data from the dataset
        self._pats = data[self._rng.integers(0, len(data), nb_patterns)]

        # Ensure the patterns are a 2D array
        if self._pats.ndim == 1:
            self._pats = np.expand_dims(self._pats, 1)
    #------------------------------------------------------------------------#

    #------------------------   Data Normalization   ------------------------#
    def _normalize(self, data):
        """ Normalize and return input data

        Similar method to `normalize_self`, but does not modify the
        class attribute.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to rescale between `BOUNDS`.

        Other Parameters
        ----------------
        [GLB] BOUNDS : 2-tuple of ints, global variable
            The range within which to rescale the data, organized as
            `(min, max)`.

        Returns
        -------
        data_norm : same type as `data`
            The data normalized between `BOUNDS`.
        """

        # Check data type and format
        data = dformat.check_data(data)

#        # Check if the data are normalized between `BOUNDS` (global)
#        norm, limits = mts.check_scale(data, BOUNDS, self.MARGINS, 0)

#        # If not, normalize them
#        if not norm:
#            data = mts.rescale(data, bounds=BOUNDS, limits=limits)[0]

        return mts.rescale(data, bounds=BOUNDS, limits=self._limits)[0]

    def _normalize_self(self, data):
        """ Normalize input data and update `norm` and `limits` attributes

        Similar method to `normalize`, but modifies the class attribute.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to rescale between `BOUNDS`.

        Other Parameters
        ----------------
        [GLB] BOUNDS : 2-tuple of ints, global variable
            The range within which to rescale the data, organized as
            `(min, max)`.

        Returns
        -------
        data_norm : same type as `data`
            The data normalized between `BOUNDS`.
        """

        # Check data type and format
        data = dformat.check_data(data)

        # Check if the data are normalized between `BOUNDS` (global),
        # and set the `_norm` and `_limits` attributes
        self._norm, self._limits = mts.check_scale(data, BOUNDS, self.MARGINS, 0)

        # If data not already between `BOUNDS`, normalize them
        if not self._norm:
            data = mts.rescale(data, bounds=BOUNDS, limits=self._limits)[0]

        return data
    #------------------------------------------------------------------------#

    #----------------------   Build Data Categories   -----------------------#
    def _build_cats(self, data):
        """ Build the categories (classes) of the input data

        Take a set of data, compute their respective distances to any
        patterns and, for each data, give it the class of the closest
        pattern. Regroup the data indexes with the same category in the
        same set, and return the set of these sets, one per pattern.

        Parameters
        ----------
        data : np.ndarray or Database
            The data to regroup by category.

        Returns
        -------
        cats : 2D np.ndarray
            The set of sets of indexes of the data belonging to the same
            category (i.e. with the same closest pattern).
        """

        # Get the categories (classes)
        if not len(data) * len(self._pats) > self._MAX_SIZE:
            cats = np.argmin([self._fdist(pat, data[:], 1)
                              for pat in self._pats], 0)

        # If required memory larger than `8*self.__MAX_SIZE`, unroll the loop
        else:
            cats = np.array([np.argmin(self._fdist(value, self._pats, 1))
                             for value in data[:]], int)

        return cats
    #------------------------------------------------------------------------#

    #----------------------   Best Matching Unit BMU   ----------------------#
    def winner(self, data):
        """ Find the Best Matching Unit for a data

        Search for the nearest prototype of a data and return its posi-
        tion number. The distance between the input data and any of the
        prototypes of the grid's nodes is computed and the one at the
        smallest distance is identified as the Best Matching Unit BMU;
        its position in the list of clusters is returned a an integer.
        Do this for any item in `data` if it contains more than 1 data.

        Note: the distance is that specified in the 'distance' attribute.

        Parameters
        ----------
        data : np.ndarray
            The data sample to categorize.

        Returns
        -------
        bmus : (array of) int(s)
            Cluster index to which belong `data`; if single array, return
            its BMU as an int; if set of arrays, return their BMUs as an
            array of ints, whose i-th value is the BMU of the i-th data.

        Examples
        --------
        # This method is shared by several classes; in the following
        # examples, assume the class is `Clustering`, that should be
        # replaced with any children class (`KohonenSOM`, `KMeans`, etc.)

        >>> import numpy as np

        # Generate a dummy database
        >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))

        # Instantiate a `Clustering` object
        >>> method = Clustering()

        # Train the clustering method
        >>> method.fit(database)

        # Build the clusters from the trained patterns
        >>> method.build(database)

        # Find the BMU of a single data
        >>> bmu = method.winner(database[15])

        # Find the BMUs of a set of data
        >>> bmus = method.winner(database[10:15])
        """

        # Wrap any scalar input into np.ndarray
        if np.ndim(data) == 0:
            data = np.array([data])

        # Normalize the data
        data = self._normalize(data)

        # If only one array
        if (self._pats.shape[-1] == 1 and len(data) == 1
            or data.ndim == 1 and self._pats.shape[-1] > 1):
            return np.argmin(self._fdist(data, self._pats, 1))

        # If several arrays, compute the BMU for each of them
        return np.array(
            [np.argmin(self._fdist(val, self._pats, 1)) for val in data])
    #------------------------------------------------------------------------#

    #-----------------------   Fit Method (Virtual)   -----------------------#
    def fit(self, data):
        """ Train the patterns with the input data """
        raise NotImplementedError
    #------------------------------------------------------------------------#

##############################################################################
