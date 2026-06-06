import numpy as np
from clustering.formats._database import *
np.set_printoptions(legacy='1.21')


def test_Database():
    """ Represent a database (values & indexes [& tags]) """
    # Generate dummy data, index, tags and notes
    value = np.arange(30).reshape(10, 3)
    index = np.arange(10)
    tags = [f'Dim{i}' for i in range(3)]
    classes = [i for i in range(len(value))]
    # Wrap the data into a Database
    dba = Database(value, index, tags, classes)
    # Normalize & denormalize the database
    print(dba.min(0))
    print(dba.max(0))
    print(dba.extrema())
    dba.normalize()
    print(dba.extrema())
    dba.denormalize()
    print(dba.extrema())
    # Compute some statistics
    print(dba.mean(0))
    print(dba.std(0))
    print(dba.mean_std(0))
    # Compute some metrics
    print(dba.density())
    print(dba.avstd())
    # Manipulate the database
    dba2 = dba.copy()
    # Append 3 new columns
    dba2.hstack(value, ['Dim3', 'Dim4', 'Dim5'])
    print(dba2.shape)
    # Append 10 new rows
    dba2.vstack(dba2.value, dba2.index, dba2.classes)
    print(dba2.shape)
    # Remove 3 rows
    dba2.remove((1, 5, -1))
    print(dba2.shape)

def test_Database_init():
    """ Instantiate a Database object (constructor) """
    # Generate dummy data, index, tags
    value = np.arange(20.).reshape(10, 2)
    index = np.arange(10.)
    tags = [f'Dim{i}' for i in range(3)]
    classes = [i for i in range(len(value))]
    # Wrap the data into a Database
    dba = Database(value, index, 5)
    dba = Database(value, index, tags)
    dba = Database(value, index, classes)

def test_Database_min():
    """ Compute the data's min and max values along `axis` """
    # Generate dummy data and compute their extremal values
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.min(0))

def test_Database_max():
    """ Compute the data's min and max values along `axis` """
    # Generate dummy data and compute their extremal values
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.max(0))

def test_Database_extrema():
    """ Compute the data's min and max values along `axis` """
    # Generate dummy data and compute their extremal values
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.extrema())

def test_Database_normalize():
    """ Normalize the data """
    # Generate dummy data and normalize it
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.extrema())
    dba.normalize()
    print(dba.extrema())

def test_Database_denormalize():
    """ Denormalize the data """
    # Generate dummy data, normalize it and denormalize it then
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.extrema())
    dba.normalize()
    print(dba.extrema())
    dba.denormalize()
    print(dba.extrema())

def test_Database_mean():
    """ Compute the mean of the data """
    # Generate dummy data and compute their mean
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.mean(1))

def test_Database_std():
    """ Compute the standard deviation of the data """
    # Generate dummy data and compute their standard deviation
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.std(1))

def test_Database_mean_std():
    """ Compute the mean and standard deviation of the data """
    # Generate dummy data and compute their mean & std
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.mean_std(1))

def test_Database_density():
    """ Compute the data's density """
    # Generate dummy data and compute their density
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.density())

def test_Database_avstd():
    """ Compute the Average Standard Deviation """
    # Generate dummy data and compute their Average Std
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    print(dba.avstd())

def test_Database_copy():
    """ Copy the current Database object """
    # Generate dummy data, normalize it and denormalize it then
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    # Copy the database
    dba2 = dba.copy()
    dba2.value += 1.
    print(np.all(dba.value == dba2.value))

def test_Database_select():
    """ Select part of the database """
    # Generate a dummy database
    dba = Database(
        np.arange(20).reshape(10, 2),
        np.arange(10),
        [f'Dim{i}' for i in range(3)],
        [i for i in range(10)])
    # Retrieve parts of the database
    dba2 = dba.select(None, None)
    dba2 = dba.select(slice(0, 5), None)
    dba2 = dba.select(None, slice(0, 3))
    dba2 = dba.select(slice(0, 5), slice(0, 3))

def test_Database_hstack():
    """ Add data column(s) to the Database (value & tags) """
    # Generate a dummy database
    vals = np.arange(20).reshape(10, 2)
    idx = np.arange(10)
    tags = ['C1', 'C2']
    dba = Database(vals, idx, tags)
    # Append 2 new columns
    dba.hstack(vals, ['C3', 'C4'])
    print(dba.shape)

def test_Database_hstack_1d():
    """ Append 1D data to a 1D Database (value & index [& classes]) """
    # Generate a dummy database
    vals = np.arange(20)
    idx = np.arange(20)
    tags = ['C1']
    cls = [i for i in range(len(vals))]
    dba = Database(vals, idx, tags, cls)
    # Append a new set of data
    dba.hstack_1d(vals, idx, cls)
    print(dba.shape)

def test_Database_vstack():
    """ Add data row(s) to the Database (index & value) """
    # Generate a dummy database
    vals = np.arange(20).reshape(10, 2)
    idx = np.arange(10)
    cls = [i for i in range(len(vals))]
    dba = Database(vals, idx, classes=cls)
    # Append 10 new rows
    dba.vstack(vals, idx, cls)
    print(dba.shape)

def test_Database_remove():
    """ Remove a (group of) data from the Database """
    # Generate a dummy database
    vals = np.arange(20).reshape(10, 2)
    idx = np.arange(10)
    dba = Database(vals, idx)
    # Remove 1 row
    dba.remove(0)
    print(dba.shape)
    # Remove 3 rows
    dba.remove((1, 5, -1))
    print(dba.shape)


def test_Cluster():
    """ Represent a cluster (values & indexes & pattern [& tags]) """
    # Generate dummy data, index, tags
    value = np.arange(30).reshape(10, 3)
    index = np.arange(10)
    pattern = value.mean(0)
    tags = [f'Dim{i}' for i in range(3)]
    classes = [i for i in range(len(value))]
    # Wrap the data into a cluster
    clt = Cluster(value, index, pattern, tags, classes)
    # Compute some statistics
    print(clt.mean(0))
    print(clt.std(0))
    print(clt.mean_std(0))
    # Compute some metrics
    print(clt.density())
    print(clt.avstd())
    # Normalize & denormalize the database
    print(clt.extrema())
    clt.normalize()
    print(clt.extrema())
    clt.denormalize()
    print(clt.extrema())
    # Manipulate the database
    clt2 = clt.copy()
    # Append 3 new columns
    clt2.hstack(value, ['Dim3', 'Dim4', 'Dim5'])
    print(clt2.shape)
    # Append 10 new rows
    clt2.vstack(clt2.value, clt2.index, clt2.classes)
    print(clt2.shape)
    # Remove 3 rows
    clt2.remove((1, 5, -1))
    # Rebuild the cluster
    clt2.set_cluster(value, index, pattern, tags, classes)

def test_Cluster_init():
    """ Instantiate a Cluster object (constructor) """
    # Generate dummy data, index, patter, tags
    value = np.arange(20.).reshape(10, 2)
    index = np.arange(10.)
    pattern = np.mean(value, 0)     # Mean as pattern
    tags = [f'Dim{i}' for i in range(5)]
    classes = [i for i in range(len(value))]
    # Wrap the data into a Cluster
    clt = Cluster(value, index)
    clt = Cluster(value, index, pattern)
    clt = Cluster(value, index, pattern, 5)
    clt = Cluster(value, index, pattern, tags)
    clt = Cluster(value, index, pattern, tags, classes)

def test_Cluster_copy():
    """ Copy the current Cluster object """
    # Generate dummy data, normalize it and denormalize it then
    clt = Cluster(np.arange(20).reshape(10, 2), np.arange(10), (9., 10.))
    # Copy the database
    clt2 = clt.copy()
    clt2.value += 1.
    print(np.all(clt.value == clt2.value))

def test_Cluster_set_cluster():
    """ Set the cluster (constructor alias) """
    # Generate a dummy database
    value = np.arange(20).reshape(10, 2)
    index = np.arange(10)
    pattern = value.mean(0)
    clt = Cluster(value, index, pattern)
    # Rebuild the cluster
    clt.set_cluster(value, index, pattern)

def test_Cluster_select():
    """ Select part of the cluster """
    # Generate a dummy cluster
    clt = Cluster(
        np.arange(20).reshape(10, 2),
        np.arange(10),
        np.arange(2, dtype=float),
        [f'Dim{i}' for i in range(3)],
        [i for i in range(10)])
    # Retrieve parts of the cluster
    clt2 = clt.select(None, None)
    clt2 = clt.select(slice(0, 5), None)
    clt2 = clt.select(None, slice(0, 3))
    clt2 = clt.select(slice(0, 5), slice(0, 3))

def test_Cluster_hstack():
    """ Add data column(s) to the cluster (value [& tags]) """
    # Generate a dummy cluster
    vals = np.arange(20).reshape(10, 2)
    idx = np.arange(10)
    pat = vals.mean(0)
    tags = ['C1', 'C2']
    clt = Cluster(vals, idx, pat, tags)
    print(clt.shape)
    print(clt.pattern.shape)
    # Append 2 new columns
    clt.hstack(vals, ['C3', 'C4'])
    print(clt.shape)
    print(clt.pattern.shape)


def test_Database_1d():
    """ Represent a database (values & indexes [& tags]) """
    #--- Test 1D row data
    value = np.arange(30)
    index = np.arange(30)
    tags = 'Dim1'
    classes = [i for i in range(len(value))]
    # Wrap the data into a Database
    dba = Database(value, index, tags, classes)
    # Normalize & denormalize the database
    print(dba.min(0))
    print(dba.max(0))
    print(dba.extrema())
    dba.normalize()
    print(dba.extrema())
    dba.denormalize()
    print(dba.extrema())
    # Compute some statistics
    print(dba.mean(0))
    print(dba.std(0))
    print(dba.mean_std(0))
    # Compute some metrics
    print(dba.density())
    print(dba.avstd())
    # Add a new set of data
    dba2 = dba.copy()
    dba2.hstack_1d(dba2.value, dba2.index, dba2.classes)
    print(dba2.shape)
    dba2.remove((1, 5, -1))
    print(dba2.shape)
    # Select some data
    dba3 = dba.select(slice(0, 5))
    print(dba3.shape)
    #--- Test 1D column data
    value = np.arange(30).reshape(-1, 1)
    index = np.arange(30)
    tags = 'Dim1'
    classes = [i for i in range(len(value))]
    # Wrap the data into a Database
    dba = Database(value, index, tags, classes)
    # Normalize & denormalize the database
    print(dba.min(0))
    print(dba.max(0))
    print(dba.extrema())
    dba.normalize()
    print(dba.extrema())
    dba.denormalize()
    print(dba.extrema())
    # Compute some statistics
    print(dba.mean(0))
    print(dba.std(0))
    print(dba.mean_std(0))
    # Compute some metrics
    print(dba.density())
    print(dba.avstd())
    # Append 1 new column
    dba2 = dba.copy()
    dba2.hstack(value, ['Dim2'])
    print(dba2.shape)
    # Append 30 new rows
    dba2 = dba.copy()
    dba2.vstack(dba2.value, dba2.index, dba2.classes)
    print(dba2.shape)
    # Remove 3 rows
    dba2.remove((1, 5, -1))
    print(dba2.shape)
    # Select some data
    dba3 = dba.select(slice(0, 5), 0)
    print(dba3.shape)

def test_Cluster_1d():
    """ Represent a cluster (values & indexes & pattern [& tags]) """
    #--- Test 1D row data
    value = np.arange(30)
    index = np.arange(30)
    pattern = value.mean(0)
    tags = 'Dim1'
    classes = [i for i in range(len(value))]
    # Wrap the data into a cluster
    clt = Cluster(value, index, pattern, tags, classes)
    # Normalize & denormalize the database
    print(clt.min(0))
    print(clt.max(0))
    print(clt.extrema())
    clt.normalize()
    print(clt.extrema())
    clt.denormalize()
    print(clt.extrema())
    # Compute some statistics
    print(clt.mean(0))
    print(clt.std(0))
    print(clt.mean_std(0))
    # Compute some metrics
    print(clt.density())
    print(clt.avstd())
    # Add a new set of data
    clt2 = clt.copy()
    clt2.hstack_1d(clt2.value, clt2.index, clt2.classes)
    print(clt2.shape)
    # Remove 3 rows
    clt2.remove((1, 5, -1))
    # Select some data
    clt3 = clt.select(slice(0, 5), 0)
    print(clt3.shape)
    # Rebuild the cluster
    clt2.set_cluster(value, index, pattern, tags, classes)
    #--- Test 1D column data
    value = np.arange(30).reshape(-1, 1)
    index = np.arange(30)
    pattern = value.mean(0)
    tags = 'Dim1'
    classes = [i for i in range(len(value))]
    # Wrap the data into a cluster
    clt = Cluster(value, index, pattern, tags, classes)
    # Normalize & denormalize the database
    print(clt.min(0))
    print(clt.max(0))
    print(clt.extrema())
    clt.normalize()
    print(clt.extrema())
    clt.denormalize()
    print(clt.extrema())
    # Compute some statistics
    print(clt.mean(0))
    print(clt.std(0))
    print(clt.mean_std(0))
    # Compute some metrics
    print(clt.density())
    print(clt.avstd())
    # Append 1 new column
    clt2 = clt.copy()
    clt2.hstack(value, ['Dim2'])
    print(clt2.shape)
    # Append 30 new rows
    clt2 = clt.copy()
    clt2.vstack(clt2.value, clt2.index, clt2.classes)
    print(clt2.shape)
    # Remove 3 rows
    clt2.remove((1, 5, -1))
    # Select some data
    clt3 = clt.select(slice(0, 5), 0)
    print(clt3.shape)
    # Rebuild the cluster
    clt2.set_cluster(value, index, pattern, tags, classes)



# Launch test/example functions
test_Database()

test_Database_init()

test_Database_min()

test_Database_max()

test_Database_extrema()

test_Database_normalize()

test_Database_denormalize()

test_Database_mean()

test_Database_std()

test_Database_mean_std()

test_Database_density()

test_Database_avstd()

test_Database_copy()

test_Database_select()

test_Database_hstack()

test_Database_hstack_1d()

test_Database_vstack()

test_Database_remove()


test_Cluster()

test_Cluster_init()

test_Cluster_copy()

test_Cluster_set_cluster()

test_Cluster_select()

test_Cluster_hstack()


test_Database_1d()

test_Cluster_1d()

