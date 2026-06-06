import numpy as np
from clustering.formats._database import Database
from clustering.formats._checkers import *
np.set_printoptions(legacy='1.21')


def test_check_data():
    """ Check data format """
    # Generate a dummy database
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    # Check the data format
    data = check_data(np.arange(5)).shape
    print(np.shape(data))
    data = check_data(dba)
    print(np.shape(data))

def test_get_index():
    """ Get the index of a database """
    # Generate a dummy database
    dba = Database(np.arange(20).reshape(10, 2), np.arange(10))
    # Retrieve some indexes
    print(get_index(np.arange(5)))
    print(get_index(dba))

def test_get_tags():
    """ Get the tags of a database """
    # Generate a dummy database
    dba = Database(
        np.arange(20).reshape(10, 2), np.arange(10), ['C1', 'C2'])
    # Retrieve some tags
    print(get_tags(np.arange(5)))
    print(get_tags(dba, idx=None))
    print(get_tags(dba, idx=1))
    print(get_tags(dba, idx=[0, 1]))

def test_get_classes():
    """ Get the classes of a database """
    # Generate a dummy database
    dba = Database(
        np.arange(20).reshape(10, 2), np.arange(10),
        classes=[i for i in range(10)])
    # Retrieve some tags
    print(get_classes(np.arange(5)))
    print(get_classes(dba, idx=None))
    print(get_classes(dba, idx=1))
    print(get_classes(dba, idx=[0, 1]))



# Launch test/example functions
test_check_data()

test_get_index()

test_get_tags()

test_get_classes()

