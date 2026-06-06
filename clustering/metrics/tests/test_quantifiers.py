import numpy as np
from clustering.metrics._quantifiers import *


def test_avstd():
    """ Average Standard Deviation """
    # Generate dummy data
    dba = np.arange(50., dtype=float).reshape(-1, 5)
    # Compute the AvStd
    mini, mean, maxi = avstd(dba, True)
    print(mini, mean, maxi)

def test_silhouettes():
    """ Silhouette Coefficient of a database composed of clusters """
    # Generate dummy data
    dba = np.arange(50., dtype=float).reshape(-1, 5)
    clts = [dba + i for i in range(3)]
    # Compute the SCs for every data of every cluster
    sils = silhouettes(clts)
    _ = [print(sil.mean()) for sil in sils]
    # Compute the SC of the data `clts[1][7]` from `clts[1]`
    sil = silhouettes(clts, (clts[1][7], 1))
    print(sil)

def test_density():
    """ Compute the density of a cluster """
    # Generate dummy data
    dba = np.arange(50., dtype=float).reshape(-1, 5)
    # Compute the density of the dataset
    print(density(dba))

def test_quantifiers():
    """ Compute the HyDensity, AvStd and Silhouettes quantifiers """
    # Generate dummy data
    dba = np.arange(50., dtype=float).reshape(-1, 5)
    clts = [dba + i for i in range(3)]
    # Compute the quantifiers of the clusters (SCs, AvStd, HyDensity)
    quants = quantifiers(clts)
    print(quants)

def test_hydas():
    """ Hybridized Density-AvStd-Silhouettes-based metric HyDAS """
    # Generate dummy data
    dba = np.arange(50., dtype=float).reshape(-1, 5)
    clts = [dba + i for i in range(3)]
    # Compute the quantifiers of the clusters (SCs, AvStd, HyDensity)
    quants = quantifiers(clts)
    # Build the HyDAS scores
    stats = hydas(clts, stats=quants)
    print(stats.round(3))

def test_get_quant_func():
    """ Get the reference to a specified quantifier function """
    # Generate dummy data
    data = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the density of the dataset
    fquant = get_quant_func('density')
    den = fquant(data,
        span='sphere_span', volume='hypersphere', distance='euclidean')
    print(den)
    # Compute the Silhouettes between two datasets
    fquant = get_quant_func('silhouettes')
    sils = fquant([data, data+1.])
    print(sils)

def test_quants_1d():
    """ Check 1D data """
    #--- Test 1D row array
    dba = np.arange(10., dtype=float)
    clts = [dba + i for i in range(3)]
    print(avstd(dba, True))
    print(density(dba))
    print(silhouettes(clts))
    print(quants:=quantifiers(clts))
    print(hydas(clts, stats=quants))
    #--- Test 1D column array
    dba = np.arange(10., dtype=float).reshape(-1, 1)
    clts = [dba + i for i in range(3)]
    print(avstd(dba, True))
    print(density(dba))
    print(silhouettes(clts))
    print(quants:=quantifiers(clts))
    print(hydas(clts, stats=quants))



# Launch test/example functions
test_avstd()

test_silhouettes()

test_density()

test_quantifiers()

test_hydas()

test_get_quant_func()

test_quants_1d()

