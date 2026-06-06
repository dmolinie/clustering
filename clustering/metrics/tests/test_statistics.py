import numpy as np
from clustering.metrics._statistics import *


def test_mean():
    """ Mean of a dataset """
    print(mean(np.arange(50., dtype=float).reshape(-1, 5)))

def test_std():
    """ Standard Deviation of a dataset """
    print(std(np.arange(50., dtype=float).reshape(-1, 5)))

def test_mean_std():
    """ Mean and Standard Deviation of a dataset """
    print(mean_std(np.arange(50., dtype=float).reshape(-1, 5)))

def test_max_span():
    """ Maximal span of a cluster """
    # Generate dummy data
    data = np.arange(50., dtype=float).reshape(-1, 5)
    # Compute the database span
    print(max_span(data, 'euclidean'))

def test_max_to_mean():
    """ Maximal distance to the mean of a cluster """
    # Generate dummy data
    data = np.arange(50., dtype=float).reshape(-1, 5)
    # Compute the database span
    print(max_to_mean(data, 'euclidean'))

def test_sphere_span():
    """ Minimal spheroid covering the data of a cluster """
    # Generate dummy noisy data
    data = np.arange(50., dtype=float) + 10.*np.random.random(50)
    data = data.reshape(-1, 5)
    # Compute the spheroid span
    print(sphere_span(data, 'euclidean', False))     # Fast estimate
    print(sphere_span(data, 'euclidean', True))      # True span

def test_get_span_func():
    """ Get the reference to a specified span estimation function """
    # Generate dummy data
    data = np.arange(50., dtype=float).reshape(-1, 5)
    # Get the spanning function
    fspan = get_span_func('sphere_span')
    # Compute the distances
    print(fspan(data))

def test_intra_distances():
    """ Compute the intra-cluster distances """
    # Generate a dummy cluster of data
    cluster = np.arange(15., dtype=float).reshape(-1, 3)
    # Compute the intra-cluster distances
    dists = intra_distances(cluster, symmetric=True, distance='euclidean')
    print(dists.round(2))

def test_inter_distances():
    """ Compute the inter-cluster distances """
    # Generate a dummy set of clusters
    clusters = [
        np.linspace(0., 1., 15).reshape(-1, 3),
        np.linspace(0., 2., 18).reshape(-1, 3),
        np.linspace(2., 5., 21).reshape(-1, 3)]
    # Compute the intra-cluster distances
    dists = inter_distances(clusters, symmetric=True, distance='euclidean')
    # Compute the maximal inter-cluster distances
    max_dists = np.array(
        [np.max(d) for dist in dists for d in dist if len(d) > 0])
    print(max_dists.round(2))
    
def test_jaccard_index():
    """ Jaccard Index between two datasets """
    # Generate dummy data
    data1 = np.arange(10., 50., dtype=float)
    data2 = np.arange(20., 90., dtype=float)
    # Compute the Jaccard Index
    print(jaccard_index(data1, data2))

def test_sorensen_index():
    """ Sorensen-Dice Index (Czekanowski) between two datasets """
    # Generate dummy data
    data1 = np.arange(10., 50., dtype=float)
    data2 = np.arange(15., 90., dtype=float)
    # Compute the Sorensen-Dice Index
    sorensen_index(data1, data2)

def test_dunn_index():
    """ Dunn Index of a set of datasets """
    # Generate a dummy set of clusters
    clusters = [
        np.linspace(0., 1., 15).reshape(-1, 3),
        np.linspace(0., 2., 18).reshape(-1, 3),
        np.linspace(2., 5., 21).reshape(-1, 3)]
    # Compute the Dunn Index
    print(dunn_index(clusters))

def test_bouldin_davies_index():
    """ Bouldin-Davies Index of a set of datasets """
    # Generate a dummy set of clusters
    clusters = [
        np.linspace(0., 1., 15).reshape(-1, 3),
        np.linspace(0., 2., 18).reshape(-1, 3),
        np.linspace(2., 5., 21).reshape(-1, 3)]
    # Bouldin-Davies Index with 'min' statistics
    print(bouldin_davies_index(clusters, 'min'))
    # Bouldin-Davies Index with 'mean' statistics
    print(bouldin_davies_index(clusters, 'mean'))
    # Bouldin-Davies Index with 'max' statistics
    print(bouldin_davies_index(clusters, 'max'))

def test_get_stat_func():
    """ Get the reference to a specified statistic function """
    # Generate dummy data
    data = np.arange(15., dtype=float).reshape(-1, 5)
    # Compute the 1st quantile (25%) of the dataset
    fstat = get_stat_func('quantile')
    quant = fstat(data, 0.25)
    print(quant)
    # Compute the Sorensen index between two datasets (must be 1D)
    data = data.ravel()
    fstat = get_stat_func('sorensen_index')
    index = fstat(data, data+3.)
    print(index)

def test_stats_1d():
    """ Check 1D data """
    #--- Test 1D row array
    arr = np.arange(50., dtype=float)
    print(max_span(arr))
    print(max_to_mean(arr))
    print(sphere_span(arr))
    print(jaccard_index(arr, arr+1.))
    print(sorensen_index(arr, arr+1.))
    clusters = [arr, arr+1., arr+2.]
    print(intra_distances(clusters[0]))
    print(inter_distances(clusters)[0][1])
    print(dunn_index(clusters))
    print(bouldin_davies_index(clusters))
    #--- Test 1D column array
    arr = np.arange(50., dtype=float).reshape(-1, 1)
    print(max_span(arr))
    print(max_to_mean(arr))
    print(sphere_span(arr))
    print(jaccard_index(arr, arr+1.))
    print(sorensen_index(arr, arr+1.))
    clusters = [arr, arr+1., arr+2.]
    print(intra_distances(clusters[0]))
    print(inter_distances(clusters)[0][1])
    print(dunn_index(clusters))
    print(bouldin_davies_index(clusters))



# Launch test/example functions
test_mean()

test_std()

test_mean_std()

test_max_span()

test_max_to_mean()

test_sphere_span()

test_get_span_func()

test_intra_distances()

test_inter_distances()

test_jaccard_index()

test_sorensen_index()

test_dunn_index()

test_bouldin_davies_index()

test_get_stat_func()

test_stats_1d()

