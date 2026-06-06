""" Tools to export clusters

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: April 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = [
    'similarity', 'save_pats', 'save_sim', 'save_dist',
    'save_patterns', 'save_clusters', 'save_statistics']

import csv
import numpy as np

import clustering.tools as tls
import clustering.metrics as mts

FILESTYLE = {'newline': '', 'encoding': 'utf-8'}


##############################################################################
##                          Build and Train Tools                           ##
##############################################################################

#-----------------   Similarity Between Two Scalar Values   -----------------#
def similarity(patterns, eps=0.1):
    """ Compute the similarities of the features of two vectors

    For a list of vectors, compare the features of each vector couple.
    If they are seen as close, the Boolean True is given to this set;
    on the contrary, it is set to False.
    In there, two scalars are considered as close if they do not differ
    from one another more than a certain percentage (`eps` argument).
    Consider two scalars a and b and let be m = (1-eps) and M = (1+eps);
    a and b are seen as close if (m*a < b < M*a) and (m*b < a < M*b).

    Parameters
    ----------
    patterns : 2D np.ndarray
        List of the patterns to compare.
    [OPT] eps : float
        Maximum percentage a scalar can differ from another to be
        considered as close.
            :Default: 0.1 (10 %)

    Returns
    -------
    sim : np.ndarray
        The similarities (as Booleans) between each feature of each
        possible couple of vectors. It is a matrix:
          - Nb of cols: same as the compared vectors (`patterns`);
          - Nb of rows: N*(N-1) / 2, where N is the nb of vectors.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(2, 2))

    # Compute the similarity between the patterns of the clusters
    >>> patterns = [clt.pattern for clt in clusters]
    >>> print(similarity(patterns, 0.5))
    [[False False False False False]
     [False False False False False]
     [False False False False False]
     [False False False False False]
     [ True  True  True  True  True]
     [False False False False False]]
    """

    # Normalize the vectors
    patterns, limits = mts.rescale(patterns, (0., 1.))

    mini = 1. - eps
    maxi = 1. + eps

    # Instantiate the matrix to be returned later on
    shape = np.shape(patterns)
    n_sim = int(shape[0] * (shape[0] - 1) / 2)    # N*(N-1) / 2
    sim = np.full((n_sim, shape[1]), False, bool)

    # Compute the similarities
    pos = 0
    for i, veci in enumerate(patterns, 1):
        for vecj in patterns[i:]:
            sim1 = np.logical_and(mini * veci < vecj, vecj < maxi * veci)
            sim2 = np.logical_and(mini * vecj < veci, veci < maxi * vecj)

            sim[pos] = np.logical_and(sim1, sim2)
            pos += 1

    sim = np.empty((n_sim, shape[1]), bool)

    # Compute the similarities
    pos, pos2 = 0, 0
    for i, veci in enumerate(patterns[:-1], 1):
        vecj = patterns[i:]
        sim1 = np.logical_and(mini * veci < vecj, vecj < maxi * veci)
        sim2 = np.logical_and(mini * vecj < veci, veci < maxi * vecj)

        pos2 = pos + len(vecj)
        sim[pos:pos2] = np.logical_and(sim1, sim2)
        pos = pos2

    # Denormalize the vectors
    patterns = mts.rescale(patterns, limits, (0., 1.))

    return sim
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                               Save results                               ##
##############################################################################

#----------------------------   Save Patterns   -----------------------------#
def save_pats(patterns, tags, folder=''):
    """ Save the patterns and the tags in a CSV file

    More generally, save a vector (tags) as the first row of the file,
    and save each row of a matrix (patterns) as a new row of the file.

    Parameters
    ----------
    patterns : 2D np.ndarray
        The matrix to save, row after row.
    tags : 1D list or 1D np.ndarray)
        The vector to save, as the first row of the file.
    [OPT] folder : str
        Folder where to save the above data. They will be saved as
        a CSV file named "patterns.csv" within this folder.
            :Default: '' (current working directory).

    Returns
    -------
    None : directly save the patterns in a CSV file.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(2, 2))

    # Save the patterns in a CSV file
    >>> patterns = [clt.pattern for clt in clusters]
    >>> save_pats(patterns, ['C1', 'C2', 'C3', 'C4', 'C5'])
    """

    # Check folder and name -- Be careful with succession of folders
    folder = tls.check_folder(folder)

    # Create (or replace) the CSV file and save the data within
    with open(folder+"Patterns.csv", 'w', **FILESTYLE) as file:
        writer = csv.writer(file)
        writer.writerow(tags)
        for pat in patterns:
            writer.writerow(pat)
#----------------------------------------------------------------------------#

#------------------   Save Similarities Between Patterns   ------------------#
def save_sim(patterns, tags, eps=0.1, folder=''):
    """ Compare the features of any couple of vectors and save them as CSV

    For a list of vectors, compare the features of each vector couple.
    If they are seen as close, the boolean True is given to this set;
    on the contrary, it is set to False.
    In there, two scalars are considered as close if they do not differ
    from one another more than a certain percentage (`eps` argument).

    Parameters
    ----------
    patterns : 2D np.ndarray
        List of the patterns to compare.
    tags : 1D list
        Names of the features / columns.
    [OPT] eps : float
        Maximum percentage a scalar can differ from another to be
        considered as close.
            :Default: 0.1 (10 %)
    [OPT] folder : str
        Folder where to save the above data. They will be saved as
        a CSV file named "pat_comp.csv" within this folder.
            :Default: '' (current working directory).

    Returns
    -------
    None : directly save the similarity scores in a CSV file.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(2, 2))

    # Compute & save the similarity between the patterns of the clusters
    >>> patterns = np.array([clt.pattern for clt in clusters])
    >>> save_sim(patterns, ['C1', 'C2', 'C3', 'C4', 'C5'])
    """

    # Check folder and name -- Be careful with succession of folders
    folder = tls.check_folder(folder)

    # Features' similarities
    sim = similarity(patterns, eps).tolist()

    # Create (or replace) the CSV file and save the data within
    with open(folder+"Pat_comp.csv", 'w', **FILESTYLE) as file:
        writer = csv.writer(file)
        pats = patterns.tolist()

        # Save each couple of patterns (clusters)
        pos = 0
        for i, pati in enumerate(pats):
            for j, patj in enumerate(pats[i+1:], i+1):
                writer.writerow([None] + tags)      # Tags
                writer.writerow([i] + pati)         # Pattern 1
                writer.writerow([j] + patj)         # Pattern 2
                writer.writerow(['Sim'] + sim[pos]) # Similarity boolean
                writer.writerow([])                 # Empty line
                pos += 1                            # Next couple of patterns
#----------------------------------------------------------------------------#

#-------------------   Save Distances Between Patterns   --------------------#
def save_dist(patterns, distance='euclidean', folder=''):
    """ Compute and save in a CSV file the distance between any
        couple of patterns

    Parameters
    ----------
    patterns : 2D np.ndarray
        List of the patterns for which to compute the distance.
    [OPT] distance : str
        The distance name; see the `get_dist_func` function from the
        `metrics` module for details.
            :Default: `euclidean`
    [OPT] folder : str
        Folder where to save the above data. They will be saved as
        a CSV file named "pat_comp.csv" within this folder.
            :Default: '' (current working directory).

    Returns
    -------
    None : directly save the distances between clusters in a CSV file.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(2, 2))

    # Compute & save the similarity between the patterns of the clusters
    >>> patterns = np.array([clt.pattern for clt in clusters])
    >>> save_dist(patterns, 'manhattan')
    """

    # Check folder and name -- Be careful with succession of folders
    folder = tls.check_folder(folder)

    # Get the distance function
    fdist = mts.get_dist_func(distance)

    # Compute the distances
    nb_clusters = len(patterns)
    distance = np.zeros((nb_clusters, nb_clusters), float)
    for i, pat in enumerate(patterns[:-1]):
        distance[i, i+1:] = np.around(fdist(pat, patterns[i+1:], 1), 3)

    # Create (or replace) the CSV file and save the patterns within
    with open(folder+"pat_comp.csv", 'w', **FILESTYLE) as file:
        writer = csv.writer(file)
        writer.writerow(['Cluster #'] + np.arange(0, nb_clusters).tolist())
        for i, dist in enumerate(distance.tolist()):
            writer.writerow([i] + dist)
#----------------------------------------------------------------------------#

#-------------------   Save Patterns and Related Stats   --------------------#
def save_patterns(clusters, eps=0.1, distance='euclidean', folder=''):
    """ Save the patterns of the objects and some related stats

    The so-called related stats are the euclidean distances between the
    patterns, and the similarities between the features of any possible
    couple of patterns. They are saved as CSV files, one for each.

    Parameters
    ----------
    clusters : list of Cluster class objects
        List of Clusters to be saved.
    [OPT] eps : float
        Maximum percentage a scalar can differ from another to be
        considered as close.
            :Default: 0.1 (10 %)
    [OPT] distance : str
        The distance name; see the `get_dist_func` function from the
        `metrics` module for details.
            :Default: `euclidean`
    [OPT] folder : str
        Folder where to save the above data. They will be saved as
        a CSV file named "pat_comp.csv" within this folder.
            :Default: '' (current working directory).

    Returns
    -------
    None : directly save the results (clusters' patterns, similarity
        scores and distances between clusters) in a CSV file.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(
    ...     np.arange(100.).reshape(-1, 2), np.arange(50), ['C1', 'C2'])
    >>> clusters = ksom_cluster(database, grid_size=(2, 2))
    >>> clusters = [clt for clt in clusters if len(clt) != 0]

    # Save the patterns and some stats about them in a file
    >>> save_patterns(clusters)
    """

    # Check folder and name
    folder = tls.check_folder(folder)

    pats = np.zeros((len(clusters), len(clusters[0].tags)), float)
    for i, clust in enumerate(clusters):
        pats[i] = np.around(clust.pattern, 3)

    # Save patterns
    save_pats(pats, clusters[0].tags, folder)

    # Save similarities between patterns
    save_sim(pats, clusters[0].tags, eps, folder)

    # Save distances between patterns
    save_dist(pats, distance, folder)
#----------------------------------------------------------------------------#

#----------------------------   Save Clusters   -----------------------------#
def save_clusters(clusters, folder=''):
    """ Save a list of clusters as unique CSV files

    For a list of clusters (Cluster class), save their tags (features
    labels), their patterns and the data (np.ndarray) contained within.

    Parameters
    ----------
    clusters : list of Cluster class objects
        List of Clusters to be saved.
    [OPT] folder : str
        Folder where to save the above data. They will be saved as
        a CSV file named "pat_comp.csv" within this folder.
            :Default: '' (current working directory).

    Returns
    -------
    None : directly save the clusters in a CSV file.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(2, 2))

    # Save the clusters in a file
    >>> save_clusters(clusters)
    """

    # Check folder and name -- Be careful with succession of folders
    folder = tls.check_folder(folder)

    # Save any cluster (node) in a new file
    folder += "cluster_"
    for i, clust in enumerate(clusters):

        # Create (or replace) the CSV file and save the node within
        with open(folder+f"{i}.csv", 'w', **FILESTYLE) as file:
            writer = csv.writer(file)

            writer.writerow(['Time'] + clusters[0].tags)
            writer.writerow(['Pattern']
                            + np.around(clust.pattern, 3).tolist())

            for idx, val in zip(clust.index, clust.value):
                writer.writerow(
                    [int(idx)] + np.around(val, 3).astype(int).tolist())
#----------------------------------------------------------------------------#

#----------------------   Save Cluster's Statistics   -----------------------#
def save_statistics(names, results, folder=''):
    """ Write some statistics of the clusters in a CSV file

    The results to save must be computed elsewhere and be gathered in a
    2D np.ndarray. An empty cluster should have a corresponding row full
    of `-1.`. The results are saved in a separate file, with name
    "filename + _exp.csv". For each matrix's column, some statistics
    (min, max, mean, median, 1st and 3rd quantiles) are computed and
    saved in a dedicated file, with name "filename + _stats.csv".

    Parameters
    ----------
    names : list of str
        Names of the features so save.
    results : np.ndarray
        Results to save.
    [OPT] folder : str
        Folder where to save the above data. They will be saved as
        a CSV file named "pat_comp.csv" within this folder.
            :Default: '' (current working directory).

    Returns
    -------
    None : directly write the results in a CSV file.

    Examples
    --------
    >>> import numpy as np
    >>> from clustering.formats import Database
    >>> from clustering.ksom import ksom_cluster

    # Generate a dummy database & cluster it
    >>> database = Database(np.arange(100.).reshape(-1, 5), np.arange(20))
    >>> clusters = ksom_cluster(database, grid_size=(2, 2))

    # Compute some characterizing metrics about the clusters
    >>> quants = mts.quantifiers(clusters)

    # Save the clusters in a file
    >>> save_statistics(['Sils', 'HyDen', 'AvStd'], quants)
    """

    # Check folder and name -- Be careful with succession of folders
    folder = tls.check_folder(folder)

    # Save the results for each cluster
    with open(folder+"Results.csv", 'w', **FILESTYLE) as file:
        writer = csv.writer(file)
        writer.writerow(names)
        empty = ['None' for i in range(len(names)-1)]

        # Results for each cluster
        for i, res in enumerate(results):
            if res[0] is None:   # Empty cluster
                writer.writerow([f'clt{i}'] + empty)
            else:
                writer.writerow([f'clt{i}'] + res.tolist())

    # Save some statistics of the results
    with open(folder+"Results_stats.csv", 'w', **FILESTYLE) as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerow(['Min'] + np.min(results, 0).tolist())
        writer.writerow(['Max'] + np.max(results, 0).tolist())
        writer.writerow(['Q1'] +\
            np.around(np.quantile(results, 0.25, 0), 2).tolist())
        writer.writerow(['Med'] +\
            np.around(np.quantile(results, 0.50, 0), 2).tolist())
        writer.writerow(['Q3'] +\
            np.around(np.quantile(results, 0.75, 0), 2).tolist())
        writer.writerow(['Mean'] +\
            np.around(np.mean(results, 0), 2).tolist())
#----------------------------------------------------------------------------#

##############################################################################
