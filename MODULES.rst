Clustering package's modules, functions & classes
+++++++++++++++++++++++++++++++++++++++++++++++++

This file aims to summarize all the objects implemented in the ``clustering`` package; to this purpose, it briefly introduces the modules of the package, as well as their respective functions and classes.

..
  Contents::


cluster
=======
Tools to ease clustering and refine the obtained groups; also provide functions to save results (CSV export).

..
  _cluster.py

* ``prune``  
        Remove the empty clusters from a list    

* ``merge``  
        Merge the closest clusters to refine splitting    

* ``sort``  
        Sort the clusters of a list by size    

* ``get_clust_func``  
        Get the reference to a specified clustering function

* ``cluster``  
        Split a a database using a given clustering method

* ``rebuild_idx``  
        Rebuild the training and testing indexes

..
  _save_clt.py

* ``similarity``  
        Compute the similarities of the features of two vectors    

* ``save_pats``  
        Save the patterns and the tags in a CSV file    

* ``save_sim``  
        Compare the features of any couple of vectors and save them as CSV    

* ``save_dist``  
        Compute and save in a CSV file the distance between any couple of patterns    

* ``save_patterns``  
        Save the patterns of the Node objects and some related stats    

* ``save_clusters``  
        Save a list of clusters as unique CSV files    

* ``save_statistics``  
        Write some statistics of the clusters in a CSV file    


datasets
========
Simple, academic datasets for testing purpose. These datasets are NumPy-based, and wrapped into the standard ``Database`` format of the project.

..
  _datasets.py

* ``nested_circles``  
        Generate two 2D nested circles    

* ``strips``  
        Generate three 2D vertical strips    

* ``gaussian``  
        Generate a database with a Gaussian distribution    

* ``gauss_3d``  
        Generate a set of 3D Gaussian distributions    


ecd
===
Provide the ECD Test, a split-and-merge method to estimate the number of regions in a feature space, by splitting a dataset into many pieces, characterizing them with their Empirical Cumulative Distributions (ECDs) and regrouping the clusters by closeness between their respective ECDs using the Modified Hausdorff Distance (MHD).

..
  _ecds.py

* ``cumulative_distributions``  
        Empirical Cumulative Distribution Functions (ECDs)

* ``cdfs_matrix``  
        Build the matrix of the MHDs between the ECDs

* ``cdfs_states``  
        Build the closeness states matrix

* ``cdfs_groups``  
        Build the closeness map

* ``ks_test``  
        Kolmogorov-Smirnov (KS) Test

..
  _ecd_test.py

* ``ECDTest``  
        Empirical Cumulative Distribution (ECD) Test

* ``ecd_test``  
        Estimate the number of regions in a feature space


formats
=======
Wrappers for data and clusters; in particular, the features of any database is a set of data (y-values), their (time)stamps (x-values), the name of the columns and some contextual information. In addition, a cluster is a database with a characterizing pattern (its prototype). The ``Database`` and ``Cluster`` classes are conceived to represent ND time-series, in which the columns represent the sensors, and the rows, the samples over time. Finally, provide a wrapper to represent a Neural Grid in the sense of the Kohonen's SOMs (cf. module ``ksom``).

..
  _database.py

* ``Database``  
        Class to represent a database (values & indexes [& tags & classes])

* ``Cluster``  
        Class to represent a cluster (values & indexes & pattern [& tags & classes])

..
  _neuralgrid.py

* ``NeuralGrid``  
        Class to represent a neural grid for the Kohonen Self-Organizing Maps

..
  _checkers.py

* ``check_data``  
        Check data format

* ``get_index``  
        Get the index of a database

* ``get_tags``  
        Get the tags of a database

* ``get_classes``  
        Get the classes of a database


kmeans
======
Implement the regular and kernel-variant K-Means algorithm, and provide classes to use them as well as standalone functions to use them directly.

..
  _kmeans.py

* ``KMeans``  
        Class that provides the Lloyd's KMeans clustering algorithm

* ``kmeans_params``  
        Check the parameters for the KMeans

* ``kmeans_cluster``  
        Instantiate & train the K-Means on a dataset and cluster it

..
  _kernel_kmeans.py

* ``KernelKMeans``  
        Class that provides the Kernel KMeans clustering algorithm

* ``kkmeans_params``  
        Check the parameters for the Kernel KMeans

* ``kkmeans_cluster``  
        Instantiate & train the Kernel K-Means on a dataset and cluster it


ksom
====
Implement the Kohonen Self-Organizing Maps (SOMs or KSOMs), that are stochastic grids that learn classes from example. They are often faster, more representative and more general-purpose than the more regular K-Means. Also provide the Bi-Level SOMs, that are an improvement to the SOMs: they train several maps, and average them all using another, final SOM; doing so is expected to improve the representativeness of clustering, and to better identify the main regions of a feature space, from a Data Mining and statistical point of view. For both SOMs and BSOMS, provide a class to use them as well as standalone shorthand functions.

..
  _ksom.py

* ``KohonenSOM``  
        Class that provides the Kohonen's Self-Organizing Maps (KSOMs)

* ``ksom_params``  
        Check the parameters for the Kohonen's SOM

* ``ksom_cluster``  
        Instantiate & train a KSOM on a dataset and cluster it

..
  _bsom.py

* ``BiLevelSOM``  
        Class that provides the Bi-Level Self-Organizing Maps (BSOM)

* ``bsom_params``  
        Check the parameters for the BSOM

* ``bsom_cluster``  
        Instantiate & train a BSOM on a dataset and cluster it

metrics
=======
Provide standard distances, kernels and linkage functions (to compare groups of data). Also provide tools to rescale a dataset, compute statistics on it and metrics to characterize it, in particular the Silhouette Coefficients, the Hyper-Density (HyDen), the Average Standard Deviation (AvStd) and a hybrid mixture of them (HyDAS).

..
  _distances.py

* ``manhattan_1d`` / ``manhattan_nd``  
        Manhattan distance for 1D / ND data

* ``euclidean_1d`` / ``euclidean_nd``  
        Euclidean distance for 1D / ND data

* ``minkowski_1d`` / ``minkowski_nd``  
        Minkowski distance for 1D / ND data

* ``canberra_1d`` / ``canberra_nd``  
        Canberra distance for 1D / ND data

* ``cosine_1d`` / ``cosine_nd``  
        Cosine distance for 1D / ND  data

* ``tanimoto_1d`` / ``tanimoto_nd``  
        Tanimoto distance for 1D / ND  data

* ``czekanowski_1d`` / ``czekanowski_nd``  
        Czekanowski distance for 1D / ND  data

* ``hausdorff``  
        Modified Hausdorff Distance between two curves (2D vectors)

* ``get_dist_func``  
        Get the reference to a specified distance function    

..
  _volumes.py

* ``hypercube``  
        Hypercube's volume and surface

* ``hypersphere``  
        Hypersphere's volume and surface

* ``get_vol_func``  
        Get the reference to a specified hyper-volume function

..
  _kernels.py

* ``linear_1d`` / ``linear_nd``  
        Linear kernel for 1D / ND  data

* ``circles_1d`` / ``circles_nd``  
        Circle kernel for 1D / ND  data

* ``gaussian_1d`` / ``gaussian_nd``  
        Gaussian kernel for 1D / ND  data 

* ``polynomial_1d`` / ``polynomial_nd``  
        Polynomial kernel for 1D / ND  data 

* ``get_ker_func``  
        Get the reference to a specified kernel function    

..
  _linkages.py

* ``linkage_single``  
        Single linkage aggregation metric    

* ``linkage_complete``  
        Complete linkage aggregation metric    

* ``linkage_mean``  
        Mean linkage aggregation metric    

* ``linkage_ward``  
        Ward linkage aggregation metric    

* ``linkages``  
        Linkage distance between any possible 2-cluster set    

* ``get_link_func``  
        Get the reference to a specified linkage function    

..
  _scaling.py

* ``extrema``  
        Minimal and maximal values of a dataset

* ``check_scale``  
        Check if a set of data is scaled between bounds

* ``rescale``  
        Rescale a dataset (e.g. normalize or denormalize it)

..
  _statistics.py

* ``mean``  
        Mean of a dataset

* ``std``  
        Standard Deviation of a dataset

* ``mean_std``  
        Mean and Standard Deviation of a dataset

* ``max_span``  
        Maximal span of a cluster

* ``max_to_mean``  
        Maximal distance to the mean of a cluster

* ``sphere_span``  
        Minimal spheroid covering the data of a cluster

* ``get_span_func``  
        Get the reference to a specified span estimation function

* ``intra_distances``  
        Compute the intra-cluster distances

* ``inter_distances``  
        Compute the inter-cluster distances

* ``jaccard_index``  
        Jaccard Index between two datasets

* ``sorensen_index``  
        Sorensen-Dice Index (Czekanowski) between two datasets

* ``dunn_index``  
        Dunn Index of a set of datasets

* ``bouldin_davies_index``  
        Bouldin-Davies Index of a set of datasets

* ``get_stat_func``  
        Get the reference to a specified statistic function

..
  _quantifiers.py

* ``avstd``  
        Average Standard Deviation

* ``silhouettes``  
        Silhouette Coefficient of a database composed of clusters

* ``density``  
        Compute the density of a cluster

* ``quantifiers``  
        Compute the HyDensity, AvStd and Silhouettes quantifiers

* ``hydas``  
        Hybridized Density-AvStd-Silhouettes-based metric HyDAS

* ``get_quant_func``  
        Get the reference to a specified quantifier function


region_growing
==============
Interface the ``DBSCAN`` and ``OPTICS`` clustering methods from the ``Scikit-Learn`` suit to the format standard of the current project. In particular, this format allows using these classes in the ``Recursive`` and ``SPRADA`` classes from the ``sprada`` module.

..
  _region_growing.py

* ``build``
        Empirical Cumulative Distribution (ECD) Test

* ``dbscan_params``
        Check the parameters for DBSCAN clustering

* ``dbscan_cluster``
        Instantiate & train DBSCAN on a dataset and cluster it

* ``optics_params``
        Check the parameters for OPTICS clustering

* ``optics_cluster``
        Instantiate & train OPTICS on a dataset and cluster it


sprada
======
Provide the ``Recursive`` class that aims to decompose a dataset into many pieces until any group satisfies a given metric (compatness, homogeneity, density, etc.). Also provide the ``SPRADA`` class, that is a high-level clustering algorithm with a split-and-merge approach: it uses the recursive decomposition of the ``Recursive`` class, and regroups the clusters with the ECD Test to rebuild the regions of the feature space.

..
  _recursive.py

* ``Recursive``  
        Recursive Decomposition

* ``recursive_params``  
        Check the parameters for Recursive clustering

* ``recursive_cluster``  
        Instantiate & train a Recursive Tree on a dataset and cluster it

..
  _sprada.py

* ``SPRADA``  
        Self-Parameterized Recursively Assessed Decomposition Algorithm

* ``sprada_params``  
        Check the parameters for SPRADA clustering

* ``sprada_cluster``  
        Instantiate & train SPRADA on a dataset and cluster it


tools
=====
Provide general-purpose tools, in particular for data format or dictionary keys check, and for saving or loading data (I/O streams).

..
  _tools.py

* ``exec_file``  
        Execute a Python script file

* ``save_as_csv``  
        Save a data set in a CSV file 

* ``load_as_csv``  
        Load the data contained in a CSV file

* ``save_ws``  
        Save a workspace as a shelve

* ``load_ws``  
        Restore a shelved workspace

* ``get_dim``  
        Get the depth (number of dimensions) of an array

* ``get_ndim``  
        Get the depth (number of dimensions) of an array

* ``flat_list``  
        Flatten a 2D list

* ``make_iter``  
        Check if `data` is scalar, and wrap it into a tuple if so

* ``check_ext``  
        Check the file name extension

* ``check_folder``  
        Check the validity of folder name

* ``check_path``  
        Check the path of a file (folders + extension)

* ``check_keys``  
        Dictionary check function

