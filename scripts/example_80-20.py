""" Example illustrating the 80% Training / 20% Testing case

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: July 2024
Last revised: June 2026

License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt

# Example datasets
from clustering import datasets as sets

# Kohonen's Self-Organizing Maps
from clustering.ksom import KohonenSOM

# Regular Pyplot colors
TABCOLORS = (
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')

# Standard variant of the TABCOLORS
COLORS = (
    'blue', 'orange', 'green', 'red', 'purple',
    'brown', 'pink', 'gray', 'olive', 'cyan')


#--------------------   Build Training & Testing Sets   ---------------------#
# Generate a simple 2D dataset
database = sets.gaussian(250, 2, (0, 100), 0.5, 12, seed=10)

# Generate a semi-random number generator
rng = np.random.default_rng(seed=0)
mask = np.full(len(database.index), False, dtype=bool)

# Generate a set of training indexes (80% of the database)
idx_trn = rng.choice(len(database), int(0.8*len(database)), replace=False)
mask[idx_trn] = True

# Separate set of indexes into training and testing indexes
ind_trn = database.index[mask]
ind_gen = database.index[~mask]

dba_trn = database.select(ind_trn)
dba_gen = database.select(ind_gen)
#----------------------------------------------------------------------------#

#----------------------   Train the KSOM & Use Them   -----------------------#
# Instantiate the Kohonen's SOM
ksom = KohonenSOM(grid_size=(3, 3), seed=0)

# Train the KSOM
ksom.fit(dba_trn, verbose=False)

## Build the clusters by building them
## (this variant does not order the clusters)
#clt_trn = ksom.build(dba_trn, cluster=True, grid=False)
#clt_gen = ksom.build(dba_gen, cluster=True, grid=False)

# Build the clusters by finding their closest nodes in the KSOM
# (this variant allows to order the clusters)
cats_trn = ksom.winner(dba_trn)     # Training dataset
cats_gen = ksom.winner(dba_gen)     # Testing dataset
#----------------------------------------------------------------------------#

#--------------------------   Display Clustering   --------------------------#
fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')

for i in range(max(cats_trn)):
    # Clusters build on the training dataset
    clust = dba_trn[cats_trn == i]
    plt.plot(clust[:, 0], clust[:, 1], '+', color=TABCOLORS[i], markersize=12)
    # Clusters build on the testing dataset
    clust = dba_gen[cats_gen == i]
    plt.plot(clust[:, 0], clust[:, 1], '*', color=COLORS[i], markersize=8)

plt.xlabel("Dimension 1 [a.u.]", size=18)
plt.ylabel("Dimension 2 [a.u.]", size=18)
plt.title("Clustering with 80% data for training and 20% for testing", size=20)
plt.legend(["Training data", "Testing data"], fontsize=14)

plt.show()
#----------------------------------------------------------------------------#
