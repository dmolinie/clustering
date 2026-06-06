import numpy as np
import matplotlib.pyplot as plt
from clustering.ecd._ecds import *
FIGSIZE = (19.20, 10.80)


##############################################################################
##                         Plot the CDFs & KS Test                          ##
##############################################################################

#------------------------------   ECDs/CDFs   -------------------------------#
def disp_cdfs(vals, cdfs, database=False, fname="KSTest_CDFs.pdf", show=False):
    """ Plot the Empirical CDFs against their increasing values

    Instantiate a figure with D plots, with `D` the last dimension of
    `cdfs` (i.e. the dimension of the space). Take the set of matrices
    `cdfs` and the set of arrays `vals`, and for any last dimension of
    both, iterate `cdfs` and plot `cdfs[i, :, d]` against `vals[:, d]`
    on the d-th graph. If `database` is True, assume the last row of
    `cdfs` is that of the database and plot it as a dashed black line.
    Finally, save the figure in a file with name `fname` and show it
    if `show` is True, or close it else.

    Parameters
    ----------
    vals : 1D np.ndarray
        The increasing threshold values of the CDFs.
    cdfs : 3D np.ndarray
        The Cumulative Distributions.
    [OPT] database : bool
        If the database's CDFs were appended to the set of CDFs
        (last row/matrix).
            :Default: False
    [OPT] fname : str
        The name of the file where to save the figure.
            :Default: "KSTest_CDFs.pdf"
    [OPT] show : bool
        Show the figure, or close it else. The figure is saved in a
        file in any case.
            :Default: False

    Other Parameters
    ----------------
    [GLB] FIGSIZE : 2-tuple of ints
        The size of the figure, in pixels / 100.

    Returns
    -------
    None : directly plot the figure and save it in a file.
    """
    plt.rcParams.update({'font.size': 18})

    # Instantiate a figure of D plots, with D the dimension of the space
    fig, axs = plt.subplots(1, cdfs.shape[-1], figsize=FIGSIZE, sharey=True)

    # If the last row is for the database, extract it
    if database:
        cdf_dba = cdfs[-1]
        cdfs = cdfs[:-1]

    for i, axe in enumerate(axs):

        # Plot the database's CDFs
        if database:
            axe.plot(vals[:, i], cdf_dba[:, i],
                     linewidth=3, color='k', linestyle='--')

        # Plot the sensors' CDFs
        for cdf in cdfs:
            axe.plot(vals[:, i], cdf[:, i], linewidth=3)

        # In a database, there is often one dimension per sensor
#        axe.set_title(f"Sensor {i}", size=19)
        axe.set_title(f"Dimension {i}", size=19)

    # Add labels
    fig.supxlabel("Values [normalized]", size=19)
    fig.supylabel("KS Test [probability]", size=19)

    # Save the figure in a file
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')

    # Show the figure if `show`, or close it else
    plt.show() if show else plt.close()
#----------------------------------------------------------------------------#

#-------------------------------   KS Test   --------------------------------#
def disp_kss(vals, cdfs, database=False, fname="KSTest.pdf", show=False):
    """ Plot the KS Test

    Instantiate a 1-plot figure. Take the set of matrices `cdfs` and
    the set of arrays `vals`, and compute their respective means along
    the last dimension (i.e. that of the space), denoted as `mu_cdfs`
    and `mu_vals`; then, plot any array of `mu_cdfs` (one per cluster)
    against `mu_vals`. If `database` is True, assume the last row of
    `cdfs` is that of the database and plot it as a dashed black line.
    Finally, save the figure in a file with name `fname` and show it
    if `show` is True, or close it else.

    Parameters
    ----------
    vals : 1D np.ndarray
        The increasing threshold values of the CDFs.
    cdfs : 3D np.ndarray
        The Cumulative Distributions.
    [OPT] database : bool
        If the database's CDFs were appended to the set of CDFs
        (last row/matrix).
            :Default: False
    [OPT] fname : str
        The name of the file where to save the figure.
            :Default: "KSTest_CDFs.pdf"
    [OPT] show : bool
        Show the figure, or close it else. The figure is saved in a
        file in any case.
            :Default: False

    Other Parameters
    ----------------
    [GLB] FIGSIZE : 2-tuple of ints
        The size of the figure, in pixels / 100.

    Returns
    -------
    None : directly plot the figure and save it in a file.
    """
    plt.rcParams.update({'font.size': 28})

    # Instantiate a figure of D plots, with D the dimension of the space
    plt.figure(figsize=FIGSIZE)

    # Compute the means of the CDFs along the last dimension (space dim.)
    means = cdfs.mean(-1)
    vals = vals.mean(-1)

    # Plot the database's KS Test
    if database:
        plt.plot(vals, means[-1], linewidth=3, color='k', linestyle='--')
        # If the last row is for the database, extract it
        means = means[:-1]

    # Plot the sensors' KS Test
    for meani in means:
        plt.plot(vals, meani, linewidth=3)

    # Add labels
    plt.xlabel("Values [normalized]", size=36)
    plt.ylabel("KS Test [probability]", size=36)
    #ax.set_title(f"Sensor")

    # Save the figure in a file
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')

    # Show the figure if `show`, or close it else
    plt.show() if show else plt.close()
#----------------------------------------------------------------------------#

##############################################################################



# Generate a dummy list of clusters
arr = np.arange(50., dtype=float).reshape(-1, 5)
arrs = [arr+i for i in range(4)]

#--- Compute the ECDs without the database
vals, cdfs = cumulative_distributions(arrs, None, 100)
print(np.shape(vals), np.shape(cdfs))

disp_cdfs(vals, cdfs, False, "CDFs-nodba.pdf", False)
disp_kss(vals, cdfs, False, "KSTest-nodba.pdf", False)

#--- Compute the ECDs with the database
vals, cdfs = cumulative_distributions(arrs, arrs[0], 100)
print(np.shape(vals), np.shape(cdfs))

disp_cdfs(vals, cdfs, True, "CDFs-dba.pdf", False)
disp_kss(vals, cdfs, True, "KSTest-dba.pdf", False)

