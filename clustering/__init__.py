""" Time-series clustering toolbox """

# Package version & author
__version__ = '1.0'
__author__ = "Dylan MOLINIE"

from importlib.util import find_spec as _find_spec

# Modules that require third-part libraries (listed as
# "dependencies" in the '.toml' installation file)
__submodules__ = {
    'tools',        # shelve
    'metrics',      # numpy
    'formats',      # numpy
    'datasets',     # numpy
    'ecd',          # numpy
    'kmeans',       # numpy
    'ksom',         # numpy
    'sprada',       # numpy
    'cluster'       # numpy
}

# Modules that require additional third-part libraries (listed as
# optional dependencies ("extras") in the '.toml' installation file)
__opt_submodules__ = {
    'region_growing': ['sklearn']
}

# For any optional submodule, check if its dependencies are installed;
# if so, add this submodule to the list of the available submodules
for _module in __opt_submodules__.items():
    if all(_find_spec(_mod) is not None for _mod in _module[1]):
        __submodules__.add(_module[0])

# Add the installed submodules to the `__all__` set
__all__ = list(__submodules__)


def __getattr__(attr):
    """ Return the correct module from its name, if it exists """
    # pylint: disable=C0415, R0402, R0911
    if attr == 'tools':
        import clustering.tools as tools
        return tools
    if attr == 'metrics':
        import clustering.metrics as metrics
        return metrics
    if attr == 'formats':
        import clustering.formats as formats
        return formats
    if attr == 'datasets':
        import clustering.datasets as datasets
        return datasets
    if attr == 'ecd':
        import clustering.ecd as ecd
        return ecd
    if attr == 'kmeans':
        import clustering.kmeans as kmeans
        return kmeans
    if attr == 'ksom':
        import clustering.ksom as ksom
        return ksom
    if attr == 'region_growing':
        import clustering.region_growing as region_growing
        return region_growing
    if attr == 'sprada':
        import clustering.sprada as sprada
        return sprada
    if attr == 'cluster':
        import clustering.cluster as cluster
        return cluster
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

def __dir__():
    """ Add the modules of `submodules` to the list of callable variables"""
    public_symbols = globals().keys() | __submodules__
    return list(public_symbols)
