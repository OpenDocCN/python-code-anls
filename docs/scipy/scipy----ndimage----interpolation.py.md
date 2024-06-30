# `D:\src\scipysrc\scipy\scipy\ndimage\interpolation.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

# Importing the _sub_module_deprecation function from scipy._lib.deprecation module
from scipy._lib.deprecation import _sub_module_deprecation

# List of public functions exposed by this module (used for __all__ attribute)
__all__ = [
    'spline_filter1d', 'spline_filter',
    'geometric_transform', 'map_coordinates',
    'affine_transform', 'shift', 'zoom', 'rotate',
]


def __dir__():
    # Special function returning the list of public functions (__all__)
    return __all__


def __getattr__(name):
    # Function invoked when trying to access an attribute not directly defined in this module
    # Generates a deprecation warning using _sub_module_deprecation function
    return _sub_module_deprecation(sub_package='ndimage', module='interpolation',
                                   private_modules=['_interpolation'], all=__all__,
                                   attribute=name)
```