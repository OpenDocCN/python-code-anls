# `D:\src\scipysrc\scipy\scipy\ndimage\__init__.py`

```
"""
=========================================================
Multidimensional image processing (:mod:`scipy.ndimage`)
=========================================================

.. currentmodule:: scipy.ndimage

This package contains various functions for multidimensional image
processing.


Filters
=======

.. autosummary::
   :toctree: generated/

   convolve - Multidimensional convolution
   convolve1d - 1-D convolution along the given axis
   correlate - Multidimensional correlation
   correlate1d - 1-D correlation along the given axis
   gaussian_filter
   gaussian_filter1d
   gaussian_gradient_magnitude
   gaussian_laplace
   generic_filter - Multidimensional filter using a given function
   generic_filter1d - 1-D generic filter along the given axis
   generic_gradient_magnitude
   generic_laplace
   laplace - N-D Laplace filter based on approximate second derivatives
   maximum_filter
   maximum_filter1d
   median_filter - Calculates a multidimensional median filter
   minimum_filter
   minimum_filter1d
   percentile_filter - Calculates a multidimensional percentile filter
   prewitt
   rank_filter - Calculates a multidimensional rank filter
   sobel
   uniform_filter - Multidimensional uniform filter
   uniform_filter1d - 1-D uniform filter along the given axis

Fourier filters
===============

.. autosummary::
   :toctree: generated/

   fourier_ellipsoid
   fourier_gaussian
   fourier_shift
   fourier_uniform

Interpolation
=============

.. autosummary::
   :toctree: generated/

   affine_transform - Apply an affine transformation
   geometric_transform - Apply an arbitrary geometric transform
   map_coordinates - Map input array to new coordinates by interpolation
   rotate - Rotate an array
   shift - Shift an array
   spline_filter
   spline_filter1d
   zoom - Zoom an array

Measurements
============

.. autosummary::
   :toctree: generated/

   center_of_mass - The center of mass of the values of an array at labels
   extrema - Min's and max's of an array at labels, with their positions
   find_objects - Find objects in a labeled array
   histogram - Histogram of the values of an array, optionally at labels
   label - Label features in an array
   labeled_comprehension
   maximum
   maximum_position
   mean - Mean of the values of an array at labels
   median
   minimum
   minimum_position
   standard_deviation - Standard deviation of an N-D image array
   sum_labels - Sum of the values of the array
   value_indices - Find indices of each distinct value in given array
   variance - Variance of the values of an N-D image array
   watershed_ift

Morphology
==========

"""
# 导入模块中定义的所有非私有函数和变量名
from ._filters import *
from ._fourier import *
from ._interpolation import *
from ._measurements import *
from ._morphology import *

# 弃用的命名空间，在 v2.0.0 版本中将移除
from . import filters
from . import fourier
from . import interpolation
from . import measurements
from . import morphology

# 将模块中所有不以 '_' 开头的名称添加到 __all__ 列表中，以便导入时被引用
__all__ = [s for s in dir() if not s.startswith('_')]

# 导入用于测试的 PytestTester 类，并将其命名为 test，然后删除 PytestTester 类本身
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
```