# `D:\src\scipysrc\scipy\scipy\spatial\__init__.py`

```
"""
=============================================================
Spatial algorithms and data structures (:mod:`scipy.spatial`)
=============================================================

.. currentmodule:: scipy.spatial

.. toctree::
   :hidden:

   spatial.distance

Spatial transformations
=======================

These are contained in the `scipy.spatial.transform` submodule.

Nearest-neighbor queries
========================
.. autosummary::
   :toctree: generated/

   KDTree      -- class for efficient nearest-neighbor queries
   cKDTree     -- class for efficient nearest-neighbor queries (faster implementation)
   Rectangle

Distance metrics
================

Distance metrics are contained in the :mod:`scipy.spatial.distance` submodule.

Delaunay triangulation, convex hulls, and Voronoi diagrams
==========================================================

.. autosummary::
   :toctree: generated/

   Delaunay    -- compute Delaunay triangulation of input points
   ConvexHull  -- compute a convex hull for input points
   Voronoi     -- compute a Voronoi diagram hull from input points
   SphericalVoronoi -- compute a Voronoi diagram from input points on the surface of a sphere
   HalfspaceIntersection -- compute the intersection points of input halfspaces

Plotting helpers
================

.. autosummary::
   :toctree: generated/

   delaunay_plot_2d     -- plot 2-D triangulation
   convex_hull_plot_2d  -- plot 2-D convex hull
   voronoi_plot_2d      -- plot 2-D Voronoi diagram

.. seealso:: :ref:`Tutorial <qhulltutorial>`


Simplex representation
======================
The simplices (triangles, tetrahedra, etc.) appearing in the Delaunay
tessellation (N-D simplices), convex hull facets, and Voronoi ridges
(N-1-D simplices) are represented in the following scheme::

    tess = Delaunay(points)
    hull = ConvexHull(points)
    voro = Voronoi(points)

    # coordinates of the jth vertex of the ith simplex
    tess.points[tess.simplices[i, j], :]        # tessellation element
    hull.points[hull.simplices[i, j], :]        # convex hull facet
    voro.vertices[voro.ridge_vertices[i, j], :] # ridge between Voronoi cells

For Delaunay triangulations and convex hulls, the neighborhood
structure of the simplices satisfies the condition:
``tess.neighbors[i,j]`` is the neighboring simplex of the ith
simplex, opposite to the ``j``-vertex. It is -1 in case of no neighbor.

Convex hull facets also define a hyperplane equation::

    (hull.equations[i,:-1] * coord).sum() + hull.equations[i,-1] == 0

Similar hyperplane equations for the Delaunay triangulation correspond
to the convex hull facets on the corresponding N+1-D
paraboloid.

The Delaunay triangulation objects offer a method for locating the
simplex containing a given point, and barycentric coordinate
computations.

Functions
---------

.. autosummary::
   :toctree: generated/

   tsearch
   distance_matrix
   minkowski_distance
   minkowski_distance_p
   procrustes
   geometric_slerp


"""

注释：
# 导入 KD 树相关模块和函数
from ._kdtree import *
# 导入并导出 C KD 树相关模块和函数
from ._ckdtree import *
# 导入并导出 Qhull 函数
from ._qhull import *
# 导入并导出球形 Voronoi 相关模块
from ._spherical_voronoi import SphericalVoronoi
# 导入并导出绘图工具函数
from ._plotutils import *
# 导入并导出 Procrustes 函数
from ._procrustes import procrustes
# 导入并导出几何 Slerp 插值函数
from ._geometric_slerp import geometric_slerp

# 被弃用的命名空间，在 v2.0.0 版本中将会被移除
# 导入并导出 C KD 树模块
from . import ckdtree
# 导入并导出 KD 树模块
from . import kdtree
# 导入并导出 Qhull 模块
from . import qhull

# 将当前模块中不以下划线开头的所有名称添加到 __all__ 列表中
__all__ = [s for s in dir() if not s.startswith('_')]

# 导入并导出距离和变换模块
from . import distance, transform

# 将 distance 和 transform 添加到 __all__ 列表中
__all__ += ['distance', 'transform']

# 导入 PytestTester 类并创建一个测试对象 test
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
# 删除 PytestTester 类的引用，以避免全局名称空间中出现不必要的对象
del PytestTester
```