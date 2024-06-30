# `D:\src\scipysrc\scipy\scipy\spatial\transform\__init__.py`

```
"""
Spatial Transformations (:mod:`scipy.spatial.transform`)
========================================================

.. currentmodule:: scipy.spatial.transform

This package implements various spatial transformations. For now,
only rotations are supported.

Rotations in 3 dimensions
-------------------------
.. autosummary::
   :toctree: generated/

   Rotation
   Slerp
   RotationSpline
"""

# 导入 Rotation 和 Slerp 类，这些类定义在 _rotation 模块中
from ._rotation import Rotation, Slerp

# 导入 RotationSpline 类，这个类定义在 _rotation_spline 模块中
from ._rotation_spline import RotationSpline

# Deprecated namespaces, to be removed in v2.0.0
# 引入旧版命名空间 rotation，计划在 v2.0.0 版本中移除
from . import rotation

# 定义 __all__ 列表，指定模块中可以被导入的对象
__all__ = ['Rotation', 'Slerp', 'RotationSpline']

# 导入 PytestTester 类，并创建一个与当前模块名相关联的测试对象 test
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)

# 删除 PytestTester 类的引用，以确保不会在模块中保留对其的引用
del PytestTester
```