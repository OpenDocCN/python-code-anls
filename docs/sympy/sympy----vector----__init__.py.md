# `D:\src\scipysrc\sympy\sympy\vector\__init__.py`

```
# 导入 SymPy 的三维坐标系对象 CoordSys3D
from sympy.vector.coordsysrect import CoordSys3D
# 导入 SymPy 的向量相关类和函数
from sympy.vector.vector import (Vector, VectorAdd, VectorMul,
                                 BaseVector, VectorZero, Cross, Dot, cross, dot)
# 导入 SymPy 的二阶张量相关类和函数
from sympy.vector.dyadic import (Dyadic, DyadicAdd, DyadicMul,
                                 BaseDyadic, DyadicZero)
# 导入 SymPy 的标量相关类
from sympy.vector.scalar import BaseScalar
# 导入 SymPy 的向量微分操作类 Del
from sympy.vector.deloperator import Del
# 导入 SymPy 的向量函数和操作
from sympy.vector.functions import (express, matrix_to_vector,
                                    laplacian, is_conservative,
                                    is_solenoidal, scalar_potential,
                                    directional_derivative,
                                    scalar_potential_difference)
# 导入 SymPy 的点对象
from sympy.vector.point import Point
# 导入 SymPy 的方向定位器类
from sympy.vector.orienters import (AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)
# 导入 SymPy 的向量运算符
from sympy.vector.operators import Gradient, Divergence, Curl, Laplacian, gradient, curl, divergence
# 导入 SymPy 的隐式区域类
from sympy.vector.implicitregion import ImplicitRegion
# 导入 SymPy 的参数区域类和相关函数
from sympy.vector.parametricregion import (ParametricRegion, parametric_region_list)
# 导入 SymPy 的参数积分类和向量积分函数
from sympy.vector.integrals import (ParametricIntegral, vector_integrate)
# 导入 SymPy 的向量种类类
from sympy.vector.kind import VectorKind

# 定义 __all__ 列表，指定可以通过 from module import * 导入的内容
__all__ = [
    'Vector', 'VectorAdd', 'VectorMul', 'BaseVector', 'VectorZero', 'Cross',
    'Dot', 'cross', 'dot',

    'VectorKind',

    'Dyadic', 'DyadicAdd', 'DyadicMul', 'BaseDyadic', 'DyadicZero',

    'BaseScalar',

    'Del',

    'CoordSys3D',

    'express', 'matrix_to_vector', 'laplacian', 'is_conservative',
    'is_solenoidal', 'scalar_potential', 'directional_derivative',
    'scalar_potential_difference',

    'Point',

    'AxisOrienter', 'BodyOrienter', 'SpaceOrienter', 'QuaternionOrienter',

    'Gradient', 'Divergence', 'Curl', 'Laplacian', 'gradient', 'curl',
    'divergence',

    'ParametricRegion', 'parametric_region_list', 'ImplicitRegion',

    'ParametricIntegral', 'vector_integrate',
]
```