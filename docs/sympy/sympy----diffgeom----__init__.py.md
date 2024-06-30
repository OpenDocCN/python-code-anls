# `D:\src\scipysrc\sympy\sympy\diffgeom\__init__.py`

```
# 从diffgeom模块中导入多个类和函数，这些类和函数包括了微分几何相关的计算和操作
from .diffgeom import (
    BaseCovarDerivativeOp, BaseScalarField, BaseVectorField, Commutator,
    contravariant_order, CoordSystem, CoordinateSymbol,
    CovarDerivativeOp, covariant_order, Differential, intcurve_diffequ,
    intcurve_series, LieDerivative, Manifold, metric_to_Christoffel_1st,
    metric_to_Christoffel_2nd, metric_to_Ricci_components,
    metric_to_Riemann_components, Patch, Point, TensorProduct, twoform_to_matrix,
    vectors_in_basis, WedgeProduct,
)

# 定义__all__列表，指定了在使用from * 导入时应该导入的符号（符号即类和函数名）
__all__ = [
    'BaseCovarDerivativeOp', 'BaseScalarField', 'BaseVectorField', 'Commutator',
    'contravariant_order', 'CoordSystem', 'CoordinateSymbol',
    'CovarDerivativeOp', 'covariant_order', 'Differential', 'intcurve_diffequ',
    'intcurve_series', 'LieDerivative', 'Manifold', 'metric_to_Christoffel_1st',
    'metric_to_Christoffel_2nd', 'metric_to_Ricci_components',
    'metric_to_Riemann_components', 'Patch', 'Point', 'TensorProduct',
    'twoform_to_matrix', 'vectors_in_basis', 'WedgeProduct',
]
```