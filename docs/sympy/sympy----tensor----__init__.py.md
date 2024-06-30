# `D:\src\scipysrc\sympy\sympy\tensor\__init__.py`

```
# 导入模块 `indexed` 中的 IndexedBase, Idx, Indexed 类
# 导入模块 `index_methods` 中的 get_contraction_structure, get_indices 函数
# 导入模块 `functions` 中的 shape 函数
# 导入模块 `array` 中的以下类和函数：
# MutableDenseNDimArray, ImmutableDenseNDimArray,
# MutableSparseNDimArray, ImmutableSparseNDimArray, NDimArray,
# tensorproduct, tensorcontraction, tensordiagonal, derive_by_array, permutedims,
# Array, DenseNDimArray, SparseNDimArray
from .indexed import IndexedBase, Idx, Indexed
from .index_methods import get_contraction_structure, get_indices
from .functions import shape
from .array import (MutableDenseNDimArray, ImmutableDenseNDimArray,
    MutableSparseNDimArray, ImmutableSparseNDimArray, NDimArray, tensorproduct,
    tensorcontraction, tensordiagonal, derive_by_array, permutedims, Array,
    DenseNDimArray, SparseNDimArray,)

# 导出以下符号到外部使用
__all__ = [
    'IndexedBase', 'Idx', 'Indexed',  # 符号：IndexedBase, Idx, Indexed

    'get_contraction_structure', 'get_indices',  # 符号：get_contraction_structure, get_indices

    'shape',  # 符号：shape

    'MutableDenseNDimArray', 'ImmutableDenseNDimArray',  # 符号：MutableDenseNDimArray, ImmutableDenseNDimArray
    'MutableSparseNDimArray', 'ImmutableSparseNDimArray', 'NDimArray',  # 符号：MutableSparseNDimArray, ImmutableSparseNDimArray, NDimArray
    'tensorproduct', 'tensorcontraction', 'tensordiagonal', 'derive_by_array', 'permutedims',  # 符号：tensorproduct, tensorcontraction, tensordiagonal, derive_by_array, permutedims
    'Array', 'DenseNDimArray', 'SparseNDimArray',  # 符号：Array, DenseNDimArray, SparseNDimArray
]
```