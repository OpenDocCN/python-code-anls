# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\from_array_to_indexed.py`

```
# 导入collections.abc模块，包含抽象基类，支持Python的集合类操作
import collections.abc
# 导入operator模块，提供了Python中常见的运算符函数
import operator
# 从itertools模块导入accumulate函数，用于累积计算
from itertools import accumulate

# 从sympy库导入特定子模块和类
from sympy import Mul, Sum, Dummy, Add
# 从sympy.tensor.array.expressions子模块导入多维数组表达式类
from sympy.tensor.array.expressions import PermuteDims, ArrayAdd, ArrayElementwiseApplyFunc, Reshape
# 从sympy.tensor.array.expressions.array_expressions子模块导入特定类和函数
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, get_rank, ArrayContraction, \
    ArrayDiagonal, get_shape, _get_array_element_or_slice, _ArrayExpr
# 从sympy.tensor.array.expressions.utils子模块导入特定函数
from sympy.tensor.array.expressions.utils import _apply_permutation_to_list


def convert_array_to_indexed(expr, indices):
    # 返回_ConvertArrayToIndexed对象的do_convert方法的结果
    return _ConvertArrayToIndexed().do_convert(expr, indices)


class _ConvertArrayToIndexed:

    def __init__(self):
        # 初始化计数器，用于记录虚拟变量（Dummy对象）的数量
        self.count_dummies = 0
```