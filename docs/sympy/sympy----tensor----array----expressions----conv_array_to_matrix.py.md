# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\conv_array_to_matrix.py`

```
# 从 sympy.tensor.array.expressions 模块中导入 from_array_to_matrix 函数
# from_array_to_matrix 函数用于将数组转换为矩阵的表示
from sympy.tensor.array.expressions import from_array_to_matrix

# 从 sympy.tensor.array.expressions.conv_array_to_indexed 模块中导入 _conv_to_from_decorator 函数
# _conv_to_from_decorator 函数用于装饰转换函数，使其支持从数组到索引的转换
from sympy.tensor.array.expressions.conv_array_to_indexed import _conv_to_from_decorator

# 将 from_array_to_matrix.convert_array_to_matrix 函数装饰为 convert_array_to_matrix 函数
# convert_array_to_matrix 函数用于执行将数组转换为矩阵的操作
convert_array_to_matrix = _conv_to_from_decorator(from_array_to_matrix.convert_array_to_matrix)

# 将 from_array_to_matrix._array2matrix 函数装饰为 _array2matrix 函数
# _array2matrix 函数用于执行内部将数组转换为矩阵的操作
_array2matrix = _conv_to_from_decorator(from_array_to_matrix._array2matrix)

# 将 from_array_to_matrix._remove_trivial_dims 函数装饰为 _remove_trivial_dims 函数
# _remove_trivial_dims 函数用于执行内部去除矩阵中不必要维度的操作
_remove_trivial_dims = _conv_to_from_decorator(from_array_to_matrix._remove_trivial_dims)
```