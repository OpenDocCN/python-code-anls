# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\conv_matrix_to_array.py`

```
# 从 sympy 库中导入将矩阵转换为数组的函数 from_matrix_to_array
from sympy.tensor.array.expressions import from_matrix_to_array
# 从 sympy 库中导入用于将数组转换为索引形式的装饰器函数 _conv_to_from_decorator
from sympy.tensor.array.expressions.conv_array_to_indexed import _conv_to_from_decorator

# 将 from_matrix_to_array 函数通过 _conv_to_from_decorator 转换为一个新的函数 convert_matrix_to_array
convert_matrix_to_array = _conv_to_from_decorator(from_matrix_to_array.convert_matrix_to_array)
```