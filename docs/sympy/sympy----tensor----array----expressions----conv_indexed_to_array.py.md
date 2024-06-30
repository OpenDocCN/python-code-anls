# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\conv_indexed_to_array.py`

```
# 从 sympy 库中导入将索引数组表达式转换为数组的函数
from sympy.tensor.array.expressions import from_indexed_to_array
# 从 sympy 库中导入用于将数组转换为索引数组表达式的装饰器函数
from sympy.tensor.array.expressions.conv_array_to_indexed import _conv_to_from_decorator

# 使用装饰器函数 _conv_to_from_decorator 将 from_indexed_to_array.convert_indexed_to_array 转换为
# convert_indexed_to_array 函数，使其具有从索引数组表达式转换为数组的功能
convert_indexed_to_array = _conv_to_from_decorator(from_indexed_to_array.convert_indexed_to_array)
```