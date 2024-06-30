# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\conv_array_to_indexed.py`

```
# 从 sympy.tensor.array.expressions 模块中导入 from_array_to_indexed 函数
from sympy.tensor.array.expressions import from_array_to_indexed
# 从 sympy.utilities.decorator 模块中导入 deprecated 装饰器
from sympy.utilities.decorator import deprecated

# 创建一个名为 _conv_to_from_decorator 的变量，使用 deprecated 装饰器对 from_array_to_indexed 函数进行处理
# 警告信息说明模块已经更名，将名称中的 'conv_' 替换为 'from_'，生效于版本 "1.11"
# active_deprecations_target 指定了针对过时的转换数组表达式模块名称的目标
_conv_to_from_decorator = deprecated(
    "module has been renamed by replacing 'conv_' with 'from_' in its name",
    deprecated_since_version="1.11",
    active_deprecations_target="deprecated-conv-array-expr-module-names",
)

# 使用 _conv_to_from_decorator 装饰 from_array_to_indexed.convert_array_to_indexed 函数
convert_array_to_indexed = _conv_to_from_decorator(from_array_to_indexed.convert_array_to_indexed)
```