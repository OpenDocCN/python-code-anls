# `D:\src\scipysrc\sympy\sympy\codegen\pynodes.py`

```
# 从当前目录下的 abstract_nodes 模块中导入 List 类，并将其命名为 AbstractList
from .abstract_nodes import List as AbstractList
# 从当前目录下的 ast 模块中导入 Token 类
from .ast import Token

# 定义一个名为 List 的类，继承自 AbstractList 类
class List(AbstractList):
    # 空的类定义，没有额外的属性或方法，直接继承自 AbstractList
    pass

# 定义一个名为 NumExprEvaluate 的类，继承自 Token 类
class NumExprEvaluate(Token):
    """表示对 numexpr 库中 evaluate 函数的调用"""

    # 定义 __slots__ 属性，限定该类的实例只能有 'expr' 属性
    __slots__ = _fields = ('expr',)
```