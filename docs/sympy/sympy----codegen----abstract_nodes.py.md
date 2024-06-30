# `D:\src\scipysrc\sympy\sympy\codegen\abstract_nodes.py`

```
"""
This module provides containers for python objects that are valid
printing targets but are not a subclass of SymPy's Printable.
"""

# 导入从 sympy.core.containers 模块中导入 Tuple 类
from sympy.core.containers import Tuple


# 定义一个名为 List 的类，继承自 Tuple 类
class List(Tuple):
    """Represents a (frozen) (Python) list (for code printing purposes)."""

    # 重写 __eq__ 方法，用于比较对象相等性
    def __eq__(self, other):
        # 如果 other 是一个 Python 的 list 类型
        if isinstance(other, list):
            # 返回当前 List 对象与使用 other 创建的 List 对象的比较结果
            return self == List(*other)
        else:
            # 否则，比较当前对象的 args 属性与 other 的相等性
            return self.args == other

    # 重写 __hash__ 方法，返回当前对象的哈希值
    def __hash__(self):
        # 调用父类 Tuple 的 __hash__ 方法计算哈希值并返回
        return super().__hash__()
```