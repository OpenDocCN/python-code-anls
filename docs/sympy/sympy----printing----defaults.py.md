# `D:\src\scipysrc\sympy\sympy\printing\defaults.py`

```
# 从 sympy.core._print_helpers 模块中导入 Printable 类
from sympy.core._print_helpers import Printable

# 将当前模块名赋值给 Printable 类的 __module__ 属性，用于兼容性
Printable.__module__ = __name__

# 设置 DefaultPrinting 变量为 Printable 类的引用，用于默认的打印操作
DefaultPrinting = Printable
```