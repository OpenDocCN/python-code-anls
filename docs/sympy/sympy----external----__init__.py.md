# `D:\src\scipysrc\sympy\sympy\external\__init__.py`

```
"""
统一的地方用于确定外部依赖是否已安装。

使用 import_module() 函数导入所有外部模块。

例如：

>>> from sympy.external import import_module
>>> numpy = import_module('numpy')

如果安装的库未安装，或者安装的版本低于指定的最小版本，函数将返回 None。
否则，将返回该库。更多信息请参阅 import_module() 函数的文档字符串。

"""

# 从 sympy.external.importtools 模块中导入 import_module 函数
from sympy.external.importtools import import_module

# 指定当前模块中可以被导出的符号列表，只包括 import_module 函数
__all__ = ['import_module']
```