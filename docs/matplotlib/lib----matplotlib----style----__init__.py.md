# `D:\src\scipysrc\matplotlib\lib\matplotlib\style\__init__.py`

```py
# 从当前包（或模块）的core模块中导入指定的变量和函数
from .core import available, context, library, reload_library, use

# 定义一个列表，包含了从core模块中导入的公开变量和函数的名称，这些名称可以通过 `from package import *` 的方式导入
__all__ = ["available", "context", "library", "reload_library", "use"]
```