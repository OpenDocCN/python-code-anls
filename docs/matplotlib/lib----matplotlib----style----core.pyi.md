# `D:\src\scipysrc\matplotlib\lib\matplotlib\style\core.pyi`

```py
from collections.abc import Generator
import contextlib

from matplotlib import RcParams
from matplotlib.typing import RcStyleType

# 用户自定义的库路径列表，类型为字符串列表
USER_LIBRARY_PATHS: list[str] = ...
# 样式扩展名，类型为字符串
STYLE_EXTENSION: str = ...

# 应用指定样式的函数，无返回值
def use(style: RcStyleType) -> None:
    ...

# 上下文管理器，用于设置样式和重置之后的操作
@contextlib.contextmanager
def context(
    style: RcStyleType, after_reset: bool = ...
) -> Generator[None, None, None]:
    ...

# 存储已加载库的样式配置字典，键为字符串（库名称），值为RcParams对象
library: dict[str, RcParams]
# 可用库名称列表，类型为字符串列表
available: list[str]

# 重新加载库的样式配置函数，无返回值
def reload_library() -> None:
    ...


这段代码主要是一些关于Matplotlib库样式管理的定义和函数声明。注释提供了对每个变量、函数和上下文管理器的简要解释和说明。
```