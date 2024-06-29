# `D:\src\scipysrc\pandas\pandas\core\window\__init__.py`

```
# 导入来自 pandas 库的窗口函数模块，用于时间序列数据的滚动、扩展和指数加权移动平均计算

from pandas.core.window.ewm import (
    ExponentialMovingWindow,         # 导入指数加权移动窗口对象
    ExponentialMovingWindowGroupby,  # 导入指数加权移动窗口分组对象
)
from pandas.core.window.expanding import (
    Expanding,                      # 导入扩展窗口对象
    ExpandingGroupby,               # 导入扩展窗口分组对象
)
from pandas.core.window.rolling import (
    Rolling,                        # 导入滚动窗口对象
    RollingGroupby,                 # 导入滚动窗口分组对象
    Window,                         # 导入窗口对象
)

__all__ = [                          # 定义了此模块中的公开接口列表，方便外部调用时的导入管理
    "Expanding",                    # 扩展窗口类
    "ExpandingGroupby",             # 扩展窗口分组类
    "ExponentialMovingWindow",      # 指数加权移动窗口类
    "ExponentialMovingWindowGroupby",   # 指数加权移动窗口分组类
    "Rolling",                      # 滚动窗口类
    "RollingGroupby",               # 滚动窗口分组类
    "Window",                       # 通用窗口类
]
```