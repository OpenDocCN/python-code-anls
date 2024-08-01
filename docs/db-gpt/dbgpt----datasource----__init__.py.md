# `.\DB-GPT-src\dbgpt\datasource\__init__.py`

```py
"""Module to define the data source connectors."""

# 导入必要的类型声明
from typing import Any

# 导入基础连接器类，忽略未使用的警告
from .base import BaseConnector  # noqa: F401


# 定义一个特殊函数 __getattr__，用于动态获取属性
def __getattr__(name: str) -> Any:
    # 如果请求的属性名是 "RDBMSConnector"
    if name == "RDBMSConnector":
        # 动态导入 RDBMSConnector 类，忽略未使用的警告
        from .rdbms.base import RDBMSConnector  # noqa: F401

        # 返回 RDBMSConnector 类
        return RDBMSConnector
    else:
        # 如果请求的属性名不是 "RDBMSConnector"，抛出属性错误异常
        raise AttributeError(f"Could not find: {name} in datasource")


# 定义一个列表，包含模块中所有公开的对象名，用于 from module import * 形式的导入
__ALL__ = ["BaseConnector", "RDBMSConnector"]
```