# `.\DB-GPT-src\dbgpt\_private\config.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 引入未来的类型注解支持，确保代码在 Python 2/3 兼容
from __future__ import annotations

# 导入操作系统模块
import os
# 导入类型检查相关模块
from typing import TYPE_CHECKING, Optional

# 导入自定义的 Singleton 单例模式实现
from dbgpt.util.singleton import Singleton

# 如果在类型检查环境下
if TYPE_CHECKING:
    # 导入 SystemApp 类和 ConnectorManager 类用于类型注解
    from dbgpt.component import SystemApp
    from dbgpt.datasource.manages import ConnectorManager

# 配置类，使用 Singleton 元类确保只有一个实例存在
class Config(metaclass=Singleton):
    """Configuration class to store the state of bools for different scripts access"""

    # 返回本地数据库管理器的属性方法，类型为 ConnectorManager
    @property
    def local_db_manager(self) -> "ConnectorManager":
        # 在需要 SYSTEM_APP 存在的情况下获取 ConnectorManager 的单例实例
        from dbgpt.datasource.manages import ConnectorManager

        # 如果 SYSTEM_APP 未设置，则引发 ValueError 异常
        if not self.SYSTEM_APP:
            raise ValueError("SYSTEM_APP is not set")
        return ConnectorManager.get_instance(self.SYSTEM_APP)
```