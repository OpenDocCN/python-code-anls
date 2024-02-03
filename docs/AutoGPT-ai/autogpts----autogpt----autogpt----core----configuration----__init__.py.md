# `.\AutoGPT\autogpts\autogpt\autogpt\core\configuration\__init__.py`

```py
# 该配置文件封装了所有 Agent 子系统的设置
# 导入必要的模块和类
from autogpt.core.configuration.schema import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
# 暴露给外部的类和模块列表
__all__ = [
    "Configurable",
    "SystemConfiguration",
    "SystemSettings",
    "UserConfigurable",
]
```