# `.\AutoGPT\autogpts\autogpt\autogpt\core\plugin\__init__.py`

```py
"""The plugin system allows the Agent to be extended with new functionality."""
# 插件系统允许Agent通过新功能进行扩展

# 导入基础插件服务类
from autogpt.core.plugin.base import PluginService

# 暴露给外部的模块列表
__all__ = [
    "PluginService",
]
```