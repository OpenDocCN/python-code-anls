# `.\AutoGPT\autogpts\autogpt\autogpt\plugins\plugin_config.py`

```py
# 导入必要的模块
from typing import Any

from pydantic import BaseModel

# 定义一个插件配置类，用于保存单个插件的配置信息
class PluginConfig(BaseModel):
    """Class for holding configuration of a single plugin"""
    
    # 插件名称，类型为字符串
    name: str
    # 插件是否启用，默认为 False
    enabled: bool = False
    # 插件的配置信息，类型为字典，键为字符串，值为任意类型
    config: dict[str, Any] = None
```