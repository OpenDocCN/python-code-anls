# `.\AutoGPT\autogpts\autogpt\autogpt\config\__init__.py`

```py
"""
This module contains the configuration classes for AutoGPT.
"""
# 导入 AIProfile 类
from .ai_profile import AIProfile
# 导入 AIDirectives 类
from .ai_directives import AIDirectives
# 导入 Config, ConfigBuilder, assert_config_has_openai_api_key 函数
from .config import Config, ConfigBuilder, assert_config_has_openai_api_key

# 定义 __all__ 列表，包含需要导出的模块成员
__all__ = [
    "assert_config_has_openai_api_key",
    "AIProfile",
    "AIDirectives",
    "Config",
    "ConfigBuilder",
]
```