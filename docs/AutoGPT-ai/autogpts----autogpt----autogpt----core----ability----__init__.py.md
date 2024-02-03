# `.\AutoGPT\autogpts\autogpt\autogpt\core\ability\__init__.py`

```py
"""The command system provides a way to extend the functionality of the AI agent."""
# 导入自动生成文本的核心能力基类、能力配置、能力注册表
from autogpt.core.ability.base import Ability, AbilityConfiguration, AbilityRegistry
# 导入自动生成文本的核心能力模式
from autogpt.core.ability.schema import AbilityResult
# 导入自动生成文本的简单能力注册表配置、能力注册表设置、简单能力注册表
from autogpt.core.ability.simple import (
    AbilityRegistryConfiguration,
    AbilityRegistrySettings,
    SimpleAbilityRegistry,
)

# 定义导出的模块成员
__all__ = [
    "Ability",
    "AbilityConfiguration",
    "AbilityRegistry",
    "AbilityResult",
    "AbilityRegistryConfiguration",
    "AbilityRegistrySettings",
    "SimpleAbilityRegistry",
]
```