# `.\AutoGPT\autogpts\autogpt\autogpt\core\ability\builtins\create_new_ability.py`

```py
# 导入 logging 模块
import logging
# 导入 ClassVar 类型提示
from typing import ClassVar

# 导入 Ability、AbilityConfiguration 类
from autogpt.core.ability.base import Ability, AbilityConfiguration
# 导入 AbilityResult 类
from autogpt.core.ability.schema import AbilityResult
# 导入 PluginLocation、PluginStorageFormat 类
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
# 导入 JSONSchema 类
from autogpt.core.utils.json_schema import JSONSchema

# 定义 CreateNewAbility 类，继承自 Ability 类
class CreateNewAbility(Ability):
    # 默认配置
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.CreateNewAbility",
        ),
    )

    # 初始化方法
    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
    ):
        # 初始化 logger 属性
        self._logger = logger
        # 初始化 configuration 属性
        self._configuration = configuration

    # 描述信息
    description: ClassVar[str] = "Create a new ability by writing python code."

    # 异步调用方法
    async def __call__(
        self,
        ability_name: str,
        description: str,
        arguments: list[dict],
        required_arguments: list[str],
        package_requirements: list[str],
        code: str,
    ) -> AbilityResult:
        # 抛出未实现错误
        raise NotImplementedError
```