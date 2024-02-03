# `.\AutoGPT\autogpts\autogpt\autogpt\core\ability\base.py`

```py
# 导入必要的模块和类
import abc
from pprint import pformat
from typing import Any, ClassVar

import inflection
from pydantic import Field

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.base import PluginLocation
from autogpt.core.resource.model_providers import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema

# 导入 AbilityResult 类
from .schema import AbilityResult

# 定义 AbilityConfiguration 类，继承自 SystemConfiguration
class AbilityConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    # 插件位置
    location: PluginLocation
    # 所需的包列表
    packages_required: list[str] = Field(default_factory=list)
    # 所需的语言模型配置
    language_model_required: LanguageModelConfiguration = None
    # 是否需要内存提供程序
    memory_provider_required: bool = False
    # 是否需要工作空间
    workspace_required: bool = False

# 定义 Ability 抽象类，继承自 ABC 类
class Ability(abc.ABC):
    """A class representing an agent ability."""

    # 默认配置
    default_configuration: ClassVar[AbilityConfiguration]

    # 返回能力的名称
    @classmethod
    def name(cls) -> str:
        """The name of the ability."""
        return inflection.underscore(cls.__name__)

    # 返回能力的描述
    @property
    @classmethod
    @abc.abstractmethod
    def description(cls) -> str:
        """A detailed description of what the ability does."""
        ...

    # 返回能力的参数
    @property
    @classmethod
    @abc.abstractmethod
    def parameters(cls) -> dict[str, JSONSchema]:
        ...

    # 调用能力的方法
    @abc.abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> AbilityResult:
        ...

    # 返回能力的字符串表示
    def __str__(self) -> str:
        return pformat(self.spec)

    # 返回能力的规范
    @property
    @classmethod
    def spec(cls) -> CompletionModelFunction:
        return CompletionModelFunction(
            name=cls.name(),
            description=cls.description,
            parameters=cls.parameters,
        )

# 定义 AbilityRegistry 抽象类，继承自 ABC 类
class AbilityRegistry(abc.ABC):
    # 注册能力的抽象方法
    @abc.abstractmethod
    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration
    ) -> None:
        ...
    # 定义一个方法，用于列出能力列表，返回一个字符串列表
    def list_abilities(self) -> list[str]:
        ...
    
    # 定义一个抽象方法，用于导出能力列表，返回一个 CompletionModelFunction 对象列表
    @abc.abstractmethod
    def dump_abilities(self) -> list[CompletionModelFunction]:
        ...
    
    # 定义一个抽象方法，用于获取指定能力的 Ability 对象
    @abc.abstractmethod
    def get_ability(self, ability_name: str) -> Ability:
        ...
    
    # 定义一个抽象方法，用于执行指定能力，返回一个 AbilityResult 对象
    @abc.abstractmethod
    async def perform(self, ability_name: str, **kwargs: Any) -> AbilityResult:
        ...
```