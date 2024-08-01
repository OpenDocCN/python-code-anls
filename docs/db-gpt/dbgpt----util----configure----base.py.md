# `.\DB-GPT-src\dbgpt\util\configure\base.py`

```py
"""Configuration base module."""

# 引入日志模块
import logging
# 引入抽象基类模块
from abc import ABC, abstractmethod
# 引入枚举类型模块
from enum import Enum
# 引入类型提示模块
from typing import Any, Optional, Union

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 表示缺失值的特殊类型
class _MISSING_TYPE:
    pass

# 表示缺失值的单例对象
_MISSING = _MISSING_TYPE()

# 配置分类枚举，用于标识配置的分类
class ConfigCategory(str, Enum):
    """The configuration category."""
    AGENT = "agent"

# 提供者类型枚举，用于标识配置提供者的类型
class ProviderType(str, Enum):
    """The provider type."""
    ENV = "env"
    PROMPT_MANAGER = "prompt_manager"

# 配置提供者的抽象基类，定义了配置提供者应具备的行为
class ConfigProvider(ABC):
    """The configuration provider."""
    
    name: ProviderType
    
    @abstractmethod
    def query(self, key: str, **kwargs) -> Any:
        """Query the configuration value by key."""

# 环境配置提供者，从环境变量中获取配置值
class EnvironmentConfigProvider(ConfigProvider):
    """Environment configuration provider.
    
    Obtain the configuration value from the environment variable.
    """
    
    name: ProviderType = ProviderType.ENV
    
    def query(self, key: str, **kwargs) -> Any:
        import os
        return os.environ.get(key, None)

# 提示管理器配置提供者，从提示管理器获取配置值（仅在运行 DB-GPT web 服务器时有效）
class PromptManagerConfigProvider(ConfigProvider):
    """Prompt manager configuration provider.
    
    Obtain the configuration value from the prompt manager.
    
    It is valid only when DB-GPT web server is running for now.
    """
    
    name: ProviderType = ProviderType.PROMPT_MANAGER
    
    def query(self, key: str, **kwargs) -> Any:
        from dbgpt._private.config import Config
        try:
            from dbgpt.serve.prompt.serve import Serve
        except ImportError:
            logger.debug("Prompt manager is not available.")
            return None
        
        # 获取配置对象并检查系统应用设置
        cfg = Config()
        sys_app = cfg.SYSTEM_APP
        if not sys_app:
            return None
        
        # 获取提示服务实例，并检查是否存在有效的提示管理器
        prompt_serve = Serve.get_instance(sys_app)
        if not prompt_serve or not prompt_serve.prompt_manager:
            return None
        
        # 获取提示管理器对象，并查询指定键的配置值
        prompt_manager = prompt_serve.prompt_manager
        value = prompt_manager.prefer_query(key, **kwargs)
        if not value:
            return None
        
        # 返回第一个配置值的模板表示
        return value[0].to_prompt_template().template

# 配置信息类，包含配置的默认值、键、提供者、是否为列表、列表分隔符和描述
class ConfigInfo:
    def __init__(
        self,
        default: Any,
        key: Optional[str] = None,
        provider: Optional[Union[str, ConfigProvider]] = None,
        is_list: bool = False,
        separator: str = "[LIST_SEP]",
        description: Optional[str] = None,
    ):
        self.default = default
        self.key = key
        self.provider = provider
        self.is_list = is_list
        self.separator = separator
        self.description = description
    # 定义一个方法 query，接受任意关键字参数并返回任意类型的值
    def query(self, **kwargs) -> Any:
        # 如果未设置 self.key，则返回默认值 self.default
        if self.key is None:
            return self.default
        # 初始化变量 value，先设为 None
        value: Any = None
        # 如果 self.provider 是 ConfigProvider 的实例
        if isinstance(self.provider, ConfigProvider):
            # 调用 self.provider 的 query 方法，传递 self.key 和其他关键字参数
            value = self.provider.query(self.key, **kwargs)
        # 如果 self.provider 是 ProviderType.ENV
        elif self.provider == ProviderType.ENV:
            # 创建 EnvironmentConfigProvider 对象，并调用其 query 方法，传递 self.key 和其他关键字参数
            value = EnvironmentConfigProvider().query(self.key, **kwargs)
        # 如果 self.provider 是 ProviderType.PROMPT_MANAGER
        elif self.provider == ProviderType.PROMPT_MANAGER:
            # 创建 PromptManagerConfigProvider 对象，并调用其 query 方法，传递 self.key 和其他关键字参数
            value = PromptManagerConfigProvider().query(self.key, **kwargs)
        # 如果 value 为 None，则将其设为默认值 self.default
        if value is None:
            value = self.default
        # 如果 value 存在且 self.is_list 为真且 value 是字符串类型，则使用分隔符 self.separator 分割字符串
        if value and self.is_list and isinstance(value, str):
            value = value.split(self.separator)
        # 返回最终的 value 值
        return value
def DynConfig(
    default: Any = _MISSING,
    *,
    category: str | ConfigCategory | None = None,
    key: str | None = None,
    provider: str | ProviderType | ConfigProvider | None = None,
    is_list: bool = False,
    separator: str = "[LIST_SEP]",
    description: str | None = None,
) -> Any:
    """Dynamic configuration.

    It allows to query the configuration value dynamically.
    It can obtain the configuration value from the specified provider.

    **Note**: Now just support obtaining string value or string list value.

    Args:
        default (Any): The default value.
        category (str | ConfigCategory | None): The configuration category.
        key (str | None): The configuration key.
        provider (str | ProviderType | ConfigProvider | None): The configuration
            provider.
        is_list (bool): Whether the value is a list.
        separator (str): The separator to split the list value.
        description (str | None): The configuration description.
    """
    # 如果 provider 为 None 且 category 是 ConfigCategory.AGENT，将 provider 设置为 ProviderType.PROMPT_MANAGER
    if provider is None and category == ConfigCategory.AGENT:
        provider = ProviderType.PROMPT_MANAGER
    # 如果 default 是 _MISSING 且 key 为 None，则抛出 ValueError
    if default == _MISSING and key is None:
        raise ValueError("Default value or key is required.")
    # 如果 default 不是 _MISSING 并且 default 是列表类型，则将 is_list 设为 True
    if default != _MISSING and isinstance(default, list):
        is_list = True
    # 返回 ConfigInfo 对象，传入各个参数
    return ConfigInfo(
        default=default,
        key=key,
        provider=provider,
        is_list=is_list,
        separator=separator,
        description=description,
    )
```