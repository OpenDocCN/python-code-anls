# `.\AutoGPT\autogpts\autogpt\autogpt\core\resource\schema.py`

```py
# 导入必要的模块
import abc
import enum

from pydantic import BaseModel, SecretBytes, SecretField, SecretStr

from autogpt.core.configuration import (
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)

# 定义资源类型的枚举类
class ResourceType(str, enum.Enum):
    """An enumeration of resource types."""
    MODEL = "model"
    MEMORY = "memory"

# 提供者使用情况的抽象基类
class ProviderUsage(SystemConfiguration, abc.ABC):
    @abc.abstractmethod
    def update_usage(self, *args, **kwargs) -> None:
        """Update the usage of the resource."""
        ...

# 提供者预算的抽象基类
class ProviderBudget(SystemConfiguration):
    total_budget: float = UserConfigurable()
    total_cost: float
    remaining_budget: float
    usage: ProviderUsage

    @abc.abstractmethod
    def update_usage_and_cost(self, *args, **kwargs) -> float:
        """Update the usage and cost of the provider.

        Returns:
            float: The (calculated) cost of the given model response.
        """
        ...

# 提供者凭据的结构体
class ProviderCredentials(SystemConfiguration):
    """Struct for credentials."""

    # 返回未掩码的凭据信息
    def unmasked(self) -> dict:
        return unmask(self)

    class Config:
        # 定义 JSON 编码器，用于处理敏感信息
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None,
            SecretBytes: lambda v: v.get_secret_value() if v else None,
            SecretField: lambda v: v.get_secret_value() if v else None,
        }

# 解除模型中的敏感信息掩码
def unmask(model: BaseModel):
    unmasked_fields = {}
    for field_name, _ in model.__fields__.items():
        value = getattr(model, field_name)
        if isinstance(value, SecretStr):
            unmasked_fields[field_name] = value.get_secret_value()
        else:
            unmasked_fields[field_name] = value
    return unmasked_fields

# 提供者设置的结构体
class ProviderSettings(SystemSettings):
    resource_type: ResourceType
    credentials: ProviderCredentials | None = None
    budget: ProviderBudget | None = None

# 用于模型提供者和内存提供者的嵌入向量类型
Embedding = list[float]
```