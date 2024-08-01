# `.\DB-GPT-src\dbgpt\core\awel\util\parameter_util.py`

```py
"""The parameter utility."""

# 导入需要的模块和类
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

# 导入内部模块和类
from dbgpt._private.pydantic import BaseModel, Field, model_validator
from dbgpt.core.interface.serialization import Serializable

# 默认的动态注册表
_DEFAULT_DYNAMIC_REGISTRY = {}

# 选项数值的定义，继承自 Serializable 和 Pydantic 的 BaseModel
class OptionValue(Serializable, BaseModel):
    """The option value of the parameter."""

    label: str = Field(..., description="The label of the option")
    name: str = Field(..., description="The name of the option")
    value: Any = Field(..., description="The value of the option")

    def to_dict(self) -> Dict:
        """Convert current metadata to json dict."""
        return self.dict()

# 动态选项的基类，继承自 Serializable 和 Pydantic 的 BaseModel，并且是抽象类
class BaseDynamicOptions(Serializable, BaseModel, ABC):
    """The base dynamic options."""

    @abstractmethod
    def option_values(self) -> List[OptionValue]:
        """Return the option values of the parameter."""

# 函数动态选项，继承自 BaseDynamicOptions
class FunctionDynamicOptions(BaseDynamicOptions):
    """The function dynamic options."""

    # 用于生成动态选项的函数，字段描述了其作用
    func: Callable[[], List[OptionValue]] = Field(
        ..., description="The function to generate the dynamic options"
    )
    # 生成动态选项函数的唯一标识符，字段描述了其作用
    func_id: str = Field(
        ..., description="The unique id of the function to generate the dynamic options"
    )

    def option_values(self) -> List[OptionValue]:
        """Return the option values of the parameter."""
        return self.func()

    # 模型验证器，用于在实例化前填充函数的 ID
    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre fill the function id."""
        if not isinstance(values, dict):
            return values
        func = values.get("func")
        if func is None:
            raise ValueError(
                "The function to generate the dynamic options is required."
            )
        func_id = _generate_unique_id(func)
        values["func_id"] = func_id
        _DEFAULT_DYNAMIC_REGISTRY[func_id] = func
        return values

    def to_dict(self) -> Dict:
        """Convert current metadata to json dict."""
        return {"func_id": self.func_id}

# 生成唯一函数 ID 的内部函数
def _generate_unique_id(func: Callable) -> str:
    if func.__name__ == "<lambda>":
        func_id = f"lambda_{inspect.getfile(func)}_{inspect.getsourcelines(func)}"
    else:
        func_id = f"{func.__module__}.{func.__name__}"
    return func_id
```