# `.\AutoGPT\autogpts\autogpt\autogpt\models\command.py`

```py
from __future__ import annotations
# 导入未来版本的注解特性

import inspect
# 导入 inspect 模块，用于检查对象

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional
# 导入类型提示相关的模块和类型

if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import Config
# 如果是类型检查阶段，则导入 BaseAgent 和 Config 类型

from .command_parameter import CommandParameter
# 从当前目录下导入 command_parameter 模块

from .context_item import ContextItem
# 从当前目录下导入 context_item 模块

CommandReturnValue = Any
# 定义 CommandReturnValue 类型为 Any

CommandOutput = CommandReturnValue | tuple[CommandReturnValue, ContextItem]
# 定义 CommandOutput 类型为 CommandReturnValue 或者元组(CommandReturnValue, ContextItem)

class Command:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """
    # 定义 Command 类，表示一个命令

    def __init__(
        self,
        name: str,
        description: str,
        method: Callable[..., CommandOutput],
        parameters: list[CommandParameter],
        enabled: Literal[True] | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
        aliases: list[str] = [],
        available: Literal[True] | Callable[[BaseAgent], bool] = True,
    ):
        # 初始化方法，接受命令的名称、描述、执行方法、参数、启用状态、禁用原因、别名、可用状态等参数
        self.name = name
        self.description = description
        self.method = method
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.aliases = aliases
        self.available = available

    @property
    def is_async(self) -> bool:
        # 定义 is_async 属性，用于检查方法是否为异步方法
        return inspect.iscoroutinefunction(self.method)
    # 定义一个特殊方法，用于调用 Command 对象
    def __call__(self, *args, agent: BaseAgent, **kwargs) -> Any:
        # 检查是否启用了该命令，并且传入的代理对象的 legacy_config 符合要求
        if callable(self.enabled) and not self.enabled(agent.legacy_config):
            # 如果命令被禁用，抛出异常并显示禁用原因
            if self.disabled_reason:
                raise RuntimeError(
                    f"Command '{self.name}' is disabled: {self.disabled_reason}"
                )
            # 如果命令被禁用但没有原因，抛出异常
            raise RuntimeError(f"Command '{self.name}' is disabled")

        # 检查是否可用该命令，并且传入的代理对象符合要求
        if callable(self.available) and not self.available(agent):
            # 如果命令不可用，抛出异常
            raise RuntimeError(f"Command '{self.name}' is not available")

        # 调用 Command 对象的方法，并传入参数和代理对象
        return self.method(*args, **kwargs, agent=agent)

    # 定义一个特殊方法，用于返回 Command 对象的字符串表示
    def __str__(self) -> str:
        # 根据参数列表生成参数字符串
        params = [
            f"{param.name}: "
            + ("%s" if param.spec.required else "Optional[%s]") % param.spec.type.value
            for param in self.parameters
        ]
        # 返回包含命令名称、描述和参数的字符串
        return (
            f"{self.name}: {self.description.rstrip('.')}. "
            f"Params: ({', '.join(params)})"
        )
```