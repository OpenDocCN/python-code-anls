# `.\AutoGPT\autogpts\autogpt\autogpt\command_decorator.py`

```py
# 导入必要的模块和类型提示
from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, ParamSpec, TypeVar

# 如果是类型检查阶段，导入特定模块
if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import Config

# 导入自定义模块
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import Command, CommandOutput, CommandParameter

# 定义 AutoGPT 命令的唯一标识符
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"

# 定义 ParamSpec 和 TypeVar
P = ParamSpec("P")
CO = TypeVar("CO", bound=CommandOutput)

# 定义装饰器 command，用于从普通函数创建 Command 对象
def command(
    name: str,
    description: str,
    parameters: dict[str, JSONSchema],
    enabled: Literal[True] | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
    aliases: list[str] = [],
    available: Literal[True] | Callable[[BaseAgent], bool] = True,
) -> Callable[[Callable[P, CO]], Callable[P, CO]]:
    """
    The command decorator is used to create Command objects from ordinary functions.
    """
    # 定义一个装饰器函数，接受一个函数作为参数并返回一个装饰后的函数
    def decorator(func: Callable[P, CO]) -> Callable[P, CO]:
        # 将参数字典转换为 CommandParameter 对象列表
        typed_parameters = [
            CommandParameter(
                name=param_name,
                spec=spec,
            )
            for param_name, spec in parameters.items()
        ]
        # 创建一个 Command 对象，包含命令的名称、描述、方法、参数、启用状态、禁用原因、别名、可用状态
        cmd = Command(
            name=name,
            description=description,
            method=func,
            parameters=typed_parameters,
            enabled=enabled,
            disabled_reason=disabled_reason,
            aliases=aliases,
            available=available,
        )

        # 如果被装饰的函数是异步函数
        if inspect.iscoroutinefunction(func):

            # 创建一个异步函数的包装器，保留原函数的参数和返回值类型
            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return await func(*args, **kwargs)

        else:
            # 创建一个普通函数的包装器，保留原函数的参数和返回值类型
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return func(*args, **kwargs)

        # 给包装器函数设置属性，包含命令对象和自动生成的 GPT 命令标识符
        setattr(wrapper, "command", cmd)
        setattr(wrapper, AUTO_GPT_COMMAND_IDENTIFIER, True)

        # 返回装饰后的函数
        return wrapper

    # 返回装饰器函数
    return decorator
```