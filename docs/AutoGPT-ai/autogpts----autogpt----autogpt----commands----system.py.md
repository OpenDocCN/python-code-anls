# `.\AutoGPT\autogpts\autogpt\autogpt\commands\system.py`

```py
"""Commands to control the internal state of the program"""

# 导入未来版本的注解特性
from __future__ import annotations

# 导入日志记录模块和类型检查模块
import logging
from typing import TYPE_CHECKING

# 导入相关模块和类
from autogpt.agents.features.context import get_agent_context
from autogpt.agents.utils.exceptions import AgentTerminated, InvalidArgumentError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

# 定义命令类别和标题
COMMAND_CATEGORY = "system"
COMMAND_CATEGORY_TITLE = "System"

# 如果是类型检查模式
if TYPE_CHECKING:
    from autogpt.agents.agent import Agent

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义命令函数"finish"
@command(
    "finish",
    "Use this to shut down once you have completed your task,"
    " or when there are insurmountable problems that make it impossible"
    " for you to finish your task.",
    {
        "reason": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A summary to the user of how the goals were accomplished",
            required=True,
        )
    },
)
def finish(reason: str, agent: Agent) -> None:
    """
    A function that takes in a string and exits the program

    Parameters:
        reason (str): A summary to the user of how the goals were accomplished.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """
    # 抛出代理终止异常，传入终止原因
    raise AgentTerminated(reason)

# 定义命令函数"close_context_item"
@command(
    "hide_context_item",
    "Hide an open file, folder or other context item, to save memory.",
    {
        "number": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The 1-based index of the context item to hide",
            required=True,
        )
    },
    # 可用性检查函数，检查代理上下文是否存在
    available=lambda a: bool(get_agent_context(a)),
)
def close_context_item(number: int, agent: Agent) -> str:
    # 确保上下文存在
    assert (context := get_agent_context(agent)) is not None

    # 如果索引超出范围或为0，抛出无效参数异常
    if number > len(context.items) or number == 0:
        raise InvalidArgumentError(f"Index {number} out of range")

    # 关闭指定索引的上下文项
    context.close(number)
    # 返回一个带有特定数字的字符串，表示上下文项目被隐藏
    return f"Context item {number} hidden ✅"
```