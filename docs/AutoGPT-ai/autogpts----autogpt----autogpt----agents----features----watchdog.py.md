# `.\AutoGPT\autogpts\autogpt\autogpt\agents\features\watchdog.py`

```py
from __future__ import annotations
# 导入用于支持类型提示的模块

import logging
# 导入日志记录模块
from contextlib import ExitStack
# 导入上下文管理模块中的 ExitStack 类
from typing import TYPE_CHECKING
# 导入用于类型检查的模块

if TYPE_CHECKING:
    from ..base import BaseAgentConfiguration
# 如果是类型检查模式，则导入 BaseAgentConfiguration 类

from autogpt.models.action_history import EpisodicActionHistory
# 导入 EpisodicActionHistory 类

from ..base import BaseAgent
# 导入 BaseAgent 类

logger = logging.getLogger(__name__)
# 获取当前模块的日志记录器

class WatchdogMixin:
    """
    Mixin that adds a watchdog feature to an agent class. Whenever the agent starts
    looping, the watchdog will switch from the FAST_LLM to the SMART_LLM and re-think.
    """
    # WatchdogMixin 类，为代理类添加了看门狗功能。每当代理开始循环时，看门狗将从 FAST_LLM 切换到 SMART_LLM 并重新思考。

    config: BaseAgentConfiguration
    # 类型注解，指定 config 属性的类型为 BaseAgentConfiguration

    event_history: EpisodicActionHistory
    # 类型注解，指定 event_history 属性的类型为 EpisodicActionHistory

    def __init__(self, **kwargs) -> None:
        # 初始化方法，接受任意关键字参数

        # 先初始化其他基类，因为我们需要从 BaseAgent 获取 event_history
        super(WatchdogMixin, self).__init__(**kwargs)
        # 调用父类的初始化方法

        if not isinstance(self, BaseAgent):
            # 如果当前实例不是 BaseAgent 的派生类
            raise NotImplementedError(
                f"{__class__.__name__} can only be applied to BaseAgent derivatives"
            )
            # 抛出未实现错误，提示 WatchdogMixin 只能应用于 BaseAgent 的派生类
    # 异步方法，用于提出行动建议，返回思考过程的输出
    async def propose_action(self, *args, **kwargs) -> BaseAgent.ThoughtProcessOutput:
        # 调用 WatchdogMixin 类的 propose_action 方法，获取命令名称、命令参数和思考过程
        command_name, command_args, thoughts = await super(
            WatchdogMixin, self
        ).propose_action(*args, **kwargs)

        # 如果不是大脑智能且快速低水平管理器与智能低水平管理器不相同
        if not self.config.big_brain and self.config.fast_llm != self.config.smart_llm:
            previous_command, previous_command_args = None, None
            # 如果事件历史记录长度大于1
            if len(self.event_history) > 1:
                # 检测重复命令
                previous_cycle = self.event_history.episodes[
                    self.event_history.cursor - 1
                ]
                previous_command = previous_cycle.action.name
                previous_command_args = previous_cycle.action.args

            rethink_reason = ""

            # 如果没有命令名称
            if not command_name:
                rethink_reason = "AI did not specify a command"
            # 如果命令名称与上一个命令名称以及命令参数与上一个命令参数相同
            elif (
                command_name == previous_command
                and command_args == previous_command_args
            ):
                rethink_reason = f"Repititive command detected ({command_name})"

            # 如果有重新考虑的原因
            if rethink_reason:
                logger.info(f"{rethink_reason}, re-thinking with SMART_LLM...")
                # 使用 ExitStack 来管理上下文
                with ExitStack() as stack:

                    @stack.callback
                    def restore_state() -> None:
                        # 在退出 ExitStack 上下文后执行
                        self.config.big_brain = False

                    # 回退当前周期的部分记录
                    self.event_history.rewind()

                    # 切换到 SMART_LLM 并重新思考
                    self.big_brain = True
                    return await self.propose_action(*args, **kwargs)

        # 返回命令名称、命令参数和思考过程
        return command_name, command_args, thoughts
```