# `.\AutoGPT\autogpts\autogpt\autogpt\agents\agent.py`

```py
# 导入必要的模块和类
from __future__ import annotations
import inspect
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

# 如果是类型检查，则导入特定模块
if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.models.command_registry import CommandRegistry

# 导入 pydantic 模块中的 Field 类
from pydantic import Field

# 导入自定义模块和类
from autogpt.core.configuration import Configurable
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import (
    AssistantChatMessage,
    ChatMessage,
    ChatModelProvider,
)
from autogpt.llm.api_manager import ApiManager
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs.utils import fmt_kwargs
from autogpt.models.action_history import (
    Action,
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
from autogpt.models.command import CommandOutput
from autogpt.models.context_item import ContextItem

# 导入自定义的基类和特性类
from .base import BaseAgent, BaseAgentConfiguration, BaseAgentSettings
from .features.context import ContextMixin
from .features.file_workspace import FileWorkspaceMixin
from .features.watchdog import WatchdogMixin
from .prompt_strategies.one_shot import (
    OneShotAgentPromptConfiguration,
    OneShotAgentPromptStrategy,
)
from .utils.exceptions import (
    AgentException,
    AgentTerminated,
    CommandExecutionError,
    UnknownCommandError,
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义 AgentConfiguration 类，继承自 BaseAgentConfiguration
class AgentConfiguration(BaseAgentConfiguration):
    pass

# 定义 AgentSettings 类，继承自 BaseAgentSettings
class AgentSettings(BaseAgentSettings):
    # 定义 config 属性，类型为 AgentConfiguration，默认值为 AgentConfiguration 的实例
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)
    # 定义 prompt_config 属性，类型为 OneShotAgentPromptConfiguration，默认值为 OneShotAgentPromptStrategy 的默认配置
    prompt_config: OneShotAgentPromptConfiguration = Field(
        default_factory=(
            lambda: OneShotAgentPromptStrategy.default_configuration.copy(deep=True)
        )
    )

# 定义 Agent 类，继承自 ContextMixin、FileWorkspaceMixin、WatchdogMixin、BaseAgent 和 Configurable[AgentSettings]
class Agent(
    ContextMixin,
    FileWorkspaceMixin,
    WatchdogMixin,
    BaseAgent,
    Configurable[AgentSettings],
):
    """AutoGPT's primary Agent; uses one-shot prompting."""
    # AutoGPT的主要Agent；使用一次性提示

    default_settings: AgentSettings = AgentSettings(
        name="Agent",
        description=__doc__,
    )
    # 设置默认设置为AgentSettings对象，包括名称和描述

    prompt_strategy: OneShotAgentPromptStrategy
    # 声明prompt_strategy变量为OneShotAgentPromptStrategy类型

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: ChatModelProvider,
        command_registry: CommandRegistry,
        legacy_config: Config,
    ):
        # 初始化函数，接受AgentSettings、ChatModelProvider、CommandRegistry和Config作为参数

        prompt_strategy = OneShotAgentPromptStrategy(
            configuration=settings.prompt_config,
            logger=logger,
        )
        # 创建OneShotAgentPromptStrategy对象，使用settings.prompt_config和logger作为参数

        super().__init__(
            settings=settings,
            llm_provider=llm_provider,
            prompt_strategy=prompt_strategy,
            command_registry=command_registry,
            legacy_config=legacy_config,
        )
        # 调用父类的初始化函数，传入相应参数

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""
        # 记录Agent创建的时间戳，仅用于结构化调试日志

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""
        # 创建LogCycleHandler对象，用于结构化调试日志

    def build_prompt(
        self,
        *args,
        extra_messages: Optional[list[ChatMessage]] = None,
        include_os_info: Optional[bool] = None,
        **kwargs,
        # 定义build_prompt方法，接受可选参数extra_messages和include_os_info
    # 定义一个方法，返回一个ChatPrompt对象
    ) -> ChatPrompt:
        # 如果没有额外消息，则将额外消息列表设置为空列表
        if not extra_messages:
            extra_messages = []

        # 在额外消息列表中添加当前时间和日期的系统消息
        extra_messages.append(
            ChatMessage.system(f"The current time and date is {time.strftime('%c')}"),
        )

        # 添加预算信息（如果有的话）到提示中
        api_manager = ApiManager()
        if api_manager.get_total_budget() > 0.0:
            remaining_budget = (
                api_manager.get_total_budget() - api_manager.get_total_cost()
            )
            if remaining_budget < 0:
                remaining_budget = 0

            # 创建包含剩余API预算信息的系统消息
            budget_msg = ChatMessage.system(
                f"Your remaining API budget is ${remaining_budget:.3f}"
                + (
                    " BUDGET EXCEEDED! SHUT DOWN!\n\n"
                    if remaining_budget == 0
                    else " Budget very nearly exceeded! Shut down gracefully!\n\n"
                    if remaining_budget < 0.005
                    else " Budget nearly exceeded. Finish up.\n\n"
                    if remaining_budget < 0.01
                    else ""
                ),
            )
            # 记录调试信息
            logger.debug(budget_msg)
            # 将预算信息添加到额外消息列表中
            extra_messages.append(budget_msg)

        # 如果include_os_info为None，则根据legacy_config.execute_local_commands的值来设置include_os_info
        if include_os_info is None:
            include_os_info = self.legacy_config.execute_local_commands

        # 调用父类的build_prompt方法，返回ChatPrompt对象
        return super().build_prompt(
            *args,
            extra_messages=extra_messages,
            include_os_info=include_os_info,
            **kwargs,
        )

    # 在思考之前执行的方法，返回一个ChatPrompt对象
    def on_before_think(self, *args, **kwargs) -> ChatPrompt:
        # 调用父类的on_before_think方法，返回一个ChatPrompt对象
        prompt = super().on_before_think(*args, **kwargs)

        # 重置日志计数器，并记录日志
        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            prompt.raw(),
            CURRENT_CONTEXT_FILE_NAME,
        )
        # 返回ChatPrompt对象
        return prompt
    # 解析和处理助手聊天消息的响应
    def parse_and_process_response(
        self, llm_response: AssistantChatMessage, *args, **kwargs
    ) -> Agent.ThoughtProcessOutput:
        # 遍历配置中的插件
        for plugin in self.config.plugins:
            # 如果插件无法处理后续规划，则跳过
            if not plugin.can_handle_post_planning():
                continue
            # 调用插件的后续规划方法，更新消息内容
            llm_response.content = plugin.post_planning(llm_response.content or "")

        # 解析响应内容，获取命令名称、参数和助手回复字典
        (
            command_name,
            arguments,
            assistant_reply_dict,
        ) = self.prompt_strategy.parse_response_content(llm_response)

        # 记录循环处理日志
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            assistant_reply_dict,
            NEXT_ACTION_FILE_NAME,
        )

        # 如果存在命令名称，则注册动作到事件历史
        if command_name:
            self.event_history.register_action(
                Action(
                    name=command_name,
                    args=arguments,
                    reasoning=assistant_reply_dict["thoughts"]["reasoning"],
                )
            )

        # 返回命令名称、参数和助手回复字典
        return command_name, arguments, assistant_reply_dict

    # 执行命令
    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
# 异步执行命令并返回结果
async def execute_command(
    # 命令名称
    command_name: str,
    # 命令参数
    arguments: dict[str, str],
    # 执行命令的代理
    agent: Agent,
) -> CommandOutput:
    """Execute the command and return the result

    Args:
        command_name (str): The name of the command to execute
        arguments (dict): The arguments for the command
        agent (Agent): The agent that is executing the command

    Returns:
        str: The result of the command
    """
    # 如果存在具有相同名称或别名的本地命令，则执行该命令
    if command := agent.command_registry.get_command(command_name):
        try:
            # 执行命令并获取结果
            result = command(**arguments, agent=agent)
            # 如果结果是可等待对象，则等待并返回结果
            if inspect.isawaitable(result):
                return await result
            return result
        except AgentException:
            raise
        except Exception as e:
            raise CommandExecutionError(str(e))

    # 处理非本地命令（例如来自插件）
    if agent._prompt_scratchpad:
        for name, command in agent._prompt_scratchpad.commands.items():
            if (
                command_name == name
                or command_name.lower() == command.description.lower()
            ):
                try:
                    return command.method(**arguments)
                except AgentException:
                    raise
                except Exception as e:
                    raise CommandExecutionError(str(e))

    # 抛出未知命令错误
    raise UnknownCommandError(
        f"Cannot execute command '{command_name}': unknown command."
    )
```