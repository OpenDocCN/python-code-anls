# `.\AutoGPT\autogpts\autogpt\autogpt\agents\base.py`

```py
# 导入必要的模块和类
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from pydantic import Field, validator

# 如果是类型检查，则导入以下模块
if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.core.prompting.base import PromptStrategy
    from autogpt.core.resource.model_providers.schema import (
        AssistantChatMessage,
        ChatModelInfo,
        ChatModelProvider,
        ChatModelResponse,
    )
    from autogpt.models.command_registry import CommandRegistry

# 导入自定义模块
from autogpt.agents.utils.prompt_scratchpad import PromptScratchpad
from autogpt.config import ConfigBuilder
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.prompting.schema import (
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAIModelName,
)
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.models.action_history import ActionResult, EpisodicActionHistory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

# 导入自定义模块
from .utils.agent_file_manager import AgentFileManager

# 设置日志记录器
logger = logging.getLogger(__name__)

# 定义类型别名
CommandName = str
CommandArgs = dict[str, str]
AgentThoughts = dict[str, Any]

# 定义基础代理配置类，继承自系统配置类
class BaseAgentConfiguration(SystemConfiguration):
    # 是否允许文件系统访问，默认为False
    allow_fs_access: bool = UserConfigurable(default=False)

    # 快速LLM模型名称，默认为GPT3_16k
    fast_llm: OpenAIModelName = UserConfigurable(default=OpenAIModelName.GPT3_16k)
    # 智能LLM模型名称，默认为GPT4
    smart_llm: OpenAIModelName = UserConfigurable(default=OpenAIModelName.GPT4)
    # 是否使用函数API，默认为False
    use_functions_api: bool = UserConfigurable(default=False)
    default_cycle_instruction: str = DEFAULT_TRIGGERING_PROMPT
    """The default instruction passed to the AI for a thinking cycle."""

    big_brain: bool = UserConfigurable(default=True)
    """
    Whether this agent uses the configured smart LLM (default) to think,
    as opposed to the configured fast LLM. Enabling this disables hybrid mode.
    """

    cycle_budget: Optional[int] = 1
    """
    The number of cycles that the agent is allowed to run unsupervised.

    `None` for unlimited continuous execution,
    `1` to require user approval for every step,
    `0` to stop the agent.
    """

    cycles_remaining = cycle_budget
    """The number of cycles remaining within the `cycle_budget`."""

    cycle_count = 0
    """The number of cycles that the agent has run since its initialization."""

    send_token_limit: Optional[int] = None
    """
    The token limit for prompt construction. Should leave room for the completion;
    defaults to 75% of `llm.max_tokens`.
    """

    summary_max_tlength: Optional[int] = None
    # TODO: move to ActionHistoryConfiguration

    plugins: list[AutoGPTPluginTemplate] = Field(default_factory=list, exclude=True)

    class Config:
        arbitrary_types_allowed = True  # Necessary for plugins

    @validator("plugins", each_item=True)
    def validate_plugins(cls, p: AutoGPTPluginTemplate | Any):
        assert issubclass(
            p.__class__, AutoGPTPluginTemplate
        ), f"{p} does not subclass AutoGPTPluginTemplate"
        assert (
            p.__class__.__name__ != "AutoGPTPluginTemplate"
        ), f"Plugins must subclass AutoGPTPluginTemplate; {p} is a template instance"
        return p

    @validator("use_functions_api")
    # 静态方法，用于验证是否支持 OpenAI Functions
    def validate_openai_functions(cls, v: bool, values: dict[str, Any]):
        # 如果 v 为真，则进行验证
        if v:
            # 获取 smart_llm 和 fast_llm 的数值
            smart_llm = values["smart_llm"]
            fast_llm = values["fast_llm"]
            # 断言 smart_llm 和 fast_llm 中不包含 "-0301" 或 "-0314"
            assert all(
                [
                    not any(s in name for s in {"-0301", "-0314"})
                    for name in {smart_llm, fast_llm}
                ]
            ), (
                # 如果断言失败，抛出异常
                f"Model {smart_llm} does not support OpenAI Functions. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )
        # 返回验证结果
        return v
# 定义一个名为 BaseAgentSettings 的类，继承自 SystemSettings
class BaseAgentSettings(SystemSettings):
    # 代理 ID，字符串类型，默认为空字符串
    agent_id: str = ""
    # 代理数据目录的路径，可选的 Path 类型，默认为 None
    agent_data_dir: Optional[Path] = None

    # AIProfile 对象，表示代理的 AI 个性，默认为 AutoGPT
    ai_profile: AIProfile = Field(default_factory=lambda: AIProfile(ai_name="AutoGPT"))
    """The AI profile or "personality" of the agent."""

    # AIDirectives 对象，代理的指令集，默认从配置文件中加载
    directives: AIDirectives = Field(
        default_factory=lambda: AIDirectives.from_file(
            ConfigBuilder.default_settings.prompt_settings_file
        )
    )
    """Directives (general instructional guidelines) for the agent."""

    # 代理的任务，字符串类型，默认为"Terminate immediately"，需要替换为 forge.sdk.schema.Task
    task: str = "Terminate immediately"  # FIXME: placeholder for forge.sdk.schema.Task
    """The user-given task that the agent is working on."""

    # BaseAgentConfiguration 对象，代理的配置信息，默认为空配置
    config: BaseAgentConfiguration = Field(default_factory=BaseAgentConfiguration)
    """The configuration for this BaseAgent subsystem instance."""

    # EpisodicActionHistory 对象，代理的行为历史记录
    history: EpisodicActionHistory = Field(default_factory=EpisodicActionHistory)
    """(STATE) The action history of the agent."""

    # 将代理的设置保存到 JSON 文件中
    def save_to_json_file(self, file_path: Path) -> None:
        with file_path.open("w") as f:
            f.write(self.json())

    # 从 JSON 文件中加载代理的设置
    @classmethod
    def load_from_json_file(cls, file_path: Path):
        return cls.parse_file(file_path)


# 定义一个名为 BaseAgent 的类，继承自 Configurable[BaseAgentSettings] 和 ABC
class BaseAgent(Configurable[BaseAgentSettings], ABC):
    """Base class for all AutoGPT agent classes."""

    # 定义一个元组类型 ThoughtProcessOutput，包含 CommandName、CommandArgs 和 AgentThoughts 三个元素
    ThoughtProcessOutput = tuple[CommandName, CommandArgs, AgentThoughts]

    # 默认的 BaseAgentSettings 对象，包含代理的名称和描述信息
    default_settings = BaseAgentSettings(
        name="BaseAgent",
        description=__doc__,
    )

    # 初始化方法，接受代理的设置、ChatModelProvider、PromptStrategy、CommandRegistry 和 Config 作为参数
    def __init__(
        self,
        settings: BaseAgentSettings,
        llm_provider: ChatModelProvider,
        prompt_strategy: PromptStrategy,
        command_registry: CommandRegistry,
        legacy_config: Config,
    # 初始化 BaseAgent 类的实例，设置状态、配置、AI 档案、指令和事件历史等属性
    ):
        self.state = settings
        self.config = settings.config
        self.ai_profile = settings.ai_profile
        self.directives = settings.directives
        self.event_history = settings.history

        self.legacy_config = legacy_config
        """LEGACY: Monolithic application configuration."""

        # 根据 settings.agent_data_dir 创建 AgentFileManager 实例，如果未配置 agent_data_dir 则为 None
        self.file_manager: AgentFileManager = (
            AgentFileManager(settings.agent_data_dir)
            if settings.agent_data_dir
            else None
        )  # type: ignore

        self.llm_provider = llm_provider

        self.prompt_strategy = prompt_strategy

        self.command_registry = command_registry
        """The registry containing all commands available to the agent."""

        self._prompt_scratchpad: PromptScratchpad | None = None

        # 支持多继承和混入，调用父类 BaseAgent 的初始化方法
        super(BaseAgent, self).__init__()

        logger.debug(f"Created {__class__} '{self.ai_profile.ai_name}'")

    # 设置 agent 的 ID，同时可以指定新的 agent 目录
    def set_id(self, new_id: str, new_agent_dir: Optional[Path] = None):
        self.state.agent_id = new_id
        if self.state.agent_data_dir:
            if not new_agent_dir:
                raise ValueError(
                    "new_agent_dir must be specified if one is currently configured"
                )
            self.attach_fs(new_agent_dir)

    # 将 agent 附加到指定的 agent 目录
    def attach_fs(self, agent_dir: Path) -> AgentFileManager:
        self.file_manager = AgentFileManager(agent_dir)
        self.file_manager.initialize()
        self.state.agent_data_dir = agent_dir
        return self.file_manager

    @property
    def llm(self) -> ChatModelInfo:
        """The LLM that the agent uses to think."""
        # 根据配置选择使用的 LLM 模型
        llm_name = (
            self.config.smart_llm if self.config.big_brain else self.config.fast_llm
        )
        return OPEN_AI_CHAT_MODELS[llm_name]

    @property
    def send_token_limit(self) -> int:
        # 返回发送令牌的限制数量，如果未配置则为 llm 最大令牌数的 3/4
        return self.config.send_token_limit or self.llm.max_tokens * 3 // 4
    # 异步方法，用于提出下一个要执行的动作，基于任务和当前状态
    async def propose_action(self) -> ThoughtProcessOutput:
        """Proposes the next action to execute, based on the task and current state.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        # 断言确保 self.file_manager 不为空，否则抛出异常提示
        assert self.file_manager, (
            f"Agent has no FileManager: call {__class__.__name__}.attach_fs()"
            " before trying to run the agent."
        )

        # 作为插件钩子的 PromptGenerator 的替代 Scratchpad
        self._prompt_scratchpad = PromptScratchpad()

        # 构建聊天提示
        prompt: ChatPrompt = self.build_prompt(scratchpad=self._prompt_scratchpad)
        # 在思考之前触发的钩子
        prompt = self.on_before_think(prompt, scratchpad=self._prompt_scratchpad)

        # 记录日志，打印执行的提示内容
        logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")
        # 使用 LLM 提供者创建聊天完成，获取响应
        response = await self.llm_provider.create_chat_completion(
            prompt.messages,
            functions=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + list(self._prompt_scratchpad.commands.values())
            if self.config.use_functions_api
            else [],
            model_name=self.llm.name,
            # 完成解析器，用于解析和处理响应
            completion_parser=lambda r: self.parse_and_process_response(
                r,
                prompt,
                scratchpad=self._prompt_scratchpad,
            ),
        )
        # 周期计数加一
        self.config.cycle_count += 1

        # 在响应上触发的钩子，返回结果
        return self.on_response(
            llm_response=response,
            prompt=prompt,
            scratchpad=self._prompt_scratchpad,
        )

    # 抽象方法，用于执行命令
    @abstractmethod
    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    # 定义一个方法，用于执行给定的命令（如果有的话），并返回代理的响应
    def execute_command(self, command_name: str, command_args: Optional[dict] = None, user_input: str = None) -> ActionResult:
        """Executes the given command, if any, and returns the agent's response.

        Params:
            command_name: The name of the command to execute, if any.
            command_args: The arguments to pass to the command, if any.
            user_input: The user's input, if any.

        Returns:
            ActionResult: An object representing the result(s) of the command.
        """
        ...

    # 定义一个方法，用于构建提示信息
    def build_prompt(
        self,
        scratchpad: PromptScratchpad,
        extra_commands: Optional[list[CompletionModelFunction]] = None,
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs a prompt using `self.prompt_strategy`.

        Params:
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)
            extra_commands: Additional commands that the agent has access to.
            extra_messages: Additional messages to include in the prompt.
        """
        # 如果没有额外的命令，将额外命令列表初始化为空
        if not extra_commands:
            extra_commands = []
        # 如果没有额外的消息，将额外消息列表初始化为空
        if not extra_messages:
            extra_messages = []

        # 应用插件的附加内容
        for plugin in self.config.plugins:
            # 如果插件无法处理后续提示，则跳过
            if not plugin.can_handle_post_prompt():
                continue
            # 调用插件的 post_prompt 方法，将 scratchpad 作为参数传入
            plugin.post_prompt(scratchpad)
        # 复制当前的指令对象
        ai_directives = self.directives.copy(deep=True)
        # 将 scratchpad 中的资源、约束和最佳实践添加到指令对象中
        ai_directives.resources += scratchpad.resources
        ai_directives.constraints += scratchpad.constraints
        ai_directives.best_practices += scratchpad.best_practices
        # 将 scratchpad 中的命令值添加到额外命令列表中
        extra_commands += list(scratchpad.commands.values())

        # 构建提示，使用 self.prompt_strategy 的 build_prompt 方法
        prompt = self.prompt_strategy.build_prompt(
            task=self.state.task,
            ai_profile=self.ai_profile,
            ai_directives=ai_directives,
            commands=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + extra_commands,
            event_history=self.event_history,
            max_prompt_tokens=self.send_token_limit,
            count_tokens=lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            count_message_tokens=lambda x: self.llm_provider.count_message_tokens(
                x, self.llm.name
            ),
            extra_messages=extra_messages,
            **extras,
        )

        return prompt

    # 在思考之前执行的方法
    def on_before_think(
        self,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ChatPrompt:
        """定义一个方法，用于在执行提示之前调用插件的on_planning钩子，并将它们的输出添加到提示中。

        参数:
            prompt: 准备执行的提示。
            scratchpad: 一个对象，用于插件写入额外的提示元素。
                (例如命令、约束、最佳实践)

        返回:
            要执行的提示
        """
        # 计算当前已使用的令牌数量
        current_tokens_used = self.llm_provider.count_message_tokens(
            prompt.messages, self.llm.name
        )
        # 获取插件数量
        plugin_count = len(self.config.plugins)
        # 遍历插件列表
        for i, plugin in enumerate(self.config.plugins):
            # 如果插件无法处理on_planning，则跳过
            if not plugin.can_handle_on_planning():
                continue
            # 调用插件的on_planning方法
            plugin_response = plugin.on_planning(scratchpad, prompt.raw())
            # 如果插件响应为空，则跳过
            if not plugin_response or plugin_response == "":
                continue
            # 创建系统消息对象
            message_to_add = ChatMessage.system(plugin_response)
            # 计算要添加的令牌数量
            tokens_to_add = self.llm_provider.count_message_tokens(
                message_to_add, self.llm.name
            )
            # 如果当前令牌使用量加上要添加的令牌数量超过发送令牌限制，则跳过该插件
            if current_tokens_used + tokens_to_add > self.send_token_limit:
                logger.debug(f"Plugin response too long, skipping: {plugin_response}")
                logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
                break
            # 将消息添加到提示中
            prompt.messages.insert(
                -1, message_to_add
            )  # HACK: assumes cycle instruction to be at the end
            current_tokens_used += tokens_to_add
        # 返回提示
        return prompt

    def on_response(
        self,
        llm_response: ChatModelResponse,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Called upon receiving a response from the chat model.

        Calls `self.parse_and_process_response()`.

        Params:
            llm_response: The raw response from the chat model.
            prompt: The prompt that was executed.
            scratchpad: An object containing additional prompt elements from plugins.
                (E.g. commands, constraints, best practices)

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """

        return llm_response.parsed_result

        # TODO: update memory/context

    @abstractmethod
    def parse_and_process_response(
        self,
        llm_response: AssistantChatMessage,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Validate, parse & process the LLM's response.

        Must be implemented by derivative classes: no base implementation is provided,
        since the implementation depends on the role of the derivative Agent.

        Params:
            llm_response: The raw response from the chat model.
            prompt: The prompt that was executed.
            scratchpad: An object containing additional prompt elements from plugins.
                (E.g. commands, constraints, best practices)

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """
        pass
```