# AutoGPT源码解析 2

# `autogpts/autogpt/autogpt/agents/base.py`

这段代码是一个自定义的 Python 类，其作用是定义了一个自动化的 GPT 语言模型的对话类。通过从 `__future__` 模式中使用 `annotations` 属性，该类可以接受未来定义的参数和返回值的说明。

具体来说，该类继承自 `ABC` 类，这是 Python 3.6 中定义的一种用于定义类和函数的模板。此外，它还继承自 `pathlib` 包中的 `Path` 类，用于文件或目录路径的表示。

在该类的内部方法中，定义了一系列抽象方法，这些方法是 GPT 语言模型的对话类所必需的。通过这些方法，可以实现自动化地生成并返回适当的回答，从而实现智能对话的功能。

另外，该类还定义了一个 `AutoGPTPluginTemplate` 类，它是 GPT 语言模型的模板。通过使用这个模板，可以更轻松地定义一个 GPT 语言模型的对话类。


```py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from pydantic import Field, validator

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.core.prompting.base import PromptStrategy
    from autogpt.core.resource.model_providers.schema import (
        ChatModelInfo,
        ChatModelProvider,
        ChatModelResponse,
    )
    from autogpt.models.command_registry import CommandRegistry

```

这段代码是一个人工智能助手，实现了基于用户输入的对话。具体来说，它实现了以下几个功能：

1. 从 `autogpt.agents.utils.prompt_scratchpad` 包中引入了用于获取用户输入的 `PromptScratchpad` 类；
2. 从 `autogpt.config` 包中引入了用于创建配置的 `ConfigBuilder` 类；
3. 从 `autogpt.config.ai_directives` 包中引入了来自 `autogpt.config.ai_profile` 的 `AIDirectives` 类；
4. 从 `autogpt.config.ai_profile` 包中引入了来自 `autogpt.config.ai_directives` 的 `AIDirectives` 类；
5. 从 `autogpt.core.configuration` 包中引入了 `Configurable`、`SystemConfiguration` 和 `SystemSettings` 类；
6. 从 `autogpt.core.prompting.schema` 包中引入了用于构建用户对话的 `ChatMessage`、`ChatPrompt` 和 `CompletionModelFunction` 类；
7. 将所有以上引入的类统一注册，并设置为可配置的实例；
8. 通过 `SystemConfiguration` 和 `SystemSettings` 获取用户输入的来源和设置，设置后，用户可以通过 `PromptScratchpad` 和 `ChatPrompt` 类来提交和获取用户输入；
9. 通过 `CompletionModelFunction` 类实现对用户输入的完整语义解析。


```py
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
```

这段代码是一个基于AutogPT库的Python代码，主要作用是实现了一个OpenAI Chat Model。这个代码的作用是：

1. 从`autogpt.core.resource.model_providers.openai`模块中定义了`OPEN_AI_CHAT_MODELS`和`OpenAIModelName`，这两个变量用于存储OpenAI Chat Model的相关信息。
2. 从`autogpt.core.runner.client_lib.logging.helpers`模块中定义了`dump_prompt`函数，用于在运行时输出提示信息。
3. 从`autogpt.llm.providers.openai`模块中定义了`get_openai_command_specs`函数，用于获取OpenAI命令规格。
4. 从`autogpt.models.action_history`模块中定义了`ActionResult`和`EpisodicActionHistory`，这两个模块用于实现动作历史记录的功能。
5. 从`./utils.agent_file_manager`模块中定义了`AgentFileManager`，这个模块的作用未知。
6. 在`main`函数中，创建了一个`AgentFileManager`实例，然后调用`run`函数，这个函数的作用未知。
7. 最后，将`CommandName`和`CommandArgs`作为参数传递给`dump_prompt`函数，用于在运行时输出提示信息。


```py
from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAIModelName,
)
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.models.action_history import ActionResult, EpisodicActionHistory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

from .utils.agent_file_manager import AgentFileManager

logger = logging.getLogger(__name__)

CommandName = str
CommandArgs = dict[str, str]
```

This is a configuration class for an AI assistant that uses OpenAI Functions, specifically the language model `openai_主要负责人-v01`.

It seems like the configuration includes several constraints and settings for the AI assistant:

1. Limits on the number of cycles the agent can run. The total number of cycles is determined by the `cycle_budget` and the number of cycles running at any given time is limited to a value between `0` and ` cycling_budget / 2`.
2. The number of times the AI assistant has been trained to use OpenAI Functions for this language model. This value is initialized to `0` and incremented by the `use_functions_api` configuration.
3. The AI assistant uses a list of plugins, which are defined as classes that inherit from `AutoGPTPluginTemplate`.
4. The AI assistant has a plotter that allows you to visualize the model's performance metrics, such as speed and accuracy.
5. The AI assistant has several settings for controlling the use of OpenAI Functions, such as the `smart_llm` and `fast_llm` settings for controlling the speed of the language model.
6. The AI assistant has a setting for enabling the use of OpenAI Functions, which is controlled by the `use_functions_api` configuration.


```py
AgentThoughts = dict[str, Any]


class BaseAgentConfiguration(SystemConfiguration):
    allow_fs_access: bool = UserConfigurable(default=False)

    fast_llm: OpenAIModelName = UserConfigurable(default=OpenAIModelName.GPT3_16k)
    smart_llm: OpenAIModelName = UserConfigurable(default=OpenAIModelName.GPT4)
    use_functions_api: bool = UserConfigurable(default=False)

    default_cycle_instruction: str = DEFAULT_TRIGGERING_PROMPT
    """The default instruction passed to the AI for a thinking cycle."""

    big_brain: bool = UserConfigurable(default=False)
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
    def validate_openai_functions(cls, v: bool, values: dict[str, Any]):
        if v:
            smart_llm = values["smart_llm"]
            fast_llm = values["fast_llm"]
            assert all(
                [
                    not any(s in name for s in {"-0301", "-0314"})
                    for name in {smart_llm, fast_llm}
                ]
            ), (
                f"Model {smart_llm} does not support OpenAI Functions. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )
        return v


```

这段代码定义了一个名为 "BaseAgentSettings" 的类，继承自 "SystemSettings" 类，用于配置一个 AI 代理的设置。这个类包含了一些属性，比如代理ID、代理数据目录、AI 配置文件、指令、任务和配置文件。

具体来说，这个类的代理配置由一个 AI 配置文件决定，如果指定了一个 AI 配置文件，那么这个文件将作为该配置文件的默认值。如果没有指定 AI 配置文件，那么将使用默认的 AI 配置，即从 "AutoGPT" 命名空间中获取一个 AI 配置。

此外，这个类包含一个指令文件，用于在运行时传递给代理的指令。这个类的配置文件将保存当前代理的配置，并在保存到 JSON 文件时写入该配置，以便在加载配置文件时使用。

最后，这个类还包含一个方法 "save_to_json_file"，用于将当前代理的配置保存到 JSON 文件中，以及一个方法 "load_from_json_file"，用于从 JSON 文件中加载代理的配置。


```py
class BaseAgentSettings(SystemSettings):
    agent_id: Optional[str] = None
    agent_data_dir: Optional[Path] = None

    ai_profile: AIProfile = Field(default_factory=lambda: AIProfile(ai_name="AutoGPT"))
    """The AI profile or "personality" of the agent."""

    directives: AIDirectives = Field(
        default_factory=lambda: AIDirectives.from_file(
            ConfigBuilder.default_settings.prompt_settings_file
        )
    )
    """Directives (general instructional guidelines) for the agent."""

    task: str = "Terminate immediately"  # FIXME: placeholder for forge.sdk.schema.Task
    """The user-given task that the agent is working on."""

    config: BaseAgentConfiguration = Field(default_factory=BaseAgentConfiguration)
    """The configuration for this BaseAgent subsystem instance."""

    history: EpisodicActionHistory = Field(default_factory=EpisodicActionHistory)
    """(STATE) The action history of the agent."""

    def save_to_json_file(self, file_path: Path) -> None:
        with file_path.open("w") as f:
            f.write(self.json())

    @classmethod
    def load_from_json_file(cls, file_path: Path):
        return cls.parse_file(file_path)


```

This is a class that defines an agent that can interact with a chat model. The agent has a method called `parse_and_process_response()` that is responsible for parsing and processing the response to the chat model.

The `parse_and_process_response()` method takes three arguments: `llm_response`, `prompt`, and `scratchpad`. It returns a `ThoughtProcessOutput` object that contains the parsed command name and command arguments, as well as any agent thoughts.

It is important to note that this class is an abstract base class and should be extended by derived classes that provide the actual implementation of the `parse_and_process_response()` method. This allows for customization of the agent based on its specific role and the requirements of the chat model it is interacting with.


```py
class BaseAgent(Configurable[BaseAgentSettings], ABC):
    """Base class for all AutoGPT agent classes."""

    ThoughtProcessOutput = tuple[CommandName, CommandArgs, AgentThoughts]

    default_settings = BaseAgentSettings(
        name="BaseAgent",
        description=__doc__,
    )

    def __init__(
        self,
        settings: BaseAgentSettings,
        llm_provider: ChatModelProvider,
        prompt_strategy: PromptStrategy,
        command_registry: CommandRegistry,
        legacy_config: Config,
    ):
        self.state = settings
        self.config = settings.config
        self.ai_profile = settings.ai_profile
        self.directives = settings.directives
        self.event_history = settings.history

        self.legacy_config = legacy_config
        """LEGACY: Monolithic application configuration."""

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

        # Support multi-inheritance and mixins for subclasses
        super(BaseAgent, self).__init__()

        logger.debug(f"Created {__class__} '{self.ai_profile.ai_name}'")

    def set_id(self, new_id: str, new_agent_dir: Optional[Path] = None):
        self.state.agent_id = new_id
        if self.state.agent_data_dir:
            if not new_agent_dir:
                raise ValueError(
                    "new_agent_dir must be specified if one is currently configured"
                )
            self.attach_fs(new_agent_dir)

    def attach_fs(self, agent_dir: Path) -> AgentFileManager:
        self.file_manager = AgentFileManager(agent_dir)
        self.file_manager.initialize()
        self.state.agent_data_dir = agent_dir
        return self.file_manager

    @property
    def llm(self) -> ChatModelInfo:
        """The LLM that the agent uses to think."""
        llm_name = (
            self.config.smart_llm if self.config.big_brain else self.config.fast_llm
        )
        return OPEN_AI_CHAT_MODELS[llm_name]

    @property
    def send_token_limit(self) -> int:
        return self.config.send_token_limit or self.llm.max_tokens * 3 // 4

    async def propose_action(self) -> ThoughtProcessOutput:
        """Runs the agent for one cycle.

        Params:
            instruction: The instruction to put at the end of the prompt.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        assert self.file_manager, (
            f"Agent has no FileManager: call {__class__.__name__}.attach_fs()"
            " before trying to run the agent."
        )

        # Scratchpad as surrogate PromptGenerator for plugin hooks
        self._prompt_scratchpad = PromptScratchpad()

        prompt: ChatPrompt = self.build_prompt(scratchpad=self._prompt_scratchpad)
        prompt = self.on_before_think(prompt, scratchpad=self._prompt_scratchpad)

        logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")
        raw_response = await self.llm_provider.create_chat_completion(
            prompt.messages,
            functions=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + list(self._prompt_scratchpad.commands.values())
            if self.config.use_functions_api
            else [],
            model_name=self.llm.name,
        )
        self.config.cycle_count += 1

        return self.on_response(
            llm_response=raw_response,
            prompt=prompt,
            scratchpad=self._prompt_scratchpad,
        )

    @abstractmethod
    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        """Executes the given command, if any, and returns the agent's response.

        Params:
            command_name: The name of the command to execute, if any.
            command_args: The arguments to pass to the command, if any.
            user_input: The user's input, if any.

        Returns:
            The results of the command.
        """
        ...

    def build_prompt(
        self,
        scratchpad: PromptScratchpad,
        extra_commands: Optional[list[CompletionModelFunction]] = None,
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. Message history of the agent, truncated & prepended with running summary as needed
        3. `cycle_instruction`

        Params:
            cycle_instruction: The final instruction for a thinking cycle
        """
        if not extra_commands:
            extra_commands = []
        if not extra_messages:
            extra_messages = []

        # Apply additions from plugins
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            plugin.post_prompt(scratchpad)
        ai_directives = self.directives.copy(deep=True)
        ai_directives.resources += scratchpad.resources
        ai_directives.constraints += scratchpad.constraints
        ai_directives.best_practices += scratchpad.best_practices
        extra_commands += list(scratchpad.commands.values())

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

    def on_before_think(
        self,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ChatPrompt:
        """Called after constructing the prompt but before executing it.

        Calls the `on_planning` hook of any enabled and capable plugins, adding their
        output to the prompt.

        Params:
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The prompt to execute
        """
        current_tokens_used = self.llm_provider.count_message_tokens(
            prompt.messages, self.llm.name
        )
        plugin_count = len(self.config.plugins)
        for i, plugin in enumerate(self.config.plugins):
            if not plugin.can_handle_on_planning():
                continue
            plugin_response = plugin.on_planning(scratchpad, prompt.raw())
            if not plugin_response or plugin_response == "":
                continue
            message_to_add = ChatMessage.system(plugin_response)
            tokens_to_add = self.llm_provider.count_message_tokens(
                message_to_add, self.llm.name
            )
            if current_tokens_used + tokens_to_add > self.send_token_limit:
                logger.debug(f"Plugin response too long, skipping: {plugin_response}")
                logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
                break
            prompt.messages.insert(
                -1, message_to_add
            )  # HACK: assumes cycle instruction to be at the end
            current_tokens_used += tokens_to_add
        return prompt

    def on_response(
        self,
        llm_response: ChatModelResponse,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Called upon receiving a response from the chat model.

        Adds the last/newest message in the prompt and the response to `history`,
        and calls `self.parse_and_process_response()` to do the rest.

        Params:
            llm_response: The raw response from the chat model
            prompt: The prompt that was executed
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """

        return self.parse_and_process_response(
            llm_response,
            prompt,
            scratchpad=scratchpad,
        )

        # TODO: update memory/context

    @abstractmethod
    def parse_and_process_response(
        self,
        llm_response: ChatModelResponse,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Validate, parse & process the LLM's response.

        Must be implemented by derivative classes: no base implementation is provided,
        since the implementation depends on the role of the derivative Agent.

        Params:
            llm_response: The raw response from the chat model
            prompt: The prompt that was executed
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """
        pass

```

# `autogpts/autogpt/autogpt/agents/planning_agent.py`

这段代码是一个自定义的 Python 模块，从未来 3.6 版本开始引入了 annotation。

具体来说，这段代码是用于导入来自未来的一些类型声明，以便在使用时可以明确地声明输入和输出的类型。

如果猜测的话，这段代码可能是用于在程序中定义一些输出或输入类型的变量，以便在程序运行时检查其类型，避免出现错误的类型实例化等。

但是，由于没有上下文，我无法确切地知道这段代码的用途。


```py
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.llm.base import ChatModelResponse, ChatSequence
    from autogpt.memory.vector import VectorMemory
    from autogpt.models.command_registry import CommandRegistry

from autogpt.agents.utils.exceptions import AgentException, InvalidAgentResponseError
from autogpt.json_utils.utilities import extract_dict_from_response, validate_dict
```

这段代码是一个自然语言处理任务中的类，包含了从多个文件中读取句子和实体，并实现了几个与该任务相关的函数和类。

具体来说，这段代码的作用如下：

1. `from autogpt.llm.base import Message` 引入了来自 `autogpt.llm.base` 包的 `Message` 类，这是该任务中用于输出对话信息的基本类。

2. `from autogpt.llm.utils import count_string_tokens` 引入了来自 `autogpt.llm.utils` 包的 `count_string_tokens` 函数，这是用于计算句子中的字符数组的方法。

3. `from autogpt.logs.log_cycle import (` 引入了来自 `autogpt.logs.log_cycle` 包的 `CURRENT_CONTEXT_FILE_NAME` 和 `NEXT_ACTION_FILE_NAME` 类，这些类用于记录当前轮次的动作和上一轮次的提示信息，以及用于指示当前正在处理的文章。

4. `from autogpt.models.action_history import (` 引入了来自 `autogpt.models.action_history` 包的 `ActionErrorResult`、`ActionInterruptedByHuman`、`ActionResult` 和 `ActionSuccessResult` 类，这些类用于记录该任务中的动作和结果。

5. `from autogpt.models.context_item import ContextItem` 引入了来自 `autogpt.models.context_item` 包的 `ContextItem` 类，该类用于记录该任务中的上下文信息。

6. `llm_道歉(text: Any, max_length: int) -> None:` 实现了该任务中的一个函数，用于创建一个用于道歉的文案，并返回是否将其应用到当前句子中。该函数使用了 `count_string_tokens` 函数来计算可能的单词数，然后使用 `Message` 类来创建一个包含该文案的 `Message` 实例。最后，返回 `None`，表示该函数没有做任何有用的事情。

7. `llm_对某个句子作出的行动(action: Action, max_length: int) -> None:` 实现了该任务中的一个函数，用于给定的动作和最大长度，返回是否应用该动作到当前句子中。该函数使用了 `action_history` 包中的类来记录该任务中的动作和结果，并检查该动作是否已经被应用过。最后，返回 `None`，表示该函数没有做任何有用的事情。


```py
from autogpt.llm.base import Message
from autogpt.llm.utils import count_string_tokens
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.models.action_history import (
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
from autogpt.models.context_item import ContextItem

```

This is a class definition for an `Assistant` object that inherits from the `ChatModelApi` class. This `Assistant` class has a PostCommandEvaluationHandler that handles the post-command evaluation of the `command_name` provided by the user.

The PostCommandEvaluationHandler has a method called `evaluate_assistant_response` which takes the response from the `llm_response` object and returns a planning agent output. This method extracts the command name and arguments from the `assistant_reply_dict` object and passes it to the `post_planning` method of the plugin. If the plugin can handle the post-planning, it will be called with the command name and arguments.

If the post-planning fails, the method returns an `InvalidAgentResponseError` with the error message.

If the response is valid, the method extracts the command name and arguments from the `assistant_reply_dict` object and passes it to the `post_planning` method of the plugin.

Finally, the method logs the event using the `log_cycle_handler` method.


```py
from .agent import execute_command, extract_command
from .base import BaseAgent
from .features.context import ContextMixin
from .features.file_workspace import FileWorkspaceMixin

logger = logging.getLogger(__name__)


class PlanningAgent(ContextMixin, FileWorkspaceMixin, BaseAgent):
    """Agent class for interacting with AutoGPT."""

    ThoughtProcessID = Literal["plan", "action", "evaluate"]

    def __init__(
        self,
        command_registry: CommandRegistry,
        memory: VectorMemory,
        triggering_prompt: str,
        config: Config,
        cycle_budget: Optional[int] = None,
    ):
        super().__init__(
            command_registry=command_registry,
            config=config,
            default_cycle_instruction=triggering_prompt,
            cycle_budget=cycle_budget,
        )

        self.memory = memory
        """VectorMemoryProvider used to manage the agent's context (TODO)"""

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

        self.plan: list[str] = []
        """List of steps that the Agent plans to take"""

    def construct_base_prompt(
        self, thought_process_id: ThoughtProcessID, **kwargs
    ) -> ChatSequence:
        prepend_messages = kwargs["prepend_messages"] = kwargs.get(
            "prepend_messages", []
        )

        # Add the current plan to the prompt, if any
        if self.plan:
            plan_section = [
                "## Plan",
                "To complete your task, you have composed the following plan:",
            ]
            plan_section += [f"{i}. {s}" for i, s in enumerate(self.plan, 1)]

            # Add the actions so far to the prompt
            if self.event_history:
                plan_section += [
                    "\n### Progress",
                    "So far, you have executed the following actions based on the plan:",
                ]
                for i, cycle in enumerate(self.event_history, 1):
                    if not (cycle.action and cycle.result):
                        logger.warn(f"Incomplete action in history: {cycle}")
                        continue

                    plan_section.append(
                        f"{i}. You executed the command `{cycle.action.format_call()}`, "
                        f"which gave the result `{cycle.result}`."
                    )

            prepend_messages.append(Message("system", "\n".join(plan_section)))

        if self.context:
            context_section = [
                "## Context",
                "Below is information that may be relevant to your task. These take up "
                "part of your working memory, which is limited, so when a context item is "
                "no longer relevant for your plan, use the `close_context_item` command to "
                "free up some memory."
                "\n",
                self.context.format_numbered(),
            ]
            prepend_messages.append(Message("system", "\n".join(context_section)))

        match thought_process_id:
            case "plan":
                # TODO: add planning instructions; details about what to pay attention to when planning
                pass
            case "action":
                # TODO: need to insert the functions here again?
                pass
            case "evaluate":
                # TODO: insert latest action (with reasoning) + result + evaluation instructions
                pass
            case _:
                raise NotImplementedError(
                    f"Unknown thought process '{thought_process_id}'"
                )

        return super().construct_base_prompt(
            thought_process_id=thought_process_id, **kwargs
        )

    def response_format_instruction(self, thought_process_id: ThoughtProcessID) -> str:
        match thought_process_id:
            case "plan":
                # TODO: add planning instructions; details about what to pay attention to when planning
                response_format = f"""```ts
                interface Response {{
                    thoughts: {{
                        // Thoughts
                        text: string;
                        // A short logical explanation about how the action is part of the earlier composed plan
                        reasoning: string;
                        // Constructive self-criticism
                        criticism: string;
                    }};
                    // A plan to achieve the goals with the available resources and/or commands.
                    plan: Array<{{
                        // An actionable subtask
                        subtask: string;
                        // Criterium to determine whether the subtask has been completed
                        completed_if: string;
                    }}>;
                }}
                ```py"""
                pass
            case "action":
                # TODO: need to insert the functions here again?
                response_format = """```ts
                interface Response {
                    thoughts: {
                        // Thoughts
                        text: string;
                        // A short logical explanation about how the action is part of the earlier composed plan
                        reasoning: string;
                        // Constructive self-criticism
                        criticism: string;
                    };
                    // The action to take, from the earlier specified list of commands
                    command: {
                        name: string;
                        args: Record<string, any>;
                    };
                }
                ```py"""
                pass
            case "evaluate":
                # TODO: insert latest action (with reasoning) + result + evaluation instructions
                response_format = f"""```ts
                interface Response {{
                    thoughts: {{
                        // Thoughts
                        text: string;
                        reasoning: string;
                        // Constructive self-criticism
                        criticism: string;
                    }};
                    result_evaluation: {{
                        // A short logical explanation of why the given partial result does or does not complete the corresponding subtask
                        reasoning: string;
                        // Whether the current subtask has been completed
                        completed: boolean;
                        // An estimate of the progress (0.0 - 1.0) that has been made on the subtask with the actions that have been taken so far
                        progress: float;
                    }};
                }}
                ```py"""
                pass
            case _:
                raise NotImplementedError(
                    f"Unknown thought process '{thought_process_id}'"
                )

        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_format,
        )

        return (
            f"Respond strictly with JSON. The JSON should be compatible with "
            "the TypeScript type `Response` from the following:\n"
            f"{response_format}\n"
        )

    def on_before_think(self, *args, **kwargs) -> ChatSequence:
        prompt = super().on_before_think(*args, **kwargs)

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.cycle_count,
            self.event_history.episodes,
            "event_history.json",
        )
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.cycle_count,
            prompt.raw(),
            CURRENT_CONTEXT_FILE_NAME,
        )
        return prompt

    def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        result: ActionResult

        if command_name == "human_feedback":
            result = ActionInterruptedByHuman(feedback=user_input)
            self.log_cycle_handler.log_cycle(
                self.ai_profile.ai_name,
                self.created_at,
                self.cycle_count,
                user_input,
                USER_INPUT_FILE_NAME,
            )

        else:
            for plugin in self.config.plugins:
                if not plugin.can_handle_pre_command():
                    continue
                command_name, arguments = plugin.pre_command(command_name, command_args)

            try:
                return_value = execute_command(
                    command_name=command_name,
                    arguments=command_args,
                    agent=self,
                )

                # Intercept ContextItem if one is returned by the command
                if type(return_value) == tuple and isinstance(
                    return_value[1], ContextItem
                ):
                    self.context.add(return_value[1])
                    return_value = return_value[0]

                result = ActionSuccessResult(outputs=return_value)
            except AgentException as e:
                result = ActionErrorResult.from_exception(e)

            result_tlength = count_string_tokens(str(result), self.llm.name)
            memory_tlength = count_string_tokens(
                str(self.event_history.fmt_paragraph()), self.llm.name
            )
            if result_tlength + memory_tlength > self.send_token_limit:
                result = ActionErrorResult(
                    reason=f"Command {command_name} returned too much output. "
                    "Do not execute this command again with the same arguments."
                )

            for plugin in self.config.plugins:
                if not plugin.can_handle_post_command():
                    continue
                if result.status == "success":
                    result.outputs = plugin.post_command(command_name, result.outputs)
                elif result.status == "error":
                    result.reason = plugin.post_command(command_name, result.reason)

        return result

    def parse_and_process_response(
        self,
        llm_response: ChatModelResponse,
        thought_process_id: ThoughtProcessID,
        *args,
        **kwargs,
    ) -> PlanningAgent.ThoughtProcessOutput:
        if not llm_response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        response_content = llm_response.content

        for plugin in self.config.plugins:
            if not plugin.can_handle_post_planning():
                continue
            response_content = plugin.post_planning(response_content)

        assistant_reply_dict = extract_dict_from_response(response_content)

        _, errors = validate_dict(assistant_reply_dict, self.config)
        if errors:
            raise InvalidAgentResponseError(
                "Validation of response failed:\n  "
                + ";\n  ".join([str(e) for e in errors])
            )

        # Get command name and arguments
        command_name, arguments = extract_command(
            assistant_reply_dict, llm_response, self.config
        )
        response = command_name, arguments, assistant_reply_dict

        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.cycle_count,
            assistant_reply_dict,
            NEXT_ACTION_FILE_NAME,
        )
        return response

```

# `autogpts/autogpt/autogpt/agents/__init__.py`

这段代码定义了一个Python classes的序列，包括：from .agent import Agent, from .base import AgentThoughts, BaseAgent, CommandArgs, CommandName, __all__ = ["BaseAgent", "Agent", "CommandName", "CommandArgs", "AgentThoughts"]。

其中，from .agent import Agent和from .base import AgentThoughts定义了两个类：Agent和AgentThoughts，分别用于实现机器人和思考功能。

另外，from .base import AgentThoughts, BaseAgent, CommandArgs, CommandName定义了四个类：BaseAgent，用于提供基本的机器人功能；CommandArgs和CommandName，用于定义命令参数和命令名称；AgentThoughts，用于提供机器人的思考功能。

最后，__all__ = ["BaseAgent", "Agent", "CommandName", "CommandArgs", "AgentThoughts"]定义了一个序列，将上述所有类都包含在内。


```py
from .agent import Agent
from .base import AgentThoughts, BaseAgent, CommandArgs, CommandName

__all__ = ["BaseAgent", "Agent", "CommandName", "CommandArgs", "AgentThoughts"]

```

# `autogpts/autogpt/autogpt/agents/features/context.py`

这段代码是一个类 `AgentContext` 的定义，用于表示一个具有 `ContextItem` 类型的上下文对象。上下文对象包含一个列表 `items`，可以用于与 ChatPrompt 和 ChatMessage 类进行交互。

具体来说，这段代码从两个来源引入了以下类和函数：

- `from autogpt.core.prompting import ChatPrompt` 和 `from autogpt.models.context_item import ContextItem`。这两个类用于从 ChatMessage 类中获取消息和上下文信息。
- `from typing import TYPE_CHECKING, Any, Optional`。这个语句用于定义变量 `items` 的类型参数，可以是一个或多个 `ContextItem` 对象。
- `__init__`、`__bool__`、`__contains__`、`add`、`close` 和 `clear` 函数，用于初始化、检查对象是否包含特定元素、添加上下文信息、添加消息和关闭对话。
- `format_numbered` 函数，用于将所有包含数字的上下文信息格式化为类似于 `"<pre>{}\n<source>{}\n"` 的字符串。


```py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from autogpt.core.prompting import ChatPrompt
    from autogpt.models.context_item import ContextItem

    from ..base import BaseAgent

from autogpt.core.resource.model_providers import ChatMessage


class AgentContext:
    items: list[ContextItem]

    def __init__(self, items: Optional[list[ContextItem]] = None):
        self.items = items or []

    def __bool__(self) -> bool:
        return len(self.items) > 0

    def __contains__(self, item: ContextItem) -> bool:
        return any([i.source == item.source for i in self.items])

    def add(self, item: ContextItem) -> None:
        self.items.append(item)

    def close(self, index: int) -> None:
        self.items.pop(index - 1)

    def clear(self) -> None:
        self.items.clear()

    def format_numbered(self) -> str:
        return "\n\n".join([f"{i}. {c.fmt()}" for i, c in enumerate(self.items, 1)])


```

这段代码定义了一个名为 ContextMixin 的类，用于在 BaseAgent 子类中添加对上下文的支持。在 mixin 的初始化方法中，创建了一个名为 `AgentContext` 的上下文对象，并将其设置为实例的初始化参数。

在 `build_prompt` 方法中，如果传递了任何额外的 `extra_messages` 参数，则将其添加到了 `extra_messages` 列表中。否则，会默认添加一个包含系统消息的列表。

最后，在 `__init__` 方法中，调用 `super()` 方法来确保继承自 BaseAgent 的类也具有 `__init__` 方法。这样做是因为 `ContextMixin` 方法中添加的上下文支持需要在继承自 `BaseAgent` 的类中才能正常工作。


```py
class ContextMixin:
    """Mixin that adds context support to a BaseAgent subclass"""

    context: AgentContext

    def __init__(self, **kwargs: Any):
        self.context = AgentContext()

        super(ContextMixin, self).__init__(**kwargs)

    def build_prompt(
        self,
        *args: Any,
        extra_messages: Optional[list[ChatMessage]] = None,
        **kwargs: Any,
    ) -> ChatPrompt:
        if not extra_messages:
            extra_messages = []

        # Add context section to prompt
        if self.context:
            extra_messages.insert(
                0,
                ChatMessage.system(
                    "## Context\n"
                    + self.context.format_numbered()
                    + "\n\nWhen a context item is no longer needed and you are not done yet,"
                    " you can hide the item by specifying its number in the list above"
                    " to `hide_context_item`.",
                ),
            )

        return super(ContextMixin, self).build_prompt(
            *args,
            extra_messages=extra_messages,
            **kwargs,
        )  # type: ignore


```

这段代码定义了一个名为 `get_agent_context` 的函数，它接受一个名为 `agent` 的参数，并返回一个名为 `AgentContext` 或 `None` 的对象。

函数内部首先检查输入参数 `agent` 是否属于一个名为 `ContextMixin` 的类。如果是，就返回该 `Agent` 对象对应的 `context` 属性；如果不是，则返回 `None`。

简而言之，该函数的作用是获取一个 `Agent` 对象对应的上下文（context），如果该对象已经定义了上下文，则返回上下文，否则返回 `None`。


```py
def get_agent_context(agent: BaseAgent) -> AgentContext | None:
    if isinstance(agent, ContextMixin):
        return agent.context

    return None

```

# `autogpts/autogpt/autogpt/agents/features/file_workspace.py`

这段代码是一个用于将 workspace 支持添加到某个类中的 mixin。通过从 `pathlib` 包中导入 `BaseAgent` 和一个自定义的 `FileWorkspace` 类，实现了一个 workspace 的初始化以及将其添加到 agent 中。

具体来说，代码首先通过 `super()` 调用父类的初始化方法，传递所有的 keyword arguments，确保子类正确地继承了父类的所有方法。然后，代码读取一个 `BaseAgentConfiguration` 类型的参数，如果该参数不是 `FileWorkspaceConfiguration` 的话，会抛出一个 `ValueError`。

接着，代码读取一个 `AgentFileManager` 对象，这是 `FileWorkspace` 中非常重要的一个组件。然后，代码调用 `__init__` 方法中的另一个参数 `config` 和 `file_manager` 来设置 workspace 和 file_manager。

最后，代码通过 `_setup_workspace` 函数将 workspace 和 config 进行初始化，并返回初始化结果。

在 `attach_fs` 方法中，代码首先调用父类的 `attach_fs` 方法，并传入 agent 的 directory。然后，代码创建一个新的 `FileWorkspace` 对象，并将它设置为当前 workspace 的实例。这样，agent 就可以访问 workspace 中定义的文件和目录了。


```py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ..base import BaseAgent

from autogpt.file_workspace import FileWorkspace

from ..base import AgentFileManager, BaseAgentConfiguration


class FileWorkspaceMixin:
    """Mixin that adds workspace support to a class"""

    workspace: FileWorkspace = None
    """Workspace that the agent has access to, e.g. for reading/writing files."""

    def __init__(self, **kwargs):
        # Initialize other bases first, because we need the config from BaseAgent
        super(FileWorkspaceMixin, self).__init__(**kwargs)

        config: BaseAgentConfiguration = getattr(self, "config")
        if not isinstance(config, BaseAgentConfiguration):
            raise ValueError(
                "Cannot initialize Workspace for Agent without compatible .config"
            )
        file_manager: AgentFileManager = getattr(self, "file_manager")
        if not file_manager:
            return

        self.workspace = _setup_workspace(file_manager, config)

    def attach_fs(self, agent_dir: Path):
        res = super(FileWorkspaceMixin, self).attach_fs(agent_dir)

        self.workspace = _setup_workspace(self.file_manager, self.config)

        return res


```

该代码定义了两个函数，分别是`_setup_workspace`和`get_agent_workspace`。它们的作用如下：

1. `_setup_workspace`函数的作用是创建一个File Workspace对象，将其存储在当前工作房的根目录下，并设置为只允许访问文件系统的根目录。然后调用该对象的`initialize()`方法来使它处于准备就绪状态，最后返回该对象。
2. `get_agent_workspace`函数的作用是获取当前Agent对象所属的工作房对象，如果当前Agent对象属于FileWorkspaceMixin类，则返回该对象的workspace对象；否则返回None。

这两个函数共同组成了一个完整的File Workspace应用，可以用来管理代理文件和存储它们的配置信息。


```py
def _setup_workspace(file_manager: AgentFileManager, config: BaseAgentConfiguration):
    workspace = FileWorkspace(
        file_manager.root / "workspace",
        restrict_to_root=not config.allow_fs_access,
    )
    workspace.initialize()
    return workspace


def get_agent_workspace(agent: BaseAgent) -> FileWorkspace | None:
    if isinstance(agent, FileWorkspaceMixin):
        return agent.workspace

    return None

```

# `autogpts/autogpt/autogpt/agents/features/watchdog.py`

这段代码是一个自定义的 Python 类，旨在实现一个带有上下文记录的会话代表器。这个会话代表器用于存储一个行动历史，并使用上下文来跟踪上下文。

具体来说，这个会话代表器实现了两个主要的函数：`get_agent_configuration`和`get_action_history`。这两个函数的实现都依赖于一个名为 `BaseAgent` 的接口类。

`get_agent_configuration`函数允许您设置或获取一个 `BaseAgent` 实例的上下文记录。上下文记录是一个包含了多个键值对的元组，每个键表示一个特定的上下文记录，而每个值则是一个字典，其中包含与上下文记录相关的信息。

`get_action_history`函数允许您获取一个行动历史。它使用 `ExitStack` 类来存储当前会话的上下文，并在您调用它时使用上下文记录中的信息来构建行动历史。

此外，这个会话代表器还实现了两个保护函数：`assert_agent_configuration`和`assert_action_history`。这些函数用于确保您在访问 `get_agent_configuration` 和 `get_action_history` 函数时使用了正确的上下文记录。


```py
from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import BaseAgentConfiguration

from autogpt.models.action_history import EpisodicActionHistory

from ..base import BaseAgent

logger = logging.getLogger(__name__)


```

This is a class called `WatchdogMixin` that inherits from `BaseAgent`. It appears to be a mixin of several other classes, including `BaseAgent`, `AnotherClass`, `SmartLLM`, `ExitStack`, and `ProductiveMixin`.

The `WatchdogMixin` class seems to be responsible for proposing actions to an agent. It has a method called `propose_action`, which takes in arguments and returns an output of type `BaseAgent.ThoughtProcessOutput`.

It appears that the method can only be used with `BaseAgent` derivatives, which seems to be a subclass of `ProductiveMixin`. This might be because `BaseAgent` and `ProductiveMixin` both inherit from `ProductiveMixin`, and `WatchdogMixin` extends from `ProductiveMixin` to add some additional functionality.

The `WatchdogMixin` class also seems to have a dependency on the `BigBrain` configuration, which seems to be a boolean value that indicates whether the agent should use SMART or regular�aw. If the `BigBrain` is `False`, the class performs actions using the regular�aw model, and if it is `True`, the class uses the SMART model.


```py
class WatchdogMixin:
    """
    Mixin that adds a watchdog feature to an agent class. Whenever the agent starts
    looping, the watchdog will switch from the FAST_LLM to the SMART_LLM and re-think.
    """

    config: BaseAgentConfiguration
    event_history: EpisodicActionHistory

    def __init__(self, **kwargs) -> None:
        # Initialize other bases first, because we need the event_history from BaseAgent
        super(WatchdogMixin, self).__init__(**kwargs)

        if not isinstance(self, BaseAgent):
            raise NotImplementedError(
                f"{__class__.__name__} can only be applied to BaseAgent derivatives"
            )

    async def propose_action(self, *args, **kwargs) -> BaseAgent.ThoughtProcessOutput:
        command_name, command_args, thoughts = await super(
            WatchdogMixin, self
        ).propose_action(*args, **kwargs)

        if not self.config.big_brain and self.config.fast_llm != self.config.smart_llm:
            previous_command, previous_command_args = None, None
            if len(self.event_history) > 1:
                # Detect repetitive commands
                previous_cycle = self.event_history.episodes[
                    self.event_history.cursor - 1
                ]
                previous_command = previous_cycle.action.name
                previous_command_args = previous_cycle.action.args

            rethink_reason = ""

            if not command_name:
                rethink_reason = "AI did not specify a command"
            elif (
                command_name == previous_command
                and command_args == previous_command_args
            ):
                rethink_reason = f"Repititive command detected ({command_name})"

            if rethink_reason:
                logger.info(f"{rethink_reason}, re-thinking with SMART_LLM...")
                with ExitStack() as stack:

                    @stack.callback
                    def restore_state() -> None:
                        # Executed after exiting the ExitStack context
                        self.config.big_brain = False

                    # Remove partial record of current cycle
                    self.event_history.rewind()

                    # Switch to SMART_LLM and re-think
                    self.big_brain = True
                    return await self.propose_action(*args, **kwargs)

        return command_name, command_args, thoughts

```

# `autogpts/autogpt/autogpt/agents/prompt_strategies/one_shot.py`

这段代码是一个Python语言的函数，其中包含了一些未来主义导入(from __future__ import annotations)，表明它从未来会导入一些新的功能或类型。

具体来说，这段代码做以下事情：

1. 从标准库(stdlib)中导入了一个名为Logger的类，用于在日志中记录信息。
2. 从标准库(stdlib)中导入了一个名为json的类，用于将JSON格式的数据转换为Python格式的数据。
3. 从标准库(stdlib)中导入了一个名为re的类，用于解析和校正字符串，以便进行一些文本处理操作。
4. 从typing import类型注释(TYPE_CHECKING)，用于强制类型断言，确保函数能够正确地接收和使用参数。
5. 从typing import Callable，用于定义一个函数，使得它可以被参数调用。
6. 从typing import Optional，用于定义一个Optional类型，可以是一个非空值，也可以是空的。
7. 从distro模块中导入了一个名为Agents的类，用于管理机器学习代理。
8. 从pydantic模块中导入了一个名为Field的类，用于定义JSON数据的字段。


```py
from __future__ import annotations

import json
import platform
import re
from logging import Logger
from typing import TYPE_CHECKING, Callable, Optional

import distro
from pydantic import Field

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.models.action_history import Episode

```

这段代码是一个自定义的Python程序，它的目的是在使用Autogpt库时处理异常情况。这个程序定义了一个名为InvalidAgentResponseError的异常类，用于表示当一个Agent在响应AIDirective时出现异常的情况。

这个程序还定义了两个与Autogpt配置相关的类，AIDirectives和AIProfile。AIDirectives类是一个包含AIDirective的配置类，而AIProfile类是一个包含AI Profile的配置类。

此外，这个程序还定义了一个系统配置类SystemConfiguration，以及一个用户可配置类UserConfigurable。系统配置类中包含一些与Prompting和ResourceModelProviders相关的类，如ChatPrompt、LanguageModelClassification和CompletionModelFunction。而用户可配置类中包含一个与AIDirectives和AIProfile类的实例。

最后，这个程序还定义了一些来自Autogpt库的函数，如extract_dict_from_response，用于从Autogpt库的响应中提取JSON数据。


```py
from autogpt.agents.utils.exceptions import InvalidAgentResponseError
from autogpt.config import AIDirectives, AIProfile
from autogpt.core.configuration.schema import SystemConfiguration, UserConfigurable
from autogpt.core.prompting import (
    ChatPrompt,
    LanguageModelClassification,
    PromptStrategy,
)
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.json_utils.utilities import extract_dict_from_response
```

This is a Python class that defines a `UserConfigurable` object for a JSON Schema.

This class allows users to customize the object by providing their own values for certain properties. These properties are defined as `UserConfigurable` which means that the default values for these properties are provided by the `to_dict()` method of the `DEFAULT_RESPONSE_SCHEMA` class.

The `UserConfigurable` class has several methods:

* `body_template`: A string containing a template for the user's body message.
* `response_schema`: A dictionary containing the JSON Schema definition for the user's response.
* `choose_action_instruction`: A string indicating whether the user should be prompted to choose an action or not.
* `use_functions_api`: A boolean indicating whether to use the user's chosen action or the default action.

It also has several methods for accessing the current state of the conversation:

* `get_current_score`: returns the current score of the game.
* `get_response_schema`: returns the JSON Schema definition for the user's response.
* `get_choose_action_instruction`: returns the user's choice for actionInstruction.

Note that the `UserConfigurable` class is designed to be extended by other classes to provide additional functionality.


```py
from autogpt.prompts.utils import format_numbered_list, indent


class OneShotAgentPromptConfiguration(SystemConfiguration):
    DEFAULT_BODY_TEMPLATE: str = (
        "## Constraints\n"
        "You operate within the following constraints:\n"
        "{constraints}\n"
        "\n"
        "## Resources\n"
        "You can leverage access to the following resources:\n"
        "{resources}\n"
        "\n"
        "## Commands\n"
        "You have access to the following commands:\n"
        "{commands}\n"
        "\n"
        "## Best practices\n"
        "{best_practices}"
    )

    DEFAULT_CHOOSE_ACTION_INSTRUCTION: str = (
        "Determine exactly one command to use next based on the given goals "
        "and the progress you have made so far, "
        "and respond using the JSON schema specified previously:"
    )

    DEFAULT_RESPONSE_SCHEMA = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "thoughts": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                required=True,
                properties={
                    "text": JSONSchema(
                        description="Thoughts",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "reasoning": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "plan": JSONSchema(
                        description="Short markdown-style bullet list that conveys the long-term plan",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "criticism": JSONSchema(
                        description="Constructive self-criticism",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "speak": JSONSchema(
                        description="Summary of thoughts, to say to user",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                },
            ),
            "command": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                required=True,
                properties={
                    "name": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "args": JSONSchema(
                        type=JSONSchema.Type.OBJECT,
                        required=True,
                    ),
                },
            ),
        },
    )

    body_template: str = UserConfigurable(default=DEFAULT_BODY_TEMPLATE)
    response_schema: dict = UserConfigurable(
        default_factory=DEFAULT_RESPONSE_SCHEMA.to_dict
    )
    choose_action_instruction: str = UserConfigurable(
        default=DEFAULT_CHOOSE_ACTION_INSTRUCTION
    )
    use_functions_api: bool = UserConfigurable(default=False)

    #########
    # State #
    #########
    # progress_summaries: dict[tuple[int, int], str] = Field(
    #     default_factory=lambda: {(0, 0): ""}
    # )


```

This is a class definition for an `Agent` that uses the text-based assisting AI platform `Aurora`. This class provides methods for generating responses to user commands, validating the responses against a response schema, and extracting the command name and arguments from the `assistant_reply_dict`.

The `parse_response_content` method is the entry point for the `Agent` class and handles the parsing of the response content from the `assistant_reply_dict`, extracting the command name and arguments, and validating the response against a predefined schema. The method returns an instance of the `ThoughtProcessOutput` class, which represents the parsed response.

The `generate_budget_constraint`, `generate_commands_list`, and `generate_response_content` methods provide additional functionality for generating responses, extracting command information, and validating the response content, respectively.


```py
class OneShotAgentPromptStrategy(PromptStrategy):
    default_configuration: OneShotAgentPromptConfiguration = (
        OneShotAgentPromptConfiguration()
    )

    def __init__(
        self,
        configuration: OneShotAgentPromptConfiguration,
        logger: Logger,
    ):
        self.config = configuration
        self.response_schema = JSONSchema.from_dict(configuration.response_schema)
        self.logger = logger

    @property
    def model_classification(self) -> LanguageModelClassification:
        return LanguageModelClassification.FAST_MODEL  # FIXME: dynamic switching

    def build_prompt(
        self,
        *,
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        event_history: list[Episode],
        include_os_info: bool,
        max_prompt_tokens: int,
        count_tokens: Callable[[str], int],
        count_message_tokens: Callable[[ChatMessage | list[ChatMessage]], int],
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. Message history of the agent, truncated & prepended with running summary as needed
        3. `cycle_instruction`

        Params:
            cycle_instruction: The final instruction for a thinking cycle
        """
        if not extra_messages:
            extra_messages = []

        system_prompt = self.build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )
        system_prompt_tlength = count_message_tokens(ChatMessage.system(system_prompt))

        user_task = f'"""{task}"""'
        user_task_tlength = count_message_tokens(ChatMessage.user(user_task))

        response_format_instr = self.response_format_instruction(
            self.config.use_functions_api
        )
        extra_messages.append(ChatMessage.system(response_format_instr))

        final_instruction_msg = ChatMessage.user(self.config.choose_action_instruction)
        final_instruction_tlength = count_message_tokens(final_instruction_msg)

        if event_history:
            progress = self.compile_progress(
                event_history,
                count_tokens=count_tokens,
                max_tokens=(
                    max_prompt_tokens
                    - system_prompt_tlength
                    - user_task_tlength
                    - final_instruction_tlength
                    - count_message_tokens(extra_messages)
                ),
            )
            extra_messages.insert(
                0,
                ChatMessage.system(f"## Progress\n\n{progress}"),
            )

        prompt = ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(user_task),
                *extra_messages,
                final_instruction_msg,
            ],
        )

        return prompt

    def build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> str:
        system_prompt_parts = (
            self._generate_intro_prompt(ai_profile)
            + (self._generate_os_info() if include_os_info else [])
            + [
                self.config.body_template.format(
                    constraints=format_numbered_list(
                        ai_directives.constraints
                        + self._generate_budget_constraint(ai_profile.api_budget)
                    ),
                    resources=format_numbered_list(ai_directives.resources),
                    commands=self._generate_commands_list(commands),
                    best_practices=format_numbered_list(ai_directives.best_practices),
                )
            ]
            + [
                "## Your Task\n"
                "The user will specify a task for you to execute, in triple quotes,"
                " in the next message. Your job is to complete the task while following"
                " your directives as given above, and terminate when your task is done."
            ]
        )

        # Join non-empty parts together into paragraph format
        return "\n\n".join(filter(None, system_prompt_parts)).strip("\n")

    def compile_progress(
        self,
        episode_history: list[Episode],
        max_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        if max_tokens and not count_tokens:
            raise ValueError("count_tokens is required if max_tokens is set")

        steps: list[str] = []
        tokens: int = 0
        start: int = len(episode_history)

        for i, c in reversed(list(enumerate(episode_history))):
            step = f"### Step {i+1}: Executed `{c.action.format_call()}`\n"
            step += f'- **Reasoning:** "{c.action.reasoning}"\n'
            step += (
                f"- **Status:** `{c.result.status if c.result else 'did_not_finish'}`\n"
            )
            if c.result:
                if c.result.status == "success":
                    result = str(c.result)
                    result = "\n" + indent(result) if "\n" in result else result
                    step += f"- **Output:** {result}"
                elif c.result.status == "error":
                    step += f"- **Reason:** {c.result.reason}\n"
                    if c.result.error:
                        step += f"- **Error:** {c.result.error}\n"
                elif c.result.status == "interrupted_by_human":
                    step += f"- **Feedback:** {c.result.feedback}\n"

            if max_tokens and count_tokens:
                step_tokens = count_tokens(step)
                if tokens + step_tokens > max_tokens:
                    break
                tokens += step_tokens

            steps.insert(0, step)
            start = i

        # TODO: summarize remaining

        part = slice(0, start)

        return "\n\n".join(steps)

    def response_format_instruction(self, use_functions_api: bool) -> str:
        response_schema = self.response_schema.copy(deep=True)
        if (
            use_functions_api
            and response_schema.properties
            and "command" in response_schema.properties
        ):
            del response_schema.properties["command"]

        # Unindent for performance
        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface("Response"),
        )

        return (
            f"Respond strictly with a JSON object{' containing your thoughts, and a function_call specifying the next command to use' if use_functions_api else ''}. "
            "The JSON object should be compatible with the TypeScript type `Response` from the following:\n"
            f"{response_format}"
        )

    def _generate_intro_prompt(self, ai_profile: AIProfile) -> list[str]:
        """Generates the introduction part of the prompt.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        return [
            f"You are {ai_profile.ai_name}, {ai_profile.ai_role.rstrip('.')}.",
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications.",
        ]

    def _generate_os_info(self) -> list[str]:
        """Generates the OS information part of the prompt.

        Params:
            config (Config): The configuration object.

        Returns:
            str: The OS information part of the prompt.
        """
        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != "Linux"
            else distro.name(pretty=True)
        )
        return [f"The OS you are running on is: {os_info}"]

    def _generate_budget_constraint(self, api_budget: float) -> list[str]:
        """Generates the budget information part of the prompt.

        Returns:
            list[str]: The budget information part of the prompt, or an empty list.
        """
        if api_budget > 0.0:
            return [
                f"It takes money to let you run. "
                f"Your API budget is ${api_budget:.3f}"
            ]
        return []

    def _generate_commands_list(self, commands: list[CompletionModelFunction]) -> str:
        """Lists the commands available to the agent.

        Params:
            agent: The agent for which the commands are being listed.

        Returns:
            str: A string containing a numbered list of commands.
        """
        try:
            return format_numbered_list([cmd.fmt_line() for cmd in commands])
        except AttributeError:
            self.logger.warn(f"Formatting commands failed. {commands}")
            raise

    def parse_response_content(
        self,
        response: AssistantChatMessageDict,
    ) -> Agent.ThoughtProcessOutput:
        if "content" not in response:
            raise InvalidAgentResponseError("Assistant response has no text content")

        assistant_reply_dict = extract_dict_from_response(response["content"])

        _, errors = self.response_schema.validate_object(
            object=assistant_reply_dict,
            logger=self.logger,
        )
        if errors:
            raise InvalidAgentResponseError(
                "Validation of response failed:\n  "
                + ";\n  ".join([str(e) for e in errors])
            )

        # Get command name and arguments
        command_name, arguments = extract_command(
            assistant_reply_dict, response, self.config.use_functions_api
        )
        return command_name, arguments, assistant_reply_dict


```

The command name and arguments can be parsed from the `assistant_reply_json` object if the `use_openai_functions_api` flag is `True`. If `use_openai_functions_api` is `True`, the function name and arguments can be extracted from the `function_call` field in the `assistant_reply` object. If the function name or arguments are not present in the `assistant_reply_json` object, a `JsonDecodeError` will be raised.


```py
#############
# Utilities #
#############


def extract_command(
    assistant_reply_json: dict,
    assistant_reply: AssistantChatMessageDict,
    use_openai_functions_api: bool,
) -> tuple[str, dict[str, str]]:
    """Parse the response and return the command name and arguments

    Args:
        assistant_reply_json (dict): The response object from the AI
        assistant_reply (ChatModelResponse): The model response from the AI
        config (Config): The config object

    Returns:
        tuple: The command name and arguments

    Raises:
        json.decoder.JSONDecodeError: If the response is not valid JSON

        Exception: If any other error occurs
    """
    if use_openai_functions_api:
        if "function_call" not in assistant_reply:
            raise InvalidAgentResponseError("No 'function_call' in assistant reply")
        assistant_reply_json["command"] = {
            "name": assistant_reply["function_call"]["name"],
            "args": json.loads(assistant_reply["function_call"]["arguments"]),
        }
    try:
        if not isinstance(assistant_reply_json, dict):
            raise InvalidAgentResponseError(
                f"The previous message sent was not a dictionary {assistant_reply_json}"
            )

        if "command" not in assistant_reply_json:
            raise InvalidAgentResponseError("Missing 'command' object in JSON")

        command = assistant_reply_json["command"]
        if not isinstance(command, dict):
            raise InvalidAgentResponseError("'command' object is not a dictionary")

        if "name" not in command:
            raise InvalidAgentResponseError("Missing 'name' field in 'command' object")

        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        return command_name, arguments

    except json.decoder.JSONDecodeError:
        raise InvalidAgentResponseError("Invalid JSON")

    except Exception as e:
        raise InvalidAgentResponseError(str(e))

```