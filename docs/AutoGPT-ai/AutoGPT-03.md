# AutoGPT源码解析 3

# `autogpts/autogpt/autogpt/agents/utils/agent_file_manager.py`

这段代码是一个名为 `AgentFileManager` 的类，它用于表示一个支持 AutoGPT 模型的 agents 的 workspace（工作区）。这个 workspace 包含一个根目录和一个初始化方法。

具体来说，这段代码的作用如下：

1. 初始化一个名为 `AgentFileManager` 的类实例，并设置其 `agent_data_dir` 属性为包含 AutoGPT 模型的数据目录。

2. 定义一个 `__init__` 方法，该方法接受一个 `agent_data_dir` 参数，并将其赋值给实例的 `_root` 属性，以便在需要时可以访问数据目录。

3. 定义一个 `root` 方法，该方法返回 workspace 的根目录。

4. 定义一个 `state_file_path` 属性，该属性返回一个代表 workspace 中状态文件的路径。

5. 定义一个 `file_ops_log_path` 属性，该属性返回一个代表 workspace 中日志文件的路径。

6. 定义一个名为 `init_file_ops_log` 的静态方法，该方法接受一个 `file_logger_path` 参数，并将其写入一个只读的文件中。

7. 可以调用 `AgentFileManager` 的 `initialize` 方法来初始化 workspace，该方法会创建一个空 workspace，并初始化其中的文件操作日志。


```py
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentFileManager:
    """A class that represents a workspace for an AutoGPT agent."""

    def __init__(self, agent_data_dir: Path):
        self._root = agent_data_dir.resolve()

    @property
    def root(self) -> Path:
        """The root directory of the workspace."""
        return self._root

    def initialize(self) -> None:
        self.root.mkdir(exist_ok=True, parents=True)
        self.init_file_ops_log(self.file_ops_log_path)

    @property
    def state_file_path(self) -> Path:
        return self.root / "state.json"

    @property
    def file_ops_log_path(self) -> Path:
        return self.root / "file_logger.log"

    @staticmethod
    def init_file_ops_log(file_logger_path: Path) -> Path:
        if not file_logger_path.exists():
            with file_logger_path.open(mode="w", encoding="utf-8") as f:
                f.write("")
        return file_logger_path

```

# `autogpts/autogpt/autogpt/agents/utils/exceptions.py`

这段代码定义了一个名为 `AgentException` 的自定义异常类，继承自广泛的 `Exception` 类。这个异常类有两个参数：`message` 和 `hint`。

`message` 参数是一个字符串，表示 agent 异常的错误消息。它是当异常信息在没有 `hint` 参数时，从参数列表中获取的默认值。如果 `hint` 参数被传入了值，那么 `message` 参数将覆盖 `hint` 的值。

`hint` 参数是一个可选的字符串，用于提供更多的信息，以帮助老师和机器学习模型(如 LLM)理解异常情况。它是在 `__init__` 方法中设置的，当 `hint` 参数被传入了值时，将覆盖 `message` 参数。如果没有提供 `hint` 参数，或者 `hint` 参数的值为空字符串，那么将不会创建任何额外的日志或错误信息。

最后，在 `__init__` 方法中，使用 `super().__init__(message, *args)` 来调用父异常类的基本 `__init__` 方法，以确保agent异常类可以正确地继承自父异常类。


```py
from typing import Optional


class AgentException(Exception):
    """Base class for specific exceptions relevant in the execution of Agents"""

    message: str

    hint: Optional[str] = None
    """A hint which can be passed to the LLM to reduce reoccurrence of this error"""

    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)


```

这段代码定义了三个不同类型的 `AgentException` 异常类：

1. `AgentTerminated`：表示代理程序已经结束或者被正常终止，通常用于程序结束时打印日志信息。
2. `ConfigurationError`：表示由于无效、不兼容或错误的配置而引发的异常。例如，当代理程序尝试使用错误的参数、配置或环境时，会引发此异常。
3. `InvalidAgentResponseError`：表示由于LLM的响应格式与预期不符而引发的异常。此异常通常用于代理程序返回非预期的响应或无法响应时。

以上异常类均从 `AgentException` 类继承而来，并可能包含一个或多个 `hint` 属性，用于在发生异常时提供更多的信息或提示。


```py
class AgentTerminated(AgentException):
    """The agent terminated or was terminated"""


class ConfigurationError(AgentException):
    """Error caused by invalid, incompatible or otherwise incorrect configuration"""


class InvalidAgentResponseError(AgentException):
    """The LLM deviated from the prescribed response format"""


class UnknownCommandError(AgentException):
    """The AI tried to use an unknown command"""

    hint = "Do not try to use this command again."


```

这段代码定义了一个异常类 `DuplicateOperationError`，它继承自 `AgentException` 类。这个异常类被用来表示一个已经执行过的提议操作。

接着，定义了另一个异常类 `CommandExecutionError`，它也继承自 `AgentException` 类。这个异常类被用来表示一个在尝试执行命令时发生的事件。

然后，定义了一个继承自 `CommandExecutionError` 的异常类 `InvalidArgumentError`，它被用来表示一个命令在接收到了无效参数时发生的事件。

最后，在 `InvalidArgumentError` 的基础上，定义了一个继承自 `CommandExecutionError` 的异常类 `OperationNotAllowedError`，它被用来表示一个提议操作在代理程序不允许执行时发生的事件。


```py
class DuplicateOperationError(AgentException):
    """The proposed operation has already been executed"""


class CommandExecutionError(AgentException):
    """An error occurred when trying to execute the command"""


class InvalidArgumentError(CommandExecutionError):
    """The command received an invalid argument"""


class OperationNotAllowedError(CommandExecutionError):
    """The agent is not allowed to execute the proposed operation"""


```

这段代码定义了三个异常类，它们都是CommandExecutionError的子类。这些异常类描述了在尝试运行 arbitrary code（非特权代码）时发生的错误情况。

具体来说，第一个异常类是AccessDeniedError，它表示在访问必要资源时被拒绝，例如尝试读取一个不存在的文件或执行一个不存在的命令。第二个异常类是CodeExecutionError，它表示在尝试运行一段 arbitrary code时发生了错误，例如代码有语法错误、缺少必要的库或模块等等。第三个异常类是TooMuchOutputError，它表示在尝试运行一段 arbitrary code时，该代码输出的内容超出了Agent（代理程序）处理的最大输出量，导致Agent无法处理。

这些异常类可以作为异常在程序中抛出，从而让程序更加鲁棒。当程序在执行某些可能危险的操作时，这些异常类可以帮助程序检测出哪些操作是不可行的，并能够提供更加详细的错误信息，有助于程序员进行调试和修复。


```py
class AccessDeniedError(CommandExecutionError):
    """The operation failed because access to a required resource was denied"""


class CodeExecutionError(CommandExecutionError):
    """The operation (an attempt to run arbitrary code) returned an error"""


class TooMuchOutputError(CommandExecutionError):
    """The operation generated more output than what the Agent can process"""

```

# `autogpts/autogpt/autogpt/agents/utils/prompt_scratchpad.py`

这段代码定义了一个名为 `CallableCompletionModelFunction` 的类，它实现了 `CompletionModelFunction` 接口。这个接口可能是某种用于自动完成文本输入操作的 API 的一部分。

这个类的 `method` 属性是一个通用的方法，可以接收一个字符串参数和一个远似完成语义的动作名称。它返回一个布尔值，表示是否完成了动作。

此代码将导入自 `logging` 模块，以及从 `typing` 模块中导入一个名为 `Callable` 的类型注。还将从 `pydantic` 模块中导入 `BaseModel` 和 `Field` 类型注。

另外，从 `autogpt.core.resource.model_providers.schema` 模块中导入 `CompletionModelFunction` 类型注，以及从 `autogpt.core.utils.json_schema` 模块中导入 `JSONSchema` 类型注。

最后，定义了一个名为 `logger` 的 logger 实例，用于输出消息。


```py
import logging
from typing import Callable

from pydantic import BaseModel, Field

from autogpt.core.resource.model_providers.schema import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger("PromptScratchpad")


class CallableCompletionModelFunction(CompletionModelFunction):
    method: Callable


```

This is a Python class that manages a command-line interface (CLI) and a set of parameters.

The CLI has a function called `execute_command` that takes a command name and a list of parameters as inputs. It then tries to execute the command using the `function` method, which should be defined in the `CommandFunction` class. If the command has any validation errors or the function cannot execute the command, it returns `None`. Otherwise, it returns the result of the function.

The CLI also has a method called `add_resource` that adds a resource to the `resources` list. It also has a method called `add_best_practice` that adds a best practice to the `best_practices` list.

To use the CLI, you would first need to import it and then call the `execute_command` function with the desired command and parameters. For example:
```py
cli = MyCLI()
cli.execute_command("my_command", ["--param1", "--param2"])
```
This would execute the `my_command` function with the `--param1` and `--param2` parameters.


```py
class PromptScratchpad(BaseModel):
    commands: dict[str, CallableCompletionModelFunction] = Field(default_factory=dict)
    resources: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    best_practices: list[str] = Field(default_factory=list)

    def add_constraint(self, constraint: str) -> None:
        """
        Add a constraint to the constraints list.

        Params:
            constraint (str): The constraint to be added.
        """
        if constraint not in self.constraints:
            self.constraints.append(constraint)

    def add_command(
        self,
        name: str,
        description: str,
        params: dict[str, str | dict],
        function: Callable,
    ) -> None:
        """
        Registers a command.

        *Should only be used by plugins.* Native commands should be added
        directly to the CommandRegistry.

        Params:
            name (str): The name of the command (e.g. `command_name`).
            description (str): The description of the command.
            params (dict, optional): A dictionary containing argument names and their
              types. Defaults to an empty dictionary.
            function (callable, optional): A callable function to be called when
                the command is executed. Defaults to None.
        """
        for p, s in params.items():
            invalid = False
            if type(s) == str and s not in JSONSchema.Type._value2member_map_:
                invalid = True
                logger.warning(
                    f"Cannot add command '{name}':"
                    f" parameter '{p}' has invalid type '{s}'."
                    f" Valid types are: {JSONSchema.Type._value2member_map_.keys()}"
                )
            elif isinstance(s, dict):
                try:
                    JSONSchema.from_dict(s)
                except KeyError:
                    invalid = True
            if invalid:
                return

        command = CallableCompletionModelFunction(
            name=name,
            description=description,
            parameters={
                name: JSONSchema(type=JSONSchema.Type._value2member_map_[spec])
                if type(spec) == str
                else JSONSchema.from_dict(spec)
                for name, spec in params.items()
            },
            method=function,
        )

        if name in self.commands:
            if description == self.commands[name].description:
                return
            logger.warning(
                f"Replacing command {self.commands[name]} with conflicting {command}"
            )
        self.commands[name] = command

    def add_resource(self, resource: str) -> None:
        """
        Add a resource to the resources list.

        Params:
            resource (str): The resource to be added.
        """
        if resource not in self.resources:
            self.resources.append(resource)

    def add_best_practice(self, best_practice: str) -> None:
        """
        Add an item to the list of best practices.

        Params:
            best_practice (str): The best practice item to be added.
        """
        if best_practice not in self.best_practices:
            self.best_practices.append(best_practice)

```

# `autogpts/autogpt/autogpt/agent_factory/configurators.py`

这段代码是一个人工智能助手，可以接受一个任务，根据用户提供的 AI 配置文件，学习一个或多个指令，并创建一个或多个新的机器人。它使用了一个名为 ChatModelProvider 的类来加载预训练的模型，还使用了一组日志配置文件。代码中还包括一些帮助函数和一些日志输出函数，以帮助调试和输出日志信息。


```py
from typing import Optional

from autogpt.agent_manager import AgentManager
from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIDirectives, AIProfile, Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.logs.config import configure_chat_plugins
from autogpt.logs.helpers import print_attribute
from autogpt.models.command_registry import CommandRegistry
from autogpt.plugins import scan_plugins


def create_agent(
    task: str,
    ai_profile: AIProfile,
    app_config: Config,
    llm_provider: ChatModelProvider,
    directives: Optional[AIDirectives] = None,
) -> Agent:
    if not task:
        raise ValueError("No task specified for new agent")
    if not directives:
        directives = AIDirectives.from_file(app_config.prompt_settings_file)

    agent = _configure_agent(
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=app_config,
        llm_provider=llm_provider,
    )

    agent.state.agent_id = AgentManager.generate_id(agent.ai_profile.ai_name)

    return agent


```

这段代码定义了一个名为 `configure_agent_with_state` 的函数，它接受三个参数：`state`、`app_config` 和 `llm_provider`。

函数首先创建一个名为 `_configure_agent` 的函数，它接受四个参数：`state`、`app_config`、`llm_provider` 和 `task`。

`_configure_agent` 函数使用 `scan_plugins` 函数扫描 `app_config` 的插件，并使用 `configure_chat_plugins` 函数配置聊天插件。

接下来，函数创建一个名为 `command_registry` 的对象，并使用 `with_command_modules` 方法扫描 `app_config` 的命令模块。

然后，函数根据传入的 `task`、`ai_profile` 和 `directives` 参数，创建一个 `Agent` 对象。

接下来，函数使用 `create_agent_state` 函数创建一个 `Agent` 的状态对象，如果传入的状态对象不为空，则将其设置为 `Agent` 的状态。

最后，函数使用 `print_attribute` 函数打印 `app_config` 的 `selenium_web_browser` 属性，然后将其返回。

函数返回一个名为 `Agent` 的对象，其中包括 `settings`、`llm_provider` 和 `command_registry`。


```py
def configure_agent_with_state(
    state: AgentSettings,
    app_config: Config,
    llm_provider: ChatModelProvider,
) -> Agent:
    return _configure_agent(
        state=state,
        app_config=app_config,
        llm_provider=llm_provider,
    )


def _configure_agent(
    app_config: Config,
    llm_provider: ChatModelProvider,
    task: str = "",
    ai_profile: Optional[AIProfile] = None,
    directives: Optional[AIDirectives] = None,
    state: Optional[AgentSettings] = None,
) -> Agent:
    if not (state or task and ai_profile and directives):
        raise TypeError(
            "Either (state) or (task, ai_profile, directives) must be specified"
        )

    app_config.plugins = scan_plugins(app_config, app_config.debug_mode)
    configure_chat_plugins(app_config)

    # Create a CommandRegistry instance and scan default folder
    command_registry = CommandRegistry.with_command_modules(
        modules=COMMAND_CATEGORIES,
        config=app_config,
    )

    agent_state = state or create_agent_state(
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=app_config,
    )

    # TODO: configure memory

    print_attribute("Configured Browser", app_config.selenium_web_browser)

    return Agent(
        settings=agent_state,
        llm_provider=llm_provider,
        command_registry=command_registry,
        legacy_config=app_config,
    )


```

这段代码定义了一个名为 create_agent_state 的函数，它接受四个参数：任务（str）、AI 配置文件（AIProfile）、指令（AIDirectives）和应用程序配置（Config）。它返回一个名为 AgentSettings 的类实例，该实例包含了定义了 agent 在执行任务过程中需要的信息和设置。

首先，函数的第一个参数是任务的名称（name），第二个参数是任务的描述（description），第三个参数是 AI 配置文件（ai_profile），第四个参数是指令（directives）。这些参数将用于定义 agent 在执行任务时需要的信息和设置。

然后，函数的第五个参数是应用程序配置（app_config），它是一个 Config 类的实例，包含了允许 agent 使用 OpenAI 函数的功能设置。

接下来，函数的第六个参数是 agent_prompt_config，它是一个复制自 default_settings.prompt_config 的 Agent.default_settings 类的实例，它使用了 deep=True 的深拷贝，从复制中继承了配置文件中的所有设置，包括 use\_functions\_api 和 plugins。

最后，函数的最后一个参数是历史记录（history），它是一个从 default\_settings.history 中拷贝的子类实例，包含了 agent 在执行任务过程中所保存的所有历史记录。

函数返回的 AgentSettings 类实例包含了以下几个方法：

* name：设置 agent 的名称。
* description：设置 agent 的描述。
* task：获取或设置 agent 在任务中需要执行的任务。
* ai_profile：获取或设置 agent 的 AI 配置文件。
* directions：获取或设置 agent 的指令。
* config：获取或设置 agent 的应用程序配置。
* prompt_config：获取或设置 agent 的 prompt 配置。
* history：获取 agent 所保存的历史记录。

通过这些方法的组合，函数可以确保 agent 在执行任务时具有所需的设置和信息。


```py
def create_agent_state(
    task: str,
    ai_profile: AIProfile,
    directives: AIDirectives,
    app_config: Config,
) -> AgentSettings:
    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = app_config.openai_functions

    return AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        config=AgentConfiguration(
            fast_llm=app_config.fast_llm,
            smart_llm=app_config.smart_llm,
            allow_fs_access=not app_config.restrict_to_workspace,
            use_functions_api=app_config.openai_functions,
            plugins=app_config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )

```

# `autogpts/autogpt/autogpt/agent_factory/generators.py`

这段代码是一个异步函数，它使用了一个特殊类型：`TYPE_CHECKING`。如果`TYPE_CHECKING`为`True`，则这段代码将在运行时执行。如果没有`TYPE_CHECKING`，则这段代码将直接跳过。

接下来，它导入了两个异步包：`autogpt.agents.agent` 和 `autogpt.config.ai_directives`。

接着，它从两个异步包中导入了两个函数：`generate_agent_profile_for_task` 和 `_configure_agent`。

`generate_agent_profile_for_task`函数接受三个参数：

* `task`：当前任务
* `app_config`：应用程序配置，包括`prompt_settings_file`属性
* `llm_provider`：对话模型提供者，它需要一个`ChatModelProvider`实例

这个函数使用`generate_agent_profile_for_task`函数生成一个AI代理的配置对象，然后使用`_configure_agent`函数将配置对象和任务直接合并，最后返回一个`Agent`实例。

`_configure_agent`函数接受四个参数：

* `task`：当前任务
* `ai_profile`：AI代理的配置对象，它需要一个`ChatModelProvider`实例作为参数
* `directives`：一个包含当前任务和任务依赖关系的列表，这些关系将决定如何生成AI代理的指令
* `app_config`：应用程序配置，包括`prompt_settings_file`属性
* `llm_provider`：对话模型提供者，它需要一个`ChatModelProvider`实例作为参数

这个函数使用`generate_agent_profile_for_task`函数生成一个AI代理的配置对象，然后使用`directives`参数将当前任务和任务依赖关系作为指令，最后使用`_configure_agent`函数将配置对象和任务合并，并返回一个`Agent`实例。


```py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.config import Config
    from autogpt.core.resource.model_providers.schema import ChatModelProvider

from autogpt.config.ai_directives import AIDirectives

from .configurators import _configure_agent
from .profile_generator import generate_agent_profile_for_task


async def generate_agent_for_task(
    task: str,
    app_config: "Config",
    llm_provider: "ChatModelProvider",
) -> "Agent":
    base_directives = AIDirectives.from_file(app_config.prompt_settings_file)
    ai_profile, task_directives = await generate_agent_profile_for_task(
        task=task,
        app_config=app_config,
        llm_provider=llm_provider,
    )
    return _configure_agent(
        task=task,
        ai_profile=ai_profile,
        directives=base_directives + task_directives,
        app_config=app_config,
        llm_provider=llm_provider,
    )

```

# `autogpts/autogpt/autogpt/agent_factory/profile_generator.py`

这段代码使用了Python的logging库来输出日志信息。它从autogpt的config模块中导入了一些配置类，包括AIDirectives、AIProfile、Config、SystemConfiguration和UserConfigurable。这些配置类用于设置人工智能的一些参数和用户可配置选项。

然后，它从autogpt的core模块中导入了一些配置类和工具函数，包括ChatPrompt、LanguageModelClassification和PromptStrategy。这些工具函数用于生成用户交互对话中的提示信息。

接着，它从autogpt的core模块的prompting工具类中导入了一些工具函数，包括json_loads，用于解析和解析JSON格式的数据。

最后，它从autogpt的core模块的resource模型提供商中导入了一些模型提供商，包括AssistantChatMessageDict、ChatMessage和ChatModelProvider。这些模型提供商用于生成机器学习模型的输出，包括聊天消息和对话模型。


```py
import logging

from autogpt.config import AIDirectives, AIProfile, Config
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.prompting import (
    ChatPrompt,
    LanguageModelClassification,
    PromptStrategy,
)
from autogpt.core.prompting.utils import json_loads
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessageDict,
    ChatMessage,
    ChatModelProvider,
    CompletionModelFunction,
)
```

This is a code snippet written in JavaScript that defines an environment for an autonomous AI agent. The environment has a user object with a name and description, as well as a set of constraints and best practices for completing a given task. The agent can be created through the `create_agent` function, which takes in the agent's name, description, and constraints as input.

The environment also includes a template for a user-provided prompt, which defines the user's objective for creating the agent. The prompt can be used to direct the agent's development process, such as setting the task to be completed or providing guidelines for creating an agent that is skilled in a particular area.

Overall, this code snippet defines an environment for creating and managing autonomous AI agents, with a user-friendly interface for controlling and customizing the agent's development.


```py
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


class AgentProfileGeneratorConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable(
        default=LanguageModelClassification.SMART_MODEL
    )
    system_prompt: str = UserConfigurable(
        default=(
            "Your job is to respond to a user-defined task, given in triple quotes, by "
            "invoking the `create_agent` function to generate an autonomous agent to "
            "complete the task. "
            "You should supply a role-based name for the agent (_GPT), "
            "an informative description for what the agent does, and "
            "1 to 5 directives in each of the categories Best Practices and Constraints, "
            "that are optimally aligned with the successful completion "
            "of its assigned task.\n"
            "\n"
            "Example Input:\n"
            '"""Help me with marketing my business"""\n\n'
            "Example Function Call:\n"
            "```\n"
            "{"
            '"name": "create_agent",'
            ' "arguments": {'
            '"name": "CMOGPT",'
            ' "description": "a professional digital marketer AI that assists Solopreneurs in'
            " growing their businesses by providing world-class expertise in solving"
            ' marketing problems for SaaS, content products, agencies, and more.",'
            ' "directives": {'
            ' "best_practices": ['
            '"Engage in effective problem-solving, prioritization, planning, and'
            " supporting execution to address your marketing needs as your virtual Chief"
            ' Marketing Officer.",'
            ' "Provide specific, actionable, and concise advice to help you make'
            " informed decisions without the use of platitudes or overly wordy"
            ' explanations.",'
            ' "Identify and prioritize quick wins and cost-effective campaigns that'
            ' maximize results with minimal time and budget investment.",'
            ' "Proactively take the lead in guiding you and offering suggestions when'
            " faced with unclear information or uncertainty to ensure your marketing"
            ' strategy remains on track."'
            "],"  # best_practices
            ' "constraints": ['
            '"Do not suggest illegal or unethical plans or strategies.",'
            ' "Take reasonable budgetary limits into account."'
            "]"  # constraints
            "}"  # directives
            "}"  # arguments
            "}\n"
            "```py"
        )
    )
    user_prompt_template: str = UserConfigurable(default='"""{user_objective}"""')
    create_agent_function: dict = UserConfigurable(
        default=CompletionModelFunction(
            name="create_agent",
            description="Create a new autonomous AI agent to complete a given task.",
            parameters={
                "name": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="A short role-based name for an autonomous agent.",
                    required=True,
                ),
                "description": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="An informative one sentence description of what the AI agent does",
                    required=True,
                ),
                "directives": JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "best_practices": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=1,
                            maxItems=5,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                            ),
                            description=(
                                "One to five highly effective best practices that are"
                                " optimally aligned with the completion of the given task."
                            ),
                            required=True,
                        ),
                        "constraints": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=1,
                            maxItems=5,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                            ),
                            description=(
                                "One to five highly effective constraints that are"
                                " optimally aligned with the completion of the given task."
                            ),
                            required=True,
                        ),
                    },
                    required=True,
                ),
            },
        ).schema
    )


```py

This is a class that defines a chatbot that can be used to extract information from users using an AI model. The chatbot has a user interface that the user can interact with by typing a message and the chatbot will return a response. The chatbot can be configured through a `ChatPrompt` class that inherits from the `ChatFramework` class. The `ChatPrompt` class has a `build_prompt` method that builds the prompt message and a `parse_response_content` method that parses the AI model's response to the user's input. The AI model is defined using the `model_classification` attribute, which is an instance of `LanguageModelClassification`.


```py
class AgentProfileGenerator(PromptStrategy):
    default_configuration: AgentProfileGeneratorConfiguration = (
        AgentProfileGeneratorConfiguration()
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt: str,
        user_prompt_template: str,
        create_agent_function: dict,
    ):
        self._model_classification = model_classification
        self._system_prompt_message = system_prompt
        self._user_prompt_template = user_prompt_template
        self._create_agent_function = CompletionModelFunction.parse(
            create_agent_function
        )

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(self, user_objective: str = "", **kwargs) -> ChatPrompt:
        system_message = ChatMessage.system(self._system_prompt_message)
        user_message = ChatMessage.user(
            self._user_prompt_template.format(
                user_objective=user_objective,
            )
        )
        prompt = ChatPrompt(
            messages=[system_message, user_message],
            functions=[self._create_agent_function],
        )
        return prompt

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> tuple[AIProfile, AIDirectives]:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        try:
            arguments = json_loads(response_content["function_call"]["arguments"])
            ai_profile = AIProfile(
                ai_name=arguments.get("name"),
                ai_role=arguments.get("description"),
            )
            ai_directives = AIDirectives(
                best_practices=arguments["directives"].get("best_practices"),
                constraints=arguments["directives"].get("constraints"),
                resources=[],
            )
        except KeyError:
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        return ai_profile, ai_directives


```py

这段代码定义了一个名为 `generate_agent_profile_for_task` 的异步函数，它接受一个任务（string）、一个应用程序配置（Config）和一个聊天模型提供商（ChatModelProvider）作为参数。

函数内部首先创建一个自定义的 LLM 配置对象，然后使用 `AgentProfileGenerator` 类的一个名为 `build_prompt` 的方法来生成与任务相关的 AI 配置对象。接着，使用 `llm_provider` 对象将生成的 AI 配置对象与用户进行交互，并从其输出中提取信息。

函数返回一个 AI Profile 和一个 AIDirectives 对象，AI Profile 是由 LLM 创建的针对用户任务的 AI 配置，AIDirectives 是一个指导如何应用 AI Profile 的指令集合。


```py
async def generate_agent_profile_for_task(
    task: str,
    app_config: Config,
    llm_provider: ChatModelProvider,
) -> tuple[AIProfile, AIDirectives]:
    """Generates an AIConfig object from the given string.

    Returns:
    AIConfig: The AIConfig object tailored to the user's input
    """
    agent_profile_generator = AgentProfileGenerator(
        **AgentProfileGenerator.default_configuration.dict()  # HACK
    )

    prompt = agent_profile_generator.build_prompt(task)

    # Call LLM with the string as user input
    output = (
        await llm_provider.create_chat_completion(
            prompt.messages,
            model_name=app_config.smart_llm,
            functions=prompt.functions,
        )
    ).response

    # Debug LLM Output
    logger.debug(f"AI Config Generator Raw Output: {output}")

    # Parse the output
    ai_profile, ai_directives = agent_profile_generator.parse_response_content(output)

    return ai_profile, ai_directives

```py

# `autogpts/autogpt/autogpt/agent_manager/agent_manager.py`

This code defines an `AgentManager` class that manages a list of agents and their associated files.

Here's what the code does:

1. Creates an agents directory within the app data directory, if it doesn't exist yet.
2. Defines a `generate_id` function to generate a unique ID for each agent.
3. Defines a `list_agents` method to list all agents in the agents directory.
4. Defines a `get_agent_dir` method to get the agent directory for a given agent ID, creates it if it doesn't exist, and checks if it does.
5. Defines a `retrieve_state` method to retrieve the agent state file for a given agent ID, reads it from the agent directory, and returns it as an `AgentSettings` object.

The `AgentManager` class is meant to be used in conjunction with other classes that will be instantiating it, such as `AutogptAgents` or `AgentsDirectory` classes.


```py
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agents.agent import AgentSettings

from autogpt.agents.utils.agent_file_manager import AgentFileManager


class AgentManager:
    def __init__(self, app_data_dir: Path):
        self.agents_dir = app_data_dir / "agents"
        if not self.agents_dir.exists():
            self.agents_dir.mkdir()

    @staticmethod
    def generate_id(agent_name: str) -> str:
        unique_id = str(uuid.uuid4())[:8]
        return f"{agent_name}-{unique_id}"

    def list_agents(self) -> list[str]:
        return [
            dir.name
            for dir in self.agents_dir.iterdir()
            if dir.is_dir() and AgentFileManager(dir).state_file_path.exists()
        ]

    def get_agent_dir(self, agent_id: str, must_exist: bool = False) -> Path:
        agent_dir = self.agents_dir / agent_id
        if must_exist and not agent_dir.exists():
            raise FileNotFoundError(f"No agent with ID '{agent_id}'")
        return agent_dir

    def retrieve_state(self, agent_id: str) -> AgentSettings:
        from autogpt.agents.agent import AgentSettings

        agent_dir = self.get_agent_dir(agent_id, True)
        state_file = AgentFileManager(agent_dir).state_file_path
        if not state_file.exists():
            raise FileNotFoundError(f"Agent with ID '{agent_id}' has no state.json")

        state = AgentSettings.load_from_json_file(state_file)
        state.agent_data_dir = agent_dir
        return state

```py

# `autogpts/autogpt/autogpt/agent_manager/__init__.py`

这段代码定义了一个名为 `AgentManager` 的类，并导入了该类的一个名为 `__all__` 的列表，使得列表中的所有成员都可以被导出。

具体来说，这段代码创建了一个名为 `AgentManager` 的类。由于使用了 `__all__ = ["AgentManager"]` 的导入选项，因此 `AgentManager` 类中的所有成员都可以被导出，这意味着可以使用以下代码来访问和使用 `AgentManager` 类中的成员：

```py
from .agent_manager import AgentManager

agent_manager = AgentManager()

# 访问 agent_manager 对象中的方法
print(agent_manager.get_agent_id())

# 使用 agent_manager 对象中的方法来获取更多信息
agent_manager.do_something()
```py

由于 `AgentManager` 类被导入了 `__all__` 列表中，因此可以使用上述代码来访问和使用该类中的所有成员。


```py
from .agent_manager import AgentManager

__all__ = ["AgentManager"]

```py

# `autogpts/autogpt/autogpt/app/agent_protocol_server.py`

这段代码使用了多个Python库，其作用是实现一个自定义的API路由器。具体来说，这段代码实现了一个简单的API，用于处理一个根路径下的/status endpoint，该endpoint返回一个JSON格式的数据。

具体来说，这段代码使用了以下几个库：

* logging：用于记录API的请求和响应信息。
* os：用于处理文件和目录操作。
* pathlib：用于处理文件路径和目录操作。
* uuid：用于生成唯一的ID。
* io：用于处理输入输出流。
* forge.sdk.db：用于数据库操作。
* forge.sdk.errors：用于处理错误。
* forge.sdk.middlewares：用于处理中间件。
* forge.sdk.routes.agent_protocol：用于实现Agent协议的路由器。
* fastapi：用于实现API。

这段代码的具体实现可以进一步分析，但是从提供的信息来看，这段代码实现了一个简单的API路由器，用于处理一个根路径下的/status endpoint，该endpoint返回一个JSON格式的数据。


```py
import logging
import os
import pathlib
from io import BytesIO
from uuid import uuid4

import orjson
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from forge.sdk.db import AgentDB
from forge.sdk.errors import NotFoundError
from forge.sdk.middlewares import AgentMiddleware
from forge.sdk.routes.agent_protocol import base_router
```py

这段代码是一个Python程序，它实现了Forge.SDK.Schema库中的几个类，用于定义任务、步骤、任务元数据等概念。具体来说，它包括：Artifact（作品）、Step（步骤）、StepRequestBody（步骤请求主体）、Task（任务）、TaskArtifactsListResponse（任务元数据列表响应）、TaskListResponse（任务列表响应）、TaskRequestBody（任务请求主体）和TaskStepsListResponse（任务步骤列表响应）。

此外，它还引入了两个来自hypercorn.asyncio和hypercorn.config的函数：serve和config。其中，serve函数用于异步启动Forge.SDK.Schema库的server，而config函数则用于设置Hypercorn的配置。


```py
from forge.sdk.schema import (
    Artifact,
    Step,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
from hypercorn.asyncio import serve as hypercorn_serve
from hypercorn.config import Config as HypercornConfig

from autogpt.agent_factory.configurators import configure_agent_with_state
from autogpt.agent_factory.generators import generate_agent_for_task
```py



This is a simple Python class that provides an interface for interacting with an S3 bucket and generating artifacts. The class is using the `boto3` library for interacting with S3 and the `google-auth` library for handling Google authentication.

The class has two methods, `generate_artifact` and `get_artifact`, which are responsible for generating and retrieving artifacts, respectively.

The `generate_artifact` method takes a task ID and an artifact ID as input, and reads the file specified by the `file_name` parameter from the S3 bucket, generates some data, and writes it to a file. The file is in binary mode, so it can be streamed and read by the `get_task_agent_file_workspace` method. Once the file is written, it is ready to be downloaded by the user.

The `get_artifact` method takes a task ID and an artifact ID as input, and retrieves the artifact corresponding to the specified ID. It reads the file specified by the `file_name` parameter from the S3 bucket, and downloads it. If the file is not found, it raises a `NotFoundError`.

Note that the class is missing some important documentation and some of the methods have been added, but not tested. Also, it is important to handle the case where the file is not found, and also the case when the user is not logged in.


```py
from autogpt.agent_manager import AgentManager
from autogpt.commands.system import finish
from autogpt.commands.user_interaction import ask_user
from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.file_workspace import FileWorkspace
from autogpt.models.action_history import ActionErrorResult, ActionSuccessResult

logger = logging.getLogger(__name__)


class AgentProtocolServer:
    def __init__(
        self,
        app_config: Config,
        database: AgentDB,
        llm_provider: ChatModelProvider,
    ):
        self.app_config = app_config
        self.db = database
        self.llm_provider = llm_provider
        self.agent_manager = AgentManager(app_data_dir=app_config.app_data_dir)

    async def start(self, port: int = 8000, router: APIRouter = base_router):
        """Start the agent server."""
        logger.debug("Starting the agent server...")
        config = HypercornConfig()
        config.bind = [f"localhost:{port}"]
        app = FastAPI(
            title="AutoGPT Server",
            description="Forked from AutoGPT Forge; Modified version of The Agent Protocol.",
            version="v0.4",
        )

        # Add CORS middleware
        origins = [
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            # Add any other origins you want to whitelist
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(router, prefix="/ap/v1")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        frontend_path = (
            pathlib.Path(script_dir)
            .joinpath("../../../../frontend/build/web")
            .resolve()
        )

        if os.path.exists(frontend_path):
            app.mount("/app", StaticFiles(directory=frontend_path), name="app")

            @app.get("/", include_in_schema=False)
            async def root():
                return RedirectResponse(url="/app/index.html", status_code=307)

        else:
            logger.warning(
                f"Frontend not found. {frontend_path} does not exist. The frontend will not be available."
            )

        # Used to access the methods on this class from API route handlers
        app.add_middleware(AgentMiddleware, agent=self)

        config.loglevel = "ERROR"
        config.bind = [f"0.0.0.0:{port}"]

        logger.info(f"AutoGPT server starting on http://localhost:{port}")
        await hypercorn_serve(app, config)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        logger.debug(f"Creating agent for task: '{task_request.input}'")
        task_agent = await generate_agent_for_task(
            task=task_request.input,
            app_config=self.app_config,
            llm_provider=self.llm_provider,
        )
        task = await self.db.create_task(
            input=task_request.input,
            additional_input=task_request.additional_input,
        )
        agent_id = task_agent.state.agent_id = task_agent_id(task.task_id)
        logger.debug(f"New agent ID: {agent_id}")
        task_agent.attach_fs(self.app_config.app_data_dir / "agents" / agent_id)
        task_agent.state.save_to_json_file(task_agent.file_manager.state_file_path)
        return task

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        logger.debug("Listing all tasks...")
        tasks, pagination = await self.db.list_tasks(page, pageSize)
        response = TaskListResponse(tasks=tasks, pagination=pagination)
        return response

    async def get_task(self, task_id: int) -> Task:
        """
        Get a task by ID.
        """
        logger.debug(f"Getting task with ID: {task_id}...")
        task = await self.db.get_task(task_id)
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        logger.debug(f"Listing all steps created by task with ID: {task_id}...")
        steps, pagination = await self.db.list_steps(task_id, page, pageSize)
        response = TaskStepsListResponse(steps=steps, pagination=pagination)
        return response

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """Create a step for the task."""
        logger.debug(f"Creating a step for task with ID: {task_id}...")

        # Restore Agent instance
        agent = configure_agent_with_state(
            state=self.agent_manager.retrieve_state(task_agent_id(task_id)),
            app_config=self.app_config,
            llm_provider=self.llm_provider,
        )
        agent.workspace.on_write_file = lambda path: self.db.create_artifact(
            task_id=task_id,
            file_name=path.parts[-1],
            relative_path=str(path),
        )

        # According to the Agent Protocol spec, the first execute_step request contains
        #  the same task input as the parent create_task request.
        # To prevent this from interfering with the agent's process, we ignore the input
        #  of this first step request, and just generate the first step proposal.
        is_init_step = not bool(agent.event_history)
        execute_command, execute_command_args, execute_result = None, None, None
        execute_approved = False

        # HACK: only for compatibility with AGBenchmark
        if step_request.input == "y":
            step_request.input = ""

        user_input = step_request.input if not is_init_step else ""

        if (
            not is_init_step
            and agent.event_history.current_episode
            and not agent.event_history.current_episode.result
        ):
            execute_command = agent.event_history.current_episode.action.name
            execute_command_args = agent.event_history.current_episode.action.args
            execute_approved = not user_input

            logger.debug(
                f"Agent proposed command"
                f" {execute_command}({fmt_kwargs(execute_command_args)})."
                f" User input/feedback: {repr(user_input)}"
            )

        # Save step request
        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            is_last=execute_command == finish.__name__ and execute_approved,
        )

        # Execute previously proposed action
        if execute_command:
            assert execute_command_args is not None

            if step.is_last and execute_command == finish.__name__:
                assert execute_command_args
                step = await self.db.update_step(
                    task_id=task_id,
                    step_id=step.step_id,
                    output=execute_command_args["reason"],
                )
                return step

            if execute_command == ask_user.__name__:  # HACK
                execute_result = ActionSuccessResult(outputs=user_input)
                agent.event_history.register_result(execute_result)
            elif execute_approved:
                step = await self.db.update_step(
                    task_id=task_id,
                    step_id=step.step_id,
                    status="running",
                )
                # Execute previously proposed action
                execute_result = await agent.execute(
                    command_name=execute_command,
                    command_args=execute_command_args,
                )
            else:
                assert user_input
                execute_result = await agent.execute(
                    command_name="human_feedback",  # HACK
                    command_args={},
                    user_input=user_input,
                )

        # Propose next action
        try:
            next_command, next_command_args, raw_output = await agent.propose_action()
            logger.debug(f"AI output: {raw_output}")
        except Exception as e:
            step = await self.db.update_step(
                task_id=task_id,
                step_id=step.step_id,
                status="completed",
                output=f"An error occurred while proposing the next action: {e}",
            )
            return step

        # Format step output
        output = (
            (
                f"Command `{execute_command}({fmt_kwargs(execute_command_args)})` returned:"
                f" {execute_result}\n\n"
            )
            if execute_command_args and execute_command != "ask_user"
            else ""
        )
        output += f"{raw_output['thoughts']['speak']}\n\n"
        output += (
            f"Next Command: {next_command}({fmt_kwargs(next_command_args)})"
            if next_command != "ask_user"
            else next_command_args["question"]
        )

        additional_output = {
            **(
                {
                    "last_action": {
                        "name": execute_command,
                        "args": execute_command_args,
                        "result": (
                            orjson.loads(execute_result.json())
                            if not isinstance(execute_result, ActionErrorResult)
                            else {
                                "error": str(execute_result.error),
                                "reason": execute_result.reason,
                            }
                        ),
                    },
                }
                if not is_init_step
                else {}
            ),
            **raw_output,
        }

        step = await self.db.update_step(
            task_id=task_id,
            step_id=step.step_id,
            status="completed",
            output=output,
            additional_output=additional_output,
        )

        agent.state.save_to_json_file(agent.file_manager.state_file_path)
        return step

    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        Get a step by ID.
        """
        step = await self.db.get_step(task_id, step_id)
        return step

    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        List the artifacts that the task has created.
        """
        artifacts, pagination = await self.db.list_artifacts(task_id, page, pageSize)
        return TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination)

    async def create_artifact(
        self, task_id: str, file: UploadFile, relative_path: str
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
        data = None
        file_name = file.filename or str(uuid4())
        data = b""
        while contents := file.file.read(1024 * 1024):
            data += contents
        # Check if relative path ends with filename
        if relative_path.endswith(file_name):
            file_path = relative_path
        else:
            file_path = os.path.join(relative_path, file_name)

        workspace = get_task_agent_file_workspace(task_id, self.agent_manager)
        await workspace.write_file(file_path, data)

        artifact = await self.db.create_artifact(
            task_id=task_id,
            file_name=file_name,
            relative_path=relative_path,
            agent_created=False,
        )
        return artifact

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        """
        Get an artifact by ID.
        """
        try:
            artifact = await self.db.get_artifact(artifact_id)
            if artifact.file_name not in artifact.relative_path:
                file_path = os.path.join(artifact.relative_path, artifact.file_name)
            else:
                file_path = artifact.relative_path
            workspace = get_task_agent_file_workspace(task_id, self.agent_manager)
            retrieved_artifact = workspace.read_file(file_path, binary=True)
        except NotFoundError as e:
            raise
        except FileNotFoundError as e:
            raise

        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={artifact.file_name}"
            },
        )


```py

这段代码定义了一个名为 `task_agent_id` 的函数，它接收一个名为 `task_id` 的参数，并返回一个字符串。函数的实现主要通过使用 Python 的字符串格式化操作，将 `task_id` 参数与一个名为 `AutoGPT-` 的前缀字符串相结合，并将结果返回。

接下来，定义了一个名为 `get_task_agent_file_workspace` 的函数，它接收一个名为 `task_id` 的参数，以及一个名为 `agent_manager` 的 `AgentManager` 实例。函数的实现主要通过创建一个名为 `FileWorkspace` 的类，将 `agent_manager` 中的 `get_agent_dir` 方法得到的 agent 目录作为根目录，创建一个 `workspace` 目录作为子目录，并将 `FileWorkspace` 类应用于创建的目录。这样，函数可以确保在生成文件 workspace 时，将 agent 目录和 workspace 目录都包含在内。

总的来说，这段代码主要定义了两个函数，一个是将任务 ID 格式化为字符串，另一个是创建一个用于存储 task agent workspace 的文件 workspace。


```py
def task_agent_id(task_id: str | int) -> str:
    return f"AutoGPT-{task_id}"


def get_task_agent_file_workspace(
    task_id: str | int,
    agent_manager: AgentManager,
) -> FileWorkspace:
    return FileWorkspace(
        root=agent_manager.get_agent_dir(
            agent_id=task_agent_id(task_id),
            must_exist=True,
        )
        / "workspace",
        restrict_to_root=True,
    )


```py

这段代码定义了一个名为 `fmt_kwargs` 的函数，它接受一个名为 `kwargs` 的字典参数，并返回一个字符串，该字符串表示了所有在 `kwargs` 中定义的参数的名称和值。

具体来说，该函数通过遍历 `kwargs` 中的每个键-值对，创建一个新的字符串，其中字符串 `"{n}={repr(v)}"` 表示参数 `n` 的名称，`repr(v)` 函数用于将参数 `v` 的对象进行字符串表示，然后将 `n` 和 `repr(v)` 组合成一个字符串，用等号 `=` 分隔。最后，该函数使用 `join()` 方法将这些字符串连接成一个大的字符串，并使用逗号 `,` 分隔。

最终，该函数返回的结果是一个字符串，其中每个参数都按照其定义的格式进行字符串表示，可以被用来格式化输出结果。


```py
def fmt_kwargs(kwargs: dict) -> str:
    return ", ".join(f"{n}={repr(v)}" for n, v in kwargs.items())

```py

# `autogpts/autogpt/autogpt/app/cli.py`

这是一个用Python编写的CLI脚本，它使用NVIDIA的PyTorch和Transformers库实现了一个自动生成语言模型的应用。通过运行这个脚本，用户可以选择不同的语言模型，例如英语、法语或西班牙语，以及需要生成的文本长度。

具体来说，这个脚本的作用是接收用户输入的命令行参数，并在这些参数的帮助下运行一个名为"run"的函数。如果用户没有提供任何参数，则脚本将默认运行"run"函数以获得帮助。

在函数"run"中，我们使用PyTorch和Transformers库训练和部署一个预训练的语言模型，以便在用户需要时生成指定长度的文本。我们可以通过设置训练文件、文本长度和模型架构来调整模型的配置。

在CLI脚本中，我们还使用NVIDIA的深度学习SDK，以便在运行脚本时自动获取 CUDA 版本号。我们还定义了一个辅助函数"click"，用于在用户输入的命令行参数中添加类型提示和错误处理。


```py
"""Main script for the autogpt package."""
from pathlib import Path
from typing import Optional

import click


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    # Invoke `run` by default
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


```py

这段代码是一个命令行工具的多行选项。具体来说，它有以下几个选项：

@cli.command()：这是一个命令行工具的标识符，表示这段代码是一个命令行工具，可以用来调用它。

@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")：这是一个开关选项，表示是否进入连续运行模式。如果选中这个选项，那么命令行工具就会在运行时自动连续运行，直到被手动停止。

@click.option(
   "--skip-reprompt",
   "-y",
   is_flag=True,
   help="Skips the re-prompting messages at the beginning of the script",
)：这也是一个开关选项，表示是否在每次运行时忽略提示消息。如果选中这个选项，那么命令行工具就会在运行时忽略提示消息，不会提示用户输入任何东西。

@click.option(
   "--ai-settings",
   "-C",
   type=click.Path(exists=True, dir_okay=False, path_type=Path),
   help=(
       "Specifies which ai_settings.yaml file to use, relative to the AutoGPT"
       " root directory. Will also automatically skip the re-prompt。"
   ),
)：这是另一个选项，表示要设置的人工智能设置。它是一个文件路径选项，用于指定一个根目录下的文件，用于设置人工智能的设置。


```py
@cli.command()
@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
@click.option(
    "--ai-settings",
    "-C",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Specifies which ai_settings.yaml file to use, relative to the AutoGPT"
        " root directory. Will also automatically skip the re-prompt."
    ),
)
```py

这段代码定义了一系列命令行选项，包括：

@click.option(
   "--prompt-settings",
   "-P",
   type=click.Path(exists=True, dir_okay=False, path_type=Path),
   help="Specifies which prompt_settings.yaml file to use.",
)
@click.option(
   "-l",
   "--continuous-limit",
   type=int,
   help="Defines the number of times to run in continuous mode",
)
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
@click.option("--debug", is_flag=True, help="Enable Debug Mode")
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
)

这个选项中的每个选项都有一个短语选项和对应的选项名称。短语选项通常是一个缩写或是一个带有冒号（:）的选项，它们告诉Click什么是这个命令行选项的实际意义。

@click.option("--prompt-settings", "-P")

这个选项是用来指定要使用的prompt_settings.yaml文件的。

@click.option("--continuous-limit", "-l")

这个选项是用来定义连续运行的最大次数的。

@click.option("--speak", is_flag=True, help="Enable Speak Mode")

这个选项是用来切换到speak模式，如果设置为True，就会啟用语音輸出。

@click.option("--debug", is_flag=True, help="Enable Debug Mode")

这个选项是用来切换到debug模式，如果设置为True，就会啟用debug模式，啟用這個模式會提供更多的信息，但是可能會導致速度變慢。

@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")

这个选项是用来切换到GPT3.5 only模式，如果设置为True，就會啟用這個模式。在這個模式下，GPT3.5 will only被使用，而不是GPT3.


```py
@click.option(
    "--prompt-settings",
    "-P",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Specifies which prompt_settings.yaml file to use.",
)
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
@click.option("--debug", is_flag=True, help="Enable Debug Mode")
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
```py

这段代码是一个 Click 选项对象，用于在命令行工具中提供一系列选项。具体来说，它有以下几个选项：

@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
   "--use-memory",
   "-m",
   "memory_type",
   type=str,
   help="Defines which Memory backend to use",
)
@click.option(
   "-b",
   "--browser-name",
   help="Specifies which web-browser to use when using Selenium to scrape the web.",
)
@click.option(
   "--allow-downloads",
   is_flag=True,
   help="Dangerous: Allows AutoGPT to download files natively.",
)

这个选项对象使用 Click 命令来创建一个带有参数的选项对象。通过调用 `click.option` 函数，我们可以获取用户在 Click 工具中输入的每个选项。

具体来说，`--gpt4only` 选项是一个布尔选项，表示是否启用 GPT4 模式。如果启用 GPT4 模式，那么它将只使用 GPT4 来运行用户提供的指令。

`--use-memory` 选项是一个字符选项，用于指定使用哪种内存 backend。这个选项允许用户指定使用 CPU 而不是 GPU。

`--browser-name` 选项是一个字符选项，用于指定在使用 Selenium 爬取网页时使用哪个浏览器。

`--allow-downloads` 选项是一个布尔选项，表示是否允许用户下载自动生成的模版文件。这个选项默认是 False，表示禁止用户下载文件。


```py
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
    "--use-memory",
    "-m",
    "memory_type",
    type=str,
    help="Defines which Memory backend to use",
)
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows AutoGPT to download files natively.",
)
```py

这段代码是一个命令行工具的选项设置，其中包括三个选项：

1. `--skip-news` 选项是一个布尔选项，表示在启动时是否输出最新的新闻。如果选中此选项，则不会输出最新的新闻，但是不会影响程序的其他部分。

2. `--workspace-directory` 选项是一个字符串选项，表示应用程序的工作区目录。如果选中此选项，则指定一个应用程序的工作区目录，可以在该目录下执行命令行工具或者在终端中运行 `cd` 命令。

3. `--install-plugin-deps` 选项是一个布尔选项，表示是否安装第三方插件的依赖项。如果选中此选项，则会在程序的安装目录下安装第三方插件的依赖项。

该命令行工具使用 `@click` 包实现选项设置。`@click` 是一个用于创建命令行工具和命令的库，可以轻松地设置选项和它们的值。


```py
@click.option(
    "--skip-news",
    is_flag=True,
    help="Specifies whether to suppress the output of latest news on startup.",
)
@click.option(
    # TODO: this is a hidden option for now, necessary for integration testing.
    #   We should make this public once we're ready to roll out agent specific workspaces.
    "--workspace-directory",
    "-w",
    type=click.Path(),
    hidden=True,
)
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
```py

这段代码使用了 Click 库定义了三个选项，分别是 `--ai-name`、`--ai-role` 和 `--constraint`。

* `--ai-name` 选项是一个字符串类型，用于指定 AI 的名称，如果用户没有指定名称，则使用 `None`。这个选项可以用来重写 AI 的名称。
* `--ai-role` 选项也是一个字符串类型，用于指定 AI 的角色，如果用户没有指定角色，则使用 `None`。这个选项也可以用来重写 AI 的角色。
* `--constraint` 是一个多选项字符串类型，用于指定是否在 AI 的提示中包含约束。如果用户没有指定约束，则使用 `None`。这个选项最多可以指定多个约束。

总的来说，这段代码定义了三个选项，用于指定 AI 的名称、角色和约束，用户可以根据自己的需要来选择使用或不使用这些选项。


```py
@click.option(
    "--ai-name",
    type=str,
    help="AI name override",
)
@click.option(
    "--ai-role",
    type=str,
    help="AI role override",
)
@click.option(
    "--constraint",
    type=str,
    multiple=True,
    help=(
        "Add or override AI constraints to include in the prompt;"
        " may be used multiple times to pass multiple constraints"
    ),
)
```py

这段代码定义了两个 Click 选项选项，分别是 `--resource` 和 `--best-practice`。

`--resource` 选项是一个字符串选项，可以有多个值，且可能有不同的含义，具体含义需要根据上下文来确定。这个选项的作用是让用户可以添加或覆盖 AI 资源的提示中包括哪些资源。

`--best-practice` 选项也是一个字符串选项，可以有多个值，且可能有不同的含义，具体含义需要根据上下文来确定。这个选项的作用是让用户可以添加或覆盖 AI 最佳实践的提示中包括哪些最佳实践。

由于 `@click.option` 装饰函数会接受一个参数 `help` 来说明该选项的作用，因此以上代码的实际作用可能会有所不同。具体如何使用这些选项，还需要根据上下文来确定。


```py
@click.option(
    "--resource",
    type=str,
    multiple=True,
    help=(
        "Add or override AI resources to include in the prompt;"
        " may be used multiple times to pass multiple resources"
    ),
)
@click.option(
    "--best-practice",
    type=str,
    multiple=True,
    help=(
        "Add or override AI best practices to include in the prompt;"
        " may be used multiple times to pass multiple best practices"
    ),
)
```py

This is a function that sets up and runs an agent based on the task specified by the user, or resumes an existing agent. It takes in several parameters, including the AI settings, prompt settings, and other preferences, such as whether to speak, how to use PromptGLM, and whether to enable or disable GPT. It also takes in additional parameters related to the AI, such as the AI name, role, and resource.


```py
@click.option(
    "--override-directives",
    is_flag=True,
    help=(
        "If specified, --constraint, --resource and --best-practice will override"
        " the AI's directives instead of being appended to them"
    ),
)
def run(
    continuous: bool,
    continuous_limit: int,
    ai_settings: Optional[Path],
    prompt_settings: Optional[Path],
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
    workspace_directory: str,
    install_plugin_deps: bool,
    ai_name: Optional[str],
    ai_role: Optional[str],
    resource: tuple[str],
    constraint: tuple[str],
    best_practice: tuple[str],
    override_directives: bool,
) -> None:
    """
    Sets up and runs an agent, based on the task specified by the user, or resumes an
    existing agent.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    from autogpt.app.main import run_auto_gpt

    run_auto_gpt(
        continuous=continuous,
        continuous_limit=continuous_limit,
        ai_settings=ai_settings,
        prompt_settings=prompt_settings,
        skip_reprompt=skip_reprompt,
        speak=speak,
        debug=debug,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        memory_type=memory_type,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
        skip_news=skip_news,
        workspace_directory=workspace_directory,
        install_plugin_deps=install_plugin_deps,
        override_ai_name=ai_name,
        override_ai_role=ai_role,
        resources=list(resource),
        constraints=list(constraint),
        best_practices=list(best_practice),
        override_directives=override_directives,
    )


```py

这段代码是一个命令行工具的CLI选项。它定义了一系列CLI选项，包括：

@cli.command：这个选项告诉程序这是一个命令行工具，而不是一个图形用户界面。

@click.option(
   "--prompt-settings",
   "-P",
   type=click.Path(exists=True, dir_okay=False, path_type=Path),
   help="Specifies which prompt_settings.yaml file to use.",
)
：这个选项告诉用户，他们应该提供一个什么样的设置，以配置prompt（提示消息）。用户需要提供一个文件路径，这个文件应该是提示消息的定义。

@click.option(
   "--debug",
   is_flag=True,
   help="Enable Debug Mode",
)
：这个选项告诉用户，他们是否想要进入调试模式。如果用户选择了这个选项，那么程序将会更加详细地输出信息。

@click.option(
   "--gpt3only",
   is_flag=True,
   help="Enable GPT3.5 Only Mode",
)
：这个选项告诉用户，他们是否想要使用GPT3.5版本。如果用户选择了这个选项，那么程序将会只使用GPT3.5版本。

@click.option(
   "--gpt4only",
   is_flag=True,
   help="Enable GPT4 Only Mode",
)
：这个选项告诉用户，他们是否想要使用GPT4版本。如果用户选择了这个选项，那么程序将会只使用GPT4版本。

@click.option(
   "--use-memory",
   "-m",
   "memory_type",
   type=str,
   help="Defines which Memory backend to use",
)
：这个选项告诉用户，他们是否想要使用哪种内存模式。有几个不同的选项，包括：

      memory_type：line
      memory_type：none
      memory_type：redis
      memory_type：water

这个选项的文档没有给出完整的选项列表，但是它告诉用户，他们可以在内存中提供额外的数据，从而提高程序的性能。


```py
@cli.command()
@click.option(
    "--prompt-settings",
    "-P",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Specifies which prompt_settings.yaml file to use.",
)
@click.option("--debug", is_flag=True, help="Enable Debug Mode")
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
    "--use-memory",
    "-m",
    "memory_type",
    type=str,
    help="Defines which Memory backend to use",
)
```py

这段代码使用了 Click 库，是一个 Python 命令行工具，用于自动化和简化命令行操作。

具体来说，这段代码定义了三个选项：

@click.option(
   "-b",
   "--browser-name",
   help="Specifies which web-browser to use when using Selenium to scrape the web.",
)
@click.option(
   "--allow-downloads",
   is_flag=True,
   help="Dangerous: Allows AutoGPT to download files natively.",
)
@click.option(
   "--install-plugin-deps",
   is_flag=True,
   help="Installs external dependencies for 3rd party plugins.",
)

第一个选项 `-b`（--browser-name）是一个选项参数，用于指定在使用 Selenium（Python Selenium库）爬取网页时使用的浏览器名称。

第二个选项 `--allow-downloads`（危险选项）是一个布尔选项，用于允许用户允许 AutoGPT 下载文件。

第三个选项 `--install-plugin-deps`（危险选项）是一个布尔选项，用于安装第三方软件包和依赖项。


```py
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows AutoGPT to download files natively.",
)
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
```py

这段代码定义了一个名为 `serve` 的函数，它接受一个名为 `prompt_settings` 的可选参数，一个表示 `debug` 变量为真值的布尔值，一个表示只使用 GPT3 的布尔值，一个表示只使用 GPT4 的布尔值，一个表示使用哪种内存类型的布尔值，一个表示是否允许下载的布尔值，和一个可选的参数 `browser_name`。

函数内部使用 `from autogpt.app.main import run_auto_gpt_server` 导入了一个名为 `run_auto_gpt_server` 的函数，并传入上述参数，然后调用该函数以创建一个符合 Agent Protocol 的自动 GPT3 或 GPT4 服务器。最终，该函数返回 `None`，表示没有做任何操作，而是直接返回了 `None`。


```py
def serve(
    prompt_settings: Optional[Path],
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    install_plugin_deps: bool,
) -> None:
    """
    Starts an Agent Protocol compliant AutoGPT server, which creates a custom agent for
    every task.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    from autogpt.app.main import run_auto_gpt_server

    run_auto_gpt_server(
        prompt_settings=prompt_settings,
        debug=debug,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        memory_type=memory_type,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
        install_plugin_deps=install_plugin_deps,
    )


```py

这段代码是一个if语句，它会判断当前脚本是否作为主程序运行。如果是，那么执行if语句块内的内容，否则跳过if语句块。

在这段代码中，只有一个函数定义，该函数被称为__main__函数，它的作用是在脚本作为主程序运行时执行。因此，如果当前脚本作为主程序运行，将调用__main__函数中的代码。

总结起来，这段代码定义了一个if语句，根据当前脚本是否作为主程序运行来执行if语句块内的内容。


```py
if __name__ == "__main__":
    cli()

```