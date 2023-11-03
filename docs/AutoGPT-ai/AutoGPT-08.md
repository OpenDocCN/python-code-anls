# AutoGPT源码解析 8

# `autogpts/autogpt/autogpt/core/ability/builtins/file_operations.py`

This appears to be a function that implements a simple ability called "file\_reader" in the "unstructured.partition" package. This ability reads a file specified by the `filename` parameter and returns an `AbilityResult` object.

The function has the following signature:
```py
class FileReaderAbility(Ability):
   def __call__(self, filename: str) -> AbilityResult:
       if result := self._check_preconditions(filename):
           return result
```
The `__call__` method checks the file specified by the `filename` parameter and returns an `AbilityResult` object. The `_check_preconditions` method is called to ensure that the file can be found and read successfully. If the file can be found and read successfully, the function returns an `AbilityResult` object. If any errors occur, the function returns an `AbilityResult` object with a success status and the error message.

The `FileReaderAbility` class似乎 to be a subclass of the `Ability` class from the `unstructured.partition` package. This class provides several methods and properties that are specific to this ability.


```py
import logging
import os
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.workspace import Workspace


class ReadFile(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.ReadFile",
        ),
        packages_required=["unstructured"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._logger = logger
        self._workspace = workspace

    description: ClassVar[str] = "Read and parse all text from a file."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to read.",
        ),
    }

    def _check_preconditions(self, filename: str) -> AbilityResult | None:
        message = ""
        try:
            pass
        except ImportError:
            message = "Package charset_normalizer is not installed."

        try:
            file_path = self._workspace.get_path(filename)
            if not file_path.exists():
                message = f"File {filename} does not exist."
            if not file_path.is_file():
                message = f"{filename} is not a file."
        except ValueError as e:
            message = str(e)

        if message:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"filename": filename},
                success=False,
                message=message,
                data=None,
            )

    def __call__(self, filename: str) -> AbilityResult:
        if result := self._check_preconditions(filename):
            return result

        from unstructured.partition.auto import partition

        file_path = self._workspace.get_path(filename)
        try:
            elements = partition(str(file_path))
            # TODO: Lots of other potentially useful information is available
            #   in the partitioned file. Consider returning more of it.
            new_knowledge = Knowledge(
                content="\n\n".join([element.text for element in elements]),
                content_type=ContentType.TEXT,
                content_metadata={"filename": filename},
            )
            success = True
            message = f"File {file_path} read successfully."
        except IOError as e:
            new_knowledge = None
            success = False
            message = str(e)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"filename": filename},
            success=success,
            message=message,
            new_knowledge=new_knowledge,
        )


```

这是一个简单的 Python class，继承自 Pyrogram.Na君主，用于将文本写入到指定的文件中。通过 `__call__` 方法实现，当需要将文本写入文件时，先检查文件是否存在，如果文件不存在，则询问用户输入要写入的文件名和内容。在文件存在的情况下，创建文件目录并写入文件内容，最后返回成功结果。


```py
class WriteFile(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.WriteFile",
        ),
        packages_required=["unstructured"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._logger = logger
        self._workspace = workspace

    description: ClassVar[str] = "Write text to a file."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to write.",
        ),
        "contents": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The contents of the file to write.",
        ),
    }

    def _check_preconditions(
        self, filename: str, contents: str
    ) -> AbilityResult | None:
        message = ""
        try:
            file_path = self._workspace.get_path(filename)
            if file_path.exists():
                message = f"File {filename} already exists."
            if len(contents):
                message = f"File {filename} was not given any content."
        except ValueError as e:
            message = str(e)

        if message:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"filename": filename, "contents": contents},
                success=False,
                message=message,
                data=None,
            )

    def __call__(self, filename: str, contents: str) -> AbilityResult:
        if result := self._check_preconditions(filename, contents):
            return result

        file_path = self._workspace.get_path(filename)
        try:
            directory = os.path.dirname(file_path)
            os.makedirs(directory)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(contents)
            success = True
            message = f"File {file_path} written successfully."
        except IOError as e:
            success = False
            message = str(e)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"filename": filename},
            success=success,
            message=message,
        )

```

# `autogpts/autogpt/autogpt/core/ability/builtins/query_language_model.py`

这段代码使用了Python的logging库将日志信息输出到控制台。同时，它引入了typing库中的ClassVar类型，以便将来声明一个泛型类。

然后，它从autogpt库中导入了一些核心模块，包括Ability和AbilityConfiguration对象，以及AbilityResult对象。还从autogpt库中的planning模块中导入LanguageModelConfiguration对象。

接下来，它定义了一个名为PluginLocation的类，用于记录插件在应用程序中的位置。它还定义了一个名为PluginStorageFormat的类，用于记录插件存储库中的数据格式。

然后，它定义了一个名为ChatMessage的类，用于表示聊天消息，并从ChatModelProvider中获取消息模型的实现。接着，它定义了一个名为OpenAIModelName的类，用于表示OpenAI模型的名称。

最后，代码导入了来自autogpt和plannedefautogpt的函数和类。


```py
import logging
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    ModelProviderName,
    OpenAIModelName,
)
from autogpt.core.utils.json_schema import JSONSchema


```

这段代码定义了一个名为 QueryLanguageModel 的类，它继承自一个名为 Ability 的类。这个类的默认配置存储在 AbilityConfiguration 类中，这个类的位置存储在 PluginLocation 类中。这个类的配置包括指定查询语言模型的名称、提供模型名称、预热模型需要花费的时间以及模型在运行时需要设置的温度。

在 QueryLanguageModel 的构造函数中，传入了一个 ChatModelProvider 类型的参数，这个参数是用于提供模型输出结果的 ChatModelProvider。构造函数还传递了一个 logger 类型的参数，用于记录模型运行时的情况。

QueryLanguageModel 的主要方法是 __call__，这个方法接收一个查询参数并返回一个 AbilityResult 类的结果。在 __call__ 方法的实现中，首先创建一个模型响应，这个响应包含模型在运行时需要设置的参数，包括模型名称、提供模型名称、预热模型需要花费的时间以及运行时需要设置的温度。最后，将模型响应作为参数传入 ChatModelProvider 的 create_chat_completion 方法，并返回一个 AbilityResult 类的结果，这个结果包含模型的名称、参数和成功状态，以及模型在运行时返回的响应内容。


```py
class QueryLanguageModel(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.QueryLanguageModel",
        ),
        language_model_required=LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT3,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.9,
        ),
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        language_model_provider: ChatModelProvider,
    ):
        self._logger = logger
        self._configuration = configuration
        self._language_model_provider = language_model_provider

    description: ClassVar[str] = (
        "Query a language model."
        " A query should be a question and any relevant context."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A query for a language model. A query should contain a question and any relevant context.",
        )
    }

    async def __call__(self, query: str) -> AbilityResult:
        model_response = await self._language_model_provider.create_chat_completion(
            model_prompt=[ChatMessage.user(query)],
            functions=[],
            model_name=self._configuration.language_model_required.model_name,
        )
        return AbilityResult(
            ability_name=self.name(),
            ability_args={"query": query},
            success=True,
            message=model_response.response["content"],
        )

```

# `autogpts/autogpt/autogpt/core/ability/builtins/__init__.py`

这段代码的作用是创建一个新的能力（Ability），该能力基于自动语法处理引擎（AutogPT）的核心能力，并使用查询语言模型（QueryLanguageModel）来处理文本数据。这个新能力可能是用于某种特定的应用场景，也可能是为了满足某种需求而开发的。


```py
from autogpt.core.ability.builtins.create_new_ability import CreateNewAbility
from autogpt.core.ability.builtins.query_language_model import QueryLanguageModel

BUILTIN_ABILITIES = {
    QueryLanguageModel.name(): QueryLanguageModel,
}

```

# `autogpts/autogpt/autogpt/core/agent/base.py`

这段代码定义了一个名为 `Agent` 的类，属于 `abc` 模块。这个类提供了一些抽象方法，用于在初始化、从工作区中创建、确定下一个能力和进行复制等操作。下面是每个方法的详细解释：

- `__init__(self, *args, **kwargs):`：这是类的构造函数，用于初始化对象的属性和方法。将传递给构造函数的所有参数传递给对象的属性。

- `from_workspace(cls, workspace_path: Path, logger: logging.Logger) -> Agent:`：这是类的一个抽象方法，用于从工作区中创建一个 `Agent` 对象。将 `workspace_path` 和 `logger` 参数传递给构造函数。构造函数返回一个指向新对象的引用。

- `__repr__(self)`：这是类的分解器方法，用于返回对象的唯一字符串表示。这个方法返回一个字符串，描述了对象的属性和方法，但没有使用任何异常或新格式化符。

- `Agent(args=None, kwargs=None)`：这是类的普通方法，用于创建一个自定义的 `Agent` 对象。这个方法没有参数，因为它是一个抽象方法，需要客户端传递参数来初始化对象的属性和方法。

- `determine_next_ability(self)`：这是类的另一个抽象方法，用于确定下一个能力。这个方法没有参数，因为它需要客户端在每次调用时传递需要评估的能力。

- `__init_subprocess(self)`：这是类的另一个抽象方法，用于在运行时初始化子进程。这个方法需要在一个运行时初始化子进程，因此它需要一个 `self` 参数，用于访问客户端的参数。


```py
import abc
import logging
from pathlib import Path


class Agent(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @classmethod
    @abc.abstractmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "Agent":
        ...

    @abc.abstractmethod
    async def determine_next_ability(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...

```

# `autogpts/autogpt/autogpt/core/agent/simple.py`

这段代码使用了Python的包和库，实现了以下功能：

1. 引入日志记录器（logging）以进行日志输出。
2. 从datetime包中导入datetime对象，用于创建描述性时间戳（datetime）类。
3. 从pathlib包中导入Path对象，用于创建文件或目录路径。
4. 从typing包中导入Any类型，表示输入数据可以是任何类型。
5. 从pydantic包中导入BaseModel类，用于定义模型和数据类型。
6. 从autogpt包中导入AbilityRegistrySettings、AbilityResult、SimpleAbilityRegistry和Agent类，实现自定义能力代理和注册功能。
7. 从autogpt的配置文件中设置AbilityRegistrySettings，指定能力注册中心。
8. 从autogpt的配置文件中设置SystemConfiguration和SystemSettings，指定系统配置和设置。
9. 从autogpt的内存管理器中设置SimpleMemory，指定简单的内存管理。


```py
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autogpt.core.ability import (
    AbilityRegistrySettings,
    AbilityResult,
    SimpleAbilityRegistry,
)
from autogpt.core.agent.base import Agent
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory import MemorySettings, SimpleMemory
```

这段代码定义了一个AgentSystem类，该类包含以下几个类：PlannerSettings、SimplePlanner、Task、TaskStatus以及一个未知类型的插件位置。

具体来说，这段代码定义了一个AgentSystem类，它继承自SystemConfiguration类。这个类包含四个属性：ability_registry、memory、openai_provider和planning，它们分别表示自定义能力注册、内存、OpenAI提供者以及规划器设置。同时，它还包含一个Workspace类和一个SimpleWorkspace类，它们用于处理模型的部署和规划。

这里使用了AutogPT的两个核心模块：Simple和Planning。Simple模块提供了一个插件机制，允许您创建自己的自定义任务、插件和能力，并使用它们来简化模型的训练和部署。Planning模块则负责生成对抗网络（GAN）环境，以在训练期间生成对抗性样本，并帮助您设置训练参数。

这段代码的目的是定义一个AgentSystem类，该类包含了一些必要的属性以及一个SimpleWorkspace类和一个未知类型的插件位置。它还继承了SystemConfiguration类，该类包含了所有必要的属性和方法，以便您创建一个完整的Agent系统。


```py
from autogpt.core.planning import PlannerSettings, SimplePlanner, Task, TaskStatus
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.resource.model_providers import (
    CompletionModelFunction,
    OpenAIProvider,
    OpenAISettings,
)
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings


class AgentSystems(SystemConfiguration):
    ability_registry: PluginLocation
    memory: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation
    workspace: PluginLocation


```

这段代码定义了一个名为AgentConfiguration的类，继承自SystemConfiguration类。AgentConfiguration类包含了一些与配置相关的属性，如循环计数、最大任务循环计数、创建时间、名称、角色和目标等。

在AgentConfiguration类中，还包含一个包含配置对象的属性，如SystemConfiguration和AgentConfiguration，以及一个包含所有与系统相关的对象的类，如AgentSystems。

接着，定义了一个名为AgentSystemSettings的类，继承自SystemSettings类。AgentSystemSettings类包含一个与配置对象的属性，如AgentConfiguration，以及一个包含能力注册文件的类，如AbilityRegistrySettings。

最后，定义了一个名为AgentSettings的类，继承自BaseModel类。AgentSettings类包含一个指向AgentSystemSettings对象的引用，一个包含配置设置的内存设置，以及一个OpenAISetting和一个PlannerSetting对象，用于配置工作区。

在这段代码中，还包含一个update_agent_name_and_goals方法，用于在创建模型时更新Agent设置类中对象的属性。


```py
class AgentConfiguration(SystemConfiguration):
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    name: str
    role: str
    goals: list[str]
    systems: AgentSystems


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    agent: AgentSystemSettings
    ability_registry: AbilityRegistrySettings
    memory: MemorySettings
    openai_provider: OpenAISettings
    planning: PlannerSettings
    workspace: WorkspaceSettings

    def update_agent_name_and_goals(self, agent_goals: dict) -> None:
        self.agent.configuration.name = agent_goals["agent_name"]
        self.agent.configuration.role = agent_goals["agent_role"]
        self.agent.configuration.goals = agent_goals["agent_goals"]


```

This is a class that appears to be responsible for managing an agent's settings and configuring an agent's workspace. The agent settings are defined in the `AgentSettings` class and the agent planner is a `SimplePlanner` that is responsible for determining the agent's name and goals. The class also has a logger that is used to log information about the agent's configuration and system state.

The `provision_agent` method is used to configure the agent's workspace and make the agent available for use. The class also has a `_get_system_instance` method that is used to retrieve a system instance by its name.


```py
class SimpleAgent(Agent, Configurable):
    default_settings = AgentSystemSettings(
        name="simple_agent",
        description="A simple agent.",
        configuration=AgentConfiguration(
            name="Entrepreneur-GPT",
            role=(
                "An AI designed to autonomously develop and run businesses with "
                "the sole goal of increasing your net worth."
            ),
            goals=[
                "Increase net worth",
                "Grow Twitter Account",
                "Develop and manage multiple businesses autonomously",
            ],
            cycle_count=0,
            max_task_cycle_count=3,
            creation_time="",
            systems=AgentSystems(
                ability_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.ability.SimpleAbilityRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.SimpleMemory",
                ),
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.resource.model_providers.OpenAIProvider",
                ),
                planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.planning.SimplePlanner",
                ),
                workspace=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.workspace.SimpleWorkspace",
                ),
            ),
        ),
    )

    def __init__(
        self,
        settings: AgentSystemSettings,
        logger: logging.Logger,
        ability_registry: SimpleAbilityRegistry,
        memory: SimpleMemory,
        openai_provider: OpenAIProvider,
        planning: SimplePlanner,
        workspace: SimpleWorkspace,
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._ability_registry = ability_registry
        self._memory = memory
        # FIXME: Need some work to make this work as a dict of providers
        #  Getting the construction of the config to work is a bit tricky
        self._openai_provider = openai_provider
        self._planning = planning
        self._workspace = workspace
        self._task_queue = []
        self._completed_tasks = []
        self._current_task = None
        self._next_ability = None

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "SimpleAgent":
        agent_settings = SimpleWorkspace.load_agent_settings(workspace_path)
        agent_args = {}

        agent_args["settings"] = agent_settings.agent
        agent_args["logger"] = logger
        agent_args["workspace"] = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger,
        )
        agent_args["openai_provider"] = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger,
        )
        agent_args["planning"] = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["openai_provider"]},
        )
        agent_args["memory"] = cls._get_system_instance(
            "memory",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
        )

        agent_args["ability_registry"] = cls._get_system_instance(
            "ability_registry",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
            memory=agent_args["memory"],
            model_providers={"openai": agent_args["openai_provider"]},
        )

        return cls(**agent_args)

    async def build_initial_plan(self) -> dict:
        plan = await self._planning.make_initial_plan(
            agent_name=self._configuration.name,
            agent_role=self._configuration.role,
            agent_goals=self._configuration.goals,
            abilities=self._ability_registry.list_abilities(),
        )
        tasks = [Task.parse_obj(task) for task in plan.parsed_result["task_list"]]

        # TODO: Should probably do a step to evaluate the quality of the generated tasks,
        #  and ensure that they have actionable ready and acceptance criteria

        self._task_queue.extend(tasks)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        self._task_queue[-1].context.status = TaskStatus.READY
        return plan.parsed_result

    async def determine_next_ability(self, *args, **kwargs):
        if not self._task_queue:
            return {"response": "I don't have any tasks to work on right now."}

        self._configuration.cycle_count += 1
        task = self._task_queue.pop()
        self._logger.info(f"Working on task: {task}")

        task = await self._evaluate_task_and_add_context(task)
        next_ability = await self._choose_next_ability(
            task,
            self._ability_registry.dump_abilities(),
        )
        self._current_task = task
        self._next_ability = next_ability.parsed_result
        return self._current_task, self._next_ability

    async def execute_next_ability(self, user_input: str, *args, **kwargs):
        if user_input == "y":
            ability = self._ability_registry.get_ability(
                self._next_ability["next_ability"]
            )
            ability_response = await ability(**self._next_ability["ability_arguments"])
            await self._update_tasks_and_memory(ability_response)
            if self._current_task.context.status == TaskStatus.DONE:
                self._completed_tasks.append(self._current_task)
            else:
                self._task_queue.append(self._current_task)
            self._current_task = None
            self._next_ability = None

            return ability_response.dict()
        else:
            raise NotImplementedError

    async def _evaluate_task_and_add_context(self, task: Task) -> Task:
        """Evaluate the task and add context to it."""
        if task.context.status == TaskStatus.IN_PROGRESS:
            # Nothing to do here
            return task
        else:
            self._logger.debug(f"Evaluating task {task} and adding relevant context.")
            # TODO: Look up relevant memories (need working memory system)
            # TODO: Evaluate whether there is enough information to start the task (language model call).
            task.context.enough_info = True
            task.context.status = TaskStatus.IN_PROGRESS
            return task

    async def _choose_next_ability(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
    ):
        """Choose the next ability to use for the task."""
        self._logger.debug(f"Choosing next ability for task {task}.")
        if task.context.cycle_count > self._configuration.max_task_cycle_count:
            # Don't hit the LLM, just set the next action as "breakdown_task" with an appropriate reason
            raise NotImplementedError
        elif not task.context.enough_info:
            # Don't ask the LLM, just set the next action as "breakdown_task" with an appropriate reason
            raise NotImplementedError
        else:
            next_ability = await self._planning.determine_next_ability(
                task, ability_specs
            )
            return next_ability

    async def _update_tasks_and_memory(self, ability_result: AbilityResult):
        self._current_task.context.cycle_count += 1
        self._current_task.context.prior_actions.append(ability_result)
        # TODO: Summarize new knowledge
        # TODO: store knowledge and summaries in memory and in relevant tasks
        # TODO: evaluate whether the task is complete

    def __repr__(self):
        return "SimpleAgent()"

    ################################################################
    # Factory interface for agent bootstrapping and initialization #
    ################################################################

    @classmethod
    def build_user_configuration(cls) -> dict[str, Any]:
        """Build the user's configuration."""
        configuration_dict = {
            "agent": cls.get_user_config(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]
        for system_name, system_location in system_locations.items():
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.get_user_config()
        configuration_dict = _prune_empty_dicts(configuration_dict)
        return configuration_dict

    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Processing agent system configuration.")
        configuration_dict = {
            "agent": cls.build_agent_configuration(
                user_configuration.get("agent", {})
            ).dict(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            logger.debug(f"Compiling configuration for system {system_name}")
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.build_agent_configuration(
                user_configuration.get(system_name, {})
            ).dict()

        return AgentSettings.parse_obj(configuration_dict)

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ) -> dict:
        logger.debug("Loading OpenAI provider.")
        provider: OpenAIProvider = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger=logger,
        )
        logger.debug("Loading agent planner.")
        agent_planner: SimplePlanner = cls._get_system_instance(
            "planning",
            agent_settings,
            logger=logger,
            model_providers={"openai": provider},
        )
        logger.debug("determining agent name and goals.")
        model_response = await agent_planner.decide_name_and_goals(
            user_objective,
        )

        return model_response.parsed_result

    @classmethod
    def provision_agent(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ):
        agent_settings.agent.configuration.creation_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        workspace: SimpleWorkspace = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger=logger,
        )
        return workspace.setup_workspace(agent_settings, logger)

    @classmethod
    def _get_system_instance(
        cls,
        system_name: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        *args,
        **kwargs,
    ):
        system_locations = agent_settings.agent.configuration.systems.dict()

        system_settings = getattr(agent_settings, system_name)
        system_class = SimplePluginService.get_plugin(system_locations[system_name])
        system_instance = system_class(
            system_settings,
            *args,
            logger=logger.getChild(system_name),
            **kwargs,
        )
        return system_instance


```

这段代码定义了一个名为 `_prune_empty_dicts` 的函数，用于对传入的 dictionary 进行操作。其作用是，对于一个 nested dictionary(也就是一个嵌套的 dictionary)，如果它的叶子节点(即字典的值)仅包含空字典，则从字典的根节点(即字典的键)开始遍历，将所有含有空字典的子节点(也就是字典的值)也删除。最终，返回一个 pruned(即去除空字典和不含有空字典的子节点的) dictionary。

该函数接收一个 dictionary 作为参数，并返回一个 pruned 的 dictionary。函数内部首先对传入的 dictionary 进行迭代，将所有含有空字典的键值对移除，并将这些移除后的字典作为新的键值对返回。在内部迭代过程中，如果移除后的字典不 empty，则将其添加到结果 dictionary 中。最终，返回结果。


```py
def _prune_empty_dicts(d: dict) -> dict:
    """
    Prune branches from a nested dictionary if the branch only contains empty dictionaries at the leaves.

    Args:
        d: The dictionary to prune.

    Returns:
        The pruned dictionary.
    """
    pruned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            pruned_value = _prune_empty_dicts(value)
            if (
                pruned_value
            ):  # if the pruned dictionary is not empty, add it to the result
                pruned[key] = pruned_value
        else:
            pruned[key] = value
    return pruned

```

# `autogpts/autogpt/autogpt/core/agent/__init__.py`

这段代码表示了一个使用LLMProvider进行指导的自主化实体，该实体使用简单Agent设置，并具有自主性。


```py
"""The Agent is an autonomouos entity guided by a LLM provider."""
from autogpt.core.agent.base import Agent
from autogpt.core.agent.simple import AgentSettings, SimpleAgent

```

# `autogpts/autogpt/autogpt/core/configuration/schema.py`

这段代码定义了一个名为 `UserConfigurable` 的类，用于表示应用程序配置中的用户可配置项。这个类使用了 `pydantic` 库中的 `Field` 类来定义数据类型。

`UserConfigurable` 的定义中包含一个静态方法 `*args, **kwargs, user_configurable=True`，这个方法使用了 `functools.wraps` 函数，表示将 `Field` 中的参数包装起来，并且在参数中添加了 `user_configurable=True` 参数，使得这个方法可以被当做普通的函数来使用。

接着，定义了一个 `SystemConfiguration` 类，它继承自 `BaseModel` 类，表示应用程序配置的整体设置。这个类中包含一个方法 `get_user_config`，这个方法返回了一个字典，其中包含了一些用户可配置的键值对，例如 `database_uri`、`log_level` 等。

另外，这个类中还定义了一个 `Config` 类，继承自 `typing.TypeVar` 类，表示这个类可以接受各种不同类型的配置。这个类中包含了一个方法 `use_enum_values`，表示如果这个应用程序支持使用 Enum 值，就使用 Enum 值，否则使用字符串字面量。另外，这个类中还定义了一个 `extra` 属性，表示额外的配置信息，这里使用了字符串 `"forbid"`。


```py
import abc
import functools
import typing
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


@functools.wraps(Field)
def UserConfigurable(*args, **kwargs):
    return Field(*args, **kwargs, user_configurable=True)
    # TODO: use this to auto-generate docs for the application configuration


class SystemConfiguration(BaseModel):
    def get_user_config(self) -> dict[str, Any]:
        return _get_user_config_fields(self)

    class Config:
        extra = "forbid"
        use_enum_values = True


```

这段代码定义了一个名为`SystemSettings`的类，它是一个继承自`BaseModel`的基类。`SystemSettings`类有两个属性，一个是`name`，表示设置的系统名称；另一个是`description`，表示设置的系统描述。

在这个类中，定义了一个名为`Config`的类，它继承自`abc.ABC`。`Configable`是`ABC`接口的实现类，它声明了一个名为`prefix`的属性，表示设置集中使用的键的前缀；还有一个名为`default_settings`的静态方法，用于获取默认设置对象。

接着，定义了一个名为`Configuration`的类，它实现了`Generic`接口，声明了一个名为`S`的类型参数。`Configuration`类有两个方法，一个是`get_user_config`，用于获取用户配置；另一个是`build_agent_configuration`，用于构建特定对象的配置。在`build_agent_configuration`方法中，使用了`deep_update`函数，将`Configuration`中的`default_settings`属性与传入的配置字典进行合并，并返回合并后的配置。

最后，在`SystemSettings`和`Configuration`类中，使用`@classmethod`装饰器来获取类的`__init__`方法，实现了一对多的初始化。在`__init__`方法中，创建了一个`Config`实例，并将其设置为类的默认设置。


```py
class SystemSettings(BaseModel):
    """A base class for all system settings."""

    name: str
    description: str

    class Config:
        extra = "forbid"
        use_enum_values = True


S = TypeVar("S", bound=SystemSettings)


class Configurable(abc.ABC, Generic[S]):
    """A base class for all configurable objects."""

    prefix: str = ""
    default_settings: typing.ClassVar[S]

    @classmethod
    def get_user_config(cls) -> dict[str, Any]:
        return _get_user_config_fields(cls.default_settings)

    @classmethod
    def build_agent_configuration(cls, configuration: dict) -> S:
        """Process the configuration for this object."""

        defaults = cls.default_settings.dict()
        final_configuration = deep_update(defaults, configuration)

        return cls.default_settings.__class__.parse_obj(final_configuration)


```

这段代码定义了一个名为 `_get_user_config_fields` 的函数，它接受一个名为 `instance` 的 Pydantic模型实例作为参数，返回该实例的用户配置字段字典。

函数内部通过遍历 `instance` 的所有属性，并检查每个属性是否属于 `__dict__` 字典，如果是，就获取该属性的用户配置字段信息。如果属性是 Pydantic 模型中的字段，并且该字段定义了一个 `SystemConfiguration` 类，那么函数将获取该字段的使用者配置。如果属性是一个或多个系统配置，那么函数将获取每个配置的使用者配置，如果属性是一个或多个列表，并且所有列表都是系统配置，那么函数将尝试获取每个列表的第一个元素，如果可以获取到，将该元素的配置作为结果返回。如果属性是一个字典，并且所有键都是系统配置，那么函数将尝试获取每个键的使用者配置，并返回一个字典，其中键是键的定义，值是键的值。

函数最终返回一个字典，其中包含所有与 `instance` 相关的用户配置字段。


```py
def _get_user_config_fields(instance: BaseModel) -> dict[str, Any]:
    """
    Get the user config fields of a Pydantic model instance.

    Args:
        instance: The Pydantic model instance.

    Returns:
        The user config fields of the instance.
    """
    user_config_fields = {}

    for name, value in instance.__dict__.items():
        field_info = instance.__fields__[name]
        if "user_configurable" in field_info.field_info.extra:
            user_config_fields[name] = value
        elif isinstance(value, SystemConfiguration):
            user_config_fields[name] = value.get_user_config()
        elif isinstance(value, list) and all(
            isinstance(i, SystemConfiguration) for i in value
        ):
            user_config_fields[name] = [i.get_user_config() for i in value]
        elif isinstance(value, dict) and all(
            isinstance(i, SystemConfiguration) for i in value.values()
        ):
            user_config_fields[name] = {
                k: v.get_user_config() for k, v in value.items()
            }

    return user_config_fields


```



该函数`deep_update`接受两个参数：`original_dict`和`update_dict`。`original_dict`是要更新的字典，`update_dict`是要更新的字典。

函数首先遍历`update_dict`中的所有键值对，然后将其中的键与`original_dict`中的键进行比较。如果两个键相同，并且它们都是字典类型，则函数递归地调用`deep_update`函数本身，更新`original_dict`中的相应键。否则，函数直接修改`original_dict`中的相应键。

函数返回`original_dict`作为更新后的字典。


```py
def deep_update(original_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update a dictionary.

    Args:
        original_dict (dict): The dictionary to be updated.
        update_dict (dict): The dictionary to update with.

    Returns:
        dict: The updated dictionary.
    """
    for key, value in update_dict.items():
        if (
            key in original_dict
            and isinstance(original_dict[key], dict)
            and isinstance(value, dict)
        ):
            original_dict[key] = deep_update(original_dict[key], value)
        else:
            original_dict[key] = value
    return original_dict

```

# `autogpts/autogpt/autogpt/core/configuration/__init__.py`

这段代码是一个语言模型的配置设置类，用于配置所有代理子系统。它通过导入来自 autogpt 库的 Configurable、SystemConfiguration、SystemSettings 和 UserConfigurable 类来提供对配置设置的访问。

具体来说，它实现了以下几个方法：

- Configurable：用于设置应用程序的配置设置类，可以根据需要进行扩展。
- SystemConfiguration：用于设置整个系统的配置设置，包括基础设施、应用程序等。
- SystemSettings：用于设置系统的设置，例如日期和时间、货币等。
- UserConfigurable：用于设置用户的配置设置，例如用户密码、用户偏好等。

此外，还实现了两个辅助方法：

- __init__：用于初始化设置，在配置文件中使用这个方法时，会读取配置文件中的设置。
- update_settings：用于更新设置，根据应用程序的当前状态更新设置。

通过这些方法，可以对代理子系统的配置进行设置，从而实现对整个系统的配置管理。


```py
"""The configuration encapsulates settings for all Agent subsystems."""
from autogpt.core.configuration.schema import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)

```

# `autogpts/autogpt/autogpt/core/memory/base.py`

这段代码定义了一个名为`MessageHistory`的类，它继承了`abc.ABC`类(代表抽象窗口模式)。在这个类中，使用了两个抽象类`abc.ABC`和`abc.MessageHistory`作为其父类，并定义了一个无参构造函数。

由于`MessageHistory`类中没有使用`abc.ABC`中的任何方法或定义，因此它的作用是定义了一个类`MessageHistory`，继承自抽象窗口模式类，即用于在应用程序中管理消息历史的类。


```py
import abc


class Memory(abc.ABC):
    pass


class MemoryItem(abc.ABC):
    pass


class MessageHistory(abc.ABC):
    pass

```

# `autogpts/autogpt/autogpt/core/memory/simple.py`

这段代码使用了两个库：json 和 logging。同时引入了三个类：MemoryConfiguration、MemorySettings 和 SystemConfiguration、SystemSettings 和 Configurable。

首先，这段代码定义了一个名为 MemoryConfiguration 的类，它继承自 SystemConfiguration 类。这个类的实例化通常在应用程序启动时进行，并负责管理应用程序的内存设置。具体来说，它将接收系统设置（通过 SystemSettings 类）作为参数，并在创建时设置它们。

接着，定义了一个名为 MemorySettings 的类，它继承自 SystemSettings 类。这个类的实例化通常在内存配置确定后进行，并设置 MemoryConfiguration 实例的参数。它将接收 MemoryConfiguration 实例传递给它的参数，并设置它们。

最后，定义了一个名为 Configurable 的类，它继承自 Configuration 类。这个类的实例化通常在应用程序启动时进行，并设置整个应用程序的配置。它将接收一个 Configurable 实例作为参数，并设置它。

总之，这段代码的作用是设置一个应用程序的内存设置，包括设置内存配置和初始化内存设置。


```py
import json
import logging

from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory
from autogpt.core.workspace import Workspace


class MemoryConfiguration(SystemConfiguration):
    pass


class MemorySettings(SystemSettings):
    configuration: MemoryConfiguration


```



这段代码定义了一个名为`MessageHistory`的类，用于保存消息历史记录。在`__init__`方法中，创建了一个包含先前消息记录的列表，并将其存储在`_message_history`成员变量中。

接着定义了一个名为`SimpleMemory`的类，该类继承自`Memory`和`Configurable`类。`SimpleMemory`类具有一个`default_settings`属性，用于设置默认的设置，然后还定义了一个`__init__`方法，用于初始化`SimpleMemory`对象，并从传入的`settings`参数中加载配置，最后将加载的消息历史记录存储在`_message_history`属性中。

最后，`SimpleMemory`类还有一个名为`_load_message_history`的静态方法，用于从工作区中加载消息历史记录。


```py
class MessageHistory:
    def __init__(self, previous_message_history: list[str]):
        self._message_history = previous_message_history


class SimpleMemory(Memory, Configurable):
    default_settings = MemorySettings(
        name="simple_memory",
        description="A simple memory.",
        configuration=MemoryConfiguration(),
    )

    def __init__(
        self,
        settings: MemorySettings,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._message_history = self._load_message_history(workspace)

    @staticmethod
    def _load_message_history(workspace: Workspace):
        message_history_path = workspace.get_path("message_history.json")
        if message_history_path.exists():
            with message_history_path.open("r") as f:
                message_history = json.load(f)
        else:
            message_history = []
        return MessageHistory(message_history)

```

# `autogpts/autogpt/autogpt/core/memory/__init__.py`

这段代码定义了一个函数，名为“manages_agent_long_term_memory”，它接受一个参数，名为“memory_settings”，也被称作“MemorySettings”。

这个函数的作用是管理一个长期的记忆，这个记忆是由一个名为“SimpleMemory”的类实现的。这个函数调用了“SimpleMemory”类的“init_记忆设置”方法，并传递了一个具体的设置，这个方法接受一个参数，称为“long_term_memory_策略”，这个参数指定了长期记忆的策略，包括对数据的访问频率、数据存储策略等。


```py
"""The memory subsystem manages the Agent's long-term memory."""
from autogpt.core.memory.base import Memory
from autogpt.core.memory.simple import MemorySettings, SimpleMemory

```

# `autogpts/autogpt/autogpt/core/planning/base.py`

这段代码定义了一个名为 "Planner" 的类，该类继承自 "abc.ABC" 类（使用 "ABC" 作为后缀）。

该类包含一个名为 "decide_name_and_goals" 的静态方法，该方法接受一个字符串类型的用户目标和一个空字符串作为参数，并返回一个 "LanguageModelResponse" 对象。

该方法的实现部分没有定义任何方法，因此该类不能直接使用。需要定义 "Planner" 类的具体行为才能理解该类的功能。


```py
# class Planner(abc.ABC):
#     """Manages the agent's planning and goal-setting by constructing language model prompts."""
#
#     @staticmethod
#     @abc.abstractmethod
#     async def decide_name_and_goals(
#         user_objective: str,
#     ) -> LanguageModelResponse:
#         """Decide the name and goals of an Agent from a user-defined objective.
#
#         Args:
#             user_objective: The user-defined objective for the agent.
#
#         Returns:
#             The agent name and goals as a response from the language model.
```

这段代码定义了一个名为 "plan" 的抽象方法，属于一个名为 "abc" 的类。这个方法接收一个名为 "context" 的对象，它是 "PlanningContext" 类的实例。这个方法的返回值类型是 "LanguageModelResponse"，表示它返回一个语言模型，使代理程序能够对上下文产生的语言进行理解和产生回复。

在方法的实现部分，定义了一个名为 "async def plan(self, context: PlanningContext) -> LanguageModelResponse："的异步函数。这个函数使用 "@abc.abstractmethod" 注解来定义一个抽象方法，意味着这个方法可以以异步方式进行实现。

在方法的参数列表中，定义了一个名为 "context" 的参数，它的类型为 "PlanningContext" 类，这个类似乎保存了代理程序当前的上下文信息。方法的返回值类型是 "LanguageModelResponse"，这个类型似乎表示要返回一个语言模型。

方法的实现部分没有对参数或返回值做出任何具体的实现，而是定义了一个名为 "async def plan(self, context: PlanningContext) -> LanguageModelResponse："的异步函数，它将在未来的需要时执行实际的计划操作。


```py
#
#         """
#         ...
#
#     @abc.abstractmethod
#     async def plan(self, context: PlanningContext) -> LanguageModelResponse:
#         """Plan the next ability for the Agent.
#
#         Args:
#             context: A context object containing information about the agent's
#                        progress, result, memories, and feedback.
#
#
#         Returns:
#             The next ability the agent should take along with thoughts and reasoning.
```

这段代码定义了一个名为“reflect”的抽象方法，属于名为“abc”的类。这个方法的参数是一个名为“ReflectionContext”的对象，它存储了关于一个预定能力（planned ability）的信息。

该方法返回一个名为“LanguageModelResponse”的对象，这个对象可能是用于表示规划或描述性能的一个响应。

方法“reflect”的作用是让类“abc”中的对象能够对预定能力进行反思，并提供自我批评。这个方法的实现可能涉及到对预定能力的评估，以及根据评估结果采取的行动。


```py
#
#         """
#         ...
#
#     @abc.abstractmethod
#     def reflect(
#         self,
#         context: ReflectionContext,
#     ) -> LanguageModelResponse:
#         """Reflect on a planned ability and provide self-criticism.
#
#
#         Args:
#             context: A context object containing information about the agent's
#                        reasoning, plan, thoughts, and criticism.
```

这段代码是一个 Python 语言中的函数，主要目的是计算并返回一个表示机器人计划有效性的分数。该分数基于一个自定义的分数计算模型，该模型根据机器人计划的最终目标与当前状态之间的距离来评估计划的有效性。具体来说，该函数将以下两个参数作为输入：

- 机器人计划的最终目标(在选项中使用“t”表示)
- 机器人当前的状态(在选项中使用“s”表示)

函数首先使用这些参数计算一个分数，该分数基于以下公式：

- 分数 = 0.2 * 最终目标距离 + 0.3 * 当前状态的枸表现

其中，0.2 是分数的系数，用于调整最终目标对当前状态的重视程度；0.3 是分数的系数，用于调整状态对最终目标的响应速度。

然后，函数将这些分数汇总到一个新分数中，该新分数基于以下公式：

- 新分数 = (最终目标分数 + 0.6 * 当前状态的枸表现)的平方根

最后，函数返回计算得到的新分数，作为机器人计划有效性的指标。


```py
#
#         Returns:
#             Self-criticism about the agent's plan.
#
#         """
#         ...

```

# `autogpts/autogpt/autogpt/core/planning/schema.py`

这段代码使用了Python的enum模块来定义了一个名为TaskType的枚举类型，它包含了7个枚举值，分别代表了 research、write、edit、code、design、test 和 plan 这8个任务类型。

该代码还使用了typing模块中的Optional，用于定义了输入数据的可选类型。

该代码最后从另一个名为AbilityResult的类中导入了一个名为AbilityResult的函数，该函数没有定义任何参数，并返回一个AbilityResult类型的结果。


```py
import enum
from typing import Optional

from pydantic import BaseModel, Field

from autogpt.core.ability.schema import AbilityResult


class TaskType(str, enum.Enum):
    RESEARCH = "research"
    WRITE = "write"
    EDIT = "edit"
    CODE = "code"
    DESIGN = "design"
    TEST = "test"
    PLAN = "plan"


```

这段代码定义了一个名为 "TaskStatus" 的枚举类型，它定义了四个值为 "backlog"、"ready"、"in_progress" 和 "done" 的枚举类型。

接着，定义了一个名为 "TaskContext" 的类，它继承自 "BaseModel" 类。在这个类中，定义了一些字段，包括一个名为 "cycle_count" 的整数类型字段，一个名为 "status" 的 "TaskStatus" 类型的字段，一个名为 "parent" 的可选整数类型字段，一个名为 "prior_actions" 的列表类型字段，一个名为 "memories" 的列表类型字段，一个名为 "user_input" 的列表类型字段，一个名为 "supplementary_info" 的列表类型字段，一个名为 "enough_info" 的布尔类型字段。

此外，还定义了一个名为 "Task" 的接口类型，它包含一个名为 "parent" 的字段和一个名为 "status" 的字段。

最后，在 "TaskStatus" 和 "Task" 类之间，使用了继承关系。可以理解为，这个 "TaskContext" 类继承了 "Task" 类中的所有字段和 "TaskStatus" 枚举类型字段。


```py
class TaskStatus(str, enum.Enum):
    BACKLOG = "backlog"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class TaskContext(BaseModel):
    cycle_count: int = 0
    status: TaskStatus = TaskStatus.BACKLOG
    parent: Optional["Task"] = None
    prior_actions: list[AbilityResult] = Field(default_factory=list)
    memories: list = Field(default_factory=list)
    user_input: list[str] = Field(default_factory=list)
    supplementary_info: list[str] = Field(default_factory=list)
    enough_info: bool = False


```

这段代码定义了一个名为 "Task" 的类，其继承自 "BaseModel" 类。

这个类有以下属性和方法：

- "objective"(目标)：一个字符串类型的成员变量，用于指定任务的目标。
- "type"(类型)：一个字符串类型的成员变量，用于指定任务的类型。这个变量被 FIXME 指出，因为 gpt 并不遵守这个类中定义的枚举类型约束。
- "priority"(优先级)：一个整数类型的成员变量，用于指定任务的优先级。
- "ready_criteria"(准备条件)：一个字符串类型的列表，用于指定任务准备好进入客户端的 criteria。
- "acceptance_criteria"(验收条件)：一个字符串类型的列表，用于指定任务是否符合客户端的要求。
- "context"(上下文)：一个名为 "TaskContext" 的成员变量，用于指定任务执行的上下文。这个变量被描述为 "default_factory=TaskContext)"，意味着如果上下文对象未被创建，则会使用 "TaskContext" 作为默认的上下文。

最后，还有一行代码 "TaskContext.update_forward_refs()"，用于解决任务和 "TaskContext" 之间的循环引用问题，因为在定义了这两个模型之后，它们之间就无法相互引用。


```py
class Task(BaseModel):
    objective: str
    type: str  # TaskType  FIXME: gpt does not obey the enum parameter in its schema
    priority: int
    ready_criteria: list[str]
    acceptance_criteria: list[str]
    context: TaskContext = Field(default_factory=TaskContext)


# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
TaskContext.update_forward_refs()

```

# `autogpts/autogpt/autogpt/core/planning/simple.py`

这段代码是一个自动执行脚本，它的作用是：

1. 导入logging、platform、time模块，用于输出日志信息和运行时间。
2. 导入distro模块，用于下载并安装指定名称的Linux发行版。
3. 定义一个名为Configurable的类，用于设置自动执行的配置。
4. 定义一个名为SystemConfiguration的类，用于设置自动执行的系统配置。
5. 定义一个名为SystemSettings的类，用于设置自动执行的系统设置。
6. 定义一个名为UserConfigurable的类，用于设置自动执行的用户配置。
7. 导入autogpt.core.planning模块，用于执行自动执行的任务。
8. 导入autogpt.core.planning.schema模块，用于定义自动执行任务的schema。
9. 定义一个名为Task的类，用于表示自动执行的任务。
10. 定义一个名为PromptStrategy的类，用于执行自动执行的任务。

该脚本在运行时，会根据用户设置的自动执行配置，下载并安装指定的Linux发行版，然后设置系统的配置和设置，接着从用户设置的配置文件中读取用户设置，然后下载并加载自动执行任务，最后根据任务设置好的prompt strategies执行相应的任务。


```py
import logging
import platform
import time

import distro

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.planning import prompt_strategies
from autogpt.core.planning.schema import Task
from autogpt.core.prompting import PromptStrategy
```

这段代码定义了一个名为`LanguageModelConfiguration`的类，它继承了`SystemConfiguration`类。这个类用于配置一个自然语言处理模型，包括模型的名称、提供者、温度等。这些配置信息用于在运行时创建一个语言模型实例。

具体来说，这个类包含以下成员变量：

- `model_name`：模型名称，是一个字符串。
- `provider_name`：提供模型的名称，是一个字符串。
- `temperature`：模型温度，是一个浮点数。

这个类的`SystemConfiguration`继承继承自`autogpt.core.resource.model_providers.ModelProvider`类，所以在创建模型实例时，需要指定 provider。


```py
from autogpt.core.prompting.schema import LanguageModelClassification
from autogpt.core.resource.model_providers import (
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderName,
    OpenAIModelName,
)
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
from autogpt.core.workspace import Workspace


class LanguageModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    model_name: str = UserConfigurable()
    provider_name: ModelProviderName = UserConfigurable()
    temperature: float = UserConfigurable()


```

这段代码定义了一个 `PromptStrategiesConfiguration` 类，它是 `SystemConfiguration` 类的子类。这个类有一个 `name_and_goals` 字段和一个 `initial_plan` 字段，它们都是 `PromptStrategiesConfiguration` 类的实例。接着，这个类有一个 `next_ability` 字段，它也是 `PromptStrategiesConfiguration` 类的实例。

接着，这个类定义了一个 `PlannerConfiguration` 类，它是 `SystemConfiguration` 类的子类。这个类有一个 `models` 字段，它是一个字典，包含两个键：`LanguageModelClassification` 和 `LanguageModelConfiguration`。接着，这个类有一个 `prompt_strategies` 字段，它是 `PromptStrategiesConfiguration` 类的实例。

最后，这个类定义了一个 `PlannerSettings` 类，它是 `SystemSettings` 类的子类。这个类有一个 `configuration` 字段，它是 `PlannerConfiguration` 类的实例。


```py
class PromptStrategiesConfiguration(SystemConfiguration):
    name_and_goals: prompt_strategies.NameAndGoalsConfiguration
    initial_plan: prompt_strategies.InitialPlanConfiguration
    next_ability: prompt_strategies.NextAbilityConfiguration


class PlannerConfiguration(SystemConfiguration):
    """Configuration for the Planner subsystem."""

    models: dict[LanguageModelClassification, LanguageModelConfiguration]
    prompt_strategies: PromptStrategiesConfiguration


class PlannerSettings(SystemSettings):
    """Settings for the Planner subsystem."""

    configuration: PlannerConfiguration


```

This is a class that defines an AI agent that can be used to complete tasks by interacting with natural language interfaces. The agent has a role, goals, and abilities that it uses to interact with the agent. The `determine_next_ability` method is used to determine the next ability to use based on the task and the agent's abilities. The `chat_with_model` method is used to interact with the agent by sending a prompt and receiving a response.


```py
class SimplePlanner(Configurable):
    """Manages the agent's planning and goal-setting by constructing language model prompts."""

    default_settings = PlannerSettings(
        name="planner",
        description="Manages the agent's planning and goal-setting by constructing language model prompts.",
        configuration=PlannerConfiguration(
            models={
                LanguageModelClassification.FAST_MODEL: LanguageModelConfiguration(
                    model_name=OpenAIModelName.GPT3,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
                LanguageModelClassification.SMART_MODEL: LanguageModelConfiguration(
                    model_name=OpenAIModelName.GPT4,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
            },
            prompt_strategies=PromptStrategiesConfiguration(
                name_and_goals=prompt_strategies.NameAndGoals.default_configuration,
                initial_plan=prompt_strategies.InitialPlan.default_configuration,
                next_ability=prompt_strategies.NextAbility.default_configuration,
            ),
        ),
    )

    def __init__(
        self,
        settings: PlannerSettings,
        logger: logging.Logger,
        model_providers: dict[ModelProviderName, ChatModelProvider],
        workspace: Workspace = None,  # Workspace is not available during bootstrapping.
    ) -> None:
        self._configuration = settings.configuration
        self._logger = logger
        self._workspace = workspace

        self._providers: dict[LanguageModelClassification, ChatModelProvider] = {}
        for model, model_config in self._configuration.models.items():
            self._providers[model] = model_providers[model_config.provider_name]

        self._prompt_strategies = {
            "name_and_goals": prompt_strategies.NameAndGoals(
                **self._configuration.prompt_strategies.name_and_goals.dict()
            ),
            "initial_plan": prompt_strategies.InitialPlan(
                **self._configuration.prompt_strategies.initial_plan.dict()
            ),
            "next_ability": prompt_strategies.NextAbility(
                **self._configuration.prompt_strategies.next_ability.dict()
            ),
        }

    async def decide_name_and_goals(self, user_objective: str) -> ChatModelResponse:
        return await self.chat_with_model(
            self._prompt_strategies["name_and_goals"],
            user_objective=user_objective,
        )

    async def make_initial_plan(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        abilities: list[str],
    ) -> ChatModelResponse:
        return await self.chat_with_model(
            self._prompt_strategies["initial_plan"],
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            abilities=abilities,
        )

    async def determine_next_ability(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
    ):
        return await self.chat_with_model(
            self._prompt_strategies["next_ability"],
            task=task,
            ability_specs=ability_specs,
        )

    async def chat_with_model(
        self,
        prompt_strategy: PromptStrategy,
        **kwargs,
    ) -> ChatModelResponse:
        model_classification = prompt_strategy.model_classification
        model_configuration = self._configuration.models[model_classification].dict()
        self._logger.debug(f"Using model configuration: {model_configuration}")
        del model_configuration["provider_name"]
        provider = self._providers[model_classification]

        template_kwargs = self._make_template_kwargs_for_strategy(prompt_strategy)
        template_kwargs.update(kwargs)
        prompt = prompt_strategy.build_prompt(**template_kwargs)

        self._logger.debug(f"Using prompt:\n{dump_prompt(prompt)}\n")
        response = await provider.create_chat_completion(
            model_prompt=prompt.messages,
            functions=prompt.functions,
            **model_configuration,
            completion_parser=prompt_strategy.parse_response_content,
        )
        return response

    def _make_template_kwargs_for_strategy(self, strategy: PromptStrategy):
        provider = self._providers[strategy.model_classification]
        template_kwargs = {
            "os_info": get_os_info(),
            "api_budget": provider.get_remaining_budget(),
            "current_time": time.strftime("%c"),
        }
        return template_kwargs


```

这段代码定义了一个名为 `get_os_info()` 的函数，它返回操作系统名称(即操作系统)。

函数首先使用 `platform.system()` 获取操作系统名称，如果操作系统名称不是 "Linux"，则会使用 `distro.name(pretty=True)` 获取操作系统名称。函数使用字符串格式化语法将操作系统名称打印出来。

函数的实现依赖于操作系统，即运行这个代码的系统。如果运行这个代码的系统是 Linux，那么函数将打印出 "Linux"。如果运行这个代码的系统不是 Linux，则需要使用 `distro.name(pretty=True)` 获取操作系统名称。


```py
def get_os_info() -> str:
    os_name = platform.system()
    os_info = (
        platform.platform(terse=True)
        if os_name != "Linux"
        else distro.name(pretty=True)
    )
    return os_info

```