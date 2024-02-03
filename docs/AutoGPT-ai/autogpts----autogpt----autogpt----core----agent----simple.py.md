# `.\AutoGPT\autogpts\autogpt\autogpt\core\agent\simple.py`

```py
# 导入日志模块
import logging
# 导入日期时间模块
from datetime import datetime
# 导入路径模块
from pathlib import Path
# 导入类型提示模块
from typing import Any

# 导入 Pydantic 模块中的基础模型类
from pydantic import BaseModel

# 导入自动生成文本模块中的能力相关类和函数
from autogpt.core.ability import (
    AbilityRegistrySettings,
    AbilityResult,
    SimpleAbilityRegistry,
)
# 导入自动生成文本模块中的基础代理类
from autogpt.core.agent.base import Agent
# 导入自动生成文本模块中的配置相关类
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
# 导入自动生成文本模块中的记忆相关类和函数
from autogpt.core.memory import MemorySettings, SimpleMemory
# 导入自动生成文本模块中的规划相关类和函数
from autogpt.core.planning import PlannerSettings, SimplePlanner, Task, TaskStatus
# 导入自动生成文本模块中的简单插件服务类
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
# 导入自动生成文本模块中的模型提供者相关类和函数
from autogpt.core.resource.model_providers import (
    CompletionModelFunction,
    OpenAIProvider,
    OpenAISettings,
)
# 导入自动生成文本模块中的简单工作空间类
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings

# 定义代理系统配置类，继承自系统配置类
class AgentSystems(SystemConfiguration):
    # 定义能力注册表的位置
    ability_registry: PluginLocation
    # 定义记忆的位置
    memory: PluginLocation
    # 定义 OpenAI 提供者的位置
    openai_provider: PluginLocation
    # 定义规划器的位置
    planning: PluginLocation
    # 定义工作空间的位置
    workspace: PluginLocation

# 定义代理配置类，继承自系统配置类
class AgentConfiguration(SystemConfiguration):
    # 定义循环次数
    cycle_count: int
    # 定义最大任务循环次数
    max_task_cycle_count: int
    # 定义创建时间
    creation_time: str
    # 定义名称
    name: str
    # 定义角色
    role: str
    # 定义目标列表
    goals: list[str]
    # 定义系统配置
    systems: AgentSystems

# 定义代理系统设置类，继承自系统设置类
class AgentSystemSettings(SystemSettings):
    # 定义配置
    configuration: AgentConfiguration

# 定义代理设置类，继承自基础模型类
class AgentSettings(BaseModel):
    # 定义代理系统设置
    agent: AgentSystemSettings
    # 定义能力注册表设置
    ability_registry: AbilityRegistrySettings
    # 定义记忆设置
    memory: MemorySettings
    # 定义 OpenAI 提供者设置
    openai_provider: OpenAISettings
    # 定义规划器设置
    planning: PlannerSettings
    # 定义工作空间设置
    workspace: WorkspaceSettings

    # 定义更新代理名称和目标的方法
    def update_agent_name_and_goals(self, agent_goals: dict) -> None:
        # 更新代理名称
        self.agent.configuration.name = agent_goals["agent_name"]
        # 更新代理角色
        self.agent.configuration.role = agent_goals["agent_role"]
        # 更新代理目标列表
        self.agent.configuration.goals = agent_goals["agent_goals"]

# 定义简单代理类，继承自代理类和可配置类
class SimpleAgent(Agent, Configurable):
    # 创建一个默认的AgentSystemSettings对象，包含了代理系统的各种设置信息
    default_settings = AgentSystemSettings(
        # 设置代理系统的名称为"simple_agent"
        name="simple_agent",
        # 设置代理系统的描述为"A simple agent."
        description="A simple agent.",
        # 配置代理系统的AgentConfiguration对象
        configuration=AgentConfiguration(
            # 设置AgentConfiguration对象的名称为"Entrepreneur-GPT"
            name="Entrepreneur-GPT",
            # 设置AgentConfiguration对象的角色描述
            role=(
                "An AI designed to autonomously develop and run businesses with "
                "the sole goal of increasing your net worth."
            ),
            # 设置AgentConfiguration对象的目标列表
            goals=[
                "Increase net worth",
                "Grow Twitter Account",
                "Develop and manage multiple businesses autonomously",
            ],
            # 设置AgentConfiguration对象的循环计数为0
            cycle_count=0,
            # 设置AgentConfiguration对象的最大任务循环计数为3
            max_task_cycle_count=3,
            # 设置AgentConfiguration对象的创建时间为空字符串
            creation_time="",
            # 配置AgentConfiguration对象的AgentSystems对象
            systems=AgentSystems(
                # 配置AgentSystems对象的ability_registry属性
                ability_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.ability.SimpleAbilityRegistry",
                ),
                # 配置AgentSystems对象的memory属性
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.SimpleMemory",
                ),
                # 配置AgentSystems对象的openai_provider属性
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route=(
                        "autogpt.core.resource.model_providers.OpenAIProvider"
                    ),
                ),
                # 配置AgentSystems对象的planning属性
                planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.planning.SimplePlanner",
                ),
                # 配置AgentSystems对象的workspace属性
                workspace=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.workspace.SimpleWorkspace",
                ),
            ),
        ),
    )
    # 初始化 AgentSystem 类的实例
    def __init__(
        self,
        settings: AgentSystemSettings,  # 接收代理系统设置对象
        logger: logging.Logger,  # 接收日志记录器对象
        ability_registry: SimpleAbilityRegistry,  # 接收简单能力注册表对象
        memory: SimpleMemory,  # 接收简单内存对象
        openai_provider: OpenAIProvider,  # 接收 OpenAI 提供者对象
        planning: SimplePlanner,  # 接收简单规划器对象
        workspace: SimpleWorkspace,  # 接收简单工作空间对象
    ):
        # 设置代理系统的配置
        self._configuration = settings.configuration
        # 设置代理系统的日志记录器
        self._logger = logger
        # 设置代理系统的能力注册表
        self._ability_registry = ability_registry
        # 设置代理系统的内存
        self._memory = memory
        # FIXME: 需要一些工作来将其作为提供者字典工作
        #  使配置的构建工作有点棘手
        # 设置代理系统的 OpenAI 提供者
        self._openai_provider = openai_provider
        # 设置代理系统的规划器
        self._planning = planning
        # 设置代理系统的工作空间
        self._workspace = workspace
        # 初始化任务队列为空列表
        self._task_queue = []
        # 初始化已完成任务列表为空列表
        self._completed_tasks = []
        # 初始化当前任务为 None
        self._current_task = None
        # 初始化下一个能力为 None
        self._next_ability = None

    @classmethod
    # 从工作空间创建 AgentSystem 类的实例
    def from_workspace(
        cls,
        workspace_path: Path,  # 接收工作空间路径
        logger: logging.Logger,  # 接收日志记录器对象
    # 定义一个类方法，返回一个SimpleAgent对象
    ) -> "SimpleAgent":
        # 从workspace_path加载agent设置
        agent_settings = SimpleWorkspace.load_agent_settings(workspace_path)
        # 初始化agent参数字典
        agent_args = {}

        # 将agent设置添加到agent参数字典中
        agent_args["settings"] = agent_settings.agent
        # 将logger添加到agent参数字典中
        agent_args["logger"] = logger
        # 获取workspace实例并添加到agent参数字典中
        agent_args["workspace"] = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger,
        )
        # 获取openai_provider实例并添加到agent参数字典中
        agent_args["openai_provider"] = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger,
        )
        # 获取planning实例并添加到agent参数字典中
        agent_args["planning"] = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["openai_provider"]},
        )
        # 获取memory实例并添加到agent参数字典中
        agent_args["memory"] = cls._get_system_instance(
            "memory",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
        )

        # 获取ability_registry实例并添加到agent参数字典中
        agent_args["ability_registry"] = cls._get_system_instance(
            "ability_registry",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
            memory=agent_args["memory"],
            model_providers={"openai": agent_args["openai_provider"]},
        )

        # 使用agent参数字典创建SimpleAgent对象并返回
        return cls(**agent_args)
    # 异步方法，构建初始计划并返回结果字典
    async def build_initial_plan(self) -> dict:
        # 调用规划器的方法生成初始计划
        plan = await self._planning.make_initial_plan(
            agent_name=self._configuration.name,
            agent_role=self._configuration.role,
            agent_goals=self._configuration.goals,
            abilities=self._ability_registry.list_abilities(),
        )
        # 解析计划中的任务列表
        tasks = [Task.parse_obj(task) for task in plan.parsed_result["task_list"]]

        # TODO: 应该执行一个步骤来评估生成的任务的质量，并确保它们具有可操作的准备和验收标准

        # 将任务添加到任务队列中
        self._task_queue.extend(tasks)
        # 根据任务优先级对任务队列进行排序
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        # 将最后一个任务标记为准备状态
        self._task_queue[-1].context.status = TaskStatus.READY
        # 返回解析后的计划结果
        return plan.parsed_result

    # 异步方法，确定下一个能力
    async def determine_next_ability(self, *args, **kwargs):
        # 如果任务队列为空，返回消息
        if not self._task_queue:
            return {"response": "I don't have any tasks to work on right now."}

        # 增加循环计数
        self._configuration.cycle_count += 1
        # 从任务队列中取出一个任务
        task = self._task_queue.pop()
        # 记录日志，表示正在处理的任务
        self._logger.info(f"Working on task: {task}")

        # 评估任务并添加上下文
        task = await self._evaluate_task_and_add_context(task)
        # 选择下一个能力
        next_ability = await self._choose_next_ability(
            task,
            self._ability_registry.dump_abilities(),
        )
        # 设置当前任务和下一个能力
        self._current_task = task
        self._next_ability = next_ability.parsed_result
        # 返回当前任务和下一个能力
        return self._current_task, self._next_ability
    # 异步执行下一个能力，根据用户输入判断是否执行
    async def execute_next_ability(self, user_input: str, *args, **kwargs):
        # 如果用户输入为"y"
        if user_input == "y":
            # 获取下一个能力对象
            ability = self._ability_registry.get_ability(
                self._next_ability["next_ability"]
            )
            # 执行下一个能力，并获取响应
            ability_response = await ability(**self._next_ability["ability_arguments"])
            # 更新任务和记忆
            await self._update_tasks_and_memory(ability_response)
            # 如果当前任务状态为完成，则将其添加到已完成任务列表中
            if self._current_task.context.status == TaskStatus.DONE:
                self._completed_tasks.append(self._current_task)
            else:
                # 否则将当前任务添加到任务队列中
                self._task_queue.append(self._current_task)
            # 重置当前任务和下一个能力
            self._current_task = None
            self._next_ability = None

            return ability_response.dict()
        else:
            # 如果用户输入不为"y"，则抛出未实现错误
            raise NotImplementedError

    # 异步评估任务并添加上下文
    async def _evaluate_task_and_add_context(self, task: Task) -> Task:
        """Evaluate the task and add context to it."""
        # 如果任务状态为进行中，则直接返回任务
        if task.context.status == TaskStatus.IN_PROGRESS:
            # Nothing to do here
            return task
        else:
            # 否则记录日志，评估任务并添加相关上下文
            self._logger.debug(f"Evaluating task {task} and adding relevant context.")
            # TODO: 查找相关记忆（需要工作记忆系统）
            # TODO: 评估是否有足够信息开始任务（使用LLM）
            # 设置任务上下文信息为有足够信息并状态为进行中
            task.context.enough_info = True
            task.context.status = TaskStatus.IN_PROGRESS
            return task

    # 选择下一个能力
    async def _choose_next_ability(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
    ):
        """Choose the next ability to use for the task."""
        # 为任务选择下一个要使用的能力
        self._logger.debug(f"Choosing next ability for task {task}.")
        # 如果任务的循环计数大于最大任务循环计数，则抛出NotImplementedError
        if task.context.cycle_count > self._configuration.max_task_cycle_count:
            # 不要触发LLM，只需将下一个动作设置为"breakdown_task"，并附上适当的原因
            raise NotImplementedError
        # 如果任务上下文中没有足够的信息，则抛出NotImplementedError
        elif not task.context.enough_info:
            # 不要询问LLM，只需将下一个动作设置为"breakdown_task"，并附上适当的原因
            raise NotImplementedError
        else:
            # 调用_planning.determine_next_ability确定下一个能力
            next_ability = await self._planning.determine_next_ability(
                task, ability_specs
            )
            return next_ability

    async def _update_tasks_and_memory(self, ability_result: AbilityResult):
        # 增加当前任务的循环计数
        self._current_task.context.cycle_count += 1
        # 将能力结果添加到当前任务的先前动作列表中
        self._current_task.context.prior_actions.append(ability_result)
        # TODO: 总结新知识
        # TODO: 将知识和总结存储在内存中和相关任务中
        # TODO: 评估任务是否完成

    def __repr__(self):
        return "SimpleAgent()"

    ################################################################
    # 代理引导和初始化的工厂接口 #
    ################################################################

    @classmethod
    def build_user_configuration(cls) -> dict[str, Any]:
        """Build the user's configuration."""
        # 构建用户的配置
        configuration_dict = {
            "agent": cls.get_user_config(),
        }

        # 获取系统位置并构建系统配置
        system_locations = configuration_dict["agent"]["configuration"]["systems"]
        for system_name, system_location in system_locations.items():
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.get_user_config()
        # 剪除空字典
        configuration_dict = _prune_empty_dicts(configuration_dict)
        return configuration_dict
    # 编译用户配置和默认配置，生成代理设置
    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        # 记录调试信息
        logger.debug("Processing agent system configuration.")
        # 创建配置字典，包含代理配置和系统配置
        configuration_dict = {
            "agent": cls.build_agent_configuration(
                user_configuration.get("agent", {})
            ).dict(),
        }

        # 获取系统配置信息
        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # 构建默认配置
        for system_name, system_location in system_locations.items():
            # 记录调试信息
            logger.debug(f"Compiling configuration for system {system_name}")
            # 获取系统类
            system_class = SimplePluginService.get_plugin(system_location)
            # 构建系统配置
            configuration_dict[system_name] = system_class.build_agent_configuration(
                user_configuration.get(system_name, {})
            ).dict()

        # 解析配置字典为AgentSettings对象并返回
        return AgentSettings.parse_obj(configuration_dict)

    # 确定代理名称和目标
    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ) -> dict:
        # 记录调试信息
        logger.debug("Loading OpenAI provider.")
        # 获取OpenAIProvider实例
        provider: OpenAIProvider = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger=logger,
        )
        # 记录调试信息
        logger.debug("Loading agent planner.")
        # 获取SimplePlanner实例
        agent_planner: SimplePlanner = cls._get_system_instance(
            "planning",
            agent_settings,
            logger=logger,
            model_providers={"openai": provider},
        )
        # 记录调试信息
        logger.debug("determining agent name and goals.")
        # 调用agent_planner的decide_name_and_goals方法获取模型响应
        model_response = await agent_planner.decide_name_and_goals(
            user_objective,
        )

        # 返回模型响应的解析结果
        return model_response.parsed_result

    # 配置代理
    @classmethod
    def provision_agent(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    # 设置代理配置的创建时间为当前时间的格式化字符串
    agent_settings.agent.configuration.creation_time = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    # 获取系统实例的工作空间
    workspace: SimpleWorkspace = cls._get_system_instance(
        "workspace",
        agent_settings,
        logger=logger,
    )
    # 设置工作空间并返回结果
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
    # 获取代理配置中系统的位置信息
    system_locations = agent_settings.agent.configuration.systems.dict()

    # 获取系统配置信息
    system_settings = getattr(agent_settings, system_name)
    # 获取系统类并实例化系统对象
    system_class = SimplePluginService.get_plugin(system_locations[system_name])
    system_instance = system_class(
        system_settings,
        *args,
        logger=logger.getChild(system_name),
        **kwargs,
    )
    # 返回系统实例
    return system_instance
# 从嵌套字典中修剪分支，如果分支只包含叶子节点为空的字典
def _prune_empty_dicts(d: dict) -> dict:
    # 创建一个空字典用于存储修剪后的结果
    pruned = {}
    # 遍历字典的键值对
    for key, value in d.items():
        # 如果值是字典类型
        if isinstance(value, dict):
            # 递归调用_prune_empty_dicts函数，修剪值中的空字典
            pruned_value = _prune_empty_dicts(value)
            # 如果修剪后的字典不为空，将其添加到结果中
            if pruned_value:
                pruned[key] = pruned_value
        else:
            # 如果值不是字典类型，直接添加到结果中
            pruned[key] = value
    # 返回修剪后的字典
    return pruned
```