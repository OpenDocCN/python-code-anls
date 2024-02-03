# `.\AutoGPT\autogpts\autogpt\autogpt\core\planning\simple.py`

```py
# 导入 logging 模块，用于记录日志信息
import logging
# 导入 platform 模块，用于获取系统平台信息
import platform
# 导入 time 模块，用于时间相关操作

# 导入 distro 模块，用于获取 Linux 发行版信息
import distro

# 从 autogpt.core.configuration 模块中导入 Configurable、SystemConfiguration、SystemSettings、UserConfigurable 类
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
# 从 autogpt.core.planning 模块中导入 prompt_strategies 模块
from autogpt.core.planning import prompt_strategies
# 从 autogpt.core.planning.schema 模块中导入 Task 类
from autogpt.core.planning.schema import Task
# 从 autogpt.core.prompting 模块中导入 PromptStrategy 类
from autogpt.core.prompting import PromptStrategy
# 从 autogpt.core.prompting.schema 模块中导入 LanguageModelClassification 类
from autogpt.core.prompting.schema import LanguageModelClassification
# 从 autogpt.core.resource.model_providers 模块中导入 ChatModelProvider、ChatModelResponse、CompletionModelFunction、ModelProviderName、OpenAIModelName 类
from autogpt.core.resource.model_providers import (
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderName,
    OpenAIModelName,
)
# 从 autogpt.core.runner.client_lib.logging.helpers 模块中导入 dump_prompt 函数
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
# 从 autogpt.core.workspace 模块中导入 Workspace 类

# 定义 LanguageModelConfiguration 类，继承自 SystemConfiguration 类
class LanguageModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""
    # 定义 model_name 属性，可由用户配置
    model_name: str = UserConfigurable()
    # 定义 provider_name 属性，可由用户配置
    provider_name: ModelProviderName = UserConfigurable()
    # 定义 temperature 属性，可由用户配置

# 定义 PromptStrategiesConfiguration 类，继承自 SystemConfiguration 类
class PromptStrategiesConfiguration(SystemConfiguration):
    # 定义 name_and_goals 属性，类型为 prompt_strategies.NameAndGoalsConfiguration
    name_and_goals: prompt_strategies.NameAndGoalsConfiguration
    # 定义 initial_plan 属性，类型为 prompt_strategies.InitialPlanConfiguration
    initial_plan: prompt_strategies.InitialPlanConfiguration
    # 定义 next_ability 属性，类型为 prompt_strategies.NextAbilityConfiguration

# 定义 PlannerConfiguration 类，继承自 SystemConfiguration 类
class PlannerConfiguration(SystemConfiguration):
    """Configuration for the Planner subsystem."""
    # 定义 models 属性，类型为字典，键为 LanguageModelClassification，值为 LanguageModelConfiguration
    models: dict[LanguageModelClassification, LanguageModelConfiguration]
    # 定义 prompt_strategies 属性，类型为 PromptStrategiesConfiguration

# 定义 PlannerSettings 类，继承自 SystemSettings 类
class PlannerSettings(SystemSettings):
    """Settings for the Planner subsystem."""
    # 定义 configuration 属性，类型为 PlannerConfiguration

# 定义 SimplePlanner 类，继承自 Configurable 类
class SimplePlanner(Configurable):
    """
    Manages the agent's planning and goal-setting
    by constructing language model prompts.
    """
    # 默认的规划器设置，包括名称、描述、配置信息
    default_settings = PlannerSettings(
        name="planner",
        description=(
            "Manages the agent's planning and goal-setting "
            "by constructing language model prompts."
        ),
        configuration=PlannerConfiguration(
            models={
                # 快速模型配置，使用 GPT3 模型，提供者为 OpenAI，温度为 0.9
                LanguageModelClassification.FAST_MODEL: LanguageModelConfiguration(
                    model_name=OpenAIModelName.GPT3,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
                # 智能模型配置，使用 GPT4 模型，提供者为 OpenAI，温度为 0.9
                LanguageModelClassification.SMART_MODEL: LanguageModelConfiguration(
                    model_name=OpenAIModelName.GPT4,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
            },
            prompt_strategies=PromptStrategiesConfiguration(
                # 名称和目标的提示策略配置
                name_and_goals=prompt_strategies.NameAndGoals.default_configuration,
                # 初始计划的提示策略配置
                initial_plan=prompt_strategies.InitialPlan.default_configuration,
                # 下一个能力的提示策略配置
                next_ability=prompt_strategies.NextAbility.default_configuration,
            ),
        ),
    )

    # 初始化方法，接受规划器设置、日志记录器、模型提供者字典和工作空间作为参数
    def __init__(
        self,
        settings: PlannerSettings,
        logger: logging.Logger,
        model_providers: dict[ModelProviderName, ChatModelProvider],
        workspace: Workspace = None,  # 在引导过程中不可用的工作空间
    # 初始化 ChatModelManager 类，设置配置、日志和工作空间
    def __init__(self, settings: Settings, logger: Logger, workspace: Workspace) -> None:
        self._configuration = settings.configuration
        self._logger = logger
        self._workspace = workspace

        # 初始化 providers 字典，用于存储不同语言模型分类对应的 ChatModelProvider
        self._providers: dict[LanguageModelClassification, ChatModelProvider] = {}
        # 遍历配置中的模型，将每个模型对应的提供者存储到 providers 字典中
        for model, model_config in self._configuration.models.items():
            self._providers[model] = model_providers[model_config.provider_name]

        # 初始化 prompt_strategies 字典，用于存储不同策略的 PromptStrategy 实例
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

    # 使用 name_and_goals 策略与模型交互，决定名称和目标
    async def decide_name_and_goals(self, user_objective: str) -> ChatModelResponse:
        return await self.chat_with_model(
            self._prompt_strategies["name_and_goals"],
            user_objective=user_objective,
        )

    # 使用 initial_plan 策略与模型交互，制定初始计划
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

    # 使用 next_ability 策略与模型交互，确定下一个能力
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

    # 与模型交互的通用方法，根据不同的 prompt_strategy 进行交互
    async def chat_with_model(
        self,
        prompt_strategy: PromptStrategy,
        **kwargs,
    # 定义一个方法，接收一个PromptStrategy对象作为参数，并返回一个ChatModelResponse对象
    ) -> ChatModelResponse:
        # 获取prompt_strategy的模型分类
        model_classification = prompt_strategy.model_classification
        # 获取模型配置信息
        model_configuration = self._configuration.models[model_classification].dict()
        # 记录使用的模型配置信息
        self._logger.debug(f"Using model configuration: {model_configuration}")
        # 删除模型配置信息中的provider_name字段
        del model_configuration["provider_name"]
        # 获取对应模型分类的provider对象
        provider = self._providers[model_classification]

        # 生成用于构建prompt的模板参数
        template_kwargs = self._make_template_kwargs_for_strategy(prompt_strategy)
        # 更新模板参数
        template_kwargs.update(kwargs)
        # 根据模板参数构建prompt
        prompt = prompt_strategy.build_prompt(**template_kwargs)

        # 记录使用的prompt信息
        self._logger.debug(f"Using prompt:\n{dump_prompt(prompt)}\n")
        # 调用provider的create_chat_completion方法，生成聊天完成的响应
        response = await provider.create_chat_completion(
            model_prompt=prompt.messages,
            functions=prompt.functions,
            **model_configuration,
            completion_parser=prompt_strategy.parse_response_content,
        )
        # 返回响应结果
        return response

    # 定义一个方法，接收一个PromptStrategy对象作为参数，并返回一个模板参数字典
    def _make_template_kwargs_for_strategy(self, strategy: PromptStrategy):
        # 获取对应模型分类的provider对象
        provider = self._providers[strategy.model_classification]
        # 构建模板参数字典
        template_kwargs = {
            "os_info": get_os_info(),
            "api_budget": provider.get_remaining_budget(),
            "current_time": time.strftime("%c"),
        }
        # 返回模板参数字典
        return template_kwargs
# 获取操作系统信息并返回字符串形式
def get_os_info() -> str:
    # 获取操作系统名称
    os_name = platform.system()
    # 如果操作系统不是 Linux，则获取操作系统的平台信息
    os_info = (
        platform.platform(terse=True)
        if os_name != "Linux"
        # 如果操作系统是 Linux，则获取 Linux 发行版的名称
        else distro.name(pretty=True)
    )
    # 返回操作系统信息
    return os_info
```