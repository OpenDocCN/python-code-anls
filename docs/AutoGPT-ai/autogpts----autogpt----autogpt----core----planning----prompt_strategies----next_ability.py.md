# `.\AutoGPT\autogpts\autogpt\autogpt\core\planning\prompt_strategies\next_ability.py`

```py
# 导入 logging 模块
import logging

# 导入自定义模块
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.planning.schema import Task
from autogpt.core.prompting import PromptStrategy
from autogpt.core.prompting.schema import ChatPrompt, LanguageModelClassification
from autogpt.core.prompting.utils import json_loads, to_numbered_list
from autogpt.core.resource.model_providers import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema

# 获取当前模块的 logger
logger = logging.getLogger(__name__)


# 定义 NextAbilityConfiguration 类，继承自 SystemConfiguration
class NextAbilityConfiguration(SystemConfiguration):
    # 定义 model_classification 属性，类型为 LanguageModelClassification，可由用户配置
    model_classification: LanguageModelClassification = UserConfigurable()
    # 定义 system_prompt_template 属性，类型为 str，可由用户配置
    system_prompt_template: str = UserConfigurable()
    # 定义 system_info 属性，类型为列表，元素为 str，可由用户配置
    system_info: list[str] = UserConfigurable()
    # 定义 user_prompt_template 属性，类型为 str，可由用户配置
    user_prompt_template: str = UserConfigurable()
    # 定义 additional_ability_arguments 属性，类型为字典，可由用户配置
    additional_ability_arguments: dict = UserConfigurable()


# 定义 NextAbility 类，继承自 PromptStrategy
class NextAbility(PromptStrategy):
    # 定义 DEFAULT_SYSTEM_PROMPT_TEMPLATE 常量，默认值为指定字符串
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = "System Info:\n{system_info}"

    # 定义 DEFAULT_SYSTEM_INFO 常量，默认值为包含系统信息的列表
    DEFAULT_SYSTEM_INFO = [
        "The OS you are running on is: {os_info}",
        "It takes money to let you run. Your API budget is ${api_budget:.3f}",
        "The current time and date is {current_time}",
    ]
    # 默认用户提示模板，包含任务目标、已完成动作次数、动作历史、额外信息、用户输入、验收标准等信息
    DEFAULT_USER_PROMPT_TEMPLATE = (
        "Your current task is is {task_objective}.\n"
        "You have taken {cycle_count} actions on this task already. "
        "Here is the actions you have taken and their results:\n"
        "{action_history}\n\n"
        "Here is additional information that may be useful to you:\n"
        "{additional_info}\n\n"
        "Additionally, you should consider the following:\n"
        "{user_input}\n\n"
        "Your task of {task_objective} is complete when the following acceptance"
        " criteria have been met:\n"
        "{acceptance_criteria}\n\n"
        "Please choose one of the provided functions to accomplish this task. "
        "Some tasks may require multiple functions to accomplish. If that is the case,"
        " choose the function that you think is most appropriate for the current"
        " situation given your progress so far."
    )
    
    # 默认附加能力参数，包含动机、自我批评、推理等信息
    DEFAULT_ADDITIONAL_ABILITY_ARGUMENTS = {
        "motivation": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "Your justification for choosing choosing this function instead of a "
                "different one."
            ),
        ),
        "self_criticism": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "Thoughtful self-criticism that explains why this function may not be "
                "the best choice."
            ),
        ),
        "reasoning": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "Your reasoning for choosing this function taking into account the "
                "`motivation` and weighing the `self_criticism`."
            ),
        ),
    }
    # 默认配置信息，包括模型分类、系统提示模板、系统信息、用户提示模板和额外能力参数
    default_configuration: NextAbilityConfiguration = NextAbilityConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        system_info=DEFAULT_SYSTEM_INFO,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        additional_ability_arguments={
            # 将默认的额外能力参数转换为字典形式
            k: v.to_dict() for k, v in DEFAULT_ADDITIONAL_ABILITY_ARGUMENTS.items()
        },
    )

    # 初始化方法，接受模型分类、系统提示模板、系统信息、用户提示模板和额外能力参数
    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt_template: str,
        system_info: list[str],
        user_prompt_template: str,
        additional_ability_arguments: dict,
    ):
        # 设置模型分类、系统提示模板、系统信息、用户提示模板和额外能力参数
        self._model_classification = model_classification
        self._system_prompt_template = system_prompt_template
        self._system_info = system_info
        self._user_prompt_template = user_prompt_template
        # 解析额外能力参数为 JSON 格式
        self._additional_ability_arguments = JSONSchema.parse_properties(
            additional_ability_arguments
        )
        # 将额外能力参数中的每个属性设置为必需属性
        for p in self._additional_ability_arguments.values():
            p.required = True

    # 返回模型分类
    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    # 构建提示信息，接受任务、能力规格、操作系统信息、API 预算、当前时间和其他参数
    def build_prompt(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
        os_info: str,
        api_budget: float,
        current_time: str,
        **kwargs,
    def parse_response_content(
        self,
        response_content: AssistantChatMessage,
    ) -> dict:
        """解析来自客观模型的实际文本响应。

        Args:
            response_content: 客观模型的原始响应内容。

        Returns:
            解析后的响应。

        """
        try:
            # 检查是否有工具调用
            if not response_content.tool_calls:
                raise ValueError("LLM did not call any function")

            # 获取第一个工具调用的函数名和参数
            function_name = response_content.tool_calls[0].function.name
            function_arguments = json_loads(
                response_content.tool_calls[0].function.arguments
            )
            # 解析响应内容，提取关键信息
            parsed_response = {
                "motivation": function_arguments.pop("motivation"),
                "self_criticism": function_arguments.pop("self_criticism"),
                "reasoning": function_arguments.pop("reasoning"),
                "next_ability": function_name,
                "ability_arguments": function_arguments,
            }
        except KeyError:
            # 如果解析失败，记录错误信息并抛出异常
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        # 返回解析后的响应
        return parsed_response
```