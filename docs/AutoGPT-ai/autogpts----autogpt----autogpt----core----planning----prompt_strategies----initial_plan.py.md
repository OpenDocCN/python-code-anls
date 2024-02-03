# `.\AutoGPT\autogpts\autogpt\autogpt\core\planning\prompt_strategies\initial_plan.py`

```py
# 导入 logging 模块
import logging

# 导入自定义模块
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.planning.schema import Task, TaskType
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

# 定义 InitialPlanConfiguration 类，继承自 SystemConfiguration
class InitialPlanConfiguration(SystemConfiguration):
    # 定义 model_classification 属性，类型为 LanguageModelClassification，可由用户配置
    model_classification: LanguageModelClassification = UserConfigurable()
    # 定义 system_prompt_template 属性，类型为 str，可由用户配置
    system_prompt_template: str = UserConfigurable()
    # 定义 system_info 属性，类型为列表，元素为 str，可由用户配置
    system_info: list[str] = UserConfigurable()
    # 定义 user_prompt_template 属性，类型为 str，可由用户配置
    user_prompt_template: str = UserConfigurable()
    # 定义 create_plan_function 属性，类型为字典，可由用户配置
    create_plan_function: dict = UserConfigurable()

# 定义 InitialPlan 类，继承自 PromptStrategy
class InitialPlan(PromptStrategy):
    # 定义 DEFAULT_SYSTEM_PROMPT_TEMPLATE 常量，包含系统提示模板的默认值
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert project planner. "
        "Your responsibility is to create work plans for autonomous agents. "
        "You will be given a name, a role, set of goals for the agent to accomplish. "
        "Your job is to break down those goals into a set of tasks that the agent can"
        " accomplish to achieve those goals. "
        "Agents are resourceful, but require clear instructions."
        " Each task you create should have clearly defined `ready_criteria` that the"
        " agent can check to see if the task is ready to be started."
        " Each task should also have clearly defined `acceptance_criteria` that the"
        " agent can check to evaluate if the task is complete. "
        "You should create as many tasks as you think is necessary to accomplish"
        " the goals.\n\n"
        "System Info:\n{system_info}"
    )
    # 默认系统信息模板，包含系统信息的占位符
    DEFAULT_SYSTEM_INFO = [
        "The OS you are running on is: {os_info}",
        "It takes money to let you run. Your API budget is ${api_budget:.3f}",
        "The current time and date is {current_time}",
    ]

    # 默认用户提示模板，包含用户信息和目标的占位符
    DEFAULT_USER_PROMPT_TEMPLATE = (
        "You are {agent_name}, {agent_role}\n" "Your goals are:\n" "{agent_goals}"
    )

    # 默认配置信息，包含模型分类、系统提示模板、系统信息、用户提示模板和创建计划函数
    default_configuration: InitialPlanConfiguration = InitialPlanConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        system_info=DEFAULT_SYSTEM_INFO,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        create_plan_function=DEFAULT_CREATE_PLAN_FUNCTION.schema,
    )

    # 初始化函数，接受模型分类、系统提示模板、系统信息、用户提示模板和创建计划函数作为参数
    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt_template: str,
        system_info: list[str],
        user_prompt_template: str,
        create_plan_function: dict,
    ):
        # 初始化对象的属性
        self._model_classification = model_classification
        self._system_prompt_template = system_prompt_template
        self._system_info = system_info
        self._user_prompt_template = user_prompt_template
        self._create_plan_function = CompletionModelFunction.parse(create_plan_function)

    # 返回模型分类属性
    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    # 构建提示信息，接受代理名称、代理角色、代理目标、能力、操作系统信息、API预算、当前时间等参数
    def build_prompt(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        abilities: list[str],
        os_info: str,
        api_budget: float,
        current_time: str,
        **kwargs,
    # 定义一个方法，返回一个ChatPrompt对象
    ) -> ChatPrompt:
        # 创建模板参数字典，包括agent_name, agent_role, os_info, api_budget, current_time和kwargs
        template_kwargs = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "os_info": os_info,
            "api_budget": api_budget,
            "current_time": current_time,
            **kwargs,
        }
        # 将agent_goals转换为编号列表，并添加到模板参数字典中
        template_kwargs["agent_goals"] = to_numbered_list(
            agent_goals, **template_kwargs
        )
        # 将abilities转换为编号列表，并添加到模板参数字典中
        template_kwargs["abilities"] = to_numbered_list(abilities, **template_kwargs)
        # 将self._system_info转换为编号列表，并添加到模板参数字典中
        template_kwargs["system_info"] = to_numbered_list(
            self._system_info, **template_kwargs
        )

        # 创建系统提示消息对象，使用系统提示模板和模板参数字典
        system_prompt = ChatMessage.system(
            self._system_prompt_template.format(**template_kwargs),
        )
        # 创建用户提示消息对象，使用用户提示模板和模板参数字典
        user_prompt = ChatMessage.user(
            self._user_prompt_template.format(**template_kwargs),
        )

        # 返回一个ChatPrompt对象，包括系统提示消息、用户提示消息和创建计划函数
        return ChatPrompt(
            messages=[system_prompt, user_prompt],
            functions=[self._create_plan_function],
            # TODO: 待完成，tokens_used暂时设为0
            tokens_used=0,
        )

    # 定义一个方法，解析响应内容
    def parse_response_content(
        self,
        response_content: AssistantChatMessage,
    ) -> dict:
        """解析来自目标模型的实际文本响应。

        Args:
            response_content: 目标模型的原始响应内容。

        Returns:
            解析后的响应。
        """
        try:
            # 检查是否没有工具调用
            if not response_content.tool_calls:
                # 如果没有工具调用，则引发值错误
                raise ValueError(
                    f"LLM did not call {self._create_plan_function.name} function; "
                    "plan creation failed"
                )
            # 解析第一个工具调用的参数
            parsed_response: object = json_loads(
                response_content.tool_calls[0].function.arguments
            )
            # 解析任务列表中的每个任务
            parsed_response["task_list"] = [
                Task.parse_obj(task) for task in parsed_response["task_list"]
            ]
        except KeyError:
            # 如果出现 KeyError，则记录错误信息并重新引发异常
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        # 返回解析后的响应
        return parsed_response
```