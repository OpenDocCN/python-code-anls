# `.\AutoGPT\autogpts\autogpt\autogpt\core\planning\prompt_strategies\name_and_goals.py`

```py
# 导入 logging 模块
import logging

# 导入自定义模块
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.prompting import PromptStrategy
from autogpt.core.prompting.schema import ChatPrompt, LanguageModelClassification
from autogpt.core.prompting.utils import json_loads
from autogpt.core.resource.model_providers import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema

# 获取当前模块的 logger
logger = logging.getLogger(__name__)


# 定义 NameAndGoalsConfiguration 类，继承自 SystemConfiguration 类
class NameAndGoalsConfiguration(SystemConfiguration):
    # 定义 model_classification 属性，类型为 LanguageModelClassification，可由用户配置
    model_classification: LanguageModelClassification = UserConfigurable()
    # 定义 system_prompt 属性，类型为 str，可由用户配置
    system_prompt: str = UserConfigurable()
    # 定义 user_prompt_template 属性，类型为 str，可由用户配置
    user_prompt_template: str = UserConfigurable()
    # 定义 create_agent_function 属性，类型为 dict，可由用户配置
    create_agent_function: dict = UserConfigurable()


# 定义 NameAndGoals 类，继承自 PromptStrategy 类
class NameAndGoals(PromptStrategy):
    # 默认系统提示信息，包括任务描述、创建代理函数调用示例等
    DEFAULT_SYSTEM_PROMPT = (
        "Your job is to respond to a user-defined task, given in triple quotes, by "
        "invoking the `create_agent` function to generate an autonomous agent to "
        "complete the task. "
        "You should supply a role-based name for the agent, "
        "an informative description for what the agent does, and "
        "1 to 5 goals that are optimally aligned with the successful completion of "
        "its assigned task.\n"
        "\n"
        "Example Input:\n"
        '"""Help me with marketing my business"""\n\n'
        "Example Function Call:\n"
        "create_agent(name='CMOGPT', "
        "description='A professional digital marketer AI that assists Solopreneurs in "
        "growing their businesses by providing world-class expertise in solving "
        "marketing problems for SaaS, content products, agencies, and more.', "
        "goals=['Engage in effective problem-solving, prioritization, planning, and "
        "supporting execution to address your marketing needs as your virtual Chief "
        "Marketing Officer.', 'Provide specific, actionable, and concise advice to "
        "help you make informed decisions without the use of platitudes or overly "
        "wordy explanations.', 'Identify and prioritize quick wins and cost-effective "
        "campaigns that maximize results with minimal time and budget investment.', "
        "'Proactively take the lead in guiding you and offering suggestions when faced "
        "with unclear information or uncertainty to ensure your marketing strategy "
        "remains on track.'])"
    )
    
    # 默认用户提示信息模板，用于填充用户的任务描述
    DEFAULT_USER_PROMPT_TEMPLATE = '"""{user_objective}"""'
    # 默认的创建代理函数，用于定义创建一个新的自主AI代理以完成给定任务的函数
    DEFAULT_CREATE_AGENT_FUNCTION = CompletionModelFunction(
        name="create_agent",
        description="Create a new autonomous AI agent to complete a given task.",
        parameters={
            "agent_name": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="A short role-based name for an autonomous agent.",
            ),
            "agent_role": JSONSchema(
                type=JSONSchema.Type.STRING,
                description=(
                    "An informative one sentence description of what the AI agent does"
                ),
            ),
            "agent_goals": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                minItems=1,
                maxItems=5,
                items=JSONSchema(
                    type=JSONSchema.Type.STRING,
                ),
                description=(
                    "One to five highly effective goals that are optimally aligned "
                    "with the completion of a specific task. "
                    "The number and complexity of the goals should correspond to the "
                    "complexity of the agent's primary objective."
                ),
            ),
        },
    )

    # 默认的配置，包含名称和目标配置
    default_configuration: NameAndGoalsConfiguration = NameAndGoalsConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        create_agent_function=DEFAULT_CREATE_AGENT_FUNCTION.schema,
    )

    # 初始化函数，接受模型分类、系统提示、用户提示模板和创建代理函数作为参数
    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt: str,
        user_prompt_template: str,
        create_agent_function: dict,
    # 初始化模型分类
    ):
        self._model_classification = model_classification
        # 初始化系统提示消息
        self._system_prompt_message = system_prompt
        # 初始化用户提示模板
        self._user_prompt_template = user_prompt_template
        # 解析创建代理函数
        self._create_agent_function = CompletionModelFunction.parse(
            create_agent_function
        )

    @property
    # 返回模型分类
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    # 构建提示信息
    def build_prompt(self, user_objective: str = "", **kwargs) -> ChatPrompt:
        # 创建系统消息
        system_message = ChatMessage.system(self._system_prompt_message)
        # 创建用户消息
        user_message = ChatMessage.user(
            self._user_prompt_template.format(
                user_objective=user_objective,
            )
        )
        # 创建提示对象
        prompt = ChatPrompt(
            messages=[system_message, user_message],
            functions=[self._create_agent_function],
            # TODO
            tokens_used=0,
        )
        return prompt

    # 解析响应内容
    def parse_response_content(
        self,
        response_content: AssistantChatMessage,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        try:
            # 检查是否有工具调用
            if not response_content.tool_calls:
                raise ValueError(
                    f"LLM did not call {self._create_agent_function} function; "
                    "agent profile creation failed"
                )
            # 解析响应内容
            parsed_response = json_loads(
                response_content.tool_calls[0].function.arguments
            )
        except KeyError:
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        return parsed_response
```