# `.\AutoGPT\autogpts\autogpt\autogpt\agent_factory\profile_generator.py`

```py
# 导入所需的模块
import json
import logging

# 从autogpt.config模块中导入AIDirectives, AIProfile, Config类
from autogpt.config import AIDirectives, AIProfile, Config
# 从autogpt.core.configuration模块中导入SystemConfiguration, UserConfigurable类
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
# 从autogpt.core.prompting模块中导入ChatPrompt, LanguageModelClassification, PromptStrategy类
from autogpt.core.prompting import ChatPrompt, LanguageModelClassification, PromptStrategy
# 从autogpt.core.prompting.utils模块中导入json_loads函数
from autogpt.core.prompting.utils import json_loads
# 从autogpt.core.resource.model_providers.schema模块中导入AssistantChatMessage, ChatMessage, ChatModelProvider, CompletionModelFunction类
from autogpt.core.resource.model_providers.schema import AssistantChatMessage, ChatMessage, ChatModelProvider, CompletionModelFunction
# 从autogpt.core.utils.json_schema模块中导入JSONSchema类
from autogpt.core.utils.json_schema import JSONSchema

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义AgentProfileGeneratorConfiguration类，继承自SystemConfiguration类
class AgentProfileGeneratorConfiguration(SystemConfiguration):
    # 定义model_classification属性，类型为LanguageModelClassification，可由用户配置，默认值为SMART_MODEL
    model_classification: LanguageModelClassification = UserConfigurable(default=LanguageModelClassification.SMART_MODEL)
    # 定义system_prompt属性，类型为str，可由用户配置，默认值为一段提示文本
    system_prompt: str = UserConfigurable(
        default=(
            "Your job is to respond to a user-defined task, given in triple quotes, by "
            "invoking the `create_agent` function to generate an autonomous agent to "
            "complete the task. "
            "You should supply a role-based name for the agent (_GPT), "
            "an informative description for what the agent does, and 1 to 5 directives "
            "in each of the categories Best Practices and Constraints, "
            "that are optimally aligned with the successful completion "
            "of its assigned task.\n"
            "\n"
            "Example Input:\n"
            '"""Help me with marketing my business"""\n\n'
            "Example Call:\n"
            "```\n"
            f"{json.dumps(_example_call, indent=4)}"
            "\n```"
        )
    )
    # 定义user_prompt_template属性，类型为str，可由用户配置，默认值为一段提示文本模板
    user_prompt_template: str = UserConfigurable(default='"""{user_objective}"""')

# 定义AgentProfileGenerator类，继承自PromptStrategy类
class AgentProfileGenerator(PromptStrategy):
    # 定义default_configuration属性，类型为AgentProfileGeneratorConfiguration，可由用户配置，默认值为AgentProfileGeneratorConfiguration的实例
    default_configuration: AgentProfileGeneratorConfiguration = AgentProfileGeneratorConfiguration()
    # 初始化方法，接受模型分类、系统提示信息、用户提示模板和创建代理函数作为参数
    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt: str,
        user_prompt_template: str,
        create_agent_function: dict,
    ):
        # 将模型分类、系统提示信息、用户提示模板和创建代理函数保存到对应的实例变量中
        self._model_classification = model_classification
        self._system_prompt_message = system_prompt
        self._user_prompt_template = user_prompt_template
        self._create_agent_function = CompletionModelFunction.parse(
            create_agent_function
        )

    # 返回模型分类
    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    # 构建提示信息，接受用户目标和其他关键字参数
    def build_prompt(self, user_objective: str = "", **kwargs) -> ChatPrompt:
        # 创建系统消息和用户消息
        system_message = ChatMessage.system(self._system_prompt_message)
        user_message = ChatMessage.user(
            self._user_prompt_template.format(
                user_objective=user_objective,
            )
        )
        # 创建提示对象，包含系统消息、用户消息和创建代理函数
        prompt = ChatPrompt(
            messages=[system_message, user_message],
            functions=[self._create_agent_function],
        )
        return prompt

    # 解析响应内容，接受助手聊天消息作为参数
    def parse_response_content(
        self,
        response_content: AssistantChatMessage,
    # 定义一个函数，用于解析来自目标模型的文本响应
    def parse_response(response_content: object) -> tuple[AIProfile, AIDirectives]:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        # 尝试解析响应内容
        try:
            # 如果没有工具调用，则引发值错误
            if not response_content.tool_calls:
                raise ValueError(
                    f"LLM did not call {self._create_agent_function.name} function; "
                    "agent profile creation failed"
                )
            # 从工具调用中获取参数并解析为对象
            arguments: object = json_loads(
                response_content.tool_calls[0].function.arguments
            )
            # 创建 AIProfile 对象
            ai_profile = AIProfile(
                ai_name=arguments.get("name"),
                ai_role=arguments.get("description"),
            )
            # 创建 AIDirectives 对象
            ai_directives = AIDirectives(
                best_practices=arguments.get("directives", {}).get("best_practices"),
                constraints=arguments.get("directives", {}).get("constraints"),
                resources=[],
            )
        # 捕获 KeyError 异常
        except KeyError:
            # 记录调试信息并重新引发异常
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        # 返回解析后的 AIProfile 和 AIDirectives 对象
        return ai_profile, ai_directives
# 异步函数，生成给定任务的代理配置文件
async def generate_agent_profile_for_task(
    task: str,  # 任务字符串
    app_config: Config,  # 应用配置对象
    llm_provider: ChatModelProvider,  # 聊天模型提供者对象
) -> tuple[AIProfile, AIDirectives]:  # 返回一个元组，包含 AIProfile 和 AIDirectives 对象

    # 创建 AgentProfileGenerator 对象，使用默认配置
    agent_profile_generator = AgentProfileGenerator(
        **AgentProfileGenerator.default_configuration.dict()  # HACK
    )

    # 生成用于构建 AI 配置的提示
    prompt = agent_profile_generator.build_prompt(task)

    # 调用 LLM，使用提示消息和函数列表作为参数
    output = (
        await llm_provider.create_chat_completion(
            prompt.messages,
            model_name=app_config.smart_llm,
            functions=prompt.functions,
        )
    ).response

    # 调试输出 LLM 的原始输出
    logger.debug(f"AI Config Generator Raw Output: {output}")

    # 解析输出，获取 AIProfile 和 AIDirectives 对象
    ai_profile, ai_directives = agent_profile_generator.parse_response_content(output)

    return ai_profile, ai_directives
```