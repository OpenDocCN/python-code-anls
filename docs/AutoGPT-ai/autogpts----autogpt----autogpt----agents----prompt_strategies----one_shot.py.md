# `.\AutoGPT\autogpts\autogpt\autogpt\agents\prompt_strategies\one_shot.py`

```py
# 导入必要的模块和库
from __future__ import annotations
import json
import platform
import re
from logging import Logger
from typing import TYPE_CHECKING, Callable, Optional
import distro

# 如果是类型检查，导入以下模块
if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.models.action_history import Episode

# 导入自定义异常
from autogpt.agents.utils.exceptions import InvalidAgentResponseError
# 导入配置文件
from autogpt.config import AIDirectives, AIProfile
# 导入系统配置和用户可配置
from autogpt.core.configuration.schema import SystemConfiguration, UserConfigurable
# 导入提示相关模块
from autogpt.core.prompting import (
    ChatPrompt,
    LanguageModelClassification,
    PromptStrategy,
)
# 导入模型提供者相关模块
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
# 导入 JSON 模式
from autogpt.core.utils.json_schema import JSONSchema
# 导入 JSON 工具
from autogpt.json_utils.utilities import extract_dict_from_response
# 导入提示工具
from autogpt.prompts.utils import format_numbered_list, indent

# 定义一个类，继承自 SystemConfiguration
class OneShotAgentPromptConfiguration(SystemConfiguration):
    # 默认的主体模板
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
        "These are the ONLY commands you can use."
        " Any action you perform must be possible through one of these commands:\n"
        "{commands}\n"
        "\n"
        "## Best practices\n"
        "{best_practices}"
    )

    # 默认的选择动作指令
    DEFAULT_CHOOSE_ACTION_INSTRUCTION: str = (
        "Determine exactly one command to use next based on the given goals "
        "and the progress you have made so far, "
        "and respond using the JSON schema specified previously:"
    )

    # 定义主体模板为可配置项
    body_template: str = UserConfigurable(default=DEFAULT_BODY_TEMPLATE)
    # 定义响应模式为可配置项
    response_schema: dict = UserConfigurable(
        default_factory=DEFAULT_RESPONSE_SCHEMA.to_dict
    )
    # 定义一个字符串类型的变量 choose_action_instruction，并设置默认值为 DEFAULT_CHOOSE_ACTION_INSTRUCTION
    choose_action_instruction: str = UserConfigurable(
        default=DEFAULT_CHOOSE_ACTION_INSTRUCTION
    )
    # 定义一个布尔类型的变量 use_functions_api，并设置默认值为 False
    use_functions_api: bool = UserConfigurable(default=False)

    #########
    # State #
    #########
    # progress_summaries: dict[tuple[int, int], str] = Field(
    #     default_factory=lambda: {(0, 0): ""}
    # )
# 定义一个名为 OneShotAgentPromptStrategy 的类，继承自 PromptStrategy 类
class OneShotAgentPromptStrategy(PromptStrategy):
    # 设置默认配置为 OneShotAgentPromptConfiguration 类的实例
    default_configuration: OneShotAgentPromptConfiguration = (
        OneShotAgentPromptConfiguration()
    )

    # 初始化方法，接受配置和日志对象作为参数
    def __init__(
        self,
        configuration: OneShotAgentPromptConfiguration,
        logger: Logger,
    ):
        # 将传入的配置赋值给实例变量 config
        self.config = configuration
        # 从配置中获取响应模式的 JSONSchema 对象，并赋值给实例变量 response_schema
        self.response_schema = JSONSchema.from_dict(configuration.response_schema)
        # 将传入的日志对象赋值给实例变量 logger
        self.logger = logger

    # 定义一个 model_classification 属性，返回 LanguageModelClassification 类的 FAST_MODEL 值
    @property
    def model_classification(self) -> LanguageModelClassification:
        return LanguageModelClassification.FAST_MODEL  # FIXME: dynamic switching

    # 构建提示方法，接受多个参数
    def build_prompt(
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
        """构建并返回一个具有以下结构的提示：
        1. 系统提示
        2. 代理的消息历史，根据需要截断并在需要时添加运行摘要
        3. `cycle_instruction`
        """
        如果没有额外消息，则将额外消息设置为空列表

        构建系统提示
        system_prompt = self.build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )
        计算系统提示的令牌长度
        system_prompt_tlength = count_message_tokens(ChatMessage.system(system_prompt))

        构建用户任务
        user_task = f'"""{task}"""'
        计算用户任务的令牌长度
        user_task_tlength = count_message_tokens(ChatMessage.user(user_task))

        构建响应格式指令
        response_format_instr = self.response_format_instruction(
            self.config.use_functions_api
        )
        将响应格式指令添加到额外消息中
        extra_messages.append(ChatMessage.system(response_format_instr))

        构建最终指令消息
        final_instruction_msg = ChatMessage.user(self.config.choose_action_instruction)
        计算最终指令消息的令牌长度
        final_instruction_tlength = count_message_tokens(final_instruction_msg)

        如果事件历史存在：
            编译进度信息
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
            在额外消息的开头插入进度信息
            extra_messages.insert(
                0,
                ChatMessage.system(f"## Progress\n\n{progress}"),
            )

        构建提示
        prompt = ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(user_task),
                *extra_messages,
                final_instruction_msg,
            ],
        )

        返回提示
        return prompt
    # 构建系统提示信息，包括 AI 个人资料、AI 指令、命令列表和是否包含操作系统信息
    def build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> str:
        # 生成系统提示信息的各个部分
        system_prompt_parts = (
            # 生成介绍性提示
            self._generate_intro_prompt(ai_profile)
            # 如果包含操作系统信息，则生成操作系统信息
            + (self._generate_os_info() if include_os_info else [])
            # 生成主体信息，包括约束、资源、命令列表和最佳实践
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
            # 添加用户任务说明
            + [
                "## Your Task\n"
                "The user will specify a task for you to execute, in triple quotes,"
                " in the next message. Your job is to complete the task while following"
                " your directives as given above, and terminate when your task is done."
            ]
        )

        # 将非空部分连接成段落格式，并去除开头和结尾的换行符
        return "\n\n".join(filter(None, system_prompt_parts)).strip("\n")

    # 编译进度信息，包括历史情节列表、最大标记数和计数标记的可选函数
    def compile_progress(
        self,
        episode_history: list[Episode],
        max_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        # 如果设置了最大令牌数但未设置计数令牌，则引发值错误
        if max_tokens and not count_tokens:
            raise ValueError("count_tokens is required if max_tokens is set")

        # 初始化步骤列表和令牌数
        steps: list[str] = []
        tokens: int = 0
        n_episodes = len(episode_history)

        # 遍历历史记录的每一集
        for i, episode in enumerate(reversed(episode_history)):
            # 对于最新的4个步骤或没有摘要的旧步骤，使用完整格式，否则使用摘要
            if i < 4 or episode.summary is None:
                step_content = indent(episode.format(), 2).strip()
            else:
                step_content = episode.summary

            step = f"* Step {n_episodes - i}: {step_content}"

            # 如果设置了最大令牌数和计数令牌
            if max_tokens and count_tokens:
                step_tokens = count_tokens(step)
                # 如果加上当前步骤的令牌数超过最大令牌数，则停止添加步骤
                if tokens + step_tokens > max_tokens:
                    break
                tokens += step_tokens

            # 将步骤插入到步骤列表的开头
            steps.insert(0, step)

        # 返回连接后的步骤字符串
        return "\n\n".join(steps)

    def response_format_instruction(self, use_functions_api: bool) -> str:
        # 复制响应模式以防止修改原始模式
        response_schema = self.response_schema.copy(deep=True)
        # 如果使用函数 API 并且响应模式包含属性且包含"command"属性，则删除"command"属性
        if (
            use_functions_api
            and response_schema.properties
            and "command" in response_schema.properties
        ):
            del response_schema.properties["command"]

        # 通过正则表达式去除响应模式中的缩进，提高性能
        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface("Response"),
        )

        # 根据使用函数 API 的情况生成指令
        instruction = (
            "Respond with pure JSON containing your thoughts, " "and invoke a tool."
            if use_functions_api
            else "Respond with pure JSON."
        )

        # 返回包含指令和响应格式的字符串
        return (
            f"{instruction} "
            "The JSON object should be compatible with the TypeScript type `Response` "
            f"from the following:\n{response_format}"
        )
    def _generate_intro_prompt(self, ai_profile: AIProfile) -> list[str]:
        """生成提示的介绍部分。

        Returns:
            list[str]: 由字符串组成的提示的介绍部分列表。
        """
        return [
            f"You are {ai_profile.ai_name}, {ai_profile.ai_role.rstrip('.')}.",
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications.",
        ]

    def _generate_os_info(self) -> list[str]:
        """生成提示的操作系统信息部分。

        Params:
            config (Config): 配置对象。

        Returns:
            str: 提示的操作系统信息部分。
        """
        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != "Linux"
            else distro.name(pretty=True)
        )
        return [f"The OS you are running on is: {os_info}"]

    def _generate_budget_constraint(self, api_budget: float) -> list[str]:
        """生成提示的预算信息部分。

        Returns:
            list[str]: 提示的预算信息部分，或者空列表。
        """
        if api_budget > 0.0:
            return [
                f"It takes money to let you run. "
                f"Your API budget is ${api_budget:.3f}"
            ]
        return []
    def _generate_commands_list(self, commands: list[CompletionModelFunction]) -> str:
        """生成代理可用的命令列表。

        Params:
            agent: 正在列出命令的代理。

        Returns:
            str: 包含命令编号列表的字符串。
        """
        try:
            # 调用 format_numbered_list 函数，将命令列表中每个命令的格式化行组成一个列表，然后返回格式化后的字符串
            return format_numbered_list([cmd.fmt_line() for cmd in commands])
        except AttributeError:
            # 如果格式化命令失败，则记录警告信息并抛出异常
            self.logger.warning(f"Formatting commands failed. {commands}")
            raise

    def parse_response_content(
        self,
        response: AssistantChatMessage,
    ) -> Agent.ThoughtProcessOutput:
        # 如果响应内容为空，则抛出无效代理响应错误
        if not response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        # 记录调试信息，包括 LLM 响应内容
        self.logger.debug(
            "LLM response content:"
            + (
                f"\n{response.content}"
                if "\n" in response.content
                else f" '{response.content}'"
            )
        )
        # 从 LLM 响应内容中提取字典对象
        assistant_reply_dict = extract_dict_from_response(response.content)
        # 记录调试信息，验证从 LLM 响应中提取的对象
        self.logger.debug(
            "Validating object extracted from LLM response:\n"
            f"{json.dumps(assistant_reply_dict, indent=4)}"
        )

        # 使用 response_schema 对象验证 assistant_reply_dict 对象，返回验证结果和错误列表
        _, errors = self.response_schema.validate_object(
            object=assistant_reply_dict,
            logger=self.logger,
        )
        # 如果存在错误，则抛出无效代理响应错误
        if errors:
            raise InvalidAgentResponseError(
                "Validation of response failed:\n  "
                + ";\n  ".join([str(e) for e in errors])
            )

        # 从 assistant_reply_dict 中提取命令名称和参数
        command_name, arguments = extract_command(
            assistant_reply_dict, response, self.config.use_functions_api
        )
        # 返回命令名称、参数和 assistant_reply_dict
        return command_name, arguments, assistant_reply_dict
# 解析 AI 的响应并返回命令名称和参数
def extract_command(
    assistant_reply_json: dict,  # AI 的响应对象
    assistant_reply: AssistantChatMessage,  # AI 的模型响应
    use_openai_functions_api: bool,  # 是否使用 OpenAI 函数 API
) -> tuple[str, dict[str, str]]:  # 返回值为命令名称和参数的元组

    # 如果使用 OpenAI 函数 API
    if use_openai_functions_api:
        # 如果助手回复中没有工具调用
        if not assistant_reply.tool_calls:
            raise InvalidAgentResponseError("No 'tool_calls' in assistant reply")
        
        # 将第一个工具调用的函数名称和参数解析为命令
        assistant_reply_json["command"] = {
            "name": assistant_reply.tool_calls[0].function.name,
            "args": json.loads(assistant_reply.tool_calls[0].function.arguments),
        }
    
    try:
        # 如果 assistant_reply_json 不是字典类型
        if not isinstance(assistant_reply_json, dict):
            raise InvalidAgentResponseError(
                f"The previous message sent was not a dictionary {assistant_reply_json}"
            )

        # 如果 assistant_reply_json 中没有 'command' 字段
        if "command" not in assistant_reply_json:
            raise InvalidAgentResponseError("Missing 'command' object in JSON")

        # 获取 'command' 字段
        command = assistant_reply_json["command"]
        
        # 如果 'command' 不是字典类型
        if not isinstance(command, dict):
            raise InvalidAgentResponseError("'command' object is not a dictionary")

        # 如果 'name' 字段不在 'command' 中
        if "name" not in command:
            raise InvalidAgentResponseError("Missing 'name' field in 'command' object")

        # 获取命令名称
        command_name = command["name"]

        # 如果 'args' 字段不存在，使用空字典
        arguments = command.get("args", {})

        return command_name, arguments

    # 捕获 JSON 解析错误
    except json.decoder.JSONDecodeError:
        raise InvalidAgentResponseError("Invalid JSON")
    # 捕获任何异常，并将异常信息转换为字符串，然后抛出一个自定义的异常InvalidAgentResponseError
    except Exception as e:
        raise InvalidAgentResponseError(str(e))
```