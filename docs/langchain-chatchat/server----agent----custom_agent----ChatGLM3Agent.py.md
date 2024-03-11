# `.\Langchain-Chatchat\server\agent\custom_agent\ChatGLM3Agent.py`

```py
"""
This file is a modified version for ChatGLM3-6B the original glm3_agent.py file from the langchain repo.
"""
# 该文件是从 langchain 仓库中的 glm3_agent.py 文件修改而来的 ChatGLM3-6B 的修改版本

from __future__ import annotations

import json
import logging
from typing import Any, List, Sequence, Tuple, Optional, Union
from pydantic.schema import model_schema

# 导入所需的模块和类

from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents.agent import Agent
from langchain.chains.llm import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.agents.agent import AgentOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.pydantic_v1 import Field
from langchain.schema import AgentAction, AgentFinish, OutputParserException, BasePromptTemplate
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool

# 定义一个消息模板
HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"
# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义一个带重试功能的输出解析器类
class StructuredChatOutputParserWithRetries(AgentOutputParser):
    """Output parser with retries for the structured chat agent."""

    # 基础解析器，使用 StructuredChatOutputParser 类
    base_parser: AgentOutputParser = Field(default_factory=StructuredChatOutputParser)
    # 输出修正解析器，可选
    output_fixing_parser: Optional[OutputFixingParser] = None
    # 解析输入的文本，返回代理动作或代理结束
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # 定义特殊标记
        special_tokens = ["Action:", "<|observation|>"]
        # 找到第一个特殊标记的位置
        first_index = min([text.find(token) if token in text else len(text) for token in special_tokens])
        # 截取文本到第一个特殊标记之前的部分
        text = text[:first_index]
        # 如果文本中包含 "tool_call"
        if "tool_call" in text:
            # 找到动作结束的位置
            action_end = text.find("```")
            # 提取动作
            action = text[:action_end].strip()
            # 找到参数字符串的起始和结束位置
            params_str_start = text.find("(") + 1
            params_str_end = text.rfind(")")
            # 提取参数字符串
            params_str = text[params_str_start:params_str_end]

            # 将参数字符串按逗号分割，再按等号分割，生成参数键值对列表
            params_pairs = [param.split("=") for param in params_str.split(",") if "=" in param]
            # 构建参数字典
            params = {pair[0].strip(): pair[1].strip().strip("'\"") for pair in params_pairs}

            # 构建动作的 JSON 对象
            action_json = {
                "action": action,
                "action_input": params
            }
        else:
            # 如果文本中不包含 "tool_call"，则构建默认动作的 JSON 对象
            action_json = {
                "action": "Final Answer",
                "action_input": text
            }
        # 构建动作的字符串表示
        action_str = f"""
# 定义一个类，表示结构化聊天代理
class StructuredGLM3ChatAgent(Agent):
    # 代理的输出解析器，默认为带重试的结构化聊天输出解析器
    output_parser: AgentOutputParser = Field(
        default_factory=StructuredChatOutputParserWithRetries
    )

    # 返回 ChatGLM3-6B 观察结果的前缀
    @property
    def observation_prefix(self) -> str:
        return "Observation:"

    # 返回 llm 调用的前缀
    @property
    def llm_prefix(self) -> str:
        return "Thought:"

    # 构建 scratchpad，包含中间步骤的信息
    def _construct_scratchpad(
            self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        # 调用父类的方法构建 scratchpad
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        # 检查 agent_scratchpad 是否为字符串类型
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")
        # 如果 agent_scratchpad 不为空，则返回包含前一工作的信息
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        # 如果 agent_scratchpad 为空，则直接返回
        else:
            return agent_scratchpad

    # 获取默认的输出解析器
    @classmethod
    def _get_default_output_parser(
            cls, llm: Optional[BaseLanguageModel] = None, **kwargs: Any
    ) -> AgentOutputParser:
        return StructuredChatOutputParserWithRetries(llm=llm)
    # 定义一个私有方法，返回一个包含"<|observation|>"的列表
    def _stop(self) -> List[str]:
        return ["<|observation|>"]

    # 创建一个提示模板，接受工具列表、提示语句、输入变量和记忆提示作为参数，返回一个基础提示模板
    @classmethod
    def create_prompt(
            cls,
            tools: Sequence[BaseTool],
            prompt: str = None,
            input_variables: Optional[List[str]] = None,
            memory_prompts: Optional[List[BasePromptTemplate]] = None,
    ) -> BasePromptTemplate:
        # 初始化工具列表和工具名称列表
        tools_json = []
        tool_names = []
        # 遍历工具列表，生成简化的工具配置信息
        for tool in tools:
            tool_schema = model_schema(tool.args_schema) if tool.args_schema else {}
            simplified_config_langchain = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool_schema.get("properties", {})
            }
            tools_json.append(simplified_config_langchain)
            tool_names.append(tool.name)
        # 格式化工具信息，替换特殊字符
        formatted_tools = "\n".join([
            f"{tool['name']}: {tool['description']}, args: {tool['parameters']}"
            for tool in tools_json
        ])
        formatted_tools = formatted_tools.replace("'", "\\'").replace("{", "{{").replace("}", "}}")
        # 根据提示语句格式化模板
        template = prompt.format(tool_names=tool_names,
                                 tools=formatted_tools,
                                 history="None",
                                 input="{input}",
                                 agent_scratchpad="{agent_scratchpad}")

        # 如果输入变量为空，则默认为["input", "agent_scratchpad"]
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        # 如果记忆提示为空，则初始化为空列表
        _memory_prompts = memory_prompts or []
        # 生成消息列表，包括系统消息和记忆提示
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            *_memory_prompts,
        ]
        # 返回聊天提示模板，包括输入变量和消息列表
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    # 类方法结束
    @classmethod
    def from_llm_and_tools(
            cls,
            llm: BaseLanguageModel,
            tools: Sequence[BaseTool],
            prompt: str = None,
            callback_manager: Optional[BaseCallbackManager] = None,
            output_parser: Optional[AgentOutputParser] = None,
            human_message_template: str = HUMAN_MESSAGE_TEMPLATE,
            input_variables: Optional[List[str]] = None,
            memory_prompts: Optional[List[BasePromptTemplate]] = None,
            **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        # 验证工具的有效性
        cls._validate_tools(tools)
        # 创建提示信息
        prompt = cls.create_prompt(
            tools,
            prompt=prompt,
            input_variables=input_variables,
            memory_prompts=memory_prompts,
        )
        # 创建 LLMChain 对象
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        # 获取工具名称列表
        tool_names = [tool.name for tool in tools]
        # 获取输出解析器，如果没有则使用默认解析器
        _output_parser = output_parser or cls._get_default_output_parser(llm=llm)
        # 返回 Agent 对象
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @property
    def _agent_type(self) -> str:
        # 抛出数值错误异常
        raise ValueError
# 初始化一个基于 GLM3 的代理程序
def initialize_glm3_agent(
        tools: Sequence[BaseTool],  # 接受一个工具序列作为参数
        llm: BaseLanguageModel,  # 接受一个语言模型对象作为参数
        prompt: str = None,  # 可选的提示字符串参数，默认为 None
        memory: Optional[ConversationBufferWindowMemory] = None,  # 可选的对话缓冲窗口内存参数，默认为 None
        agent_kwargs: Optional[dict] = None,  # 可选的代理参数字典，默认为 None
        *,
        tags: Optional[Sequence[str]] = None,  # 可选的标签序列参数，默认为 None
        **kwargs: Any,  # 接受任意其他关键字参数
) -> AgentExecutor:  # 返回一个代理执行器对象
    tags_ = list(tags) if tags else []  # 如果 tags 存在则转换为列表，否则为空列表
    agent_kwargs = agent_kwargs or {}  # 如果代理参数存在则使用，否则为空字典
    agent_obj = StructuredGLM3ChatAgent.from_llm_and_tools(  # 从语言模型和工具创建一个结构化的 GLM3 聊天代理对象
        llm=llm,  # 传入语言模型参数
        tools=tools,  # 传入工具参数
        prompt=prompt,  # 传入提示字符串参数
        **agent_kwargs  # 传入代理参数字典
    )
    return AgentExecutor.from_agent_and_tools(  # 从代理对象和工具创建一个代理执行器对象
        agent=agent_obj,  # 传入代理对象参数
        tools=tools,  # 传入工具参数
        memory=memory,  # 传入内存参数
        tags=tags_,  # 传入标签参数
        **kwargs,  # 传入其他关键字参数
    )
```