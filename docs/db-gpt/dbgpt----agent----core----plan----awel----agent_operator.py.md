# `.\DB-GPT-src\dbgpt\agent\core\plan\awel\agent_operator.py`

```py
"""Agent Operator for AWEL."""

# 从 ABC 模块导入 ABC 类，用于定义抽象基类
from abc import ABC
# 从 typing 模块导入 List、Optional、Type 类型提示工具
from typing import List, Optional, Type

# 从 dbgpt.core.awel 模块导入 MapOperator 类
from dbgpt.core.awel import MapOperator
# 从 dbgpt.core.awel.flow 模块导入 IOField、OperatorCategory、OperatorType、Parameter、ViewMetadata 类
from dbgpt.core.awel.flow import (
    IOField,
    OperatorCategory,
    OperatorType,
    Parameter,
    ViewMetadata,
)
# 从 dbgpt.core.awel.trigger.base 模块导入 Trigger 类
from dbgpt.core.awel.trigger.base import Trigger
# 从 dbgpt.core.interface.message 模块导入 ModelMessageRoleType 类
from dbgpt.core.interface.message import ModelMessageRoleType

# TODO: 不依赖 MixinLLMOperator
# 从 dbgpt.model.operators.llm_operator 模块导入 MixinLLMOperator 类
from dbgpt.model.operators.llm_operator import MixinLLMOperator

# 从 ....resource.manage 模块导入 get_resource_manager 函数
from ....resource.manage import get_resource_manager
# 从 ....util.llm.llm 模块导入 LLMConfig 类
from ....util.llm.llm import LLMConfig
# 从 ...agent 模块导入 Agent、AgentGenerateContext、AgentMessage 类
from ...agent import Agent, AgentGenerateContext, AgentMessage
# 从 ...agent_manage 模块导入 get_agent_manager 函数
from ...agent_manage import get_agent_manager
# 从 ...base_agent 模块导入 ConversableAgent 类
from ...base_agent import ConversableAgent
# 从 .agent_operator_resource 模块导入 AWELAgent 类
from .agent_operator_resource import AWELAgent


class BaseAgentOperator:
    """The abstract operator for an Agent."""

    # 共享数据键名，代表模型名称
    SHARE_DATA_KEY_MODEL_NAME = "share_data_key_agent_name"

    def __init__(self, agent: Optional[Agent] = None):
        """Create an AgentOperator."""
        # 初始化方法，接收一个可选的 Agent 实例作为参数
        self._agent = agent

    @property
    def agent(self) -> Agent:
        """Return the Agent."""
        # 返回当前实例的 Agent 属性，如果未设置则抛出 ValueError 异常
        if not self._agent:
            raise ValueError("agent is not set")
        return self._agent


class WrappedAgentOperator(
    BaseAgentOperator, MapOperator[AgentGenerateContext, AgentGenerateContext], ABC
):
    """The Agent operator.

    Wrap the agent and trigger the agent to generate a reply.
    """

    def __init__(self, agent: Agent, **kwargs):
        """Create an WrappedAgentOperator."""
        # 调用父类 BaseAgentOperator 的初始化方法，传入 agent 参数
        super().__init__(agent=agent)
        # 调用 MapOperator 的初始化方法，传入任意额外关键字参数
        MapOperator.__init__(self, **kwargs)
    # 异步方法定义，用于映射处理输入消息并生成回复的过程
    async def map(self, input_value: AgentGenerateContext) -> AgentGenerateContext:
        """Trigger agent to generate a reply."""
        # 初始化一个空列表，用于存储当前回复的消息
        now_rely_messages: List[AgentMessage] = []

        # 如果输入消息为空，则抛出数值错误异常
        if not input_value.message:
            raise ValueError("The message is empty.")

        # 复制输入消息，以便后续操作
        input_message = input_value.message.copy()

        # 确定当前目标，基于代理名称或角色
        _goal = self.agent.name if self.agent.name else self.agent.role
        current_goal = f"[{_goal}]:"

        # 如果输入消息包含内容，则将其添加到当前目标中
        if input_message.content:
            current_goal += input_message.content

        # 将当前目标附加到输入消息的属性中
        input_message.current_goal = current_goal

        # 复制人类消息，设置其角色为人类，并添加到当前回复消息列表中
        human_message = input_message.copy()
        human_message.role = ModelMessageRoleType.HUMAN
        now_rely_messages.append(human_message)

        # 准备要发送的消息，并检查是否需要依赖前序消息
        now_message = input_message
        if input_value.rely_messages and len(input_value.rely_messages) > 0:
            now_message = input_value.rely_messages[-1]

        # 如果发送者为空，则抛出数值错误异常
        if not input_value.sender:
            raise ValueError("The sender is empty.")

        # 使用异步发送消息，参数为当前消息、代理、评审者和False（表示不需要回复）
        await input_value.sender.send(
            now_message, self.agent, input_value.reviewer, False
        )

        # 调用代理的生成回复方法，获取代理回复消息
        agent_reply_message = await self.agent.generate_reply(
            received_message=input_message,
            sender=input_value.sender,
            reviewer=input_value.reviewer,
            rely_messages=input_value.rely_messages,
        )

        # 检查代理回复消息的成功状态，若不成功则抛出数值错误异常，附带失败原因
        is_success = agent_reply_message.success
        if not is_success:
            raise ValueError(
                f"The task failed at step {self.agent.role} and the attempt "
                f"to repair it failed. The final reason for "
                f"failure:{agent_reply_message.content}!"
            )

        # 复制AI消息，设置其角色为AI，并添加到当前回复消息列表中
        ai_message = agent_reply_message.copy()
        ai_message.role = ModelMessageRoleType.AI
        now_rely_messages.append(ai_message)

        # 返回生成的上下文对象，包括输入消息、发送者、评审者、当前消息列表、和是否静默处理的标志
        return AgentGenerateContext(
            message=input_message,
            sender=self.agent,
            reviewer=input_value.reviewer,
            rely_messages=now_rely_messages,
            silent=input_value.silent,
        )
class AWELAgentOperator(
    MixinLLMOperator, MapOperator[AgentGenerateContext, AgentGenerateContext]
):
    """The Agent operator for AWEL."""

    # 定义元数据，描述 AWEL Agent 运算符的属性和行为
    metadata = ViewMetadata(
        label="AWEL Agent Operator",
        name="agent_operator",
        category=OperatorCategory.AGENT,
        description="The Agent operator.",
        parameters=[
            Parameter.build_from(
                "Agent",
                "awel_agent",
                AWELAgent,
                description="The dbgpt agent.",
            ),
        ],
        inputs=[
            IOField.build_from(
                "Agent Operator Request",
                "agent_operator_request",
                AgentGenerateContext,
                "The Agent Operator request.",
            )
        ],
        outputs=[
            IOField.build_from(
                "Agent Operator Output",
                "agent_operator_output",
                AgentGenerateContext,
                description="The Agent Operator output.",
            )
        ],
    )

    def __init__(self, awel_agent: AWELAgent, **kwargs):
        """Create an AgentOperator."""
        # 初始化操作符实例，设置 AWEL Agent 并调用父类构造函数
        MixinLLMOperator.__init__(self)
        MapOperator.__init__(self, **kwargs)
        self.awel_agent = awel_agent

    async def map(
        self,
        input_value: AgentGenerateContext,
    ) -> AgentGenerateContext:
        """Build the agent."""
        # 根据 AWEL Agent 的配置构建 ConversableAgent 对象

        # 获取 agent_cls 类型，根据 awel_agent.agent_profile 从管理器中获取对应的 agent 类型
        agent_cls: Type[ConversableAgent] = get_agent_manager().get_by_name(
            self.awel_agent.agent_profile
        )
        # 获取 awel_agent 的 llm_config
        llm_config = self.awel_agent.llm_config

        # 如果 llm_config 不存在，则根据 input_value 或默认值创建新的 LLMConfig
        if not llm_config:
            if input_value.llm_client:
                llm_config = LLMConfig(llm_client=input_value.llm_client)
            else:
                llm_config = LLMConfig(llm_client=self.llm_client)
        else:
            # 如果 llm_config 存在但 llm_client 为空，则根据 input_value 或默认值设置 llm_client
            if not llm_config.llm_client:
                if input_value.llm_client:
                    llm_config.llm_client = input_value.llm_client
                else:
                    llm_config.llm_client = self.llm_client

        kwargs = {}
        # 如果 awel_agent 的 role_name 存在，则设置 kwargs 中的 "name"
        if self.awel_agent.role_name:
            kwargs["name"] = self.awel_agent.role_name
        # 如果 awel_agent 的 fixed_subgoal 存在，则设置 kwargs 中的 "fixed_subgoal"
        if self.awel_agent.fixed_subgoal:
            kwargs["fixed_subgoal"] = self.awel_agent.fixed_subgoal

        # 根据 awel_agent 的 resources 属性构建 resource 对象
        resource = get_resource_manager().build_resource(self.awel_agent.resources)
        
        # 使用 agent_cls 类型构建 agent 对象，绑定各种配置和资源
        agent = (
            await agent_cls(**kwargs)
            .bind(input_value.memory)
            .bind(llm_config)
            .bind(input_value.agent_context)
            .bind(resource)
            .build()
        )

        return agent


class AgentDummyTrigger(Trigger):
    """Http trigger for AWEL.

    Http trigger is used to trigger a DAG by http request.
    """
    metadata = ViewMetadata(
        label="Agent Trigger",  # 设置视图的标签为 "Agent Trigger"
        name="agent_trigger",  # 设置视图的名称为 "agent_trigger"
        category=OperatorCategory.AGENT,  # 将视图分类为代理类别
        operator_type=OperatorType.INPUT,  # 设置操作类型为输入操作
        description="Trigger your workflow by agent",  # 设置视图的描述信息
        inputs=[],  # 设置视图的输入为空列表
        parameters=[],  # 设置视图的参数为空列表
        outputs=[
            IOField.build_from(
                "Agent Operator Context",  # 输出字段的名称为 "Agent Operator Context"
                "agent_operator_context",  # 输出字段的标识符为 "agent_operator_context"
                AgentGenerateContext,  # 输出字段的类型为 AgentGenerateContext
                description="The Agent Operator output.",  # 输出字段的描述信息
            )
        ],  # 设置视图的输出为包含一个输出字段的列表
    )
    
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Initialize a HttpTrigger."""  # 初始化方法，用于实例化 HttpTrigger 类
    
        super().__init__(**kwargs)  # 调用父类的初始化方法，并传递所有关键字参数
    
    async def trigger(self, **kwargs) -> None:
        """Trigger the DAG. Not used in HttpTrigger."""  # 触发方法，用于触发 DAG，但在 HttpTrigger 中未使用
    
        raise NotImplementedError("Dummy trigger does not support trigger.")
        # 抛出 NotImplementedError 异常，指示该触发方法未实现，不支持触发操作
```