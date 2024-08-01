# `.\DB-GPT-src\dbgpt\agent\core\agent.py`

```py
"""Agent Interface."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from dbgpt.core import LLMClient  # 导入LLMClient类
from dbgpt.util.annotations import PublicAPI  # 导入PublicAPI注解

from .action.base import ActionOutput  # 导入ActionOutput类
from .memory.agent_memory import AgentMemory  # 导入AgentMemory类


class Agent(ABC):
    """Agent Interface."""

    @abstractmethod
    async def send(  # 定义异步抽象方法send，用于发送消息给接收者代理
        self,
        message: AgentMessage,  # 参数：消息内容，类型为AgentMessage
        recipient: Agent,  # 参数：接收者代理，类型为Agent
        reviewer: Optional[Agent] = None,  # 可选参数：评审者代理，类型为Agent，默认为None
        request_reply: Optional[bool] = True,  # 可选参数：是否请求回复，类型为bool，默认为True
        is_recovery: Optional[bool] = False,  # 可选参数：消息是否为恢复消息，类型为bool，默认为False
    ) -> None:
        """Send a message to recipient agent.

        Args:
            message(AgentMessage): the message to be sent.
            recipient(Agent): the recipient agent.
            reviewer(Agent): the reviewer agent.
            request_reply(bool): whether to request a reply.
            is_recovery(bool): whether the message is a recovery message.

        Returns:
            None
        """

    @abstractmethod
    async def receive(  # 定义异步抽象方法receive，用于接收来自其他代理的消息
        self,
        message: AgentMessage,  # 参数：接收到的消息，类型为AgentMessage
        sender: Agent,  # 参数：发送者代理，类型为Agent
        reviewer: Optional[Agent] = None,  # 可选参数：评审者代理，类型为Agent，默认为None
        request_reply: Optional[bool] = None,  # 可选参数：是否请求回复，类型为bool，默认为None
        silent: Optional[bool] = False,  # 可选参数：是否静默，类型为bool，默认为False
        is_recovery: Optional[bool] = False,  # 可选参数：消息是否为恢复消息，类型为bool，默认为False
    ) -> None:
        """Receive a message from another agent.

        Args:
            message(AgentMessage): the received message.
            sender(Agent): the sender agent.
            reviewer(Agent): the reviewer agent.
            request_reply(bool): whether to request a reply.
            silent(bool): whether to be silent.
            is_recovery(bool): whether the message is a recovery message.

        Returns:
            None
        """

    @abstractmethod
    async def generate_reply(  # 定义异步抽象方法generate_reply，基于接收到的消息生成回复
        self,
        received_message: AgentMessage,  # 参数：接收到的消息，类型为AgentMessage
        sender: Agent,  # 参数：消息发送者的Agent实例
        reviewer: Optional[Agent] = None,  # 可选参数：评审者的Agent实例，默认为None
        rely_messages: Optional[List[AgentMessage]] = None,  # 可选参数：接收到的消息列表，默认为None
        **kwargs,  # 其他关键字参数
    ) -> AgentMessage:
        """Generate a reply based on the received messages.

        Args:
            received_message(AgentMessage): the received message.
            sender: sender of an Agent instance.
            reviewer: reviewer of an Agent instance.
            rely_messages: a list of messages received.

        Returns:
            AgentMessage: the generated reply. If None, no reply is generated.
        """

    @abstractmethod
    async def thinking(  # 定义异步抽象方法thinking，用于处理思考过程
        self, messages: List[AgentMessage], prompt: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        定义一个方法签名，指定返回值类型为元组，包含两个可选的字符串或None。
        """

    @abstractmethod
    async def review(self, message: Optional[str], censored: Agent) -> Tuple[bool, Any]:
        """
        抽象方法：基于被审查的消息进行审核。

        Args:
            message: 待审核的消息
            censored: 被审查的代理

        Returns:
            bool: 消息是否被审查
            Any: 被审查后的消息
        """

    @abstractmethod
    async def act(
        self,
        message: Optional[str],
        sender: Optional[Agent] = None,
        reviewer: Optional[Agent] = None,
        **kwargs,
    ) -> Optional[ActionOutput]:
        """
        抽象方法：基于LLM推理结果执行操作。

        Args:
            message: 待执行的消息
            sender: 发送者的代理实例
            reviewer: 审查者的代理实例
            **kwargs: 其他参数

        Returns:
             ActionOutput: 代理的操作输出
        """

    @abstractmethod
    async def verify(
        self,
        message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[str]]:
        """
        抽象方法：验证当前执行结果是否符合目标期望。

        Args:
            message: 待验证的消息
            sender: 发送者的代理实例
            reviewer: 审查者的代理实例
            **kwargs: 其他参数

        Returns:
            Tuple[bool, Optional[str]]: 验证是否成功及验证结果
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        抽象属性：返回代理的名称。
        """

    @property
    @abstractmethod
    def role(self) -> str:
        """
        抽象属性：返回代理的角色。
        """

    @property
    @abstractmethod
    def desc(self) -> Optional[str]:
        """
        抽象属性：返回代理的描述信息。
        """
@dataclasses.dataclass
@PublicAPI(stability="beta")
class AgentReviewInfo:
    """Message object for agent communication."""

    # 是否批准，默认为 False
    approve: bool = False
    # 审核意见，可选
    comments: Optional[str] = None

    def copy(self) -> "AgentReviewInfo":
        """Return a copy of the current AgentReviewInfo."""
        return AgentReviewInfo(approve=self.approve, comments=self.comments)


@dataclasses.dataclass
@PublicAPI(stability="beta")
class AgentMessage:
    """Message object for agent communication."""

    # 消息内容，可选
    content: Optional[str] = None
    # 消息名称，可选
    name: Optional[str] = None
    # 消息上下文，可以是字符串或字典类型
    context: Optional[MessageContextType] = None
    # 动作报告类型，可以是任意字典类型
    action_report: Optional[ActionReportType] = None
    # 审核信息，AgentReviewInfo 类型，可选
    review_info: Optional[AgentReviewInfo] = None
    # 当前目标，可选
    current_goal: Optional[str] = None
    # 模型名称，可选
    model_name: Optional[str] = None
    # 角色，可选
    role: Optional[str] = None
    # 成功标志，可选
    success: Optional[bool] = None

    def to_dict(self) -> Dict:
        """Return a dictionary representation of the AgentMessage."""
        return dataclasses.asdict(self)

    def to_llm_message(self) -> Dict[str, Any]:
        """Return a dictionary representation of the AgentMessage."""
        return {
            "content": self.content,
            "context": self.context,
            "role": self.role,
        }

    @classmethod
    def from_llm_message(cls, message: Dict[str, Any]) -> AgentMessage:
        """Create an AgentMessage object from a dictionary."""
        return cls(
            content=message.get("content"),
            context=message.get("context"),
            role=message.get("role"),
        )

    @classmethod
    # 根据给定的消息字典列表创建 AgentMessage 对象列表
    def from_messages(cls, messages: List[Dict[str, Any]]) -> List[AgentMessage]:
        """Create a list of AgentMessage objects from a list of dictionaries."""
        # 获取类的所有字段名列表
        field_names = [f.name for f in dataclasses.fields(cls)]
        # 初始化结果列表
        results = []
        # 遍历每个消息字典
        for message in messages:
            # 从消息字典中筛选出与类字段名匹配的键值对作为关键字参数
            kwargs = {
                key: value for key, value in message.items() if key in field_names
            }
            # 使用关键字参数创建 AgentMessage 对象，并添加到结果列表中
            results.append(cls(**kwargs))
        # 返回创建的 AgentMessage 对象列表
        return results

    # 返回当前 AgentMessage 对象的副本
    def copy(self) -> "AgentMessage":
        """Return a copy of the current AgentMessage."""
        # 初始化副本的上下文信息
        copied_context: Optional[MessageContextType] = None
        # 如果存在上下文信息
        if self.context:
            # 如果上下文信息是字典类型，则进行深拷贝
            if isinstance(self.context, dict):
                copied_context = self.context.copy()
            else:
                # 否则直接赋值给副本
                copied_context = self.context
        # 复制动作报告，如果存在的话
        copied_action_report = self.action_report.copy() if self.action_report else None
        # 复制审核信息，如果存在的话
        copied_review_info = self.review_info.copy() if self.review_info else None
        # 返回一个新的 AgentMessage 对象，包含复制的各项属性
        return AgentMessage(
            content=self.content,
            name=self.name,
            context=copied_context,
            action_report=copied_action_report,
            review_info=copied_review_info,
            current_goal=self.current_goal,
            model_name=self.model_name,
            role=self.role,
            success=self.success,
        )

    # 将当前 AgentMessage 对象的上下文信息转换为字典形式并返回
    def get_dict_context(self) -> Dict[str, Any]:
        """Return the context as a dictionary."""
        # 如果上下文信息是字典类型，则直接返回
        if isinstance(self.context, dict):
            return self.context
        # 否则返回一个空字典
        return {}
```