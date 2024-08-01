# `.\DB-GPT-src\dbgpt\agent\core\memory\gpts\base.py`

```py
"""Base memory interface for agents."""

import dataclasses  # 导入dataclasses模块，用于支持数据类的定义
from abc import ABC, abstractmethod  # 从abc模块导入ABC抽象基类和abstractmethod装饰器
from datetime import datetime  # 导入datetime模块中的datetime类
from typing import Any, Dict, List, Optional  # 导入类型提示相关的类和装饰器

from ...schema import Status  # 从特定位置导入Status类


@dataclasses.dataclass
class GptsPlan:
    """Gpts plan."""

    conv_id: str  # 对话ID，字符串类型
    sub_task_num: int  # 子任务数量，整数类型
    sub_task_content: Optional[str]  # 子任务内容，可选的字符串类型
    sub_task_title: Optional[str] = None  # 子任务标题，可选的字符串类型，默认为None
    sub_task_agent: Optional[str] = None  # 子任务代理，可选的字符串类型，默认为None
    resource_name: Optional[str] = None  # 资源名称，可选的字符串类型，默认为None
    rely: Optional[str] = None  # 依赖关系，可选的字符串类型，默认为None
    agent_model: Optional[str] = None  # 代理模型，可选的字符串类型，默认为None
    retry_times: int = 0  # 重试次数，整数类型，默认为0
    max_retry_times: int = 5  # 最大重试次数，整数类型，默认为5
    state: Optional[str] = Status.TODO.value  # 状态，可选的字符串类型，初始为Status.TODO.value
    result: Optional[str] = None  # 结果，可选的字符串类型，默认为None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GptsPlan":
        """Create a GptsPlan object from a dictionary."""
        return GptsPlan(
            conv_id=d["conv_id"],  # 从字典中获取conv_id字段赋给对象属性
            sub_task_num=d["sub_task_num"],  # 从字典中获取sub_task_num字段赋给对象属性
            sub_task_content=d["sub_task_content"],  # 从字典中获取sub_task_content字段赋给对象属性
            sub_task_agent=d["sub_task_agent"],  # 从字典中获取sub_task_agent字段赋给对象属性
            resource_name=d["resource_name"],  # 从字典中获取resource_name字段赋给对象属性
            rely=d["rely"],  # 从字典中获取rely字段赋给对象属性
            agent_model=d["agent_model"],  # 从字典中获取agent_model字段赋给对象属性
            retry_times=d["retry_times"],  # 从字典中获取retry_times字段赋给对象属性
            max_retry_times=d["max_retry_times"],  # 从字典中获取max_retry_times字段赋给对象属性
            state=d["state"],  # 从字典中获取state字段赋给对象属性
            result=d["result"],  # 从字典中获取result字段赋给对象属性
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the GptsPlan object."""
        return dataclasses.asdict(self)  # 将数据类对象转换为字典表示


@dataclasses.dataclass
class GptsMessage:
    """Gpts message."""

    conv_id: str  # 对话ID，字符串类型
    sender: str  # 发送者，字符串类型
    receiver: str  # 接收者，字符串类型
    role: str  # 角色，字符串类型
    content: str  # 内容，字符串类型
    rounds: Optional[int]  # 轮次，可选的整数类型
    current_goal: Optional[str] = None  # 当前目标，可选的字符串类型，默认为None
    context: Optional[str] = None  # 上下文，可选的字符串类型，默认为None
    review_info: Optional[str] = None  # 审查信息，可选的字符串类型，默认为None
    action_report: Optional[str] = None  # 行动报告，可选的字符串类型，默认为None
    model_name: Optional[str] = None  # 模型名称，可选的字符串类型，默认为None
    created_at: datetime = dataclasses.field(default_factory=datetime.utcnow)  # 创建时间，datetime类型，默认为当前UTC时间
    updated_at: datetime = dataclasses.field(default_factory=datetime.utcnow)  # 更新时间，datetime类型，默认为当前UTC时间

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GptsMessage":
        """Create a GptsMessage object from a dictionary."""
        return GptsMessage(
            conv_id=d["conv_id"],  # 从字典中获取conv_id字段赋给对象属性
            sender=d["sender"],  # 从字典中获取sender字段赋给对象属性
            receiver=d["receiver"],  # 从字典中获取receiver字段赋给对象属性
            role=d["role"],  # 从字典中获取role字段赋给对象属性
            content=d["content"],  # 从字典中获取content字段赋给对象属性
            rounds=d["rounds"],  # 从字典中获取rounds字段赋给对象属性
            model_name=d["model_name"],  # 从字典中获取model_name字段赋给对象属性
            current_goal=d["current_goal"],  # 从字典中获取current_goal字段赋给对象属性
            context=d["context"],  # 从字典中获取context字段赋给对象属性
            review_info=d["review_info"],  # 从字典中获取review_info字段赋给对象属性
            action_report=d["action_report"],  # 从字典中获取action_report字段赋给对象属性
            created_at=d["created_at"],  # 从字典中获取created_at字段赋给对象属性
            updated_at=d["updated_at"],  # 从字典中获取updated_at字段赋给对象属性
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the GptsMessage object."""
        return dataclasses.asdict(self)  # 将数据类对象转换为字典表示


class GptsPlansMemory(ABC):
    """Gpts plans memory interface."""

    @abstractmethod  # 抽象方法声明，用于子类实现
    @abstractmethod
    def batch_save(self, plans: List[GptsPlan]) -> None:
        """Save plans in batch.

        Args:
            plans: List of GptsPlan objects to be saved in batch
        """
    
    @abstractmethod
    def get_by_conv_id(self, conv_id: str) -> List[GptsPlan]:
        """Get plans by conversation id.

        Args:
            conv_id: Conversation id to retrieve plans for

        Returns:
            List[GptsPlan]: List of GptsPlan objects associated with the conversation id
        """

    @abstractmethod
    def get_by_conv_id_and_num(
        self, conv_id: str, task_nums: List[int]
    ) -> List[GptsPlan]:
        """Get plans by conversation id and task numbers.

        Args:
            conv_id(str): Conversation id to retrieve plans for
            task_nums(List[int]): List of task numbers to retrieve

        Returns:
            List[GptsPlan]: List of GptsPlan objects matching the conversation id and task numbers
        """

    @abstractmethod
    def get_todo_plans(self, conv_id: str) -> List[GptsPlan]:
        """Get unfinished planning steps for a conversation.

        Args:
            conv_id(str): Conversation id to retrieve unfinished plans for

        Returns:
            List[GptsPlan]: List of unfinished GptsPlan objects for the conversation
        """

    @abstractmethod
    def complete_task(self, conv_id: str, task_num: int, result: str) -> None:
        """Set a planning step as complete.

        Args:
            conv_id(str): Conversation id
            task_num(int): Task number to mark as complete
            result(str): Result of the planning step
        """

    @abstractmethod
    def update_task(
        self,
        conv_id: str,
        task_num: int,
        state: str,
        retry_times: int,
        agent: Optional[str] = None,
        model: Optional[str] = None,
        result: Optional[str] = None,
    ) -> None:
        """Update information of a planning step.

        Args:
            conv_id(str): Conversation id
            task_num(int): Task number to update
            state(str): Updated status of the task
            retry_times(int): Number of retries attempted
            agent(str, optional): Agent's name involved in the task
            model(str, optional): Model name used for the task
            result(str, optional): Result of the planning step
        """

    @abstractmethod
    def remove_by_conv_id(self, conv_id: str) -> None:
        """Remove plans associated with a conversation id.

        Args:
            conv_id(str): Conversation id to remove plans for
        """
class GptsMessageMemory(ABC):
    """Gpts message memory interface."""

    @abstractmethod
    def append(self, message: GptsMessage) -> None:
        """Add a message.

        Args:
            message(GptsMessage): Message object
        """
        
    @abstractmethod
    def get_by_agent(self, conv_id: str, agent: str) -> Optional[List[GptsMessage]]:
        """Return all messages of the agent in the conversation.

        Args:
            conv_id(str): Conversation id
            agent(str): Agent's name

        Returns:
            List[GptsMessage]: List of messages
        """

    @abstractmethod
    def get_between_agents(
        self,
        conv_id: str,
        agent1: str,
        agent2: str,
        current_goal: Optional[str] = None,
    ) -> List[GptsMessage]:
        """Get messages between two agents.

        Query information related to an agent

        Args:
            conv_id(str): Conversation id
            agent1(str): Agent1's name
            agent2(str): Agent2's name
            current_goal(str): Current goal

        Returns:
            List[GptsMessage]: List of messages
        """

    @abstractmethod
    def get_by_conv_id(self, conv_id: str) -> List[GptsMessage]:
        """Return all messages in the conversation.

        Query messages by conv id.

        Args:
            conv_id(str): Conversation id
        Returns:
            List[GptsMessage]: List of messages
        """

    @abstractmethod
    def get_last_message(self, conv_id: str) -> Optional[GptsMessage]:
        """Return the last message in the conversation.

        Args:
            conv_id(str): Conversation id

        Returns:
            GptsMessage: The last message in the conversation
        """



# GptsMessageMemory 类定义了一个抽象基类接口，用于管理消息存储和检索的操作。

## append 方法

- 添加一条消息到存储器中。
- 参数 `message`：GptsMessage 对象，表示要添加的消息。

## get_by_agent 方法

- 返回指定会话中特定代理人的所有消息。
- 参数 `conv_id`：会话 ID。
- 参数 `agent`：代理人的名称。
- 返回值：一个可选的 GptsMessage 对象列表，代表特定代理人的所有消息。

## get_between_agents 方法

- 获取两个代理人之间的消息列表。
- 参数 `conv_id`：会话 ID。
- 参数 `agent1`：第一个代理人的名称。
- 参数 `agent2`：第二个代理人的名称。
- 参数 `current_goal`（可选）：当前的目标。
- 返回值：一个 GptsMessage 对象列表，代表两个代理人之间的所有消息。

## get_by_conv_id 方法

- 返回特定会话中的所有消息。
- 参数 `conv_id`：会话 ID。
- 返回值：一个 GptsMessage 对象列表，代表特定会话中的所有消息。

## get_last_message 方法

- 返回特定会话中的最后一条消息。
- 参数 `conv_id`：会话 ID。
- 返回值：一个可选的 GptsMessage 对象，代表会话中的最后一条消息。
```