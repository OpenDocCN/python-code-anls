# `.\DB-GPT-src\dbgpt\core\interface\message.py`

```py
"""The conversation and message module."""

# 引入未来的注解功能，用于支持自引用类型的类型注解
from __future__ import annotations

# 引入必要的模块和类
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

# 引入 dbgpt 内部的模型和字段定义
from dbgpt._private.pydantic import BaseModel, Field, model_to_dict
# 引入存储接口相关类
from dbgpt.core.interface.storage import (
    InMemoryStorage,
    ResourceIdentifier,
    StorageInterface,
    StorageItem,
)


class BaseMessage(BaseModel, ABC):
    """Message object."""

    # 消息内容
    content: str
    # 消息索引，默认为0
    index: int = 0
    # 消息在对话中的回合索引，默认为0
    round_index: int = 0
    """The round index of the message in the conversation"""
    # 附加关键字参数
    additional_kwargs: dict = Field(default_factory=dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""

    @property
    def pass_to_model(self) -> bool:
        """Whether the message will be passed to the model."""
        return True

    def to_dict(self) -> Dict:
        """Convert to dict.

        Returns:
            Dict: The dict object
        """
        # 将消息对象转换为字典形式
        return {
            "type": self.type,
            "data": model_to_dict(self),
            "index": self.index,
            "round_index": self.round_index,
        }

    @staticmethod
    def messages_to_string(messages: List["BaseMessage"]) -> str:
        """Convert messages to str.

        Args:
            messages (List[BaseMessage]): The messages

        Returns:
            str: The str messages
        """
        # 将消息对象列表转换为字符串形式
        return _messages_to_str(messages)


class HumanMessage(BaseMessage):
    """Type of message that is spoken by the human."""

    # 是否为示例消息，默认为False
    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "human"


class AIMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    # 是否为示例消息，默认为False
    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ai"


class ViewMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    # 是否为示例消息，默认为False
    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "view"

    @property
    def pass_to_model(self) -> bool:
        """Whether the message will be passed to the model.

        The view message will not be passed to the model
        """
        return False


class SystemMessage(BaseMessage):
    """Type of message that is a system message."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "system"


class ModelMessageRoleType:
    """Type of ModelMessage role."""

    # 定义模型消息的角色类型常量
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"
    VIEW = "view"


class ModelMessage(BaseModel):
    """Type of message that interaction between dbgpt-server and llm-server."""

    """Similar to openai's message format"""
    # 消息的角色类型
    role: str
    # 消息内容
    content: str
    # 定义一个类属性 `round_index`，类型为 Optional[int]，初始值为 0
    round_index: Optional[int] = 0

    @property
    def pass_to_model(self) -> bool:
        """Whether the message will be passed to the model.

        The view message will not be passed to the model

        Returns:
            bool: Whether the message will be passed to the model
        """
        # 检查当前对象的角色是否在允许传递给模型的角色列表中
        return self.role in [
            ModelMessageRoleType.SYSTEM,
            ModelMessageRoleType.HUMAN,
            ModelMessageRoleType.AI,
        ]

    @staticmethod
    def from_base_messages(messages: List[BaseMessage]) -> List["ModelMessage"]:
        """Covert BaseMessage format to current ModelMessage format.

        Args:
            messages (List[BaseMessage]): The base messages

        Returns:
            List[ModelMessage]: The model messages
        """
        # 将 BaseMessage 格式转换为 ModelMessage 格式
        result = []
        for message in messages:
            content, round_index = message.content, message.round_index
            if isinstance(message, HumanMessage):
                # 如果消息类型为 HumanMessage，创建对应的 ModelMessage 对象并添加到结果列表
                result.append(
                    ModelMessage(
                        role=ModelMessageRoleType.HUMAN,
                        content=content,
                        round_index=round_index,
                    )
                )
            elif isinstance(message, AIMessage):
                # 如果消息类型为 AIMessage，创建对应的 ModelMessage 对象并添加到结果列表
                result.append(
                    ModelMessage(
                        role=ModelMessageRoleType.AI,
                        content=content,
                        round_index=round_index,
                    )
                )
            elif isinstance(message, SystemMessage):
                # 如果消息类型为 SystemMessage，创建对应的 ModelMessage 对象并添加到结果列表
                result.append(
                    ModelMessage(
                        role=ModelMessageRoleType.SYSTEM, content=message.content
                    )
                )
        return result

    @staticmethod
    def from_openai_messages(
        messages: Union[str, List[Dict[str, str]]]
    ) -> List["ModelMessage"]:
        """Openai message format to current ModelMessage format."""
        if isinstance(messages, str):
            # 如果 messages 是字符串，则创建一个包含单个 HumanMessage 的 ModelMessage 列表
            return [ModelMessage(role=ModelMessageRoleType.HUMAN, content=messages)]
        result = []
        for message in messages:
            msg_role = message["role"]
            content = message["content"]
            if msg_role == "system":
                # 如果消息角色为 "system"，创建一个对应的 ModelMessage 对象并添加到结果列表
                result.append(
                    ModelMessage(role=ModelMessageRoleType.SYSTEM, content=content)
                )
            elif msg_role == "user":
                # 如果消息角色为 "user"，创建一个对应的 ModelMessage 对象并添加到结果列表
                result.append(
                    ModelMessage(role=ModelMessageRoleType.HUMAN, content=content)
                )
            elif msg_role == "assistant":
                # 如果消息角色为 "assistant"，创建一个对应的 ModelMessage 对象并添加到结果列表
                result.append(
                    ModelMessage(role=ModelMessageRoleType.AI, content=content)
                )
            else:
                # 如果消息角色未知，抛出 ValueError 异常
                raise ValueError(f"Unknown role: {msg_role}")
        return result

    @staticmethod
    def to_common_messages(
        messages: List["ModelMessage"],
        convert_to_compatible_format: bool = False,
        support_system_role: bool = True,
    ) -> List[Dict[str, str]]:
        """Cover to common message format.

        Convert each ModelMessage in the input list to a standardized format
        based on its role (user, system, assistant). Optionally reorganizes
        the message history to place the last user input at the end.

        Args:
            messages (List["ModelMessage"]): List of ModelMessage objects to convert.
            convert_to_compatible_format (bool): Whether to reorganize messages for compatibility.
            support_system_role (bool): Whether the model supports system role messages.

        Returns:
            List[Dict[str, str]]: List of dictionaries representing converted messages.

        Raises:
            ValueError: If the message role is not supported and support_system_role is False.
        """
        history = []
        # Add history conversation
        for message in messages:
            if message.role == ModelMessageRoleType.HUMAN:
                history.append({"role": "user", "content": message.content})
            elif message.role == ModelMessageRoleType.SYSTEM:
                if not support_system_role:
                    raise ValueError("Current model not support system role")
                history.append({"role": "system", "content": message.content})
            elif message.role == ModelMessageRoleType.AI:
                history.append({"role": "assistant", "content": message.content})
            else:
                pass
        if convert_to_compatible_format:
            # Move the last user's information to the end
            last_user_input_index = None
            for i in range(len(history) - 1, -1, -1):
                if history[i]["role"] == "user":
                    last_user_input_index = i
                    break
            if last_user_input_index:
                last_user_input = history.pop(last_user_input_index)
                history.append(last_user_input)
        return history

    @staticmethod
    def to_dict_list(messages: List["ModelMessage"]) -> List[Dict[str, str]]:
        """Convert ModelMessage objects to a list of dictionaries.

        Args:
            messages (List["ModelMessage"]): List of ModelMessage objects to convert.

        Returns:
            List[Dict[str, str]]: List of dictionaries representing each ModelMessage.

        """
        return list(map(lambda m: model_to_dict(m), messages))

    @staticmethod
    def build_human_message(content: str) -> "ModelMessage":
        """Create a ModelMessage object representing a human message.

        Args:
            content (str): The content of the human message.

        Returns:
            ModelMessage: A ModelMessage object with role set to HUMAN and given content.

        """
        return ModelMessage(role=ModelMessageRoleType.HUMAN, content=content)
    # 获取可打印的消息字符串。
    def get_printable_message(messages: List["ModelMessage"]) -> str:
        """Get the printable message.

        Args:
            messages (List["ModelMessage"]): The model messages

        Returns:
            str: The printable message
        """
        # 初始化空字符串以存储可打印的消息
        str_msg = ""
        # 遍历给定的消息列表
        for message in messages:
            # 根据消息的属性构建当前消息的格式化字符串
            curr_message = (
                f"(Round {message.round_index}) {message.role}: {message.content} "
            )
            # 将当前消息字符串去除右侧空白字符并添加换行符后追加到总消息字符串
            str_msg += curr_message.rstrip() + "\n"

        # 返回最终构建的可打印消息字符串
        return str_msg

    @staticmethod
    def messages_to_string(
        messages: List["ModelMessage"],
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        system_prefix: str = "System",
    ) -> str:
        """Convert messages to str.

        Args:
            messages (List[ModelMessage]): The messages
            human_prefix (str): The human prefix
            ai_prefix (str): The ai prefix
            system_prefix (str): The system prefix

        Returns:
            str: The str messages
        """
        # 调用内部函数 _messages_to_str 将消息列表转换为字符串
        return _messages_to_str(messages, human_prefix, ai_prefix, system_prefix)
_SingleRoundMessage = List[BaseMessage]
_MultiRoundMessageMapper = Callable[[List[_SingleRoundMessage]], List[BaseMessage]]


def _message_to_dict(message: BaseMessage) -> Dict:
    # 将 BaseMessage 对象转换成字典形式
    return message.to_dict()


def _messages_to_dict(messages: List[BaseMessage]) -> List[Dict]:
    # 将列表中的多个 BaseMessage 对象转换成字典形式的列表
    return [_message_to_dict(m) for m in messages]


def _messages_to_str(
    messages: Union[List[BaseMessage], List[ModelMessage]],
    human_prefix: str = "Human",
    ai_prefix: str = "AI",
    system_prefix: str = "System",
) -> str:
    """Convert messages to str.

    Args:
        messages (List[Union[BaseMessage, ModelMessage]]): The messages
        human_prefix (str): The human prefix
        ai_prefix (str): The ai prefix
        system_prefix (str): The system prefix

    Returns:
        str: The str messages
    """
    str_messages = []
    for message in messages:
        role = None
        if isinstance(message, HumanMessage):
            # 判断消息类型，设置角色为人类消息前缀
            role = human_prefix
        elif isinstance(message, AIMessage):
            # 判断消息类型，设置角色为 AI 消息前缀
            role = ai_prefix
        elif isinstance(message, SystemMessage):
            # 判断消息类型，设置角色为系统消息前缀
            role = system_prefix
        elif isinstance(message, ViewMessage):
            # 忽略视图消息
            pass
        elif isinstance(message, ModelMessage):
            # 对于模型消息，使用消息本身的角色
            role = message.role
        else:
            # 抛出异常，表示遇到不支持的消息类型
            raise ValueError(f"Got unsupported message type: {message}")
        if role:
            # 如果角色不为空，则添加到字符串消息列表中
            str_messages.append(f"{role}: {message.content}")
    # 将字符串消息列表连接成一个换行分隔的字符串
    return "\n".join(str_messages)


def _message_from_dict(message: Dict) -> BaseMessage:
    # 根据字典中的类型字段创建对应的 BaseMessage 对象
    _type = message["type"]
    if _type == "human":
        return HumanMessage(**message["data"])
    elif _type == "ai":
        return AIMessage(**message["data"])
    elif _type == "system":
        return SystemMessage(**message["data"])
    elif _type == "view":
        return ViewMessage(**message["data"])
    else:
        # 抛出异常，表示遇到意外的消息类型
        raise ValueError(f"Got unexpected type: {_type}")


def _messages_from_dict(messages: List[Dict]) -> List[BaseMessage]:
    # 将字典形式的消息列表转换成 BaseMessage 对象的列表
    return [_message_from_dict(m) for m in messages]


def parse_model_messages(
    messages: List[ModelMessage],
) -> Tuple[str, List[str], List[List[str]]]:
    """Parse model messages.

    Parse model messages to extract the user prompt, system messages, and a history of
    conversation.

    This function analyzes a list of ModelMessage objects, identifying the role of each
    message (e.g., human, system, ai)
    and categorizes them accordingly. The last message is expected to be from the user
    (human), and it's treated as
    the current user prompt. System messages are extracted separately, and the
    conversation history is compiled into pairs of human and AI messages.

    Args:
        messages (List[ModelMessage]): List of messages from a chat conversation.

    Returns:
        Tuple[str, List[str], List[List[str]]]: A tuple containing user prompt, system
        messages, and conversation history.
    """
    """
    Parse a list of ModelMessage objects into user prompt, system messages, and conversation history.

    Args:
        messages (list): A list of ModelMessage objects representing a conversation.

    Returns:
        tuple: A tuple containing the user prompt, list of system messages, and the
            conversation history.
            The conversation history is a list of message pairs, each containing a
            user message and the corresponding AI response.

    Examples:
        .. code-block:: python

            # Example 1: Single round of conversation
            messages = [
                ModelMessage(role="human", content="Hello"),
                ModelMessage(role="ai", content="Hi there!"),
                ModelMessage(role="human", content="How are you?"),
            ]
            user_prompt, system_messages, history = parse_model_messages(messages)
            # user_prompt: "How are you?"
            # system_messages: []
            # history: [["Hello", "Hi there!"]]

            # Example 2: Conversation with system messages
            messages = [
                ModelMessage(role="system", content="System initializing..."),
                ModelMessage(role="human", content="Is it sunny today?"),
                ModelMessage(role="ai", content="Yes, it's sunny."),
                ModelMessage(role="human", content="Great!"),
            ]
            user_prompt, system_messages, history = parse_model_messages(messages)
            # user_prompt: "Great!"
            # system_messages: ["System initializing..."]
            # history: [["Is it sunny today?", "Yes, it's sunny."]]

            # Example 3: Multiple rounds with system message
            messages = [
                ModelMessage(role="human", content="Hi"),
                ModelMessage(role="ai", content="Hello!"),
                ModelMessage(role="system", content="Error 404"),
                ModelMessage(role="human", content="What's the error?"),
                ModelMessage(role="ai", content="Just a joke."),
                ModelMessage(role="human", content="Funny!"),
            ]
            user_prompt, system_messages, history = parse_model_messages(messages)
            # user_prompt: "Funny!"
            # system_messages: ["Error 404"]
            # history: [["Hi", "Hello!"], ["What's the error?", "Just a joke."]]
    """
    # Initialize empty lists for system messages and history messages
    system_messages: List[str] = []
    history_messages: List[List[str]] = [[]]

    # Iterate through all messages except the last one
    for message in messages[:-1]:
        # Check the role of the message and append content accordingly
        if message.role == "human":
            history_messages[-1].append(message.content)  # Append user message to current history
        elif message.role == "system":
            system_messages.append(message.content)  # Collect system message
        elif message.role == "ai":
            history_messages[-1].append(message.content)  # Append AI response to current history
            history_messages.append([])  # Start a new empty history for the next round

    # Validate the last message must be from the user
    if messages[-1].role != "human":
        raise ValueError("Hi! What do you want to talk about？")

    # Filter out incomplete history pairs and set the user prompt
    history_messages = list(filter(lambda x: len(x) == 2, history_messages))
    user_prompt = messages[-1].content  # Set user prompt as the last message content

    return user_prompt, system_messages, history_messages
class OnceConversation:
    """Once conversation.

    All the information of a conversation, the current single service in memory,
    can expand cache and database support distributed services.
    """

    def __init__(
        self,
        chat_mode: str,
        user_name: Optional[str] = None,
        sys_code: Optional[str] = None,
        summary: Optional[str] = None,
        **kwargs,
    ):
        """Create a new conversation."""
        # 初始化对话对象，设置对话模式、用户名、系统代码和摘要
        self.chat_mode: str = chat_mode
        self.user_name: Optional[str] = user_name
        self.sys_code: Optional[str] = sys_code
        self.summary: Optional[str] = summary

        # 初始化消息列表、开始日期、对话顺序、模型名称、参数类型、参数值、成本、令牌和消息索引
        self.messages: List[BaseMessage] = kwargs.get("messages", [])
        self.start_date: str = kwargs.get("start_date", "")
        self.chat_order: int = int(kwargs.get("chat_order", 0))
        self.model_name: str = kwargs.get("model_name", "")
        self.param_type: str = kwargs.get("param_type", "")
        self.param_value: str = kwargs.get("param_value", "")
        self.cost: int = int(kwargs.get("cost", 0))
        self.tokens: int = int(kwargs.get("tokens", 0))
        self._message_index: int = int(kwargs.get("message_index", 0))

    def _append_message(self, message: BaseMessage) -> None:
        # 添加消息到消息列表中，并更新消息索引和对话顺序
        index = self._message_index
        self._message_index += 1
        message.index = index
        message.round_index = self.chat_order
        message.additional_kwargs["param_type"] = self.param_type
        message.additional_kwargs["param_value"] = self.param_value
        message.additional_kwargs["model_name"] = self.model_name
        self.messages.append(message)

    def start_new_round(self) -> None:
        """Start a new round of conversation.

        Example:
            >>> conversation = OnceConversation("chat_normal")
            >>> # The chat order will be 0, then we start a new round of conversation
            >>> assert conversation.chat_order == 0
            >>> conversation.start_new_round()
            >>> # Now the chat order will be 1
            >>> assert conversation.chat_order == 1
            >>> conversation.add_user_message("hello")
            >>> conversation.add_ai_message("hi")
            >>> conversation.end_current_round()
            >>> # Now the chat order will be 1, then we start a new round of
            >>> # conversation
            >>> conversation.start_new_round()
            >>> # Now the chat order will be 2
            >>> assert conversation.chat_order == 2
            >>> conversation.add_user_message("hello")
            >>> conversation.add_ai_message("hi")
            >>> conversation.end_current_round()
            >>> assert conversation.chat_order == 2
        """
        # 开始一个新的对话轮次，对话顺序加1
        self.chat_order += 1
    def end_current_round(self) -> None:
        """Execute the end of the current round of conversation.

        We do noting here, just for the interface
        """
        # 这个方法用于结束当前对话轮次，但实际上不执行任何操作，只是为了接口的完整性而存在
        pass

    def add_user_message(
        self, message: str, check_duplicate_type: Optional[bool] = False
    ) -> None:
        """Save a user message to the conversation.

        Args:
            message (str): The message content
            check_duplicate_type (bool): Whether to check the duplicate message type

        Raises:
            ValueError: If the message is duplicate and check_duplicate_type is True
        """
        # 将用户消息保存到对话中
        if check_duplicate_type:
            # 检查是否已存在用户消息
            has_message = any(
                isinstance(instance, HumanMessage) for instance in self.messages
            )
            if has_message:
                raise ValueError("Already Have Human message")
        self._append_message(HumanMessage(content=message))

    def add_ai_message(
        self, message: str, update_if_exist: Optional[bool] = False
    ) -> None:
        """Save an AI message to current conversation.

        Args:
            message (str): The message content
            update_if_exist (bool): Whether to update the message if the message type
                is duplicate
        """
        # 将AI消息保存到当前对话中
        if not update_if_exist:
            # 如果不需要更新已存在的消息类型，则直接添加新的AI消息
            self._append_message(AIMessage(content=message))
            return
        # 检查是否已存在AI消息
        has_message = any(isinstance(instance, AIMessage) for instance in self.messages)
        if has_message:
            # 如果已存在AI消息，则更新消息内容
            self._update_ai_message(message)
        else:
            # 如果不存在AI消息，则添加新的AI消息
            self._append_message(AIMessage(content=message))

    def _update_ai_message(self, new_message: str) -> None:
        """Update the all AI message to new message.

        stream out message update

        Args:
            new_message (str): The new message
        """
        # 更新所有AI消息到新的消息内容
        for item in self.messages:
            if item.type == "ai":
                item.content = new_message

    def add_view_message(self, message: str) -> None:
        """Save a view message to current conversation."""
        # 将视图消息保存到当前对话中
        self._append_message(ViewMessage(content=message))

    def add_system_message(self, message: str) -> None:
        """Save a system message to current conversation."""
        # 将系统消息保存到当前对话中
        self._append_message(SystemMessage(content=message))

    def set_start_time(self, datatime: datetime):
        """Set the start time of the conversation."""
        # 设置对话开始时间
        dt_str = datatime.strftime("%Y-%m-%d %H:%M:%S")
        self.start_date = dt_str

    def clear(self) -> None:
        """Remove all messages from the store."""
        # 清空存储中的所有消息
        self.messages.clear()

    def get_latest_user_message(self) -> Optional[HumanMessage]:
        """Get the latest user message."""
        # 获取最新的用户消息
        for message in self.messages[::-1]:
            if isinstance(message, HumanMessage):
                return message
        return None
    def get_system_messages(self) -> List[SystemMessage]:
        """获取系统消息列表。

        Returns:
            List[SystemMessage]: 系统消息列表
        """
        return cast(
            List[SystemMessage],
            list(filter(lambda x: isinstance(x, SystemMessage), self.messages)),
        )



    def _to_dict(self) -> Dict:
        """将对象转换为字典表示。"""
        return _conversation_to_dict(self)



    def from_conversation(self, conversation: OnceConversation) -> None:
        """从存储中加载对话内容。

        Args:
            conversation (OnceConversation): 要加载的对话对象
        """
        self.chat_mode = conversation.chat_mode  # 设置聊天模式
        self.messages = conversation.messages  # 设置消息列表
        self.start_date = conversation.start_date  # 设置对话开始日期
        self.chat_order = conversation.chat_order  # 设置聊天顺序
        if not self.model_name and conversation.model_name:
            self.model_name = conversation.model_name  # 设置模型名称（如果当前未设置且对话中有提供）
        if not self.param_type and conversation.param_type:
            self.param_type = conversation.param_type  # 设置参数类型（如果当前未设置且对话中有提供）
        if not self.param_value and conversation.param_value:
            self.param_value = conversation.param_value  # 设置参数值（如果当前未设置且对话中有提供）
        self.cost = conversation.cost  # 设置对话成本
        self.tokens = conversation.tokens  # 设置对话令牌
        self.user_name = conversation.user_name  # 设置用户名
        self.sys_code = conversation.sys_code  # 设置系统代码
        self._message_index = conversation._message_index  # 设置消息索引



    def get_messages_by_round(self, round_index: int) -> List[BaseMessage]:
        """按照轮次索引获取消息列表。

        Args:
            round_index (int): 轮次索引

        Returns:
            List[BaseMessage]: 消息列表
        """
        return list(filter(lambda x: x.round_index == round_index, self.messages))



    def get_latest_round(self) -> List[BaseMessage]:
        """获取最新一轮的消息列表。

        Returns:
            List[BaseMessage]: 消息列表
        """
        return self.get_messages_by_round(self.chat_order)
    def get_messages_with_round(self, round_count: int) -> List[BaseMessage]:
        """Get the messages with round count.

        If the round count is 1, the history messages will not be included.

        Example:
            .. code-block:: python

                conversation = OnceConversation()
                conversation.start_new_round()
                conversation.add_user_message("hello, this is the first round")
                conversation.add_ai_message("hi")
                conversation.end_current_round()
                conversation.start_new_round()
                conversation.add_user_message("hello, this is the second round")
                conversation.add_ai_message("hi")
                conversation.end_current_round()
                conversation.start_new_round()
                conversation.add_user_message("hello, this is the third round")
                conversation.add_ai_message("hi")
                conversation.end_current_round()

                assert len(conversation.get_messages_with_round(1)) == 2
                assert (
                    conversation.get_messages_with_round(1)[0].content
                    == "hello, this is the third round"
                )
                assert conversation.get_messages_with_round(1)[1].content == "hi"

                assert len(conversation.get_messages_with_round(2)) == 4
                assert (
                    conversation.get_messages_with_round(2)[0].content
                    == "hello, this is the second round"
                )
                assert conversation.get_messages_with_round(2)[1].content == "hi"

        Args:
            round_count (int): The round count

        Returns:
            List[BaseMessage]: The messages
        """
        # 获取最新回合的索引
        latest_round_index = self.chat_order
        # 计算起始回合的索引，确保不小于1
        start_round_index = max(1, latest_round_index - round_count + 1)
        # 初始化消息列表
        messages = []
        # 遍历需要获取消息的回合索引范围
        for round_index in range(start_round_index, latest_round_index + 1):
            # 获取特定回合索引的消息列表，并添加到结果消息列表中
            messages.extend(self.get_messages_by_round(round_index))
        # 返回所有获取到的消息列表
        return messages
    def get_model_messages(self) -> List[ModelMessage]:
        """获取模型消息。

        模型消息包括人类消息、AI 消息和系统消息。
        模型消息可能包括历史消息，消息的顺序与对话中消息的顺序相同，最后一条消息是最新的消息。

        如果您想根据自己的逻辑处理消息，可以重写此方法。

        示例:
            如果您不需要历史消息，可以像这样重写此方法:
            .. code-block:: python

                def get_model_messages(self) -> List[ModelMessage]:
                    messages = []
                    for message in self.get_latest_round():
                        if message.pass_to_model:
                            messages.append(
                                ModelMessage(role=message.type, content=message.content)
                            )
                    return messages

            如果您想添加一轮历史消息，可以像这样重写此方法:
            .. code-block:: python

                def get_model_messages(self) -> List[ModelMessage]:
                    messages = []
                    latest_round_index = self.chat_order
                    round_count = 1
                    start_round_index = max(1, latest_round_index - round_count + 1)
                    for round_index in range(start_round_index, latest_round_index + 1):
                        for message in self.get_messages_by_round(round_index):
                            if message.pass_to_model:
                                messages.append(
                                    ModelMessage(
                                        role=message.type, content=message.content
                                    )
                                )
                    return messages

        返回:
            List[ModelMessage]: 模型消息
        """
        messages = []
        for message in self.messages:
            if message.pass_to_model:
                messages.append(
                    ModelMessage(
                        role=message.type,
                        content=message.content,
                        round_index=message.round_index,
                    )
                )
        return messages

    def get_history_message(
        self, include_system_message: bool = False
    ) -> List[BaseMessage]:
        """获取历史消息。

        不包括系统消息。

        Args:
            include_system_message (bool): 是否包括系统消息

        Returns:
            List[BaseMessage]: 历史消息列表
        """
        messages = []  # 初始化空列表，用于存储符合条件的消息
        for message in self.messages:  # 遍历实例对象中的消息列表
            if (
                message.pass_to_model  # 如果消息标记为需要传递给模型
                and include_system_message  # 并且参数中指定要包括系统消息
                or message.type != "system"  # 或者消息类型不是系统消息
            ):
                messages.append(message)  # 将符合条件的消息添加到列表中
        return messages  # 返回符合条件的消息列表
class ConversationIdentifier(ResourceIdentifier):
    """Conversation identifier."""

    def __init__(self, conv_uid: str, identifier_type: str = "conversation"):
        """Create a conversation identifier.

        Args:
            conv_uid (str): The conversation uid
            identifier_type (str): The identifier type, default is "conversation"
        """
        self.conv_uid = conv_uid  # 设置会话的唯一标识符
        self.identifier_type = identifier_type  # 设置标识符类型，默认为"conversation"

    @property
    def str_identifier(self) -> str:
        """Return the str identifier."""
        return f"{self.identifier_type}:{
    def to_message(self) -> BaseMessage:
        """Convert to message object.

        Returns:
            BaseMessage: The message object

        Raises:
            ValueError: If the message type is not supported
        """
        # 调用内部函数将对象转换为消息对象，并返回结果
        return _message_from_dict(self.message_detail)

    def merge(self, other: "StorageItem") -> None:
        """Merge the other message to self.

        Args:
            other (StorageItem): The other message
        """
        # 检查参数是否为 MessageStorageItem 类型，如果不是则抛出 ValueError 异常
        if not isinstance(other, MessageStorageItem):
            raise ValueError(f"Can not merge {other} to {self}")
        # 将参数对象的消息细节复制到当前对象的消息细节中
        self.message_detail = other.message_detail
    @property
    def identifier(self) -> ConversationIdentifier:
        """Return the identifier."""
        return self._id



        """Return the identifier."""
        # 返回当前对话的标识符



    def to_dict(self) -> Dict:
        """Convert to dict."""
        # 将对象转换为字典表示
        dict_data = self._to_dict()
        messages: Dict = dict_data.pop("messages")
        message_ids = []
        index = 0
        for message in messages:
            if "index" in message:
                message_idx = message["index"]
            else:
                message_idx = index
                index += 1
            message_ids.append(
                MessageIdentifier(self.conv_uid, message_idx).str_identifier
            )
        # Replace message with message ids
        dict_data["conv_uid"] = self.conv_uid
        dict_data["message_ids"] = message_ids
        dict_data["save_message_independent"] = self.save_message_independent
        return dict_data



        """Convert to dict."""
        # 将对象转换为字典表示



        dict_data = self._to_dict()
        messages: Dict = dict_data.pop("messages")



        messages: Dict = dict_data.pop("messages")
        # 从字典数据中取出消息列表



        message_ids = []
        index = 0
        for message in messages:



            if "index" in message:
                message_idx = message["index"]
            else:
                message_idx = index
                index += 1



            message_ids.append(
                MessageIdentifier(self.conv_uid, message_idx).str_identifier
            )



        # Replace message with message ids
        dict_data["conv_uid"] = self.conv_uid
        dict_data["message_ids"] = message_ids
        dict_data["save_message_independent"] = self.save_message_independent



        # 将消息列表替换为消息ID列表，更新字典数据



        return dict_data



        # 返回更新后的字典数据



    def merge(self, other: "StorageItem") -> None:
        """Merge the other conversation to self.

        Args:
            other (StorageItem): The other conversation
        """
        # 合并另一个对话到当前对话

        if not isinstance(other, StorageConversation):
            raise ValueError(f"Can not merge {other} to {self}")
        # 检查是否为相同类型的对话对象，否则引发错误

        self.from_conversation(other)



        if not isinstance(other, StorageConversation):
            raise ValueError(f"Can not merge {other} to {self}")



            raise ValueError(f"Can not merge {other} to {self}")
        # 如果不是相同类型的对话对象，则引发值错误异常



        self.from_conversation(other)



        # 调用方法将另一个对话合并到当前对话



    def __init__(
        self,
        conv_uid: str,
        chat_mode: str = "chat_normal",
        user_name: Optional[str] = None,
        sys_code: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        summary: Optional[str] = None,
        save_message_independent: bool = True,
        conv_storage: Optional[StorageInterface] = None,
        message_storage: Optional[StorageInterface] = None,
        load_message: bool = True,
        **kwargs,
    ):
        """Create a conversation."""
        # 初始化对话对象的属性和状态

        super().__init__(chat_mode, user_name, sys_code, summary, **kwargs)
        # 调用父类的初始化方法，设置基本属性

        self.conv_uid = conv_uid
        # 设置对话的唯一标识符

        self._message_ids = message_ids
        # 设置消息ID列表

        # Record the message index last time saved to the storage,
        # next time save messages which index is _has_stored_message_index + 1
        self._has_stored_message_index = (
            len(kwargs["messages"]) - 1 if "messages" in kwargs else -1
        )
        # 记录上次保存到存储的消息索引，下次保存从_has_stored_message_index + 1开始

        # Whether to load the message from the storage
        self._load_message = load_message
        # 是否从存储加载消息的标志

        self.save_message_independent = save_message_independent
        # 是否独立保存消息的标志

        self._id = ConversationIdentifier(conv_uid)
        # 创建对话标识符对象

        if conv_storage is None:
            conv_storage = InMemoryStorage()
        # 如果对话存储对象为None，则使用内存存储

        if message_storage is None:
            message_storage = InMemoryStorage()
        # 如果消息存储对象为None，则使用内存存储

        self.conv_storage = conv_storage
        # 设置对话存储对象

        self.message_storage = message_storage
        # 设置消息存储对象

        # Load from storage
        self.load_from_storage(self.conv_storage, self.message_storage)
        # 从存储加载数据到对话对象



        """Create a conversation."""
        # 创建一个新的对话对象的实例



        super().__init__(chat_mode, user_name, sys_code, summary, **kwargs)



        # 调用父类的初始化方法，设置基本属性



        self.conv_uid = conv_uid



        # 设置对话的唯一标识符



        self._message_ids = message_ids



        # 设置消息ID列表



        self._has_stored_message_index = (
            len(kwargs["messages"]) - 1 if "messages" in kwargs else -1
        )



        # 记录上次保存到存储的消息索引，下次保存从_has_stored_message_index + 1开始



        self._load_message = load_message



        # 是否从存储加载消息的标志



        self.save_message_independent = save_message_independent



        # 是否独立保存消息的标志



        self._id = ConversationIdentifier(conv_uid)



        # 创建对话标识符对象



        if conv_storage is None:
            conv_storage = InMemoryStorage()



        # 如果对话存储对象为None，则使用内存存储



        if message_storage is None:
            message_storage = InMemoryStorage()



        # 如果消息存储对象为None，则使用内存存储



        self.conv_storage = conv_storage



        # 设置对话存储对象



        self.message_storage = message_storage



        # 设置消息存储对象



        self.load_from_storage(self.conv_storage, self.message_storage)



        # 从存储加载数据到对话对象
    def message_ids(self) -> List[str]:
        """Return the message ids.

        Returns:
            List[str]: The message ids
        """
        return self._message_ids if self._message_ids else []



    def end_current_round(self) -> None:
        """End the current round of conversation.

        Save the conversation to the storage after a round of conversation
        """
        # 调用 save_to_storage 方法保存当前会话到存储中
        self.save_to_storage()



    def _get_message_items(self) -> List[MessageStorageItem]:
        """Generate MessageStorageItem objects for each message in self.messages.

        Returns:
            List[MessageStorageItem]: List of MessageStorageItem objects
        """
        return [
            MessageStorageItem(self.conv_uid, message.index, message.to_dict())
            for message in self.messages
        ]



    def save_to_storage(self) -> None:
        """Save the conversation to the storage."""
        # 获取所有消息的 MessageStorageItem 对象列表
        message_list = self._get_message_items()
        # 更新 _message_ids 列表为所有消息的标识符
        self._message_ids = [
            message.identifier.str_identifier for message in message_list
        ]
        # 筛选出尚未保存的消息
        messages_to_save = message_list[self._has_stored_message_index + 1 :]
        # 更新 _has_stored_message_index 到最后一个保存的消息索引
        self._has_stored_message_index = len(message_list) - 1
        if self.save_message_independent:
            # 若设置为独立保存消息，则调用 message_storage.save_list 方法保存消息列表
            self.message_storage.save_list(messages_to_save)
        # 保存当前会话到存储中
        self.conv_storage.save_or_update(self)



    def load_from_storage(
        self, conv_storage: StorageInterface, message_storage: StorageInterface
    ) -> None:
        """Load conversation and messages from storage.

        Args:
            conv_storage (StorageInterface): Storage interface for conversation
            message_storage (StorageInterface): Storage interface for messages
        """
        # TODO: Add implementation to load conversation and messages from storage
        pass
    ) -> None:
        """Load the conversation from the storage.

        Warning: This will overwrite the current conversation.

        Args:
            conv_storage (StorageInterface): The storage interface
            message_storage (StorageInterface): The storage interface
        """
        # 从存储中加载对话
        conversation: Optional[StorageConversation] = conv_storage.load(
            self._id, StorageConversation
        )
        if conversation is None:
            return
        message_ids = conversation._message_ids or []

        if self._load_message:
            # 加载消息
            message_list = message_storage.load_list(
                [
                    MessageIdentifier.from_str_identifier(message_id)
                    for message_id in message_ids
                ],
                MessageStorageItem,
            )
            # 转换消息列表中的每条消息为消息对象
            messages = [message.to_message() for message in message_list]
        else:
            messages = []
        # 如果没有加载到消息，则使用对话对象中的消息列表
        real_messages = messages or conversation.messages
        conversation.messages = real_messages
        # 此索引用于保存消息到存储中（尚未保存）
        # 新消息附加到消息列表后，因此索引为 len(messages)
        conversation._message_index = len(real_messages)
        # 计算对话中消息的排序顺序
        conversation.chat_order = (
            max(m.round_index for m in real_messages) if real_messages else 0
        )
        # 添加额外的关键参数到对话对象中
        self._append_additional_kwargs(conversation, real_messages)
        self._message_ids = message_ids
        # 保存消息索引的标志位，初始化为实际消息数量减一
        self._has_stored_message_index = len(real_messages) - 1
        # 从对话对象中获取是否独立保存消息的设置
        self.save_message_independent = conversation.save_message_independent
        # 将当前加载的对话应用到当前对象中
        self.from_conversation(conversation)

    def _append_additional_kwargs(
        self, conversation: StorageConversation, messages: List[BaseMessage]
    ) -> None:
        """Parse the additional kwargs and append to the conversation.

        Args:
            conversation (StorageConversation): The conversation
            messages (List[BaseMessage]): The messages
        """
        param_type = ""
        param_value = ""
        # 倒序遍历消息列表，找到第一个包含附加参数的消息
        for message in messages[::-1]:
            if message.additional_kwargs:
                # 获取消息中的参数类型和参数值
                param_type = message.additional_kwargs.get("param_type", "")
                param_value = message.additional_kwargs.get("param_value", "")
                break
        # 如果对话对象没有设置参数类型，则设置为找到的参数类型
        if not conversation.param_type:
            conversation.param_type = param_type
        # 如果对话对象没有设置参数值，则设置为找到的参数值
        if not conversation.param_value:
            conversation.param_value = param_value
    def delete(self) -> None:
        """Delete all the messages and conversation."""
        # 获取当前会话的所有消息项
        message_list = self._get_message_items()
        # 从消息项列表中提取消息的标识符
        message_ids = [message.identifier for message in message_list]
        # 使用消息存储对象删除消息列表中的所有消息
        self.message_storage.delete_list(message_ids)
        # 使用会话存储对象删除当前会话
        self.conv_storage.delete(self.identifier)
        # 用空会话对象覆盖当前会话
        self.from_conversation(
            StorageConversation(
                self.conv_uid,
                save_message_independent=self.save_message_independent,
                conv_storage=self.conv_storage,
                message_storage=self.message_storage,
            )
        )
def _conversation_to_dict(once: OnceConversation) -> Dict:
    # 初始化空字符串，用于存储格式化后的开始日期字符串
    start_str: str = ""
    # 检查是否存在开始日期属性且非空
    if hasattr(once, "start_date") and once.start_date:
        # 如果开始日期是 datetime 类型，则格式化为指定格式的字符串
        if isinstance(once.start_date, datetime):
            start_str = once.start_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # 否则直接赋值开始日期
            start_str = once.start_date

    # 构建并返回对话对象的字典表示
    return {
        "chat_mode": once.chat_mode,          # 对话模式
        "model_name": once.model_name,        # 模型名称
        "chat_order": once.chat_order,        # 对话顺序
        "start_date": start_str,              # 开始日期字符串
        "cost": once.cost if once.cost else 0,    # 费用，若无则默认为0
        "tokens": once.tokens if once.tokens else 0,    # 令牌数，若无则默认为0
        "messages": _messages_to_dict(once.messages),   # 将消息列表转换为字典形式
        "param_type": once.param_type,        # 参数类型
        "param_value": once.param_value,      # 参数值
        "user_name": once.user_name,          # 用户名
        "sys_code": once.sys_code,            # 系统代码
        "summary": once.summary if once.summary else "",   # 摘要，若无则为空字符串
    }


def _conversations_to_dict(conversations: List[OnceConversation]) -> List[dict]:
    # 将每个 OnceConversation 对象转换为字典形式，并存入列表中
    return [_conversation_to_dict(m) for m in conversations]


def _conversation_from_dict(once: dict) -> OnceConversation:
    # 根据字典创建 OnceConversation 对象
    conversation = OnceConversation(
        once.get("chat_mode", ""), once.get("user_name"), once.get("sys_code")
    )
    conversation.cost = once.get("cost", 0)    # 设置费用，若字典中无则默认为0
    conversation.chat_mode = once.get("chat_mode", "chat_normal")   # 设置对话模式，默认为 'chat_normal'
    conversation.tokens = once.get("tokens", 0)    # 设置令牌数，若字典中无则默认为0
    conversation.start_date = once.get("start_date", "")    # 设置开始日期，若字典中无则为空字符串
    conversation.chat_order = int(once.get("chat_order", 0))  # 设置对话顺序，若字典中无则默认为0
    conversation.param_type = once.get("param_type", "")    # 设置参数类型，若字典中无则为空字符串
    conversation.param_value = once.get("param_value", "")  # 设置参数值，若字典中无则为空字符串
    conversation.model_name = once.get("model_name", "proxyllm")   # 设置模型名称，默认为 'proxyllm'
    print(once.get("messages"))    # 打印消息，用于调试
    conversation.messages = _messages_from_dict(once.get("messages", []))   # 将字典中的消息转换为消息对象列表
    return conversation


def _split_messages_by_round(messages: List[BaseMessage]) -> List[List[BaseMessage]]:
    """按照轮次索引将消息分组。

    Args:
        messages (List[BaseMessage]): 消息列表。

    Returns:
        List[List[BaseMessage]]: 按照轮次索引分组后的消息列表。
    """
    messages_by_round: List[List[BaseMessage]] = []    # 初始化存储按轮次分组的消息列表
    last_round_index = 0    # 初始化上一个轮次索引为0
    for message in messages:
        if not message.round_index:
            # 如果消息的轮次索引未设置，则抛出数值错误异常
            raise ValueError("Message round_index is not set")
        if message.round_index > last_round_index:
            # 如果消息的轮次索引大于上一个轮次索引，则更新上一个轮次索引并添加新的空列表
            last_round_index = message.round_index
            messages_by_round.append([])
        # 将消息添加到当前轮次的列表中
        messages_by_round[-1].append(message)
    return messages_by_round


def _append_view_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """向消息列表中添加视图消息。

    仅用于在 DB-GPT-Web 中显示。
    如果已经存在视图消息，则不执行任何操作。

    Args:
        messages (List[BaseMessage]): 消息列表。

    Returns:
        List[BaseMessage]: 添加了视图消息的消息列表。
    """
    messages_by_round = _split_messages_by_round(messages)   # 将消息按照轮次索引分组
    # 遍历每一轮消息列表
    for current_round in messages_by_round:
        ai_message = None  # 初始化 AI 消息为 None
        view_message = None  # 初始化视图消息为 None
        
        # 遍历当前轮的消息列表
        for message in current_round:
            if message.type == "ai":
                ai_message = message  # 如果消息类型为 "ai"，将其赋值给 ai_message
            elif message.type == "view":
                view_message = message  # 如果消息类型为 "view"，将其赋值给 view_message
        
        # 如果已经有视图消息存在，则跳过当前轮次的处理
        if view_message:
            continue
        
        # 如果存在 AI 消息，则创建一个对应的视图消息
        if ai_message:
            view_message = ViewMessage(
                content=ai_message.content,  # 使用 AI 消息的内容
                index=ai_message.index,  # 使用 AI 消息的索引
                round_index=ai_message.round_index,  # 使用 AI 消息的轮次索引
                additional_kwargs=(  # 复制 AI 消息的额外关键字参数（如果存在）
                    ai_message.additional_kwargs.copy() if ai_message.additional_kwargs else {}
                ),
            )
            current_round.append(view_message)  # 将创建的视图消息添加到当前轮次的消息列表中
    
    # 返回所有轮次消息列表中的所有消息的总和列表
    return sum(messages_by_round, [])
```