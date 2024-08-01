# `.\DB-GPT-src\dbgpt\core\interface\operators\message_operator.py`

```py
"""The message operator."""
# 导入所需的模块和库
import logging  # 导入日志记录模块
import uuid  # 导入生成唯一标识符的模块
from abc import ABC  # 导入抽象基类 ABC
from typing import Any, Callable, Dict, List, Optional, Union, cast  # 导入类型提示相关的模块

# 导入 dbgpt.core 下的多个模块和类
from dbgpt.core import (
    InMemoryStorage,
    LLMClient,
    MessageStorageItem,
    ModelMessage,
    ModelMessageRoleType,
    ModelRequest,
    ModelRequestContext,
    StorageConversation,
    StorageInterface,
)
# 导入 dbgpt.core.awel 中的 BaseOperator 和 MapOperator
from dbgpt.core.awel import BaseOperator, MapOperator
# 导入 dbgpt.core.awel.flow 中的 IOField, OperatorCategory, Parameter, ViewMetadata
from dbgpt.core.awel.flow import IOField, OperatorCategory, Parameter, ViewMetadata
# 导入 dbgpt.core.interface.message 中的多个类和函数
from dbgpt.core.interface.message import (
    BaseMessage,
    _messages_to_str,
    _MultiRoundMessageMapper,
    _split_messages_by_round,
)
# 导入 dbgpt.util.i18n_utils 中的 _ 函数
from dbgpt.util.i18n_utils import _

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


class BaseConversationOperator(BaseOperator, ABC):
    """Base class for conversation operators."""

    # 定义用于共享数据的键名常量
    SHARE_DATA_KEY_STORAGE_CONVERSATION = "share_data_key_storage_conversation"
    SHARE_DATA_KEY_MODEL_REQUEST = "share_data_key_model_request"
    SHARE_DATA_KEY_MODEL_REQUEST_CONTEXT = "share_data_key_model_request_context"

    # 类属性，指示是否检查存储的标志
    _check_storage: bool = True

    def __init__(
        self,
        storage: Optional[StorageInterface[StorageConversation, Any]] = None,
        message_storage: Optional[StorageInterface[MessageStorageItem, Any]] = None,
        check_storage: bool = True,
        **kwargs,
    ):
        """Create a new BaseConversationOperator."""
        # 初始化方法，设置属性
        self._check_storage = check_storage  # 设置检查存储的标志
        self._storage = storage  # 设置存储对象
        self._message_storage = message_storage  # 设置消息存储对象

    @property
    def storage(self) -> Optional[StorageInterface[StorageConversation, Any]]:
        """Return the LLM client."""
        # 返回存储对象的属性
        if not self._storage:  # 如果存储对象未设置
            if self._check_storage:  # 如果设置了检查存储标志
                raise ValueError("Storage is not set")  # 抛出数值错误异常
            return None  # 返回 None
        return self._storage  # 返回存储对象

    @property
    def message_storage(self) -> Optional[StorageInterface[MessageStorageItem, Any]]:
        """Return the LLM client."""
        # 返回消息存储对象的属性
        if not self._message_storage:  # 如果消息存储对象未设置
            if self._check_storage:  # 如果设置了检查存储标志
                raise ValueError("Message storage is not set")  # 抛出数值错误异常
            return None  # 返回 None
        return self._message_storage  # 返回消息存储对象

    async def get_storage_conversation(self) -> Optional[StorageConversation]:
        """Get the storage conversation from share data.

        Returns:
            StorageConversation: The storage conversation.
        """
        # 异步方法，从共享数据中获取存储对话
        storage_conv: StorageConversation = (
            await self.current_dag_context.get_from_share_data(
                self.SHARE_DATA_KEY_STORAGE_CONVERSATION
            )
        )
        if not storage_conv:  # 如果未获取到存储对话对象
            if self._check_storage:  # 如果设置了检查存储标志
                raise ValueError("Storage conversation is not set")  # 抛出数值错误异常
            return None  # 返回 None
        return storage_conv  # 返回存储对话对象
    def check_messages(self, messages: List[ModelMessage]) -> None:
        """检查消息列表是否符合要求。

        Args:
            messages (List[ModelMessage]): 消息列表。

        Raises:
            ValueError: 如果消息列表为空。
            ValueError: 如果消息角色不在支持的角色列表中。
        """
        # 检查消息列表是否为空，如果是空列表则抛出异常
        if not messages:
            raise ValueError("Input messages is empty")
        
        # 遍历消息列表，检查每个消息的角色是否在支持的角色列表中
        for message in messages:
            if message.role not in [
                ModelMessageRoleType.HUMAN,
                ModelMessageRoleType.SYSTEM,
            ]:
                # 如果消息角色不在支持的角色列表中，抛出异常
                raise ValueError(f"Message role {message.role} is not supported")
# 使用 Union 定义了 ChatHistoryLoadType 类型，可以是 ModelRequest、ModelRequestContext 或 Dict[str, Any] 中的一种
ChatHistoryLoadType = Union[ModelRequest, ModelRequestContext, Dict[str, Any]]


class PreChatHistoryLoadOperator(
    BaseConversationOperator, MapOperator[ChatHistoryLoadType, List[BaseMessage]]
):
    """The operator to prepare the storage conversation.

    In DB-GPT, conversation record and the messages in the conversation are stored in
    the storage,
    and they can store in different storage(for high performance).

    This operator just load the conversation and messages from storage.
    """

    # 定义了视图元数据，用于描述这个操作符的信息，包括名称、类别、描述等
    metadata = ViewMetadata(
        label=_("Chat History Load Operator"),
        name="chat_history_load_operator",
        category=OperatorCategory.CONVERSION,
        description=_("The operator to load chat history from storage."),
        parameters=[
            # 参数1：Conversation Storage，用于存储对话项（不包括消息项）
            Parameter.build_from(
                label=_("Conversation Storage"),
                name="storage",
                type=StorageInterface,
                optional=True,
                default=None,
                description=_(
                    "The conversation storage, store the conversation items("
                    "Not include message items). If None, we will use InMemoryStorage."
                ),
            ),
            # 参数2：Message Storage，用于存储一次对话的消息
            Parameter.build_from(
                label=_("Message Storage"),
                name="message_storage",
                type=StorageInterface,
                optional=True,
                default=None,
                description=_(
                    "The message storage, store the messages of one "
                    "conversation. If None, we will use InMemoryStorage."
                ),
            ),
        ],
        inputs=[
            # 输入字段定义：Model Request，模型请求的类型
            IOField.build_from(
                label=_("Model Request"),
                name="input_value",
                type=ModelRequest,
                description=_("The model request."),
            )
        ],
        outputs=[
            # 输出字段定义：Stored Messages，存储在存储中的消息列表
            IOField.build_from(
                label=_("Stored Messages"),
                name="output_value",
                type=BaseMessage,
                description=_("The messages stored in the storage."),
                is_list=True,
            )
        ],
    )

    def __init__(
        self,
        # 初始化方法，接受参数 storage 和 message_storage，分别为对话存储和消息存储的接口类型
        storage: Optional[StorageInterface[StorageConversation, Any]] = None,
        message_storage: Optional[StorageInterface[MessageStorageItem, Any]] = None,
        include_system_message: bool = False,
        use_in_memory_storage_if_not_set: bool = True,
        **kwargs,
    ):
        """
        Create a new PreChatHistoryLoadOperator.
        """
        # 如果存储对象未设置且允许使用内存存储，则使用内存存储
        if not storage and use_in_memory_storage_if_not_set:
            logger.info(
                "Storage is not set, use the InMemoryStorage as the conversation "
                "storage."
            )
            storage = InMemoryStorage()
        # 如果消息存储对象未设置且允许使用内存存储，则使用内存存储
        if not message_storage and use_in_memory_storage_if_not_set:
            logger.info(
                "Message storage is not set, use the InMemoryStorage as the message "
            )
            message_storage = InMemoryStorage()
        # 调用父类初始化方法，设置存储对象和消息存储对象
        super().__init__(storage=storage, message_storage=message_storage)
        # 调用 MapOperator 的初始化方法，传入额外参数
        MapOperator.__init__(self, **kwargs)
        # 设置是否包含系统消息的标志位
        self._include_system_message = include_system_message

    async def map(self, input_value: ChatHistoryLoadType) -> List[BaseMessage]:
        """
        Map the input value to a ModelRequest.

        Args:
            input_value (ChatHistoryLoadType): The input value.

        Returns:
            List[BaseMessage]: The messages stored in the storage.
        """
        # 如果输入值为空，则抛出数值错误异常
        if not input_value:
            raise ValueError("Model request context can't be None")
        # 如果输入值为字典，则将其转换为 ModelRequestContext 对象
        if isinstance(input_value, dict):
            input_value = ModelRequestContext(**input_value)
        # 如果输入值为 ModelRequest 对象且其上下文为空，则抛出数值错误异常
        elif isinstance(input_value, ModelRequest):
            if not input_value.context:
                raise ValueError("Model request context can't be None")
            input_value = input_value.context
        # 将输入值强制转换为 ModelRequestContext 类型
        input_value = cast(ModelRequestContext, input_value)
        # 如果会话 UID 不存在，则生成一个新的随机 UUID
        if not input_value.conv_uid:
            input_value.conv_uid = str(uuid.uuid4())
        # 如果额外参数为空，则设置为空字典
        if not input_value.extra:
            input_value.extra = {}

        # 获取聊天模式
        chat_mode = input_value.chat_mode

        # 创建存储会话对象，从存储中加载对话历史，因此需要异步执行
        storage_conv: StorageConversation = await self.blocking_func_to_async(
            StorageConversation,
            conv_uid=input_value.conv_uid,
            chat_mode=chat_mode,
            user_name=input_value.user_name,
            sys_code=input_value.sys_code,
            conv_storage=self.storage,
            message_storage=self.message_storage,
            param_type="",
            param_value=input_value.chat_param,
        )

        # 将存储的会话对象保存以共享数据，供子操作使用
        await self.current_dag_context.save_to_share_data(
            self.SHARE_DATA_KEY_STORAGE_CONVERSATION, storage_conv
        )
        # 将输入的请求上下文保存以共享数据
        await self.current_dag_context.save_to_share_data(
            self.SHARE_DATA_KEY_MODEL_REQUEST_CONTEXT, input_value
        )
        # 从存储中获取历史消息
        history_messages: List[BaseMessage] = storage_conv.get_history_message(
            include_system_message=self._include_system_message
        )
        return history_messages
class ConversationMapperOperator(
    BaseConversationOperator, MapOperator[List[BaseMessage], List[BaseMessage]]
):
    """The base conversation mapper operator."""

    def __init__(
        self, message_mapper: Optional[_MultiRoundMessageMapper] = None, **kwargs
    ):
        """
        Create a new ConversationMapperOperator.

        Args:
            message_mapper: Optional mapper for multi-round messages.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        # 调用父类 MapOperator 的初始化方法，传入额外的参数
        MapOperator.__init__(self, **kwargs)
        self._message_mapper = message_mapper

    async def map(self, input_value: List[BaseMessage]) -> List[BaseMessage]:
        """Map the input value to a ModelRequest."""
        # 调用 map_messages 方法进行消息映射
        return await self.map_messages(input_value)

    async def map_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Map multi round messages to a list of BaseMessage."""
        # 将消息按回合分组
        messages_by_round: List[List[BaseMessage]] = _split_messages_by_round(messages)
        # 确定要使用的消息映射器，默认为 map_multi_round_messages 方法
        message_mapper = self._message_mapper or self.map_multi_round_messages
        # 调用消息映射器进行映射处理
        return message_mapper(messages_by_round)

    def map_multi_round_messages(
        self, messages_by_round: List[List[BaseMessage]]
    ) -> List[BaseMessage]:
        """Map multi-round messages to a list of BaseMessage."""
        # 实际的消息映射处理方法，由子类实现
        # 在此处应该有具体的映射逻辑，但未在提供的代码中给出
    ) -> List[BaseMessage]:
        """Map multi round messages to a list of BaseMessage.

        By default, just merge all multi round messages to a list of BaseMessage
        according origin order.
        And you can overwrite this method to implement your own logic.

        Examples:
            Merge multi round messages to a list of BaseMessage according origin order.

            >>> from dbgpt.core.interface.message import (
            ...     AIMessage,
            ...     HumanMessage,
            ...     SystemMessage,
            ... )
            >>> messages_by_round = [
            ...     [
            ...         HumanMessage(content="Hi", round_index=1),
            ...         AIMessage(content="Hello!", round_index=1),
            ...     ],
            ...     [
            ...         HumanMessage(content="What's the error?", round_index=2),
            ...         AIMessage(content="Just a joke.", round_index=2),
            ...     ],
            ... ]
            >>> operator = ConversationMapperOperator()
            >>> messages = operator.map_multi_round_messages(messages_by_round)
            >>> assert messages == [
            ...     HumanMessage(content="Hi", round_index=1),
            ...     AIMessage(content="Hello!", round_index=1),
            ...     HumanMessage(content="What's the error?", round_index=2),
            ...     AIMessage(content="Just a joke.", round_index=2),
            ... ]

            Map multi round messages to a list of BaseMessage just keep the last one
            round.

            >>> class MyMapper(ConversationMapperOperator):
            ...     def __init__(self, **kwargs):
            ...         super().__init__(**kwargs)
            ...
            ...     def map_multi_round_messages(
            ...         self, messages_by_round: List[List[BaseMessage]]
            ...     ) -> List[BaseMessage]:
            ...         return messages_by_round[-1]
            ...
            >>> operator = MyMapper()
            >>> messages = operator.map_multi_round_messages(messages_by_round)
            >>> assert messages == [
            ...     HumanMessage(content="What's the error?", round_index=2),
            ...     AIMessage(content="Just a joke.", round_index=2),
            ... ]

        Args:
            messages_by_round (List[List[BaseMessage]]):
                The messages grouped by round.
        """
        # Just merge and return
        return _merge_multi_round_messages(messages_by_round)
# 定义一个 BufferedConversationMapperOperator 类，继承自 ConversationMapperOperator 类。
class BufferedConversationMapperOperator(ConversationMapperOperator):
    """
    Buffered conversation mapper operator.

    The buffered conversation mapper operator which can be configured to keep
    a certain number of starting and/or ending rounds of a conversation.

    Args:
        keep_start_rounds (Optional[int]): Number of initial rounds to keep.
        keep_end_rounds (Optional[int]): Number of final rounds to keep.
    """

    def __init__(
        self,
        keep_start_rounds: Optional[int] = None,
        keep_end_rounds: Optional[int] = None,
        message_mapper: Optional[_MultiRoundMessageMapper] = None,
        **kwargs,
    ):
        """Create a new BufferedConversationMapperOperator."""
        # 创建一个新的 BufferedConversationMapperOperator 类

        # 验证输入参数
        if keep_start_rounds is None:
            keep_start_rounds = 0
        if keep_end_rounds is None:
            keep_end_rounds = 0
        if keep_start_rounds < 0:
            raise ValueError("keep_start_rounds must be non-negative")
        if keep_end_rounds < 0:
            raise ValueError("keep_end_rounds must be non-negative")

        self._keep_start_rounds = keep_start_rounds
        self._keep_end_rounds = keep_end_rounds
        if message_mapper:

            def new_message_mapper(
                messages_by_round: List[List[BaseMessage]],
            ) -> List[BaseMessage]:
                # 过滤每轮的消息
                messages_by_round = self._filter_round_messages(messages_by_round)
                return message_mapper(messages_by_round)

        else:

            def new_message_mapper(
                messages_by_round: List[List[BaseMessage]],
            ) -> List[BaseMessage]:
                # 过滤每轮的消息
                messages_by_round = self._filter_round_messages(messages_by_round)
                # 合并多轮消息
                return _merge_multi_round_messages(messages_by_round)

        # 调用父类的初始化方法
        super().__init__(new_message_mapper, **kwargs)

    def _filter_round_messages(
        self, messages_by_round: List[List[BaseMessage]]
# 定义一个类型别名 EvictionPolicyType，表示一个回调函数类型，接受一个列表，其中每个元素是一个列表，其内部元素是 BaseMessage 类型，返回值也是一个类似结构的列表
EvictionPolicyType = Callable[[List[List[BaseMessage]]], List[List[BaseMessage]]]


class TokenBufferedConversationMapperOperator(ConversationMapperOperator):
    """The token buffered conversation mapper operator.

    If the token count of the messages is greater than the max token limit, we will
    evict the messages by round.

    Args:
        model (str): The model name.
        llm_client (LLMClient): The LLM client.
        max_token_limit (int): The max token limit.
        eviction_policy (EvictionPolicyType): The eviction policy.
        message_mapper (_MultiRoundMessageMapper): The message mapper, it applies after
            all messages are handled.
    """

    def __init__(
        self,
        model: str,
        llm_client: LLMClient,
        max_token_limit: int = 2000,
        eviction_policy: Optional[EvictionPolicyType] = None,
        message_mapper: Optional[_MultiRoundMessageMapper] = None,
        **kwargs,
    ):
        """Create a new TokenBufferedConversationMapperOperator."""
        # 如果最大令牌限制小于0，抛出 ValueError 异常
        if max_token_limit < 0:
            raise ValueError("Max token limit can't be negative")
        self._model = model  # 设置模型名称
        self._llm_client = llm_client  # 设置 LLM 客户端
        self._max_token_limit = max_token_limit  # 设置最大令牌限制
        self._eviction_policy = eviction_policy  # 设置驱逐策略
        self._message_mapper = message_mapper  # 设置消息映射器
        super().__init__(**kwargs)  # 调用父类的初始化方法

    async def map_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Map multi round messages to a list of BaseMessage."""
        # 获取驱逐策略，如果没有设置，则使用默认的驱逐策略
        eviction_policy = self._eviction_policy or self.eviction_policy
        # 将消息按照轮次拆分为列表的列表
        messages_by_round: List[List[BaseMessage]] = _split_messages_by_round(messages)
        # 将所有消息转换为字符串表示，并合并多轮次消息
        messages_str = _messages_to_str(_merge_multi_round_messages(messages_by_round))
        # 第一次计算消息的令牌数
        current_tokens = await self._llm_client.count_token(self._model, messages_str)

        while current_tokens > self._max_token_limit:
            # 当令牌数超过最大限制时，依据驱逐策略进行消息驱逐
            # TODO: 我们应该找到一种高效的方法来执行这个操作
            messages_by_round = eviction_policy(messages_by_round)
            messages_str = _messages_to_str(
                _merge_multi_round_messages(messages_by_round)
            )
            # 重新计算当前令牌数
            current_tokens = await self._llm_client.count_token(
                self._model, messages_str
            )
        # 获取消息映射器，如果没有设置，则使用默认的多轮次消息映射器
        message_mapper = self._message_mapper or self.map_multi_round_messages
        # 对处理后的消息应用消息映射器
        return message_mapper(messages_by_round)

    def eviction_policy(
        self, messages_by_round: List[List[BaseMessage]]
    ) -> List[List[BaseMessage]]:
        # 此方法需要被子类实现，用于执行消息驱逐策略
        pass  # 这里使用 pass 占位符，因为该方法需要在子类中具体实现
    ) -> List[List[BaseMessage]]:
        """Evict the messages by round, default is FIFO.

        Args:
            messages_by_round (List[List[BaseMessage]]): The messages by round.
                A list of lists where each inner list contains BaseMessage objects for each round.

        Returns:
            List[List[BaseMessage]]: The evicted messages by round.
                The updated list of lists after removing the first round of messages.
        """
        # Remove the first round of messages from the list
        messages_by_round.pop(0)
        # Return the updated list of lists after eviction
        return messages_by_round
# 合并多个轮次的消息列表为单个消息列表
def _merge_multi_round_messages(messages: List[List[BaseMessage]]) -> List[BaseMessage]:
    # 使用内置的sum函数，将多个列表中的元素依次合并到一个新列表中
    # 示例中的断言表达式演示了sum函数的用法和效果
    return sum(messages, [])
```