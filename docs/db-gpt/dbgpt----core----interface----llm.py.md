# `.\DB-GPT-src\dbgpt\core\interface\llm.py`

```py
"""The interface for LLM."""

import collections  # 导入 collections 模块，用于支持额外的数据结构
import copy  # 导入 copy 模块，用于复制对象
import logging  # 导入 logging 模块，用于日志记录
import time  # 导入 time 模块，用于处理时间相关功能
from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 和 abstractmethod，用于定义抽象基类和抽象方法
from dataclasses import asdict, dataclass, field  # 导入 dataclasses 模块相关函数，用于数据类的处理
from typing import Any, AsyncIterator, Dict, List, Optional, Union  # 导入 typing 模块，用于类型提示

from cachetools import TTLCache  # 导入 TTLCache 类，用于实现 TTL 缓存

from dbgpt._private.pydantic import BaseModel, model_to_dict  # 从私有模块导入 BaseModel 和 model_to_dict 函数
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType  # 导入消息相关类
from dbgpt.util import BaseParameters  # 导入 BaseParameters 类，用于基础参数处理
from dbgpt.util.annotations import PublicAPI  # 导入 PublicAPI 装饰器，标记为公共 API
from dbgpt.util.model_utils import GPUInfo  # 导入 GPUInfo 类，用于 GPU 相关信息处理

logger = logging.getLogger(__name__)  # 获取当前模块的 logger 对象


@dataclass
@PublicAPI(stability="beta")
class ModelInferenceMetrics:
    """A class to represent metrics for assessing the inference performance of a LLM."""
    
    collect_index: Optional[int] = 0
    """The index used for collecting metrics in a sequence."""

    start_time_ms: Optional[int] = None
    """The timestamp (in milliseconds) when the model inference starts."""

    end_time_ms: Optional[int] = None
    """The timestamp (in milliseconds) when the model inference ends."""

    current_time_ms: Optional[int] = None
    """The current timestamp (in milliseconds) when the model inference returns partially output(stream)."""

    first_token_time_ms: Optional[int] = None
    """The timestamp (in milliseconds) when the first token is generated."""

    first_completion_time_ms: Optional[int] = None
    """The timestamp (in milliseconds) when the first completion is generated."""

    first_completion_tokens: Optional[int] = None
    """The number of tokens when the first completion is generated."""

    prompt_tokens: Optional[int] = None
    """The number of tokens in the input prompt."""

    completion_tokens: Optional[int] = None
    """The number of tokens in the generated completion."""

    total_tokens: Optional[int] = None
    """The total number of tokens (prompt plus completion)."""

    speed_per_second: Optional[float] = None
    """The average number of tokens generated per second."""

    current_gpu_infos: Optional[List[GPUInfo]] = None
    """Current GPU information, all devices"""

    avg_gpu_infos: Optional[List[GPUInfo]] = None
    """Average memory usage across all collection points"""

    @staticmethod
    def create_metrics(
        last_metrics: Optional["ModelInferenceMetrics"] = None,
        """Static method to create a new instance of ModelInferenceMetrics based on optional last_metrics."""
    ) -> "ModelInferenceMetrics":
        """Create metrics for model inference.

        Args:
            last_metrics(ModelInferenceMetrics): The last metrics.

        Returns:
            ModelInferenceMetrics: The metrics for model inference.
        """
        # 如果传入了上一次的 metrics 对象，则从中获取各项指标，否则设为 None
        start_time_ms = last_metrics.start_time_ms if last_metrics else None
        first_token_time_ms = last_metrics.first_token_time_ms if last_metrics else None
        first_completion_time_ms = (
            last_metrics.first_completion_time_ms if last_metrics else None
        )
        first_completion_tokens = (
            last_metrics.first_completion_tokens if last_metrics else None
        )
        prompt_tokens = last_metrics.prompt_tokens if last_metrics else None
        completion_tokens = last_metrics.completion_tokens if last_metrics else None
        total_tokens = last_metrics.total_tokens if last_metrics else None
        speed_per_second = last_metrics.speed_per_second if last_metrics else None
        current_gpu_infos = last_metrics.current_gpu_infos if last_metrics else None
        avg_gpu_infos = last_metrics.avg_gpu_infos if last_metrics else None

        # 如果 start_time_ms 为 None，则获取当前时间的毫秒数作为开始时间
        if not start_time_ms:
            start_time_ms = time.time_ns() // 1_000_000
        current_time_ms = time.time_ns() // 1_000_000
        end_time_ms = current_time_ms

        # 返回一个新的 ModelInferenceMetrics 对象，包含各项指标
        return ModelInferenceMetrics(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            current_time_ms=current_time_ms,
            first_token_time_ms=first_token_time_ms,
            first_completion_time_ms=first_completion_time_ms,
            first_completion_tokens=first_completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            speed_per_second=speed_per_second,
            current_gpu_infos=current_gpu_infos,
            avg_gpu_infos=avg_gpu_infos,
        )

    def to_dict(self) -> Dict:
        """Convert the model inference metrics to dict."""
        # 将当前对象转换为字典形式并返回
        return asdict(self)
@dataclass
@PublicAPI(stability="beta")
class ModelRequestContext:
    """用于表示LLM模型请求上下文的类。"""

    stream: bool = False
    """是否返回响应流。"""

    cache_enable: bool = False
    """是否启用模型推理的缓存。"""

    user_name: Optional[str] = None
    """模型请求的用户名。"""

    sys_code: Optional[str] = None
    """模型请求的系统代码。"""

    conv_uid: Optional[str] = None
    """模型推理的会话ID。"""

    span_id: Optional[str] = None
    """模型推理的跨度ID。"""

    chat_mode: Optional[str] = None
    """模型推理的聊天模式。"""

    chat_param: Optional[str] = None
    """聊天模式的聊天参数。"""

    extra: Optional[Dict[str, Any]] = field(default_factory=dict)
    """模型推理的额外信息。"""

    request_id: Optional[str] = None
    """模型推理的请求ID。"""


@dataclass
@PublicAPI(stability="beta")
class ModelOutput:
    """用于表示LLM的输出结果的类。"""

    text: str
    """生成的文本。"""
    
    error_code: int
    """模型推理的错误代码。若推理成功，错误代码为0。"""

    incremental: bool = False
    """是否增量输出结果。"""

    model_context: Optional[Dict] = None
    """模型推理的上下文信息。"""

    finish_reason: Optional[str] = None
    """模型推理完成的原因。"""

    usage: Optional[Dict[str, Any]] = None
    """模型推理的使用情况。"""

    metrics: Optional[ModelInferenceMetrics] = None
    """模型推理的一些指标。"""

    def to_dict(self) -> Dict:
        """将模型输出转换为字典格式。"""
        return asdict(self)

    @property
    def success(self) -> bool:
        """检查模型推理是否成功。"""
        return self.error_code == 0


_ModelMessageType = Union[List[ModelMessage], List[Dict[str, Any]]]


@dataclass
@PublicAPI(stability="beta")
class ModelRequest:
    """表示模型请求的类。"""

    model: str
    """模型的名称。"""

    messages: _ModelMessageType
    """输入消息。可以是模型消息的列表或字典的列表。"""

    temperature: Optional[float] = None
    """模型推理的温度参数。"""

    max_new_tokens: Optional[int] = None
    """生成的最大标记数。"""

    stop: Optional[str] = None
    """模型推理的停止条件。"""

    stop_token_ids: Optional[List[int]] = None
    """模型推理的停止标记ID。"""

    context_len: Optional[int] = None
    """模型推理的上下文长度。"""

    echo: Optional[bool] = False
    """是否回显输入消息。"""

    span_id: Optional[str] = None
    """模型推理的跨度ID。"""

    context: Optional[ModelRequestContext] = field(
        default_factory=lambda: ModelRequestContext()
    )
    """模型推理的上下文信息。"""

    @property
    def
    def stream(self) -> bool:
        """Whether to return a stream of responses."""
        # 返回一个布尔值，表示是否返回响应流，依据是上下文存在且请求指定了流模式
        return bool(self.context and self.context.stream)

    def copy(self) -> "ModelRequest":
        """Copy the model request.

        Returns:
            ModelRequest: The copied model request.
        """
        # 深拷贝当前模型请求对象
        new_request = copy.deepcopy(self)
        # 转换消息列表为 List[ModelMessage]
        new_request.messages = new_request.get_messages()
        return new_request

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model request to dict.

        Returns:
            Dict[str, Any]: The model request in dict.
        """
        # 深拷贝当前模型请求对象
        new_request = copy.deepcopy(self)
        new_messages = []
        # 遍历消息列表，如果消息是字典则直接添加，否则调用其 dict() 方法并添加
        for message in new_request.messages:
            if isinstance(message, dict):
                new_messages.append(message)
            else:
                new_messages.append(message.dict())
        new_request.messages = new_messages
        # 返回转换后的字典，跳过值为 None 的字段
        return {k: v for k, v in asdict(new_request).items() if v is not None}

    def to_trace_metadata(self) -> Dict[str, Any]:
        """Convert the model request to trace metadata.

        Returns:
            Dict[str, Any]: The trace metadata.
        """
        # 获取模型请求的字典表示
        metadata = self.to_dict()
        # 将请求的消息转换为字符串并添加到元数据中的 "prompt" 键
        metadata["prompt"] = self.messages_to_string()
        return metadata

    def get_messages(self) -> List[ModelMessage]:
        """Get the messages.

        If the messages is not a list of ModelMessage, it will be converted to a list
        of ModelMessage.

        Returns:
            List[ModelMessage]: The messages.
        """
        # 获取消息列表，确保其类型为 ModelMessage 的列表
        messages = []
        for message in self.messages:
            if isinstance(message, dict):
                messages.append(ModelMessage(**message))
            else:
                messages.append(message)
        return messages

    def get_single_user_message(self) -> Optional[ModelMessage]:
        """Get the single user message.

        Returns:
            Optional[ModelMessage]: The single user message.
        """
        # 获取单个用户消息，如果不符合条件则引发 ValueError 异常
        messages = self.get_messages()
        if len(messages) != 1 and messages[0].role != ModelMessageRoleType.HUMAN:
            raise ValueError("The messages is not a single user message")
        return messages[0]

    @staticmethod
    def build_request(
        model: str,
        messages: List[ModelMessage],
        context: Optional[Union[ModelRequestContext, Dict[str, Any], BaseModel]] = None,
        stream: bool = False,
        echo: bool = False,
        **kwargs,
        ):
        """Build a model request object.

        Args:
            model (str): The model identifier.
            messages (List[ModelMessage]): The list of messages.
            context (Optional[Union[ModelRequestContext, Dict[str, Any], BaseModel]], optional):
                Optional context for the model request. Defaults to None.
            stream (bool, optional): Whether to use streaming mode. Defaults to False.
            echo (bool, optional): Whether to echo messages. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelRequest: The constructed model request object.
        """
        # 构建一个模型请求对象，可以包括模型标识、消息列表、上下文等信息
        pass  # 此处只是说明，实际上不执行任何操作
    ):
        """
        Build a model request.

        Args:
            model(str): The model name.
                The name of the model for which the request is being built.
            messages(List[ModelMessage]): The messages.
                List of ModelMessage objects representing the input messages.
            context(Optional[Union[ModelRequestContext, Dict[str, Any], BaseModel]]):
                The context.
                Optional context information for the model request, can be a ModelRequestContext object,
                a dictionary, or a BaseModel instance.
            stream(bool): Whether to return a stream of responses. Defaults to False.
                Flag indicating whether the response should be streamed.
            echo(bool): Whether to echo the input messages. Defaults to False.
                Flag indicating whether to echo the input messages in the response.
            **kwargs: Other arguments.
                Additional keyword arguments that can be passed to the ModelRequest.

        Returns:
            ModelRequest: The constructed model request object.

        """
        if not context:
            context = ModelRequestContext(stream=stream)
        elif not isinstance(context, ModelRequestContext):
            context_dict = None
            if isinstance(context, dict):
                context_dict = context
            elif isinstance(context, BaseModel):
                context_dict = model_to_dict(context)
            if context_dict and "stream" not in context_dict:
                context_dict["stream"] = stream
            if context_dict:
                context = ModelRequestContext(**context_dict)
            else:
                context = ModelRequestContext(stream=stream)
        return ModelRequest(
            model=model,
            messages=messages,
            context=context,
            echo=echo,
            **kwargs,
        )

    @staticmethod
    def _build(model: str, prompt: str, **kwargs):
        """
        Build a model request for a prompt.

        Args:
            model(str): The model name.
                The name of the model for which the request is being built.
            prompt(str): The prompt text.
                The text prompt to be sent as a message.
            **kwargs: Other arguments.
                Additional keyword arguments that can be passed to the ModelRequest.

        Returns:
            ModelRequest: The constructed model request object.

        """
        return ModelRequest(
            model=model,
            messages=[ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)],
            **kwargs,
        )

    def to_common_messages(
        self, support_system_role: bool = True
        ):
        """
        Convert to common messages.

        Args:
            support_system_role(bool): Whether to support system role. Defaults to True.
                Flag indicating whether to include system role messages.

        Returns:
            List[ModelMessage]: A list of common model messages.

        """
    ) -> List[Dict[str, Any]]:
        """Convert the messages to the common format(like OpenAI API).

        This function will move last user message to the end of the list.

        Args:
            support_system_role (bool): Whether to support system role

        Returns:
            List[Dict[str, Any]]: The messages in the format of OpenAI API.

        Raises:
            ValueError: If the message role is not supported

        Examples:
            .. code-block:: python

                from dbgpt.core.interface.message import (
                    ModelMessage,
                    ModelMessageRoleType,
                )

                messages = [
                    ModelMessage(role=ModelMessageRoleType.HUMAN, content="Hi"),
                    ModelMessage(
                        role=ModelMessageRoleType.AI, content="Hi, I'm a robot."
                    ),
                    ModelMessage(
                        role=ModelMessageRoleType.HUMAN, content="Who are your"
                    ),
                ]
                openai_messages = ModelRequest.to_openai_messages(messages)
                assert openai_messages == [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hi, I'm a robot."},
                    {"role": "user", "content": "Who are your"},
                ]
        """
        # Convert all messages to instances of ModelMessage if they are not already
        messages = [
            m if isinstance(m, ModelMessage) else ModelMessage(**m)
            for m in self.messages
        ]
        # Convert messages to the common format expected by OpenAI API
        return ModelMessage.to_common_messages(
            messages, support_system_role=support_system_role
        )

    def messages_to_string(self) -> str:
        """Convert the messages to string.

        Returns:
            str: The messages in string format.
        """
        # Get all messages and convert them to a single string representation
        return ModelMessage.messages_to_string(self.get_messages())
# 使用 dataclass 装饰器定义 ModelExtraMedata 类，表示 LLM 的额外元数据
@dataclass
class ModelExtraMedata(BaseParameters):
    """A class to represent the extra metadata of a LLM."""

    # 定义 prompt_roles 属性，表示提示的角色列表，默认包含系统、人类和 AI
    prompt_roles: List[str] = field(
        default_factory=lambda: [
            ModelMessageRoleType.SYSTEM,
            ModelMessageRoleType.HUMAN,
            ModelMessageRoleType.AI,
        ],
        metadata={"help": "The roles of the prompt"},
    )

    # 定义 prompt_sep 属性，表示提示在多轮对话中的分隔符，默认为换行符
    prompt_sep: Optional[str] = field(
        default="\n",
        metadata={"help": "The separator of the prompt between multiple rounds"},
    )

    # 定义 prompt_chat_template 属性，表示聊天模板，可在模型仓库的 tokenizer 配置中找到
    prompt_chat_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "The chat template, see: "
            "https://huggingface.co/docs/transformers/main/en/chat_templating"
        },
    )

    # 定义 support_system_message 属性，表示模型是否支持系统消息
    @property
    def support_system_message(self) -> bool:
        """Whether the model supports system message.

        Returns:
            bool: Whether the model supports system message.
        """
        return ModelMessageRoleType.SYSTEM in self.prompt_roles


# 使用 dataclass 装饰器定义 ModelMetadata 类，表示 LLM 模型
@dataclass
@PublicAPI(stability="beta")
class ModelMetadata(BaseParameters):
    """A class to represent a LLM model."""

    # 定义 model 属性，表示模型名称
    model: str = field(
        metadata={"help": "Model name"},
    )
    # 定义 context_length 属性，表示模型的上下文长度，默认为 4096
    context_length: Optional[int] = field(
        default=4096,
        metadata={"help": "Context length of model"},
    )
    # 定义 chat_model 属性，表示模型是否是聊天模型，默认为 True
    chat_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether the model is a chat model"},
    )
    # 定义 is_function_calling_model 属性，表示模型是否是函数调用模型，默认为 False
    is_function_calling_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether the model is a function calling model"},
    )
    # 定义 metadata 属性，表示模型元数据，默认为空字典
    metadata: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "Model metadata"},
    )
    # 定义 ext_metadata 属性，表示模型的额外元数据，默认为 ModelExtraMedata 的实例
    ext_metadata: Optional[ModelExtraMedata] = field(
        default_factory=ModelExtraMedata,
        metadata={"help": "Model extra metadata"},
    )

    # 定义 from_dict 类方法，从字典创建新的模型元数据
    @classmethod
    def from_dict(
        cls, data: dict, ignore_extra_fields: bool = False
    ) -> "ModelMetadata":
        """Create a new model metadata from a dict."""
        # 如果字典中包含 ext_metadata 键，则将其转换为 ModelExtraMedata 实例
        if "ext_metadata" in data:
            data["ext_metadata"] = ModelExtraMedata(**data["ext_metadata"])
        return cls(**data)


# 定义 MessageConverter 抽象类，用于消息转换器
class MessageConverter(ABC):
    r"""An abstract class for message converter.

    Different LLMs may have different message formats, this class is used to convert
    the messages to the format of the LLM.
    @abstractmethod
    def convert(
        self,
        messages: List[ModelMessage],
        model_metadata: Optional[ModelMetadata] = None,
    ) -> List[ModelMessage]:
        """Convert the messages.

        Args:
            messages(List[ModelMessage]): The messages.
            model_metadata(ModelMetadata): The model metadata.

        Returns:
            List[ModelMessage]: The converted messages.
        """
    # 创建一个默认消息转换器类，继承自MessageConverter类
    """The default message converter."""

    # 初始化方法，创建一个新的默认消息转换器实例
    def __init__(self, prompt_sep: Optional[str] = None):
        """Create a new default message converter."""
        # 设置一个用于分隔提示的可选参数
        self._prompt_sep = prompt_sep

    # 转换消息的方法，执行消息转换的三个步骤：
    def convert(
        self,
        messages: List[ModelMessage],
        model_metadata: Optional[ModelMetadata] = None,
    ) -> List[ModelMessage]:
        """Convert the messages.

        There are three steps to convert the messages:

        1. Just keep system, human and AI messages
        2. Move the last user's message to the end of the list
        3. Convert the messages to no system message if the model does not support
        system message

        Args:
            messages(List[ModelMessage]): The messages.
            model_metadata(ModelMetadata): The model metadata.

        Returns:
            List[ModelMessage]: The converted messages.
        """
        # 1. 只保留系统消息、人类消息和AI消息
        messages = list(filter(lambda m: m.pass_to_model, messages))
        # 2. 将最后一个用户消息移到列表末尾
        messages = self.move_last_user_message_to_end(messages)

        # 检查模型元数据和外部元数据是否存在
        if not model_metadata or not model_metadata.ext_metadata:
            # 如果没有模型元数据，记录警告并返回原始消息列表
            logger.warning("No model metadata, skip message system message conversion")
            return messages
        if not model_metadata.ext_metadata.support_system_message:
            # 3. 如果模型不支持系统消息，转换消息为不含系统消息
            return self.convert_to_no_system_message(messages, model_metadata)
        # 否则返回处理后的消息列表
        return messages

    # 将消息转换为不包含系统消息的方法
    def convert_to_no_system_message(
        self,
        messages: List[ModelMessage],
        model_metadata: Optional[ModelMetadata] = None,
    ) -> List[ModelMessage]:
    ) -> List[ModelMessage]:
        r"""Convert the messages to no system message.

        Examples:
            >>> # Convert the messages to no system message, just merge system messages
            >>> # to the last user message
            >>> from typing import List
            >>> from dbgpt.core.interface.message import (
            ...     ModelMessage,
            ...     ModelMessageRoleType,
            ... )
            >>> from dbgpt.core.interface.llm import (
            ...     DefaultMessageConverter,
            ...     ModelMetadata,
            ... )
            >>> messages = [
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="Who are you"
            ...     ),
            ... ]
            >>> converter = DefaultMessageConverter()
            >>> model_metadata = ModelMetadata(model="test")
            >>> converted_messages = converter.convert_to_no_system_message(
            ...     messages, model_metadata
            ... )
            >>> assert converted_messages == [
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN,
            ...         content="You are a helpful assistant\nWho are you",
            ...     ),
            ... ]
        """
        # 如果模型元数据不存在或者其扩展元数据不存在，则记录警告并返回原始消息列表
        if not model_metadata or not model_metadata.ext_metadata:
            logger.warning("No model metadata, skip message conversion")
            return messages
        
        # 获取模型的扩展元数据
        ext_metadata = model_metadata.ext_metadata
        
        # 初始化系统消息和结果消息列表
        system_messages = []
        result_messages = []
        
        # 遍历每条消息
        for message in messages:
            if message.role == ModelMessageRoleType.SYSTEM:
                # 如果消息角色是系统消息，则将其添加到系统消息列表中
                system_messages.append(message)
            elif message.role in [
                ModelMessageRoleType.HUMAN,
                ModelMessageRoleType.AI,
            ]:
                # 如果消息角色是人类消息或AI消息，则将其添加到结果消息列表中
                result_messages.append(message)
        
        # 获取提示分隔符，优先使用对象的分隔符，否则使用扩展元数据中的分隔符，默认为换行符
        prompt_sep = self._prompt_sep or ext_metadata.prompt_sep or "\n"
        
        # 初始化系统消息字符串为None
        system_message_str = None
        
        # 如果系统消息列表中有多条消息
        if len(system_messages) > 1:
            # 记录警告信息，指出系统消息数量超过一条
            logger.warning("Your system messages have more than one message")
            # 将系统消息列表中所有消息的内容用提示分隔符连接起来作为系统消息字符串
            system_message_str = prompt_sep.join([m.content for m in system_messages])
        # 如果系统消息列表中只有一条消息
        elif len(system_messages) == 1:
            # 将系统消息字符串设置为该条消息的内容
            system_message_str = system_messages[0].content
        
        # 如果存在系统消息字符串且结果消息列表不为空
        if system_message_str and result_messages:
            # 不支持系统消息，将系统消息合并到最后一条用户消息的内容中
            result_messages[-1].content = (
                system_message_str + prompt_sep + result_messages[-1].content
            )
        
        # 返回处理后的结果消息列表
        return result_messages
    ) -> List[ModelMessage]:
        """尝试将最后一条用户消息移动到列表的末尾。

        Examples:
            >>> from typing import List
            >>> from dbgpt.core.interface.message import (
            ...     ModelMessage,
            ...     ModelMessageRoleType,
            ... )
            >>> from dbgpt.core.interface.llm import DefaultMessageConverter
            >>> messages = [
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="Who are you"
            ...     ),
            ...     ModelMessage(role=ModelMessageRoleType.AI, content="I'm a robot"),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="What's your name"
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ... ]
            >>> converter = DefaultMessageConverter()
            >>> converted_messages = converter.move_last_user_message_to_end(messages)
            >>> assert converted_messages == [
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="Who are you"
            ...     ),
            ...     ModelMessage(role=ModelMessageRoleType.AI, content="I'm a robot"),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.SYSTEM,
            ...         content="You are a helpful assistant",
            ...     ),
            ...     ModelMessage(
            ...         role=ModelMessageRoleType.HUMAN, content="What's your name"
            ...     ),
            ... ]

        Args:
            messages(List[ModelMessage]): 要处理的消息列表。

        Returns:
            List[ModelMessage]: 处理后的消息列表，最后一条用户消息被移动到末尾。
        """
        last_user_input_index = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == ModelMessageRoleType.HUMAN:
                last_user_input_index = i
                break
        if last_user_input_index is not None:
            last_user_input = messages.pop(last_user_input_index)
            messages.append(last_user_input)
        return messages
# 使用 @PublicAPI 注解标识此类是公共 API 的一部分，稳定性为 beta
class LLMClient(ABC):
    """An abstract class for LLM client."""

    # 缓存模型元数据，缓存项最大数为 100，过期时间为 60 秒
    _MODEL_CACHE_ = TTLCache(maxsize=100, ttl=60)

    @property
    def cache(self) -> collections.abc.MutableMapping:
        """Return the cache object to cache the model metadata.

        You can override this property to use your own cache object.
        Returns:
            collections.abc.MutableMapping: The cache object.
        """
        return self._MODEL_CACHE_

    @abstractmethod
    async def generate(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelOutput:
        """Generate a response for a given model request.

        Sometimes, different LLMs may have different message formats,
        you can use the message converter to convert the messages to the format of the
        LLM.

        Args:
            request(ModelRequest): The model request.
            message_converter(MessageConverter): The message converter.

        Returns:
            ModelOutput: The model output.

        """

    @abstractmethod
    async def generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> AsyncIterator[ModelOutput]:
        """Generate a stream of responses for a given model request.

        Sometimes, different LLMs may have different message formats,
        you can use the message converter to convert the messages to the format of the
        LLM.

        Args:
            request(ModelRequest): The model request.
            message_converter(MessageConverter): The message converter.

        Returns:
            AsyncIterator[ModelOutput]: The model output stream.
        """

    @abstractmethod
    async def models(self) -> List[ModelMetadata]:
        """Get all the models.

        Returns:
            List[ModelMetadata]: A list of model metadata.
        """

    @abstractmethod
    async def count_token(self, model: str, prompt: str) -> int:
        """Count the number of tokens in a given prompt.

        Args:
            model(str): The model name.
            prompt(str): The prompt.

        Returns:
            int: The number of tokens.
        """

    async def covert_message(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelRequest:
        """Convert a model request message.

        This method is meant to be overridden to handle specific message conversions.

        Args:
            request(ModelRequest): The model request to convert.
            message_converter(MessageConverter): The message converter.

        Returns:
            ModelRequest: The converted model request.
        """
    async def convert_message(
        self, request: ModelRequest, message_converter: MessageConverter = None
    ) -> ModelRequest:
        """Convert the message.

        If no message converter is provided, the original request will be returned.

        Args:
            request(ModelRequest): The model request.
            message_converter(MessageConverter): The message converter.

        Returns:
            ModelRequest: The converted model request.
        """
        # 如果没有提供消息转换器，直接返回原始请求
        if not message_converter:
            return request
        # 复制原始请求，以免修改原始数据
        new_request = request.copy()
        # 获取指定模型的元数据
        model_metadata = await self.get_model_metadata(request.model)
        # 使用消息转换器将原始消息转换为新消息
        new_messages = message_converter.convert(request.get_messages(), model_metadata)
        # 更新新请求的消息数据
        new_request.messages = new_messages
        return new_request

    async def cached_models(self) -> List[ModelMetadata]:
        """Get all the models from the cache or the llm server.

        If the model metadata is not in the cache, it will be fetched from the
        llm server.

        Returns:
            List[ModelMetadata]: A list of model metadata.
        """
        # 缓存中模型数据的键
        key = "____$llm_client_models$____"
        # 如果键不在缓存中，则从服务器获取模型数据并缓存
        if key not in self.cache:
            models = await self.models()
            self.cache[key] = models
            # 将每个模型的元数据也缓存起来
            for model in models:
                model_metadata_key = (
                    f"____$llm_client_models_metadata_{model.model}$____"
                )
                self.cache[model_metadata_key] = model
        return self.cache[key]

    async def get_model_metadata(self, model: str) -> ModelMetadata:
        """Get the model metadata.

        Args:
            model(str): The model name.

        Returns:
            ModelMetadata: The model metadata.

        Raises:
            ValueError: If the model is not found.
        """
        # 构造指定模型元数据在缓存中的键
        model_metadata_key = f"____$llm_client_models_metadata_{model}$____"
        # 如果键不在缓存中，从缓存模型数据中获取
        if model_metadata_key not in self.cache:
            await self.cached_models()
        # 获取模型元数据
        model_metadata = self.cache.get(model_metadata_key)
        # 如果未获取到模型元数据，则抛出异常
        if not model_metadata:
            raise ValueError(f"Model {model} not found")
        return model_metadata
    def __call__(self, *args, **kwargs) -> ModelOutput:
        """
        Return the model output.

        Call the LLM client to generate the response for the given message.

        Please do not use this method in the production environment, it is only used
        for debugging.
        """
        from dbgpt.util import get_or_create_event_loop  # 导入获取或创建事件循环的函数

        messages = kwargs.get("messages")  # 从 kwargs 中获取 "messages" 参数
        model = kwargs.get("model")  # 从 kwargs 中获取 "model" 参数
        if messages:
            del kwargs["messages"]  # 如果存在 "messages" 参数，则从 kwargs 中删除它
            model_messages = ModelMessage.from_openai_messages(messages)  # 将 OpenAI 消息转换为 ModelMessage 对象
        else:
            model_messages = [ModelMessage.build_human_message(args[0])]  # 否则，根据第一个参数创建一个人类消息的 ModelMessage 对象
        if not model:
            if hasattr(self, "default_model"):
                model = getattr(self, "default_model")  # 如果没有传入 model 参数，则尝试获取默认模型
            else:
                raise ValueError("The default model is not set")  # 如果没有默认模型，则抛出 ValueError 异常
        if "model" in kwargs:
            del kwargs["model"]  # 如果 kwargs 中包含 "model" 参数，则删除它
        req = ModelRequest.build_request(model, model_messages, **kwargs)  # 构建模型请求对象
        loop = get_or_create_event_loop()  # 获取或创建事件循环
        return loop.run_until_complete(self.generate(req))  # 使用事件循环运行生成方法，并返回生成的模型输出
```