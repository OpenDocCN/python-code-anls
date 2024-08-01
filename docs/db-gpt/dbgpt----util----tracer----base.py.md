# `.\DB-GPT-src\dbgpt\util\tracer\base.py`

```py
# 从未来导入注释的功能，允许在类中使用类型注释
from __future__ import annotations

# 导入用于处理 JSON 数据的模块
import json
# 导入用于生成安全随机数的模块
import secrets
# 导入用于生成唯一标识符的模块
import uuid
# 导入抽象基类模块，用于定义抽象类
from abc import ABC, abstractmethod
# 导入数据类模块，用于定义数据类
from dataclasses import dataclass
# 导入日期时间模块，用于处理日期和时间
from datetime import datetime
# 导入枚举类型模块
from enum import Enum
# 导入用于类型注释的模块
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 dbgpt.component 模块中的特定内容
from dbgpt.component import BaseComponent, ComponentType, SystemApp

# 定义用于追踪的全局常量
DBGPT_TRACER_SPAN_ID = "DB-GPT-Trace-Span-Id"

# 兼容性：OpenTelemetry API 中的最大跟踪和跨度 ID 值
_TRACE_ID_MAX_VALUE = 2**128 - 1
_SPAN_ID_MAX_VALUE = 2**64 - 1
INVALID_SPAN_ID = 0x0000000000000000
INVALID_TRACE_ID = 0x00000000000000000000000000000000

# 定义用于表示跨度类型的枚举
class SpanType(str, Enum):
    BASE = "base"
    RUN = "run"
    CHAT = "chat"
    AGENT = "agent"

# 定义用于表示运行时跨度名称的枚举
class SpanTypeRunName(str, Enum):
    WEBSERVER = "Webserver"
    WORKER_MANAGER = "WorkerManager"
    MODEL_WORKER = "ModelWorker"
    EMBEDDING_MODEL = "EmbeddingModel"

    @staticmethod
    def values():
        return [item.value for item in SpanTypeRunName]

# 表示一个被追踪的工作单元的类
class Span:
    """Represents a unit of work that is being traced.
    This can be any operation like a function call or a database query.
    """

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        span_type: SpanType = None,
        parent_span_id: str = None,
        operation_name: str = None,
        metadata: Dict = None,
        end_caller: Callable[[Span], None] = None,
    ):
        # 如果未指定跨度类型，默认为 SpanType.BASE
        if not span_type:
            span_type = SpanType.BASE
        # 设置跨度类型
        self.span_type = span_type
        # 整个追踪的唯一标识符
        self.trace_id = trace_id
        # 在追踪中此跨度的唯一标识符
        self.span_id = span_id
        # 父跨度的标识符，如果这是一个子跨度
        self.parent_span_id = parent_span_id
        # 正在追踪操作的描述性名称
        self.operation_name = operation_name
        # 跨度开始时的时间戳
        self.start_time = datetime.now()
        # 跨度结束时的时间戳，初始为 None
        self.end_time = None
        # 与跨度相关的额外元数据
        self.metadata = metadata or {}
        # 注册用于结束跨度时调用的回调函数
        self._end_callers = []
        if end_caller:
            self._end_callers.append(end_caller)

    # 标记跨度结束，记录当前时间
    def end(self, **kwargs):
        self.end_time = datetime.now()
        # 如果传递了 metadata 参数，则更新元数据
        if "metadata" in kwargs:
            self.metadata = kwargs.get("metadata")
        # 调用所有注册的结束回调函数
        for caller in self._end_callers:
            caller(self)

    # 添加用于结束时调用的回调函数
    def add_end_caller(self, end_caller: Callable[[Span], None]):
        if end_caller:
            self._end_callers.append(end_caller)

    # 上下文管理器方法，在退出上下文时自动结束跨度
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return False
    # 返回一个字典，包含当前对象的各个属性
    def to_dict(self) -> Dict:
        return {
            # 获取当前对象的 span_type 属性的枚举值
            "span_type": self.span_type.value,
            # 获取当前对象的 trace_id 属性
            "trace_id": self.trace_id,
            # 获取当前对象的 span_id 属性
            "span_id": self.span_id,
            # 获取当前对象的 parent_span_id 属性
            "parent_span_id": self.parent_span_id,
            # 获取当前对象的 operation_name 属性
            "operation_name": self.operation_name,
            # 将当前对象的 start_time 属性转换为指定格式的字符串，精确到毫秒
            "start_time": (
                None
                if not self.start_time
                else self.start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            ),
            # 将当前对象的 end_time 属性转换为指定格式的字符串，精确到毫秒
            "end_time": (
                None
                if not self.end_time
                else self.end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            ),
            # 清理当前对象的 metadata 属性以便转换为 JSON 格式，若为空则为 None
            "metadata": _clean_for_json(self.metadata) if self.metadata else None,
        }

    # 创建当前 Span 对象的副本
    def copy(self) -> Span:
        """Create a copy of this span."""
        # 复制当前对象的 metadata 属性，若为空则设为 None
        metadata = self.metadata.copy() if self.metadata else None
        # 创建一个新的 Span 对象，使用当前对象的属性值（除了 metadata 以外）
        span = Span(
            self.trace_id,
            self.span_id,
            self.span_type,
            self.parent_span_id,
            self.operation_name,
            metadata=metadata,
        )
        # 复制当前对象的 start_time 属性到新的 Span 对象
        span.start_time = self.start_time
        # 复制当前对象的 end_time 属性到新的 Span 对象
        span.end_time = self.end_time
        # 返回新创建的 Span 对象
        return span
class SpanStorageType(str, Enum):
    ON_CREATE = "on_create"  # 定义枚举类型，表示跟踪数据存储的不同方式，例如在创建时存储、在结束时存储、同时在创建和结束时存储
    ON_END = "on_end"
    ON_CREATE_END = "on_create_end"


class SpanStorage(BaseComponent, ABC):
    """Abstract base class for storing spans.
    
    This allows different storage mechanisms (e.g., in-memory, database) to be implemented.
    """

    name = ComponentType.TRACER_SPAN_STORAGE.value  # 设置存储组件的名称为 TRACER_SPAN_STORAGE

    def init_app(self, system_app: SystemApp):
        """Initialize the storage with the given application context."""
        pass  # 初始化存储器，使用给定的应用程序上下文

    @abstractmethod
    def append_span(self, span: Span):
        """Store the given span. This needs to be implemented by subclasses."""
        # 存储给定的跟踪数据。具体实现由子类完成

    def append_span_batch(self, spans: List[Span]):
        """Store the span batch"""
        for span in spans:
            self.append_span(span)  # 存储跟踪数据的批量处理功能


class Tracer(BaseComponent, ABC):
    """Abstract base class for tracing operations.
    Provides the core logic for starting, ending, and retrieving spans.
    """

    name = ComponentType.TRACER.value  # 设置跟踪组件的名称为 TRACER

    def __init__(self, system_app: SystemApp | None = None):
        super().__init__(system_app)
        self.system_app = system_app  # 应用程序上下文

    def init_app(self, system_app: SystemApp):
        """Initialize the tracer with the given application context."""
        self.system_app = system_app  # 使用给定的应用程序上下文初始化跟踪器

    @abstractmethod
    def append_span(self, span: Span):
        """Append the given span to storage. This needs to be implemented by subclasses."""
        # 将给定的跟踪数据追加到存储中。具体实现由子类完成

    @abstractmethod
    def start_span(
        self,
        operation_name: str,
        parent_span_id: str = None,
        span_type: SpanType = None,
        metadata: Dict = None,
    ) -> Span:
        """Begin a new span for the given operation. If provided, the span will be
        a child of the span with the given parent_span_id.
        """
        # 开始一个新的跟踪操作。如果提供了 parent_span_id，则该操作将作为其子操作进行跟踪

    @abstractmethod
    def end_span(self, span: Span, **kwargs):
        """
        End the given span.
        """
        # 结束给定的跟踪操作

    @abstractmethod
    def get_current_span(self) -> Optional[Span]:
        """
        Retrieve the span that is currently being traced.
        """
        # 获取当前正在跟踪的跟踪操作

    @abstractmethod
    def _get_current_storage(self) -> SpanStorage:
        """
        Get the storage mechanism currently in use for storing spans.
        This needs to be implemented by subclasses.
        """
        # 获取当前用于存储跟踪数据的存储机制。子类需要实现该方法

    def _new_uuid(self) -> str:
        """
        Generate a new unique identifier.
        """
        return str(uuid.uuid4())  # 生成新的唯一标识符

    def _new_random_trace_id(self) -> str:
        """Create a new random trace ID."""
        return _new_random_trace_id()  # 创建新的随机跟踪 ID

    def _new_random_span_id(self) -> str:
        """Create a new random span ID."""
        return _new_random_span_id()  # 创建新的随机跟踪操作 ID


def _new_random_trace_id() -> str:
    """Create a new random trace ID."""
    # Generate a 128-bit hex string
    return secrets.token_hex(16)  # 生成一个 128 位的十六进制字符串


def _is_valid_trace_id(trace_id: Union[str, int]) -> bool:
    """Check if the given trace ID is valid."""
    # 检查给定的跟踪 ID 是否有效
    ...
    # 如果 trace_id 是字符串类型
    if isinstance(trace_id, str):
        try:
            # 尝试将十六进制字符串转换为整数
            trace_id = int(trace_id, 16)
        except ValueError:
            # 如果转换失败，返回 False
            return False
    # 检查 trace_id 是否在有效范围内
    return INVALID_TRACE_ID < int(trace_id) <= _TRACE_ID_MAX_VALUE
# 创建一个新的随机跟踪 ID，以 64 位十六进制字符串表示
def _new_random_span_id() -> str:
    """Create a new random span ID."""
    # 生成一个 64 位的十六进制字符串
    return secrets.token_hex(8)


# 检查给定的跟踪 ID 是否有效
def _is_valid_span_id(span_id: Union[str, int]) -> bool:
    if isinstance(span_id, str):
        try:
            # 尝试将十六进制的字符串转换为整数
            span_id = int(span_id, 16)
        except ValueError:
            return False
    # 检查跟踪 ID 是否在有效范围内
    return INVALID_SPAN_ID < int(span_id) <= _SPAN_ID_MAX_VALUE


# 解析给定的跟踪 ID 字符串，返回其包含的 trace_id 和 parent_span_id
def _split_span_id(span_id: str) -> Tuple[int, int]:
    parent_span_id_parts = span_id.split(":")
    if len(parent_span_id_parts) != 2:
        return 0, 0
    trace_id, parent_span_id = parent_span_id_parts
    try:
        # 尝试将十六进制的 trace_id 和 parent_span_id 转换为整数
        trace_id = int(trace_id, 16)
        span_id = int(parent_span_id, 16)
        return trace_id, span_id
    except ValueError:
        return 0, 0


# 定义一个跟踪器上下文的数据类，用于保存当前的 span_id
@dataclass
class TracerContext:
    span_id: Optional[str] = None


# 清理数据以便于 JSON 序列化，如果数据无法清理，则返回 None
def _clean_for_json(data: Optional[str, Any] = None):
    if data is None:
        return None
    if isinstance(data, dict):
        cleaned_dict = {}
        for key, value in data.items():
            # 尝试递归清理子项数据
            cleaned_value = _clean_for_json(value)
            if cleaned_value is not None:
                # 只有清理后的值不为 None 才添加到清理后的字典中
                try:
                    json.dumps({key: cleaned_value})
                    cleaned_dict[key] = cleaned_value
                except TypeError:
                    # 如果无法序列化，则跳过该键值对
                    pass
        return cleaned_dict
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            cleaned_item = _clean_for_json(item)
            if cleaned_item is not None:
                try:
                    json.dumps(cleaned_item)
                    cleaned_list.append(cleaned_item)
                except TypeError:
                    pass
        return cleaned_list
    else:
        try:
            # 尝试将数据转换为 JSON 格式
            json.dumps(data)
            return data
        except TypeError:
            return None


# 解析请求或数据对象中的跟踪 ID，返回有效的 span_id
def _parse_span_id(body: Any) -> Optional[str]:
    from starlette.requests import Request
    from dbgpt._private.pydantic import BaseModel, model_to_dict

    span_id: Optional[str] = None
    if isinstance(body, Request):
        # 如果 body 是 Request 对象，则从头部获取 DBGPT_TRACER_SPAN_ID
        span_id = body.headers.get(DBGPT_TRACER_SPAN_ID)
    elif isinstance(body, dict):
        # 如果 body 是字典，则尝试从 DBGPT_TRACER_SPAN_ID 或 "span_id" 键获取
        span_id = body.get(DBGPT_TRACER_SPAN_ID) or body.get("span_id")
    elif isinstance(body, BaseModel):
        # 如果 body 是 BaseModel 实例，则将其转换为字典后再获取 span_id
        dict_body = model_to_dict(body)
        span_id = dict_body.get(DBGPT_TRACER_SPAN_ID) or dict_body.get("span_id")
    if not span_id:
        return None
    else:
        # 解析 span_id，获取其包含的 trace_id 和 parent_span_id
        int_trace_id, int_span_id = _split_span_id(span_id)
        if not int_trace_id:
            return None
        # 检查 span_id 和 trace_id 是否有效
        if _is_valid_span_id(int_span_id) and _is_valid_trace_id(int_trace_id):
            return span_id
        else:
            return span_id
```