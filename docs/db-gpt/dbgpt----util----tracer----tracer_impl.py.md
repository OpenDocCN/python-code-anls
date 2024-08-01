# `.\DB-GPT-src\dbgpt\util\tracer\tracer_impl.py`

```py
import asyncio  # 导入异步IO库，用于异步操作
import inspect  # 导入inspect模块，用于获取对象信息
import logging  # 导入日志模块，用于记录日志
from contextvars import ContextVar  # 从contextvars模块导入ContextVar，用于上下文变量管理
from functools import wraps  # 导入wraps装饰器，用于保留函数元信息
from typing import Any, AsyncIterator, Dict, Optional  # 导入类型提示相关模块

from dbgpt.component import ComponentType, SystemApp  # 导入调试组件相关类和系统应用类
from dbgpt.util.module_utils import import_from_checked_string  # 从模块工具中导入字符串安全导入函数
from dbgpt.util.tracer.base import (  # 从追踪器基础模块导入追踪器相关类和枚举
    Span,
    SpanStorage,
    SpanStorageType,
    SpanType,
    Tracer,
    TracerContext,
)
from dbgpt.util.tracer.span_storage import MemorySpanStorage  # 从追踪器存储模块导入内存追踪器存储类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class DefaultTracer(Tracer):
    def __init__(
        self,
        system_app: SystemApp | None = None,
        default_storage: SpanStorage = None,
        span_storage_type: SpanStorageType = SpanStorageType.ON_CREATE_END,
    ):
        super().__init__(system_app)  # 调用父类构造函数初始化系统应用
        self._span_stack_var = ContextVar("span_stack", default=[])  # 创建上下文变量"span_stack"，用于存储跨度堆栈

        if not default_storage:
            default_storage = MemorySpanStorage(system_app)  # 如果未提供默认存储，则使用内存跨度存储
        self._default_storage = default_storage  # 设置默认存储
        self._span_storage_type = span_storage_type  # 设置跨度存储类型

    def append_span(self, span: Span):
        self._get_current_storage().append_span(span.copy())  # 将给定的跨度添加到当前存储中

    def start_span(
        self,
        operation_name: str,
        parent_span_id: str = None,
        span_type: SpanType = None,
        metadata: Dict = None,
    ) -> Span:
        trace_id = (
            self._new_random_trace_id()  # 生成新的随机追踪ID
            if parent_span_id is None
            else parent_span_id.split(":")[0]  # 如果提供了父跨度ID，则从中提取追踪ID
        )
        span_id = f"{trace_id}:{self._new_random_span_id()}"  # 使用追踪ID和新的随机跨度ID生成跨度ID

        span = Span(  # 创建新的跨度对象
            trace_id,
            span_id,
            span_type,
            parent_span_id,
            operation_name,
            metadata=metadata,
        )

        if self._span_storage_type in [
            SpanStorageType.ON_END,
            SpanStorageType.ON_CREATE_END,
        ]:
            span.add_end_caller(self.append_span)  # 如果跨度存储类型为ON_END或ON_CREATE_END，则在结束时调用append_span方法

        if self._span_storage_type in [
            SpanStorageType.ON_CREATE,
            SpanStorageType.ON_CREATE_END,
        ]:
            self.append_span(span)  # 如果跨度存储类型为ON_CREATE或ON_CREATE_END，则立即调用append_span方法存储跨度

        current_stack = self._span_stack_var.get()  # 获取当前跨度堆栈
        current_stack.append(span)  # 将新创建的跨度添加到堆栈中
        self._span_stack_var.set(current_stack)  # 更新上下文变量中的跨度堆栈

        span.add_end_caller(self._remove_from_stack_top)  # 在跨度结束时调用_remove_from_stack_top方法
        return span  # 返回创建的跨度对象

    def end_span(self, span: Span, **kwargs):
        """"""
        span.end(**kwargs)  # 结束指定的跨度对象

    def _remove_from_stack_top(self, span: Span):
        current_stack = self._span_stack_var.get()  # 获取当前跨度堆栈
        if current_stack:
            current_stack.pop()  # 弹出堆栈顶部的跨度
        self._span_stack_var.set(current_stack)  # 更新上下文变量中的跨度堆栈

    def get_current_span(self) -> Optional[Span]:
        current_stack = self._span_stack_var.get()  # 获取当前跨度堆栈
        return current_stack[-1] if current_stack else None  # 返回堆栈顶部的跨度对象，如果堆栈为空则返回None

    def _get_current_storage(self) -> SpanStorage:
        return self.system_app.get_component(  # 获取系统应用中的跟踪器跨度存储组件
            ComponentType.TRACER_SPAN_STORAGE, SpanStorage, self._default_storage
        )


class TracerManager:
    """The manager of current tracer"""

    # 初始化方法，设置初始属性
    def __init__(self) -> None:
        # 系统应用的实例，可选类型为 SystemApp
        self._system_app: Optional[SystemApp] = None
        # 上下文变量，用于存储跟踪器上下文，类型为 ContextVar[TracerContext]
        self._trace_context_var: ContextVar[TracerContext] = ContextVar(
            "trace_context",
            default=TracerContext(),
        )

    # 初始化管理器，设置系统应用实例和跟踪器上下文变量
    def initialize(
        self, system_app: SystemApp, trace_context_var: ContextVar[TracerContext] = None
    ) -> None:
        # 设置系统应用实例
        self._system_app = system_app
        # 如果提供了跟踪器上下文变量，则使用提供的，否则使用默认值
        if trace_context_var:
            self._trace_context_var = trace_context_var

    # 获取跟踪器实例的私有方法
    def _get_tracer(self) -> Tracer:
        # 如果系统应用实例不存在，则返回空值
        if not self._system_app:
            return None
        # 否则，从系统应用实例中获取跟踪器组件
        return self._system_app.get_component(ComponentType.TRACER, Tracer, None)

    # 开始一个新的跟踪 span
    def start_span(
        self,
        operation_name: str,
        parent_span_id: str = None,
        span_type: SpanType = None,
        metadata: Dict = None,
    ) -> Span:
        """Start a new span with operation_name
        This method must not throw an exception under any case and try not to block as much as possible
        """
        # 获取当前跟踪器实例
        tracer = self._get_tracer()
        # 如果跟踪器实例不存在，返回一个空的 Span 对象
        if not tracer:
            return Span(
                "empty_span", "empty_span", span_type=span_type, metadata=metadata
            )
        # 如果没有指定父 span ID，则使用当前活动的 span ID
        if not parent_span_id:
            parent_span_id = self.get_current_span_id()
        # 如果未指定 span 类型且存在父 span ID，则获取当前活动 span 的类型
        if not span_type and parent_span_id:
            span_type = self._get_current_span_type()
        # 使用跟踪器实例开始一个新的 span
        return tracer.start_span(
            operation_name, parent_span_id, span_type=span_type, metadata=metadata
        )

    # 结束给定的 span
    def end_span(self, span: Span, **kwargs):
        # 获取当前跟踪器实例
        tracer = self._get_tracer()
        # 如果跟踪器实例不存在或未提供 span 对象，则直接返回
        if not tracer or not span:
            return
        # 使用跟踪器实例结束指定的 span
        tracer.end_span(span, **kwargs)

    # 获取当前活动的 span
    def get_current_span(self) -> Optional[Span]:
        # 获取当前跟踪器实例
        tracer = self._get_tracer()
        # 如果跟踪器实例不存在，则返回空值
        if not tracer:
            return None
        # 否则，返回当前跟踪器实例的当前活动 span
        return tracer.get_current_span()

    # 获取当前活动的 span ID
    def get_current_span_id(self) -> Optional[str]:
        # 获取当前活动的 span
        current_span = self.get_current_span()
        # 如果存在当前活动 span，则返回其 span ID
        if current_span:
            return current_span.span_id
        # 否则，获取跟踪器上下文变量中的 span ID（如果存在）
        ctx = self._trace_context_var.get()
        return ctx.span_id if ctx else None

    # 获取当前活动 span 的类型
    def _get_current_span_type(self) -> Optional[SpanType]:
        # 获取当前活动 span
        current_span = self.get_current_span()
        # 如果存在当前活动 span，则返回其类型
        return current_span.span_type if current_span else None

    # 解析给定主体的 span ID
    def _parse_span_id(self, body: Any) -> Optional[str]:
        # 导入并调用 _parse_span_id 方法来解析 span ID
        from .base import _parse_span_id

        return _parse_span_id(body)

    # 异步流的包装器方法，用于在异步生成器中创建 span
    def wrapper_async_stream(
        self,
        generator: AsyncIterator[Any],
        operation_name: str,
        parent_span_id: str = None,
        span_type: SpanType = None,
        metadata: Dict = None,
        **kwargs
    ) -> AsyncIterator[Any]:
        # 在异步生成器中启动新的 span
        return self.start_span(operation_name, parent_span_id, span_type, metadata, **kwargs)
        ) -> AsyncIterator[Any]:
        """Wrap an async generator with a span"""

        # 获取当前 span 的父 span ID，如果没有指定则使用当前 span 的 ID
        parent_span_id = parent_span_id or self.get_current_span_id()

        # 定义一个内部函数 wrapper，用于包装 async generator
        async def wrapper():
            # 创建一个新的 span，并传入操作名称、父 span ID、span 类型和元数据
            span = self.start_span(operation_name, parent_span_id, span_type, metadata)
            try:
                # 遍历 async generator，yield 每个 item
                async for item in generator:
                    yield item
            finally:
                # 结束 span
                span.end()

        # 返回内部函数 wrapper
        return wrapper()
root_tracer: TracerManager = TracerManager()

# 定义一个全局的 TracerManager 对象 root_tracer，用于管理和跟踪系统中的操作和函数调用


def trace(operation_name: Optional[str] = None, **trace_kwargs):
    # 定义一个装饰器函数 trace，用于添加跟踪操作到函数上
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 定义同步函数装饰器 sync_wrapper，负责同步函数的跟踪逻辑
            name = (
                operation_name if operation_name else _parse_operation_name(func, *args)
            )
            # 获取操作名，如果未提供则从函数和参数中解析
            with root_tracer.start_span(name, **trace_kwargs):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 定义异步函数装饰器 async_wrapper，负责异步函数的跟踪逻辑
            name = (
                operation_name if operation_name else _parse_operation_name(func, *args)
            )
            # 获取操作名，如果未提供则从函数和参数中解析
            with root_tracer.start_span(name, **trace_kwargs):
                return await func(*args, **kwargs)

        # 根据函数是否为异步函数返回相应的装饰器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _parse_operation_name(func, *args):
    # 解析操作名的辅助函数，根据函数签名和参数判断是否需要包含 self 的类名
    self_name = None
    if inspect.signature(func).parameters.get("self"):
        self_name = args[0].__class__.__name__
    func_name = func.__name__
    if self_name:
        return f"{self_name}.{func_name}"
    return func_name


def initialize_tracer(
    tracer_filename: str,
    root_operation_name: str = "DB-GPT-Webserver",
    system_app: Optional[SystemApp] = None,
    tracer_storage_cls: Optional[str] = None,
    create_system_app: bool = False,
    enable_open_telemetry: bool = False,
    otlp_endpoint: Optional[str] = None,
    otlp_insecure: Optional[bool] = None,
    otlp_timeout: Optional[int] = None,
):
    """初始化跟踪器，设置跟踪的文件名和系统应用。"""
    from dbgpt.util.tracer.span_storage import FileSpanStorage, SpanStorageContainer

    if not system_app and create_system_app:
        system_app = SystemApp()
    # 如果没有提供系统应用并且需要创建，则创建一个新的 SystemApp 对象
    if not system_app:
        return
    # 如果系统应用为空，则直接返回

    trace_context_var = ContextVar(
        "trace_context",
        default=TracerContext(),
    )
    # 创建跟踪上下文变量 trace_context_var，并使用默认的 TracerContext

    tracer = DefaultTracer(system_app)
    # 创建一个默认的 Tracer 对象，使用提供的系统应用

    storage_container = SpanStorageContainer(system_app)
    # 创建一个 SpanStorageContainer 对象，用于管理跟踪数据的存储

    storage_container.append_storage(FileSpanStorage(tracer_filename))
    # 将文件存储的 SpanStorage 添加到 storage_container 中，用于持久化跟踪数据

    if enable_open_telemetry:
        from dbgpt.util.tracer.opentelemetry import OpenTelemetrySpanStorage

        # 如果开启了 OpenTelemetry 支持，则添加 OpenTelemetrySpanStorage
        storage_container.append_storage(
            OpenTelemetrySpanStorage(
                service_name=root_operation_name,
                otlp_endpoint=otlp_endpoint,
                otlp_insecure=otlp_insecure,
                otlp_timeout=otlp_timeout,
            )
        )

    if tracer_storage_cls:
        logger.info(f"Begin parse storage class {tracer_storage_cls}")
        # 如果提供了自定义的跟踪存储类，记录日志信息
        storage = import_from_checked_string(tracer_storage_cls, SpanStorage)
        # 通过字符串导入跟踪存储类
        storage_container.append_storage(storage())
        # 将自定义的跟踪存储类添加到 storage_container 中

    system_app.register_instance(storage_container)
    # 将 storage_container 注册为系统应用的实例，用于管理跟踪数据存储

    system_app.register_instance(tracer)
    # 将 tracer 注册为系统应用的实例，用于管理跟踪操作和生成跟踪数据

    root_tracer.initialize(system_app, trace_context_var)
    # 初始化根跟踪器，使用系统应用和跟踪上下文变量
    # 如果 system_app.app 存在（即不为 None），则执行以下操作
    if system_app.app:
        # 导入 TraceIDMiddleware 类，用于追踪 ID 的中间件
        from dbgpt.util.tracer.tracer_middleware import TraceIDMiddleware
    
        # 将 TraceIDMiddleware 中间件添加到 system_app.app 中
        # 参数说明：
        #   - TraceIDMiddleware: 要添加的中间件类
        #   - trace_context_var: 跟踪上下文变量
        #   - tracer: 追踪器对象
        #   - root_operation_name: 根操作名称
        system_app.app.add_middleware(
            TraceIDMiddleware,
            trace_context_var=trace_context_var,
            tracer=tracer,
            root_operation_name=root_operation_name,
        )
```