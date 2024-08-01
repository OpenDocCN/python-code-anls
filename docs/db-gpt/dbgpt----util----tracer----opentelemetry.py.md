# `.\DB-GPT-src\dbgpt\util\tracer\opentelemetry.py`

```py
# 从 typing 模块中导入 Dict、List、Optional 类型
from typing import Dict, List, Optional
# 从当前包中导入 Span、SpanStorage、_split_span_id
from .base import Span, SpanStorage, _split_span_id

# 尝试导入 OpenTelemetry 相关模块
try:
    # 导入 trace 相关模块
    from opentelemetry import trace
    # 导入 OTLPSpanExporter 类
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    # 导入 Resource 类
    from opentelemetry.sdk.resources import Resource
    # 导入 Span 和 TracerProvider 类
    from opentelemetry.sdk.trace import Span as OTSpan
    from opentelemetry.sdk.trace import TracerProvider
    # 导入 BatchSpanProcessor 类
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    # 导入 SpanContext 和 SpanKind
    from opentelemetry.trace import SpanContext, SpanKind
# 如果 ImportError 发生，则抛出 ImportError 异常
except ImportError:
    raise ImportError(
        "To use OpenTelemetrySpanStorage, you must install opentelemetry-api, "
        "opentelemetry-sdk and opentelemetry-exporter-otlp."
        "You can install it via `pip install opentelemetry-api opentelemetry-sdk "
        "opentelemetry-exporter-otlp`"
    )

# 定义 OpenTelemetrySpanStorage 类，继承自 SpanStorage
class OpenTelemetrySpanStorage(SpanStorage):
    """OpenTelemetry span storage."""

    # 初始化方法，接收 service_name、otlp_endpoint、otlp_insecure 和 otlp_timeout 参数
    def __init__(
        self,
        service_name: str,
        otlp_endpoint: Optional[str] = None,
        otlp_insecure: Optional[bool] = None,
        otlp_timeout: Optional[int] = None,
    ):
        # 调用父类 SpanStorage 的初始化方法
        super().__init__()
        # 将 service_name 参数赋值给实例变量 self.service_name
        self.service_name = service_name

        # 创建 Resource 对象，包含服务名的属性
        resource = Resource(attributes={"service.name": service_name})
        # 创建 TracerProvider 对象，传入 Resource 对象
        self.tracer_provider = TracerProvider(resource=resource)
        # 获取当前模块的 Tracer 对象
        self.tracer = self.tracer_provider.get_tracer(__name__)
        
        # 存储未结束的 span 的字典，键为 span 的 ID，值为 OTSpan 对象
        self.spans: Dict[str, OTSpan] = {}

        # 创建 OTLPSpanExporter 对象，传入 otlp_endpoint、otlp_insecure、otlp_timeout 参数
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=otlp_insecure,
            timeout=otlp_timeout,
        )
        # 创建 BatchSpanProcessor 对象，传入 OTLPSpanExporter 对象
        span_processor = BatchSpanProcessor(otlp_exporter)
        # 将 span_processor 添加到 TracerProvider 中
        self.tracer_provider.add_span_processor(span_processor)
        
        # 设置全局的 TracerProvider 为当前实例的 TracerProvider
        trace.set_tracer_provider(self.tracer_provider)
    # 将给定的 span 对象添加到当前跟踪器中
    def append_span(self, span: Span):
        # 获取 span 的唯一标识符
        span_id = span.span_id

        # 如果该 span_id 已存在于 spans 字典中
        if span_id in self.spans:
            # 弹出已存在的 span 对象
            otel_span = self.spans.pop(span_id)
            # 更新 span 的结束时间和属性
            end_time = int(span.end_time.timestamp() * 1e9) if span.end_time else None
            # 如果有元数据，更新到 otel_span 的属性中
            if span.metadata:
                for key, value in span.metadata.items():
                    # 只有特定类型的值才能作为属性值
                    if isinstance(value, (bool, str, bytes, int, float)) or (
                        isinstance(value, list)
                        and all(
                            isinstance(i, (bool, str, bytes, int, float)) for i in value
                        )
                    ):
                        otel_span.set_attribute(key, value)
            # 设置 span 的结束时间
            if end_time:
                otel_span.end(end_time=end_time)
            else:
                otel_span.end()
        else:
            # 创建父上下文，用于关联跟踪
            parent_context = self._create_parent_context(span)
            # 将 span 的开始时间从 datetime 转换为整数
            start_time = int(span.start_time.timestamp() * 1e9)

            # 在跟踪器中创建新的 span 对象
            otel_span = self.tracer.start_span(
                span.operation_name,
                context=parent_context,
                kind=SpanKind.INTERNAL,
                start_time=start_time,
            )

            # 设置调试标识符到 otel_span 的属性中
            otel_span.set_attribute("dbgpt_trace_id", span.trace_id)
            otel_span.set_attribute("dbgpt_span_id", span.span_id)

            # 如果存在父 span 的标识符，则设置到 otel_span 的属性中
            if span.parent_span_id:
                otel_span.set_attribute("dbgpt_parent_span_id", span.parent_span_id)

            # 设置 span 类型到 otel_span 的属性中
            otel_span.set_attribute("span_type", span.span_type.value)

            # 如果有元数据，更新到 otel_span 的属性中
            if span.metadata:
                for key, value in span.metadata.items():
                    if isinstance(value, (bool, str, bytes, int, float)) or (
                        isinstance(value, list)
                        and all(
                            isinstance(i, (bool, str, bytes, int, float)) for i in value
                        )
                    ):
                        otel_span.set_attribute(key, value)

            # 如果 span 没有结束时间，将其添加到 spans 字典中
            if not span.end_time:
                self.spans[span_id] = otel_span

    # 将给定的 spans 列表中的每个 span 添加到当前跟踪器中
    def append_span_batch(self, spans: List[Span]):
        for span in spans:
            self.append_span(span)

    # 根据给定的 span 创建父上下文
    def _create_parent_context(self, span: Span):
        # 如果 span 没有父 span 标识符，返回无效的跟踪上下文
        if not span.parent_span_id:
            return trace.set_span_in_context(trace.INVALID_SPAN)

        # 从 span 的父 span 标识符中分割出 trace_id 和 parent_span_id
        trace_id, parent_span_id = _split_span_id(span.parent_span_id)
        # 如果没有有效的 trace_id，返回无效的跟踪上下文
        if not trace_id:
            return trace.set_span_in_context(trace.INVALID_SPAN)

        # 创建 SpanContext 对象作为父上下文，并设置为远程跟踪
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=trace.TraceFlags(0x01),  # 默认为采样
        )
        return trace.set_span_in_context(trace.NonRecordingSpan(span_context))

    # 关闭跟踪器提供者，释放资源
    def close(self):
        self.tracer_provider.shutdown()
```