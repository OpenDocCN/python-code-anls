# `.\DB-GPT-src\dbgpt\util\tracer\tests\test_base.py`

```py
# 导入所需模块和类
from typing import Dict

from dbgpt.component import SystemApp  # 导入 SystemApp 类
from dbgpt.util.tracer import Span, SpanStorage, SpanType, Tracer  # 导入 Span 相关类和 Tracer 类

# Mock 实现

# MockSpanStorage 类，继承自 SpanStorage 类
class MockSpanStorage(SpanStorage):
    def __init__(self):
        self.spans = []  # 初始化空的 spans 列表

    # 添加 span 到 spans 列表
    def append_span(self, span: Span):
        self.spans.append(span)


# MockTracer 类，继承自 Tracer 类
class MockTracer(Tracer):
    def __init__(self, system_app: SystemApp | None = None):
        super().__init__(system_app)
        self.current_span = None  # 当前 span 初始化为 None
        self.storage = MockSpanStorage()  # 初始化 MockSpanStorage 对象作为存储

    # 添加 span 到 storage 中
    def append_span(self, span: Span):
        self.storage.append_span(span)

    # 开始一个 span
    def start_span(
        self, operation_name: str, parent_span_id: str = None, metadata: Dict = None
    ) -> Span:
        # 如果没有指定 parent_span_id，则生成新的 trace_id
        trace_id = (
            self._new_uuid() if parent_span_id is None else parent_span_id.split(":")[0]
        )
        # 生成新的 span_id
        span_id = f"{trace_id}:{self._new_uuid()}"
        # 创建 Span 对象
        span = Span(
            trace_id, span_id, SpanType.BASE, parent_span_id, operation_name, metadata
        )
        self.current_span = span  # 更新当前 span
        return span  # 返回创建的 span 对象

    # 结束一个 span
    def end_span(self, span: Span):
        span.end()  # 调用 span 的 end 方法
        self.append_span(span)  # 将 span 添加到 storage 中

    # 获取当前的 span
    def get_current_span(self) -> Span:
        return self.current_span

    # 获取当前的 storage
    def _get_current_storage(self) -> SpanStorage:
        return self.storage


# 测试用例

# 测试 Span 类的创建和属性设置
def test_span_creation():
    span = Span(
        "trace_id",
        "span_id",
        SpanType.BASE,
        "parent_span_id",
        "operation",
        {"key": "value"},
    )
    assert span.trace_id == "trace_id"  # 断言 trace_id 是否正确设置
    assert span.span_id == "span_id"  # 断言 span_id 是否正确设置
    assert span.parent_span_id == "parent_span_id"  # 断言 parent_span_id 是否正确设置
    assert span.operation_name == "operation"  # 断言 operation_name 是否正确设置
    assert span.metadata == {"key": "value"}  # 断言 metadata 是否正确设置


# 测试 Span 类的结束方法
def test_span_end():
    span = Span("trace_id", "span_id")
    assert span.end_time is None  # 断言 span 的 end_time 是否为 None
    span.end()  # 调用 span 的 end 方法
    assert span.end_time is not None  # 断言 span 的 end_time 是否不为 None


# 测试 MockTracer 类的 start_span 方法
def test_mock_tracer_start_span():
    tracer = MockTracer()
    span = tracer.start_span("operation")
    assert span.operation_name == "operation"  # 断言 span 的 operation_name 是否正确设置
    assert tracer.get_current_span() == span  # 断言 tracer 中的当前 span 是否与新创建的 span 相同


# 测试 MockTracer 类的 end_span 方法
def test_mock_tracer_end_span():
    tracer = MockTracer()
    span = tracer.start_span("operation")
    tracer.end_span(span)
    assert span in tracer._get_current_storage().spans  # 断言 span 是否被正确添加到 storage 中


# 测试 MockTracer 类的 append_span 方法
def test_mock_tracer_append_span():
    tracer = MockTracer()
    span = Span("trace_id", "span_id")
    tracer.append_span(span)
    assert span in tracer._get_current_storage().spans  # 断言 span 是否被正确添加到 storage 中


# 测试父子 span 的关系
def test_parent_child_span_relation():
    tracer = MockTracer()

    # 开始一个父 span
    parent_span = tracer.start_span("parent_operation")

    # 开始一个子 span，使用父 span 的 ID 作为 parent_span_id
    child_span = tracer.start_span(
        "child_operation", parent_span_id=parent_span.span_id
    )

    # 断言父子 span 的关系
    assert child_span.parent_span_id == parent_span.span_id  # 断言 child_span 的 parent_span_id 是否等于 parent_span 的 span_id
    assert (
        child_span.trace_id == parent_span.trace_id
    )  # 假设子 span 和父 span 共享相同的 trace_id

    # 结束 spans
    tracer.end_span(child_span)
    # 结束指定的跟踪span
    tracer.end_span(parent_span)
    
    # 断言child_span存在于当前跟踪器的存储中
    assert child_span in tracer._get_current_storage().spans
    
    # 断言parent_span存在于当前跟踪器的存储中
    assert parent_span in tracer._get_current_storage().spans
# 测试确保生成的 UUID 是唯一的
# 注意：这是一个简单的测试，不能保证在大量生成 UUID 的情况下的唯一性。

def test_new_uuid_unique():
    # 创建 MockTracer 实例用于跟踪
    tracer = MockTracer()
    # 生成 1000 个 UUID，并存储在集合 uuid_set 中
    uuid_set = {tracer._new_uuid() for _ in range(1000)}
    # 断言集合中的 UUID 数量是否为 1000，以确保生成的 UUID 均不重复
    assert len(uuid_set) == 1000
```