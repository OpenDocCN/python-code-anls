# `.\DB-GPT-src\dbgpt\util\tracer\tests\test_tracer_impl.py`

```py
# 导入 pytest 库，用于测试框架
import pytest

# 从 dbgpt.component 模块导入 SystemApp 类
from dbgpt.component import SystemApp
# 从 dbgpt.util.tracer 模块导入以下类和枚举
from dbgpt.util.tracer import (
    DefaultTracer,
    MemorySpanStorage,
    Span,
    SpanStorage,
    SpanStorageType,
    Tracer,
    TracerManager,
)

# 定义系统应用的 pytest fixture
@pytest.fixture
def system_app():
    return SystemApp()

# 定义存储的 pytest fixture，初始化一个 MemorySpanStorage 实例并注册到 system_app
@pytest.fixture
def storage(system_app: SystemApp):
    ms = MemorySpanStorage(system_app)
    system_app.register_instance(ms)
    return ms

# 定义跟踪器的 pytest fixture，根据请求参数创建 DefaultTracer 实例
@pytest.fixture
def tracer(request, system_app: SystemApp):
    if not request or not hasattr(request, "param"):
        return DefaultTracer(system_app)
    else:
        # 从请求参数中获取 span_storage_type，默认为 SpanStorageType.ON_CREATE_END
        span_storage_type = request.param.get(
            "span_storage_type", SpanStorageType.ON_CREATE_END
        )
        return DefaultTracer(system_app, span_storage_type=span_storage_type)

# 定义跟踪器管理器的 pytest fixture，注册 tracer 到 system_app 并初始化 TracerManager 实例
@pytest.fixture
def tracer_manager(system_app: SystemApp, tracer: Tracer):
    system_app.register_instance(tracer)
    manager = TracerManager()
    manager.initialize(system_app)
    return manager

# 测试函数：测试开始和结束跟踪 span 的功能
def test_start_and_end_span(tracer: Tracer):
    # 开始一个名为 "operation" 的 span
    span = tracer.start_span("operation")
    assert isinstance(span, Span)  # 断言 span 是 Span 类的实例
    assert span.operation_name == "operation"  # 断言 span 的操作名称为 "operation"

    # 结束 span
    tracer.end_span(span)
    assert span.end_time is not None  # 断言 span 的结束时间不为 None

    # 断言存储的 span 中包含结束的 span
    stored_span = tracer._get_current_storage().spans[0]
    assert stored_span.span_id == span.span_id  # 断言存储的 span 的 span_id 与原始 span 的一致

# 测试函数：测试使用 TracerManager 开始和结束跟踪 span 的功能
def test_start_and_end_span_with_tracer_manager(tracer_manager: TracerManager):
    # 使用 TracerManager 开始一个名为 "operation" 的 span
    span = tracer_manager.start_span("operation")
    assert isinstance(span, Span)  # 断言 span 是 Span 类的实例
    assert span.operation_name == "operation"  # 断言 span 的操作名称为 "operation"

    # 结束 span
    tracer_manager.end_span(span)
    assert span.end_time is not None  # 断言 span 的结束时间不为 None

# 测试函数：测试父子 span 的关系
def test_parent_child_span_relation(tracer: Tracer):
    # 开始一个名为 "parent_operation" 的父 span
    parent_span = tracer.start_span("parent_operation")
    # 开始一个名为 "child_operation" 的子 span，指定父 span 的 span_id
    child_span = tracer.start_span(
        "child_operation", parent_span_id=parent_span.span_id
    )

    # 断言子 span 的 parent_span_id 与父 span 的 span_id 相同
    assert child_span.parent_span_id == parent_span.span_id
    # 断言子 span 的 trace_id 与父 span 的 trace_id 相同
    assert child_span.trace_id == parent_span.trace_id

    # 结束子 span 和父 span
    tracer.end_span(child_span)
    tracer.end_span(parent_span)

    # 断言存储的 spans 中包含父 span 和子 span
    assert parent_span.operation_name in [
        s.operation_name for s in tracer._get_current_storage().spans
    ]
    assert child_span.operation_name in [
        s.operation_name for s in tracer._get_current_storage().spans
    ]

# 测试函数：使用 pytest.mark.parametrize 参数化测试不同的 tracer 和预期计数
@pytest.mark.parametrize(
    "tracer, expected_count, after_create_inc_count",
    [
        ({"span_storage_type": SpanStorageType.ON_CREATE}, 1, 1),
        ({"span_storage_type": SpanStorageType.ON_END}, 1, 0),
        ({"span_storage_type": SpanStorageType.ON_CREATE_END}, 2, 1),
    ],
    indirect=["tracer"],  # 指定 tracer 参数为间接参数
)
def test_tracer_span_storage_type_and_with(
    tracer: Tracer,
    expected_count: int,
    after_create_inc_count: int,
    storage: SpanStorage,
):
    # 开始一个名为 "new_span" 的 span
    span = tracer.start_span("new_span")
    # 结束 span
    span.end()
    # 断言存储的 spans 的数量符合预期
    assert len(storage.spans) == expected_count

    # 使用 with 语句开始一个名为 "with_span" 的 span
    with tracer.start_span("with_span") as ws:
        # 断言存储的 spans 的数量符合预期
        assert len(storage.spans) == expected_count + after_create_inc_count
    # 断言存储的 spans 的数量符合预期
    assert len(storage.spans) == expected_count + expected_count
```