# `.\DB-GPT-src\dbgpt\util\tracer\tests\test_span_storage.py`

```py
# 引入 asyncio 模块，用于支持异步编程
import asyncio
# 引入 json 模块，用于 JSON 数据的序列化和反序列化
import json
# 引入 os 模块，提供操作系统相关的功能
import os
# 引入 tempfile 模块，用于创建临时文件和目录
import tempfile
# 引入 time 模块，提供时间相关的功能，如 sleep()
import time
# 引入 datetime 和 timedelta 类，用于处理日期和时间
from datetime import datetime, timedelta
# 从 unittest.mock 模块中导入 patch 函数，用于模拟对象和函数
from unittest.mock import patch

# 引入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 dbgpt.util.tracer 模块中导入以下类和枚举
from dbgpt.util.tracer import (
    FileSpanStorage,    # 文件存储跨度的类
    Span,               # 跨度对象的类
    SpanStorage,        # 跨度存储的接口类
    SpanStorageContainer,   # 跨度存储容器类
    SpanType,           # 跨度类型的枚举类
)


@pytest.fixture
def storage(request):
    # 检查请求是否存在或是否有 "param" 属性
    if not request or not hasattr(request, "param"):
        file_does_not_exist = False
    else:
        file_does_not_exist = request.param.get("file_does_not_exist", False)

    # 如果文件不存在，则使用临时目录创建一个临时文件来初始化存储
    if file_does_not_exist:
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "non_existent_file.jsonl")
            storage_instance = FileSpanStorage(filename)
            yield storage_instance
    else:
        # 否则，使用具有自动删除特性的命名临时文件来初始化存储
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            filename = tmp_file.name
            storage_instance = FileSpanStorage(filename)
            yield storage_instance


@pytest.fixture
def storage_container(request):
    # 检查请求是否存在或是否有 "param" 属性
    if not request or not hasattr(request, "param"):
        batch_size = 10
        flush_interval = 10
    else:
        # 从请求参数中获取批量大小和刷新间隔
        batch_size = request.param.get("batch_size", 10)
        flush_interval = request.param.get("flush_interval", 10)
    
    # 初始化一个 SpanStorageContainer 实例作为存储容器
    storage_container = SpanStorageContainer(
        batch_size=batch_size, flush_interval=flush_interval
    )
    yield storage_container


def read_spans_from_file(filename):
    # 打开指定文件，逐行读取 JSON 数据并返回为列表
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def test_write_span(storage: SpanStorage):
    # 创建一个 Span 对象并将其追加到存储中
    span = Span("1", "a", SpanType.BASE, "b", "op1")
    storage.append_span(span)
    # 等待一段时间，确保写入操作完成
    time.sleep(0.1)

    # 从存储的文件中读取跨度数据，并进行断言验证
    spans_in_file = read_spans_from_file(storage.filename)
    assert len(spans_in_file) == 1
    assert spans_in_file[0]["trace_id"] == "1"


def test_incremental_write(storage: SpanStorage):
    # 创建两个 Span 对象并依次追加到存储中
    span1 = Span("1", "a", SpanType.BASE, "b", "op1")
    span2 = Span("2", "c", SpanType.BASE, "d", "op2")

    storage.append_span(span1)
    storage.append_span(span2)
    # 等待一段时间，确保写入操作完成
    time.sleep(0.1)

    # 从存储的文件中读取跨度数据，并进行断言验证
    spans_in_file = read_spans_from_file(storage.filename)
    assert len(spans_in_file) == 2


def test_sync_and_async_append(storage: SpanStorage):
    # 创建一个 Span 对象并同时进行同步和异步方式的追加操作
    span = Span("1", "a", SpanType.BASE, "b", "op1")

    storage.append_span(span)

    # 定义异步函数，在其中进行追加操作
    async def async_append():
        storage.append_span(span)

    # 运行异步函数
    asyncio.run(async_append())

    # 等待一段时间，确保所有的写入操作完成
    time.sleep(0.1)
    
    # 从存储的文件中读取跨度数据，并进行断言验证
    spans_in_file = read_spans_from_file(storage.filename)
    assert len(spans_in_file) == 2


@pytest.mark.parametrize("storage", [{"file_does_not_exist": True}], indirect=True)
def test_non_existent_file(storage: SpanStorage):
    # 在存储中追加两个 Span 对象，并对文件存在性参数进行测试
    span = Span("1", "a", SpanType.BASE, "b", "op1")
    span2 = Span("2", "c", SpanType.BASE, "d", "op2")
    storage.append_span(span)
    time.sleep(0.1)

    # 读取存储的文件中的跨度数据，并进行断言验证
    spans_in_file = read_spans_from_file(storage.filename)
    assert len(spans_in_file) == 1

    # 再次向存储中追加一个 Span 对象，并验证其写入结果
    storage.append_span(span2)
    time.sleep(0.1)
    spans_in_file = read_spans_from_file(storage.filename)
    # 断言确保 spans_in_file 列表长度为 2
    assert len(spans_in_file) == 2
    # 断言确保 spans_in_file 列表中第一个元素的 "trace_id" 键为 "1"
    assert spans_in_file[0]["trace_id"] == "1"
    # 断言确保 spans_in_file 列表中第二个元素的 "trace_id" 键为 "2"
    assert spans_in_file[1]["trace_id"] == "2"
# 使用 pytest 的参数化装饰器，指定参数为 {"file_does_not_exist": True}，并且间接调用 fixture
@pytest.mark.parametrize("storage", [{"file_does_not_exist": True}], indirect=True)
# 定义测试函数 test_log_rollover，接受名为 storage 的 SpanStorage 类型参数
def test_log_rollover(storage: SpanStorage):
    # 模拟起始日期为 2023 年 10 月 18 日 23 时 59 分
    mock_start_date = datetime(2023, 10, 18, 23, 59)

    # 使用 patch 函数将 datetime.datetime 替换为 mock_datetime，并设定返回值为 mock_start_date
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_start_date

        # 创建 Span 对象 span1，并将其添加到 storage
        span1 = Span("1", "a", SpanType.BASE, "b", "op1")
        storage.append_span(span1)
        # 等待 0.1 秒
        time.sleep(0.1)

        # 模拟进入新的一天
        mock_datetime.now.return_value = mock_start_date + timedelta(minutes=1)

        # 创建 Span 对象 span2，并将其添加到 storage
        span2 = Span("2", "c", SpanType.BASE, "d", "op2")
        storage.append_span(span2)
        # 等待 0.1 秒

    # 断言原始文件名 storage.filename 应存在于文件系统中
    assert os.path.exists(storage.filename)

    # 构建滚动文件名 dated_filename，格式为原始文件名加上日期后缀
    dated_filename = os.path.join(
        os.path.dirname(storage.filename),
        f"{os.path.basename(storage.filename).split('.')[0]}_2023-10-18.jsonl",
    )

    # 断言滚动文件名 dated_filename 应存在于文件系统中
    assert os.path.exists(dated_filename)

    # 从原始文件 storage.filename 中读取 Span 列表 spans_in_original_file
    spans_in_original_file = read_spans_from_file(storage.filename)
    # 断言 spans_in_original_file 的长度为 1
    assert len(spans_in_original_file) == 1
    # 断言 spans_in_original_file 的第一个元素的 trace_id 为 "2"
    assert spans_in_original_file[0]["trace_id"] == "2"

    # 从滚动文件 dated_filename 中读取 Span 列表 spans_in_dated_file
    spans_in_dated_file = read_spans_from_file(dated_filename)
    # 断言 spans_in_dated_file 的长度为 1
    assert len(spans_in_dated_file) == 1
    # 断言 spans_in_dated_file 的第一个元素的 trace_id 为 "1"


# 使用 pytest 的 asyncio 参数化装饰器
@pytest.mark.asyncio
# 参数化 fixture storage_container，设置 batch_size 为 5，间接调用
@pytest.mark.parametrize("storage_container", [{"batch_size": 5}], indirect=True)
# 异步测试函数 test_container_flush_policy，接受 storage_container 和 storage 两个参数
async def test_container_flush_policy(
    storage_container: SpanStorageContainer, storage: FileSpanStorage
):
    # 将 storage 添加到 storage_container
    storage_container.append_storage(storage)
    # 创建 Span 对象 span，trace_id 为 "1"
    span = Span("1", "a", SpanType.BASE, "b", "op1")

    # 获取 storage 的文件名 filename
    filename = storage.filename

    # 循环执行 storage_container.batch_size - 1 次，向 storage_container 中添加 span
    for _ in range(storage_container.batch_size - 1):
        storage_container.append_span(span)

    # 从文件 filename 中读取 Span 列表 spans_in_file
    spans_in_file = read_spans_from_file(filename)
    # 断言 spans_in_file 的长度为 0
    assert len(spans_in_file) == 0

    # 触发批量写入
    storage_container.append_span(span)
    # 等待 0.1 秒
    await asyncio.sleep(0.1)

    # 重新读取文件 filename 中的 Span 列表 spans_in_file
    spans_in_file = read_spans_from_file(filename)
    # 断言 spans_in_file 的长度为 storage_container.batch_size
    assert len(spans_in_file) == storage_container.batch_size
```