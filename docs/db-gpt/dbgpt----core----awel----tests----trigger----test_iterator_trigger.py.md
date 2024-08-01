# `.\DB-GPT-src\dbgpt\core\awel\tests\trigger\test_iterator_trigger.py`

```py
# 导入必要的模块
from typing import AsyncIterator
import pytest
from dbgpt.core.awel import (
    DAG,
    InputSource,
    MapOperator,
    StreamifyAbsOperator,
    TransformStreamAbsOperator,
)
from dbgpt.core.awel.trigger.iterator_trigger import IteratorTrigger

# 定义一个流式操作符，用于生成从0到n-1的数字流
class NumberProducerOperator(StreamifyAbsOperator[int, int]):
    """Create a stream of numbers from 0 to `n-1`"""
    async def streamify(self, n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

# 定义一个自定义的流式操作符，用于对输入流中的数据进行平方操作
class MyStreamingOperator(TransformStreamAbsOperator[int, int]):
    async def transform_stream(self, data: AsyncIterator[int]) -> AsyncIterator[int]:
        async for i in data:
            yield i * i

# 辅助函数，用于检查流式操作的结果
async def _check_stream_results(stream_results, expected_len):
    assert len(stream_results) == expected_len
    for _, result in stream_results:
        i = 0
        async for num in result:
            assert num == i * i
            i += 1

# 测试用例1：测试单个数据
@pytest.mark.asyncio
async def test_single_data():
    # 创建一个DAG
    with DAG("test_single_data"):
        trigger_task = IteratorTrigger(data=2)
        task = MapOperator(lambda x: x * x)
        trigger_task >> task
    results = await trigger_task.trigger()
    assert len(results) == 1
    assert results[0][1] == 4

    # 创建另一个DAG，测试流式数据
    with DAG("test_single_data_stream"):
        trigger_task = IteratorTrigger(data=2, streaming_call=True)
        number_task = NumberProducerOperator()
        task = MyStreamingOperator()
        trigger_task >> number_task >> task
    stream_results = await trigger_task.trigger()
    await _check_stream_results(stream_results, 1)

# 测试用例2：测试列表数据
@pytest.mark.asyncio
async def test_list_data():
    with DAG("test_list_data"):
        trigger_task = IteratorTrigger(data=[0, 1, 2, 3])
        task = MapOperator(lambda x: x * x)
        trigger_task >> task
    results = await trigger_task.trigger()
    assert len(results) == 4
    assert results == [(0, 0), (1, 1), (2, 4), (3, 9)]

    with DAG("test_list_data_stream"):
        trigger_task = IteratorTrigger(data=[0, 1, 2, 3], streaming_call=True)
        number_task = NumberProducerOperator()
        task = MyStreamingOperator()
        trigger_task >> number_task >> task
    stream_results = await trigger_task.trigger()
    await _check_stream_results(stream_results, 4)

# 测试用例3：测试异步迭代器数据
@pytest.mark.asyncio
async def test_async_iterator_data():
    async def async_iter():
        for i in range(4):
            yield i

    with DAG("test_async_iterator_data"):
        trigger_task = IteratorTrigger(data=async_iter())
        task = MapOperator(lambda x: x * x)
        trigger_task >> task
    results = await trigger_task.trigger()
    assert len(results) == 4
    assert results == [(0, 0), (1, 1), (2, 4), (3, 9)]

    with DAG("test_async_iterator_data_stream"):
        trigger_task = IteratorTrigger(data=async_iter(), streaming_call=True)
        number_task = NumberProducerOperator()
        task = MyStreamingOperator()
        trigger_task >> number_task >> task
    # 等待异步任务 `trigger_task.trigger()` 完成并获取其结果，将结果赋给 `stream_results`
    stream_results = await trigger_task.trigger()
    
    # 调用异步函数 `_check_stream_results`，传入 `stream_results` 和参数 `4`，并等待其完成
    await _check_stream_results(stream_results, 4)
# 使用 pytest 的 asyncio 标记来定义异步测试函数
@pytest.mark.asyncio
async def test_input_source_data():
    # 创建一个 DAG 对象，用于组织任务流程
    with DAG("test_input_source_data"):
        # 创建一个迭代器触发器任务，从可迭代对象中获取数据
        trigger_task = IteratorTrigger(data=InputSource.from_iterable([0, 1, 2, 3]))
        # 创建一个映射操作任务，对数据进行平方操作
        task = MapOperator(lambda x: x * x)
        # 设置任务之间的依赖关系
        trigger_task >> task
    # 触发任务流程并获取结果
    results = await trigger_task.trigger()
    # 断言结果的长度为4
    assert len(results) == 4
    # 断言结果与预期相符
    assert results == [(0, 0), (1, 1), (2, 4), (3, 9)]

    # 创建另一个 DAG 对象，用于组织任务流程
    with DAG("test_input_source_data_stream"):
        # 创建一个迭代器触发器任务，从可迭代对象中获取数据，并设置为流式调用
        trigger_task = IteratorTrigger(
            data=InputSource.from_iterable([0, 1, 2, 3]),
            streaming_call=True,
        )
        # 创建一个数字生产操作任务
        number_task = NumberProducerOperator()
        # 创建一个自定义的流式操作任务
        task = MyStreamingOperator()
        # 设置任务之间的依赖关系
        trigger_task >> number_task >> task
    # 触发任务流程并获取结果
    stream_results = await trigger_task.trigger()
    # 检查流式结果
    await _check_stream_results(stream_results, 4)


# 使用 pytest 的 asyncio 标记来定义异步测试函数
@pytest.mark.asyncio
async def test_parallel_safe():
    # 创建一个 DAG 对象，用于组织任务流程
    with DAG("test_parallel_safe"):
        # 创建一个迭代器触发器任务，从可迭代对象中获取数据
        trigger_task = IteratorTrigger(data=InputSource.from_iterable([0, 1, 2, 3]))
        # 创建一个映射操作任务，对数据进行平方操作
        task = MapOperator(lambda x: x * x)
        # 设置任务之间的依赖关系
        trigger_task >> task
    # 触发任务流程并获取结果，设置并行数为3
    results = await trigger_task.trigger(parallel_num=3)
    # 断言结果的长度为4
    assert len(results) == 4
    # 断言结果与预期相符
    assert results == [(0, 0), (1, 1), (2, 4), (3, 9)]

    # 创建另一个 DAG 对象，用于组织任务流程
    with DAG("test_input_source_data_stream"):
        # 创建一个迭代器触发器任务，从可迭代对象中获取数据，并设置为流式调用
        trigger_task = IteratorTrigger(
            data=InputSource.from_iterable([0, 1, 2, 3]),
            streaming_call=True,
        )
        # 创建一个数字生产操作任务
        number_task = NumberProducerOperator()
        # 创建一个自定义的流式操作任务
        task = MyStreamingOperator()
        # 设置任务之间的依赖关系
        trigger_task >> number_task >> task
    # 触发任务流程并获取结果，设置并行数为3
    stream_results = await trigger_task.trigger(parallel_num=3)
    # 检查流式结果
    await _check_stream_results(stream_results, 4)
```