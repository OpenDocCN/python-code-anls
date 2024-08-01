# `.\DB-GPT-src\dbgpt\core\awel\tests\conftest.py`

```py
# 导入必要的模块和函数
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, List
import pytest
import pytest_asyncio

# 导入需要的类和函数
from .. import (
    DAGContext,
    DefaultWorkflowRunner,
    InputOperator,
    SimpleInputSource,
    TaskState,
    WorkflowRunner,
)
from ..task.task_impl import _is_async_iterator

# 定义一个 pytest fixture，返回默认的工作流运行器
@pytest.fixture
def runner():
    return DefaultWorkflowRunner()

# 定义一个函数，创建包含多个异步迭代器的列表
def _create_stream(num_nodes) -> List[AsyncIterator[int]]:
    iters = []
    for _ in range(num_nodes):
        # 定义一个异步生成器函数，生成从 0 到 9 的整数序列
        async def stream_iter():
            for i in range(10):
                yield i

        # 调用异步生成器函数，获取异步迭代器对象
        stream_iter = stream_iter()
        # 断言确保对象是异步迭代器
        assert _is_async_iterator(stream_iter)
        iters.append(stream_iter)
    return iters

# 定义一个函数，从给定的二维整数列表中创建包含多个异步迭代器的列表
def _create_stream_from(output_streams: List[List[int]]) -> List[AsyncIterator[int]]:
    iters = []
    for single_stream in output_streams:
        # 定义一个异步生成器函数，生成单个列表中的整数序列
        async def stream_iter():
            for i in single_stream:
                yield i

        # 调用异步生成器函数，获取异步迭代器对象
        stream_iter = stream_iter()
        # 断言确保对象是异步迭代器
        assert _is_async_iterator(stream_iter)
        iters.append(stream_iter)
    return iters

# 定义一个异步上下文管理器，根据参数创建输入节点
@asynccontextmanager
async def _create_input_node(**kwargs):
    num_nodes = kwargs.get("num_nodes")
    is_stream = kwargs.get("is_stream", False)
    
    # 如果需要创建流式输入节点
    if is_stream:
        outputs = kwargs.get("output_streams")
        if outputs:
            # 检查节点数与输出流列表长度是否匹配
            if num_nodes and num_nodes != len(outputs):
                raise ValueError(
                    f"num_nodes {num_nodes} != the length of output_streams {len(outputs)}"
                )
            # 从二维列表创建多个异步迭代器
            outputs = _create_stream_from(outputs)
        else:
            # 如果没有输出流列表，创建默认数量的异步迭代器
            num_nodes = num_nodes or 1
            outputs = _create_stream(num_nodes)
    else:
        # 如果不是流式输入节点，使用默认输出列表
        outputs = kwargs.get("outputs", ["Hello."])
    
    nodes = []
    for i, output in enumerate(outputs):
        print(f"output: {output}")
        # 创建输入源对象
        input_source = SimpleInputSource(output)
        # 创建输入操作符对象，带有任务ID
        input_node = InputOperator(input_source, task_id="input_node_" + str(i))
        nodes.append(input_node)
    
    # 返回输入节点列表作为上下文管理器的结果
    yield nodes

# 定义一个 pytest-asyncio fixture，用于单个输入节点
@pytest_asyncio.fixture
async def input_node(request):
    param = getattr(request, "param", {})
    async with _create_input_node(**param) as input_nodes:
        yield input_nodes[0]

# 定义一个 pytest-asyncio fixture，用于流式输入节点
@pytest_asyncio.fixture
async def stream_input_node(request):
    param = getattr(request, "param", {})
    param["is_stream"] = True
    async with _create_input_node(**param) as input_nodes:
        yield input_nodes[0]

# 定义一个 pytest-asyncio fixture，用于多个输入节点
@pytest_asyncio.fixture
async def input_nodes(request):
    param = getattr(request, "param", {})
    async with _create_input_node(**param) as input_nodes:
        yield input_nodes

# 定义一个 pytest-asyncio fixture，用于多个流式输入节点
@pytest_asyncio.fixture
async def stream_input_nodes(request):
    param = getattr(request, "param", {})
    param["is_stream"] = True
    async with _create_input_node(**param) as input_nodes:
        yield input_nodes
```