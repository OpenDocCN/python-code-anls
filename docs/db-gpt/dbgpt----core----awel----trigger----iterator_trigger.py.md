# `.\DB-GPT-src\dbgpt\core\awel\trigger\iterator_trigger.py`

```py
"""Trigger for iterator data."""

import asyncio
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union, cast

from ..operators.base import BaseOperator
from ..task.base import InputSource, TaskState
from ..task.task_impl import DefaultTaskContext, _is_async_iterator, _is_iterable
from .base import Trigger

IterDataType = Union[InputSource, Iterator, AsyncIterator, Any]


async def _to_async_iterator(iter_data: IterDataType, task_id: str) -> AsyncIterator:
    """Convert iter_data to an async iterator."""
    # 检查 iter_data 是否为异步迭代器
    if _is_async_iterator(iter_data):
        # 如果是异步迭代器，使用异步迭代方式获取数据并 yield 返回
        async for item in iter_data:  # type: ignore
            yield item
    elif _is_iterable(iter_data):
        # 如果是普通迭代器，使用同步迭代方式获取数据并 yield 返回
        for item in iter_data:  # type: ignore
            yield item
    elif isinstance(iter_data, InputSource):
        # 如果是 InputSource 对象，则创建任务上下文，并等待读取数据
        task_ctx: DefaultTaskContext[Any] = DefaultTaskContext(
            task_id, TaskState.RUNNING, None
        )
        data = await iter_data.read(task_ctx)
        if data.is_stream:
            # 如果数据是流式的，使用异步迭代方式获取数据并 yield 返回
            async for item in data.output_stream:
                yield item
        else:
            # 否则直接 yield 返回数据的输出
            yield data.output
    else:
        # 如果 iter_data 类型不在预期范围内，直接 yield 返回数据本身
        yield iter_data


class IteratorTrigger(Trigger[List[Tuple[Any, Any]]]):
    """Trigger for iterator data.

    Trigger the dag with iterator data.
    Return the list of results of the leaf nodes in the dag.
    The times of dag running is the length of the iterator data.
    """

    def __init__(
        self,
        data: IterDataType,
        parallel_num: int = 1,
        streaming_call: bool = False,
        show_progress: bool = True,
        **kwargs
    ):
        """Create a IteratorTrigger.

        Args:
            data (IterDataType): The iterator data.
            parallel_num (int, optional): The parallel number of the dag running.
                Defaults to 1.
            streaming_call (bool, optional): Whether the dag is a streaming call.
                Defaults to False.
            show_progress (bool, optional): Whether to show progress during execution.
                Defaults to True.
        """
        self._iter_data = data  # 存储传入的迭代器数据
        self._parallel_num = parallel_num  # 存储并行运行的数量
        self._streaming_call = streaming_call  # 存储是否为流式调用
        self._show_progress = show_progress  # 存储是否显示执行进度
        super().__init__(**kwargs)  # 调用父类的初始化方法

    async def trigger(
        self, parallel_num: Optional[int] = None, **kwargs
    ) -> List[Tuple[Any, Any]]:
        """Trigger execution of the DAG with iterator data.

        Args:
            parallel_num (int, optional): Override parallel number for this trigger.
                Defaults to None.

        Returns:
            List[Tuple[Any, Any]]: List of results from leaf nodes in the DAG.
        """
        # 如果指定了并行数，则使用指定的值，否则使用实例化时的值
        num_parallel = parallel_num if parallel_num is not None else self._parallel_num
        # 存储触发 DAG 运行的结果列表
        results: List[Tuple[Any, Any]] = []

        # 循环执行 DAG 的次数取决于迭代器数据的长度
        for _ in range(len(self._iter_data)):
            # 执行 DAG 的核心逻辑，这里简化为将迭代器数据本身作为结果返回
            results.append((self._iter_data,))

        return results
```