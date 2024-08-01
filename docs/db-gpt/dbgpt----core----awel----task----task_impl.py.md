# `.\DB-GPT-src\dbgpt\core\awel\task\task_impl.py`

```py
"""The default implementation of Task.

This implementation can run workflow in local machine.
"""
# 导入必要的模块和库
import asyncio  # 异步编程库
import logging  # 日志记录模块
from abc import ABC, abstractmethod  # 抽象基类和抽象方法装饰器
from typing import (
    Any,  # 任意类型
    AsyncIterator,  # 异步迭代器类型
    Callable,  # 可调用对象类型
    Coroutine,  # 协程类型
    Dict,  # 字典类型
    Generic,  # 泛型类
    List,  # 列表类型
    Optional,  # 可选类型
    Tuple,  # 元组类型
    Union,  # 联合类型
    cast,  # 类型强制转换函数
)

from .base import (  # 导入自定义模块中的相关类和函数
    _EMPTY_DATA_TYPE,  # 空数据类型
    EMPTY_DATA,  # 空数据对象
    OUT,  # 输出对象标记
    PLACEHOLDER_DATA,  # 占位数据对象
    InputContext,  # 输入上下文类
    InputSource,  # 输入源类
    MapFunc,  # 映射函数类型
    PredicateFunc,  # 谓词函数类型
    ReduceFunc,  # 减少函数类型
    StreamFunc,  # 流函数类型
    T,  # 泛型类型 T
    TaskContext,  # 任务上下文类
    TaskOutput,  # 任务输出类
    TaskState,  # 任务状态类
    TransformFunc,  # 转换函数类型
    UnStreamFunc,  # 取消流函数类型
    is_empty_data,  # 判断是否为空数据的函数
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


async def _reduce_stream(stream: AsyncIterator, reduce_function) -> Any:
    # 初始化累加器
    try:
        accumulator = await stream.__anext__()  # 获取流的第一个元素作为初始值
    except StopAsyncIteration:
        raise ValueError("Stream is empty")  # 如果流为空则抛出异常
    is_async = asyncio.iscoroutinefunction(reduce_function)  # 判断是否为异步函数
    async for element in stream:
        if is_async:
            accumulator = await reduce_function(accumulator, element)
        else:
            accumulator = reduce_function(accumulator, element)
    return accumulator  # 返回累加器的最终值


class SimpleTaskOutput(TaskOutput[T], Generic[T]):
    """The default implementation of TaskOutput.

    It wraps the no stream data and provide some basic data operations.
    """

    def __init__(self, data: Union[T, _EMPTY_DATA_TYPE] = EMPTY_DATA) -> None:
        """Create a SimpleTaskOutput.

        Args:
            data (Union[T, _EMPTY_DATA_TYPE], optional): The output data. Defaults to
                EMPTY_DATA.
        """
        super().__init__()
        self._data = data  # 初始化存储数据的私有属性

    @property
    def output(self) -> T:
        """Return the output data."""
        if EMPTY_DATA.is_same(self._data):
            raise ValueError("No output data for current task output")  # 如果数据为空则抛出异常
        return cast(T, self._data)  # 返回存储的数据

    def set_output(self, output_data: T | AsyncIterator[T]) -> None:
        """Save the output data to current object.

        Args:
            output_data (T | AsyncIterator[T]): The output data.
        """
        if _is_async_iterator(output_data):
            raise ValueError(
                f"Can not set stream data {output_data} to SimpleTaskOutput"
            )  # 如果是异步迭代器则抛出异常
        self._data = cast(T, output_data)  # 设置存储的数据

    def new_output(self) -> TaskOutput[T]:
        """Create new output object with empty data."""
        return SimpleTaskOutput()  # 返回一个新的空输出对象

    @property
    def is_empty(self) -> bool:
        """Return True if the output data is empty."""
        return is_empty_data(self._data)  # 判断数据是否为空

    @property
    def is_none(self) -> bool:
        """Return True if the output data is None."""
        return self._data is None  # 判断数据是否为 None

    async def _apply_func(self, func) -> Any:
        """Apply the function to current output data."""
        if asyncio.iscoroutinefunction(func):
            out = await func(self._data)  # 如果是异步函数，则使用 await 调用
        else:
            out = func(self._data)  # 否则同步调用
        return out  # 返回函数应用后的结果
    # 将给定的映射函数应用于任务的输出。
    async def map(self, map_func: MapFunc) -> TaskOutput[OUT]:
        """Apply a mapping function to the task's output.

        Args:
            map_func (MapFunc): A function to apply to the task's output.

        Returns:
            TaskOutput[OUT]: The result of applying the mapping function.
        """
        # 调用 _apply_func 方法，将映射函数应用于任务的输出
        out = await self._apply_func(map_func)
        # 返回一个简单任务输出对象，包含应用映射函数后的结果
        return SimpleTaskOutput(out)

    # 检查条件函数的执行结果，并返回相应的任务输出对象
    async def check_condition(self, condition_func) -> TaskOutput[OUT]:
        """Check the condition function."""
        # 调用 _apply_func 方法，检查条件函数的执行结果
        out = await self._apply_func(condition_func)
        # 如果条件函数返回 True，则返回一个包含占位数据的简单任务输出对象
        if out:
            return SimpleTaskOutput(PLACEHOLDER_DATA)
        # 否则返回一个包含空数据的简单任务输出对象
        return SimpleTaskOutput(EMPTY_DATA)

    # 将任务的输出转换为流输出，并返回相应的任务输出对象
    async def streamify(self, transform_func: StreamFunc) -> TaskOutput[OUT]:
        """Transform the task's output to a stream output.

        Args:
            transform_func (StreamFunc): A function to transform the task's output to a
                stream output.

        Returns:
            TaskOutput[OUT]: The result of transforming the task's output to a stream
                output.
        """
        # 调用 _apply_func 方法，将转换函数应用于任务的输出
        out = await self._apply_func(transform_func)
        # 返回一个简单流任务输出对象，包含应用转换函数后的结果
        return SimpleStreamTaskOutput(out)
class SimpleStreamTaskOutput(TaskOutput[T], Generic[T]):
    """The default stream implementation of TaskOutput."""

    def __init__(
        self, data: Union[AsyncIterator[T], _EMPTY_DATA_TYPE] = EMPTY_DATA
    ) -> None:
        """Create a SimpleStreamTaskOutput.

        Args:
            data (Union[AsyncIterator[T], _EMPTY_DATA_TYPE], optional): The output data.
                Defaults to EMPTY_DATA.
        """
        super().__init__()
        self._data = data  # 初始化对象的输出数据，可以是异步迭代器或空数据类型

    @property
    def is_stream(self) -> bool:
        """Return True if the output data is a stream."""
        return True  # 判断输出数据是否为流数据，始终返回True

    @property
    def is_empty(self) -> bool:
        """Return True if the output data is empty."""
        return is_empty_data(self._data)  # 判断输出数据是否为空，调用辅助函数is_empty_data

    @property
    def is_none(self) -> bool:
        """Return True if the output data is None."""
        return self._data is None  # 判断输出数据是否为None，始终返回True或False

    @property
    def output_stream(self) -> AsyncIterator[T]:
        """Return the output data.

        Returns:
            AsyncIterator[T]: The output data.

        Raises:
            ValueError: If the output data is empty.
        """
        if EMPTY_DATA.is_same(self._data):
            raise ValueError("No output data for current task output")  # 如果输出数据为空，则抛出异常
        return cast(AsyncIterator[T], self._data)  # 返回异步迭代器形式的输出数据

    def set_output(self, output_data: T | AsyncIterator[T]) -> None:
        """Save the output data to current object.

        Raises:
            ValueError: If the output data is not a stream.
        """
        if not _is_async_iterator(output_data):
            raise ValueError(
                f"Can not set non-stream data {output_data} to SimpleStreamTaskOutput"
            )  # 如果设置的输出数据不是异步迭代器，则抛出异常
        self._data = cast(AsyncIterator[T], output_data)  # 将输出数据设置为异步迭代器形式

    def new_output(self) -> TaskOutput[T]:
        """Create new output object with empty data."""
        return SimpleStreamTaskOutput()  # 创建一个新的SimpleStreamTaskOutput对象，数据为空

    async def map(self, map_func: MapFunc) -> TaskOutput[OUT]:
        """Apply a mapping function to the task's output."""
        is_async = asyncio.iscoroutinefunction(map_func)  # 检查传入的映射函数是否是异步函数

        async def new_iter() -> AsyncIterator[OUT]:
            async for out in self.output_stream:
                if is_async:
                    new_out: OUT = await map_func(out)  # 如果是异步函数，使用await调用映射函数
                else:
                    new_out = cast(OUT, map_func(out))  # 否则直接调用映射函数
                yield new_out

        return SimpleStreamTaskOutput(new_iter())  # 返回一个新的SimpleStreamTaskOutput对象，应用映射函数后的数据

    async def reduce(self, reduce_func: ReduceFunc) -> TaskOutput[OUT]:
        """Apply a reduce function to the task's output."""
        out = await _reduce_stream(self.output_stream, reduce_func)  # 调用_reduce_stream函数，应用归约函数
        return SimpleTaskOutput(out)  # 返回一个新的SimpleTaskOutput对象，包含归约后的输出数据
    async def unstreamify(self, transform_func: UnStreamFunc) -> TaskOutput[OUT]:
        """Transform the task's output to a non-stream output."""
        # 检查 transform_func 是否是协程函数
        if asyncio.iscoroutinefunction(transform_func):
            # 如果是协程函数，等待使用输出流作为参数调用它
            out = await transform_func(self.output_stream)
        else:
            # 如果不是协程函数，直接使用输出流作为参数调用 transform_func
            out = transform_func(self.output_stream)
        # 将输出包装成 SimpleTaskOutput 对象并返回
        return SimpleTaskOutput(out)

    async def transform_stream(self, transform_func: TransformFunc) -> TaskOutput[OUT]:
        """Transform an AsyncIterator[T] to another AsyncIterator[T].

        Args:
            transform_func (Callable[[AsyncIterator[T]], AsyncIterator[T]]): Function to
                 apply to the AsyncIterator[T].

        Returns:
            TaskOutput[T]: The result of applying the reducing function.
        """
        # 检查 transform_func 是否是协程函数
        if asyncio.iscoroutinefunction(transform_func):
            # 如果是协程函数，等待使用输出流作为参数调用它，并将结果声明为 AsyncIterator[OUT]
            out: AsyncIterator[OUT] = await transform_func(self.output_stream)
        else:
            # 如果不是协程函数，使用输出流作为参数调用 transform_func，并将结果强制转换为 AsyncIterator[OUT]
            out = cast(AsyncIterator[OUT], transform_func(self.output_stream))
        # 将输出包装成 SimpleStreamTaskOutput 对象并返回
        return SimpleStreamTaskOutput(out)
# 检查对象是否是异步迭代器
def _is_async_iterator(obj):
    return (
        hasattr(obj, "__anext__")  # 检查对象是否有 __anext__ 方法
        and callable(getattr(obj, "__anext__", None))  # 确保 __anext__ 方法是可调用的
        and hasattr(obj, "__aiter__")  # 检查对象是否有 __aiter__ 方法
        and callable(getattr(obj, "__aiter__", None))  # 确保 __aiter__ 方法是可调用的
    )


# 检查对象是否是异步可迭代的
def _is_async_iterable(obj):
    return hasattr(obj, "__aiter__")  # 检查对象是否有 __aiter__ 方法
           and callable(getattr(obj, "__aiter__", None))  # 确保 __aiter__ 方法是可调用的


# 检查对象是否是普通迭代器
def _is_iterator(obj):
    return (
        hasattr(obj, "__iter__")  # 检查对象是否有 __iter__ 方法
        and callable(getattr(obj, "__iter__", None))  # 确保 __iter__ 方法是可调用的
        and hasattr(obj, "__next__")  # 检查对象是否有 __next__ 方法
        and callable(getattr(obj, "__next__", None))  # 确保 __next__ 方法是可调用的
    )


# 检查对象是否是普通可迭代的
def _is_iterable(obj):
    return hasattr(obj, "__iter__")  # 检查对象是否有 __iter__ 方法
           and callable(getattr(obj, "__iter__", None))  # 确保 __iter__ 方法是可调用的


# 将对象转换为异步迭代器
async def _to_async_iterator(obj) -> AsyncIterator:
    if _is_async_iterable(obj):  # 如果对象是异步可迭代的
        async for item in obj:  # 异步迭代对象
            yield item  # 返回每个迭代项
    elif _is_iterable(obj):  # 如果对象是普通可迭代的
        for item in obj:  # 迭代对象
            yield item  # 返回每个迭代项
    else:
        raise ValueError(f"Can not convert {obj} to AsyncIterator")  # 抛出值错误异常，无法转换为异步迭代器


class BaseInputSource(InputSource, ABC):
    """The base class of InputSource."""

    def __init__(self, streaming: Optional[bool] = None) -> None:
        """Create a BaseInputSource."""
        super().__init__()
        self._is_read = False  # 初始化 _is_read 标志为 False，表示尚未读取
        self._streaming_data = streaming  # 初始化 _streaming_data 属性为传入的流数据选项

    @abstractmethod
    def _read_data(self, task_ctx: TaskContext) -> Any:
        """Return data with task context."""

    async def read(self, task_ctx: TaskContext) -> TaskOutput:
        """Read data with task context.

        Args:
            task_ctx (TaskContext): The task context.

        Returns:
            TaskOutput: The task output.

        Raises:
            ValueError: If the input source is a stream and has been read.
        """
        data = self._read_data(task_ctx)  # 调用 _read_data 方法读取数据
        if self._streaming_data is None:
            streaming_data = _is_async_iterator(data) or _is_iterator(data)  # 检查数据是否是流数据（异步或普通迭代器）
        else:
            streaming_data = self._streaming_data  # 使用初始化时设置的流数据选项

        if streaming_data:  # 如果数据是流数据
            if self._is_read:
                raise ValueError(f"Input iterator {data} has been read!")  # 如果已经读取过则抛出值错误异常
            it_data = _to_async_iterator(data)  # 将数据转换为异步迭代器
            output: TaskOutput = SimpleStreamTaskOutput(it_data)  # 使用异步迭代器创建简单流任务输出对象
        else:  # 如果数据不是流数据
            output = SimpleTaskOutput(data)  # 创建简单任务输出对象

        self._is_read = True  # 将 _is_read 标志设置为 True，表示已经读取过数据
        return output  # 返回任务输出对象


class SimpleInputSource(BaseInputSource):
    """The default implementation of InputSource."""

    def __init__(self, data: Any, streaming: Optional[bool] = None) -> None:
        """Create a SimpleInputSource.

        Args:
            data (Any): The input data.
        """
        super().__init__(streaming=streaming)  # 调用父类初始化方法
        self._data = data  # 设置对象的输入数据

    def _read_data(self, task_ctx: TaskContext) -> Any:
        return self._data  # 直接返回对象的输入数据


class SimpleCallDataInputSource(BaseInputSource):
    """The implementation of InputSource for call data."""

    def __init__(self) -> None:
        """Create a SimpleCallDataInputSource."""
        super().__init__()  # 调用父类初始化方法
    def _read_data(self, task_ctx: TaskContext) -> Any:
        """Read data from task context.

        Returns:
            Any: The data.

        Raises:
            ValueError: If the call data is empty.
        """
        # 从任务上下文中获取调用数据
        call_data = task_ctx.call_data
        # 获取键为"data"的数据，如果不存在则使用EMPTY_DATA
        data = call_data.get("data", EMPTY_DATA) if call_data else EMPTY_DATA
        # 检查获取的数据是否为空
        if is_empty_data(data):
            # 如果数据为空，则抛出值错误异常
            raise ValueError("No call data for current SimpleCallDataInputSource")
        # 返回读取到的数据
        return data
class DefaultTaskContext(TaskContext, Generic[T]):
    """DefaultTaskContext 类，继承自 TaskContext 类，泛型类型为 T"""

    def __init__(
        self,
        task_id: str,
        task_state: TaskState,
        task_output: Optional[TaskOutput[T]] = None,
        log_index: int = 0,
    ) -> None:
        """初始化 DefaultTaskContext 对象。

        Args:
            task_id (str): 任务 ID。
            task_state (TaskState): 任务状态。
            task_output (Optional[TaskOutput[T]], optional): 任务输出。默认为 None。
            log_index (int, optional): 日志索引。默认为 0。
        """
        super().__init__()
        self._task_id = task_id
        self._task_state = task_state
        self._output: Optional[TaskOutput[T]] = task_output
        self._task_input: Optional[InputContext] = None
        self._metadata: Dict[str, Any] = {}
        self._log_index = log_index

    @property
    def task_id(self) -> str:
        """返回任务 ID。"""
        return self._task_id

    @property
    def log_id(self) -> str:
        """返回日志 ID。"""
        return f"{self._task_id}_{self._log_index}"

    @property
    def task_input(self) -> InputContext:
        """返回任务输入。如果任务输入为空，则引发 ValueError 异常。"""
        if not self._task_input:
            raise ValueError("No input for current task context")
        return self._task_input

    def set_task_input(self, input_ctx: InputContext) -> None:
        """保存任务输入到当前任务。"""
        self._task_input = input_ctx

    @property
    def task_output(self) -> TaskOutput:
        """返回任务输出。如果任务输出为空，则引发 ValueError 异常。"""
        if not self._output:
            raise ValueError("No output for current task context")
        return self._output

    def set_task_output(self, task_output: TaskOutput) -> None:
        """保存任务输出到当前任务。"""
        self._output = task_output

    @property
    def current_state(self) -> TaskState:
        """返回当前任务状态。"""
        return self._task_state

    def set_current_state(self, task_state: TaskState) -> None:
        """保存当前任务状态到当前任务。"""
        self._task_state = task_state

    def new_ctx(self) -> TaskContext:
        """创建一个新的任务上下文，输出为空。如果当前任务输出为空，则引发 ValueError 异常。"""
        if not self._output:
            raise ValueError("No output for current task context")
        new_output = self._output.new_output()
        return DefaultTaskContext(
            self._task_id, self._task_state, new_output, self._log_index
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """返回当前任务的元数据。"""
        return self._metadata
    # 异步方法，返回当前任务的调用数据，可能为空
    async def _call_data_to_output(self) -> Optional[TaskOutput[T]]:
        """Return the call data of current task.

        Returns:
            Optional[TaskOutput[T]]: The call data.
        """
        # 获取当前任务的调用数据
        call_data = self.call_data
        # 如果调用数据为空，返回空值
        if not call_data:
            return None
        # 创建简单的调用数据输入源
        input_source = SimpleCallDataInputSource()
        # 使用输入源读取数据，返回结果
        return await input_source.read(self)
class DefaultInputContext(InputContext):
    """The default implementation of InputContext.

    It wraps all inputs from parent tasks and provides some basic data operations.
    """

    def __init__(self, outputs: List[TaskContext]) -> None:
        """Create a DefaultInputContext.

        Args:
            outputs (List[TaskContext]): The outputs from parent tasks.
        """
        super().__init__()
        self._outputs = outputs  # Initialize the _outputs attribute with the provided outputs

    @property
    def parent_outputs(self) -> List[TaskContext]:
        """Return the outputs from parent tasks.

        Returns:
            List[TaskContext]: The outputs from parent tasks.
        """
        return self._outputs  # Return the stored parent task outputs

    async def _apply_func(
        self, func: Callable[[Any], Any], apply_type: str = "map"
    ) -> Tuple[List[TaskContext], List[TaskOutput]]:
        """Apply the function to all parent outputs.

        Args:
            func (Callable[[Any], Any]): The function to apply.
            apply_type (str, optional): The apply type. Defaults to "map".

        Returns:
            Tuple[List[TaskContext], List[TaskOutput]]: The new parent outputs and the
                results of applying the function.
        """
        new_outputs: List[TaskContext] = []  # Initialize an empty list for new task contexts
        map_tasks = []  # Initialize an empty list for coroutine tasks
        for out in self._outputs:  # Iterate over each parent output
            new_outputs.append(out.new_ctx())  # Create a new context for each output
            if apply_type == "map":
                result: Coroutine[Any, Any, TaskOutput[Any]] = out.task_output.map(func)  # Apply 'map' function on task output
            elif apply_type == "reduce":
                reduce_func = cast(ReduceFunc, func)
                result = out.task_output.reduce(reduce_func)  # Apply 'reduce' function on task output
            elif apply_type == "check_condition":
                result = out.task_output.check_condition(func)  # Check condition on task output
            else:
                raise ValueError(f"Unsupport apply type {apply_type}")  # Raise error for unsupported apply types
            map_tasks.append(result)  # Append the result to map_tasks list
        results = await asyncio.gather(*map_tasks)  # Gather results of all coroutine tasks
        return new_outputs, results  # Return new task contexts and results of function applications

    async def map(self, map_func: Callable[[Any], Any]) -> InputContext:
        """Apply a mapping function to all parent outputs."""
        new_outputs, results = await self._apply_func(map_func)  # Apply _apply_func with 'map' type
        for i, task_ctx in enumerate(new_outputs):
            task_ctx = cast(TaskContext, task_ctx)
            task_ctx.set_task_output(results[i])  # Set task output for each new task context
        return DefaultInputContext(new_outputs)  # Return a new DefaultInputContext with updated outputs
    async def map_all(self, map_func: Callable[..., Any]) -> InputContext:
        """Apply a mapping function to all parent outputs.

        The parent outputs will be unpacked and passed to the mapping function.

        Args:
            map_func (Callable[..., Any]): The mapping function.

        Returns:
            InputContext: The new input context.
        """
        if not self._outputs:
            # 如果没有父输出，则返回一个包含空列表的默认输入上下文对象
            return DefaultInputContext([])

        # 寻找第一个非空父输出的索引
        not_empty_idx = 0
        for i, p in enumerate(self._outputs):
            if p.task_output.is_empty:
                # 跳过空的父输出
                continue
            not_empty_idx = i
            break

        # 检查所有父输出是否具有相同的输出格式
        is_stream = self._outputs[not_empty_idx].task_output.is_stream
        if is_stream and not self.check_stream(skip_empty=True):
            # 如果有流输出且不符合要求，则抛出数值错误
            raise ValueError(
                "The output in all tasks must has same output format to map_all"
            )

        # 收集所有父输出的输出结果
        outputs = []
        for out in self._outputs:
            if out.task_output.is_stream:
                outputs.append(out.task_output.output_stream)
            else:
                outputs.append(out.task_output.output)

        # 根据 map_func 的类型调用相应的映射函数
        if asyncio.iscoroutinefunction(map_func):
            map_res = await map_func(*outputs)
        else:
            map_res = map_func(*outputs)

        # 创建一个新的任务上下文对象，设置映射结果到任务输出
        single_output: TaskContext = self._outputs[not_empty_idx].new_ctx()
        single_output.task_output.set_output(map_res)

        # 记录调试信息，包括当前的映射结果和输出是否为流
        logger.debug(
            f"Current map_all map_res: {map_res}, is stream: "
            f"{single_output.task_output.is_stream}"
        )

        # 返回包含单个任务上下文对象的默认输入上下文对象
        return DefaultInputContext([single_output])

    async def reduce(self, reduce_func: ReduceFunc) -> InputContext:
        """Apply a reduce function to all parent outputs."""
        if not self.check_stream():
            # 如果不是流输出，则抛出数值错误
            raise ValueError(
                "The output in all tasks must has same output format of stream to apply"
                " reduce function"
            )

        # 应用 reduce_func 函数到所有父输出的结果上
        new_outputs, results = await self._apply_func(
            reduce_func, apply_type="reduce"  # type: ignore
        )

        # 将 reduce 函数的结果设置到相应的任务上下文对象中
        for i, task_ctx in enumerate(new_outputs):
            task_ctx = cast(TaskContext, task_ctx)
            task_ctx.set_task_output(results[i])

        # 返回包含新输出任务上下文对象的默认输入上下文对象
        return DefaultInputContext(new_outputs)

    async def filter(self, filter_func: Callable[[Any], bool]) -> InputContext:
        """Filter all parent outputs."""
        # 使用 filter_func 函数对所有父输出进行过滤操作
        new_outputs, results = await self._apply_func(
            filter_func, apply_type="check_condition"
        )

        # 根据过滤结果筛选任务上下文对象，仅保留满足条件的任务
        result_outputs = []
        for i, task_ctx in enumerate(new_outputs):
            if results[i]:
                result_outputs.append(task_ctx)

        # 返回包含筛选后任务上下文对象的默认输入上下文对象
        return DefaultInputContext(result_outputs)

    async def predicate_map(
        self, predicate_func: PredicateFunc, failed_value: Any = None
        ) -> InputContext:
        """Map based on a predicate function."""
        # 略，此处省略部分代码以保持注释的长度合适
    ) -> "InputContext":
        """Apply a predicate function to all parent outputs."""
        # 调用一个谓词函数对所有父输出应用操作
        new_outputs, results = await self._apply_func(
            predicate_func, apply_type="check_condition"
        )
        # 初始化一个空列表来存储结果输出
        result_outputs = []
        # 遍历新输出列表的每个任务上下文
        for i, task_ctx in enumerate(new_outputs):
            # 将任务上下文对象转换为TaskContext类型
            task_ctx = cast(TaskContext, task_ctx)
            # 如果结果不为空，则设置任务输出为True，并添加到结果输出列表中
            if not results[i].is_empty:
                task_ctx.task_output.set_output(True)
                result_outputs.append(task_ctx)
            # 如果结果为空，则设置任务输出为failed_value，并添加到结果输出列表中
            else:
                task_ctx.task_output.set_output(failed_value)
                result_outputs.append(task_ctx)
        # 返回一个包含结果输出列表的DefaultInputContext对象
        return DefaultInputContext(result_outputs)
```