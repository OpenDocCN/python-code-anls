# `.\DB-GPT-src\dbgpt\core\awel\task\base.py`

```py
"""Base classes for task-related objects."""
# 导入必要的模块和类型定义
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

# 定义类型变量
IN = TypeVar("IN")
OUT = TypeVar("OUT")
T = TypeVar("T")


class _EMPTY_DATA_TYPE:
    """A special type to represent empty data."""

    def __init__(self, name: str = "EMPTY_DATA"):
        self.name = name

    def __bool__(self):
        return False

    def __str__(self):
        return f"EmptyData({self.name})"

    def is_same(self, obj: Any) -> bool:
        """Check if the object is the same as the current object.

        Args:
            obj (Any): The object to compare with.

        Returns:
            bool: True if the object is the same as the current object, False otherwise.
        """
        # 判断对象是否与当前对象相同
        if not isinstance(obj, _EMPTY_DATA_TYPE):
            return False
        return self == obj


# 定义特殊的空数据常量
EMPTY_DATA = _EMPTY_DATA_TYPE("EMPTY_DATA")
SKIP_DATA = _EMPTY_DATA_TYPE("SKIP_DATA")
PLACEHOLDER_DATA = _EMPTY_DATA_TYPE("PLACEHOLDER_DATA")


def is_empty_data(data: Any):
    """Check if the data is empty."""
    # 检查数据是否为空数据
    if isinstance(data, _EMPTY_DATA_TYPE):
        return data in (EMPTY_DATA, SKIP_DATA)
    elif hasattr(data, "empty"):
        return getattr(data, "empty", False)
    return False


# 定义不同类型的函数类型别名
MapFunc = Union[Callable[[IN], OUT], Callable[[IN], Awaitable[OUT]]]
ReduceFunc = Union[Callable[[IN, IN], OUT], Callable[[IN, IN], Awaitable[OUT]]]
StreamFunc = Callable[[IN], Awaitable[AsyncIterator[OUT]]]
UnStreamFunc = Callable[[AsyncIterator[IN]], OUT]
TransformFunc = Callable[[AsyncIterator[IN]], Awaitable[AsyncIterator[OUT]]]
PredicateFunc = Union[Callable[[IN], bool], Callable[[IN], Awaitable[bool]]]
JoinFunc = Union[Callable[..., OUT], Callable[..., Awaitable[OUT]]]


class TaskState(str, Enum):
    """Enumeration representing the state of a task in the workflow.

    This Enum defines various states a task can be in during its lifecycle in the DAG.
    """

    # 任务状态枚举类，定义了任务在DAG中生命周期中可能处于的各种状态
    INIT = "init"  # Initial state of the task, not yet started
    SKIP = "skip"  # State indicating the task was skipped
    RUNNING = "running"  # State indicating the task is currently running
    SUCCESS = "success"  # State indicating the task completed successfully
    FAILED = "failed"  # State indicating the task failed during execution


class TaskOutput(ABC, Generic[T]):
    """Abstract base class representing the output of a task.

    This class encapsulates the output of a task and provides methods to access the
    output data.It can be subclassed to implement specific output behaviors.
    """

    @property
    def is_stream(self) -> bool:
        """Check if the output is a stream.

        Returns:
            bool: True if the output is a stream, False otherwise.
        """
        # 检查输出是否为流
        return False

    @property
    # 检查输出是否为空
    def is_empty(self) -> bool:
        """Check if the output is empty.

        Returns:
            bool: True if the output is empty, False otherwise.
        """
        return False

    @property
    # 检查输出是否为 None
    def is_none(self) -> bool:
        """Check if the output is None.

        Returns:
            bool: True if the output is None, False otherwise.
        """
        return False

    @property
    # 返回任务的输出
    def output(self) -> T:
        """Return the output of the task.

        Returns:
            T: The output of the task.
        """
        raise NotImplementedError

    @property
    # 返回任务输出作为异步流的形式
    def output_stream(self) -> AsyncIterator[T]:
        """Return the output of the task as an asynchronous stream.

        Returns:
            AsyncIterator[T]: An asynchronous iterator over the output. None if the
                output is empty.
        """
        raise NotImplementedError

    @abstractmethod
    # 设置输出数据到当前对象
    def set_output(self, output_data: Union[T, AsyncIterator[T]]) -> None:
        """Set the output data to current object.

        Args:
            output_data (Union[T, AsyncIterator[T]]): Output data.
        """

    @abstractmethod
    # 创建新的输出对象
    def new_output(self) -> "TaskOutput[T]":
        """Create new output object."""

    async def map(self, map_func: MapFunc) -> "TaskOutput[OUT]":
        """Apply a mapping function to the task's output.

        Args:
            map_func (MapFunc): A function to apply to the task's output.

        Returns:
            TaskOutput[OUT]: The result of applying the mapping function.
        """
        raise NotImplementedError

    async def reduce(self, reduce_func: ReduceFunc) -> "TaskOutput[OUT]":
        """Apply a reducing function to the task's output.

        Stream TaskOutput to no stream TaskOutput.

        Args:
            reduce_func: A reducing function to apply to the task's output.

        Returns:
            TaskOutput[OUT]: The result of applying the reducing function.
        """
        raise NotImplementedError

    async def streamify(self, transform_func: StreamFunc) -> "TaskOutput[T]":
        """Convert a value of type T to an AsyncIterator[T] using a transform function.

        Args:
            transform_func (StreamFunc): Function to transform a T value into an
                AsyncIterator[OUT].

        Returns:
            TaskOutput[T]: The result of applying the reducing function.
        """
        raise NotImplementedError

    async def transform_stream(
        self, transform_func: TransformFunc
    ) -> "TaskOutput[OUT]":
        """Transform an AsyncIterator[T] to another AsyncIterator[T].

        Args:
            transform_func (Callable[[AsyncIterator[T]], AsyncIterator[T]]): Function to
                 apply to the AsyncIterator[T].

        Returns:
            TaskOutput[T]: The result of applying the reducing function.
        """
        raise NotImplementedError
    # 将异步迭代器的输出转换为类型 T 的值，使用指定的转换函数
    async def unstreamify(self, transform_func: UnStreamFunc) -> "TaskOutput[OUT]":
        """Convert an AsyncIterator[T] to a value of type T using a transform function.

        Args:
            transform_func (UnStreamFunc): Function to transform an AsyncIterator[T]
                into a T value.

        Returns:
            TaskOutput[T]: The result of applying the reducing function.
        """
        # 抛出未实现错误，需要在子类中实现具体逻辑
        raise NotImplementedError

    # 检查当前输出是否满足给定条件
    async def check_condition(self, condition_func) -> "TaskOutput[OUT]":
        """Check if current output meets a given condition.

        Args:
            condition_func: A function to determine if the condition is met.
        Returns:
            TaskOutput[T]: The result of applying the reducing function.
                If the condition is not met, return empty output.
        """
        # 抛出未实现错误，需要在子类中实现具体逻辑
        raise NotImplementedError
class TaskContext(ABC, Generic[T]):
    """Abstract base class representing the context of a task within a DAG.
    
    This class provides the interface for accessing task-related information
    and manipulating task output.
    """

    @property
    @abstractmethod
    def task_id(self) -> str:
        """Return the unique identifier of the task.

        Returns:
            str: The unique identifier of the task.
        """

    @property
    @abstractmethod
    def task_input(self) -> "InputContext":
        """Return the InputContext of current task.

        Returns:
            InputContext: The InputContext of current task.

        Raises:
            Exception: If the InputContext is not set.
        """

    @abstractmethod
    def set_task_input(self, input_ctx: "InputContext") -> None:
        """Set the InputContext object to current task.

        Args:
            input_ctx (InputContext): The InputContext of current task
        """

    @property
    @abstractmethod
    def task_output(self) -> TaskOutput[T]:
        """Return the output object of the task.

        Returns:
            TaskOutput[T]: The output object of the task.
        """

    @abstractmethod
    def set_task_output(self, task_output: TaskOutput[T]) -> None:
        """Set the output object to current task."""

    @property
    @abstractmethod
    def current_state(self) -> TaskState:
        """Get the current state of the task.

        Returns:
            TaskState: The current state of the task.
        """

    @abstractmethod
    def set_current_state(self, task_state: TaskState) -> None:
        """Set current task state.

        Args:
            task_state (TaskState): The task state to be set.
        """

    @abstractmethod
    def new_ctx(self) -> "TaskContext":
        """Create new task context.

        Returns:
            TaskContext: A new instance of a TaskContext.
        """

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return the metadata of current task.

        Returns:
            Dict[str, Any]: The metadata
        """

    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata with key and value.

        Args:
            key (str): The key of metadata
            value (str): The value to be add to metadata
        """
        self.metadata[key] = value

    @property
    def call_data(self) -> Optional[Dict]:
        """Return the call data for current data."""
        return self.metadata.get("call_data")

    @abstractmethod
    async def _call_data_to_output(self) -> Optional[TaskOutput[T]]:
        """Get the call data for current data."""

    def set_call_data(self, call_data: Dict) -> None:
        """Save the call data for current task."""
        self.update_metadata("call_data", call_data)


class InputContext(ABC):
    """Abstract base class representing the context of inputs to a operator node.
    
    This class defines the structure for managing input data to an operator node
    within a computational DAG.
    """
    """
    This class defines methods to manipulate and access the inputs for a operator node.
    """

    @property
    @abstractmethod
    def parent_outputs(self) -> List[TaskContext]:
        """Get the outputs from the parent nodes.

        Returns:
            List[TaskContext]: A list of contexts of the parent nodes' outputs.
        """

    @abstractmethod
    async def map(self, map_func: Callable[[Any], Any]) -> "InputContext":
        """Apply a mapping function to the inputs.

        Args:
            map_func (Callable[[Any], Any]): A function to be applied to the inputs.

        Returns:
            InputContext: A new InputContext instance with the mapped inputs.
        """

    @abstractmethod
    async def map_all(self, map_func: Callable[..., Any]) -> "InputContext":
        """Apply a mapping function to all inputs.

        Args:
            map_func (Callable[..., Any]): A function to be applied to all inputs.

        Returns:
            InputContext: A new InputContext instance with the mapped inputs.
        """

    @abstractmethod
    async def reduce(self, reduce_func: ReduceFunc) -> "InputContext":
        """Apply a reducing function to the inputs.

        Args:
            reduce_func (Callable[[Any], Any]): A function that reduces the inputs.

        Returns:
            InputContext: A new InputContext instance with the reduced inputs.
        """

    @abstractmethod
    async def filter(self, filter_func: Callable[[Any], bool]) -> "InputContext":
        """Filter the inputs based on a provided function.

        Args:
            filter_func (Callable[[Any], bool]): A function that returns True for
                inputs to keep.

        Returns:
            InputContext: A new InputContext instance with the filtered inputs.
        """

    @abstractmethod
    async def predicate_map(
        self, predicate_func: PredicateFunc, failed_value: Any = None
    ) -> "InputContext":
        """Predicate the inputs based on a provided function.

        Args:
            predicate_func (Callable[[Any], bool]): A function that returns True for
                inputs is predicate True.
            failed_value (Any): The value to be set if the return value of predicate
                function is False

        Returns:
            InputContext: A new InputContext instance with the predicate inputs.
        """

    def check_single_parent(self) -> bool:
        """Check if there is only a single parent output.

        Returns:
            bool: True if there is only one parent output, False otherwise.
        """
        return len(self.parent_outputs) == 1
    def check_stream(self, skip_empty: bool = False) -> bool:
        """检查所有父输出是否为流对象。

        Args:
            skip_empty (bool): 是否跳过空输出。

        Returns:
            bool: 如果所有父输出都是流对象则返回True，否则返回False。
        """
        # 遍历所有父输出
        for out in self.parent_outputs:
            # 如果输出为空且跳过空输出标志为True，则继续下一次循环
            if out.task_output.is_empty and skip_empty:
                continue
            # 如果输出不为空且不是流对象，则返回False
            if not (out.task_output and out.task_output.is_stream):
                return False
        # 如果所有父输出都是流对象，则返回True
        return True
# 定义一个抽象基类 InputSource，表示数据流图节点的输入源
class InputSource(ABC, Generic[T]):
    """Abstract base class representing the source of inputs to a DAG node."""

    # 声明一个抽象方法 read，用于从当前输入源读取数据
    @abstractmethod
    async def read(self, task_ctx: TaskContext) -> TaskOutput[T]:
        """Read the data from current input source.

        Returns:
            TaskOutput[T]: The output object read from current source
        """

    # 类方法 from_data，从给定的数据创建一个 InputSource 实例
    @classmethod
    def from_data(cls, data: T) -> "InputSource[T]":
        """Create an InputSource from data.

        Args:
            data (T): The data to create the InputSource from.

        Returns:
            InputSource[T]: The InputSource created from the data.
        """
        # 导入 SimpleInputSource 类，用于创建具体的 InputSource 对象
        from .task_impl import SimpleInputSource

        # 使用 SimpleInputSource 创建一个非流式的 InputSource 实例
        return SimpleInputSource(data, streaming=False)

    # 类方法 from_iterable，从可迭代对象创建一个 InputSource 实例
    @classmethod
    def from_iterable(
        cls, iterable: Union[AsyncIterable[T], Iterable[T]]
    ) -> "InputSource[T]":
        """Create an InputSource from an iterable.

        Args:
            iterable (List[T]): The iterable to create the InputSource from.

        Returns:
            InputSource[T]: The InputSource created from the iterable.
        """
        # 导入 SimpleInputSource 类，用于创建具体的 InputSource 对象
        from .task_impl import SimpleInputSource

        # 使用 SimpleInputSource 创建一个流式的 InputSource 实例
        return SimpleInputSource(iterable, streaming=True)

    # 类方法 from_callable，从可调用对象创建一个 InputSource 实例
    @classmethod
    def from_callable(cls) -> "InputSource[T]":
        """Create an InputSource from a callable."""
        # 导入 SimpleCallDataInputSource 类，用于创建具体的 InputSource 对象
        from .task_impl import SimpleCallDataInputSource

        # 使用 SimpleCallDataInputSource 创建一个 InputSource 实例
        return SimpleCallDataInputSource()
```