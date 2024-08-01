# `.\DB-GPT-src\dbgpt\core\awel\operators\base.py`

```py
"""Base classes for operators that can be executed within a workflow."""

import asyncio  # 异步编程库，用于处理异步任务
import functools  # 函数工具库，用于高阶函数的操作
from abc import ABC, ABCMeta, abstractmethod  # 导入抽象基类相关模块
from contextvars import ContextVar  # 上下文变量模块，用于跨协程存储和传递上下文
from types import FunctionType  # 导入 FunctionType 类型，用于函数类型的声明
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    Iterator,
    Optional,
    TypeVar,
    Union,
    cast,
)  # 引入类型提示模块，用于类型标注和提示

from dbgpt.component import ComponentType, SystemApp  # 导入组件类型和系统应用模块
from dbgpt.util.executor_utils import (
    AsyncToSyncIterator,
    BlockingFunction,
    DefaultExecutorFactory,
    blocking_func_to_async,
)  # 导入执行器工具函数
from dbgpt.util.tracer import root_tracer  # 导入追踪器

from ..dag.base import DAG, DAGContext, DAGNode, DAGVar  # 导入 DAG 相关模块
from ..task.base import EMPTY_DATA, OUT, T, TaskOutput, is_empty_data  # 导入任务相关模块

F = TypeVar("F", bound=FunctionType)  # 声明一个类型变量 F，绑定到 FunctionType 类型

CALL_DATA = Union[Dict[str, Any], Any]  # 声明 CALL_DATA 类型别名，可以是字典或任意类型
CURRENT_DAG_CONTEXT: ContextVar[Optional[DAGContext]] = ContextVar(
    "current_dag_context", default=None
)  # 定义一个上下文变量 CURRENT_DAG_CONTEXT，存储当前 DAG 的上下文信息，默认为 None


class WorkflowRunner(ABC, Generic[T]):
    """Abstract base class representing a runner for executing workflows in a DAG.

    This class defines the interface for executing workflows within the DAG,
    handling the flow from one DAG node to another.
    """

    @abstractmethod
    async def execute_workflow(
        self,
        node: "BaseOperator",
        call_data: Optional[CALL_DATA] = None,
        streaming_call: bool = False,
        exist_dag_ctx: Optional[DAGContext] = None,
    ) -> DAGContext:
        """Execute the workflow starting from a given operator.

        Args:
            node (RunnableDAGNode): The starting node of the workflow to be executed.
            call_data (CALL_DATA): The data pass to root operator node.
            streaming_call (bool): Whether the call is a streaming call.
            exist_dag_ctx (DAGContext): The context of the DAG when this node is run,
                Defaults to None.
        Returns:
            DAGContext: The context after executing the workflow, containing the final
                state and data.
        """


default_runner: Optional[WorkflowRunner] = None  # 定义一个可选的 WorkflowRunner 对象，默认为 None


def _dev_mode() -> bool:
    """Check if the operator is in dev mode.

    In production mode, the default runner is not None, and the operator will run in
    the same process with the DB-GPT webserver.
    """
    return default_runner is None  # 返回当前是否处于开发模式，即 default_runner 是否为 None


class BaseOperatorMeta(ABCMeta):
    """Metaclass of BaseOperator."""

    @classmethod
    def _apply_defaults(cls, func: F) -> F:
        # 定义内部函数apply_defaults，用于应用默认参数和包装原始函数
        @functools.wraps(func)
        def apply_defaults(self: "BaseOperator", *args: Any, **kwargs: Any) -> Any:
            # 获取关键字参数中的"dag"，如果不存在则使用当前的DAG
            dag: Optional[DAG] = kwargs.get("dag") or DAGVar.get_current_dag()
            # 获取关键字参数中的"task_id"
            task_id: Optional[str] = kwargs.get("task_id")
            # 获取关键字参数中的"system_app"，如果不存在则使用当前的系统应用
            system_app: Optional[SystemApp] = (
                kwargs.get("system_app") or DAGVar.get_current_system_app()
            )
            # 获取关键字参数中的"executor"
            executor = kwargs.get("executor") or DAGVar.get_executor()
            # 如果executor不存在，则根据情况选择默认的执行器
            if not executor:
                if system_app:
                    executor = system_app.get_component(
                        ComponentType.EXECUTOR_DEFAULT,
                        DefaultExecutorFactory,
                        default_component=DefaultExecutorFactory(),
                    ).create()  # type: ignore
                else:
                    executor = DefaultExecutorFactory().create()
                DAGVar.set_executor(executor)

            # 如果task_id不存在且存在dag，则生成一个新的任务ID
            if not task_id and dag:
                task_id = dag._new_node_id()
            # 获取关键字参数中的"runner"，如果不存在则使用默认的运行器
            runner: Optional[WorkflowRunner] = kwargs.get("runner") or default_runner

            # 如果关键字参数中没有指定的值，则使用默认参数补充
            if not kwargs.get("dag"):
                kwargs["dag"] = dag
            if not kwargs.get("task_id"):
                kwargs["task_id"] = task_id
            if not kwargs.get("runner"):
                kwargs["runner"] = runner
            if not kwargs.get("system_app"):
                kwargs["system_app"] = system_app
            if not kwargs.get("executor"):
                kwargs["executor"] = executor

            # 调用原始函数，返回实际对象
            real_obj = func(self, *args, **kwargs)
            return real_obj

        return cast(F, apply_defaults)

    def __new__(cls, name, bases, namespace, **kwargs):
        """创建一个新的BaseOperator类，并应用默认参数。"""
        # 调用父类的__new__方法创建新的类
        new_cls = super().__new__(cls, name, bases, namespace, **kwargs)
        # 将新类的__init__方法替换为经过默认参数处理的_apply_defaults返回的函数
        new_cls.__init__ = cls._apply_defaults(new_cls.__init__)
        # 在定义之后执行后续定义的操作
        new_cls.after_define()
        # 返回创建好的新类
        return new_cls
class BaseOperator(DAGNode, ABC, Generic[OUT], metaclass=BaseOperatorMeta):
    """Abstract base class for operator nodes that can be executed within a workflow.

    This class extends DAGNode by adding execution capabilities.
    """

    streaming_operator: bool = False  # 指示是否为流操作器
    incremental_output: bool = False  # 指示是否具有增量输出
    output_format: Optional[str] = None  # 输出格式，可以为None

    def __init__(
        self,
        task_id: Optional[str] = None,
        task_name: Optional[str] = None,
        dag: Optional[DAG] = None,
        runner: Optional[WorkflowRunner] = None,
        can_skip_in_branch: bool = True,
        **kwargs,
    ) -> None:
        """Create a BaseOperator with an optional workflow runner.

        Args:
            runner (WorkflowRunner, optional): The runner used to execute the workflow.
                Defaults to None.
        """
        super().__init__(node_id=task_id, node_name=task_name, dag=dag, **kwargs)  # 调用父类的构造函数
        if not runner:
            from dbgpt.core.awel import DefaultWorkflowRunner

            runner = DefaultWorkflowRunner()  # 如果没有指定runner，则使用默认的工作流程运行器
        if "incremental_output" in kwargs:
            self.incremental_output = bool(kwargs["incremental_output"])  # 如果在kwargs中指定了增量输出，则设置增量输出标志
        if "output_format" in kwargs:
            self.output_format = kwargs["output_format"]  # 如果在kwargs中指定了输出格式，则设置输出格式

        self._runner: WorkflowRunner = runner  # 设置工作流程运行器
        self._dag_ctx: Optional[DAGContext] = None  # DAG上下文，默认为None
        self._can_skip_in_branch = can_skip_in_branch  # 设置是否可以在分支中跳过任务的标志

    @property
    def current_dag_context(self) -> DAGContext:
        """Return the current DAG context."""
        ctx = CURRENT_DAG_CONTEXT.get()  # 获取当前的DAG上下文
        if not ctx:
            raise ValueError("DAGContext is not set")  # 如果没有找到上下文，则引发值错误异常
        return ctx

    @property
    def dev_mode(self) -> bool:
        """Whether the operator is in dev mode.

        In production mode, the default runner is not None, and the operator will run in
        the same process with the DB-GPT webserver.

        Returns:
            bool: Whether the operator is in dev mode. True if the
                default runner is None.
        """
        return _dev_mode()  # 返回当前是否处于开发模式的状态

    async def _run(self, dag_ctx: DAGContext, task_log_id: str) -> TaskOutput[OUT]:
        if not self.node_id:
            raise ValueError(f"The DAG Node ID can't be empty, current node {self}")  # 如果节点ID为空，则引发值错误异常
        if not task_log_id:
            raise ValueError(f"The task log ID can't be empty, current node {self}")  # 如果任务日志ID为空，则引发值错误异常
        CURRENT_DAG_CONTEXT.set(dag_ctx)  # 设置当前的DAG上下文
        return await self._do_run(dag_ctx)  # 调用具体的运行方法来执行任务

    @abstractmethod
    async def _do_run(self, dag_ctx: DAGContext) -> TaskOutput[OUT]:
        """
        Abstract method to run the task within the DAG node.

        Args:
            dag_ctx (DAGContext): The context of the DAG when this node is run.

        Returns:
            TaskOutput[OUT]: The task output after this node has been run.
        """

    async def call(
        self,
        call_data: Optional[CALL_DATA] = EMPTY_DATA,
        dag_ctx: Optional[DAGContext] = None,
        ...
    ):
        """Async method to perform operator call within a workflow."""
        # 省略部分参数说明
        ...
    ) -> OUT:
        """
        Execute the node and return the output.

        This method is a high-level wrapper for executing the node.

        Args:
            call_data (CALL_DATA): The data passed to the root operator node.
            dag_ctx (DAGContext): The context of the DAG when this node is run.
                Defaults to None.

        Returns:
            OUT: The output of the node after execution.
        """
        if not is_empty_data(call_data):
            # Ensure call_data is wrapped in a dictionary if not empty
            call_data = {"data": call_data}
        with root_tracer.start_span("dbgpt.awel.operator.call"):
            # Start a tracing span for debugging purposes
            out_ctx = await self._runner.execute_workflow(
                self, call_data, exist_dag_ctx=dag_ctx
            )
            # Execute the workflow and retrieve the output context
            return out_ctx.current_task_context.task_output.output

    def _blocking_call(
        self,
        call_data: Optional[CALL_DATA] = EMPTY_DATA,
        loop: Optional[asyncio.BaseEventLoop] = None,
    ) -> OUT:
        """
        Execute the node and return the output in a blocking manner.

        This method is a high-level wrapper for executing the node.
        This method is primarily used for debugging. It's recommended to use `call` method instead.

        Args:
            call_data (CALL_DATA): The data passed to the root operator node.
            loop (asyncio.BaseEventLoop): The event loop to run the call operation. If not provided, a new event loop is created.

        Returns:
            OUT: The output of the node after execution.
        """
        from dbgpt.util.utils import get_or_create_event_loop

        if not loop:
            # If no event loop is provided, obtain or create a new one
            loop = get_or_create_event_loop()
        loop = cast(asyncio.BaseEventLoop, loop)
        # Run the `call` method asynchronously and return its result
        return loop.run_until_complete(self.call(call_data))

    async def call_stream(
        self,
        call_data: Optional[CALL_DATA] = EMPTY_DATA,
        dag_ctx: Optional[DAGContext] = None,
    ) -> AsyncIterator[OUT]:
        """
        Execute the node and return the output as a stream.

        This method is used for nodes where the output is a stream.

        Args:
            call_data (CALL_DATA): The data passed to the root operator node.
            dag_ctx (DAGContext): The context of the DAG when this node is run,
                Defaults to None.

        Returns:
            AsyncIterator[OUT]: An asynchronous iterator over the output stream.
        """
        if call_data != EMPTY_DATA:
            # Wrap call_data in a dictionary if it's not empty
            call_data = {"data": call_data}
        # Start a tracing span for debugging purposes
        with root_tracer.start_span("dbgpt.awel.operator.call_stream"):
            # Execute the workflow asynchronously to obtain output context
            out_ctx = await self._runner.execute_workflow(
                self, call_data, streaming_call=True, exist_dag_ctx=dag_ctx
            )

            # Retrieve the task output from the output context
            task_output = out_ctx.current_task_context.task_output
            if task_output.is_stream:
                # If task output is a stream, use its generator
                stream_generator = (
                    out_ctx.current_task_context.task_output.output_stream
                )
            else:
                # If no stream output, wrap the output in a stream generator function
                async def _gen():
                    yield task_output.output

                stream_generator = _gen()
            # Return the asynchronous stream wrapped with a tracing span
            return root_tracer.wrapper_async_stream(
                stream_generator, "dbgpt.awel.operator.call_stream.iterate"
            )

    def _blocking_call_stream(
        self,
        call_data: Optional[CALL_DATA] = EMPTY_DATA,
        loop: Optional[asyncio.BaseEventLoop] = None,
    ) -> Iterator[OUT]:
        """
        Execute the node and return the output as a stream.

        This method is used for nodes where the output is a stream.
        This method just for debug. Please use `call_stream` method instead.

        Args:
            call_data (CALL_DATA): The data passed to the root operator node.

        Returns:
            Iterator[OUT]: An iterator over the output stream.
        """
        from dbgpt.util.utils import get_or_create_event_loop

        # Get or create the event loop if not provided
        if not loop:
            loop = get_or_create_event_loop()
        # Convert async iterator to synchronous iterator and return
        return AsyncToSyncIterator(self.call_stream(call_data), loop)

    async def blocking_func_to_async(
        self, func: BlockingFunction, *args, **kwargs
    ) -> Any:
        """
        Execute a blocking function asynchronously.

        In AWEL, the operators are executed asynchronously. However,
        some functions are blocking, we run them in a separate thread.

        Args:
            func (BlockingFunction): The blocking function to be executed.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        """
        if not self._executor:
            raise ValueError("Executor is not set")
        # Execute the blocking function asynchronously using the executor
        return await blocking_func_to_async(self._executor, func, *args, **kwargs)

    @property
    def current_event_loop_task_id(self) -> int:
        """Get the current event loop task id."""
        # Return the id of the current asyncio event loop task
        return id(asyncio.current_task())
    # 定义一个方法，用于检查操作符在分支中是否可以跳过
    def can_skip_in_branch(self) -> bool:
        """Check if the operator can be skipped in the branch."""
        # 返回一个布尔值，表示是否可以跳过当前操作符在分支中的使用
        return self._can_skip_in_branch
# 定义一个函数，用于初始化默认的工作流程运行器
def initialize_runner(runner: WorkflowRunner):
    # 声明 default_runner 变量为全局变量，用于存储传入的运行器对象
    global default_runner
    # 将传入的 runner 参数赋值给全局变量 default_runner
    default_runner = runner
```