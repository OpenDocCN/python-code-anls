# `.\DB-GPT-src\dbgpt\core\awel\operators\common_operator.py`

```py
"""Common operators of AWEL."""

import asyncio  # 引入异步IO库，用于支持异步操作
import logging  # 引入日志记录库，用于输出日志信息
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Union  # 引入类型提示模块，用于声明函数和变量的类型

from ..dag.base import DAGContext  # 从上级目录中导入DAG上下文类
from ..task.base import (  # 从任务基础模块导入多个符号
    IN,
    OUT,
    SKIP_DATA,
    InputContext,
    InputSource,
    JoinFunc,
    MapFunc,
    ReduceFunc,
    TaskContext,
    TaskOutput,
    is_empty_data,
)
from .base import BaseOperator  # 从当前目录下的基础操作模块导入基础操作类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class JoinOperator(BaseOperator, Generic[OUT]):
    """Operator that joins inputs using a custom combine function.

    This node type is useful for combining the outputs of upstream nodes.
    """

    def __init__(
        self, combine_function: JoinFunc, can_skip_in_branch: bool = True, **kwargs
    ):
        """Create a JoinDAGNode with a combine function.

        Args:
            combine_function: A function that defines how to combine inputs.
            can_skip_in_branch(bool): Whether the node can be skipped in a branch.
        """
        super().__init__(can_skip_in_branch=can_skip_in_branch, **kwargs)
        if not callable(combine_function):
            raise ValueError("combine_function must be callable")
        self.combine_function = combine_function

    async def _do_run(self, dag_ctx: DAGContext) -> TaskOutput[OUT]:
        """Run the join operation on the DAG context's inputs.

        Args:
            dag_ctx (DAGContext): The current context of the DAG.

        Returns:
            TaskOutput[OUT]: The task output after this node has been run.
        """
        curr_task_ctx: TaskContext[OUT] = dag_ctx.current_task_context
        input_ctx: InputContext = await curr_task_ctx.task_input.map_all(
            self.combine_function
        )
        # All join result store in the first parent output
        join_output = input_ctx.parent_outputs[0].task_output
        curr_task_ctx.set_task_output(join_output)
        return join_output

    async def _return_first_non_empty(self, *inputs):
        for data in inputs:
            if not is_empty_data(data):
                return data
        raise ValueError("All inputs are empty")


class ReduceStreamOperator(BaseOperator, Generic[IN, OUT]):
    """Operator that reduces inputs using a custom reduce function."""

    def __init__(self, reduce_function: Optional[ReduceFunc] = None, **kwargs):
        """Create a ReduceStreamOperator with a combine function.

        Args:
            reduce_function: A function that defines how to reduce inputs.

        Raises:
            ValueError: If the reduce_function is not callable.
        """
        super().__init__(**kwargs)
        if reduce_function and not callable(reduce_function):
            raise ValueError("reduce_function must be callable")
        self.reduce_function = reduce_function
    async def _do_run(self, dag_ctx: DAGContext) -> TaskOutput[OUT]:
        """Run the join operation on the DAG context's inputs.

        Args:
            dag_ctx (DAGContext): The current context of the DAG.

        Returns:
            TaskOutput[OUT]: The task output after this node has been run.
        """
        # 获取当前任务的上下文
        curr_task_ctx: TaskContext[OUT] = dag_ctx.current_task_context
        # 获取当前任务的输入数据
        task_input = curr_task_ctx.task_input
        # 检查输入数据是否为流数据，如果不是则抛出异常
        if not task_input.check_stream():
            raise ValueError("ReduceStreamOperator expects stream data")
        # 检查输入数据是否只有一个父任务，如果不是则抛出异常
        if not task_input.check_single_parent():
            raise ValueError("ReduceStreamOperator expects single parent")

        # 获取执行缩减操作的函数，如果未指定则使用默认的 self.reduce_function 或 self.reduce
        reduce_function = self.reduce_function or self.reduce

        # 调用输入数据的 reduce 方法来执行缩减操作，得到一个输入上下文对象
        input_ctx: InputContext = await task_input.reduce(reduce_function)
        # 将所有的缩减结果存储在第一个父任务的输出中
        reduce_output = input_ctx.parent_outputs[0].task_output
        # 设置当前任务的输出结果
        curr_task_ctx.set_task_output(reduce_output)
        # 返回缩减后的结果
        return reduce_output

    async def reduce(self, a: IN, b: IN) -> OUT:
        """Reduce the input stream to a single value."""
        # 抽象方法，用于将输入流缩减为单个值，需要在子类中实现
        raise NotImplementedError
class MapOperator(BaseOperator, Generic[IN, OUT]):
    """Map operator that applies a mapping function to its inputs.

    This operator transforms its input data using a provided mapping function and
    passes the transformed data downstream.
    """

    def __init__(self, map_function: Optional[MapFunc] = None, **kwargs):
        """Create a MapDAGNode with a mapping function.

        Args:
            map_function: A function that defines how to map the input data.

        Raises:
            ValueError: If the map_function is not callable.
        """
        super().__init__(**kwargs)
        # 检查 map_function 是否可调用，如果不可调用则抛出 ValueError 异常
        if map_function and not callable(map_function):
            raise ValueError("map_function must be callable")
        self.map_function = map_function

    async def _do_run(self, dag_ctx: DAGContext) -> TaskOutput[OUT]:
        """Run the mapping operation on the DAG context's inputs.

        This method applies the mapping function to the input context and updates
        the DAG context with the new data.

        Args:
            dag_ctx (DAGContext[IN]): The current context of the DAG.

        Returns:
            TaskOutput[OUT]: The task output after this node has been run.

        Raises:
            ValueError: If not a single parent or the map_function is not callable
        """
        curr_task_ctx: TaskContext[OUT] = dag_ctx.current_task_context
        call_data = curr_task_ctx.call_data
        # 如果没有 call_data 并且任务输入不只一个父节点，则抛出 ValueError 异常
        if not call_data and not curr_task_ctx.task_input.check_single_parent():
            num_parents = len(curr_task_ctx.task_input.parent_outputs)
            raise ValueError(
                f"task {curr_task_ctx.task_id} MapDAGNode expects single parent,"
                f"now number of parents: {num_parents}"
            )
        # 获取 map_function，若未指定则使用默认的 self.map
        map_function = self.map_function or self.map

        # 如果有 call_data，则将其转换为输出数据并进行映射操作
        if call_data:
            wrapped_call_data = await curr_task_ctx._call_data_to_output()
            if not wrapped_call_data:
                raise ValueError(
                    f"task {curr_task_ctx.task_id} MapDAGNode expects wrapped_call_data"
                )
            # 对 wrapped_call_data 应用映射函数，更新任务输出
            output: TaskOutput[OUT] = await wrapped_call_data.map(map_function)
            curr_task_ctx.set_task_output(output)
            return output

        # 否则，对输入上下文应用映射函数，所有连接结果存储在第一个父节点输出中
        input_ctx: InputContext = await curr_task_ctx.task_input.map(map_function)
        output = input_ctx.parent_outputs[0].task_output
        curr_task_ctx.set_task_output(output)
        return output

    async def map(self, input_value: IN) -> OUT:
        """Map the input data to a new value."""
        # 抽象方法，子类需要实现具体的映射逻辑
        raise NotImplementedError


BranchFunc = Union[Callable[[IN], bool], Callable[[IN], Awaitable[bool]]]
# 返回任务名称的函数类型定义
BranchTaskType = Union[str, Callable[[IN], str], Callable[[IN], Awaitable[str]]]


class BranchOperator(BaseOperator, Generic[IN, OUT]):
    """Operator node that branches the workflow based on a provided function.
    """
    This node filters its input data using a branching function and
    allows for conditional paths in the workflow.
    """

    def __init__(
        self,
        branches: Optional[Dict[BranchFunc[IN], BranchTaskType]] = None,
        **kwargs,
    ):
        """
        Create a BranchDAGNode with a branching function.

        Args:
            branches (Dict[BranchFunc[IN], Union[BaseOperator, str]]):
                Dict of functions defining the branching conditions.

        Raises:
            ValueError: If the branch_function is not callable.
        """
        super().__init__(**kwargs)
        if branches:
            for branch_function, value in branches.items():
                # Check if the branch_function is callable
                if not callable(branch_function):
                    raise ValueError("branch_function must be callable")
                if isinstance(value, BaseOperator):
                    # Ensure the BaseOperator has a node_name set
                    if not value.node_name:
                        raise ValueError("branch node name must be set")
                    branches[branch_function] = value.node_name
                elif callable(value):
                    # Ensure BranchTaskType is either a string or BaseOperator
                    raise ValueError(
                        "BranchTaskType must be str or BaseOperator on init"
                    )
        self._branches = branches
    async def _do_run(self, dag_ctx: DAGContext) -> TaskOutput[OUT]:
        """Run the branching operation on the DAG context's inputs.

        This method applies the branching function to the input context to determine
        the path of execution in the workflow.

        Args:
            dag_ctx (DAGContext[IN]): The current context of the DAG.

        Returns:
            TaskOutput[OUT]: The task output after this node has been run.
        """
        # 获取当前任务上下文
        curr_task_ctx: TaskContext[OUT] = dag_ctx.current_task_context
        # 获取当前任务的输入
        task_input = curr_task_ctx.task_input
        # 检查任务输入中是否包含流数据，如果有则抛出异常
        if task_input.check_stream():
            raise ValueError("BranchDAGNode expects no stream data")
        # 检查任务输入是否只有一个父节点，如果不是则抛出异常
        if not task_input.check_single_parent():
            raise ValueError("BranchDAGNode expects single parent")

        # 获取分支函数列表
        branches = self._branches
        # 如果分支列表为空，则调用异步方法获取分支逻辑
        if not branches:
            branches = await self.branches()

        # 用于存储分支函数任务和分支节点名任务的列表
        branch_func_tasks = []
        branch_name_tasks = []

        # 遍历分支函数和节点名的映射关系
        for func, node_name in branches.items():
            # 添加分支函数任务到列表中
            branch_func_tasks.append(
                curr_task_ctx.task_input.predicate_map(func, failed_value=None)
            )
            # 如果节点名是可调用的，则定义异步函数用于映射节点名
            if callable(node_name):

                async def map_node_name(func) -> str:
                    # 使用映射函数获取输入上下文，并获取任务名
                    input_context = await curr_task_ctx.task_input.map(func)
                    task_name = input_context.parent_outputs[0].task_output.output
                    return task_name

                # 将映射节点名的异步函数添加到任务列表中
                branch_name_tasks.append(map_node_name(node_name))

            else:
                # 如果节点名不是可调用的，则定义临时异步函数返回节点名本身
                async def _tmp_map_node_name(task_name: str) -> str:
                    return task_name

                # 将临时异步函数添加到任务列表中
                branch_name_tasks.append(_tmp_map_node_name(node_name))

        # 执行所有分支函数任务并获取结果列表
        branch_input_ctxs: List[InputContext] = await asyncio.gather(*branch_func_tasks)
        # 执行所有节点名任务并获取结果列表
        branch_nodes: List[str] = await asyncio.gather(*branch_name_tasks)

        # 获取父任务的输出
        parent_output = task_input.parent_outputs[0].task_output
        # 设置当前任务的输出为父任务的输出
        curr_task_ctx.set_task_output(parent_output)

        # 用于存储跳过节点名的列表
        skip_node_names = []
        # 遍历分支输入上下文列表
        for i, ctx in enumerate(branch_input_ctxs):
            node_name = branch_nodes[i]
            # 获取分支的输出
            branch_out = ctx.parent_outputs[0].task_output
            # 记录日志，包含分支输入结果和是否为空的信息
            logger.info(
                f"branch_input_ctxs {i} result {branch_out.output}, "
                f"is_empty: {branch_out.is_empty}"
            )
            # 如果分支输出为空，则记录日志并将节点名添加到跳过列表中
            if ctx.parent_outputs[0].task_output.is_none:
                logger.info(f"Skip node name {node_name}")
                skip_node_names.append(node_name)

        # 更新当前任务的元数据，记录跳过的节点名列表
        curr_task_ctx.update_metadata("skip_node_names", skip_node_names)
        # 返回父任务的输出作为当前任务的输出
        return parent_output
# 定义一个名为 BranchJoinOperator 的类，它继承自 JoinOperator，并支持泛型 OUT
class BranchJoinOperator(JoinOperator, Generic[OUT]):
    """Operator that joins inputs using a custom combine function.

    This node type is useful for combining the outputs of upstream nodes.
    """

    def __init__(
        self,
        combine_function: Optional[JoinFunc] = None,
        can_skip_in_branch: bool = False,
        **kwargs,
    ):
        """Create a JoinDAGNode with a combine function.

        Args:
            combine_function: A function that defines how to combine inputs.
            can_skip_in_branch(bool): Whether the node can be skipped in a branch (default True).
        """
        # 调用父类 JoinOperator 的构造函数，设置 combine_function 和 can_skip_in_branch 属性
        super().__init__(
            combine_function=combine_function or self._return_first_non_empty,
            can_skip_in_branch=can_skip_in_branch,
            **kwargs,
        )


# 定义一个名为 InputOperator 的类，它继承自 BaseOperator，并支持泛型 OUT
class InputOperator(BaseOperator, Generic[OUT]):
    """Operator node that reads data from an input source."""

    def __init__(self, input_source: InputSource[OUT], **kwargs) -> None:
        """Create an InputDAGNode with an input source."""
        # 调用父类 BaseOperator 的构造函数，设置 _input_source 属性
        super().__init__(**kwargs)
        self._input_source = input_source

    async def _do_run(self, dag_ctx: DAGContext) -> TaskOutput[OUT]:
        # 获取当前任务上下文
        curr_task_ctx: TaskContext[OUT] = dag_ctx.current_task_context
        # 从输入源读取任务输出
        task_output = await self._input_source.read(curr_task_ctx)
        # 将任务输出设置到当前任务上下文中
        curr_task_ctx.set_task_output(task_output)
        return task_output

    @classmethod
    def dummy_input(cls, dummy_data: Any = SKIP_DATA, **kwargs) -> "InputOperator[OUT]":
        """Create a dummy InputOperator with a given input value."""
        # 使用给定的虚拟数据创建一个虚拟的 InputOperator 对象
        return cls(input_source=InputSource.from_data(dummy_data), **kwargs)


# 定义一个名为 TriggerOperator 的类，它继承自 InputOperator，并支持泛型 OUT
class TriggerOperator(InputOperator[OUT], Generic[OUT]):
    """Operator node that triggers the DAG to run."""

    def __init__(self, **kwargs) -> None:
        """Create a TriggerDAGNode."""
        # 导入需要的模块
        from ..task.task_impl import SimpleCallDataInputSource
        # 调用父类 InputOperator 的构造函数，设置 input_source 属性为 SimpleCallDataInputSource 实例
        super().__init__(input_source=SimpleCallDataInputSource(), **kwargs)
```