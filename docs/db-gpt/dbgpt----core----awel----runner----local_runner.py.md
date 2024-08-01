# `.\DB-GPT-src\dbgpt\core\awel\runner\local_runner.py`

```py
"""Local runner for workflow.

This runner will run the workflow in the current process.
"""

import asyncio  # 引入异步编程的 asyncio 库
import logging  # 引入日志记录的 logging 库
import traceback  # 引入异常追踪的 traceback 库
from typing import Any, Dict, List, Optional, Set, cast  # 引入类型注解相关的模块

from dbgpt.component import SystemApp  # 从 dbgpt.component 模块引入 SystemApp 类
from dbgpt.util.tracer import root_tracer  # 从 dbgpt.util.tracer 模块引入 root_tracer 函数

from ..dag.base import DAGContext, DAGVar  # 从当前包的 ..dag.base 模块引入 DAGContext 和 DAGVar 类型
from ..operators.base import CALL_DATA, BaseOperator, WorkflowRunner  # 从当前包的 ..operators.base 模块引入 CALL_DATA, BaseOperator 和 WorkflowRunner 类
from ..operators.common_operator import BranchOperator  # 从当前包的 ..operators.common_operator 模块引入 BranchOperator 类
from ..task.base import SKIP_DATA, TaskContext, TaskState  # 从当前包的 ..task.base 模块引入 SKIP_DATA, TaskContext 和 TaskState 类
from ..task.task_impl import DefaultInputContext, DefaultTaskContext, SimpleTaskOutput  # 从当前包的 ..task.task_impl 模块引入 DefaultInputContext, DefaultTaskContext 和 SimpleTaskOutput 类
from .job_manager import JobManager  # 从当前包的 .job_manager 模块引入 JobManager 类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class DefaultWorkflowRunner(WorkflowRunner):
    """The default workflow runner."""

    def __init__(self):
        """Init the default workflow runner."""
        self._running_dag_ctx: Dict[str, DAGContext] = {}  # 初始化正在运行的 DAG 上下文的字典
        self._task_log_index_map: Dict[str, int] = {}  # 初始化任务日志索引的字典
        self._lock = asyncio.Lock()  # 初始化异步锁对象

    async def _log_task(self, task_id: str) -> int:
        async with self._lock:  # 异步上下文管理器，用于确保线程安全
            if task_id not in self._task_log_index_map:
                self._task_log_index_map[task_id] = 0
            self._task_log_index_map[task_id] += 1  # 增加任务日志索引
            logger.debug(
                f"Task {task_id} log index {self._task_log_index_map[task_id]}"
            )  # 记录调试级别的日志信息，显示任务和其日志索引
            return self._task_log_index_map[task_id]

    async def execute_workflow(
        self,
        node: BaseOperator,
        call_data: Optional[CALL_DATA] = None,
        streaming_call: bool = False,
        exist_dag_ctx: Optional[DAGContext] = None,
    ) -> DAGContext:
        """Execute the workflow.

        Args:
            node (BaseOperator): The end node of the workflow.
            call_data (Optional[CALL_DATA], optional): The call data of the end node.
                Defaults to None.
            streaming_call (bool, optional): Whether the call is a streaming call.
                Defaults to False.
            exist_dag_ctx (Optional[DAGContext], optional): The existing DAG context.
                Defaults to None.
        """
        # Save node output
        # 根据传入的结束节点和调用数据构建作业管理器对象
        job_manager = JobManager.build_from_end_node(node, call_data)
        
        if not exist_dag_ctx:
            # Create a new DAG context if none exists
            # 如果不存在现有的 DAG 上下文，则创建一个新的 DAG 上下文
            node_outputs: Dict[str, TaskContext] = {}
            share_data: Dict[str, Any] = {}
            event_loop_task_id = id(asyncio.current_task())
        else:
            # Share node outputs and shared data with the existing DAG context
            # 如果存在现有的 DAG 上下文，则共享节点输出和共享数据
            node_outputs = exist_dag_ctx._node_to_outputs
            share_data = exist_dag_ctx._share_data
            event_loop_task_id = exist_dag_ctx._event_loop_task_id
        
        # 构建并返回新的 DAG 上下文对象
        dag_ctx = DAGContext(
            event_loop_task_id=event_loop_task_id,
            node_to_outputs=node_outputs,
            share_data=share_data,
            streaming_call=streaming_call,
            node_name_to_ids=job_manager._node_name_to_ids,
        )
        
        # Log the beginning of the workflow run from the end operator
        # 记录工作流程的运行开始，包括节点 ID 和执行者信息
        logger.info(
            f"Begin run workflow from end operator, id: {node.node_id}, runner: {self}"
        )
        logger.debug(f"Node id {node.node_id}, call_data: {call_data}")
        
        # Initialize a set to track skipped node IDs
        # 初始化一个集合来跟踪跳过的节点 ID
        skip_node_ids: Set[str] = set()
        
        # Retrieve the current system application if available
        # 获取当前系统应用的实例（如果可用）
        system_app: Optional[SystemApp] = DAGVar.get_current_system_app()

        if node.dag:
            # Save the DAG context for the current node's DAG if it exists
            # 如果节点所属的 DAG 存在，则保存当前 DAG 上下文
            await node.dag._save_dag_ctx(dag_ctx)
        
        # Execute pre-DAG run operations from the job manager
        # 执行作业管理器的 DAG 运行前操作
        await job_manager.before_dag_run()

        # Start a root tracer span for workflow execution
        # 开始一个根跟踪器的 span 用于工作流执行
        with root_tracer.start_span(
            "dbgpt.awel.workflow.run_workflow",
            metadata={
                "exist_dag_ctx": exist_dag_ctx is not None,
                "event_loop_task_id": event_loop_task_id,
                "streaming_call": streaming_call,
                "awel_node_id": node.node_id,
                "awel_node_name": node.node_name,
            },
        ):
            # Execute the node within the workflow
            # 在工作流中执行节点
            await self._execute_node(
                job_manager, node, dag_ctx, node_outputs, skip_node_ids, system_app
            )
        
        if not streaming_call and node.dag and exist_dag_ctx is None:
            # Perform post-DAG end operations if not a streaming call and the current DAG is not a sub-DAG
            # 如果不是流式调用且当前 DAG 不是子 DAG，则执行 DAG 结束后的操作
            await node.dag._after_dag_end(dag_ctx._event_loop_task_id)
        
        # Return the constructed DAG context object
        # 返回构建的 DAG 上下文对象
        return dag_ctx
    # 异步方法 `_execute_node`，用于执行单个节点的操作
    async def _execute_node(
        # 传入的参数：
        # job_manager: 作业管理器对象，用于管理节点的执行作业
        job_manager: JobManager,
        # node: 当前要执行的操作节点，继承自 BaseOperator
        node: BaseOperator,
        # dag_ctx: DAG 上下文，包含与 DAG 相关的上下文信息
        dag_ctx: DAGContext,
        # node_outputs: 字典，包含节点输出的任务上下文信息，键为输出名称，值为任务上下文对象
        node_outputs: Dict[str, TaskContext],
        # skip_node_ids: 集合，包含应跳过执行的节点 ID
        skip_node_ids: Set[str],
        # system_app: 可选的系统应用对象，表示当前操作所属的系统应用
        system_app: Optional[SystemApp],
# 根据节点名称跳过当前分支的下游操作节点
def _skip_current_downstream_by_node_name(
    branch_node: BranchOperator, skip_nodes: List[str], skip_node_ids: Set[str]
):
    # 如果没有需要跳过的节点列表，则直接返回
    if not skip_nodes:
        return
    # 遍历当前分支节点的所有下游节点
    for child in branch_node.downstream:
        # 将子节点视为基本操作节点
        child = cast(BaseOperator, child)
        # 如果子节点的名称在跳过列表中或者其节点ID在跳过集合中，则记录日志并跳过其下游节点
        if child.node_name in skip_nodes or child.node_id in skip_node_ids:
            logger.info(f"Skip node name {child.node_name}, node id {child.node_id}")
            _skip_downstream_by_id(child, skip_node_ids)


# 根据节点ID跳过节点及其所有下游节点
def _skip_downstream_by_id(node: BaseOperator, skip_node_ids: Set[str]):
    # 如果当前节点不允许跳过，则跳过其所有下游节点
    if not node.can_skip_in_branch():
        # 当前节点无法跳过，因此跳过其所有下游节点
        return
    # 将当前节点的ID加入跳过集合
    skip_node_ids.add(node.node_id)
    # 遍历当前节点的所有下游节点
    for child in node.downstream:
        # 将子节点视为基本操作节点
        child = cast(BaseOperator, child)
        # 递归调用以跳过当前节点的每个下游节点及其子节点
        _skip_downstream_by_id(child, skip_node_ids)
```