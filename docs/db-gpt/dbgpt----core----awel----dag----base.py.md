# `.\DB-GPT-src\dbgpt\core\awel\dag\base.py`

```py
"""The base module of DAG.

DAG is the core component of AWEL, it is used to define the relationship between tasks.
"""

import asyncio  # 异步编程的支持库
import contextvars  # 上下文变量支持库，用于跨协程和线程传递上下文信息
import logging  # 日志记录库
import threading  # 多线程支持库
import uuid  # 生成唯一标识符的库
from abc import ABC, abstractmethod  # 抽象基类和抽象方法的支持
from collections import deque  # 双端队列的支持
from concurrent.futures import Executor  # 并发执行器的支持
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union, cast  # 类型提示支持

from dbgpt.component import SystemApp  # 调试工具相关组件

from ..flow.base import ViewMixin  # 流视图相关基础组件
from ..resource.base import ResourceGroup  # 资源组相关基础组件
from ..task.base import TaskContext, TaskOutput  # 任务上下文和任务输出相关基础组件

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

DependencyType = Union["DependencyMixin", Sequence["DependencyMixin"]]  # 依赖关系类型的别名定义


def _is_async_context():
    """Check if the current context is within an async context.

    Returns:
        bool: True if currently within an async context, False otherwise.
    """
    try:
        loop = asyncio.get_running_loop()  # 获取当前正在运行的事件循环
        return asyncio.current_task(loop=loop) is not None  # 检查当前任务是否存在于指定事件循环中
    except RuntimeError:
        return False


class DependencyMixin(ABC):
    """The mixin class for DAGNode.

    This class defines the interface for setting upstream and downstream nodes.

    And it also implements the operator << and >> for setting upstream
    and downstream nodes.
    """

    @abstractmethod
    def set_upstream(self, nodes: DependencyType) -> None:
        """Set one or more upstream nodes for this node.

        Args:
            nodes (DependencyType): Upstream nodes to be set to current node.

        Raises:
            ValueError: If no upstream nodes are provided or if an argument is
            not a DependencyMixin.
        """

    @abstractmethod
    def set_downstream(self, nodes: DependencyType) -> None:
        """Set one or more downstream nodes for this node.

        Args:
            nodes (DependencyType): Downstream nodes to be set to current node.

        Raises:
            ValueError: If no downstream nodes are provided or if an argument is
            not a DependencyMixin.
        """

    def __lshift__(self, nodes: DependencyType) -> DependencyType:
        """Set upstream nodes for current node.

        Implements: self << nodes.

        Example:
            .. code-block:: python

                # means node.set_upstream(input_node)
                node << input_node
                # means node2.set_upstream([input_node])
                node2 << [input_node]

        """
        self.set_upstream(nodes)
        return nodes

    def __rshift__(self, nodes: DependencyType) -> DependencyType:
        """Set downstream nodes for current node.

        Implements: self >> nodes.

        Examples:
            .. code-block:: python

                # means node.set_downstream(next_node)
                node >> next_node

                # means node2.set_downstream([next_node])
                node2 >> [next_node]

        """
        self.set_downstream(nodes)
        return nodes

    def __rrshift__(self, nodes: DependencyType) -> "DependencyMixin":
        """Set upstream nodes for current node.

        Implements: [node] >> self
        """
        self.__lshift__(nodes)
        return self
    # 定义一个特殊方法 __rlshift__，用于设置当前节点的下游节点
    def __rlshift__(self, nodes: DependencyType) -> "DependencyMixin":
        """Set downstream nodes for current node.

        Implements: [node] << self
        """
        # 调用 __rshift__ 方法来设置当前节点的下游节点
        self.__rshift__(nodes)
        # 返回当前对象自身
        return self
class DAGVar:
    """The DAGVar is used to store the current DAG context."""

    # 线程本地存储，用于存储线程局部变量
    _thread_local = threading.local()
    # 异步上下文变量，用于存储上下文相关的变量
    _async_local: contextvars.ContextVar = contextvars.ContextVar(
        "current_dag_stack", default=deque()
    )
    # 当前系统应用对象的可选引用
    _system_app: Optional[SystemApp] = None
    # 当前DAG的执行器，用于在异步DAG中运行同步任务
    _executor: Optional[Executor] = None

    @classmethod
    def enter_dag(cls, dag) -> None:
        """Enter a DAG context.

        Args:
            dag (DAG): The DAG to enter
        """
        is_async = _is_async_context()
        if is_async:
            stack = cls._async_local.get()
            stack.append(dag)
            cls._async_local.set(stack)
        else:
            # 如果线程不是异步的，将DAG压入线程本地栈中
            if not hasattr(cls._thread_local, "current_dag_stack"):
                cls._thread_local.current_dag_stack = deque()
            cls._thread_local.current_dag_stack.append(dag)

    @classmethod
    def exit_dag(cls) -> None:
        """Exit a DAG context."""
        is_async = _is_async_context()
        if is_async:
            stack = cls._async_local.get()
            if stack:
                stack.pop()
                cls._async_local.set(stack)
        else:
            # 如果线程不是异步的，从线程本地栈中弹出当前DAG
            if (
                hasattr(cls._thread_local, "current_dag_stack")
                and cls._thread_local.current_dag_stack
            ):
                cls._thread_local.current_dag_stack.pop()

    @classmethod
    def get_current_dag(cls) -> Optional["DAG"]:
        """Get the current DAG.

        Returns:
            Optional[DAG]: The current DAG
        """
        is_async = _is_async_context()
        if is_async:
            stack = cls._async_local.get()
            return stack[-1] if stack else None
        else:
            # 如果线程不是异步的，返回线程本地栈中的最后一个DAG对象
            if (
                hasattr(cls._thread_local, "current_dag_stack")
                and cls._thread_local.current_dag_stack
            ):
                return cls._thread_local.current_dag_stack[-1]
            return None

    @classmethod
    def get_current_system_app(cls) -> Optional[SystemApp]:
        """Get the current system app.

        Returns:
            Optional[SystemApp]: The current system app
        """
        # 返回当前系统应用对象的引用，如果未设置则返回None
        return cls._system_app

    @classmethod
    def set_current_system_app(cls, system_app: SystemApp) -> None:
        """Set the current system app.

        Args:
            system_app (SystemApp): The system app to set
        """
        # 设置当前系统应用对象，如果已设置则发出警告
        if cls._system_app:
            logger.warning("System APP has already set, nothing to do")
        else:
            cls._system_app = system_app

    @classmethod
    def get_executor(cls) -> Optional[Executor]:
        """Get the current executor.

        Returns:
            Optional[Executor]: The current executor
        """
        # 返回当前的执行器对象
        return cls._executor

    @classmethod
    def set_executor(cls, executor: Executor) -> None:
        """Set the current executor.

        Args:
            executor (Executor): The executor object to be set as the current executor.

        """
        # 设置类变量 _executor 为传入的 executor 参数
        cls._executor = executor
class DAGLifecycle:
    """The lifecycle of DAG."""

    async def before_dag_run(self):
        """Execute before DAG run."""
        pass

    async def after_dag_end(self, event_loop_task_id: int):
        """Execute after DAG end.

        This method may be called multiple times, please make sure it is idempotent.
        """
        pass


class DAGNode(DAGLifecycle, DependencyMixin, ViewMixin, ABC):
    """The base class of DAGNode."""

    resource_group: Optional[ResourceGroup] = None
    """The resource group of current DAGNode"""

    def __init__(
        self,
        dag: Optional["DAG"] = None,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        system_app: Optional[SystemApp] = None,
        executor: Optional[Executor] = None,
        **kwargs,
    ) -> None:
        """Initialize a DAGNode.

        Args:
            dag (Optional["DAG"], optional): The DAG to add this node to.
            Defaults to None.
            node_id (Optional[str], optional): The node id. Defaults to None.
            node_name (Optional[str], optional): The node name. Defaults to None.
            system_app (Optional[SystemApp], optional): The system app.
            Defaults to None.
            executor (Optional[Executor], optional): The executor. Defaults to None.
        """
        super().__init__()  # 调用父类的初始化方法
        self._upstream: List["DAGNode"] = []  # 初始化上游节点列表为空列表
        self._downstream: List["DAGNode"] = []  # 初始化下游节点列表为空列表
        self._dag: Optional["DAG"] = dag or DAGVar.get_current_dag()  # 设置当前节点所属的DAG对象
        self._system_app: Optional[SystemApp] = (
            system_app or DAGVar.get_current_system_app()  # 设置当前节点关联的系统应用对象
        )
        self._executor: Optional[Executor] = executor or DAGVar.get_executor()  # 设置当前节点使用的执行器对象
        if not node_id and self._dag:
            node_id = self._dag._new_node_id()  # 如果节点ID未指定且存在DAG对象，则生成新的节点ID
        self._node_id: Optional[str] = node_id  # 设置当前节点的节点ID
        self._node_name: Optional[str] = node_name  # 设置当前节点的节点名称
        if self._dag:
            self._dag._append_node(self)  # 将当前节点添加到其所属的DAG对象中

    @property
    def node_id(self) -> str:
        """Return the node id of current DAGNode."""
        if not self._node_id:
            raise ValueError("Node id not set for current DAGNode")  # 如果节点ID未设置，则抛出数值错误
        return self._node_id  # 返回当前节点的节点ID

    @property
    @abstractmethod
    def dev_mode(self) -> bool:
        """Whether current DAGNode is in dev mode."""
        # 抽象属性，需要在子类中实现，用于判断当前节点是否处于开发模式

    @property
    def system_app(self) -> Optional[SystemApp]:
        """Return the system app of current DAGNode."""
        return self._system_app  # 返回当前节点关联的系统应用对象

    def set_system_app(self, system_app: SystemApp) -> None:
        """Set system app for current DAGNode.

        Args:
            system_app (SystemApp): The system app
        """
        self._system_app = system_app  # 设置当前节点关联的系统应用对象

    def set_node_id(self, node_id: str) -> None:
        """Set node id for current DAGNode.

        Args:
            node_id (str): The node id
        """
        self._node_id = node_id  # 设置当前节点的节点ID
    def __hash__(self) -> int:
        """Return the hash value of current DAGNode.

        If the node_id is not None, return the hash value of node_id.
        """
        if self.node_id:
            # 如果节点的 node_id 不为 None，则返回其哈希值
            return hash(self.node_id)
        else:
            # 否则调用父类的哈希函数返回哈希值
            return super().__hash__()

    def __eq__(self, other: Any) -> bool:
        """Return whether the current DAGNode is equal to other DAGNode."""
        if not isinstance(other, DAGNode):
            # 如果 other 不是 DAGNode 类型，则返回 False
            return False
        return self.node_id == other.node_id  # 比较当前节点和另一个节点的 node_id 是否相等

    @property
    def node_name(self) -> Optional[str]:
        """Return the node name of current DAGNode.

        Returns:
            Optional[str]: The node name of current DAGNode
        """
        return self._node_name  # 返回当前节点的名称

    @property
    def dag(self) -> Optional["DAG"]:
        """Return the DAG of current DAGNode.

        Returns:
            Optional["DAG"]: The DAG of current DAGNode
        """
        return self._dag  # 返回当前节点所属的 DAG 对象

    def set_upstream(self, nodes: DependencyType) -> None:
        """Set upstream nodes for current node.

        Args:
            nodes (DependencyType): Upstream nodes to be set to current node.
        """
        self.set_dependency(nodes)  # 设置当前节点的上游节点依赖关系

    def set_downstream(self, nodes: DependencyType) -> None:
        """Set downstream nodes for current node.

        Args:
            nodes (DependencyType): Downstream nodes to be set to current node.
        """
        self.set_dependency(nodes, is_upstream=False)  # 设置当前节点的下游节点依赖关系

    @property
    def upstream(self) -> List["DAGNode"]:
        """Return the upstream nodes of current DAGNode.

        Returns:
            List["DAGNode"]: The upstream nodes of current DAGNode
        """
        return self._upstream  # 返回当前节点的所有上游节点列表

    @property
    def downstream(self) -> List["DAGNode"]:
        """Return the downstream nodes of current DAGNode.

        Returns:
            List["DAGNode"]: The downstream nodes of current DAGNode
        """
        return self._downstream  # 返回当前节点的所有下游节点列表
    def set_dependency(self, nodes: DependencyType, is_upstream: bool = True) -> None:
        """Set dependency for current node.

        Args:
            nodes (DependencyType): The nodes to set dependency to current node.
            is_upstream (bool, optional): Whether set upstream nodes. Defaults to True.
        """
        # 如果nodes不是Sequence类型，则转换为列表
        if not isinstance(nodes, Sequence):
            nodes = [nodes]
        # 检查所有节点是否都是DAGNode的实例，如果不是则抛出错误
        if not all(isinstance(node, DAGNode) for node in nodes):
            raise ValueError(
                "all nodes to set dependency to current node must be instance "
                "of 'DAGNode'"
            )
        # 将nodes强制转换为Sequence[DAGNode]
        nodes = cast(Sequence[DAGNode], nodes)
        # 收集所有节点所属的DAG对象，去除空值
        dags = set([node.dag for node in nodes if node.dag])  # noqa: C403
        # 如果当前节点有所属的DAG对象，则加入dags集合中
        if self.dag:
            dags.add(self.dag)
        # 如果没有找到任何有效的DAG对象，则抛出错误
        if not dags:
            raise ValueError("set dependency to current node must in a DAG context")
        # 如果找到的DAG对象超过一个，则抛出错误
        if len(dags) != 1:
            raise ValueError(
                "set dependency to current node just support in one DAG context"
            )
        # 从集合中取出唯一的DAG对象，并将当前节点加入到该DAG对象中
        dag = dags.pop()
        self._dag = dag

        # 将当前节点加入到DAG对象的节点列表中
        dag._append_node(self)
        # 根据is_upstream标志，设置当前节点的上游或下游节点
        for node in nodes:
            if is_upstream and node not in self.upstream:
                node._dag = dag
                dag._append_node(node)

                self._upstream.append(node)
                node._downstream.append(self)
            elif node not in self._downstream:
                node._dag = dag
                dag._append_node(node)

                self._downstream.append(node)
                node._upstream.append(self)

    def __repr__(self):
        """Return the representation of current DAGNode."""
        # 返回当前DAGNode对象的字符串表示形式
        cls_name = self.__class__.__name__
        if self.node_id and self.node_name:
            return f"{cls_name}(node_id={self.node_id}, node_name={self.node_name})"
        if self.node_id:
            return f"{cls_name}(node_id={self.node_id})"
        if self.node_name:
            return f"{cls_name}(node_name={self.node_name})"
        else:
            return f"{cls_name}"

    @property
    def graph_str(self):
        """Return the graph string of current DAGNode."""
        # 返回当前DAGNode对象的图形字符串表示形式
        cls_name = self.__class__.__name__
        if self.node_id and self.node_name:
            return f"{self.node_id}({cls_name},{self.node_name})"
        if self.node_id:
            return f"{self.node_id}({cls_name})"
        if self.node_name:
            return f"{self.node_name}_{cls_name}({cls_name})"
        else:
            return f"{cls_name}"

    def __str__(self):
        """Return the string of current DAGNode."""
        # 返回当前DAGNode对象的字符串表示形式
        return self.__repr__()
# 构建任务键的函数，用于生成唯一标识任务的键值
def _build_task_key(task_name: str, key: str) -> str:
    return f"{task_name}___$$$$$$___{key}"

# 表示当前DAG的上下文，当DAG运行时创建
class DAGContext:
    """The context of current DAG, created when the DAG is running.

    Every DAG has been triggered will create a new DAGContext.
    """

    def __init__(
        self,
        node_to_outputs: Dict[str, TaskContext],
        share_data: Dict[str, Any],
        event_loop_task_id: int,
        streaming_call: bool = False,
        node_name_to_ids: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize a DAGContext.

        Args:
            node_to_outputs (Dict[str, TaskContext]): The task outputs of current DAG.
            share_data (Dict[str, Any]): The share data of current DAG.
            streaming_call (bool, optional): Whether the current DAG is streaming call.
                Defaults to False.
            node_name_to_ids (Optional[Dict[str, str]], optional): The node name to node
                IDs mapping for current DAG. Defaults to None.
        """
        # 如果没有提供节点名称到节点ID的映射，设为空字典
        if not node_name_to_ids:
            node_name_to_ids = {}
        self._streaming_call = streaming_call
        self._curr_task_ctx: Optional[TaskContext] = None
        self._share_data: Dict[str, Any] = share_data
        self._node_to_outputs: Dict[str, TaskContext] = node_to_outputs
        self._node_name_to_ids: Dict[str, str] = node_name_to_ids
        self._event_loop_task_id = event_loop_task_id

    @property
    def _task_outputs(self) -> Dict[str, TaskContext]:
        """Return the task outputs of current DAG.

        Just use for internal for now.
        Returns:
            Dict[str, TaskContext]: The task outputs of current DAG
        """
        return self._node_to_outputs

    @property
    def current_task_context(self) -> TaskContext:
        """Return the current task context."""
        # 如果当前任务上下文未设置，抛出运行时错误
        if not self._curr_task_ctx:
            raise RuntimeError("Current task context not set")
        return self._curr_task_ctx

    @property
    def streaming_call(self) -> bool:
        """Whether the current DAG is streaming call."""
        return self._streaming_call

    def set_current_task_context(self, _curr_task_ctx: TaskContext) -> None:
        """Set the current task context.

        When the task is running, the current task context
        will be set to the task context.

        TODO: We should support parallel task running in the future.
        """
        self._curr_task_ctx = _curr_task_ctx
    # 根据任务名获取任务输出对象
    def get_task_output(self, task_name: str) -> TaskOutput:
        """Get the task output by task name.

        Args:
            task_name (str): The task name

        Returns:
            TaskOutput: The task output
        """
        # 如果任务名为 None，则抛出数值错误异常
        if task_name is None:
            raise ValueError("task_name can't be None")
        
        # 根据任务名获取对应的节点 ID
        node_id = self._node_name_to_ids.get(task_name)
        
        # 如果未找到对应的节点 ID，则抛出数值错误异常
        if not node_id:
            raise ValueError(f"Task name {task_name} not in DAG")
        
        # 根据节点 ID 获取任务输出对象
        task_output = self._task_outputs.get(node_id)
        
        # 如果未找到任务输出对象，则抛出数值错误异常
        if not task_output:
            raise ValueError(f"Task output for task {task_name} not exists")
        
        # 返回任务输出对象中的具体任务输出
        return task_output.task_output

    # 异步方法：根据键获取共享数据
    async def get_from_share_data(self, key: str) -> Any:
        """Get share data by key.

        Args:
            key (str): The share data key

        Returns:
            Any: The share data, you can cast it to the real type
        """
        # 记录调试日志：根据键从共享数据中获取数据
        logger.debug(f"Get share data by key {key} from {id(self._share_data)}")
        
        # 返回共享数据中指定键的数据
        return self._share_data.get(key)

    # 异步方法：保存数据到共享数据
    async def save_to_share_data(
        self, key: str, data: Any, overwrite: bool = False
    ) -> None:
        """Save share data by key.

        Args:
            key (str): The share data key
            data (Any): The share data
            overwrite (bool): Whether overwrite the share data if the key
                already exists. Defaults to None.
        """
        # 如果键已存在于共享数据中且不允许覆盖，则抛出数值错误异常
        if key in self._share_data and not overwrite:
            raise ValueError(f"Share data key {key} already exists")
        
        # 记录调试日志：保存数据到共享数据中
        logger.debug(f"Save share data by key {key} to {id(self._share_data)}")
        
        # 将数据保存到共享数据中指定键的位置
        self._share_data[key] = data

    # 异步方法：根据任务名和键获取共享数据
    async def get_task_share_data(self, task_name: str, key: str) -> Any:
        """Get share data by task name and key.

        Args:
            task_name (str): The task name
            key (str): The share data key

        Returns:
            Any: The share data
        """
        # 如果任务名为 None，则抛出数值错误异常
        if task_name is None:
            raise ValueError("task_name can't be None")
        
        # 如果键为 None，则抛出数值错误异常
        if key is None:
            raise ValueError("key can't be None")
        
        # 根据任务名和键构建具体任务键，然后从共享数据中获取数据
        return self.get_from_share_data(_build_task_key(task_name, key))

    # 异步方法：根据任务名和键保存数据到共享数据
    async def save_task_share_data(
        self, task_name: str, key: str, data: Any, overwrite: bool = False
    ) -> None:
        """Save share data by task name and key.

        Args:
            task_name (str): The task name
            key (str): The share data key
            data (Any): The share data
            overwrite (bool): Whether overwrite the share data if the key
                already exists. Defaults to None.

        Raises:
            ValueError: If the share data key already exists and overwrite is not True
        """
        # 如果任务名为 None，则抛出数值错误异常
        if task_name is None:
            raise ValueError("task_name can't be None")
        
        # 如果键为 None，则抛出数值错误异常
        if key is None:
            raise ValueError("key can't be None")
        
        # 根据任务名和键构建具体任务键，然后保存数据到共享数据中
        await self.save_to_share_data(_build_task_key(task_name, key), data, overwrite)
    # 定义一个异步方法 `_clean_all`，用于执行清理操作，暂时未实现具体功能
    async def _clean_all(self):
        # `pass` 关键字表示该方法暂时不执行任何操作，保留为将来实现功能预留空间
        pass
    @property
    def dag_id(self) -> str:
        """Return the dag id of current DAG."""
        return self._dag_id



    @property
    def tags(self) -> Dict[str, str]:
        """Return the tags of current DAG."""
        return self._tags



    @property
    def description(self) -> Optional[str]:
        """Return the description of current DAG."""
        return self._description



    @property
    def dev_mode(self) -> bool:
        """Whether the current DAG is in dev mode.

        Returns:
            bool: Whether the current DAG is in dev mode
        """
        from ..operators.base import _dev_mode

        return _dev_mode()



    def _build(self) -> None:
        from ..operators.common_operator import TriggerOperator

        # 初始化一个空的节点集合
        nodes: Set[DAGNode] = set()
        # 遍历 DAG 的节点映射，获取所有节点
        for _, node in self.node_map.items():
            # 将节点及其后续节点添加到集合中
            nodes = nodes.union(_get_nodes(node))
        # 设置根节点为没有上游节点的节点集合
        self._root_nodes = list(set(filter(lambda x: not x.upstream, nodes)))
        # 设置叶子节点为没有下游节点的节点集合
        self._leaf_nodes = list(set(filter(lambda x: not x.downstream, nodes)))
        # 设置触发器节点为所有触发器操作节点的集合
        self._trigger_nodes = list(
            set(filter(lambda x: isinstance(x, TriggerOperator), nodes))
        )



    def _append_node(self, node: DAGNode) -> None:
        # 如果节点 ID 已存在于节点映射中，则直接返回
        if node.node_id in self.node_map:
            return
        # 如果节点有名称，并且名称已存在于名称到节点的映射中，则引发异常
        if node.node_name:
            if node.node_name in self.node_name_to_node:
                raise ValueError(
                    f"Node name {node.node_name} already exists in DAG {self.dag_id}"
                )
            # 将节点名称映射到节点对象
            self.node_name_to_node[node.node_name] = node
        # 获取节点的 ID
        node_id = node.node_id
        # 如果节点 ID 为空，则引发异常
        if not node_id:
            raise ValueError("Node id can't be None")
        # 将节点添加到节点映射中
        self.node_map[node_id] = node
        # 清空缓存的根节点和叶子节点列表
        self._root_nodes = []
        self._leaf_nodes = []



    def _new_node_id(self) -> str:
        # 返回一个新的 UUID 作为节点 ID
        return str(uuid.uuid4())



    def __init__(
        self,
        dag_id: str,
        resource_group: Optional[ResourceGroup] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initialize a DAG."""
        # 初始化 DAG 对象
        self._dag_id = dag_id
        # 初始化标签字典，如果未提供则为空字典
        self._tags: Dict[str, str] = tags or {}
        # 初始化描述信息，如果未提供则为 None
        self._description = description
        # 初始化节点映射和节点名称到节点的映射
        self.node_map: Dict[str, DAGNode] = {}
        self.node_name_to_node: Dict[str, DAGNode] = {}
        # 初始化根节点、叶子节点和触发器节点列表为空列表
        self._root_nodes: List[DAGNode] = []
        self._leaf_nodes: List[DAGNode] = []
        self._trigger_nodes: List[DAGNode] = []
        # 初始化资源组和异步锁
        self._resource_group: Optional[ResourceGroup] = resource_group
        self._lock = asyncio.Lock()
        # 初始化事件循环任务 ID 到上下文的映射
        self._event_loop_task_id_to_ctx: Dict[int, DAGContext] = {}
    def root_nodes(self) -> List[DAGNode]:
        """Return the root nodes of current DAG.

        Returns:
            List[DAGNode]: The root nodes of current DAG, no repeat
        """
        # 如果根节点列表为空，则调用 _build 方法构建 DAG
        if not self._root_nodes:
            self._build()
        # 返回当前 DAG 的根节点列表
        return self._root_nodes

    @property
    def leaf_nodes(self) -> List[DAGNode]:
        """Return the leaf nodes of current DAG.

        Returns:
            List[DAGNode]: The leaf nodes of current DAG, no repeat
        """
        # 如果叶子节点列表为空，则调用 _build 方法构建 DAG
        if not self._leaf_nodes:
            self._build()
        # 返回当前 DAG 的叶子节点列表
        return self._leaf_nodes

    @property
    def trigger_nodes(self) -> List[DAGNode]:
        """Return the trigger nodes of current DAG.

        Returns:
            List[DAGNode]: The trigger nodes of current DAG, no repeat
        """
        # 如果触发节点列表为空，则调用 _build 方法构建 DAG
        if not self._trigger_nodes:
            self._build()
        # 返回当前 DAG 的触发节点列表
        return self._trigger_nodes

    async def _save_dag_ctx(self, dag_ctx: DAGContext) -> None:
        # 使用异步锁保护操作
        async with self._lock:
            event_loop_task_id = dag_ctx._event_loop_task_id
            current_task = asyncio.current_task()
            task_name = current_task.get_name() if current_task else None
            # 将 DAG 上下文存储到事件循环任务 ID 映射中
            self._event_loop_task_id_to_ctx[event_loop_task_id] = dag_ctx
            logger.debug(
                f"Save DAG context {dag_ctx} to event loop task {event_loop_task_id}, "
                f"task_name: {task_name}"
            )

    async def _after_dag_end(self, event_loop_task_id: Optional[int] = None) -> None:
        """Execute after DAG end."""
        tasks = []
        event_loop_task_id = event_loop_task_id or id(asyncio.current_task())
        # 对所有节点执行 DAG 结束后的操作
        for node in self.node_map.values():
            tasks.append(node.after_dag_end(event_loop_task_id))
        await asyncio.gather(*tasks)

        # 清理 DAG 上下文
        async with self._lock:
            current_task = asyncio.current_task()
            task_name = current_task.get_name() if current_task else None
            # 如果在事件循环任务 ID 映射中找不到指定的 DAG 上下文，则引发异常
            if event_loop_task_id not in self._event_loop_task_id_to_ctx:
                raise RuntimeError(
                    f"DAG context not found with event loop task id "
                    f"{event_loop_task_id}, task_name: {task_name}"
                )
            # 记录清理 DAG 上下文的调试信息
            logger.debug(
                f"Clean DAG context with event loop task id {event_loop_task_id}, "
                f"task_name: {task_name}"
            )
            # 从事件循环任务 ID 映射中移除并清理指定的 DAG 上下文
            dag_ctx = self._event_loop_task_id_to_ctx.pop(event_loop_task_id)
            await dag_ctx._clean_all()

    def print_tree(self) -> None:
        """Print the DAG tree"""  # noqa: D400
        # 调用 _print_format_dag_tree 函数打印 DAG 树形结构
        _print_format_dag_tree(self)
    def visualize_dag(self, view: bool = True, **kwargs) -> Optional[str]:
        """Visualize the DAG.

        Args:
            view (bool, optional): Whether view the DAG graph. Defaults to True,
                if True, it will open the graph file with your default viewer.
        """
        # 打印DAG的树形结构
        self.print_tree()
        # 调用_visualize_dag函数可视化DAG，并返回可视化结果
        return _visualize_dag(self, view=view, **kwargs)

    def show(self, mermaid: bool = False) -> Any:
        """Return the graph of current DAG."""
        # 获取当前DAG的图形表示
        dot, mermaid_str = _get_graph(self)
        # 如果mermaid为True，则返回mermaid格式的图形表示，否则返回dot格式的图形表示
        return mermaid_str if mermaid else dot

    def __enter__(self):
        """Enter a DAG context."""
        # 进入DAG上下文
        DAGVar.enter_dag(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit a DAG context."""
        # 退出DAG上下文
        DAGVar.exit_dag()

    def __hash__(self) -> int:
        """Return the hash value of current DAG.

        If the dag_id is not None, return the hash value of dag_id.
        """
        # 如果dag_id不为None，则返回dag_id的哈希值，否则返回父类的哈希值
        if self.dag_id:
            return hash(self.dag_id)
        else:
            return super().__hash__()

    def __eq__(self, other):
        """Return whether the current DAG is equal to other DAG."""
        # 判断当前DAG是否与另一个DAG相等
        if not isinstance(other, DAG):
            return False
        return self.dag_id == other.dag_id

    def __repr__(self):
        """Return the representation of current DAG."""
        # 返回当前DAG的表示形式
        return f"DAG(dag_id={self.dag_id})"
# 获取与给定节点相关的节点集合，包括上游或下游节点
def _get_nodes(node: DAGNode, is_upstream: Optional[bool] = True) -> Set[DAGNode]:
    nodes: Set[DAGNode] = set()
    if not node:
        return nodes
    nodes.add(node)
    # 根据 is_upstream 决定是获取上游还是下游节点集合
    stream_nodes = node.upstream if is_upstream else node.downstream
    # 递归地将相关节点添加到集合中
    for node in stream_nodes:
        nodes = nodes.union(_get_nodes(node, is_upstream))
    return nodes


# 打印格式化的DAG树结构
def _print_format_dag_tree(dag: DAG) -> None:
    for node in dag.root_nodes:
        _print_dag(node)


# 递归打印DAG节点及其子节点
def _print_dag(
    node: DAGNode,
    level: int = 0,
    prefix: str = "",
    last: bool = True,
    level_dict: Optional[Dict[int, Any]] = None,
):
    if level_dict is None:
        level_dict = {}

    connector = " -> " if level != 0 else ""
    new_prefix = prefix
    if last:
        if level != 0:
            new_prefix += "  "
        # 打印节点信息，带有适当的前缀和连接符
        print(prefix + connector + str(node))
    else:
        if level != 0:
            new_prefix += "| "
        print(prefix + connector + str(node))

    # 维护当前层级节点数目的字典
    level_dict[level] = level_dict.get(level, 0) + 1
    num_children = len(node.downstream)
    for i, child in enumerate(node.downstream):
        _print_dag(child, level + 1, new_prefix, i == num_children - 1, level_dict)


# 打印DAG树的根节点及其子节点
def _print_dag_tree(root_nodes: List[DAGNode], level_sep: str = "  ") -> None:
    def _print_node(node: DAGNode, level: int) -> None:
        print(f"{level_sep * level}{node}")

    _apply_root_node(root_nodes, _print_node)


# 对每个根节点应用给定的函数
def _apply_root_node(
    root_nodes: List[DAGNode],
    func: Callable[[DAGNode, int], None],
) -> None:
    for dag_node in root_nodes:
        _handle_dag_nodes(False, 0, dag_node, func)


# 处理DAG节点，根据方向（从上到下或从下到上）递归应用给定的函数
def _handle_dag_nodes(
    is_down_to_up: bool,
    level: int,
    dag_node: DAGNode,
    func: Callable[[DAGNode, int], None],
):
    if not dag_node:
        return
    # 应用给定的函数到当前节点
    func(dag_node, level)
    # 获取当前节点的相关节点集合（上游或下游）
    stream_nodes = dag_node.upstream if is_down_to_up else dag_node.downstream
    level += 1
    # 递归地处理相关节点
    for node in stream_nodes:
        _handle_dag_nodes(is_down_to_up, level, node, func)


# 获取DAG的图形表示
def _get_graph(dag: DAG):
    try:
        from graphviz import Digraph
    except ImportError:
        logger.warn("Can't import graphviz, skip visualize DAG")
        return None, None
    # 创建一个Digraph对象表示DAG
    dot = Digraph(name=dag.dag_id)
    # 初始化Mermaid图的定义
    mermaid_str = "graph TD;\n"
    # 记录已添加的边，避免添加重复的边
    added_edges = set()

    # 添加边的函数，用于递归添加DAG节点之间的边
    def add_edges(node: DAGNode):
        nonlocal mermaid_str
        if node.downstream:
            for downstream_node in node.downstream:
                # 检查是否已经添加了该边
                if (str(node), str(downstream_node)) not in added_edges:
                    dot.edge(str(node), str(downstream_node))
                    mermaid_str += f"    {node.graph_str} --> {downstream_node.graph_str};\n"  # noqa
                    added_edges.add((str(node), str(downstream_node)))
                add_edges(downstream_node)

    # 对每个根节点调用添加边的函数
    for root in dag.root_nodes:
        add_edges(root)
    return dot, mermaid_str
def _visualize_dag(
    dag: DAG, view: bool = True, generate_mermaid: bool = True, **kwargs
) -> Optional[str]:
    """Visualize the DAG.

    Args:
        dag (DAG): The DAG object to visualize
        view (bool, optional): Whether to display the DAG graph after rendering. Defaults to True.
        generate_mermaid (bool, optional): Whether to generate a Mermaid syntax file alongside the visualization.
            Defaults to True.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Optional[str]: The filename of the rendered DAG graph if successful; otherwise, None.
    """
    # Retrieve the DOT and Mermaid syntax representations of the DAG
    dot, mermaid_str = _get_graph(dag)
    
    # If DOT representation is not available, return None
    if not dot:
        return None
    
    # Determine the filename for the rendered DAG graph
    filename = f"dag-vis-{dag.dag_id}.gv"
    
    # Override filename if 'filename' is provided in kwargs
    if "filename" in kwargs:
        filename = kwargs["filename"]
        del kwargs["filename"]
    
    # Set default directory for saving files if 'directory' not provided in kwargs
    if "directory" not in kwargs:
        from dbgpt.configs.model_config import LOGDIR
        kwargs["directory"] = LOGDIR
    
    # Generate Mermaid syntax file if requested
    if generate_mermaid:
        mermaid_filename = filename.replace(".gv", ".md")
        with open(f"{kwargs.get('directory', '')}/{mermaid_filename}", "w") as mermaid_file:
            logger.info(f"Writing Mermaid syntax to {mermaid_filename}")
            mermaid_file.write(mermaid_str)
    
    # Render the DOT file and return its filename
    return dot.render(filename, view=view, **kwargs)
```